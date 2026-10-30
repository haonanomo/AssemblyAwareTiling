"""Batched disassembly environment with geometric, IK, and stability constraints.

Inputs are tiled boards with -1 outside, 0 for virtual unfilled support, and
positive real-piece IDs. Part states are 0=removed, 1=free, and 2=held.
Actions 0..P-1 hold a part; P..2P-1 remove a held part. Completed episodes
reset automatically; step also returns the raw state before reset.
"""

import logging
import pickle
from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tiling_rl.physics.simulator_newton_seq import check_stability_newton
from tiling_rl.sequence.ik_lookup import IKLookupTable, resolve_ik_table_path
from tiling_rl.stability.stability_checker_jax_seq import checker_predict_batch_jax

logger = logging.getLogger(__name__)


def pack_ik_tables_for_jax(ik_tables, max_parts, max_samples=None, max_r1=None, max_r2=None):
    """Pad per-board IK tables to fixed shapes for batched JAX evaluation."""
    batch = len(ik_tables)

    if max_samples is None:
        max_samples = max(int(t.sample_piece_id.shape[0]) for t in ik_tables)
    if max_r1 is None:
        max_r1 = max(int(t.robot1_sample_ids.shape[0]) for t in ik_tables)
    if max_r2 is None:
        max_r2 = max(int(t.robot2_sample_ids.shape[0]) for t in ik_tables)

    sample_piece_id = np.zeros((batch, max_samples), dtype=np.int32)
    sample_robot_id = np.zeros((batch, max_samples), dtype=np.int32)
    collision_mask = np.ones((batch, max_samples, max_parts), dtype=np.bool_)

    robot1_sample_ids = -np.ones((batch, max_r1), dtype=np.int32)
    robot2_sample_ids = -np.ones((batch, max_r2), dtype=np.int32)
    pair_collision = np.ones((batch, max_r1, max_r2), dtype=np.bool_)

    num_samples = np.zeros((batch,), dtype=np.int32)
    num_r1 = np.zeros((batch,), dtype=np.int32)
    num_r2 = np.zeros((batch,), dtype=np.int32)

    for b, t in enumerate(ik_tables):
        s = int(t.sample_piece_id.shape[0])
        r1 = int(t.robot1_sample_ids.shape[0])
        r2 = int(t.robot2_sample_ids.shape[0])

        if s > max_samples or r1 > max_r1 or r2 > max_r2:
            raise ValueError(
                f"IK table exceeds padded size: "
                f"s={s}/{max_samples}, r1={r1}/{max_r1}, r2={r2}/{max_r2}"
            )

        num_samples[b] = s
        num_r1[b] = r1
        num_r2[b] = r2

        sample_piece_id[b, :s] = t.sample_piece_id.astype(np.int32)
        sample_robot_id[b, :s] = t.sample_robot_id.astype(np.int32)

        cm = t.collision_mask.astype(np.bool_)
        ncol = min(cm.shape[1], max_parts)
        collision_mask[b, :s, :ncol] = cm[:, :ncol]

        robot1_sample_ids[b, :r1] = t.robot1_sample_ids.astype(np.int32)
        robot2_sample_ids[b, :r2] = t.robot2_sample_ids.astype(np.int32)

        pc = t.pair_collision.astype(np.bool_)
        pair_collision[b, :r1, :r2] = pc[:r1, :r2]

    return {
        "ik_sample_piece_id": sample_piece_id,
        "ik_sample_robot_id": sample_robot_id,
        "ik_collision_mask": collision_mask,
        "ik_robot1_sample_ids": robot1_sample_ids,
        "ik_robot2_sample_ids": robot2_sample_ids,
        "ik_pair_collision": pair_collision,
        "ik_num_samples": num_samples,
        "ik_num_r1": num_r1,
        "ik_num_r2": num_r2,
    }


class SeqEnvState(NamedTuple):
    """Batched scene, part state, and padded IK tables used by JAX search."""

    board: jnp.ndarray
    part_states: jnp.ndarray
    piece_masks: jnp.ndarray
    contact_matrix: jnp.ndarray
    insertion_table: jnp.ndarray
    n_parts: jnp.ndarray
    stable: jnp.ndarray

    ik_sample_piece_id: jnp.ndarray
    ik_sample_robot_id: jnp.ndarray
    ik_collision_mask: jnp.ndarray
    ik_robot1_sample_ids: jnp.ndarray
    ik_robot2_sample_ids: jnp.ndarray
    ik_pair_collision: jnp.ndarray
    ik_num_samples: jnp.ndarray
    ik_num_r1: jnp.ndarray
    ik_num_r2: jnp.ndarray


def piece_ids_from_board(board):
    return sorted(int(x) for x in np.unique(board) if x > 0)


def normalize_piece_ids(board):
    board = np.asarray(board, dtype=np.int32)
    out = board.copy()
    pids = piece_ids_from_board(board)
    mapping = {old: new for new, old in enumerate(pids, start=1)}
    for old, new in mapping.items():
        out[board == old] = new
    return out, mapping


def shift_mask(mask, dx, dy):
    h, w = mask.shape
    out = np.zeros_like(mask)
    coords = np.argwhere(mask > 0)
    if coords.shape[0] == 0:
        return out
    for r, c in coords:
        nr, nc = r + dx, c + dy
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            return None
        out[nr, nc] = 1
    return out


def compute_insertion_table_2d(board, max_shift=10, max_parts=None):
    pids = piece_ids_from_board(board)
    n_part = len(pids)
    if max_parts is None:
        max_parts = n_part

    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)
    table = np.ones((4, max_parts, max_parts), dtype=bool)
    piece_masks = {pid: (board == pid).astype(np.uint8) for pid in pids}

    for pid_i in pids:
        mask_i = piece_masks[pid_i]
        i = pid_i - 1
        for d_id, (dx, dy) in enumerate(dirs):
            ok_dir = np.ones((max_parts,), dtype=bool)
            for step in range(1, max_shift + 1):
                moved = shift_mask(mask_i, dx * step, dy * step)
                if moved is None:
                    break
                for pid_j in pids:
                    if pid_i == pid_j:
                        continue
                    j = pid_j - 1
                    if np.any((moved > 0) & (piece_masks[pid_j] > 0)):
                        ok_dir[j] = False
            table[d_id, i, :] = ok_dir
            table[d_id, i, i] = True

    return table


def compute_contact_matrix_2d(board, max_parts=None):
    pids = piece_ids_from_board(board)
    n_part = len(pids)
    if max_parts is None:
        max_parts = n_part

    contact = np.zeros((max_parts, max_parts), dtype=np.float32)
    h, w = board.shape

    for r in range(h):
        for c in range(w):
            pid = int(board[r, c])
            if pid <= 0:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if rr >= h or cc >= w:
                    continue
                qid = int(board[rr, cc])
                if qid <= 0 or qid == pid:
                    continue
                i, j = pid - 1, qid - 1
                if i < max_parts and j < max_parts:
                    contact[i, j] = 1.0
                    contact[j, i] = 1.0

    return contact


def compute_part_features_2d(board, max_parts=None):
    pids = piece_ids_from_board(board)
    n_part = len(pids)
    if max_parts is None:
        max_parts = n_part

    h, w = board.shape
    feats = np.zeros((max_parts, 6), dtype=np.float32)

    total_valid = max(float(np.sum(board != -1)), 1.0)

    for pid in pids:
        idx = pid - 1
        if idx >= max_parts:
            continue

        coords = np.argwhere(board == pid)
        if coords.shape[0] == 0:
            continue

        rows = coords[:, 0].astype(np.float32)
        cols = coords[:, 1].astype(np.float32)

        size = coords.shape[0] / total_valid
        cy = rows.mean() / max(h - 1, 1)
        cx = cols.mean() / max(w - 1, 1)
        ymin = rows.min() / max(h - 1, 1)
        ymax = rows.max() / max(h - 1, 1)
        xmin = cols.min() / max(w - 1, 1)
        xmax = cols.max() / max(w - 1, 1)

        feats[idx] = np.array([size, cy, cx, ymax - ymin, xmax - xmin, 1.0], dtype=np.float32)

    return feats


def compute_piece_masks_2d(board, max_parts=None):
    pids = piece_ids_from_board(board)
    n_part = len(pids)
    if max_parts is None:
        max_parts = n_part

    h, w = board.shape
    masks = np.zeros((max_parts, h, w), dtype=np.float32)

    for pid in pids:
        idx = pid - 1
        if idx >= max_parts:
            continue
        masks[idx] = (board == pid).astype(np.float32)

    return masks


def hold_mask_from_state(full_board, part_states):
    hold_mask = (full_board == 0).astype(np.float32)
    part_states = np.asarray(part_states, dtype=np.int32)

    for pid in range(1, part_states.shape[0] + 1):
        if part_states[pid - 1] == 2:
            hold_mask[full_board == pid] = 1.0

    hold_mask[full_board < 0] = 0.0
    return hold_mask


def held_piece_ids_from_state(part_states):
    part_states = np.asarray(part_states, dtype=np.int32)
    return set((np.where(part_states == 2)[0] + 1).astype(int).tolist())


@jax.jit
def _ik_one_robot_piece_feasible_single(
    state,
    pid,
    robot_id,
    sample_piece_id,
    sample_robot_id,
    collision_mask,
    num_samples,
):
    num_parts = state.shape[0]
    sample_idx = jnp.arange(sample_piece_id.shape[0], dtype=jnp.int32)
    valid_sample = sample_idx < num_samples

    is_same_piece = sample_piece_id == pid
    is_same_robot = sample_robot_id == robot_id

    active = state > 0
    pids = jnp.arange(1, num_parts + 1, dtype=jnp.int32)
    active_other = active & (pids != pid)

    collide_active = jnp.any(collision_mask & active_other[None, :], axis=1)

    ok = valid_sample & is_same_piece & is_same_robot & (~collide_active)
    return jnp.any(ok)


@jax.jit
def _held_pid_at(state, k):
    num_parts = state.shape[0]
    pids = jnp.arange(1, num_parts + 1, dtype=jnp.int32)
    held_pids = jnp.where(state == 2, pids, 0)

    def body(carry, x):
        count, out = carry
        is_held = x > 0
        take = is_held & (count == k)
        out = jnp.where(take, x, out)
        count = count + is_held.astype(jnp.int32)
        return (count, out), None

    (count, out), _ = jax.lax.scan(body, (jnp.int32(0), jnp.int32(0)), held_pids)
    return out


@jax.jit
def _ik_pair_assignment_feasible_single(
    state,
    pid_a,
    pid_b,
    robot_a,
    robot_b,
    sample_piece_id,
    sample_robot_id,
    collision_mask,
    robot1_sample_ids,
    robot2_sample_ids,
    pair_collision,
    num_samples,
    num_r1,
    num_r2,
):
    num_parts = state.shape[0]
    active = state > 0
    pids = jnp.arange(1, num_parts + 1, dtype=jnp.int32)

    sample_idx = jnp.arange(sample_piece_id.shape[0], dtype=jnp.int32)
    valid_sample = sample_idx < num_samples

    def valid_for(pid, rid):
        active_other = active & (pids != pid)
        collide_active = jnp.any(collision_mask & active_other[None, :], axis=1)
        return (
            valid_sample
            & (sample_piece_id == pid)
            & (sample_robot_id == rid)
            & (~collide_active)
        )

    valid_a = valid_for(pid_a, robot_a)
    valid_b = valid_for(pid_b, robot_b)

    r1_idx = jnp.arange(robot1_sample_ids.shape[0], dtype=jnp.int32)
    r2_idx = jnp.arange(robot2_sample_ids.shape[0], dtype=jnp.int32)

    r1_valid = r1_idx < num_r1
    r2_valid = r2_idx < num_r2

    r1_global = robot1_sample_ids
    r2_global = robot2_sample_ids

    r1_safe = jnp.clip(r1_global, 0, sample_piece_id.shape[0] - 1)
    r2_safe = jnp.clip(r2_global, 0, sample_piece_id.shape[0] - 1)

    ok_r1_ab = r1_valid & valid_a[r1_safe]
    ok_r2_ab = r2_valid & valid_b[r2_safe]

    ok_r1_ba = r1_valid & valid_b[r1_safe]
    ok_r2_ba = r2_valid & valid_a[r2_safe]

    use_ab = robot_a == 1
    ok_r1 = jnp.where(use_ab, ok_r1_ab, ok_r1_ba)
    ok_r2 = jnp.where(use_ab, ok_r2_ab, ok_r2_ba)
    ok_pair = ok_r1[:, None] & ok_r2[None, :] & (~pair_collision)
    return jnp.any(ok_pair)


@jax.jit
def ik_state_feasible_single(
    state,
    sample_piece_id,
    sample_robot_id,
    collision_mask,
    robot1_sample_ids,
    robot2_sample_ids,
    pair_collision,
    num_samples,
    num_r1,
    num_r2,
):
    held_count = jnp.sum(state == 2)

    def case_zero(_):
        return jnp.array(True)

    def case_one(_):
        pid0 = _held_pid_at(state, 0)

        one_r1 = _ik_one_robot_piece_feasible_single(
            state, pid0, jnp.int32(1),
            sample_piece_id, sample_robot_id, collision_mask, num_samples,
        )
        one_r2 = _ik_one_robot_piece_feasible_single(
            state, pid0, jnp.int32(2),
            sample_piece_id, sample_robot_id, collision_mask, num_samples,
        )
        return one_r1 | one_r2

    def case_two(_):
        pid0 = _held_pid_at(state, 0)
        pid1 = _held_pid_at(state, 1)

        pair_12 = _ik_pair_assignment_feasible_single(
            state, pid0, pid1, 1, 2,
            sample_piece_id, sample_robot_id, collision_mask,
            robot1_sample_ids, robot2_sample_ids, pair_collision,
            num_samples, num_r1, num_r2,
        )

        pair_21 = _ik_pair_assignment_feasible_single(
            state, pid0, pid1, 2, 1,
            sample_piece_id, sample_robot_id, collision_mask,
            robot1_sample_ids, robot2_sample_ids, pair_collision,
            num_samples, num_r1, num_r2,
        )

        return pair_12 | pair_21

    def case_bad(_):
        return jnp.array(False)

    return jax.lax.switch(
        jnp.minimum(held_count, 3),
        [case_zero, case_one, case_two, case_bad],
        operand=None,
    )


@jax.jit
def ik_state_feasible_batch_jax(
    part_states,
    ik_sample_piece_id,
    ik_sample_robot_id,
    ik_collision_mask,
    ik_robot1_sample_ids,
    ik_robot2_sample_ids,
    ik_pair_collision,
    ik_num_samples,
    ik_num_r1,
    ik_num_r2,
):
    return jax.vmap(ik_state_feasible_single)(
        part_states,
        ik_sample_piece_id,
        ik_sample_robot_id,
        ik_collision_mask,
        ik_robot1_sample_ids,
        ik_robot2_sample_ids,
        ik_pair_collision,
        ik_num_samples,
        ik_num_r1,
        ik_num_r2,
    )


@partial(jax.jit, static_argnums=(2, 3))
def ik_action_mask_batch_jax(state: SeqEnvState, base_mask, n_actions, n_robot):
    batch, num_parts = state.part_states.shape
    actions = jnp.arange(n_actions, dtype=jnp.int32)

    def one_env_actions(
        part_state,
        base_row,
        sample_piece_id,
        sample_robot_id,
        collision_mask,
        robot1_sample_ids,
        robot2_sample_ids,
        pair_collision,
        num_samples,
        num_r1,
        num_r2,
    ):
        def one_action(a):
            pid_idx = a % num_parts
            is_hold = a < num_parts

            hold_state = part_state.at[pid_idx].set(2)

            hold_ik_ok = ik_state_feasible_single(
                hold_state,
                sample_piece_id,
                sample_robot_id,
                collision_mask,
                robot1_sample_ids,
                robot2_sample_ids,
                pair_collision,
                num_samples,
                num_r1,
                num_r2,
            )

            return jnp.where(is_hold, hold_ik_ok, True)

        ik_ok_all = jax.vmap(one_action)(actions)
        return base_row & ik_ok_all

    return jax.vmap(one_env_actions)(
        state.part_states,
        base_mask,
        state.ik_sample_piece_id,
        state.ik_sample_robot_id,
        state.ik_collision_mask,
        state.ik_robot1_sample_ids,
        state.ik_robot2_sample_ids,
        state.ik_pair_collision,
        state.ik_num_samples,
        state.ik_num_r1,
        state.ik_num_r2,
    )


@jax.jit
def ik_selected_action_legal_batch_jax(state: SeqEnvState, base_mask, actions):
    batch, num_parts = state.part_states.shape

    action_base_legal = jnp.take_along_axis(
        base_mask,
        actions[:, None],
        axis=1,
    )[:, 0]

    pid_idx = actions % num_parts
    is_hold = actions < num_parts

    def one_env(
        part_state,
        pid_i,
        is_h,
        sample_piece_id,
        sample_robot_id,
        collision_mask,
        robot1_sample_ids,
        robot2_sample_ids,
        pair_collision,
        num_samples,
        num_r1,
        num_r2,
    ):
        hold_state = part_state.at[pid_i].set(2)

        hold_ik_ok = ik_state_feasible_single(
            hold_state,
            sample_piece_id,
            sample_robot_id,
            collision_mask,
            robot1_sample_ids,
            robot2_sample_ids,
            pair_collision,
            num_samples,
            num_r1,
            num_r2,
        )

        return jnp.where(is_h, hold_ik_ok, True)

    ik_ok = jax.vmap(one_env)(
        state.part_states,
        pid_idx,
        is_hold,
        state.ik_sample_piece_id,
        state.ik_sample_robot_id,
        state.ik_collision_mask,
        state.ik_robot1_sample_ids,
        state.ik_robot2_sample_ids,
        state.ik_pair_collision,
        state.ik_num_samples,
        state.ik_num_r1,
        state.ik_num_r2,
    )

    return action_base_legal & ik_ok


@jax.jit
def _check_extractable_jax(state_row, pid_idx, insertion_table):
    exist_flag = state_row > 0
    exist_flag = exist_flag.at[pid_idx].set(False)
    tab_i = insertion_table[:, pid_idx, :]
    ok_dir = jnp.all(tab_i | (~exist_flag)[None, :], axis=-1)
    return jnp.any(ok_dir)


@partial(jax.jit, static_argnums=(3,))
def action_is_legal(part_states, insertion_table, action, n_robot):
    num_parts = part_states.shape[0]
    pid_idx = action % num_parts
    is_hold = action < num_parts
    is_remove = action >= num_parts

    n_held = jnp.sum(part_states == 2)

    can_hold_basic = (part_states[pid_idx] == 1) & (n_held < n_robot)
    can_remove = part_states[pid_idx] == 2

    can_extract = _check_extractable_jax(part_states, pid_idx, insertion_table)
    can_hold = can_hold_basic & can_extract

    return jnp.where(is_hold, can_hold, jnp.where(is_remove, can_remove, False))


@partial(jax.jit, static_argnums=(2, 3))
def legal_action_mask_single(part_states, insertion_table, n_actions, n_robot):
    actions = jnp.arange(n_actions, dtype=jnp.int32)
    return jax.vmap(
        lambda a: action_is_legal(part_states, insertion_table, a, n_robot)
    )(actions)


@partial(jax.jit, static_argnums=(2, 3))
def legal_action_mask_batch(part_states, insertion_table, n_actions, n_robot):
    return jax.vmap(
        legal_action_mask_single,
        in_axes=(0, 0, None, None)
    )(part_states, insertion_table, n_actions, n_robot)


@partial(jax.jit, static_argnums=(2,))
def pure_step_jax(state: SeqEnvState, action: int, n_robot: int):
    """Apply geometric hold/remove dynamics without IK or physics evaluation."""
    num_parts = state.part_states.shape[0]
    pid_idx = action % num_parts
    pid = pid_idx + 1

    legal = action_is_legal(
        state.part_states,
        state.insertion_table,
        action,
        n_robot,
    )

    is_hold = action < num_parts
    is_remove = action >= num_parts

    real_mask = jnp.arange(num_parts) < state.n_parts
    old_removed = jnp.sum((state.part_states == 0) & real_mask)

    hold_state = state.part_states.at[pid_idx].set(2)
    remove_state = state.part_states.at[pid_idx].set(0)

    next_part_states = jnp.where(
        legal & is_hold,
        hold_state,
        jnp.where(legal & is_remove, remove_state, state.part_states)
    )

    next_board = jnp.where(
        legal & is_remove,
        jnp.where(state.board == pid, -1, state.board),
        state.board
    )

    new_removed = jnp.sum((next_part_states == 0) & real_mask)
    delta_removed = (new_removed - old_removed).astype(jnp.float32)
    n_parts = jnp.maximum(state.n_parts.astype(jnp.float32), 1.0)
    progress_reward = delta_removed / n_parts

    done_success = new_removed >= state.n_parts
    return next_board, next_part_states, done_success, legal, is_remove, progress_reward


class SequenceEnv:
    """Evaluate hold/remove episodes using offline IK and Newton stability."""

    def __init__(
        self,
        n_robot=2,
        max_shift=10,
        checker_ckpt_path=None,
        checker_threshold=0.5,
        fixed_max_parts=None,
        use_ik=True,
        ik_table_suffix="_ik_table.npz",
        fixed_max_ik_samples=None,
        fixed_max_ik_r1=None,
        fixed_max_ik_r2=None,
    ):
        self.fixed_max_ik_samples = fixed_max_ik_samples
        self.fixed_max_ik_r1 = fixed_max_ik_r1
        self.fixed_max_ik_r2 = fixed_max_ik_r2
        self.use_ik = bool(use_ik)
        self.ik_table_suffix = str(ik_table_suffix)
        self.ik_tables = None
        self.ik_table_cache = {}
        self.n_robot = int(n_robot)
        self.max_shift = int(max_shift)
        self.checker_threshold = float(checker_threshold)
        self.fixed_max_parts = None if fixed_max_parts is None else int(fixed_max_parts)

        self.checker = None
        self.checker_params = None
        if checker_ckpt_path is not None:
            self.load_checker(checker_ckpt_path, threshold=self.checker_threshold)

    def load_checker(self, ckpt_path, threshold=0.5):
        self.checker_threshold = threshold
        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)

        self.checker_params = payload.get("params", payload)
        self.checker = True
        logger.debug(f"[SeqChecker] loaded from {ckpt_path}")

    def _ik_path_from_board_path(self, board_path):
        return resolve_ik_table_path(board_path, self.ik_table_suffix)


    def _load_ik_tables(self, board_paths):
        if board_paths is None:
            if self.use_ik:
                raise ValueError("use_ik=True requires board_paths in reset_batch.")
            self.ik_tables = None
            return

        tables = []
        for p in board_paths:
            ik_path = self._ik_path_from_board_path(p)
            ik_path = Path(ik_path).resolve()

            if not ik_path.exists():
                raise FileNotFoundError(f"Missing IK table for {p}: expected {ik_path}")

            key = str(ik_path)
            if key not in self.ik_table_cache:
                self.ik_table_cache[key] = IKLookupTable(ik_path)

            tables.append(self.ik_table_cache[key])

        self.ik_tables = tables

    def reset_batch(self, boards, board_paths=None):
        """Reset tiled boards; IK mode requires one matching board path per scene."""
        boards_np = np.asarray(boards, dtype=np.int32)
        if boards_np.ndim == 2:
            boards_np = boards_np[None, ...]

        if boards_np.ndim != 3 or boards_np.shape[0] == 0:
            raise ValueError("boards must be a nonempty batch of 2D arrays")
        if any(size == 0 for size in boards_np.shape[1:]):
            raise ValueError("Board dimensions must be positive")
        if self.use_ik and board_paths is not None and len(board_paths) != len(boards_np):
            raise ValueError("board_paths must contain one path per board")

        processed = []
        for b in boards_np:
            b, _ = normalize_piece_ids(b)
            processed.append(b.astype(np.int32))

        if any(not piece_ids_from_board(b) for b in processed):
            raise ValueError("Each sequence board must contain at least one real piece")
        batch_max_parts = max(len(piece_ids_from_board(b)) for b in processed)
        if self.fixed_max_parts is None:
            max_parts = batch_max_parts
        else:
            max_parts = self.fixed_max_parts
            if batch_max_parts > max_parts:
                raise ValueError(
                    f"batch_max_parts={batch_max_parts} exceeds fixed_max_parts={max_parts}"
                )

        batch = len(processed)
        h, w = processed[0].shape

        board_batch = np.full((batch, h, w), -1, dtype=np.int32)
        part_states = np.zeros((batch, max_parts), dtype=np.int32)
        piece_masks = np.zeros((batch, max_parts, h, w), dtype=np.float32)
        contact_matrix = np.zeros((batch, max_parts, max_parts), dtype=np.float32)
        insertion_table = np.ones((batch, 4, max_parts, max_parts), dtype=bool)
        n_parts = np.zeros((batch,), dtype=np.int32)

        for i, b in enumerate(processed):
            pids = piece_ids_from_board(b)
            n = len(pids)

            board_batch[i] = b
            part_states[i, :n] = 1
            n_parts[i] = n

            piece_masks[i] = compute_piece_masks_2d(b, max_parts=max_parts)
            contact_matrix[i] = compute_contact_matrix_2d(b, max_parts=max_parts)
            insertion_table[i] = compute_insertion_table_2d(
                b,
                max_shift=self.max_shift,
                max_parts=max_parts,
            )
        if self.use_ik:
            self._load_ik_tables(board_paths)
        else:
            self.ik_tables = None

        if self.use_ik and self.ik_tables is not None:
            ik_pack = pack_ik_tables_for_jax(
                self.ik_tables,
                max_parts=max_parts,
                max_samples=self.fixed_max_ik_samples,
                max_r1=self.fixed_max_ik_r1,
                max_r2=self.fixed_max_ik_r2,
            )
        else:
            bsz = batch
            s = int(self.fixed_max_ik_samples or 1)
            r1 = int(self.fixed_max_ik_r1 or 1)
            r2 = int(self.fixed_max_ik_r2 or 1)
            ik_pack = {
                "ik_sample_piece_id": np.zeros((bsz, s), dtype=np.int32),
                "ik_sample_robot_id": np.zeros((bsz, s), dtype=np.int32),
                "ik_collision_mask": np.ones((bsz, s, max_parts), dtype=np.bool_),
                "ik_robot1_sample_ids": -np.ones((bsz, r1), dtype=np.int32),
                "ik_robot2_sample_ids": -np.ones((bsz, r2), dtype=np.int32),
                "ik_pair_collision": np.ones((bsz, r1, r2), dtype=np.bool_),
                "ik_num_samples": np.zeros((bsz,), dtype=np.int32),
                "ik_num_r1": np.zeros((bsz,), dtype=np.int32),
                "ik_num_r2": np.zeros((bsz,), dtype=np.int32),
            }
        self.state = SeqEnvState(
            board=jnp.array(board_batch),
            part_states=jnp.array(part_states),
            piece_masks=jnp.array(piece_masks),
            contact_matrix=jnp.array(contact_matrix),
            insertion_table=jnp.array(insertion_table),
            n_parts=jnp.array(n_parts),
            stable=jnp.array([True] * batch),

            ik_sample_piece_id=jnp.array(ik_pack["ik_sample_piece_id"]),
            ik_sample_robot_id=jnp.array(ik_pack["ik_sample_robot_id"]),
            ik_collision_mask=jnp.array(ik_pack["ik_collision_mask"]),
            ik_robot1_sample_ids=jnp.array(ik_pack["ik_robot1_sample_ids"]),
            ik_robot2_sample_ids=jnp.array(ik_pack["ik_robot2_sample_ids"]),
            ik_pair_collision=jnp.array(ik_pack["ik_pair_collision"]),
            ik_num_samples=jnp.array(ik_pack["ik_num_samples"]),
            ik_num_r1=jnp.array(ik_pack["ik_num_r1"]),
            ik_num_r2=jnp.array(ik_pack["ik_num_r2"]),
        )

        self.reset_state = jax.tree_util.tree_map(lambda x: x, self.state)
        return self.reset_state

    def _get_obs(self):
        return jax.tree_util.tree_map(lambda x: x, self.state)

    def ik_action_mask_batch(self, state=None):
        if state is None:
            state = self.state

        n_actions = state.part_states.shape[1] * 2
        base_mask = legal_action_mask_batch(
            state.part_states,
            state.insertion_table,
            n_actions,
            self.n_robot,
        )

        if not self.use_ik:
            return base_mask

        return ik_action_mask_batch_jax(
            state,
            base_mask,
            n_actions,
            self.n_robot,
        )

    def ik_feasible_batch(self, part_states_np):
        part_states = jnp.array(part_states_np, dtype=jnp.int32)

        if not self.use_ik:
            return np.ones((part_states_np.shape[0],), dtype=np.bool_)

        out = ik_state_feasible_batch_jax(
            part_states,
            self.state.ik_sample_piece_id,
            self.state.ik_sample_robot_id,
            self.state.ik_collision_mask,
            self.state.ik_robot1_sample_ids,
            self.state.ik_robot2_sample_ids,
            self.state.ik_pair_collision,
            self.state.ik_num_samples,
            self.state.ik_num_r1,
            self.state.ik_num_r2,
        )
        return np.array(out).astype(np.bool_)

    def step(self, actions, has_any_legal_now=None):
        """Return observation, reward, done, success, raw state, and audit arrays.

        The five audit arrays contain simulated boards, hold masks, Newton
        labels, checker probabilities, and checker labels; unavailable arrays
        are None. Only removal actions trigger Newton simulation.
        """
        if has_any_legal_now is None:
            legal_mask_now = self.ik_action_mask_batch(self.state)
            has_any_legal_now = jnp.any(legal_mask_now, axis=1)

        next_board, next_part_states, done_success, legal, is_remove, progress_reward = jax.vmap(
            pure_step_jax,
            in_axes=(0, 0, None),
        )(self.state, actions, self.n_robot)

        legal_np = np.array(legal).astype(bool)
        is_remove_np = np.array(is_remove).astype(bool)

        next_part_states_np_all = np.array(next_part_states).astype(np.int32)
        ik_ok = self.ik_feasible_batch(next_part_states_np_all)

        legal_after_ik_np = legal_np & ik_ok
        legal_after_ik = jnp.array(legal_after_ik_np)

        next_board = jnp.where(
            legal_after_ik[:, None, None],
            next_board,
            self.state.board,
        )
        next_part_states = jnp.where(
            legal_after_ik[:, None],
            next_part_states,
            self.state.part_states,
        )
        done_success = done_success & legal_after_ik
        is_remove = is_remove & legal_after_ik
        progress_reward = jnp.where(legal_after_ik, progress_reward, 0.0)

        to_simulate = np.array(self.state.stable).astype(bool) & legal_after_ik_np & is_remove_np
        is_stable = np.array(self.state.stable)

        newton_boards = None
        newton_hold_masks = None
        newton_labels = None
        checker_probs = None
        checker_labels = None

        if to_simulate.any():
            boards_np = np.array(next_board)[to_simulate]
            states_np = np.array(next_part_states)[to_simulate]
            full_boards_np = np.array(self.reset_state.board)[to_simulate]

            held_piece_ids = [
                held_piece_ids_from_state(row)
                for row in states_np
            ]

            result = np.array(
                check_stability_newton(
                    boards_np,
                    held_piece_ids=held_piece_ids,
                    render=False,
                    keep_run=False,
                )
            ).astype(np.bool_)

            is_stable[to_simulate] = result
            hold_masks = np.stack([
                hold_mask_from_state(fb, st)
                for fb, st in zip(full_boards_np, states_np)
            ], axis=0).astype(np.float32)

            newton_boards = boards_np.astype(np.int32)
            newton_hold_masks = hold_masks
            newton_labels = result.astype(np.float32)
            if self.checker is not None:
                checker_p = np.array(
                    checker_predict_batch_jax(
                        jnp.array(newton_boards),
                        jnp.array(newton_hold_masks),
                        self.checker_params,
                    )
                ).reshape(-1)

                checker_probs = checker_p.astype(np.float32)
                checker_labels = (checker_p > self.checker_threshold).astype(np.float32)

        is_stable = jnp.array(is_stable)

        n_actions = self.state.part_states.shape[1] * 2
        tmp_next_state = self.state._replace(
            board=next_board,
            part_states=next_part_states,
        )

        next_legal_mask = self.ik_action_mask_batch(tmp_next_state)
        has_any_legal_next = jnp.any(next_legal_mask, axis=1)

        done_unstable = ~is_stable
        done_no_action = (~has_any_legal_now) | (~has_any_legal_next)

        done = done_success | done_unstable | done_no_action
        success = done_success & is_stable & legal_after_ik

        collapse_penalty = 0.02
        success_bonus = 0.2

        base_reward = jnp.where(
            is_stable & legal_after_ik,
            progress_reward,
            0.0,
        )

        collapse_term = jnp.where(
            done_unstable & legal,
            -collapse_penalty,
            0.0,
        )

        success_term = jnp.where(
            success,
            success_bonus,
            0.0,
        )
        reward = base_reward + collapse_term + success_term

        raw_next_state = self.state._replace(
            board=next_board,
            part_states=next_part_states,
            stable=is_stable,
        )

        self.state = raw_next_state

        def select_active_state(reset_val, current_val):
            broadcast_dims = (done.shape[0],) + (1,) * (current_val.ndim - 1)
            return jnp.where(done.reshape(broadcast_dims), reset_val, current_val)

        self.state = jax.tree_util.tree_map(
            select_active_state,
            self.reset_state,
            self.state,
        )

        return (
            self._get_obs(),
            reward,
            done,
            success,
            raw_next_state,
            newton_boards,
            newton_hold_masks,
            newton_labels,
            checker_probs,
            checker_labels,
        )
