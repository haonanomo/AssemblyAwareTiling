"""Batched tiling dynamics and physics-based episode evaluation.

Boards use -1 for blocked cells, 0 for empty target cells, and positive IDs for
placed pieces. Reset methods consume blocked masks decoded from input PNGs;
1 marks blocked cells and 0 marks target cells.
"""

import logging
import pickle
import time
from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from tiling_rl.physics.simulator_newton_til import check_stability_newton
from tiling_rl.sequence.model_seq import AttentionPolicyPatchSeq as SeqAttentionPolicy
from tiling_rl.stability.stability_checker_jax_til import (
    build_checker_model,
    checker_predict_batch_no_hold_jax,
)

logger = logging.getLogger(__name__)


def convert_jax_params_7ch_to_8ch(params):
    """Extend legacy checker input weights with a zero-initialized eighth channel."""
    params = jax.device_get(params)
    new_params = {}

    converted = False

    for module_name, module_params in params.items():
        new_module_params = {}

        for param_name, value in module_params.items():
            arr = np.asarray(value)

            if (
                "input_proj" in module_name
                and param_name == "w"
                and arr.ndim == 4
                and arr.shape[2] == 7
            ):
                new_arr = np.zeros(
                    (arr.shape[0], arr.shape[1], 8, arr.shape[3]),
                    dtype=arr.dtype,
                )
                new_arr[:, :, :7, :] = arr
                new_module_params[param_name] = jnp.asarray(new_arr)
                converted = True
            else:
                new_module_params[param_name] = jnp.asarray(arr)

        new_params[module_name] = new_module_params

    if converted:
        logger.debug("Converted stability checker input from 7 to 8 channels")

    return new_params


def seq_forward_fn(board, part_states, contact_matrix, piece_masks, is_sampling):
    policy = SeqAttentionPolicy()
    return policy(
        board=board,
        part_states=part_states,
        contact_matrix=contact_matrix,
        piece_masks=piece_masks,
        is_sampling=is_sampling,
    )


seq_p_model = hk.transform_with_state(seq_forward_fn)
SEQ_MAX_PARTS = 128


@partial(jax.jit, static_argnums=(1,))
def tiling_board_to_seq_inputs(board, max_parts):
    """Extract batched piece occupancy and contacts for the sequence value model."""
    pids = jnp.arange(1, max_parts + 1, dtype=board.dtype)

    piece_masks = (
        board[:, None, :, :] == pids[None, :, None, None]
    ).astype(jnp.float32)

    part_states = (
        jnp.sum(piece_masks, axis=(2, 3)) > 0
    ).astype(jnp.int32)

    h_contact = jnp.einsum(
        "bpij,bqij->bpq",
        piece_masks[:, :, :, :-1],
        piece_masks[:, :, :, 1:],
    )

    v_contact = jnp.einsum(
        "bpij,bqij->bpq",
        piece_masks[:, :, :-1, :],
        piece_masks[:, :, 1:, :],
    )

    contact = (
        h_contact
        + jnp.swapaxes(h_contact, 1, 2)
        + v_contact
        + jnp.swapaxes(v_contact, 1, 2)
    )

    eye = jnp.eye(max_parts, dtype=contact.dtype)[None, :, :]
    contact = jnp.where(eye > 0, 0.0, contact)
    contact_matrix = (contact > 0).astype(jnp.float32)

    return part_states, contact_matrix, piece_masks


@jax.jit
def seq_value_from_tiling_board(seq_policy_params, seq_policy_states, board):
    """Return assembly values, or ones when the sequence policy is disabled."""
    if seq_policy_params is None or seq_policy_states is None:
        return jnp.ones((board.shape[0],), dtype=jnp.float32)

    part_states, contact_matrix, piece_masks = tiling_board_to_seq_inputs(
        board,
        SEQ_MAX_PARTS,
    )

    (_, values), _ = seq_p_model.apply(
        seq_policy_params,
        seq_policy_states,
        None,
        board=board,
        part_states=part_states,
        contact_matrix=contact_matrix,
        piece_masks=piece_masks,
        is_sampling=False,
    )

    return jnp.clip(jnp.reshape(values, (-1,)), 0.0, 1.0)


class EnvState(NamedTuple):
    """Tiling state; leading batch axes are added by environment reset methods."""

    board: jnp.ndarray
    inventory: jnp.ndarray
    pos: jnp.ndarray
    pieces: jnp.ndarray
    pieces_id: jnp.ndarray
    stable: jnp.ndarray


@jax.jit
def get_board_rb_most_empty_pixel(board):
    """Finds the empty (0) cell with the highest index."""
    empty_mask = (board == 0)
    height, width = board.shape[-2], board.shape[-1]
    indices = jnp.arange(height * width).reshape(height, width)
    weighted_indices = jnp.where(empty_mask, indices, -1)
    target_idx = jnp.max(weighted_indices)
    pos = jnp.array([target_idx // width, target_idx % width])
    return pos


@jax.jit
def get_piece_rb_most_pixel(piece):
    """Finds the bottom-rightmost block of the piece."""
    rows, cols = jnp.where(piece > 0, size=16, fill_value=-1)
    pixel_indices = jnp.where(rows != -1, rows * 4 + cols, -1)
    target_pixel = jnp.max(pixel_indices)
    return target_pixel // 4, target_pixel % 4


@jax.jit
def pure_step_jax(state: EnvState, action: int):
    """Return board, inventory, position, geometric completion, and legality."""
    inventory = state.inventory
    board = state.board
    piece = state.pieces[action, ...]
    pos = state.pos
    piece_id = state.pieces_id[action]

    # Align the bottom-right occupied piece cell with the target position.
    pr, pc = get_piece_rb_most_pixel(piece)
    height, width = board.shape[-2], board.shape[-1]
    br, bc = pos[0], pos[1]
    row_offset, col_offset = br - pr, bc - pc
    row_indices = jnp.arange(height)[:, None] - row_offset
    col_indices = jnp.arange(width)[None, :] - col_offset
    within_piece = (row_indices >= 0) & (row_indices < 4) & \
                   (col_indices >= 0) & (col_indices < 4)
    placed_piece = jnp.where(within_piece, piece[row_indices, col_indices], 0)

    # Validate action
    is_full = (board[br, bc] != 0)
    overlap = jnp.any((board != 0) & (placed_piece > 0))
    out_of_bounds = jnp.sum(placed_piece) < jnp.sum(piece)
    has_inventory = inventory[action] > 0
    illegal = overlap | out_of_bounds | is_full | (~has_inventory)

    # All orientations of a piece share the same inventory count.
    curr_step = jnp.max(board)
    new_board = jnp.where(illegal, board, board + placed_piece * (curr_step + 1))
    mask = (state.pieces_id == piece_id)
    new_inv = jnp.where(mask, inventory - 1, inventory)
    new_inv = jnp.where(illegal, inventory, new_inv)

    indices = jnp.arange(height * width).reshape(height, width)
    temp_board = jnp.where(indices < indices[pos[0], pos[1]], new_board, -1)
    new_pos = get_board_rb_most_empty_pixel(temp_board)

    done = (temp_board[new_pos[0], new_pos[1]] != 0)

    return new_board, new_inv, new_pos, done, ~illegal


@jax.jit
def action_is_legal(board, inventory_count, pos, piece):
    pr, pc = get_piece_rb_most_pixel(piece)
    height, width = board.shape[-2], board.shape[-1]
    br, bc = pos[0], pos[1]

    row_offset, col_offset = br - pr, bc - pc
    row_indices = jnp.arange(height)[:, None] - row_offset
    col_indices = jnp.arange(width)[None, :] - col_offset

    within_piece = (
        (row_indices >= 0) & (row_indices < 4) &
        (col_indices >= 0) & (col_indices < 4)
    )
    placed_piece = jnp.where(within_piece, piece[row_indices, col_indices], 0)

    is_full = (board[br, bc] != 0)
    overlap = jnp.any((board != 0) & (placed_piece > 0))
    out_of_bounds = jnp.sum(placed_piece) < jnp.sum(piece)
    has_inventory = inventory_count > 0

    illegal = overlap | out_of_bounds | is_full | (~has_inventory)
    return ~illegal


@jax.jit
def legal_action_mask_single(board, inventory, pos, pieces):
    return jax.vmap(
        action_is_legal,
        in_axes=(None, 0, None, 0)
    )(board, inventory, pos, pieces)


@jax.jit
def legal_action_mask_batch(board, inventory, pos, pieces):
    return jax.vmap(
        legal_action_mask_single,
        in_axes=(0, 0, 0, 0)
    )(board, inventory, pos, pieces)


def canonicalize_blocked_mask(mask, board_size=(16, 16)):
    """Crop a blocked mask to its target bounds and align it to the bottom right."""
    mask = np.asarray(mask, dtype=np.int32)
    if mask.ndim != 2:
        raise ValueError("Blocked mask must be a 2D array")

    valid = (mask == 0)
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Mask has no valid region (no zeros found).")

    cropped = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    H, W = board_size
    h, w = cropped.shape
    if h > H or w > W:
        raise ValueError(f"Cropped mask shape {(h, w)} exceeds board size {(H, W)}")

    out = np.ones((H, W), dtype=np.int32)
    out[H - h:, W - w:] = cropped
    return jnp.array(out)


def build_multi_board_batch_shared_inventory(shape_specs,
                                             board_size=(16, 16),
                                             shapes_per_iter=4,
                                             envs_per_shape=4,
                                             rng=None,
                                             shape_probs=None):
    """Sample distinct PNG shape specs and repeat each mask for parallel episodes."""
    if shapes_per_iter < 1 or envs_per_shape < 1:
        raise ValueError("shapes_per_iter and envs_per_shape must be positive")
    if rng is None:
        rng = np.random.default_rng()

    if len(shape_specs) < shapes_per_iter:
        raise ValueError("Not enough shapes in shape_specs")

    if shape_probs is None:
        chosen_ids = rng.choice(len(shape_specs), size=shapes_per_iter, replace=False)
    else:
        chosen_ids = rng.choice(len(shape_specs), size=shapes_per_iter, replace=False, p=shape_probs)

    blocked_masks = []
    shape_names = []

    for sid in chosen_ids:
        spec = shape_specs[sid]
        canon_mask = canonicalize_blocked_mask(spec["mask"], board_size=board_size)

        for _ in range(envs_per_shape):
            blocked_masks.append(canon_mask)
            shape_names.append(spec["name"])

    blocked_mask_batch = jnp.stack(blocked_masks, axis=0)

    return {
        "blocked_mask": blocked_mask_batch,
        "shape_names": shape_names,
    }


class TetrixEnv:
    """Batched environment that automatically resets terminated episodes."""

    def __init__(
        self,
        board_size=(6, 6),
        checker_ckpt_path=None,
        seq_policy_params=None,
        seq_policy_states=None,
    ):
        super().__init__()
        self.board_size = tuple(board_size)

        self.checker = None
        self.checker_threshold = 0.5
        self.last_step_timing = {"newton": 0.0, "sequence": 0.0, "total": 0.0}

        self.seq_policy_params = seq_policy_params
        self.seq_policy_states = seq_policy_states

        if checker_ckpt_path is not None:
            self.load_checker(checker_ckpt_path)

    def set_seq_policy(self, seq_policy_params, seq_policy_states):
        self.seq_policy_params = seq_policy_params
        self.seq_policy_states = seq_policy_states

    def load_checker(self, ckpt_path, threshold=0.5):
        self.checker_threshold = threshold

        with open(ckpt_path, "rb") as f:
            payload = pickle.load(f)

        params = payload["params"]
        params = convert_jax_params_7ch_to_8ch(params)

        self.checker_params = params
        self.checker_model = build_checker_model()
        self.checker = True

        logger.debug("Loaded stability checker: %s", ckpt_path)

    def step(self, actions):
        """Place pieces, evaluate rewards, and reset completed episodes.

        Returns observation, reward, done, success, simulated boards, physics
        labels, checker probabilities, and checker labels. Diagnostic arrays
        are None when no boards were evaluated by the corresponding checker.
        """
        step_t0 = time.perf_counter()
        newton_time = 0.0
        sequence_time = 0.0

        legal_mask_now = legal_action_mask_batch(
            self.state.board,
            self.state.inventory,
            self.state.pos,
            self.state.pieces
        )
        has_any_legal_now = jnp.any(legal_mask_now, axis=1)
        new_board, new_inv, new_pos, done, legal = jax.vmap(pure_step_jax)(self.state, actions)

        # Newton determines stability; the learned checker is audited against it.
        to_simulate = np.array(self.state.stable & legal)
        is_stable = np.array(self.state.stable)

        newton_boards = None
        newton_labels = None
        checker_probs = None
        checker_labels = None

        if to_simulate.any():
            to_check_board = np.array(new_board)[to_simulate, ...].astype(np.int32)

            t0 = time.perf_counter()
            result = np.array(
                check_stability_newton(to_check_board, render=False)
            ).astype(np.bool_)
            newton_time += time.perf_counter() - t0

            is_stable[to_simulate] = result

            newton_boards = to_check_board
            newton_labels = result.astype(np.float32)

            if self.checker is not None:
                checker_p = np.array(
                    checker_predict_batch_no_hold_jax(
                        jnp.array(to_check_board),
                        self.checker_params,
                    )
                ).reshape(-1)

                checker_probs = checker_p.astype(np.float32)
                checker_labels = (checker_p > self.checker_threshold).astype(np.float32)


        is_stable = jnp.array(is_stable)
        done_geom = done
        done_unstable = ~is_stable
        done_uncover = ~has_any_legal_now
        done = done_geom | done_unstable | done_uncover
        success = done_geom & is_stable & (~done_uncover)

        self.state = self.state._replace(board=new_board,
                                         pos=new_pos,
                                         inventory=new_inv,
                                         stable=is_stable)

        # Scale placed area by the assembly value estimate.
        piece_pixels = jnp.sum(self.state.pieces, axis=(-2, -1))
        total_valid_pixels = jnp.sum(new_board != -1, axis=(-2, -1))
        pieces_reward = piece_pixels / jnp.maximum(total_valid_pixels[..., jnp.newaxis], 1)
        batch_indices = jnp.arange(actions.shape[0])

        if self.seq_policy_params is not None and self.seq_policy_states is not None:
            t0 = time.perf_counter()
            seq_value = seq_value_from_tiling_board(
                self.seq_policy_params,
                self.seq_policy_states,
                new_board,
            )
            seq_value.block_until_ready()
            sequence_time += time.perf_counter() - t0
        else:
            seq_value = jnp.ones((actions.shape[0],), dtype=jnp.float32)

        collapse_penalty = 0.02
        uncover_penalty = 0.02
        success_bonus = 0.2

        base_reward = jnp.where(
            is_stable & legal & (~done_uncover),
            seq_value * pieces_reward[batch_indices, actions],
            0.0
        )
        collapse_term = jnp.where(done_unstable & legal, -collapse_penalty, 0.0)
        uncover_term = jnp.where(done_uncover, -uncover_penalty, 0.0)
        success_term = jnp.where(success, success_bonus, 0.0)
        reward = base_reward + collapse_term + uncover_term + success_term

        def select_active_state(reset_val, current_val):
            broadcast_dims = (done.shape[0],) + (1,) * (current_val.ndim - 1)
            return jnp.where(done.reshape(broadcast_dims), reset_val, current_val)

        # Apply to the whole state pytree
        self.state = jax.tree_util.tree_map(
            select_active_state,
            self.reset_state,
            self.state
        )
        self.last_step_timing = {
            "total": float(time.perf_counter() - step_t0),
            "newton": float(newton_time),
            "sequence": float(sequence_time),
        }
        return self._get_obs(), reward, done, success, newton_boards, newton_labels, checker_probs, checker_labels

    def _get_obs(self):
        state_copy = jax.tree_util.tree_map(lambda x: x, self.state)
        return state_copy

    def reset(self,
              pieces: jnp.ndarray,
              pieces_id: jnp.ndarray,
              inventory: jnp.ndarray,
              blocked_mask: jnp.ndarray,
              num_envs: int):
        """Reset parallel episodes to the same decoded PNG mask and inventory."""
        inventory = inventory[jnp.newaxis, ...].repeat(num_envs, axis=0)
        pieces = pieces[jnp.newaxis, ...].repeat(num_envs, axis=0)
        pieces_id = pieces_id[jnp.newaxis, ...].repeat(num_envs, axis=0)
        blocked_mask = blocked_mask[jnp.newaxis, ...].repeat(num_envs, axis=0)

        num_env = inventory.shape[0]
        board = jnp.zeros((num_env, *self.board_size), dtype=jnp.int32)
        board = jnp.where(blocked_mask > 0, -1, board)

        self.state = EnvState(
            board=board,
            pieces=pieces,
            pieces_id=pieces_id,
            inventory=inventory,
            pos=jax.vmap(get_board_rb_most_empty_pixel)(board),
            stable=jnp.array([True] * num_env)
        )
        self.reset_state = jax.tree_util.tree_map(lambda x: x, self.state)

        return self.reset_state

    def reset_batch(self,
                    pieces: jnp.ndarray,
                    pieces_id: jnp.ndarray,
                    inventory: jnp.ndarray,
                    blocked_mask: jnp.ndarray):
        """Reset one episode per decoded mask, sharing the initial piece inventory."""
        num_env = blocked_mask.shape[0]

        inventory = inventory[jnp.newaxis, ...].repeat(num_env, axis=0)
        pieces = pieces[jnp.newaxis, ...].repeat(num_env, axis=0)
        pieces_id = pieces_id[jnp.newaxis, ...].repeat(num_env, axis=0)

        board = jnp.zeros((num_env, *self.board_size), dtype=jnp.int32)
        board = jnp.where(blocked_mask > 0, -1, board)

        self.state = EnvState(
            board=board,
            pieces=pieces,
            pieces_id=pieces_id,
            inventory=inventory,
            pos=jax.vmap(get_board_rb_most_empty_pixel)(board),
            stable=jnp.array([True] * num_env)
        )
        self.reset_state = jax.tree_util.tree_map(lambda x: x, self.state)
        return self.reset_state
