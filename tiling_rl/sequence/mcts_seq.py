"""MCTS, policy optimization, and evaluation for sequence policies.

Dataset selection and runtime configuration live in train_seq.py.
"""

import copy
import logging
import os
import pickle
import random
import time
from collections import deque
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np
import optax

from tiling_rl.sequence.env_seq import (
    SequenceEnv,
    pure_step_jax,
    legal_action_mask_batch,
    normalize_piece_ids,
    piece_ids_from_board,
    held_piece_ids_from_state,
    ik_action_mask_batch_jax,
    ik_selected_action_legal_batch_jax,
)
from tiling_rl.sequence.model_seq import AttentionPolicyPatchSeq as AttentionPolicy
from tiling_rl.sequence.ik_lookup import resolve_ik_table_path
from tiling_rl.physics.simulator_newton_seq import check_stability_newton
from tiling_rl.stability.stability_checker_jax_seq import checker_predict_batch_jax as checker_predict_seq_batch_jax

logger = logging.getLogger(__name__)


def compute_global_ik_padding(npy_files):
    """Return maximum table sizes for fixed JAX batch shapes."""
    max_samples = 1
    max_r1 = 1
    max_r2 = 1

    for p in npy_files:
        ik_path = resolve_ik_table_path(p)
        with np.load(ik_path) as data:
            max_samples = max(max_samples, int(data["sample_piece_id"].shape[0]))
            max_r1 = max(max_r1, int(data["robot1_sample_ids"].shape[0]))
            max_r2 = max(max_r2, int(data["robot2_sample_ids"].shape[0]))

    return max_samples, max_r1, max_r2


def p_forward_fn(board, part_states, contact_matrix, piece_masks, is_sampling):
    """Apply the default sequence policy and value network."""
    policy = AttentionPolicy()
    return policy(
        board=board,
        part_states=part_states,
        contact_matrix=contact_matrix,
        piece_masks=piece_masks,
        is_sampling=is_sampling,
    )


p_model = hk.transform_with_state(p_forward_fn)


def save_checkpoint(path, policy_params, policy_states, extra=None):
    """Save policy weights, model state, and optional iteration metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_params": policy_params,
        "policy_states": policy_states,
        "extra": extra,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path):
    """Load policy weights, model state, and optional metadata."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["policy_params"], payload["policy_states"], payload.get("extra", None)


def hold_mask_batch_from_state(board, part_states):
    """Mark held real pieces and virtual zero-region supports."""
    bsz, h, w = board.shape
    num_parts = part_states.shape[1]

    pids = jnp.arange(1, num_parts + 1, dtype=board.dtype)
    held = part_states == 2

    eq = board[..., None] == pids[None, None, None, :]
    real_held_mask = jnp.any(eq & held[:, None, None, :], axis=-1)

    virtual_held_mask = board == 0
    mask = real_held_mask | virtual_held_mask
    mask = jnp.where(board >= 0, mask, False)

    return mask.astype(jnp.float32)


def count_real_parts(board):
    board, _ = normalize_piece_ids(board)
    return len(piece_ids_from_board(board))


def compute_global_step_per_iterations(npy_files):
    """Return a rollout horizon of twice the maximum real-piece count."""
    max_parts = 0

    for p in npy_files:
        board = np.load(p).astype(np.int32)
        n_parts = count_real_parts(board)
        max_parts = max(max_parts, n_parts)

    step_per_iterations = 2 * max(1, max_parts)
    return step_per_iterations, max_parts


def rollout_and_validate(
    policy_params,
    policy_states,
    board_np,
    n_robot=2,
    max_shift=10,
    mode="policy_only",
    checker_params=None,
    rng_key=None,
    fixed_max_parts=None,
    board_path=None,
    fixed_max_ik_samples=None,
    fixed_max_ik_r1=None,
    fixed_max_ik_r2=None,
    *,
    p_model=p_model,
):
    """Roll out one board with IK constraints and validate removals with Newton."""
    eval_env = SequenceEnv(
        n_robot=n_robot,
        max_shift=max_shift,
        checker_ckpt_path=None,
        fixed_max_parts=fixed_max_parts,
        fixed_max_ik_samples=fixed_max_ik_samples,
        fixed_max_ik_r1=fixed_max_ik_r1,
        fixed_max_ik_r2=fixed_max_ik_r2,
    )
    eval_env.reset_batch(boards=[board_np], board_paths=[board_path] if board_path is not None else None)
    obs = eval_env._get_obs()

    n_parts = int(obs.n_parts[0])
    max_steps = max(1, 2 * n_parts)

    remove_boards = []
    remove_held_piece_ids = []
    remove_pre_states = []
    remove_post_states = []

    final_part_state = np.array(obs.part_states[0]).astype(np.int32)
    done_by_policy = False

    forward_time = 0.0
    sim_time = 0.0

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    for _ in range(max_steps):
        n_actions = obs.part_states.shape[1] * 2
        legal_mask = eval_env.ik_action_mask_batch(obs)

        if not bool(jnp.any(legal_mask[0])):
            break

        forward_t0 = time.perf_counter()

        if mode == "policy_only":
            (logits, _), _ = p_model.apply(
                policy_params,
                policy_states,
                None,
                board=obs.board,
                part_states=obs.part_states,
                contact_matrix=obs.contact_matrix,
                piece_masks=obs.piece_masks,
                is_sampling=False,
            )

            logits = jnp.where(legal_mask, logits, -1e9)
            action = int(jnp.argmax(logits[0]))
            actions = jnp.array([action], dtype=jnp.int32)

        elif mode == "mcts":
            if checker_params is None:
                raise ValueError("mode='mcts' requires checker_params")

            rng_key, subkey = jax.random.split(rng_key)

            actions, _ = mcts_action(
                policy_params,
                policy_states,
                subkey,
                obs,
                checker_params,
                n_robot,
                False,
                p_model=p_model,
            )

            actions = jnp.asarray(actions, dtype=jnp.int32)

        else:
            raise ValueError(f"Unknown eval mode: {mode}")

        next_board, next_part_states, done_success, legal, is_remove, _ = jax.vmap(
            pure_step_jax,
            in_axes=(0, 0, None),
        )(obs, actions, n_robot)

        next_board.block_until_ready()
        forward_time += time.perf_counter() - forward_t0

        if not bool(legal[0]):
            break

        pre_state_np = np.array(obs.part_states[0]).astype(np.int32)
        post_state_np = np.array(next_part_states[0]).astype(np.int32)

        if bool(is_remove[0]):
            nb = np.array(next_board[0]).astype(np.int32)
            held_ids = held_piece_ids_from_state(post_state_np)

            remove_boards.append(nb)
            remove_held_piece_ids.append(held_ids)
            remove_pre_states.append(pre_state_np)
            remove_post_states.append(post_state_np)

        final_part_state = post_state_np

        obs = obs._replace(
            board=next_board,
            part_states=next_part_states,
            stable=jnp.array([True]),
        )

        if bool(done_success[0]):
            done_by_policy = True
            break

    if len(remove_boards) == 0:
        progress = float(np.sum(final_part_state[:n_parts] == 0) / max(1, n_parts))
        return {
            "success": False,
            "progress": progress,
            "n_parts": n_parts,
            "num_remove": 0,
            "num_stable_remove": 0,
            "num_unstable_remove": 0,
            "forward_time": float(forward_time),
            "sim_time": float(sim_time),
        }

    boards_np = np.stack(remove_boards, axis=0).astype(np.int32)

    sim_t0 = time.perf_counter()
    labels = np.array(
        check_stability_newton(
            boards_np,
            held_piece_ids=remove_held_piece_ids,
            render=False,
            keep_run=False,
        )
    ).astype(np.bool_)
    sim_time += time.perf_counter() - sim_t0

    unstable_idx = np.where(~labels)[0]

    if len(unstable_idx) > 0:
        first_bad = int(unstable_idx[0])
        valid_state = remove_pre_states[first_bad]
        success = False
    else:
        valid_state = final_part_state
        success = bool(done_by_policy and np.all(final_part_state[:n_parts] == 0))

    progress = float(np.sum(valid_state[:n_parts] == 0) / max(1, n_parts))

    return {
        "success": success,
        "progress": progress,
        "n_parts": n_parts,
        "num_remove": int(len(remove_boards)),
        "num_stable_remove": int(np.sum(labels)),
        "num_unstable_remove": int(np.sum(~labels)),
        "forward_time": float(forward_time),
        "sim_time": float(sim_time),
    }


def evaluate_policy_only_batched(
    policy_params,
    policy_states,
    npy_files,
    iteration,
    n_robot=2,
    max_shift=10,
    fixed_max_parts=None,
    board_cache=None,
    fixed_max_ik_samples=None,
    fixed_max_ik_r1=None,
    fixed_max_ik_r2=None,
    *,
    p_model=p_model,
):
    """Evaluate board/IK pairs together using greedy policy actions."""
    eval_t0 = time.perf_counter()
    forward_time = 0.0
    sim_time = 0.0

    if board_cache is None:
        boards = [np.load(p).astype(np.int32) for p in npy_files]
    else:
        boards = [board_cache[p] for p in npy_files]

    names = [Path(p).stem for p in npy_files]

    eval_env = SequenceEnv(
        n_robot=n_robot,
        max_shift=max_shift,
        checker_ckpt_path=None,
        fixed_max_parts=fixed_max_parts,
        fixed_max_ik_samples=fixed_max_ik_samples,
        fixed_max_ik_r1=fixed_max_ik_r1,
        fixed_max_ik_r2=fixed_max_ik_r2,
    )
    eval_env.reset_batch(boards=boards, board_paths=npy_files)
    obs = eval_env._get_obs()

    num_shapes = len(boards)
    n_parts_np = np.array(obs.n_parts).astype(np.int32)
    max_steps = int(2 * max(1, int(np.max(n_parts_np))))

    active = np.ones((num_shapes,), dtype=bool)
    done_by_policy = np.zeros((num_shapes,), dtype=bool)
    final_part_states = np.array(obs.part_states).astype(np.int32)

    remove_boards = []
    remove_held_piece_ids = []
    remove_pre_states = []
    remove_shape_ids = []

    for _ in range(max_steps):
        if not np.any(active):
            break

        n_actions = obs.part_states.shape[1] * 2

        forward_t0 = time.perf_counter()

        active_jnp = jnp.array(active)
        legal_mask = eval_env.ik_action_mask_batch(obs)
        legal_mask = legal_mask & active_jnp[:, None]
        has_legal = jnp.any(legal_mask, axis=1)

        if not bool(jnp.any(has_legal)):
            break

        (logits, _), _ = p_model.apply(
            policy_params,
            policy_states,
            None,
            board=obs.board,
            part_states=obs.part_states,
            contact_matrix=obs.contact_matrix,
            piece_masks=obs.piece_masks,
            is_sampling=False,
        )

        logits = jnp.where(legal_mask, logits, -1e9)
        actions = jnp.argmax(logits, axis=1).astype(jnp.int32)

        next_board, next_part_states, done_success, legal, is_remove, _ = jax.vmap(
            pure_step_jax,
            in_axes=(0, 0, None),
        )(obs, actions, n_robot)

        next_board.block_until_ready()
        forward_time += time.perf_counter() - forward_t0

        legal_np = np.array(legal).astype(bool)
        is_remove_np = np.array(is_remove).astype(bool)
        done_success_np = np.array(done_success).astype(bool)

        prev_part_states_np = np.array(obs.part_states).astype(np.int32)
        next_part_states_np = np.array(next_part_states).astype(np.int32)
        next_board_np = np.array(next_board).astype(np.int32)

        has_legal_np = np.array(has_legal).astype(bool)
        step_valid = active & has_legal_np & legal_np

        for i in np.where(step_valid & is_remove_np)[0]:
            remove_boards.append(next_board_np[i])
            remove_held_piece_ids.append(held_piece_ids_from_state(next_part_states_np[i]))
            remove_pre_states.append(prev_part_states_np[i].copy())
            remove_shape_ids.append(int(i))

        final_part_states[step_valid] = next_part_states_np[step_valid]

        done_by_policy = done_by_policy | (step_valid & done_success_np)

        step_valid_mask = jnp.array(step_valid)
        obs = obs._replace(
            board=jnp.where(step_valid_mask[:, None, None], next_board, obs.board),
            part_states=jnp.where(step_valid_mask[:, None], next_part_states, obs.part_states),
            stable=jnp.ones((num_shapes,), dtype=bool),
        )

        active = active & has_legal_np & legal_np & (~done_success_np)

    labels = None
    if len(remove_boards) > 0:
        boards_np = np.stack(remove_boards, axis=0).astype(np.int32)

        sim_t0 = time.perf_counter()
        labels = np.array(
            check_stability_newton(
                boards_np,
                held_piece_ids=remove_held_piece_ids,
                render=False,
                keep_run=False,
            )
        ).astype(np.bool_)
        sim_time += time.perf_counter() - sim_t0

    valid_part_states = final_part_states.copy()

    if labels is not None:
        first_bad_by_shape = {}
        for k, stable in enumerate(labels):
            if not stable:
                sid = remove_shape_ids[k]
                if sid not in first_bad_by_shape:
                    first_bad_by_shape[sid] = k

        for sid, k in first_bad_by_shape.items():
            valid_part_states[sid] = remove_pre_states[k]

    results = []
    for i in range(num_shapes):
        n = int(n_parts_np[i])
        progress = float(np.sum(valid_part_states[i, :n] == 0) / max(1, n))

        if labels is None:
            shape_remove_idx = []
        else:
            shape_remove_idx = [k for k, sid in enumerate(remove_shape_ids) if sid == i]

        if len(shape_remove_idx) == 0:
            stable_count = 0
            unstable_count = 0
        else:
            shape_labels = labels[shape_remove_idx]
            stable_count = int(np.sum(shape_labels))
            unstable_count = int(np.sum(~shape_labels))

        success = bool(
            done_by_policy[i]
            and unstable_count == 0
            and np.all(valid_part_states[i, :n] == 0)
        )

        results.append({
            "name": names[i],
            "success": success,
            "progress": progress,
            "n_parts": n,
            "num_remove": int(len(shape_remove_idx)),
            "num_stable_remove": stable_count,
            "num_unstable_remove": unstable_count,
            "forward_time": 0.0,
            "sim_time": 0.0,
        })

    num_success = sum(int(r["success"]) for r in results)
    progresses = np.array([r["progress"] for r in results], dtype=np.float32)

    total_remove = sum(r["num_remove"] for r in results)
    total_stable_remove = sum(r["num_stable_remove"] for r in results)
    total_unstable_remove = sum(r["num_unstable_remove"] for r in results)

    eval_time = time.perf_counter() - eval_t0
    other_time = eval_time - forward_time - sim_time

    logger.debug(
        f"[policy_only_batched Eval iter={iteration:04d}] "
        f"shapes={num_shapes} | "
        f"success={num_success}/{num_shapes} ({num_success / max(1, num_shapes) * 100:.2f}%) | "
        f"mean_progress={float(np.mean(progresses)):.4f} | "
        f"min_progress={float(np.min(progresses)):.4f} | "
        f"max_progress={float(np.max(progresses)):.4f} | "
        f"remove_states={total_remove} | "
        f"stable_remove={total_stable_remove} | "
        f"unstable_remove={total_unstable_remove} | "
        f"time={eval_time:.2f}s | "
        f"forward_time={forward_time:.2f}s | "
        f"sim_time={sim_time:.2f}s | "
        f"other_time={other_time:.2f}s"
    )

    return results


def evaluate_all_shapes(
    policy_params,
    policy_states,
    npy_files,
    iteration,
    n_robot=2,
    max_shift=10,
    mode="policy_only",
    checker_params=None,
    fixed_max_parts=None,
    board_cache=None,
    fixed_max_ik_samples=None,
    fixed_max_ik_r1=None,
    fixed_max_ik_r2=None,
    *,
    p_model=p_model,
):
    """Evaluate explicit board files with greedy policy or MCTS actions."""
    if mode == "policy_only":
        return evaluate_policy_only_batched(
            policy_params=policy_params,
            policy_states=policy_states,
            npy_files=npy_files,
            iteration=iteration,
            n_robot=n_robot,
            max_shift=max_shift,
            fixed_max_parts=fixed_max_parts,
            board_cache=board_cache,
            fixed_max_ik_samples=fixed_max_ik_samples,
            fixed_max_ik_r1=fixed_max_ik_r1,
            fixed_max_ik_r2=fixed_max_ik_r2,
            p_model=p_model,
        )

    eval_t0 = time.perf_counter()
    rng_key = jax.random.PRNGKey(10000 + int(iteration))

    results = []
    for i, p in enumerate(npy_files, start=1):
        board_np = np.load(p).astype(np.int32)

        rng_key, subkey = jax.random.split(rng_key)

        r = rollout_and_validate(
            policy_params=policy_params,
            policy_states=policy_states,
            board_np=board_np,
            n_robot=n_robot,
            max_shift=max_shift,
            mode=mode,
            checker_params=checker_params,
            rng_key=subkey,
            fixed_max_parts=fixed_max_parts,
            board_path=p,
            fixed_max_ik_samples=fixed_max_ik_samples,
            fixed_max_ik_r1=fixed_max_ik_r1,
            fixed_max_ik_r2=fixed_max_ik_r2,
            p_model=p_model,
        )

        r["name"] = Path(p).stem
        results.append(r)

    num_shapes = len(results)
    num_success = sum(int(r["success"]) for r in results)
    progresses = np.array([r["progress"] for r in results], dtype=np.float32)

    total_remove = sum(r["num_remove"] for r in results)
    total_stable_remove = sum(r["num_stable_remove"] for r in results)
    total_unstable_remove = sum(r["num_unstable_remove"] for r in results)

    total_forward_time = sum(float(r.get("forward_time", 0.0)) for r in results)
    total_sim_time = sum(float(r.get("sim_time", 0.0)) for r in results)

    eval_time = time.perf_counter() - eval_t0
    other_time = eval_time - total_forward_time - total_sim_time

    logger.debug(
        f"[{mode} Eval iter={iteration:04d}] "
        f"shapes={num_shapes} | "
        f"success={num_success}/{num_shapes} ({num_success / max(1, num_shapes) * 100:.2f}%) | "
        f"mean_progress={float(np.mean(progresses)):.4f} | "
        f"min_progress={float(np.min(progresses)):.4f} | "
        f"max_progress={float(np.max(progresses)):.4f} | "
        f"remove_states={total_remove} | "
        f"stable_remove={total_stable_remove} | "
        f"unstable_remove={total_unstable_remove} | "
        f"time={eval_time:.2f}s | "
        f"forward_time={total_forward_time:.2f}s | "
        f"sim_time={total_sim_time:.2f}s | "
        f"other_time={other_time:.2f}s"
    )

    return results


@partial(jax.jit, static_argnames=("optimizer", "n_robot", "weights", "p_model"))
def train_step(
    optimizer,
    params,
    state,
    opt_state,
    board,
    ps,
    pm,
    cm,
    it,
    pr,
    vs,
    lm,
    n_robot,
    weights=(1.0, 1.0),
    *,
    p_model=p_model,
):
    """Update policy/value heads from search distributions and episode returns."""
    def loss_fn(params):
        (logits, values), new_state = p_model.apply(
            params,
            state,
            None,
            board=board,
            part_states=ps,
            contact_matrix=cm,
            piece_masks=pm,
            is_sampling=False,
        )

        legal_mask = lm.astype(jnp.bool_)
        has_any_legal = jnp.any(legal_mask, axis=1)
        logits = jnp.where(legal_mask, logits, -1e9)

        policy_ce = optax.softmax_cross_entropy(logits, pr)
        policy_loss = jnp.where(
            jnp.any(has_any_legal),
            jnp.sum(policy_ce * has_any_legal.astype(jnp.float32)) /
            jnp.sum(has_any_legal.astype(jnp.float32)),
            0.0,
        )

        v_min = -0.02
        v_max = 1.2
        vs_scaled = (vs - v_min) / (v_max - v_min)
        vs_scaled = jnp.clip(vs_scaled, 0.0, 1.0)
        value_loss = optax.squared_error(values, vs_scaled).mean()

        probs = jax.nn.softmax(logits)
        probs = jnp.where(has_any_legal[:, None], probs, 0.0)
        entropy_per_sample = jnp.where(
            has_any_legal,
            -jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1),
            0.0,
        )
        entropy = jnp.where(
            jnp.any(has_any_legal),
            jnp.sum(entropy_per_sample) / jnp.sum(has_any_legal.astype(jnp.float32)),
            0.0,
        )

        total_loss = weights[0] * policy_loss + weights[1] * value_loss
        return total_loss, (new_state, policy_loss, value_loss, entropy)

    grads, (new_state, policy_loss, value_loss, entropy) = jax.grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state, policy_loss, value_loss, entropy


@partial(jax.jit, static_argnames=("n_robot", "is_sampling", "p_model"))
def mcts_action(policy_params, policy_states, rng_key, env_state, checker_params,
                n_robot, is_sampling=False, *, p_model=p_model):
    """Search hold/remove actions using IK constraints and learned stability."""
    def recurrent_fn(p, key, actions, e_state):
        n_actions = e_state.part_states.shape[1] * 2
        base_legal_mask_now = legal_action_mask_batch(
            e_state.part_states,
            e_state.insertion_table,
            n_actions,
            n_robot,
        )

        action_ik_legal = ik_selected_action_legal_batch_jax(
            e_state,
            base_legal_mask_now,
            actions,
        )

        has_any_legal_now = jnp.any(base_legal_mask_now, axis=1)

        next_board, next_part_states, next_done_success, next_legal, is_remove, progress_reward = jax.vmap(
            pure_step_jax,
            in_axes=(0, 0, None),
        )(e_state, actions, n_robot)

        next_legal = next_legal & action_ik_legal

        next_board = jnp.where(
            next_legal[:, None, None],
            next_board,
            e_state.board,
        )
        next_part_states = jnp.where(
            next_legal[:, None],
            next_part_states,
            e_state.part_states,
        )
        next_done_success = next_done_success & next_legal
        is_remove = is_remove & next_legal
        progress_reward = jnp.where(next_legal, progress_reward, 0.0)

        tmp_next_state = e_state._replace(
            board=next_board,
            part_states=next_part_states,
        )
        base_next_legal_mask = legal_action_mask_batch(
            next_part_states,
            e_state.insertion_table,
            n_actions,
            n_robot,
        )
        next_legal_mask = ik_action_mask_batch_jax(
            tmp_next_state,
            base_next_legal_mask,
            n_actions,
            n_robot,
        )
        next_has_any_legal = jnp.any(next_legal_mask, axis=1)
        done_no_action = (~has_any_legal_now) | (~next_has_any_legal)

        (logits, val), _ = p_model.apply(
            p,
            policy_states,
            key,
            board=next_board,
            part_states=next_part_states,
            contact_matrix=e_state.contact_matrix,
            piece_masks=e_state.piece_masks,
            is_sampling=is_sampling,
        )
        logits = jnp.where(next_legal_mask, logits, -1e9)

        hold_mask = hold_mask_batch_from_state(next_board, next_part_states)
        stability_p = checker_predict_seq_batch_jax(next_board, hold_mask, checker_params)
        checker_is_stable = stability_p > 0.5

        checked_stable = jnp.where(is_remove, checker_is_stable, e_state.stable)

        collapse_penalty = 0.02
        success_bonus = 0.2

        base_reward = jnp.where(
            next_legal,
            jnp.where(is_remove, stability_p * progress_reward, progress_reward),
            0.0,
        )

        done_unstable = next_legal & is_remove & (~checked_stable)

        collapse_term = jnp.where(
            done_unstable,
            -collapse_penalty,
            0.0,
        )

        success_term = jnp.where(
            next_done_success & checked_stable & next_legal,
            success_bonus,
            0.0,
        )

        reward_raw = base_reward + collapse_term + success_term

        terminal = next_done_success | done_no_action | done_unstable
        discount = jnp.where(terminal, 0.0, 1.0)

        v_min = -0.02
        v_max = 1.2
        v_scale = v_max - v_min
        reward = (reward_raw + discount * v_min - v_min) / v_scale

        next_e_state = e_state._replace(
            board=next_board,
            part_states=next_part_states,
            stable=checked_stable,
        )

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=val,
        ), next_e_state

    n_actions = env_state.part_states.shape[1] * 2
    root_base_legal_mask = legal_action_mask_batch(
        env_state.part_states,
        env_state.insertion_table,
        n_actions,
        n_robot,
    )
    root_legal_mask = ik_action_mask_batch_jax(
        env_state,
        root_base_legal_mask,
        n_actions,
        n_robot,
    )

    (init_logits, init_value), _ = p_model.apply(
        policy_params,
        policy_states,
        rng_key,
        board=env_state.board,
        part_states=env_state.part_states,
        contact_matrix=env_state.contact_matrix,
        piece_masks=env_state.piece_masks,
        is_sampling=is_sampling,
    )
    init_logits = jnp.where(root_legal_mask, init_logits, -1e9)

    root = mctx.RootFnOutput(
        prior_logits=init_logits,
        value=init_value,
        embedding=env_state,
    )

    policy_output = mctx.gumbel_muzero_policy(
        params=policy_params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=192,
        max_depth=30,
        invalid_actions=~root_legal_mask,
        max_num_considered_actions=n_actions,
    )

    return policy_output.action, policy_output.action_weights


def train_alphazero_seq(
    env,
    p_model,
    result_save_folder,
    num_iterations=10,
    num_parallel_envs=4,
    step_per_iterations=100,
    batch_size=512,
    num_train_steps=5,
    batch_reset_fn=None,
    checkpoint_every=5,
    init_checkpoint_path=None,
    eval_files=None,
    eval_every=2,
    eval_mode="policy_only",
    seed=42,
    *,
    board_cache=None,
):
    """Train from board/IK episodes, with optional evaluation on explicit files."""
    Path(result_save_folder).mkdir(parents=True, exist_ok=True)
    data_dir = Path(result_save_folder) / "stability_data"
    checkpoint_dir = Path(result_save_folder) / "checkpoints"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    rng_key = jax.random.PRNGKey(seed)
    rng_key, subkey = jax.random.split(rng_key, 2)

    logger.debug("Random seed: %d", seed)

    if init_checkpoint_path is not None:
        policy_params, policy_states, extra = load_checkpoint(init_checkpoint_path)
        logger.debug("Loaded initial policy: %s", init_checkpoint_path)
        if extra is not None:
            logger.debug("Checkpoint metadata: %s", extra)
    else:
        policy_params, policy_states = p_model.init(
            subkey,
            board=env.state.board,
            part_states=env.state.part_states,
            contact_matrix=env.state.contact_matrix,
            piece_masks=env.state.piece_masks,
            is_sampling=False,
        )
        logger.debug("[Init] initialized sequence policy from scratch")

    logger.info(
        "Policy: %s | Stability checker: %s",
        "Loaded" if init_checkpoint_path is not None else "Initialized",
        "Loaded" if env.checker_params is not None else "Disabled",
    )
    logger.info("Output: %s", result_save_folder)

    optimizer = optax.adam(learning_rate=5e-5)
    opt_state = optimizer.init(policy_params)

    replay_buffer = deque(maxlen=200000)
    game_history = [[] for _ in range(num_parallel_envs)]
    loss_weights = (1.0, 1.0)

    best_result_by_shape = {}
    obs = env._get_obs()

    logger.debug("Starting sequence training: iterations=%d, environments=%d",
                num_iterations, num_parallel_envs)
    total_train_wall_t0 = time.perf_counter()

    for iteration in range(num_iterations):
        is_sampling = iteration < 3

        if batch_reset_fn is not None:
            reset_kwargs = batch_reset_fn(iteration)
            env.reset_batch(
                boards=reset_kwargs["boards"],
                board_paths=reset_kwargs.get("board_paths", None),
            )
            obs = env._get_obs()
            game_history = [[] for _ in range(num_parallel_envs)]
            shape_names_for_envs = reset_kwargs.get(
                "shape_names",
                ["single_shape"] * num_parallel_envs,
            )
        else:
            shape_names_for_envs = ["single_shape"] * num_parallel_envs

        logger.debug("Iteration %d/%d", iteration + 1, num_iterations)
        iter_t0 = time.perf_counter()

        num_success = 0
        iteration_newton_boards = []
        iteration_newton_hold_masks = []
        iteration_newton_labels = []
        checker_match_count = 0
        checker_total_count = 0
        checker_stable_match = 0
        checker_stable_total = 0
        checker_unstable_match = 0
        checker_unstable_total = 0

        for istep in range(step_per_iterations):
            rng_key, subkey = jax.random.split(rng_key)

            n_actions = env.state.part_states.shape[1] * 2
            legal_mask_now = env.ik_action_mask_batch(env.state)
            has_any_legal_now = jnp.any(legal_mask_now, axis=1)

            dummy_actions = jnp.zeros((num_parallel_envs,), dtype=jnp.int32)
            dummy_probs = jnp.zeros((num_parallel_envs, n_actions), dtype=jnp.float32)

            if bool(jnp.any(has_any_legal_now)):
                safe_state = jax.tree_util.tree_map(
                    lambda cur, reset: jnp.where(
                        has_any_legal_now.reshape((num_parallel_envs,) + (1,) * (cur.ndim - 1)),
                        cur,
                        reset,
                    ),
                    env.state,
                    env.reset_state,
                )

                mcts_actions, mcts_probs = mcts_action(
                    policy_params,
                    policy_states,
                    subkey,
                    safe_state,
                    env.checker_params,
                    env.n_robot,
                    is_sampling,
                    p_model=p_model,
                )

                actions = jnp.where(has_any_legal_now, mcts_actions, dummy_actions)
                probs = jnp.where(has_any_legal_now[:, None], mcts_probs, dummy_probs)
            else:
                actions = dummy_actions
                probs = dummy_probs
            (
                next_obs,
                reward,
                done,
                success,
                raw_next_state,
                newton_boards,
                newton_hold_masks,
                newton_labels,
                checker_probs,
                checker_labels,
            ) = env.step(actions, has_any_legal_now=has_any_legal_now)
            if newton_boards is not None and newton_labels is not None:
                iteration_newton_boards.append(newton_boards)
                iteration_newton_hold_masks.append(newton_hold_masks)
                iteration_newton_labels.append(newton_labels)

                if checker_labels is not None:
                    newton_np = np.asarray(newton_labels).astype(np.bool_).reshape(-1)
                    checker_np = np.asarray(checker_labels).astype(np.bool_).reshape(-1)

                    match = checker_np == newton_np

                    checker_match_count += int(np.sum(match))
                    checker_total_count += int(newton_np.shape[0])

                    stable_mask = newton_np
                    unstable_mask = ~newton_np

                    checker_stable_match += int(np.sum(match & stable_mask))
                    checker_stable_total += int(np.sum(stable_mask))

                    checker_unstable_match += int(np.sum(match & unstable_mask))
                    checker_unstable_total += int(np.sum(unstable_mask))

            for env_idx in range(num_parallel_envs):
                game_history[env_idx].append({
                    "board": np.array(obs.board[env_idx]),
                    "part_states": np.array(obs.part_states[env_idx]),
                    "piece_masks": np.array(obs.piece_masks[env_idx]),
                    "contact_matrix": np.array(obs.contact_matrix[env_idx]),
                    "insertion_table": np.array(obs.insertion_table[env_idx]),
                    "probs": np.array(probs[env_idx]),
                    "legal_mask": np.array(legal_mask_now[env_idx]),
                    "done": bool(done[env_idx]),
                    "reward": float(reward[env_idx]),
                })

            num_success += int(np.sum(np.array(success)))

            for env_idx in range(num_parallel_envs):
                shape_name = shape_names_for_envs[env_idx]
                key = shape_name
                if key not in best_result_by_shape:
                    best_result_by_shape[key] = {
                        "name": shape_name,
                        "progress": -1.0,
                        "part_states": None,
                    }

                if bool(done[env_idx]):
                    if bool(success[env_idx]):
                        ps = np.array(raw_next_state.part_states[env_idx])
                    else:
                        ps = np.array(obs.part_states[env_idx])

                    n_real = int(obs.n_parts[env_idx])
                    progress = float(np.sum(ps[:n_real] == 0) / max(1, n_real))

                    if progress > best_result_by_shape[key]["progress"]:
                        best_result_by_shape[key]["progress"] = progress
                        best_result_by_shape[key]["part_states"] = ps.copy()
                        logger.debug(
                            f"[BEST] {shape_name}: progress -> {progress:.3f} "
                            f"(iter {iteration + 1}, step {istep + 1})"
                        )

            obs = next_obs

        accum_rewards = []
        for env_idx in range(num_parallel_envs):
            if len(game_history[env_idx]) == 0:
                continue

            istep = len(game_history[env_idx]) - 1
            while istep >= 0 and not game_history[env_idx][istep]["done"]:
                istep -= 1

            if istep < 0:
                save_to_buffer = copy.deepcopy(game_history[env_idx])
                game_history[env_idx] = []
            else:
                save_to_buffer = copy.deepcopy(game_history[env_idx][:istep + 1])
                game_history[env_idx] = game_history[env_idx][istep + 1:]

            v = 0.0
            for data in reversed(save_to_buffer):
                r = data["reward"]
                done = data["done"]
                if done:
                    accum_rewards.append(r)
                elif len(accum_rewards) > 0:
                    accum_rewards[-1] += r

                v = r + v * (1.0 - float(done))

                replay_buffer.append((
                    data["board"],
                    data["part_states"],
                    data["piece_masks"],
                    data["contact_matrix"],
                    data["insertion_table"],
                    data["probs"],
                    data["legal_mask"],
                    v,
                ))
        if len(iteration_newton_boards) > 0:
            boards_all = np.concatenate(iteration_newton_boards, axis=0).astype(np.int32)
            hold_masks_all = np.concatenate(iteration_newton_hold_masks, axis=0).astype(np.float32)
            labels_all = np.concatenate(iteration_newton_labels, axis=0).astype(np.float32)

            seq_data_path = os.path.join(
                data_dir,
                f"sequence_stability_data_iter_{iteration + 1:04d}.npz",
            )
            np.savez_compressed(
                seq_data_path,
                boards=boards_all,
                hold_masks=hold_masks_all,
                labels=labels_all,
            )
            logger.debug(f"[SeqStabilityData] saved {boards_all.shape[0]} samples to {seq_data_path}")

        if len(accum_rewards) == 0:
            accum_rewards.append(0.0)

        iter_time = time.perf_counter() - iter_t0
        total_time = time.perf_counter() - total_train_wall_t0
        buffer_size = len(replay_buffer)
        buffer_batches = buffer_size // batch_size
        avg_return = float(np.mean(accum_rewards)) if len(accum_rewards) > 0 else 0.0
        if checker_total_count > 0:
            checker_match_ratio = checker_match_count / checker_total_count
            checker_stable_ratio = checker_stable_match / max(checker_stable_total, 1)
            checker_unstable_ratio = checker_unstable_match / max(checker_unstable_total, 1)

            checker_log = (
                f"checker_match={checker_match_ratio:.4f} "
                f"({checker_match_count}/{checker_total_count}) | "
                f"checker_stable_match={checker_stable_ratio:.4f} "
                f"({checker_stable_match}/{checker_stable_total}) | "
                f"checker_unstable_match={checker_unstable_ratio:.4f} "
                f"({checker_unstable_match}/{checker_unstable_total})"
            )
        else:
            checker_log = "checker_match=NA"
        logger.debug(
            f"[Time] iter={iter_time:.2f}s | total={total_time:.2f}s | "
            f"buffer={buffer_size} | buffer_batches={buffer_batches} | "
            f"success={num_success} | avg_return={avg_return:.4f} | "
            f"is_sampling={is_sampling} | {checker_log}"
        )

        phase = "Sampling" if is_sampling else "Collecting samples"
        loss_log = f"Policy loss: —, Value loss: —, Entropy: — ({phase})"
        if len(replay_buffer) >= batch_size and not is_sampling:
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy = 0.0
            train_steps = 0

            for _ in range(num_train_steps):
                batch = random.sample(replay_buffer, batch_size)
                b_board = jnp.array([x[0] for x in batch])
                b_ps = jnp.array([x[1] for x in batch])
                b_pm = jnp.array([x[2] for x in batch])
                b_cm = jnp.array([x[3] for x in batch])
                b_it = jnp.array([x[4] for x in batch])
                b_pr = jnp.array([x[5] for x in batch])
                b_lm = jnp.array([x[6] for x in batch])
                b_v = jnp.array([x[7] for x in batch])

                (policy_params, policy_states, opt_state,
                p_loss, v_loss, entropy) = train_step(
                    optimizer,
                    policy_params,
                    policy_states,
                    opt_state,
                    b_board,
                    b_ps,
                    b_pm,
                    b_cm,
                    b_it,
                    b_pr,
                    b_v,
                    b_lm,
                    env.n_robot,
                    loss_weights,
                    p_model=p_model,
                )

                avg_policy_loss += p_loss
                avg_value_loss += v_loss
                avg_entropy += entropy
                train_steps += 1

            avg_policy_loss /= train_steps
            avg_value_loss /= train_steps
            avg_entropy /= train_steps

            loss_log = (
                f"Policy loss: {avg_policy_loss:.4f}, "
                f"Value loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}"
            )
        if (
            eval_files is not None
            and eval_every is not None
            and eval_every > 0
            and ((iteration + 1) % eval_every) == 0
        ):
            evaluate_all_shapes(
                policy_params=policy_params,
                policy_states=policy_states,
                npy_files=eval_files,
                iteration=iteration + 1,
                n_robot=env.n_robot,
                max_shift=env.max_shift,
                mode=eval_mode,
                checker_params=env.checker_params,
                fixed_max_parts=env.fixed_max_parts,
                board_cache=board_cache,
                fixed_max_ik_samples=env.fixed_max_ik_samples,
                fixed_max_ik_r1=env.fixed_max_ik_r1,
                fixed_max_ik_r2=env.fixed_max_ik_r2,
                p_model=p_model,
            )

        logger.info("Iteration %d/%d | Buffer: %d", iteration + 1, num_iterations, len(replay_buffer))
        logger.info(
            "[Time] Iteration: %.2fs | Total: %.2fs",
            time.perf_counter() - iter_t0,
            time.perf_counter() - total_train_wall_t0,
        )
        logger.info(loss_log)

        if ((iteration + 1) % checkpoint_every) == 0:
            ckpt_path = checkpoint_dir / f"seq_policy_iter_{iteration + 1}.pkl"
            save_checkpoint(
                ckpt_path,
                policy_params,
                policy_states,
                extra={"iteration": iteration + 1},
            )
            logger.info("Saved policy checkpoint: %s", ckpt_path)

    return policy_params, policy_states
