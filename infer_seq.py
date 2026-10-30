"""Run batched sequence inference and save boards, trajectories, and metrics."""

import argparse
import json
import os
import pickle
import time
from functools import partial
from pathlib import Path

import haiku as hk
import jax
import jax.numpy as jnp
import mctx
import numpy as np

from tiling_rl.sequence.env_seq import (
    SequenceEnv,
    pure_step_jax,
    legal_action_mask_batch,
    piece_ids_from_board,
    normalize_piece_ids,
    ik_action_mask_batch_jax,
    ik_selected_action_legal_batch_jax,
    held_piece_ids_from_state,
)
from tiling_rl.sequence.model_seq import AttentionPolicyPatchSeq as AttentionPolicy
from tiling_rl.sequence.ik_lookup import resolve_ik_table_path
from tiling_rl.stability.stability_checker_jax_seq import checker_predict_batch_jax as checker_predict_seq_batch_jax
from tiling_rl.physics.simulator_newton_seq import check_stability_newton

try:
    from tiling_rl.common.render import render_equivalent_sets
except Exception:
    render_equivalent_sets = None


PROJECT_DIR = Path(__file__).resolve().parent

def p_forward_fn(board, part_states, contact_matrix, piece_masks, is_sampling):
    policy = AttentionPolicy()
    return policy(
        board=board,
        part_states=part_states,
        contact_matrix=contact_matrix,
        piece_masks=piece_masks,
        is_sampling=is_sampling,
    )


p_model = hk.transform_with_state(p_forward_fn)


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["policy_params"], payload["policy_states"], payload.get("extra", None)


def load_checker_params(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload.get("params", payload)


def held_hold_mask_batch_from_state(board, part_states):
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


def compute_global_padding(npy_files, use_ik=True, ik_table_suffix="_ik_table.npz"):
    max_parts = 1
    max_samples = 1
    max_r1 = 1
    max_r2 = 1

    for p in npy_files:
        board = np.load(p).astype(np.int32)
        max_parts = max(max_parts, count_real_parts(board))

        if use_ik:
            ik_path = resolve_ik_table_path(p, ik_table_suffix)
            with np.load(ik_path) as data:
                max_samples = max(max_samples, int(data["sample_piece_id"].shape[0]))
                max_r1 = max(max_r1, int(data["robot1_sample_ids"].shape[0]))
                max_r2 = max(max_r2, int(data["robot2_sample_ids"].shape[0]))

    return max_parts, max_samples, max_r1, max_r2


def collect_npy_files(npy_dir, npy_glob="*.npy", use_ik=True, ik_table_suffix="_ik_table.npz", limit=None):
    npy_dir = Path(npy_dir)
    all_files = sorted(
        p for p in npy_dir.glob(npy_glob)
        if p.is_file() and not p.name.endswith("_ik_table.npy")
    )

    if use_ik:
        valid = []
        missing = []
        for p in all_files:
            if resolve_ik_table_path(p, ik_table_suffix).is_file():
                valid.append(p)
            else:
                missing.append(p)
        if missing:
            print(f"Warning: Skipping {len(missing)} boards without IK tables", flush=True)
        all_files = valid

    if limit is not None and int(limit) > 0:
        all_files = all_files[: int(limit)]

    if len(all_files) == 0:
        raise FileNotFoundError(f"No valid npy assemblies found in {npy_dir}")

    return all_files


@partial(jax.jit, static_argnames=("n_robot", "use_ik"))
def policy_only_action(policy_params, policy_states, rng_key, env_state, n_robot=2, use_ik=True):
    n_actions = env_state.part_states.shape[1] * 2
    base_legal_mask = legal_action_mask_batch(
        env_state.part_states,
        env_state.insertion_table,
        n_actions,
        n_robot,
    )
    if use_ik:
        legal_mask = ik_action_mask_batch_jax(
            env_state,
            base_legal_mask,
            n_actions,
            n_robot,
        )
    else:
        legal_mask = base_legal_mask

    (logits, _), _ = p_model.apply(
        policy_params,
        policy_states,
        rng_key,
        board=env_state.board,
        part_states=env_state.part_states,
        contact_matrix=env_state.contact_matrix,
        piece_masks=env_state.piece_masks,
        is_sampling=False,
    )

    logits = jnp.where(legal_mask, logits, -1e9)
    actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    probs = jax.nn.softmax(logits, axis=-1)
    has_any_legal = jnp.any(legal_mask, axis=-1)
    return actions, probs, has_any_legal


@jax.jit
def value_inference_batch(policy_params, policy_states, env_state):
    (_, values), _ = p_model.apply(
        policy_params,
        policy_states,
        None,
        board=env_state.board,
        part_states=env_state.part_states,
        contact_matrix=env_state.contact_matrix,
        piece_masks=env_state.piece_masks,
        is_sampling=False,
    )
    return values

@partial(jax.jit, static_argnames=("n_robot", "use_ik", "is_sampling", "num_simulations", "max_depth"))
def mcts_action(
    policy_params,
    policy_states,
    rng_key,
    env_state,
    checker_params,
    n_robot=2,
    is_sampling=False,
    num_simulations=192,
    max_depth=30,
    use_ik=True,
):
    def recurrent_fn(p, key, actions, e_state):
        n_actions = e_state.part_states.shape[1] * 2
        base_legal_mask_now = legal_action_mask_batch(
            e_state.part_states,
            e_state.insertion_table,
            n_actions,
            n_robot,
        )

        if use_ik:
            action_ik_legal = ik_selected_action_legal_batch_jax(
                e_state,
                base_legal_mask_now,
                actions,
            )
        else:
            action_ik_legal = jnp.ones((actions.shape[0],), dtype=jnp.bool_)

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
        if use_ik:
            next_legal_mask = ik_action_mask_batch_jax(
                tmp_next_state,
                base_next_legal_mask,
                n_actions,
                n_robot,
            )
        else:
            next_legal_mask = base_next_legal_mask
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

        hold_mask = held_hold_mask_batch_from_state(next_board, next_part_states)
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
        collapse_term = jnp.where(done_unstable, -collapse_penalty, 0.0)
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
    if use_ik:
        root_legal_mask = ik_action_mask_batch_jax(
            env_state,
            root_base_legal_mask,
            n_actions,
            n_robot,
        )
    else:
        root_legal_mask = root_base_legal_mask

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
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=~root_legal_mask,
        max_num_considered_actions=n_actions,
    )

    return policy_output.action, policy_output.action_weights


def build_assembly_trial_batch(npy_files, num_trials, seed):
    boards = []
    board_paths = []
    shape_names = []
    shape_indices = []
    trial_indices = []
    trial_seeds = []

    for shape_idx, p in enumerate(npy_files):
        board = np.load(p).astype(np.int32)
        for trial in range(num_trials):
            boards.append(board.copy())
            board_paths.append(p)
            shape_names.append(p.stem)
            shape_indices.append(shape_idx)
            trial_indices.append(trial)
            trial_seeds.append(int(seed + shape_idx * 100000 + trial))

    return {
        "boards": boards,
        "board_paths": board_paths,
        "shape_names": shape_names,
        "shape_indices": np.asarray(shape_indices, dtype=np.int32),
        "trial_indices": np.asarray(trial_indices, dtype=np.int32),
        "trial_seeds": np.asarray(trial_seeds, dtype=np.int64),
    }


def build_single_trial_batch(npy_files, trial, seed):
    boards = []
    board_paths = []
    shape_names = []
    shape_indices = []
    trial_indices = []
    trial_seeds = []

    for shape_idx, p in enumerate(npy_files):
        board = np.load(p).astype(np.int32)
        boards.append(board.copy())
        board_paths.append(p)
        shape_names.append(p.stem)
        shape_indices.append(shape_idx)
        trial_indices.append(int(trial))
        trial_seeds.append(int(seed + shape_idx * 100000 + int(trial)))

    return {
        "boards": boards,
        "board_paths": board_paths,
        "shape_names": shape_names,
        "shape_indices": np.asarray(shape_indices, dtype=np.int32),
        "trial_indices": np.asarray(trial_indices, dtype=np.int32),
        "trial_seeds": np.asarray(trial_seeds, dtype=np.int64),
    }


def progress_from_part_states(part_states, n_parts):
    n_parts = np.asarray(n_parts).astype(np.int32)
    removed = []
    progress = []
    for i in range(part_states.shape[0]):
        n = int(n_parts[i])
        r = int(np.sum(part_states[i, :n] == 0))
        removed.append(r)
        progress.append(float(r / max(n, 1)))
    return np.asarray(removed, dtype=np.int32), np.asarray(progress, dtype=np.float32)


def save_board_outputs(board, out_dir, stem, save_png=True):
    out_dir = Path(out_dir)
    (out_dir / "boards").mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "boards" / f"{stem}.npy"
    np.save(npy_path, np.asarray(board, dtype=np.int32))

    png_path = None
    if save_png and render_equivalent_sets is not None:
        (out_dir / "previews").mkdir(parents=True, exist_ok=True)
        png_path = out_dir / "previews" / f"{stem}.png"
        try:
            render_equivalent_sets(np.asarray(board, dtype=np.int32), save_path=str(png_path))
        except Exception as e:
            print(f"Warning: Failed to render {png_path}: {e}", flush=True)
            png_path = None

    return str(npy_path), None if png_path is None else str(png_path)


def board_from_part_state(full_board, part_state):
    out = np.asarray(full_board, dtype=np.int32).copy()
    part_state = np.asarray(part_state, dtype=np.int32)
    for pid in range(1, part_state.shape[0] + 1):
        if part_state[pid - 1] == 0:
            out[out == pid] = -1
    return out


def run_batched_policy_only_deferred_newton(
    env,
    policy_params,
    policy_states,
    rng_key,
    shape_names,
    max_steps=200,
    n_robot=2,
    verbose=False,
):
    rollout_t0 = time.perf_counter()
    obs = env._get_obs()
    batch_size = int(obs.board.shape[0])

    active = np.ones((batch_size,), dtype=np.bool_)
    done_by_policy = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_no_action = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_unstable = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_max_steps = np.ones((batch_size,), dtype=np.bool_)
    steps_taken = np.zeros((batch_size,), dtype=np.int32)

    initial_boards = np.asarray(obs.board).astype(np.int32)
    n_parts = np.asarray(obs.n_parts).astype(np.int32)
    final_part_states = np.asarray(obs.part_states).astype(np.int32)

    remove_boards = []
    remove_held_piece_ids = []
    remove_pre_states = []
    remove_post_states = []
    remove_shape_ids = []
    remove_orders = []

    forward_time = 0.0

    for step in range(max_steps):
        if not active.any():
            break

        rng_key, subkey = jax.random.split(rng_key)

        safe_state = jax.tree_util.tree_map(
            lambda cur, reset: jnp.where(
                jnp.asarray(active).reshape((batch_size,) + (1,) * (cur.ndim - 1)),
                cur,
                reset,
            ),
            env.state,
            env.reset_state,
        )

        forward_t0 = time.perf_counter()
        actions, probs, has_any_legal = policy_only_action(
            policy_params=policy_params,
            policy_states=policy_states,
            rng_key=subkey,
            env_state=safe_state,
            n_robot=n_robot,
            use_ik=env.use_ik,
        )

        has_any_legal_np = np.asarray(has_any_legal).astype(np.bool_)
        apply_mask = active & has_any_legal_np

        no_legal = active & (~has_any_legal_np)
        if np.any(no_legal):
            terminated_by_no_action[no_legal] = True
            terminated_by_max_steps[no_legal] = False
            active[no_legal] = False
            steps_taken[no_legal] = step + 1

        if not apply_mask.any():
            forward_time += time.perf_counter() - forward_t0
            continue

        dummy_actions = jnp.zeros((batch_size,), dtype=jnp.int32)
        actions = jnp.where(jnp.asarray(apply_mask), actions, dummy_actions)

        next_board, next_part_states, done_success, legal, is_remove, _ = jax.vmap(
            pure_step_jax,
            in_axes=(0, 0, None),
        )(safe_state, actions, n_robot)
        next_board.block_until_ready()
        forward_time += time.perf_counter() - forward_t0

        legal_np = np.asarray(legal).astype(np.bool_)
        is_remove_np = np.asarray(is_remove).astype(np.bool_)
        done_success_np = np.asarray(done_success).astype(np.bool_)

        prev_part_states_np = np.asarray(env.state.part_states).astype(np.int32)
        next_part_states_np = np.asarray(next_part_states).astype(np.int32)
        next_board_np = np.asarray(next_board).astype(np.int32)

        step_valid = apply_mask & legal_np

        for idx in np.where(step_valid & is_remove_np)[0]:
            remove_boards.append(next_board_np[idx].copy())
            remove_held_piece_ids.append(held_piece_ids_from_state(next_part_states_np[idx]))
            remove_pre_states.append(prev_part_states_np[idx].copy())
            remove_post_states.append(next_part_states_np[idx].copy())
            remove_shape_ids.append(int(idx))
            remove_orders.append(int(np.sum(next_part_states_np[idx, :n_parts[idx]] == 0)))

        final_part_states[step_valid] = next_part_states_np[step_valid]
        done_by_policy = done_by_policy | (step_valid & done_success_np)

        step_valid_jnp = jnp.asarray(step_valid)
        env.state = env.state._replace(
            board=jnp.where(step_valid_jnp[:, None, None], next_board, env.state.board),
            part_states=jnp.where(step_valid_jnp[:, None], next_part_states, env.state.part_states),
            stable=jnp.ones((batch_size,), dtype=jnp.bool_),
        )

        for idx in np.where(step_valid)[0]:
            steps_taken[idx] = step + 1
            removed = int(np.sum(final_part_states[idx, :n_parts[idx]] == 0))
            progress = removed / max(int(n_parts[idx]), 1)
            if done_success_np[idx]:
                terminated_by_max_steps[idx] = False

        active = active & step_valid & (~done_success_np)

    rollout_time = time.perf_counter() - rollout_t0

    sim_time = 0.0
    valid_part_states = final_part_states.copy()
    collapse_next_boards = [None for _ in range(batch_size)]
    collapse_next_part_states = [None for _ in range(batch_size)]

    total_candidate_remove_states = int(len(remove_boards))
    total_newton_states = 0
    total_stable_remove = 0
    total_unstable_remove = 0
    skipped_newton_states = 0

    first_bad_by_shape = {}
    alive_for_newton = np.ones((batch_size,), dtype=np.bool_)

    if len(remove_boards) > 0:
        sim_t0 = time.perf_counter()
        remove_orders_np = np.asarray(remove_orders, dtype=np.int32)
        remove_shape_ids_np = np.asarray(remove_shape_ids, dtype=np.int32)

        for order in sorted(np.unique(remove_orders_np).tolist()):
            group = np.where(remove_orders_np == int(order))[0]
            if group.shape[0] == 0:
                continue

            keep = np.asarray([alive_for_newton[int(remove_shape_ids_np[k])] for k in group], dtype=np.bool_)
            run_ids = group[keep]
            skipped_newton_states += int(group.shape[0] - run_ids.shape[0])

            if run_ids.shape[0] == 0:
                continue

            boards_np = np.stack([remove_boards[int(k)] for k in run_ids], axis=0).astype(np.int32)
            held_ids = [remove_held_piece_ids[int(k)] for k in run_ids]

            labels = np.asarray(
                check_stability_newton(
                    boards_np,
                    held_piece_ids=held_ids,
                    render=False,
                    keep_run=False,
                )
            ).astype(np.bool_).reshape(-1)

            total_newton_states += int(labels.shape[0])
            total_stable_remove += int(np.sum(labels))
            total_unstable_remove += int(np.sum(~labels))

            bad_local = np.where(~labels)[0]
            for local_idx in bad_local:
                k = int(run_ids[int(local_idx)])
                sid = int(remove_shape_ids[k])
                if sid not in first_bad_by_shape:
                    first_bad_by_shape[sid] = k
                    alive_for_newton[sid] = False

            if not np.any(alive_for_newton):
                remaining = int(np.sum(remove_orders_np > int(order)))
                skipped_newton_states += remaining
                break

        sim_time = time.perf_counter() - sim_t0

        for sid, k in first_bad_by_shape.items():
            valid_part_states[sid] = remove_pre_states[k]
            collapse_next_boards[sid] = remove_boards[k]
            collapse_next_part_states[sid] = remove_post_states[k]
            terminated_by_unstable[sid] = True
            terminated_by_max_steps[sid] = False
            done_by_policy[sid] = False

    best_removed, best_progress = progress_from_part_states(valid_part_states, n_parts)
    best_boards = np.stack(
        [board_from_part_state(initial_boards[i], valid_part_states[i]) for i in range(batch_size)],
        axis=0,
    ).astype(np.int32)

    final_success = np.zeros((batch_size,), dtype=np.bool_)
    for i in range(batch_size):
        n = int(n_parts[i])
        final_success[i] = bool(
            done_by_policy[i]
            and (not terminated_by_unstable[i])
            and np.all(valid_part_states[i, :n] == 0)
        )
        if final_success[i]:
            terminated_by_max_steps[i] = False


    return [
        {
            "best_board": best_boards[i].astype(np.int32),
            "best_part_states": valid_part_states[i].astype(np.int32),
            "best_removed": int(best_removed[i]),
            "n_parts": int(n_parts[i]),
            "progress": float(best_progress[i]),
            "collapse_next_board": collapse_next_boards[i],
            "collapse_next_part_states": collapse_next_part_states[i],
            "final_success": bool(final_success[i]),
            "terminated_by_no_action": bool(terminated_by_no_action[i]),
            "terminated_by_unstable": bool(terminated_by_unstable[i]),
            "terminated_by_max_steps": bool(terminated_by_max_steps[i]),
            "steps": int(steps_taken[i]),
            "total_newton_states": int(total_newton_states),
            "total_stable_remove": int(total_stable_remove),
            "total_unstable_remove": int(total_unstable_remove),
            "batch_total_time": float(rollout_time + sim_time),
            "batch_forward_time": float(forward_time),
            "batch_sim_time": float(sim_time),
        }
        for i in range(batch_size)
    ]

def run_batched_episodes(
    env,
    policy_params,
    policy_states,
    rng_key,
    shape_names,
    mode="mcts",
    num_simulations=192,
    max_depth=30,
    max_steps=200,
    n_robot=2,
    verbose=False,
):
    if mode in ("policy_only", "greedy"):
        return run_batched_policy_only_deferred_newton(
            env=env,
            policy_params=policy_params,
            policy_states=policy_states,
            rng_key=rng_key,
            shape_names=shape_names,
            max_steps=max_steps,
            n_robot=n_robot,
            verbose=verbose,
        )

    obs = env._get_obs()
    batch_size = int(obs.board.shape[0])

    active = np.ones((batch_size,), dtype=np.bool_)
    final_success = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_no_action = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_unstable = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_max_steps = np.ones((batch_size,), dtype=np.bool_)
    steps_taken = np.zeros((batch_size,), dtype=np.int32)

    initial_boards = np.asarray(obs.board).astype(np.int32)
    n_parts = np.asarray(obs.n_parts).astype(np.int32)

    best_boards = initial_boards.copy()
    best_part_states = np.asarray(obs.part_states).astype(np.int32)
    best_removed, best_progress = progress_from_part_states(best_part_states, n_parts)

    collapse_next_boards = [None for _ in range(batch_size)]
    collapse_next_part_states = [None for _ in range(batch_size)]

    total_newton_states = 0
    total_stable_remove = 0
    total_unstable_remove = 0

    for step in range(max_steps):
        if not active.any():
            break

        legal_mask_now = env.ik_action_mask_batch(env.state)
        has_any_legal = np.asarray(jnp.any(legal_mask_now, axis=1)).astype(np.bool_)
        apply_mask = active & has_any_legal

        no_legal = active & (~has_any_legal)
        if np.any(no_legal):
            terminated_by_no_action[no_legal] = True
            terminated_by_max_steps[no_legal] = False
            active[no_legal] = False
            steps_taken[no_legal] = step + 1

        if not apply_mask.any():
            continue

        rng_key, subkey = jax.random.split(rng_key)

        safe_state = jax.tree_util.tree_map(
            lambda cur, reset: jnp.where(
                jnp.asarray(apply_mask).reshape((batch_size,) + (1,) * (cur.ndim - 1)),
                cur,
                reset,
            ),
            env.state,
            env.reset_state,
        )

        if mode in ("policy_only", "greedy"):
            actions, probs, _ = policy_only_action(
                policy_params=policy_params,
                policy_states=policy_states,
                rng_key=subkey,
                env_state=safe_state,
                n_robot=n_robot,
                use_ik=env.use_ik,
            )
        elif mode == "mcts":
            actions, probs = mcts_action(
                policy_params=policy_params,
                policy_states=policy_states,
                rng_key=subkey,
                env_state=safe_state,
                checker_params=env.checker_params,
                n_robot=n_robot,
                is_sampling=False,
                num_simulations=num_simulations,
                max_depth=max_depth,
                use_ik=env.use_ik,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dummy_actions = jnp.zeros((batch_size,), dtype=jnp.int32)
        actions = jnp.where(jnp.asarray(apply_mask), actions, dummy_actions)

        prev_state = env.state
        step_state = prev_state._replace(
            stable=jnp.asarray(apply_mask) & prev_state.stable,
        )
        env.state = step_state

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
        ) = env.step(actions, has_any_legal_now=jnp.asarray(has_any_legal))

        done_np = np.asarray(done).astype(np.bool_)
        success_np = np.asarray(success).astype(np.bool_)
        raw_stable_np = np.asarray(raw_next_state.stable).astype(np.bool_)
        raw_board_np = np.asarray(raw_next_state.board).astype(np.int32)
        raw_part_states_np = np.asarray(raw_next_state.part_states).astype(np.int32)
        current_part_states_np = np.asarray(next_obs.part_states).astype(np.int32)

        if newton_labels is not None:
            labels = np.asarray(newton_labels).astype(np.bool_).reshape(-1)
            total_newton_states += int(labels.shape[0])
            total_stable_remove += int(np.sum(labels))
            total_unstable_remove += int(np.sum(~labels))

        keep_next = jnp.asarray(apply_mask)
        env.state = jax.tree_util.tree_map(
            lambda old, new: jnp.where(
                keep_next.reshape((batch_size,) + (1,) * (old.ndim - 1)),
                new,
                old,
            ),
            prev_state,
            env.state,
        )

        raw_removed, raw_progress = progress_from_part_states(raw_part_states_np, n_parts)
        current_removed, current_progress = progress_from_part_states(current_part_states_np, n_parts)

        for idx in np.where(apply_mask)[0]:
            steps_taken[idx] = step + 1

            if success_np[idx]:
                final_success[idx] = True
                terminated_by_max_steps[idx] = False
                if raw_progress[idx] >= best_progress[idx]:
                    best_progress[idx] = raw_progress[idx]
                    best_removed[idx] = raw_removed[idx]
                    best_boards[idx] = raw_board_np[idx].copy()
                    best_part_states[idx] = raw_part_states_np[idx].copy()
            elif not done_np[idx]:
                if raw_stable_np[idx]:
                    if current_progress[idx] >= best_progress[idx]:
                        best_progress[idx] = current_progress[idx]
                        best_removed[idx] = current_removed[idx]
                        best_boards[idx] = np.asarray(next_obs.board)[idx].astype(np.int32).copy()
                        best_part_states[idx] = current_part_states_np[idx].copy()
            else:
                terminated_by_max_steps[idx] = False
                if not raw_stable_np[idx]:
                    terminated_by_unstable[idx] = True
                    collapse_next_boards[idx] = raw_board_np[idx].copy()
                    collapse_next_part_states[idx] = raw_part_states_np[idx].copy()
                else:
                    terminated_by_no_action[idx] = True


        active[np.where(apply_mask)[0]] &= ~done_np[np.where(apply_mask)[0]]

    return [
        {
            "best_board": best_boards[i].astype(np.int32),
            "best_part_states": best_part_states[i].astype(np.int32),
            "best_removed": int(best_removed[i]),
            "n_parts": int(n_parts[i]),
            "progress": float(best_progress[i]),
            "collapse_next_board": collapse_next_boards[i],
            "collapse_next_part_states": collapse_next_part_states[i],
            "final_success": bool(final_success[i]),
            "terminated_by_no_action": bool(terminated_by_no_action[i]),
            "terminated_by_unstable": bool(terminated_by_unstable[i]),
            "terminated_by_max_steps": bool(terminated_by_max_steps[i]),
            "steps": int(steps_taken[i]),
            "total_newton_states": int(total_newton_states),
            "total_stable_remove": int(total_stable_remove),
            "total_unstable_remove": int(total_unstable_remove),
            "batch_total_time": 0.0,
            "batch_forward_time": 0.0,
            "batch_sim_time": 0.0,
        }
        for i in range(batch_size)
    ]


def infer_seq_folder(
    npy_dir=str(PROJECT_DIR / "dataset/sequence/test"),
    out_root=str(PROJECT_DIR / "results/inference_seq"),
    ckpt_path=str(PROJECT_DIR / "checkpoints/policy_seq.pkl"),
    checker_ckpt_path=str(PROJECT_DIR / "checkpoints/stability_checker_jax_seq.pkl"),
    npy_glob="*.npy",
    mode="mcts",
    num_trials=1,
    num_simulations=192,
    max_depth=30,
    max_steps=200,
    seed=0,
    n_robot=2,
    max_shift=10,
    use_ik=True,
    ik_table_suffix="_ik_table.npz",
    limit=None,
    save_png=True,
    verbose=False,
):
    if num_trials < 1:
        raise ValueError("num_trials must be positive")
    npy_files = collect_npy_files(
        npy_dir=npy_dir,
        npy_glob=npy_glob,
        use_ik=use_ik,
        ik_table_suffix=ik_table_suffix,
        limit=limit,
    )

    board_shapes = {np.load(path, mmap_mode="r", allow_pickle=False).shape for path in npy_files}
    if len(board_shapes) != 1 or any(len(shape) != 2 for shape in board_shapes):
        raise ValueError("Sequence inference requires 2D boards of the same shape")

    policy_params, policy_states, extra = load_checkpoint(str(ckpt_path))

    checker_params = None
    if mode == "mcts":
        checker_params = load_checker_params(str(checker_ckpt_path))

    fixed_max_parts, fixed_max_ik_samples, fixed_max_ik_r1, fixed_max_ik_r2 = compute_global_padding(
        npy_files,
        use_ik=use_ik,
        ik_table_suffix=ik_table_suffix,
    )

    npy_dir_name = Path(npy_dir).name
    ckpt_stem = Path(ckpt_path).stem
    save_root = Path(out_root) / npy_dir_name / mode / ckpt_stem
    save_root.mkdir(parents=True, exist_ok=True)

    config = {
        "npy_dir": str(npy_dir),
        "npy_glob": str(npy_glob),
        "out_root": str(out_root),
        "save_root": str(save_root),
        "ckpt_path": str(ckpt_path),
        "checker_ckpt_path": str(checker_ckpt_path),
        "mode": str(mode),
        "num_trials": int(num_trials),
        "num_simulations": int(num_simulations),
        "max_depth": int(max_depth),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "n_robot": int(n_robot),
        "max_shift": int(max_shift),
        "use_ik": bool(use_ik),
        "ik_table_suffix": str(ik_table_suffix),
        "fixed_max_parts": int(fixed_max_parts),
        "fixed_max_ik_samples": int(fixed_max_ik_samples),
        "fixed_max_ik_r1": int(fixed_max_ik_r1),
        "fixed_max_ik_r2": int(fixed_max_ik_r2),
        "batched": True,
        "trial_separate": True,
        "batch_size_per_trial": int(len(npy_files)),
    }

    print(f"Inference: Sequence | Mode: {'MCTS' if mode == 'mcts' else 'Policy only'}", flush=True)
    print(
        f"Data: {len(npy_files)} boards | Trials per board: {num_trials} | Batch size: {len(npy_files)}",
        flush=True,
    )
    print(
        f"Policy: Loaded | Stability checker: {'Loaded' if checker_params is not None else 'Disabled'} | "
        f"IK tables: {'Loaded' if use_ik else 'Disabled'}",
        flush=True,
    )
    print(f"Output: {save_root}", flush=True)

    t0 = time.perf_counter()
    all_results = []
    all_batch_infos = []
    all_init_values = []
    all_init_values_unscaled = []
    trial_time_logs = []

    for trial in range(int(num_trials)):
        trial_t0 = time.perf_counter()
        batch_info = build_single_trial_batch(
            npy_files=npy_files,
            trial=trial,
            seed=seed,
        )

        env = SequenceEnv(
            n_robot=n_robot,
            max_shift=max_shift,
            checker_ckpt_path=str(checker_ckpt_path) if (mode == "mcts" and checker_ckpt_path is not None) else None,
            fixed_max_parts=fixed_max_parts,
            use_ik=use_ik,
            ik_table_suffix=ik_table_suffix,
            fixed_max_ik_samples=fixed_max_ik_samples,
            fixed_max_ik_r1=fixed_max_ik_r1,
            fixed_max_ik_r2=fixed_max_ik_r2,
        )
        if mode == "mcts":
            env.checker_params = checker_params
            env.checker = True

        env.reset_batch(
            boards=batch_info["boards"],
            board_paths=batch_info["board_paths"] if use_ik else None,
        )

        value_t0 = time.perf_counter()
        init_values = np.asarray(
            value_inference_batch(
                policy_params,
                policy_states,
                env.state,
            )
        ).reshape(-1).astype(np.float32)
        value_time = time.perf_counter() - value_t0

        v_min = -0.02
        v_max = 1.2
        init_values_unscaled = init_values * (v_max - v_min) + v_min


        rng_key = jax.random.PRNGKey(int(seed) + int(trial) * 1000003)
        results = run_batched_episodes(
            env=env,
            policy_params=policy_params,
            policy_states=policy_states,
            rng_key=rng_key,
            shape_names=batch_info["shape_names"],
            mode=mode,
            num_simulations=num_simulations,
            max_depth=max_depth,
            max_steps=max_steps,
            n_robot=n_robot,
            verbose=verbose,
        )

        trial_total_time = time.perf_counter() - trial_t0
        if len(results) > 0:
            forward_time = float(results[0].get("batch_forward_time", 0.0))
            sim_time = float(results[0].get("batch_sim_time", 0.0))
        else:
            forward_time = 0.0
            sim_time = 0.0

        trial_success = sum(int(r["final_success"]) for r in results)
        trial_progress = float(np.mean([float(r["progress"]) for r in results])) if results else 0.0
        trial_time_logs.append({
            "trial": int(trial),
            "total_time": float(trial_total_time),
            "forward_time": float(forward_time),
            "sim_time": float(sim_time),
            "success_count": int(trial_success),
            "success_rate": float(trial_success / max(len(results), 1)),
            "mean_progress": float(trial_progress),
        })
        print(
            f"Batch {trial + 1}/{num_trials} | Completed: {(trial + 1) * len(npy_files)}/{num_trials * len(npy_files)} | "
            f"Time: {trial_total_time:.2f}s",
            flush=True,
        )

        all_results.extend(results)
        all_batch_infos.extend([batch_info] * len(results))
        all_init_values.extend(init_values.tolist())
        all_init_values_unscaled.extend(init_values_unscaled.tolist())

    shape_logs_by_idx = {}
    total_success = 0
    total_trials = 0

    for flat_idx, result in enumerate(all_results):
        batch_info = all_batch_infos[flat_idx]
        local_idx = flat_idx % len(batch_info["shape_names"])
        shape_idx = int(batch_info["shape_indices"][local_idx])
        trial = int(batch_info["trial_indices"][local_idx])
        trial_seed = int(batch_info["trial_seeds"][local_idx])
        shape_name = batch_info["shape_names"][local_idx]
        init_value = float(all_init_values[flat_idx])
        init_value_unscaled = float(all_init_values_unscaled[flat_idx])

        trial_stem = f"{shape_name}_trial_{trial:03d}"

        max_npy, max_png = save_board_outputs(
            result["best_board"],
            save_root,
            trial_stem,
            save_png=save_png,
        )
        (save_root / "piece_states").mkdir(parents=True, exist_ok=True)
        part_state_path = save_root / "piece_states" / f"{trial_stem}.npy"
        np.save(part_state_path, result["best_part_states"].astype(np.int32))

        collapse_npy = None
        collapse_png = None
        collapse_part_state_path = None
        if result["collapse_next_board"] is not None:
            collapse_npy, collapse_png = save_board_outputs(
                result["collapse_next_board"],
                save_root / "failures",
                trial_stem,
                save_png=save_png,
            )
            (save_root / "failures" / "piece_states").mkdir(parents=True, exist_ok=True)
            collapse_part_state_path = save_root / "failures" / "piece_states" / f"{trial_stem}.npy"
            np.save(collapse_part_state_path, result["collapse_next_part_states"].astype(np.int32))

        success_bool = bool(result["final_success"])
        total_success += int(success_bool)
        total_trials += 1

        row = {
            "trial": int(trial),
            "seed": trial_seed,
            "success": success_bool,
            "best_removed": int(result["best_removed"]),
            "n_parts": int(result["n_parts"]),
            "progress": float(result["progress"]),
            "steps": int(result["steps"]),
            "terminated_by_no_action": bool(result["terminated_by_no_action"]),
            "terminated_by_unstable": bool(result["terminated_by_unstable"]),
            "terminated_by_max_steps": bool(result["terminated_by_max_steps"]),
            "has_collapse_next": result["collapse_next_board"] is not None,
            "maxprogress_npy": max_npy,
            "maxprogress_png": max_png,
            "maxprogress_part_states_npy": str(part_state_path),
            "collapse_next_npy": collapse_npy,
            "collapse_next_png": collapse_png,
            "collapse_next_part_states_npy": None if collapse_part_state_path is None else str(collapse_part_state_path),
            "init_value": init_value,
            "init_value_unscaled": init_value_unscaled,        
        }

        if shape_idx not in shape_logs_by_idx:
            shape_logs_by_idx[shape_idx] = {
                "shape_name": shape_name,
                "source_npy": str(npy_files[shape_idx]),
                "num_trials": int(num_trials),
                "success_count": 0,
                "success_rate": 0.0,
                "best_removed": -1,
                "best_progress": -1.0,
                "best_trial": -1,
                "best_over_trials_npy": None,
                "best_over_trials_png": None,
                "best_over_trials_part_states_npy": None,
                "trials": [],
                "_best_board": None,
                "_best_part_states": None,
                "value_mean": 0.0,
                "value_unscaled_mean": 0.0,
            }

        shape_log = shape_logs_by_idx[shape_idx]
        shape_log["trials"].append(row)
        shape_log["success_count"] += int(success_bool)
        if float(result["progress"]) > float(shape_log["best_progress"]):
            shape_log["best_progress"] = float(result["progress"])
            shape_log["best_removed"] = int(result["best_removed"])
            shape_log["best_trial"] = int(trial)
            shape_log["_best_board"] = result["best_board"].copy()
            shape_log["_best_part_states"] = result["best_part_states"].copy()


    shape_logs = []
    for shape_idx in sorted(shape_logs_by_idx):
        shape_summary = shape_logs_by_idx[shape_idx]
        shape_name = shape_summary["shape_name"]
        shape_out_dir = save_root / "logs"
        shape_out_dir.mkdir(parents=True, exist_ok=True)

        best_board = shape_summary.pop("_best_board")
        best_part_states = shape_summary.pop("_best_part_states")
        if best_board is not None:
            best_npy, best_png = save_board_outputs(
                best_board,
                save_root / "best",
                shape_name,
                save_png=save_png,
            )
            shape_summary["best_over_trials_npy"] = best_npy
            shape_summary["best_over_trials_png"] = best_png
            (save_root / "best" / "piece_states").mkdir(parents=True, exist_ok=True)
            best_ps_path = save_root / "best" / "piece_states" / f"{shape_name}.npy"
            np.save(best_ps_path, best_part_states.astype(np.int32))
            shape_summary["best_over_trials_part_states_npy"] = str(best_ps_path)

        shape_summary["trials"] = sorted(shape_summary["trials"], key=lambda x: x["trial"])
        shape_summary["success_rate"] = float(
            shape_summary["success_count"] / max(shape_summary["num_trials"], 1)
        )

        if len(shape_summary["trials"]) > 0:
            shape_summary["value_mean"] = float(
                np.mean([t["init_value"] for t in shape_summary["trials"]])
            )
            shape_summary["value_unscaled_mean"] = float(
                np.mean([t["init_value_unscaled"] for t in shape_summary["trials"]])
            )

        shape_json_path = shape_out_dir / f"{shape_name}.json"
        with open(shape_json_path, "w", encoding="utf-8") as f:
            json.dump(shape_summary, f, indent=2, ensure_ascii=False)

        shape_logs.append(shape_summary)

    total_time = time.perf_counter() - t0
    first_trials = [s["trials"][0] for s in shape_logs if len(s["trials"]) > 0]
    first_progress = float(np.mean([t["progress"] for t in first_trials])) if first_trials else 0.0
    best_progress = float(np.mean([s["best_progress"] for s in shape_logs])) if shape_logs else 0.0

    first_value = float(np.mean([t["init_value"] for t in first_trials])) if first_trials else 0.0
    first_value_unscaled = float(np.mean([t["init_value_unscaled"] for t in first_trials])) if first_trials else 0.0
    value_success = [t["init_value"] for s in shape_logs for t in s["trials"] if t["success"]]
    value_fail = [t["init_value"] for s in shape_logs for t in s["trials"] if not t["success"]]
    value_success_mean = float(np.mean(value_success)) if len(value_success) > 0 else 0.0
    value_fail_mean = float(np.mean(value_fail)) if len(value_fail) > 0 else 0.0
    overall = {
        "config": config,
        "total_shapes": int(len(npy_files)),
        "total_trials": int(total_trials),
        "success_count": int(total_success),
        "success_rate": float(total_success / max(total_trials, 1)),
        "first_trial_mean_progress": first_progress,
        "first_trial_mean_value": first_value,
        "first_trial_mean_value_unscaled": first_value_unscaled,
        "value_success_mean": value_success_mean,
        "value_fail_mean": value_fail_mean,
        "best_over_trials_mean_progress": best_progress,
        "runtime_sec": float(total_time),
        "trial_time_logs": trial_time_logs,
        "shapes": shape_logs,
    }

    summary_json_path = save_root / "infer_seq_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    print(
        f"Finished | Successful trials: {total_success}/{total_trials} ({overall['success_rate']:.1%}) | "
        f"Total time: {total_time:.2f}s",
        flush=True,
    )

    return overall


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiling_folder", "--board_dir", type=str, default=str(PROJECT_DIR / "dataset/sequence/test"))
    parser.add_argument("--out_root", type=str, default=str(PROJECT_DIR / "results/inference_seq"))
    parser.add_argument("--ckpt_path", type=str, default=str(PROJECT_DIR / "checkpoints/policy_seq.pkl"))
    parser.add_argument("--checker_ckpt_path", type=str, default=str(PROJECT_DIR / "checkpoints/stability_checker_jax_seq.pkl"))
    parser.add_argument("--mode", type=str, default="policy_only", choices=["mcts", "policy_only"])
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    # Accept the legacy flag; inference always uses concise logs.
    parser.add_argument("--hide_step_log", action="store_true", default=False, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.num_trials < 1:
        parser.error("--num_trials must be positive")
    if not Path(args.tiling_folder).is_dir():
        parser.error(f"Board directory does not exist: {args.tiling_folder}. Provide --board_dir with NPY boards and matching IK tables.")
    names = ("ckpt_path", "checker_ckpt_path") if args.mode == "mcts" else ("ckpt_path",)
    for name in names:
        if not Path(getattr(args, name)).is_file():
            parser.error(f"--{name} file does not exist: {getattr(args, name)}")
    return args


def main(argv=None):
    args = parse_args(argv)

    infer_seq_folder(
        npy_dir=args.tiling_folder,
        out_root=args.out_root,
        ckpt_path=args.ckpt_path,
        checker_ckpt_path=args.checker_ckpt_path,
        mode=args.mode,
        num_trials=args.num_trials,
        seed=args.seed,
        verbose=not args.hide_step_log,
    )


if __name__ == "__main__":
    main()
