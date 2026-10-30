"""Run batched tiling inference and save boards, trajectories, and metrics."""

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

from tiling_rl.tiling.env_til import (
    TetrixEnv,
    pure_step_jax,
    legal_action_mask_batch,
    seq_value_from_tiling_board,
)
from tiling_rl.tiling.model_til import AttentionPolicyPatch as AttentionPolicy
from tiling_rl.stability.stability_checker_jax_til import checker_predict_batch_no_hold_jax
from tiling_rl.common.pieces import generate_polyominoes_from_list
from tiling_rl.common.render import render_equivalent_sets
from tiling_rl.common.masks import load_masks_from_png_dir


PROJECT_DIR = Path(__file__).resolve().parent

def p_forward_fn(board, inventory, pos, pieces, is_sampling):
    policy = AttentionPolicy()
    return policy(board, inventory, pos, pieces, is_sampling)


p_model = hk.transform_with_state(p_forward_fn)


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["policy_params"], payload["policy_states"], payload.get("extra", None)

def load_seq_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["policy_params"], payload["policy_states"], payload.get("extra", None)

def canonicalize_mask(mask, board_size):
    mask = np.asarray(mask, dtype=np.int32)
    valid = mask == 0
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Mask has no valid region.")

    cropped = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    H, W = board_size
    h, w = cropped.shape
    if h > H or w > W:
        raise ValueError(f"Cropped mask shape {(h, w)} exceeds board size {(H, W)}")

    canon = np.ones((H, W), dtype=np.int32)
    canon[H - h:, W - w:] = cropped
    return jnp.array(canon)


def build_single_shape_batch(spec, board_size, pieces, pieces_id, inventory):
    blocked_mask = canonicalize_mask(spec["mask"], board_size)[None, ...]
    return {
        "pieces": pieces,
        "pieces_id": pieces_id,
        "inventory": inventory,
        "blocked_mask": blocked_mask,
        "shape_names": [spec["name"]],
    }

def canonicalize_poly_shape(shape_grid):
    shape_grid = np.asarray(shape_grid, dtype=np.int32)

    def trim(x):
        rmask = np.any(x > 0, axis=1)
        cmask = np.any(x > 0, axis=0)
        return x[np.ix_(rmask, cmask)]

    symmetries = []
    curr = shape_grid.copy()
    for _ in range(4):
        curr = np.rot90(curr)
        symmetries.append(trim(curr))
        symmetries.append(trim(np.flipud(curr)))

    forms = [tuple(map(tuple, s.tolist())) for s in symmetries]
    return min(forms)


POLYOMINO_SHAPES = {
    "3I": [[1, 1, 1]],
    "3L": [[1, 1],
           [1, 0]],

    "4I": [[1, 1, 1, 1]],
    "4O": [[1, 1],
           [1, 1]],
    "4T": [[1, 1, 1],
           [0, 1, 0]],
    "4L": [[1, 1, 1],
           [1, 0, 0]],
    "4S": [[0, 1, 1],
           [1, 1, 0]],

    "5I": [[1, 1, 1, 1, 1]],
    "5L": [[1, 1, 1, 1],
           [1, 0, 0, 0]],
    "5P": [[1, 1],
           [1, 1],
           [1, 0]],
    "5T": [[1, 1, 1],
           [0, 1, 0],
           [0, 1, 0]],
    "5U": [[1, 0, 1],
           [1, 1, 1]],
    "5V": [[1, 0, 0],
           [1, 0, 0],
           [1, 1, 1]],
    "5W": [[1, 0, 0],
           [1, 1, 0],
           [0, 1, 1]],
    "5X": [[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]],
    "5Y": [[1, 1, 1, 1],
           [0, 1, 0, 0]],
    "5Z": [[1, 1, 0],
           [0, 1, 0],
           [0, 1, 1]],
    "5F": [[0, 1, 1],
           [1, 1, 0],
           [0, 1, 0]],
    "5N": [[1, 1, 0, 0],
           [0, 1, 1, 1]],
}


POLYOMINO_CANONICAL = {
    name: canonicalize_poly_shape(np.asarray(shape, dtype=np.int32))
    for name, shape in POLYOMINO_SHAPES.items()
}


def canonicalize_piece_from_grid(piece_grid):
    binary = (np.asarray(piece_grid) > 0).astype(np.int32)
    if np.sum(binary) == 0:
        return None

    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]
    binary = binary[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    return canonicalize_poly_shape(binary)


def make_inventory(mode, pieces, inventory_pieces=None, inventory_count=99):
    pieces_np = np.asarray(pieces)
    piece_sizes = np.sum(pieces_np > 0, axis=(1, 2)).astype(np.int32)

    inv = np.zeros((pieces_np.shape[0],), dtype=np.int32)

    if mode == "default":
        inv[:] = 10

    elif mode == "tri_only":
        inv[piece_sizes == 3] = inventory_count

    elif mode == "tetra_only":
        inv[piece_sizes == 4] = inventory_count

    elif mode == "penta_only":
        inv[piece_sizes == 5] = inventory_count

    elif mode == "custom":
        if inventory_pieces is None or inventory_pieces.strip() == "":
            raise ValueError("mode='custom' requires --inventory_pieces, e.g. 3I,4I,4O,5P")

        selected_names = [
            x.strip().upper()
            for x in inventory_pieces.split(",")
            if x.strip()
        ]

        unknown = [x for x in selected_names if x not in POLYOMINO_CANONICAL]
        if len(unknown) > 0:
            raise ValueError(
                f"Unknown inventory pieces: {unknown}. "
                f"Available: {sorted(POLYOMINO_CANONICAL.keys())}"
            )

        selected_canonicals = {
            POLYOMINO_CANONICAL[name]
            for name in selected_names
        }

        for i, p in enumerate(pieces_np):
            canonical = canonicalize_piece_from_grid(p)
            if canonical in selected_canonicals:
                inv[i] = inventory_count

    else:
        raise ValueError(f"Unknown inventory mode: {mode}")

    return jnp.asarray(inv, dtype=jnp.int32)

@jax.jit
def policy_only_action(policy_params, policy_states, rng_key, env_state):
    legal_mask = legal_action_mask_batch(
        env_state.board,
        env_state.inventory,
        env_state.pos,
        env_state.pieces,
    )

    (logits, _), _ = p_model.apply(
        policy_params,
        policy_states,
        rng_key,
        board=env_state.board,
        inventory=env_state.inventory,
        pos=env_state.pos,
        pieces=env_state.pieces,
        is_sampling=False,
    )

    logits = jnp.where(legal_mask, logits, -1e9)
    actions = jnp.argmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    has_any_legal = jnp.any(legal_mask, axis=-1)
    return actions, probs, has_any_legal


@partial(jax.jit, static_argnames=("num_simulations", "max_depth"))
def mcts_action(
    policy_params,
    policy_states,
    rng_key,
    env_state,
    checker_params,
    seq_policy_params,
    seq_policy_states,
    num_simulations=256,
    max_depth=50,
):
    def recurrent_fn(p, key, actions, e_state):
        legal_mask_now = legal_action_mask_batch(
            e_state.board,
            e_state.inventory,
            e_state.pos,
            e_state.pieces,
        )
        has_any_legal_now = jnp.any(legal_mask_now, axis=1)
        done_uncover = ~has_any_legal_now

        next_board, next_inventory, next_pos, next_done, next_legal = jax.vmap(
            pure_step_jax
        )(e_state, actions)

        next_legal_mask = legal_action_mask_batch(
            next_board,
            next_inventory,
            next_pos,
            e_state.pieces,
        )

        (logits, val), _ = p_model.apply(
            p,
            policy_states,
            key,
            board=next_board,
            inventory=next_inventory,
            pos=next_pos,
            pieces=e_state.pieces,
            is_sampling=False,
        )
        logits = jnp.where(next_legal_mask, logits, -1e9)

        stability_p = checker_predict_batch_no_hold_jax(next_board, checker_params)
        seq_value = seq_value_from_tiling_board(
            seq_policy_params,
            seq_policy_states,
            next_board,
        )
        is_stable = stability_p > 0.5

        piece_pixels = jnp.sum(e_state.pieces, axis=(-2, -1))
        total_valid_pixels = jnp.sum(next_board != -1, axis=(-2, -1))
        pieces_reward = piece_pixels / jnp.maximum(total_valid_pixels[..., None], 1)
        batch_indices = jnp.arange(actions.shape[0])

        collapse_penalty = 0.02
        uncover_penalty = 0.02
        success_bonus = 0.2

        base_reward = jnp.where(
            next_legal & (~done_uncover),
            stability_p * seq_value * pieces_reward[batch_indices, actions],
            0.0,
        )
        collapse_term = jnp.where(next_legal & (~is_stable), -collapse_penalty, 0.0)
        uncover_term = jnp.where(done_uncover, -uncover_penalty, 0.0)
        success_term = jnp.where(next_done & is_stable & (~done_uncover), success_bonus, 0.0)
        reward_raw = base_reward + collapse_term + uncover_term + success_term

        terminal = next_done | done_uncover
        discount = jnp.where(terminal, 0.0, 1.0)

        v_min = -0.02
        v_max = 1.2
        v_scale = v_max - v_min
        reward = (reward_raw + discount * v_min - v_min) / v_scale

        next_e_state = e_state._replace(
            board=next_board,
            pos=next_pos,
            inventory=next_inventory,
            stable=is_stable,
        )

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=val,
        ), next_e_state

    root_legal_mask = legal_action_mask_batch(
        env_state.board,
        env_state.inventory,
        env_state.pos,
        env_state.pieces,
    )

    (init_logits, init_value), _ = p_model.apply(
        policy_params,
        policy_states,
        rng_key,
        board=env_state.board,
        inventory=env_state.inventory,
        pos=env_state.pos,
        pieces=env_state.pieces,
        is_sampling=False,
    )

    init_logits = jnp.where(root_legal_mask, init_logits, -1e9)
    invalid_actions = ~root_legal_mask

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
        invalid_actions=invalid_actions,
        max_num_considered_actions=env_state.inventory.shape[1],
    )
    return policy_output.action, policy_output.action_weights


def build_shape_trial_batch(shape_specs, board_size, pieces, pieces_id, inventory, num_trials, seed):
    blocked_masks = []
    shape_names = []
    shape_indices = []
    trial_indices = []
    trial_seeds = []

    for shape_idx, spec in enumerate(shape_specs):
        canon = canonicalize_mask(spec["mask"], board_size)
        for trial in range(num_trials):
            blocked_masks.append(canon)
            shape_names.append(spec["name"])
            shape_indices.append(shape_idx)
            trial_indices.append(trial)
            trial_seeds.append(int(seed + shape_idx * 100000 + trial))

    blocked_mask = jnp.stack(blocked_masks, axis=0)
    return {
        "pieces": pieces,
        "pieces_id": pieces_id,
        "inventory": inventory,
        "blocked_mask": blocked_mask,
        "shape_names": shape_names,
        "shape_indices": np.asarray(shape_indices, dtype=np.int32),
        "trial_indices": np.asarray(trial_indices, dtype=np.int32),
        "trial_seeds": np.asarray(trial_seeds, dtype=np.int64),
    }


def mask_state_batch(state, active_mask):
    active = jnp.asarray(active_mask, dtype=bool)

    def select_active(x, reset_x):
        shape = (active.shape[0],) + (1,) * (x.ndim - 1)
        return jnp.where(active.reshape(shape), x, reset_x)

    return jax.tree_util.tree_map(select_active, state, state)


def run_batched_episodes(
    env,
    policy_params,
    policy_states,
    seq_policy_params,
    seq_policy_states,
    rng_key,
    shape_names,
    mode="mcts",
    num_simulations=256,
    max_depth=50,
    max_steps=200,
    verbose=False,
):
    obs = env._get_obs()
    batch_size = int(obs.board.shape[0])

    active = np.ones((batch_size,), dtype=np.bool_)
    final_success = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_uncover = np.zeros((batch_size,), dtype=np.bool_)
    terminated_by_max_steps = np.ones((batch_size,), dtype=np.bool_)
    steps_taken = np.zeros((batch_size,), dtype=np.int32)

    initial_boards = np.asarray(obs.board).astype(np.int32)
    total_voxels = np.sum(initial_boards != -1, axis=(1, 2)).astype(np.int32)

    best_boards = initial_boards.copy()
    best_cover = np.sum(best_boards > 0, axis=(1, 2)).astype(np.int32)
    collapse_next_boards = [None for _ in range(batch_size)]

    for step in range(max_steps):
        if not active.any():
            break

        legal_mask_now = legal_action_mask_batch(
            env.state.board,
            env.state.inventory,
            env.state.pos,
            env.state.pieces,
        )
        has_any_legal = np.asarray(jnp.any(legal_mask_now, axis=1)).astype(np.bool_)
        apply_mask = active & has_any_legal

        no_legal = active & (~has_any_legal)
        if np.any(no_legal):
            terminated_by_uncover[no_legal] = True
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
            )
        elif mode == "mcts":
            actions, probs = mcts_action(
                policy_params=policy_params,
                policy_states=policy_states,
                rng_key=subkey,
                env_state=safe_state,
                checker_params=env.checker_params,
                seq_policy_params=seq_policy_params,
                seq_policy_states=seq_policy_states,
                num_simulations=num_simulations,
                max_depth=max_depth,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dummy_actions = jnp.zeros((batch_size,), dtype=jnp.int32)
        actions = jnp.where(jnp.asarray(apply_mask), actions, dummy_actions)

        obs_before = env._get_obs()
        sim_board, _, _, _, sim_legal = jax.vmap(pure_step_jax)(obs_before, actions)
        sim_board_np = np.asarray(sim_board).astype(np.int32)
        sim_legal_np = np.asarray(sim_legal).astype(np.bool_)

        # Mark inactive rows unstable to skip their Newton checks in env.step.
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
            newton_boards,
            newton_labels,
            checker_probs,
            checker_labels,
        ) = env.step(actions)

        done_np = np.asarray(done).astype(np.bool_)
        success_np = np.asarray(success).astype(np.bool_)
        stable_np = np.asarray(next_obs.stable).astype(np.bool_)
        next_board_np = np.asarray(next_obs.board).astype(np.int32)

        # Restore inactive rows so they do not get accidentally advanced/reset in logs.
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

        active_indices = np.where(apply_mask)[0]
        for idx in active_indices:
            steps_taken[idx] = step + 1
            if success_np[idx]:
                final_success[idx] = True
                terminated_by_max_steps[idx] = False
                cover = int(np.sum(sim_board_np[idx] > 0))
                if cover >= best_cover[idx]:
                    best_cover[idx] = cover
                    best_boards[idx] = sim_board_np[idx].copy()
            elif not done_np[idx]:
                if stable_np[idx]:
                    cover = int(np.sum(next_board_np[idx] > 0))
                    if cover >= best_cover[idx]:
                        best_cover[idx] = cover
                        best_boards[idx] = next_board_np[idx].copy()
            else:
                terminated_by_max_steps[idx] = False
                if sim_legal_np[idx]:
                    collapse_next_boards[idx] = sim_board_np[idx].copy()


        active[active_indices] &= ~done_np[active_indices]

    return [
        {
            "best_board": best_boards[i].astype(np.int32),
            "best_cover": int(best_cover[i]),
            "total_voxels": int(total_voxels[i]),
            "cover_ratio": float(best_cover[i] / max(int(total_voxels[i]), 1)),
            "collapse_next_board": collapse_next_boards[i],
            "final_success": bool(final_success[i]),
            "terminated_by_uncover": bool(terminated_by_uncover[i]),
            "terminated_by_max_steps": bool(terminated_by_max_steps[i]),
            "steps": int(steps_taken[i]),
        }
        for i in range(batch_size)
    ]


def save_board_outputs(board, out_dir, stem):
    out_dir = Path(out_dir)
    (out_dir / "boards").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "boards" / f"{stem}.npy"
    png_path = out_dir / "previews" / f"{stem}.png"
    np.save(npy_path, np.asarray(board, dtype=np.int32))
    render_equivalent_sets(board, save_path=str(png_path))
    return str(npy_path), str(png_path)


def infer_folder(
    png_dir,
    out_root,
    ckpt_path,
    checker_ckpt_path,
    seq_policy_ckpt_path=None,
    board_size=(16, 16),
    mode="mcts",
    num_trials=1,
    num_simulations=256,
    max_depth=50,
    max_steps=200,
    seed=0,
    verbose=False,
    inventory_mode="default",
    inventory_pieces=None,
    inventory_count=99,
    batch_size=64,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_trials <= 0:
        raise ValueError("num_trials must be positive")

    pieces, pieces_id = generate_polyominoes_from_list([3, 4, 5], grid_size=4)
    inventory = make_inventory(
        inventory_mode,
        pieces,
        inventory_pieces=inventory_pieces,
        inventory_count=inventory_count,
    )

    shape_specs = load_masks_from_png_dir(Path(png_dir))
    if len(shape_specs) == 0:
        raise FileNotFoundError(f"No masks loaded from {png_dir}")

    policy_params, policy_states, extra = load_checkpoint(str(ckpt_path))

    seq_policy_params, seq_policy_states = None, None
    if seq_policy_ckpt_path is not None:
        seq_policy_params, seq_policy_states, seq_extra = load_seq_checkpoint(
            str(seq_policy_ckpt_path)
        )

    batch_info = build_shape_trial_batch(
        shape_specs=shape_specs,
        board_size=board_size,
        pieces=pieces,
        pieces_id=pieces_id,
        inventory=inventory,
        num_trials=num_trials,
        seed=seed,
    )

    env = TetrixEnv(
        board_size=board_size,
        checker_ckpt_path=str(checker_ckpt_path),
        seq_policy_params=seq_policy_params,
        seq_policy_states=seq_policy_states,
    )

    png_dir_name = Path(png_dir).name
    ckpt_stem = Path(ckpt_path).stem
    inventory_tag = inventory_mode
    if inventory_mode == "custom":
        inventory_tag = "custom_" + str(inventory_pieces).replace(",", "_").replace(" ", "")

    seq_ckpt_stem = Path(seq_policy_ckpt_path).stem if seq_policy_ckpt_path is not None else "none"
    
    save_root = Path(out_root) / png_dir_name / mode / f"{ckpt_stem}_seq_{seq_ckpt_stem}" / inventory_tag
    save_root.mkdir(parents=True, exist_ok=True)

    total_episodes = len(batch_info["shape_names"])
    num_batches = (total_episodes + batch_size - 1) // batch_size

    config = {
        "png_dir": str(png_dir),
        "out_root": str(out_root),
        "save_root": str(save_root),
        "ckpt_path": str(ckpt_path),
        "checker_ckpt_path": str(checker_ckpt_path),
        "board_size": list(board_size),
        "mode": mode,
        "num_trials": int(num_trials),
        "num_simulations": int(num_simulations),
        "max_depth": int(max_depth),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "batched": True,
        "batch_size": int(min(batch_size, total_episodes)),
        "max_batch_size": int(batch_size),
        "total_episodes": int(total_episodes),
        "num_batches": int(num_batches),
        "inventory_mode": str(inventory_mode),
        "inventory_pieces": inventory_pieces,
        "inventory_count": int(inventory_count),
        "seq_policy_ckpt_path": str(seq_policy_ckpt_path) if seq_policy_ckpt_path is not None else None,
    }

    print(f"Inference: Tiling | Mode: {'MCTS' if mode == 'mcts' else 'Policy only'}", flush=True)
    print(
        f"Data: {len(shape_specs)} shapes | Trials per shape: {num_trials} | "
        f"Batch size: {min(batch_size, total_episodes)}",
        flush=True,
    )
    print(
        f"Policy: Loaded | Stability checker: Loaded | "
        f"Sequence policy: {'Loaded' if seq_policy_params is not None else 'Disabled'}",
        flush=True,
    )
    print(f"Output: {save_root}", flush=True)

    t0 = time.perf_counter()
    base_rng_key = jax.random.PRNGKey(int(seed))
    shape_logs_by_idx = {}
    total_success = 0
    total_trials = 0

    for batch_idx, start in enumerate(range(0, total_episodes, batch_size)):
        end = min(start + batch_size, total_episodes)
        batch_t0 = time.perf_counter()
        env.reset_batch(
            pieces=batch_info["pieces"],
            pieces_id=batch_info["pieces_id"],
            inventory=batch_info["inventory"],
            blocked_mask=batch_info["blocked_mask"][start:end],
        )
        # Preserve the old RNG key for a single batch; give later batches distinct keys.
        rng_key = base_rng_key if batch_idx == 0 else jax.random.fold_in(base_rng_key, batch_idx)
        results = run_batched_episodes(
            env=env,
            policy_params=policy_params,
            policy_states=policy_states,
            seq_policy_params=seq_policy_params,
            seq_policy_states=seq_policy_states,
            rng_key=rng_key,
            shape_names=batch_info["shape_names"][start:end],
            mode=mode,
            num_simulations=num_simulations,
            max_depth=max_depth,
            max_steps=max_steps,
            verbose=verbose,
        )
        if len(results) != end - start:
            raise RuntimeError("Batch result count does not match the input count")

        for flat_idx, result in enumerate(results, start=start):
            shape_idx = int(batch_info["shape_indices"][flat_idx])
            trial = int(batch_info["trial_indices"][flat_idx])
            trial_seed = int(batch_info["trial_seeds"][flat_idx])
            shape_name = batch_info["shape_names"][flat_idx]

            trial_stem = f"{shape_name}_trial_{trial:03d}"

            max_npy, max_png = save_board_outputs(
                result["best_board"],
                save_root,
                trial_stem,
            )

            collapse_npy = None
            collapse_png = None
            if result["collapse_next_board"] is not None:
                collapse_npy, collapse_png = save_board_outputs(
                    result["collapse_next_board"],
                    save_root / "failures",
                    trial_stem,
                )

            success_bool = bool(result["final_success"])
            total_success += int(success_bool)
            total_trials += 1

            row = {
                "trial": int(trial),
                "seed": trial_seed,
                "success": success_bool,
                "best_cover": int(result["best_cover"]),
                "total_voxels": int(result["total_voxels"]),
                "cover_ratio": float(result["cover_ratio"]),
                "steps": int(result["steps"]),
                "terminated_by_uncover": bool(result["terminated_by_uncover"]),
                "terminated_by_max_steps": bool(result["terminated_by_max_steps"]),
                "has_collapse_next": result["collapse_next_board"] is not None,
                "maxcover_npy": max_npy,
                "maxcover_png": max_png,
                "collapse_next_npy": collapse_npy,
                "collapse_next_png": collapse_png,
            }

            if shape_idx not in shape_logs_by_idx:
                shape_logs_by_idx[shape_idx] = {
                    "shape_name": shape_name,
                    "num_trials": int(num_trials),
                    "success_count": 0,
                    "success_rate": 0.0,
                    "best_cover": -1,
                    "best_trial": -1,
                    "best_over_trials_npy": None,
                    "best_over_trials_png": None,
                    "trials": [],
                    "_best_board": None,
                }

            shape_log = shape_logs_by_idx[shape_idx]
            shape_log["trials"].append(row)
            shape_log["success_count"] += int(success_bool)
            if int(result["best_cover"]) > int(shape_log["best_cover"]):
                shape_log["best_cover"] = int(result["best_cover"])
                shape_log["best_trial"] = int(trial)
                shape_log["_best_board"] = result["best_board"].copy()


        print(
            f"Batch {batch_idx + 1}/{num_batches} | Completed: {total_trials}/{total_episodes} | "
            f"Time: {time.perf_counter() - batch_t0:.2f}s",
            flush=True,
        )
        del results

    shape_logs = []
    for shape_idx in sorted(shape_logs_by_idx):
        shape_summary = shape_logs_by_idx[shape_idx]
        shape_name = shape_summary["shape_name"]
        shape_out_dir = save_root / "logs"
        shape_out_dir.mkdir(parents=True, exist_ok=True)

        best_board = shape_summary.pop("_best_board")
        if best_board is not None:
            best_npy, best_png = save_board_outputs(
                best_board,
                save_root / "best",
                shape_name,
            )
            shape_summary["best_over_trials_npy"] = best_npy
            shape_summary["best_over_trials_png"] = best_png

        shape_summary["trials"] = sorted(shape_summary["trials"], key=lambda x: x["trial"])
        shape_summary["success_rate"] = float(
            shape_summary["success_count"] / max(shape_summary["num_trials"], 1)
        )

        shape_json_path = shape_out_dir / f"{shape_name}.json"
        with open(shape_json_path, "w", encoding="utf-8") as f:
            json.dump(shape_summary, f, indent=2, ensure_ascii=False)

        shape_logs.append(shape_summary)

    total_time = time.perf_counter() - t0
    overall = {
        "config": config,
        "total_shapes": int(len(shape_specs)),
        "total_trials": int(total_trials),
        "success_count": int(total_success),
        "success_rate": float(total_success / max(total_trials, 1)),
        "runtime_sec": float(total_time),
        "shapes": shape_logs,
    }

    summary_json_path = save_root / "infer_summary.json"
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
    parser.add_argument("--png_dir", "--mask_dir", type=str, default=str(PROJECT_DIR / "dataset/png"), help='default: %(default)s')
    parser.add_argument("--out_root", type=str, default=str(PROJECT_DIR / "results/inference_tiling"), help='default: %(default)s')
    parser.add_argument("--ckpt_path", type=str, default=str(PROJECT_DIR / "checkpoints/policy_til.pkl"), help='default: %(default)s')
    parser.add_argument("--checker_ckpt_path", type=str, default=str(PROJECT_DIR / "checkpoints/stability_checker_jax_til.pkl"), help='default: %(default)s')
    parser.add_argument(
        "--seq_policy_ckpt_path",
        type=str,
        default=None,
        help='default: %(default)s',
    )
    parser.add_argument("--mode", type=str, default="mcts", choices=["mcts", "policy_only"], help='default: %(default)s')
    parser.add_argument("--num_trials", type=int, default=10, help='default: %(default)s')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='default: %(default)s')
    parser.add_argument("--seed", type=int, default=0, help='default: %(default)s')
    # Accept the legacy flag; inference always uses concise logs.
    parser.add_argument("--hide_step_log", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument(
        "--inventory_mode",
        type=str,
        default="default",
        choices=[
            "default",
            "tri_only",
            "tetra_only",
            "penta_only",
            "custom",
        ],
        help='default: %(default)s',
    )

    parser.add_argument(
        "--inventory_pieces",
        type=str,
        default=None,
        help='default: %(default)s',
    )

    parser.add_argument(
        "--inventory_count",
        type=int,
        default=99,
        help='default: %(default)s',
    )

    parser.add_argument(
        "--inventory_runs",
        type=str,
        default=None,
        help='default: %(default)s',
    )
    

    args = parser.parse_args(argv)
    if args.num_trials < 1:
        parser.error("--num_trials must be positive")
    if args.batch_size < 1 or args.inventory_count < 1:
        parser.error("--batch_size and --inventory_count must be positive")
    if args.inventory_runs is not None and not any(item.strip() for item in args.inventory_runs.split(";")):
        parser.error("--inventory_runs must contain at least one run")
    if not Path(args.png_dir).is_dir():
        parser.error(f"PNG directory does not exist: {args.png_dir}")
    for name in ("ckpt_path", "checker_ckpt_path", "seq_policy_ckpt_path"):
        path = getattr(args, name)
        if path is not None and not Path(path).is_file():
            parser.error(f"--{name} file does not exist: {path}")
    return args


def main(argv=None):
    args = parse_args(argv)

    if args.inventory_runs is None:
        run_specs = [(args.inventory_mode, args.inventory_pieces)]
    else:
        run_specs = []
        for item in args.inventory_runs.split(";"):
            item = item.strip()
            if item == "":
                continue

            if item in ["default", "tri_only", "tetra_only", "penta_only"]:
                run_specs.append((item, None))
            else:
                run_specs.append(("custom", item))

    for inv_mode, inv_pieces in run_specs:

        infer_folder(
            png_dir=args.png_dir,
            out_root=args.out_root,
            ckpt_path=args.ckpt_path,
            checker_ckpt_path=args.checker_ckpt_path,
            seq_policy_ckpt_path=args.seq_policy_ckpt_path,
            mode=args.mode,
            num_trials=args.num_trials,
            batch_size=args.batch_size,
            seed=args.seed,
            verbose=not args.hide_step_log,
            inventory_mode=inv_mode,
            inventory_pieces=inv_pieces,
            inventory_count=args.inventory_count,
        )


if __name__ == "__main__":
    main()
