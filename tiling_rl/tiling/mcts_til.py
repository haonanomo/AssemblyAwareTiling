"""MCTS, policy optimization, and validation for assembly-aware tiling.

Training configuration and PNG dataset loading are handled by train_til.py.
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

from tiling_rl.common.render import render_equivalent_sets
from tiling_rl.physics.simulator_newton_til import check_stability_newton
from tiling_rl.stability.stability_checker_jax_til import checker_predict_batch_no_hold_jax
from tiling_rl.tiling.env_til import (
    EnvState,
    canonicalize_blocked_mask,
    get_board_rb_most_empty_pixel,
    legal_action_mask_batch,
    pure_step_jax,
    seq_value_from_tiling_board,
)
from tiling_rl.tiling.model_til import AttentionPolicyPatch as AttentionPolicy

logger = logging.getLogger(__name__)


def p_forward_fn(board, inventory, pos, pieces, is_sampling):
    policy = AttentionPolicy()
    return policy(board, inventory, pos, pieces, is_sampling)

p_model = hk.transform_with_state(p_forward_fn)


def load_seq_checkpoint(path):
    """Load auxiliary sequence policy parameters, state, and metadata."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["policy_params"], payload["policy_states"], payload.get("extra", None)


def save_checkpoint(path, policy_params, policy_states, extra=None):
    """Save policy parameters, state, and optional training metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_params": policy_params,
        "policy_states": policy_states,
        "extra": extra,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path):
    """Load policy parameters, state, and optional training metadata."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["policy_params"], payload["policy_states"], payload.get("extra", None)


def compute_checker_audit_stats(probs, labels, threshold=0.5):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)

    if probs.size == 0:
        return None

    pred = probs > threshold
    y = labels > 0.5

    stable_probs = probs[y]
    unstable_probs = probs[~y]

    tp = int(np.sum(pred & y))
    tn = int(np.sum((~pred) & (~y)))
    fp = int(np.sum(pred & (~y)))
    fn = int(np.sum((~pred) & y))

    acc = (tp + tn) / max(len(y), 1)
    stable_acc = tp / max(int(np.sum(y)), 1)
    unstable_acc = tn / max(int(np.sum(~y)), 1)
    balanced_acc = 0.5 * (stable_acc + unstable_acc)

    brier = float(np.mean((probs - labels) ** 2))

    eps = 1e-6
    nll = -np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))

    confident_wrong = np.mean(((probs > 0.8) & (~y)) | ((probs < 0.2) & y))
    confident_count = np.sum((probs > 0.8) | (probs < 0.2))
    confident_correct = np.sum(((probs > 0.8) & y) | ((probs < 0.2) & (~y)))
    confident_acc = confident_correct / max(confident_count, 1)

    try:
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(probs), dtype=np.float32)

        n_pos = np.sum(y)
        n_neg = np.sum(~y)
        if n_pos > 0 and n_neg > 0:
            rank_sum_pos = np.sum(ranks[y])
            auc = (rank_sum_pos - n_pos * (n_pos - 1) / 2.0) / max(n_pos * n_neg, 1)
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    def safe_mean(x):
        return float(np.mean(x)) if x.size > 0 else float("nan")

    def safe_std(x):
        return float(np.std(x)) if x.size > 0 else float("nan")

    def safe_q(x, q):
        return float(np.quantile(x, q)) if x.size > 0 else float("nan")

    return {
        "n": int(len(probs)),
        "acc": float(acc),
        "balanced_acc": float(balanced_acc),
        "stable_acc": float(stable_acc),
        "unstable_acc": float(unstable_acc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "brier": brier,
        "nll": float(nll),
        "auc": float(auc),
        "p_mean": safe_mean(probs),
        "p_std": safe_std(probs),
        "p_stable_mean": safe_mean(stable_probs),
        "p_unstable_mean": safe_mean(unstable_probs),
        "p_stable_q10": safe_q(stable_probs, 0.10),
        "p_stable_q50": safe_q(stable_probs, 0.50),
        "p_stable_q90": safe_q(stable_probs, 0.90),
        "p_unstable_q10": safe_q(unstable_probs, 0.10),
        "p_unstable_q50": safe_q(unstable_probs, 0.50),
        "p_unstable_q90": safe_q(unstable_probs, 0.90),
        "confident_wrong_rate": float(confident_wrong),
        "confident_acc": float(confident_acc),
        "confident_count": int(confident_count),
    }


def format_checker_audit_stats(stats):
    if stats is None:
        return "checker_audit=NA"

    return (
        f"checker_n={stats['n']} acc={stats['acc']:.4f} "
        f"balanced_acc={stats['balanced_acc']:.4f} "
        f"fp={stats['fp']} fn={stats['fn']} brier={stats['brier']:.4f}"
    )


@partial(jax.jit, static_argnames=("optimizer", "p_model"))
def train_step(optimizer, params, state, opt_state, bd, inv, pos, pc, pr, vs,
               weights=(1.0, 1.0), *, p_model=p_model):
    """Optimize masked policy cross-entropy and scaled value regression."""
    def loss_fn(params):
        (logits, values), new_state = p_model.apply(
            params, state, None,
            board=bd,
            inventory=inv,
            pos=pos,
            pieces=pc,
            is_sampling=False
        )
        legal_mask = legal_action_mask_batch(bd, inv, pos, pc)
        has_any_legal = jnp.any(legal_mask, axis=1)
        logits = jnp.where(legal_mask, logits, -1e9)
        policy_ce = optax.softmax_cross_entropy(logits, pr)
        policy_loss = jnp.where(
            jnp.any(has_any_legal),
            jnp.sum(policy_ce * has_any_legal.astype(jnp.float32)) /
            jnp.sum(has_any_legal.astype(jnp.float32)),
            0.0
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
            0.0
        )
        entropy = jnp.where(
            jnp.any(has_any_legal),
            jnp.sum(entropy_per_sample) / jnp.sum(has_any_legal.astype(jnp.float32)),
            0.0
        )

        total_loss = weights[0] * policy_loss + weights[1] * value_loss
        return total_loss, (new_state, policy_loss, value_loss, entropy)

    grads, (new_state, policy_loss, value_loss, entropy) = jax.grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state, policy_loss, value_loss, entropy


@partial(jax.jit, static_argnames=("p_model",))
def policy_only_action(policy_params, policy_states, rng_key, env_state, *, p_model=p_model):
    """Choose the highest-logit legal action for each board."""
    legal_mask = legal_action_mask_batch(
        env_state.board,
        env_state.inventory,
        env_state.pos,
        env_state.pieces
    )

    (logits, _), _ = p_model.apply(
        policy_params,
        policy_states,
        rng_key,
        board=env_state.board,
        inventory=env_state.inventory,
        pos=env_state.pos,
        pieces=env_state.pieces,
        is_sampling=False
    )

    logits = jnp.where(legal_mask, logits, -1e9)
    actions = jnp.argmax(logits, axis=-1)
    has_any_legal = jnp.any(legal_mask, axis=-1)

    return actions, has_any_legal


def validate_policy_only(policy_params, policy_states, iteration, *, shape_specs,
                         board_size, inventory, pieces, pieces_id, result_save_folder,
                         eval_batch_size=64, p_model=p_model):
    """Evaluate a PNG-derived shape set with greedy actions and final physics checks."""
    t0 = time.perf_counter()

    if not shape_specs:
        raise ValueError("Validation requires at least one PNG shape")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    Path(result_save_folder).mkdir(parents=True, exist_ok=True)
    max_valid_cells = max(int(np.sum(np.asarray(spec["mask"]) == 0)) for spec in shape_specs)
    max_steps = max_valid_cells // 3 + 8

    all_final_boards = []
    all_geom_success = []
    all_shape_names = []
    all_cover = []

    rng_key = jax.random.PRNGKey(100000 + iteration)

    for start in range(0, len(shape_specs), eval_batch_size):
        end = min(start + eval_batch_size, len(shape_specs))
        specs = shape_specs[start:end]
        batch_size = len(specs)

        blocked_masks = []
        shape_names = []

        for spec in specs:
            blocked_masks.append(
                canonicalize_blocked_mask(spec["mask"], board_size=board_size)
            )
            shape_names.append(spec["name"])

        blocked_mask = jnp.stack(blocked_masks, axis=0)

        batch_inventory = inventory[jnp.newaxis, ...].repeat(batch_size, axis=0)
        batch_pieces = pieces[jnp.newaxis, ...].repeat(batch_size, axis=0)
        batch_pieces_id = pieces_id[jnp.newaxis, ...].repeat(batch_size, axis=0)

        board = jnp.zeros((batch_size, *board_size), dtype=jnp.int32)
        board = jnp.where(blocked_mask > 0, -1, board)

        state = EnvState(
            board=board,
            inventory=batch_inventory,
            pos=jax.vmap(get_board_rb_most_empty_pixel)(board),
            pieces=batch_pieces,
            pieces_id=batch_pieces_id,
            stable=jnp.ones((batch_size,), dtype=bool),
        )

        active = np.ones((batch_size,), dtype=np.bool_)
        geom_success = np.zeros((batch_size,), dtype=np.bool_)

        for _ in range(max_steps):
            if not active.any():
                break

            rng_key, subkey = jax.random.split(rng_key)

            actions, has_any_legal = policy_only_action(
                policy_params,
                policy_states,
                subkey,
                state,
                p_model=p_model,
            )

            active_jnp = jnp.asarray(active)
            apply_mask = active_jnp & has_any_legal

            dummy_actions = jnp.zeros((batch_size,), dtype=jnp.int32)
            safe_actions = jnp.where(apply_mask, actions, dummy_actions)

            new_board, new_inv, new_pos, done_geom, legal = jax.vmap(
                pure_step_jax
            )(state, safe_actions)

            new_state = state._replace(
                board=new_board,
                inventory=new_inv,
                pos=new_pos,
            )

            def select_updated(old, new):
                shape = (batch_size,) + (1,) * (old.ndim - 1)
                return jnp.where(apply_mask.reshape(shape), new, old)

            state = jax.tree_util.tree_map(select_updated, state, new_state)

            done_np = np.asarray(done_geom)
            legal_np = np.asarray(legal)
            apply_np = np.asarray(apply_mask)
            has_legal_np = np.asarray(has_any_legal)

            geom_success |= apply_np & legal_np & done_np
            active = active & has_legal_np & (~done_np)

        final_boards = np.asarray(state.board).astype(np.int32)
        cover = np.sum(final_boards > 0, axis=(1, 2))

        all_final_boards.append(final_boards)
        all_geom_success.append(geom_success)
        all_shape_names.extend(shape_names)
        all_cover.append(cover)

    final_boards = np.concatenate(all_final_boards, axis=0)
    geom_success = np.concatenate(all_geom_success, axis=0)
    cover = np.concatenate(all_cover, axis=0)

    stable = np.zeros((final_boards.shape[0],), dtype=np.bool_)

    if np.any(geom_success):
        to_check = final_boards[geom_success]
        stable_result = np.asarray(
            check_stability_newton(to_check, render=False)
        ).astype(np.bool_)
        stable[geom_success] = stable_result

    success = geom_success & stable

    success_n = int(np.sum(success))
    total_n = int(len(success))
    geom_n = int(np.sum(geom_success))
    stable_n = int(np.sum(stable))
    avg_cover = float(np.mean(cover)) if len(cover) > 0 else 0.0
    max_cover = int(np.max(cover)) if len(cover) > 0 else 0

    validation_dir = Path(result_save_folder) / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(
        validation_dir,
        f"policy_only_validation_iter_{iteration + 1:04d}.npz"
    )

    np.savez_compressed(
        save_path,
        boards=final_boards,
        geom_success=geom_success.astype(np.float32),
        stable=stable.astype(np.float32),
        success=success.astype(np.float32),
        cover=cover.astype(np.int32),
        shape_names=np.asarray(all_shape_names),
    )

    dt = time.perf_counter() - t0

    logger.debug(
        f"[Validation] iter={iteration + 1} | "
        f"success: {success_n}/{total_n} | "
        f"geom: {geom_n}/{total_n} | "
        f"stable: {stable_n}/{max(geom_n, 1)} | "
        f"avg_cover={avg_cover:.2f} | max_cover={max_cover} | "
        f"time={dt:.2f}s | saved={save_path}"
    )


@partial(jax.jit, static_argnames=("p_model",))
def mcts_action(policy_params, policy_states, rng_key, env_state, checker_params,
                seq_policy_params, seq_policy_states, is_sampling=False, *, p_model=p_model):
    """Run Gumbel MuZero search with stability and assembly-aware rewards."""
    def recurrent_fn(p, key, actions, e_state):
        legal_mask_now = legal_action_mask_batch(
            e_state.board,
            e_state.inventory,
            e_state.pos,
            e_state.pieces
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
            e_state.pieces
        )

        (logits, val), _ = p_model.apply(
            p,
            policy_states,
            key,
            board=next_board,
            inventory=next_inventory,
            pos=next_pos,
            pieces=e_state.pieces,
            is_sampling=is_sampling
        )

        logits = jnp.where(next_legal_mask, logits, -1e9)

        stability_p = checker_predict_batch_no_hold_jax(next_board, checker_params)
        seq_value = seq_value_from_tiling_board(seq_policy_params, seq_policy_states, next_board)
        is_stable = (stability_p > 0.5)

        piece_pixels = jnp.sum(e_state.pieces, axis=(-2, -1))
        total_valid_pixels = jnp.sum(next_board != -1, axis=(-2, -1))
        pieces_reward = piece_pixels / jnp.maximum(total_valid_pixels[..., jnp.newaxis], 1)
        batch_indices = jnp.arange(actions.shape[0])
        collapse_penalty = 0.02
        uncover_penalty = 0.02
        success_bonus = 0.2
        base_reward = jnp.where(
            next_legal & (~done_uncover),
            stability_p * seq_value * pieces_reward[batch_indices, actions],
            0.0
        )
        collapse_term = jnp.where(
            next_legal & (~is_stable),
            -collapse_penalty,
            0.0
        )
        uncover_term = jnp.where(
            done_uncover,
            -uncover_penalty,
            0.0
        )
        success_term = jnp.where(
            next_done & is_stable & (~done_uncover),
            success_bonus,
            0.0
        )
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
            stable=is_stable
        )

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=val
        ), next_e_state

    root_legal_mask = legal_action_mask_batch(
        env_state.board,
        env_state.inventory,
        env_state.pos,
        env_state.pieces
    )

    (init_logits, init_value), _ = p_model.apply(
        policy_params,
        policy_states,
        rng_key,
        board=env_state.board,
        inventory=env_state.inventory,
        pos=env_state.pos,
        pieces=env_state.pieces,
        is_sampling=is_sampling
    )
    init_logits = jnp.where(root_legal_mask, init_logits, -1e9)
    invalid_actions = ~root_legal_mask
    root = mctx.RootFnOutput(
        prior_logits=init_logits,
        value=init_value,
        embedding=env_state
    )
    policy_output = mctx.gumbel_muzero_policy(
        params=policy_params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=192,
        max_depth=30,
        invalid_actions=invalid_actions,
        max_num_considered_actions=env_state.inventory.shape[1]
    )
    return policy_output.action, policy_output.action_weights


def train_alphazero(env,
                    p_model,
                    result_save_folder,
                    seq_policy_params,
                    seq_policy_states,
                    num_iterations=10,
                    num_parallel_envs=4,
                    step_per_iterations=100,
                    batch_size=512,
                    num_train_steps=5,
                    batch_reset_fn=None,
                    checkpoint_every=5,
                    init_checkpoint_path=None,
                    seed=42,
                    validation_fn=None):
    """Train on a reset environment, optionally evaluating every ten iterations.

    batch_reset_fn(iteration) supplies PNG-derived masks and shared inventory.
    validation_fn receives policy_params, policy_states, and iteration as keywords.
    """
    Path(result_save_folder).mkdir(parents=True, exist_ok=True)

    output_dirs = {
        name: Path(result_save_folder) / name
        for name in ("boards", "previews", "stability_data", "checkpoints")
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    rng_key = jax.random.PRNGKey(seed)
    logger.debug("Random seed: %d", seed)

    rng_key, subkey = jax.random.split(rng_key, 2)

    if init_checkpoint_path is not None:
        policy_params, policy_states, extra = load_checkpoint(init_checkpoint_path)
        logger.debug("Loaded initial policy: %s", init_checkpoint_path)
        if extra is not None:
            logger.debug("Checkpoint metadata: %s", extra)
    else:
        policy_params, policy_states = p_model.init(
            subkey,
            board=env.state.board,
            inventory=env.state.inventory,
            pos=env.state.pos,
            pieces=env.state.pieces,
            is_sampling=False
        )
        logger.debug("Initialized policy from scratch")

    logger.info(
        "Policy: %s | Stability checker: %s | Sequence policy: %s",
        "Loaded" if init_checkpoint_path is not None else "Initialized",
        "Loaded" if env.checker else "Disabled",
        "Loaded" if seq_policy_params is not None else "Disabled",
    )
    logger.info("Output: %s", result_save_folder)

    optimizer = optax.adam(learning_rate=5e-5)
    opt_state = optimizer.init(policy_params)

    replay_buffer = deque(maxlen=200000)

    game_history = [[] for _ in range(num_parallel_envs)]
    loss_weights = [1.0, 1.0]

    best_result_by_shape = {}
    obs = env._get_obs()
    last_stable_buffer = [None] * num_parallel_envs


    logger.debug("Starting AlphaZero training: iterations=%d, environments=%d",
                num_iterations, num_parallel_envs)
    total_train_wall_t0 = time.perf_counter()
    for iteration in range(num_iterations):
        is_sampling = (iteration < 3)
        if batch_reset_fn is not None:
            reset_kwargs = batch_reset_fn(iteration)
            env.reset_batch(
                pieces=reset_kwargs["pieces"],
                pieces_id=reset_kwargs["pieces_id"],
                inventory=reset_kwargs["inventory"],
                blocked_mask=reset_kwargs["blocked_mask"],
            )
            obs = env._get_obs()
            game_history = [[] for _ in range(num_parallel_envs)]
            shape_names_for_envs = reset_kwargs["shape_names"]
            last_stable_buffer = [None] * num_parallel_envs
        else:
            shape_names_for_envs = ["single_shape"] * num_parallel_envs
        logger.debug("Iteration %d/%d", iteration + 1, num_iterations)
        iter_t0 = time.perf_counter()
        timing_mcts = 0.0
        timing_env_step = 0.0
        timing_newton = 0.0
        timing_sequence = 0.0

        num_success = 0
        num_done = 0
        iteration_newton_boards = []
        iteration_newton_labels = []

        checker_prob_buffer = []
        checker_label_buffer = []

        for istep in range(step_per_iterations):
            # Collect self-play transitions.
            rng_key, subkey = jax.random.split(rng_key)
            legal_mask_now = legal_action_mask_batch(
                env.state.board,
                env.state.inventory,
                env.state.pos,
                env.state.pieces
            )
            has_any_legal_now = jnp.any(legal_mask_now, axis=1)

            dummy_actions = jnp.zeros((num_parallel_envs,), dtype=jnp.int32)
            dummy_probs = jnp.zeros_like(env.state.inventory, dtype=jnp.float32)

            if bool(jnp.any(has_any_legal_now)):
                safe_state = jax.tree_util.tree_map(
                    lambda cur, reset: jnp.where(
                        has_any_legal_now.reshape((num_parallel_envs,) + (1,) * (cur.ndim - 1)),
                        cur,
                        reset
                    ),
                    env.state,
                    env.reset_state
                )

                t_mcts0 = time.perf_counter()
                mcts_actions, mcts_probs = mcts_action(
                    policy_params,
                    policy_states,
                    subkey,
                    safe_state,
                    env.checker_params,
                    seq_policy_params,
                    seq_policy_states,
                    is_sampling,
                    p_model=p_model,
                )
                mcts_actions.block_until_ready()
                mcts_probs.block_until_ready()
                timing_mcts += time.perf_counter() - t_mcts0

                actions = jnp.where(has_any_legal_now, mcts_actions, dummy_actions)
                probs = jnp.where(has_any_legal_now[:, None], mcts_probs, dummy_probs)
            else:
                actions = dummy_actions
                probs = dummy_probs

            t_step0 = time.perf_counter()
            next_obs, reward, done, success, newton_boards, newton_labels, checker_probs, checker_labels = env.step(actions)
            next_obs.board.block_until_ready()
            timing_env_step += time.perf_counter() - t_step0
            step_timing = getattr(env, "last_step_timing", {})
            timing_newton += float(step_timing.get("newton", 0.0))
            timing_sequence += float(step_timing.get("sequence", 0.0))

            if newton_boards is not None and newton_labels is not None:
                iteration_newton_boards.append(newton_boards)
                iteration_newton_labels.append(newton_labels)

                if checker_labels is not None and checker_probs is not None:
                    newton_np = np.asarray(newton_labels).astype(np.bool_).reshape(-1)
                    checker_prob_np = np.asarray(checker_probs).astype(np.float32).reshape(-1)

                    checker_prob_buffer.append(checker_prob_np)
                    checker_label_buffer.append(newton_np.astype(np.float32))

            for env_idx in range(num_parallel_envs):

                if obs.stable[env_idx]:
                    last_stable_buffer[env_idx] = jax.tree_util.tree_map(lambda x: x[env_idx], obs)

                game_history[env_idx].append({
                    "board": np.array(obs.board[env_idx, ...]),
                    "inventory": np.array(obs.inventory[env_idx, ...]),
                    "pos": np.array(obs.pos[env_idx, ...]),
                    "pieces": np.array(obs.pieces[env_idx, ...]),
                    "probs": np.array(probs[env_idx, ...]),
                    "stable": obs.stable[env_idx],
                    "done": done[env_idx],
                    "reward": reward[env_idx],
                })
            num_success += int(np.sum(np.asarray(success)))
            num_done += int(np.sum(np.asarray(done)))

            for env_idx in range(num_parallel_envs):
                shape_name = shape_names_for_envs[env_idx]
                shape_key = tuple(np.asarray(obs.board[env_idx] == -1).ravel().tolist())

                if shape_key not in best_result_by_shape:
                    best_result_by_shape[shape_key] = {
                        "name": shape_name,
                        "cover": -1,
                        "board": None,
                    }

                if done[env_idx]:
                    obs_i = last_stable_buffer[env_idx]
                    if obs_i is None:
                        continue

                    if success[env_idx]:
                        new_bd, _, _, _, _ = pure_step_jax(obs_i, actions[env_idx])
                    else:
                        new_bd = obs_i.board

                    cover = np.sum(np.array(new_bd) > 0)

                    best_cover = best_result_by_shape[shape_key]["cover"]

                    if cover > best_cover:
                        best_result_by_shape[shape_key]["cover"] = cover
                        best_result_by_shape[shape_key]["board"] = np.array(new_bd)

                        fname = output_dirs["previews"] / f"{shape_name}_maxcover_best.png"
                        render_equivalent_sets(new_bd, save_path=str(fname))

                        fname_np = output_dirs["boards"] / f"{shape_name}_maxcover_best.npy"
                        np.save(fname_np, np.array(new_bd))

                        logger.debug("Best coverage for %s: %d (iteration %d, step %d)",
                                    shape_name, cover, iteration + 1, istep + 1)

            obs = next_obs


        accum_rewards = []


        for env_idx in range(num_parallel_envs):
            istep = len(game_history[env_idx]) - 1
            while not game_history[env_idx][istep]["done"] and istep >= 0:
                istep = istep - 1
            save_to_buffer = copy.deepcopy(game_history[env_idx][: (istep + 1)])
            game_history[env_idx] = game_history[env_idx][istep + 1:]

            v = 0
            for data in reversed(save_to_buffer):
                r = data['reward']
                done = data['done']
                if done:
                    accum_rewards.append(r)
                else:
                    accum_rewards[-1] += r
                v = r + v * (1.0 - done)
                replay_buffer.append((data["board"],
                                      data['inventory'],
                                      data['pos'],
                                      data['pieces'],
                                      data["probs"],
                                      v,))


        if len(iteration_newton_boards) > 0:
            boards_all = np.concatenate(iteration_newton_boards, axis=0).astype(np.int32)
            labels_all = np.concatenate(iteration_newton_labels, axis=0).astype(np.float32)

            stability_npz_path = os.path.join(
                output_dirs["stability_data"],
                f"stability_data_iter_{iteration + 1:04d}.npz"
            )
            np.savez_compressed(
                stability_npz_path,
                boards=boards_all,
                labels=labels_all,
            )
            logger.debug("Saved %d stability samples: %s", boards_all.shape[0], stability_npz_path)

        if len(accum_rewards) == 0:
            accum_rewards.append(0)

        iter_time = time.perf_counter() - iter_t0
        total_time = time.perf_counter() - total_train_wall_t0
        buffer_size = len(replay_buffer)
        buffer_batches = buffer_size // batch_size

        if len(checker_prob_buffer) > 0:
            checker_probs_all = np.concatenate(checker_prob_buffer, axis=0)
            checker_labels_all = np.concatenate(checker_label_buffer, axis=0)
            checker_audit = compute_checker_audit_stats(
                probs=checker_probs_all,
                labels=checker_labels_all,
                threshold=env.checker_threshold,
            )
            checker_log = format_checker_audit_stats(checker_audit)
        else:
            checker_audit = None
            checker_log = "checker_audit=NA"

        logger.debug(
            f"[Time] iter={iter_time:.2f}s | total={total_time:.2f}s | "
            f"mcts={timing_mcts:.2f}s | env_step={timing_env_step:.2f}s | "
            f"newton={timing_newton:.2f}s | sequence={timing_sequence:.2f}s | "
            f"buffer={buffer_size} | buffer_batches={buffer_batches} | "
            f"is_sampling={is_sampling} | "
            f"success: {num_success}/{num_done} | "
            f"{checker_log}"
        )
        if validation_fn is not None and (iteration + 1) % 10 == 0:
            validation_fn(
                policy_params=policy_params,
                policy_states=policy_states,
                iteration=iteration,
            )

        # Optimize the policy and value heads.
        phase = "Sampling" if is_sampling else "Collecting samples"
        loss_log = f"Policy loss: —, Value loss: —, Entropy: — ({phase})"
        if len(replay_buffer) >= 7 * batch_size and not is_sampling:
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy = 0.0
            train_steps = 0

            for _ in range(num_train_steps):
                batch = random.sample(replay_buffer, batch_size)
                b_b = jnp.array([x[0] for x in batch])
                b_i = jnp.array([x[1] for x in batch])
                b_pos = jnp.array([x[2] for x in batch])
                b_pc = jnp.array([x[3] for x in batch])
                b_pr = jnp.array([x[4] for x in batch])
                b_v = jnp.array([x[5] for x in batch])

                (policy_params, policy_states, opt_state,
                 p_loss, v_loss, entropy) = train_step(
                    optimizer, policy_params, policy_states, opt_state,
                    b_b, b_i, b_pos, b_pc, b_pr, b_v, loss_weights,
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


        logger.info("Iteration %d/%d | Buffer: %d", iteration + 1, num_iterations, len(replay_buffer))
        logger.info(
            "[Time] Iteration: %.2fs | Total: %.2fs",
            time.perf_counter() - iter_t0,
            time.perf_counter() - total_train_wall_t0,
        )
        logger.info(loss_log)

        if ((iteration + 1) % checkpoint_every) == 0:
            ckpt_path = output_dirs["checkpoints"] / f"policy_iter_{iteration + 1}.pkl"
            save_checkpoint(
                ckpt_path,
                policy_params,
                policy_states,
                extra={"iteration": iteration + 1}
            )
            logger.info("Saved policy checkpoint: %s", ckpt_path)

    return policy_params, policy_states
