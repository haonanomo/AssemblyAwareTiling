"""Train the assembly-aware tiling policy on a directory of PNG masks."""

import argparse
import logging
import os
from functools import partial
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mask_dir', type=Path, default=PROJECT_DIR / 'dataset/png',
                        help='default: %(default)s')
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '42')), help='default: %(default)s')
    parser.add_argument('--out_dir', type=Path, default=os.environ.get('RESULT_DIR'),
                        help=f"default: {os.environ.get('RESULT_DIR') or PROJECT_DIR / 'results' / os.environ.get('RUN_NAME', 'train_tiling_seed<seed>')}")
    parser.add_argument('--checker_ckpt_path', type=Path,
                        default=PROJECT_DIR / 'checkpoints/stability_checker_jax_til.pkl', help='default: %(default)s')
    parser.add_argument('--seq_policy_ckpt_path', type=Path, default=None,
                        help='default: %(default)s')
    parser.add_argument('--init_checkpoint_path', type=Path, default=None,
                        help='default: %(default)s')
    parser.add_argument('--num_iterations', type=int, default=1000, help='default: %(default)s')
    parser.add_argument('--shapes_per_iter', type=int, default=64, help='default: %(default)s')
    parser.add_argument('--envs_per_shape', type=int, default=1, help='default: %(default)s')
    parser.add_argument('--batch_size', type=int, default=1024, help='default: %(default)s')
    parser.add_argument('--num_train_steps', type=int, default=15, help='default: %(default)s')
    parser.add_argument('--checkpoint_every', type=int, default=5, help='default: %(default)s')
    args = parser.parse_args(argv)
    for name in ('num_iterations', 'shapes_per_iter', 'envs_per_shape', 'batch_size',
                 'num_train_steps', 'checkpoint_every'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    if not args.mask_dir.is_dir():
        parser.error(f'PNG mask directory does not exist: {args.mask_dir}')
    for name in ('checker_ckpt_path', 'seq_policy_ckpt_path', 'init_checkpoint_path'):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f'--{name} file does not exist: {path}')
    if args.out_dir is None:
        run_name = os.environ.get('RUN_NAME', f'train_tiling_seed{args.seed}')
        args.out_dir = PROJECT_DIR / 'results' / run_name
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    import jax
    import jax.numpy as jnp
    import numpy as np

    from tiling_rl.common.masks import load_masks_from_png_dir
    from tiling_rl.common.pieces import generate_polyominoes_from_list
    from tiling_rl.tiling.env_til import (
        TetrixEnv,
        build_multi_board_batch_shared_inventory,
        canonicalize_blocked_mask,
    )
    from tiling_rl.tiling.mcts_til import (
        load_seq_checkpoint,
        p_model,
        train_alphazero,
        validate_policy_only,
    )

    board_size = (16, 16)
    shape_specs = load_masks_from_png_dir(args.mask_dir)
    if len(shape_specs) < args.shapes_per_iter:
        raise ValueError(
            f'Found {len(shape_specs)} PNG shapes, but shapes_per_iter={args.shapes_per_iter}. '
            'Set --shapes_per_iter to at most the number of PNGs.'
        )
    for spec in shape_specs:
        try:
            canonicalize_blocked_mask(spec['mask'], board_size)
        except ValueError as exc:
            raise ValueError(f"Invalid PNG mask {spec['name']}: {exc}") from exc
    logger.info('Training: Tiling')
    logger.info('Data: %d shapes | Environments: %d | Seed: %d',
                len(shape_specs), args.shapes_per_iter * args.envs_per_shape, args.seed)
    logger.info('Device: %s', ', '.join(str(device) for device in jax.devices()))
    logger.debug('JAX devices: %s', jax.devices())

    pieces, pieces_id = generate_polyominoes_from_list([3, 4, 5], grid_size=4)
    inventory = 10 * jnp.ones(pieces.shape[0], dtype=jnp.int32)
    num_parallel_envs = args.shapes_per_iter * args.envs_per_shape

    def batch_reset_fn(iteration):
        batch_info = build_multi_board_batch_shared_inventory(
            shape_specs=shape_specs,
            board_size=board_size,
            shapes_per_iter=args.shapes_per_iter,
            envs_per_shape=args.envs_per_shape,
            rng=np.random.default_rng(seed=args.seed + iteration),
        )
        batch_info.update(pieces=pieces, pieces_id=pieces_id, inventory=inventory)
        return batch_info

    seq_params, seq_states = None, None
    if args.seq_policy_ckpt_path is not None:
        seq_params, seq_states, _ = load_seq_checkpoint(args.seq_policy_ckpt_path)
        logger.debug('Loaded sequence policy: %s', args.seq_policy_ckpt_path)
    else:
        logger.debug('Sequence policy disabled')
    env = TetrixEnv(
        board_size=board_size,
        checker_ckpt_path=args.checker_ckpt_path,
        seq_policy_params=seq_params,
        seq_policy_states=seq_states,
    )
    init_batch = batch_reset_fn(0)
    env.reset_batch(
        pieces=init_batch['pieces'],
        pieces_id=init_batch['pieces_id'],
        inventory=init_batch['inventory'],
        blocked_mask=init_batch['blocked_mask'],
    )
    validation_fn = partial(
        validate_policy_only,
        shape_specs=shape_specs,
        board_size=board_size,
        inventory=inventory,
        pieces=pieces,
        pieces_id=pieces_id,
        result_save_folder=args.out_dir,
        p_model=p_model,
    )
    max_valid_cells = max(int(np.sum(np.asarray(spec['mask']) == 0)) for spec in shape_specs)
    result = train_alphazero(
        env=env,
        p_model=p_model,
        result_save_folder=args.out_dir,
        seq_policy_params=seq_params,
        seq_policy_states=seq_states,
        num_iterations=args.num_iterations,
        num_parallel_envs=num_parallel_envs,
        step_per_iterations=max_valid_cells // 3 + 4,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        batch_reset_fn=batch_reset_fn,
        checkpoint_every=args.checkpoint_every,
        init_checkpoint_path=args.init_checkpoint_path,
        seed=args.seed,
        validation_fn=validation_fn,
    )
    logger.debug('Training complete: %s', args.out_dir)
    return result


if __name__ == '__main__':
    main()
