"""Train a sequence policy from tiled NPY boards and matching IK tables."""

import argparse
import logging
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board_dir', type=Path,
                        default=os.environ.get('BOARD_DIR', str(PROJECT_DIR / 'dataset/sequence/train')),
                        help='default: %(default)s')
    parser.add_argument('--out_dir', type=Path, default=os.environ.get('RESULT_DIR'),
                        help=f"default: {os.environ.get('RESULT_DIR') or PROJECT_DIR / 'results' / os.environ.get('RUN_NAME', 'train_sequence_seed<seed>')}")
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '42')))
    parser.add_argument('--checker_ckpt_path', type=Path,
                        default=PROJECT_DIR / 'checkpoints/stability_checker_jax_seq.pkl',
                        help='default: %(default)s')
    parser.add_argument('--init_checkpoint_path', type=Path,
                        help='default: %(default)s')
    parser.add_argument('--num_parallel_envs', type=int,
                        default=int(os.environ.get('NUM_PARALLEL_ENVS', '96')))
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_train_steps', type=int, default=10)
    parser.add_argument('--checkpoint_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=5,
                        help='default: %(default)s')
    parser.add_argument('--eval_mode', choices=('policy_only', 'mcts'), default='policy_only')
    args = parser.parse_args(argv)
    for name in ('num_parallel_envs', 'num_iterations', 'batch_size',
                 'num_train_steps', 'checkpoint_every'):
        if getattr(args, name) < 1:
            parser.error(f'--{name} must be positive')
    if args.eval_every < 0:
        parser.error('--eval_every must be nonnegative')
    if not args.board_dir.is_dir():
        parser.error(f'Board directory does not exist: {args.board_dir}')
    for name in ('checker_ckpt_path', 'init_checkpoint_path'):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f'--{name} file does not exist: {path}')
    if args.out_dir is None:
        run_name = os.environ.get('RUN_NAME', f'train_sequence_seed{args.seed}')
        args.out_dir = PROJECT_DIR / 'results' / run_name
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    import jax
    import numpy as np

    from tiling_rl.sequence.env_seq import SequenceEnv
    from tiling_rl.sequence.mcts_seq import (
        compute_global_ik_padding,
        compute_global_step_per_iterations,
        p_model,
        train_alphazero_seq,
    )

    from tiling_rl.sequence.ik_lookup import resolve_ik_table_path

    all_files = sorted(args.board_dir.glob('*.npy'))
    if not all_files:
        raise ValueError(f'No NPY boards found in {args.board_dir}')
    npy_files = [p for p in all_files if resolve_ik_table_path(p).is_file()]
    missing = len(all_files) - len(npy_files)
    if missing:
        logger.warning('Warning: Skipping %d boards without matching IK tables in %s', missing, args.board_dir)
    if not npy_files:
        raise ValueError(f'No board/IK-table pairs found in {args.board_dir}')
    logger.info('Training: Sequence')
    logger.info('Data: %d boards | IK tables: Loaded | Environments: %d | Seed: %d',
                len(npy_files), args.num_parallel_envs, args.seed)
    logger.info('Device: %s', ', '.join(str(device) for device in jax.devices()))

    max_samples, max_r1, max_r2 = compute_global_ik_padding(npy_files)
    step_per_iterations, max_parts = compute_global_step_per_iterations(npy_files)
    board_cache = {p: np.load(p, allow_pickle=False).astype(np.int32) for p in npy_files}
    shapes = {board.shape for board in board_cache.values()}
    if len(shapes) != 1 or any(len(shape) != 2 for shape in shapes):
        raise ValueError('Sequence training requires 2D boards of the same shape')
    if any(not np.any(board > 0) for board in board_cache.values()):
        raise ValueError('Each sequence board must contain at least one real piece')
    logger.debug('Padding: parts=%d, IK samples=%d, robot1=%d, robot2=%d; steps=%d',
                max_parts, max_samples, max_r1, max_r2, step_per_iterations)

    def batch_reset_fn(iteration):
        rng = np.random.default_rng(seed=args.seed + iteration)
        ids = rng.choice(len(npy_files), size=args.num_parallel_envs,
                         replace=len(npy_files) < args.num_parallel_envs)
        paths = [npy_files[i] for i in ids]
        return {
            'boards': [board_cache[p] for p in paths],
            'board_paths': paths,
            'shape_names': [f'{p.stem}_env{j}' for j, p in enumerate(paths)],
        }

    env = SequenceEnv(
        n_robot=2,
        max_shift=10,
        checker_ckpt_path=args.checker_ckpt_path,
        fixed_max_parts=max_parts,
        fixed_max_ik_samples=max_samples,
        fixed_max_ik_r1=max_r1,
        fixed_max_ik_r2=max_r2,
    )
    initial = batch_reset_fn(0)
    env.reset_batch(boards=initial['boards'], board_paths=initial['board_paths'])
    result = train_alphazero_seq(
        env=env,
        p_model=p_model,
        result_save_folder=args.out_dir,
        num_iterations=args.num_iterations,
        num_parallel_envs=args.num_parallel_envs,
        step_per_iterations=step_per_iterations,
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        batch_reset_fn=batch_reset_fn,
        checkpoint_every=args.checkpoint_every,
        init_checkpoint_path=args.init_checkpoint_path,
        eval_files=npy_files,
        eval_every=args.eval_every,
        eval_mode=args.eval_mode,
        seed=args.seed,
        board_cache=board_cache,
    )
    logger.debug('Sequence training complete: %s', args.out_dir)
    return result


if __name__ == '__main__':
    main()
