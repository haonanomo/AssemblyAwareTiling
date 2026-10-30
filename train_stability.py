"""Launch stability-checker training for tiling or sequence tasks."""

import argparse
import importlib
import logging
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--task", choices=("tiling", "seq"), default="tiling",
        help='default: %(default)s',
    )
    parser.add_argument("--data_dir", type=Path, nargs="+", required=True,
                        help='required')
    parser.add_argument("--save_dir", "--out_dir", type=Path,
                        help='default: results/stability_{til,seq}')
    parser.add_argument("--init_ckpt_path", "--init_checkpoint_path", type=Path,
                        help='default: %(default)s')
    parser.add_argument("--device", default="auto", help='default: %(default)s')
    parser.add_argument("--epochs", type=int, default=1000, help='default: %(default)s')
    parser.add_argument("--lr", type=float, help='default: tiling=3e-4, seq=2e-4')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='default: %(default)s')
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help='default: %(default)s')
    parser.add_argument("--samples_per_shape_per_batch", type=int, default=6,
                        help='default: %(default)s')
    parser.add_argument("--shapes_per_batch", type=int, default=200, help='default: %(default)s')
    parser.add_argument("--batches_per_epoch", type=int, default=20, help='default: %(default)s')
    parser.add_argument("--num_workers", type=int, default=8,
                        help='default: %(default)s')
    parser.add_argument("--seed", type=int, default=42, help='default: %(default)s')
    parser.add_argument("--validate_every", type=int, default=10,
                        help='default: %(default)s')
    parser.add_argument("--val_eval_fraction", type=float, default=0.2,
                        help='default: %(default)s')
    parser.add_argument("--log_level", choices=("DEBUG", "INFO", "WARNING"), default="INFO",
                        help='default: %(default)s')
    args = parser.parse_args(argv)
    suffix = "til" if args.task == "tiling" else "seq"
    if args.save_dir is None:
        args.save_dir = PROJECT_DIR / "results" / f"stability_{suffix}"
    if args.lr is None:
        args.lr = 3e-4 if args.task == "tiling" else 2e-4
    for directory in args.data_dir:
        if not directory.is_dir():
            parser.error(f"Data directory does not exist: {directory}")
    if args.init_ckpt_path is not None and not args.init_ckpt_path.is_file():
        parser.error(f"Initial checkpoint does not exist: {args.init_ckpt_path}")
    for name in ("epochs", "shapes_per_batch", "batches_per_epoch", "num_workers", "validate_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.samples_per_shape_per_batch < 2:
        parser.error("--samples_per_shape_per_batch must be at least 2")
    if not 0 <= args.val_ratio < 1:
        parser.error("--val_ratio must be in [0, 1)")
    if not 0 < args.val_eval_fraction <= 1:
        parser.error("--val_eval_fraction must be in (0, 1]")
    if not args.lr > 0 or not args.weight_decay >= 0:
        parser.error("--lr must be positive and --weight_decay must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(message)s")
    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    modules = {
        "tiling": "tiling_rl.stability.stability_checker_til",
        "seq": "tiling_rl.stability.stability_checker_seq",
    }
    trainer = importlib.import_module(modules[args.task])
    logging.getLogger(__name__).info(
        "Training: Stability | Task: %s | Device: %s",
        "Tiling" if args.task == "tiling" else "Sequence", args.device.upper(),
    )
    options = vars(args).copy()
    options.pop("task")
    options.pop("log_level")
    return trainer.train_stability_checker(**options)


if __name__ == "__main__":
    main()
