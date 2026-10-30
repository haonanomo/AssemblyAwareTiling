# AssemblyAwareTiling

Implementation of the following paper:

Assembly-aware Tiling for Standing Polyominoes (SIGGRAPH Asia 2026)

## Installation

- **Platform:** Linux or Windows with WSL2 for GPU acceleration. On native Windows, JAX supports CPU only.
- **Python:** 3.11.
- **Libraries:** JAX, Haiku, Optax, MCTX, PyTorch, Newton, Warp, and cuRobo.

Run installation and all commands below from the project directory.

```bash
pip install -e .
```

## Data Terms

- **Tiled boards:** Cell-level piece IDs representing full or partial tilings.
- **Piece states:** Piece-level status: 0 = removed, 1 = placed, 2 = held.
- **Hold masks:** Cell-level binary masks: 1 = held pieces or virtual support; 0 = all other cells.

## Tiling Policy

### Training

- **Input:** 16x16 PNG masks, a JAX tiling stability checker, and an optional sequence policy. With the sequence policy, training is assembly-aware; without it, training is stability-aware.
- **Output:** Policy checkpoints in `checkpoints/`, tiled boards in `boards/`, PNG previews in `previews/`, stability training data in `stability_data/`, and validation results in `validation/`.

```bash
python train_til.py
```

### Inference

- **Input:** PNG masks, a tiling policy, a JAX tiling stability checker, and an optional sequence policy.
- **Output:** All trial boards in `boards/`, PNG previews in `previews/`, best-per-shape results in `best/`, and collapse diagnostics in `failures/`. Trial filenames include the shape and trial index.

```bash
python infer_til.py
```

## Sequence Policy

IK tables contain grasp candidates and collision data, stored as NPZ files. Pass a Tiling output's `boards/` directory to `--board_dir`. Generate the tables separately before training or inference:

```bash
python -m tiling_rl.sequence.ik_batched precompute_dir --board_dir <board_dir>
```

Tables are saved in `<board_dir>/ik_tables/` and loaded automatically by Sequence. Legacy tables beside the boards are also supported. Batch errors are recorded in `ik_precompute_log.json` and produce a nonzero exit status after all files are processed.

### Training

- **Input:** Tiled boards, IK tables, and a JAX sequence stability checker.
- **Output:** Sequence policy checkpoints in `checkpoints/` and stability training data in `stability_data/`.

```bash
python train_seq.py --board_dir <board_dir>
```

### Inference

- **Input:** Tiled boards, IK tables, and a sequence policy; MCTS also requires a JAX sequence stability checker.
- **Output:** Tiled boards in `boards/`, piece states in `piece_states/`, and PNG previews in `previews/`, with matching filenames. Best results and collapse diagnostics use `best/` and `failures/`.

```bash
python infer_seq.py --board_dir <board_dir>
```

## Stability Checker

Checker training uses PyTorch; MCTS in both Tiling and Sequence uses JAX. Convert PyTorch checkpoints to JAX with `convert_checker_til.py` or `convert_checker_seq.py` in the stability module.

### Training

- **Input:** Tiled boards, stability labels, and optional hold masks, stored together in NPZ files. Use the training output's `stability_data/` directory for `--data_dir`.
- **Output:** PyTorch stability checker checkpoints.

Tiling checker training automatically relabels each four-connected zero region as a separate rigid piece. Source NPZ files stay unchanged; Sequence keeps zero regions as virtual support.

```bash
python train_stability.py --task tiling --data_dir <data_dir>
python train_stability.py --task seq --data_dir <data_dir>
```

### Inference

- **Input:** Tiled boards, a JAX stability checker, and optional hold masks.
- **Output:** Stability predictions.
