# AssemblyAwareTiling

Implementation of the following paper:

Assembly-aware Tiling for Standing Polyominoes (SIGGRAPH Asia 2026)

## Installation

- **Platform:** Linux or Windows with WSL2 for GPU acceleration. On native Windows, JAX supports CPU only.
- **Python:** >=3.11.

```bash
pip install -e .
```

## Data Terms

- **Tiled boards:** Cell-level piece IDs representing full or partial tilings.
- **Piece states:** Piece-level status: 0 = removed, 1 = placed, 2 = held.
- **Hold masks:** Cell-level binary masks: 1 = held pieces or virtual support; 0 = all other cells.

## Tiling Policy

### Training

- **Input:** PNG masks, a JAX tiling stability checker, and an optional sequence policy.
- **Output:** Tiling policy checkpoints, tiled boards, and stability training data.

```bash
python train_til.py
```

### Inference

- **Input:** PNG masks, a tiling policy, a JAX tiling stability checker, and an optional sequence policy.
- **Output:** Tiled boards.

```bash
python infer_til.py
```

## Sequence Policy

IK tables contain grasp candidates and collision data. Generate them separately from tiled boards before training or inference:

```bash
python -m tiling_rl.sequence.ik_batched precompute_dir --board_dir <board_dir>
```

### Training

- **Input:** Tiled boards, IK tables, and a JAX sequence stability checker.
- **Output:** Sequence policy checkpoints and stability training data.

```bash
python train_seq.py --board_dir <board_dir>
```

### Inference

- **Input:** Tiled boards, IK tables, a sequence policy, and a JAX sequence stability checker.
- **Output:** Tiled boards and piece states.

```bash
python infer_seq.py --board_dir <board_dir>
```

## Stability Checker

Checker training uses PyTorch; MCTS in both tiling and sequence uses JAX. Convert PyTorch checkpoints to JAX with `convert_checker_til.py` or `convert_checker_seq.py` in the stability module. Note that untiled regions are treated as connected rigid pieces for tiling and as virtual support for sequence.

### Training

- **Input:** Tiled boards, stability labels, and optional hold masks in NPZ files.
- **Output:** PyTorch stability checker checkpoints.

```bash
python train_stability.py --task tiling --data_dir <data_dir>
python train_stability.py --task seq --data_dir <data_dir>
```

### Inference

- **Input:** Tiled boards, a JAX stability checker, and optional hold masks.
- **Output:** Stability predictions.
