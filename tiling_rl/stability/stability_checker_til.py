"""PyTorch stability checker and supervised training for tiling scenes.

Training paths and experiment settings are supplied by train_stability.py.
NPZ data contains boards (N, H, W), binary labels (N,), and optional hold_masks.
Zero-connected regions are relabeled as rigid pieces when loading data.
Checkpoints written here contain PyTorch state; convert them for JAX inference.
"""

import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import label
from torch.utils.data import Dataset, DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

floatType = torch.float32


def relabel_zero_components_as_pieces(board: np.ndarray) -> np.ndarray:
    """Give each four-connected zero region a distinct positive piece ID."""
    board = np.asarray(board, dtype=np.int32)
    if board.ndim != 2:
        raise ValueError(f"Expected a 2D board, got shape {board.shape}")
    out = board.copy()
    zero_mask = board == 0
    if np.any(zero_mask):
        components = label(zero_mask, connectivity=1)
        max_piece_id = int(board[board > 0].max()) if np.any(board > 0) else 0
        out[zero_mask] = max_piece_id + components[zero_mask]
    return out


def pad_board_top_left(board_np: np.ndarray, target_h: int = 16, target_w: int = 16) -> np.ndarray:
    """Pad above and to the left, keeping the board aligned to the bottom right."""
    board_np = np.asarray(board_np, dtype=np.int32)

    if board_np.ndim == 2:
        h, w = board_np.shape
        if h > target_h or w > target_w:
            raise ValueError(f"Board shape {(h, w)} exceeds target size {(target_h, target_w)}")
        out = -np.ones((target_h, target_w), dtype=np.int32)
        out[target_h - h:, target_w - w:] = board_np
        return out

    if board_np.ndim == 3:
        b, h, w = board_np.shape
        if h > target_h or w > target_w:
            raise ValueError(f"Board shape {(h, w)} exceeds target size {(target_h, target_w)}")
        out = -np.ones((b, target_h, target_w), dtype=np.int32)
        out[:, target_h - h:, target_w - w:] = board_np
        return out

    raise ValueError(f"Expected 2D or 3D board, got shape {board_np.shape}")


def pad_hold_mask_top_left(mask_np: np.ndarray, target_h: int = 16, target_w: int = 16) -> np.ndarray:
    """Pad above and to the left with zero hold values."""
    mask_np = np.asarray(mask_np, dtype=np.float32)

    if mask_np.ndim == 2:
        h, w = mask_np.shape
        if h > target_h or w > target_w:
            raise ValueError(f"Hold mask shape {(h, w)} exceeds target size {(target_h, target_w)}")
        out = np.zeros((target_h, target_w), dtype=np.float32)
        out[target_h - h:, target_w - w:] = mask_np
        return out

    if mask_np.ndim == 3:
        b, h, w = mask_np.shape
        if h > target_h or w > target_w:
            raise ValueError(f"Hold mask shape {(h, w)} exceeds target size {(target_h, target_w)}")
        out = np.zeros((b, target_h, target_w), dtype=np.float32)
        out[:, target_h - h:, target_w - w:] = mask_np
        return out

    raise ValueError(f"Expected 2D or 3D hold mask, got shape {mask_np.shape}")


def get_adjacency_map(board: torch.Tensor) -> torch.Tensor:
    """
    board: (B, H, W) int tensor
    return: (B, 4, H, W)
        channels: up, down, left, right
        1 means neighbor is different rigid piece or boundary
    """
    B, H, W = board.shape

    up_diff = (board[:, 1:, :] != board[:, :-1, :])
    down_diff = (board[:, :-1, :] != board[:, 1:, :])
    left_diff = (board[:, :, 1:] != board[:, :, :-1])
    right_diff = (board[:, :, :-1] != board[:, :, 1:])

    up_boundary = (board[:, 0, :] != -1)[:, None, :]
    up = torch.cat([up_boundary, up_diff], dim=1)

    down_boundary = (board[:, H - 1, :] != -1)[:, None, :]
    down = torch.cat([down_diff, down_boundary], dim=1)

    left_boundary = (board[:, :, 0] != -1)[:, :, None]
    left = torch.cat([left_boundary, left_diff], dim=2)

    right_boundary = (board[:, :, W - 1] != -1)[:, :, None]
    right = torch.cat([right_diff, right_boundary], dim=2)

    return torch.stack([up, down, left, right], dim=1).to(floatType)


def compute_piece_size_map(board: torch.Tensor) -> torch.Tensor:
    """
    board: (B, H, W) int tensor
    return: (B, 1, H, W)
        each rigid piece stores its normalized size on all of its cells
    """
    B, H, W = board.shape
    out = torch.zeros((B, 1, H, W), device=board.device, dtype=floatType)
    norm = float(max(H * W, 1))

    for b in range(B):
        ids = torch.unique(board[b])
        ids = ids[ids >= 0]
        for pid in ids:
            mask = (board[b] == pid)
            size = mask.sum().to(floatType) / norm
            out[b, 0][mask] = size

    return out


def compute_bottom_support_map(board: torch.Tensor) -> torch.Tensor:
    """
    board: (B, H, W) int tensor
    return: (B, 1, H, W)
    1 if a rigid cell is supported from below by:
      - bottom boundary
      - another non-blocked rigid cell directly below
    """
    B, H, W = board.shape
    occupied = (board >= 0)

    support = torch.zeros((B, H, W), device=board.device, dtype=torch.bool)
    support[:, H - 1, :] = occupied[:, H - 1, :]

    if H > 1:
        below = board[:, 1:, :]
        curr = board[:, :-1, :]
        has_support = (curr >= 0) & (below >= 0)
        support[:, :-1, :] = has_support

    return support[:, None, :, :].to(floatType)


def encode_board_features_2d(board, hold_mask=None, device="cuda", target_h=16, target_w=16):
    """Encode boards as float features [B,8,target_h,target_w].

    board and optional hold_mask accept NumPy arrays or Torch tensors
    of shape [H,W] or [B,H,W]. Board values: -1 outside, 0 unfilled,
    positive IDs real pieces. Zero-connected regions become separate rigid
    pieces and are dynamic unless explicitly marked in hold_mask.
    Channels: four adjacencies (up/down/left/right), blocked cells,
    piece size, bottom support, and hold mask.
    """
    if isinstance(board, torch.Tensor):
        board_np = board.detach().cpu().numpy().astype(np.int32)
    else:
        board_np = np.asarray(board, dtype=np.int32)

    board_np = pad_board_top_left(board_np, target_h=target_h, target_w=target_w)

    if board_np.ndim == 2:
        board_np = board_np[None, ...]

    for i, sample in enumerate(board_np):
        if np.any(sample == 0):
            board_np[i] = relabel_zero_components_as_pieces(sample)

    if hold_mask is None:
        hold_mask_np = np.zeros_like(board_np, dtype=np.float32)
    else:
        if isinstance(hold_mask, torch.Tensor):
            hold_mask_np = hold_mask.detach().cpu().numpy().astype(np.float32)
        else:
            hold_mask_np = np.asarray(hold_mask, dtype=np.float32)

        hold_mask_np = pad_hold_mask_top_left(
            hold_mask_np,
            target_h=target_h,
            target_w=target_w,
        )

        if hold_mask_np.ndim == 2:
            hold_mask_np = hold_mask_np[None, ...]

    if hold_mask_np.shape != board_np.shape:
        raise ValueError(f"hold_mask shape {hold_mask_np.shape} does not match board shape {board_np.shape}")

    hold_mask_np = np.where(board_np > 0, hold_mask_np, 0.0).astype(np.float32)

    board_t = torch.as_tensor(board_np, device=device, dtype=torch.int64)

    adjacency = get_adjacency_map(board_t)
    blocked = (board_t == -1).to(floatType).unsqueeze(1)
    piece_size = compute_piece_size_map(board_t)
    support = compute_bottom_support_map(board_t)
    hold = torch.as_tensor(hold_mask_np, device=device, dtype=floatType).unsqueeze(1)

    feats = torch.cat([adjacency, blocked, piece_size, support, hold], dim=1)
    return feats


def augment_board_features_2d(feats: torch.Tensor, labels: torch.Tensor = None):
    return feats, labels


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1 or classname.find("GroupNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class ResidualLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = max(out_channels // 4, 8)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.norm1 = nn.GroupNorm(4 if mid_channels % 4 == 0 else 1, mid_channels)
        self.norm2 = nn.GroupNorm(4 if mid_channels % 4 == 0 else 1, mid_channels)
        self.norm3 = nn.GroupNorm(8 if out_channels % 8 == 0 else 1, out_channels)

        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else None

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        h = self.norm3(self.conv3(h))
        if self.shortcut is not None:
            x = self.shortcut(x)
        return self.act(x + h)


class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class StabilityUNet2D(nn.Module):
    def __init__(
        self,
        in_channels=8,
        base_channels=64,
        scale=3,
        num_blocks=1,
        feat_channels=64,
        head_hidden=64,
        dropout=0.10,
    ):
        super().__init__()

        stem_groups = 8 if base_channels % 8 == 0 else (4 if base_channels % 4 == 0 else 1)

        self.input_proj = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.stem_norm = nn.GroupNorm(stem_groups, base_channels)
        self.act = nn.ReLU(inplace=True)

        ch = base_channels
        self.down = nn.ModuleList()
        for i in range(scale):
            out_ch = ch if i == 0 else 2 * ch
            block = nn.ModuleDict({
                "down": Downsample2D(ch, out_ch),
                "res": nn.ModuleList([ResidualLayer2D(out_ch, out_ch) for _ in range(num_blocks)]),
            })
            self.down.append(block)
            ch = out_ch

        self.mid = nn.ModuleDict({
            "block1": ResidualLayer2D(ch, ch),
            "block2": ResidualLayer2D(ch, ch),
        })

        self.up = nn.ModuleList()
        for i in range(scale):
            if i < scale - 1:
                block = nn.ModuleDict({
                    "merge": ResidualLayer2D(2 * ch, ch),
                    "res": nn.ModuleList([ResidualLayer2D(ch, ch) for _ in range(num_blocks)]),
                    "up": Upsample2D(ch, ch // 2),
                })
                self.up.append(block)
                ch = ch // 2
            else:
                block = nn.ModuleDict({
                    "merge": ResidualLayer2D(2 * ch, ch),
                    "res": nn.ModuleList([ResidualLayer2D(ch, ch) for _ in range(num_blocks)]),
                    "up": Upsample2D(ch, ch),
                })
                self.up.append(block)

        self.dropout = nn.Dropout2d(p=dropout)
        self.conv_out_feat = nn.Conv2d(ch, feat_channels, kernel_size=3, stride=1, padding=1)
        self.head = nn.Sequential(
            nn.Linear(feat_channels, head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden, 1),
        )

        self.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.act(self.stem_norm(self.input_proj(x)))

        hs = []
        for block in self.down:
            x = self.act(block["down"](x))
            for layer in block["res"]:
                x = layer(x)
            hs.append(x)

        x = self.mid["block1"](x)
        x = self.mid["block2"](x)
        x = self.dropout(x)

        for block in self.up:
            skip = hs.pop()
            x = torch.cat([x, skip], dim=1)
            x = block["merge"](x)
            for layer in block["res"]:
                x = layer(x)
            x = self.act(block["up"](x))

        feat = self.conv_out_feat(x)
        feat = feat.mean(dim=(-2, -1))
        logits = self.head(feat).squeeze(-1)
        return logits


class StabilityChecker2D(nn.Module):
    """Eight-channel PyTorch classifier returning one stability logit per board."""
    def __init__(self):
        super().__init__()
        cfg = {}

        self.model = StabilityUNet2D(
            in_channels=cfg.get("in_channels", 8),
            base_channels=cfg.get("base_channels", 64),
            scale=cfg.get("scale", 3),
            num_blocks=cfg.get("num_blocks", 1),
            feat_channels=cfg.get("feat_channels", 64),
            head_hidden=cfg.get("head_hidden", 64),
            dropout=cfg.get("dropout", 0.10),
        )

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))

    @torch.no_grad()
    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) > threshold).to(torch.float32)


def load_checkpoint_flexible_7ch_to_8ch(model, ckpt_path, device="cpu"):
    """Load compatible PyTorch weights, extending legacy seven-channel inputs with zeros."""
    ckpt = torch.load(ckpt_path, map_location=device)
    old_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    new_state = model.state_dict()

    loaded = []
    skipped = []

    for k, v in old_state.items():
        if k not in new_state:
            skipped.append(k)
            continue

        if k == "model.input_proj.weight":
            if v.ndim == 4 and new_state[k].ndim == 4 and v.shape[1] == 7 and new_state[k].shape[1] == 8:
                new_state[k][:, :7, :, :] = v
                new_state[k][:, 7:8, :, :] = 0.0
                loaded.append(k)
            elif v.shape == new_state[k].shape:
                new_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        elif v.shape == new_state[k].shape:
            new_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(new_state)
    logger.debug(f"[CKPT] loaded={len(loaded)} | skipped={len(skipped)}")
    if len(skipped) > 0:
        logger.warning("Warning: Checkpoint parameters skipped due to missing keys or incompatible shapes: %s", skipped[:10])
    return model


def _load_npz(npz_path: str):
    """Load and relabel tiling samples in memory, preserving the source NPZ."""
    with np.load(npz_path) as data:
        boards = data["boards"].astype(np.int32)
        labels = data["labels"].astype(np.float32)

        if "hold_masks" in data:
            hold_masks = data["hold_masks"].astype(np.float32)
        else:
            hold_masks = np.zeros_like(boards, dtype=np.float32)

    if boards.ndim != 3:
        raise ValueError(f"Expected boards with shape (N,H,W) in {npz_path}, got {boards.shape}")
    for i, board in enumerate(boards):
        if np.any(board == 0):
            boards[i] = relabel_zero_components_as_pieces(board)

    labels = torch.from_numpy(labels)
    return npz_path, boards, hold_masks, labels


def preload_feature_cache(items: List[Tuple[str, int]], max_workers: int = 8):
    """Cache relabeled boards, hold masks, and labels, converting each file once."""
    unique_paths = sorted({npz_path for npz_path, _ in items})
    cache = {}

    file_sample_counts = {}
    total_samples = 0
    for npz_path in unique_paths:
        with np.load(npz_path) as data:
            n = int(data["labels"].shape[0])
        file_sample_counts[npz_path] = n
        total_samples += n

    processed_samples = 0
    t0 = time.perf_counter()

    logger.debug(
        f"[CACHE] start preload boards+hold_masks+labels {len(unique_paths)} npz files | "
        f"total_samples={total_samples} | workers={max_workers}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_load_npz, p) for p in unique_paths]

        for i, fut in enumerate(as_completed(futures), start=1):
            npz_path, boards, hold_masks, labels = fut.result()
            cache[npz_path] = {
                "boards": boards,           # np.ndarray (N, H, W)
                "hold_masks": hold_masks,   # np.ndarray (N, H, W)
                "labels": labels,           # torch.Tensor (N,)
            }

            processed_samples += file_sample_counts[npz_path]

            elapsed = time.perf_counter() - t0
            logger.debug(
                "Cache progress: files=%d/%d, samples=%d/%d, elapsed=%.1fs",
                i, len(unique_paths), processed_samples, total_samples, elapsed,
            )

    total_time = time.perf_counter() - t0
    logger.debug(f"[CACHE] preload finished in {total_time:.1f}s")
    return cache


class NPZShapeDataset(Dataset):
    """Encode cached board/hold-mask samples on demand."""
    def __init__(self, items: List[Tuple[str, int]], feature_cache):
        self.items = items
        self.feature_cache = feature_cache

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npz_path, sample_idx = self.items[idx]
        board = self.feature_cache[npz_path]["boards"][sample_idx]
        hold_mask = self.feature_cache[npz_path]["hold_masks"][sample_idx]
        label = self.feature_cache[npz_path]["labels"][sample_idx]

        feats = encode_board_features_2d(
            board,
            hold_mask=hold_mask,
            device="cpu",
        ).squeeze(0).cpu()

        return feats, label


class RandomShapeBalancedBatchSampler:
    """Sample shapes with equal positive and negative counts per selected shape."""
    def __init__(
        self,
        dataset_items: List[Tuple[str, int]],
        shape_to_pos: Dict[str, List[Tuple[str, int]]],
        shape_to_neg: Dict[str, List[Tuple[str, int]]],
        shape_names: List[str],
        samples_per_shape: int,
        shapes_per_batch: int,
        batches_per_epoch: int,
    ):
        self.dataset_items = dataset_items
        self.item_to_dataset_index = {item: i for i, item in enumerate(dataset_items)}
        self.shape_to_pos = shape_to_pos
        self.shape_to_neg = shape_to_neg
        self.shape_names = list(shape_names)

        self.samples_per_shape = max(int(samples_per_shape), 2)
        self.target_half = max(self.samples_per_shape // 2, 1)

        self.shapes_per_batch = min(int(shapes_per_batch), len(self.shape_names))
        self.batches_per_epoch = max(int(batches_per_epoch), 1)

        self.shape_half = {}
        for name in self.shape_names:
            pos_n = len(self.shape_to_pos[name])
            neg_n = len(self.shape_to_neg[name])
            half = min(self.target_half, pos_n, neg_n)
            self.shape_half[name] = half

        self.valid_shape_names = [name for name in self.shape_names if self.shape_half[name] > 0]

        if len(self.valid_shape_names) == 0:
            raise ValueError("No valid shapes available for balanced batch sampling")

        self.shapes_per_batch = min(self.shapes_per_batch, len(self.valid_shape_names))

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            selected_shapes = random.sample(self.valid_shape_names, k=self.shapes_per_batch)

            batch = []
            for name in selected_shapes:
                half = self.shape_half[name]
                pos_items = random.sample(self.shape_to_pos[name], k=half)
                neg_items = random.sample(self.shape_to_neg[name], k=half)

                batch.extend(self.item_to_dataset_index[item] for item in pos_items)
                batch.extend(self.item_to_dataset_index[item] for item in neg_items)

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


def split_indices_stratified_per_shape(
    npz_paths: List[str],
    val_ratio: float = 0.2,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    train_items = []
    val_items = []

    for npz_path in npz_paths:
        with np.load(npz_path) as data:
            labels = data["labels"].astype(np.uint8)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        n_val_pos = int(round(len(pos_idx) * val_ratio))
        n_val_neg = int(round(len(neg_idx) * val_ratio))
        if len(pos_idx) > 0:
            n_val_pos = min(max(n_val_pos, 1), len(pos_idx)) if val_ratio > 0 else 0
        if len(neg_idx) > 0:
            n_val_neg = min(max(n_val_neg, 1), len(neg_idx)) if val_ratio > 0 else 0

        val_pos = pos_idx[:n_val_pos]
        train_pos = pos_idx[n_val_pos:]
        val_neg = neg_idx[:n_val_neg]
        train_neg = neg_idx[n_val_neg:]

        train_items.extend([(npz_path, int(i)) for i in train_pos])
        train_items.extend([(npz_path, int(i)) for i in train_neg])
        val_items.extend([(npz_path, int(i)) for i in val_pos])
        val_items.extend([(npz_path, int(i)) for i in val_neg])

    return train_items, val_items


def collect_npz_paths(data_dirs):
    """Collect unique NPZ files directly inside one or more data directories."""
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]

    npz_paths = []
    for d in data_dirs:
        d = Path(d)
        if not d.exists():
            raise ValueError(f"data_dir does not exist: {d}")
        found = sorted(str(p) for p in d.glob("*.npz"))
        logger.debug(f"[DATA DIR] {d} | npz={len(found)}")
        npz_paths.extend(found)

    npz_paths = sorted(set(npz_paths))
    if len(npz_paths) == 0:
        raise ValueError(f"No npz files found in data_dirs={data_dirs}")

    return npz_paths


def shape_key_from_path(npz_path: str):
    p = Path(npz_path)
    return f"{p.parent.name}/{p.stem}"


def build_shape_index(items: List[Tuple[str, int]]):
    shape_to_pos = {}
    shape_to_neg = {}
    label_cache = {}

    for npz_path, idx in items:
        key = shape_key_from_path(npz_path)
        if key not in shape_to_pos:
            shape_to_pos[key] = []
            shape_to_neg[key] = []
        if npz_path not in label_cache:
            with np.load(npz_path) as data:
                label_cache[npz_path] = data["labels"].astype(np.uint8)
        y = int(label_cache[npz_path][idx])
        if y == 1:
            shape_to_pos[key].append((npz_path, idx))
        else:
            shape_to_neg[key].append((npz_path, idx))

    valid_shapes = [k for k in shape_to_pos.keys() if len(shape_to_pos[k]) > 0 and len(shape_to_neg[k]) > 0]
    return shape_to_pos, shape_to_neg, sorted(valid_shapes)


def compute_binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).to(torch.float32)
    labels = labels.to(torch.float32)

    acc = (preds == labels).float().mean().item() if labels.numel() > 0 else 0.0

    pos_mask = (labels == 1)
    neg_mask = (labels == 0)
    pos_acc = (preds[pos_mask] == labels[pos_mask]).float().mean().item() if pos_mask.any() else 0.0
    neg_acc = (preds[neg_mask] == labels[neg_mask]).float().mean().item() if neg_mask.any() else 0.0
    bal_acc = 0.5 * (pos_acc + neg_acc)

    return {
        "acc": acc,
        "stable_acc": pos_acc,
        "unstable_acc": neg_acc,
        "balanced_acc": bal_acc,
    }


def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_n = 0
    logits_all = []
    labels_all = []

    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(feats)
            loss = criterion(logits, labels)

            bs = labels.shape[0]
            total_loss += loss.item() * bs
            total_n += bs
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

    if total_n == 0:
        return {"loss": 0.0, "acc": 0.0, "stable_acc": 0.0, "unstable_acc": 0.0, "balanced_acc": 0.0}

    logits_all = torch.cat(logits_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    metrics = compute_binary_metrics(logits_all, labels_all)
    metrics["loss"] = total_loss / total_n
    return metrics


def make_val_loader(val_dataset, train_shapes, samples_per_shape_per_batch, device, num_workers, val_eval_fraction, seed, epoch):
    val_count = len(val_dataset)
    if val_count == 0:
        return None, 0

    eval_count = max(1, int(round(val_count * val_eval_fraction)))
    eval_count = min(eval_count, val_count)

    rng = np.random.default_rng(seed + epoch)
    subset_indices = rng.choice(val_count, size=eval_count, replace=False)
    subset_indices = subset_indices.tolist()

    val_subset = Subset(val_dataset, subset_indices)
    val_batch_size = max(len(train_shapes) * samples_per_shape_per_batch, 32)
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )
    return val_loader, eval_count


def train_stability_checker(
    data_dir,
    save_dir,
    device="cuda",
    epochs=20,
    lr=1e-3,
    weight_decay=1e-4,
    val_ratio=0.2,
    samples_per_shape_per_batch=8,
    shapes_per_batch=64,
    batches_per_epoch=20,
    num_workers=4,
    seed=42,
    validate_every=10,
    val_eval_fraction=0.2,
    init_ckpt_path=None,
):
    """Train from explicit NPZ directories and save PyTorch checkpoints.

    init_ckpt_path initializes compatible model weights only; optimizer and epoch
    state are initialized anew. num_workers controls the cache-loading threads.
    Returns the trained model and per-epoch metrics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = collect_npz_paths(data_dir)

    total = 0
    hold_stable = 0
    hold_unstable = 0
    no_hold_stable = 0
    no_hold_unstable = 0

    for p in npz_paths:
        with np.load(p) as data:
            labels = data["labels"].astype(np.uint8).reshape(-1)

            if "hold_masks" in data:
                hold_masks = data["hold_masks"]
                has_hold = np.any(hold_masks > 0, axis=(1, 2))
            else:
                has_hold = np.zeros(labels.shape[0], dtype=bool)

        stable = labels == 1
        unstable = labels == 0

        total += int(labels.shape[0])
        hold_stable += int(np.sum(has_hold & stable))
        hold_unstable += int(np.sum(has_hold & unstable))
        no_hold_stable += int(np.sum((~has_hold) & stable))
        no_hold_unstable += int(np.sum((~has_hold) & unstable))

    hold_total = hold_stable + hold_unstable
    no_hold_total = no_hold_stable + no_hold_unstable

    logger.debug(
        f"[DATA DIST] total={total} | "
        f"hold={hold_total} stable={hold_stable} unstable={hold_unstable} | "
        f"no_hold={no_hold_total} stable={no_hold_stable} unstable={no_hold_unstable}"
    )

    train_items, val_items = split_indices_stratified_per_shape(npz_paths, val_ratio=val_ratio, seed=seed)
    train_pos, train_neg, train_shapes = build_shape_index(train_items)
    if len(train_shapes) == 0:
        raise ValueError("No valid shapes with both stable and unstable samples in training split")
    feature_cache = preload_feature_cache(train_items + val_items, max_workers=num_workers)
    train_dataset = NPZShapeDataset(train_items, feature_cache=feature_cache)
    val_dataset = NPZShapeDataset(val_items, feature_cache=feature_cache)

    train_batch_sampler = RandomShapeBalancedBatchSampler(
        dataset_items=train_items,
        shape_to_pos=train_pos,
        shape_to_neg=train_neg,
        shape_names=train_shapes,
        samples_per_shape=samples_per_shape_per_batch,
        shapes_per_batch=shapes_per_batch,
        batches_per_epoch=batches_per_epoch,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    model = StabilityChecker2D().to(device)
    if init_ckpt_path is not None:
        model = load_checkpoint_flexible_7ch_to_8ch(model, init_ckpt_path, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    min_lr = 1e-6
    criterion = nn.BCEWithLogitsLoss()

    init_eval_items = train_items + val_items

    init_pos_items = []
    init_neg_items = []
    label_cache = {}

    for npz_path, idx in init_eval_items:
        if npz_path not in label_cache:
            with np.load(npz_path) as data:
                label_cache[npz_path] = data["labels"].astype(np.uint8)

        y = int(label_cache[npz_path][idx])
        if y == 1:
            init_pos_items.append((npz_path, idx))
        else:
            init_neg_items.append((npz_path, idx))

    init_eval_count = max(2, int(round(len(init_eval_items) * 0.10)))
    init_half = min(init_eval_count // 2, len(init_pos_items), len(init_neg_items))

    rng = np.random.default_rng(seed)
    init_eval_sampled_items = (
        [init_pos_items[i] for i in rng.choice(len(init_pos_items), size=init_half, replace=False)]
        +
        [init_neg_items[i] for i in rng.choice(len(init_neg_items), size=init_half, replace=False)]
    )
    rng.shuffle(init_eval_sampled_items)

    init_eval_dataset = NPZShapeDataset(init_eval_sampled_items, feature_cache=feature_cache)
    init_eval_loader = DataLoader(
        init_eval_dataset,
        batch_size=max(shapes_per_batch * samples_per_shape_per_batch, 32),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    init_eval_count = len(init_eval_sampled_items)

    init_metrics = evaluate_model(model, init_eval_loader, device, criterion)
    logger.debug(
        f"[INIT EVAL 10%] n={init_eval_count} | "
        f"loss={init_metrics['loss']:.4f} | "
        f"bal_acc={init_metrics['balanced_acc']:.4f} | "
        f"stable={init_metrics['stable_acc']:.4f} | "
        f"unstable={init_metrics['unstable_acc']:.4f}"
    )

    best_val = -1.0
    history = []

    max_batch_size = train_batch_sampler.shapes_per_batch * train_batch_sampler.samples_per_shape
    logger.info("Data: %d training samples | %d validation samples", len(train_items), len(val_items))
    logger.info("Output: %s", save_dir)
    logger.debug("Split fractions: train=%g, val=%g", 1.0 - val_ratio, val_ratio)
    logger.debug(
        f"[BATCH] samples_per_shape_per_batch={samples_per_shape_per_batch} | "
        f"shapes_per_batch={train_batch_sampler.shapes_per_batch} | "
        f"batches_per_epoch={len(train_batch_sampler)} | "
        f"max_batch_size={max_batch_size}"
    )
    logger.debug(f"[VAL] validate_every={validate_every} | val_eval_fraction={val_eval_fraction}")

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        logits_all = []
        labels_all = []

        for feats, labels in train_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.shape[0]
            train_loss_sum += loss.item() * bs
            train_n += bs
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())

        train_logits = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
        train_labels = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0)
        train_metrics = compute_binary_metrics(train_logits, train_labels) if train_n > 0 else {
            "acc": 0.0,
            "stable_acc": 0.0,
            "unstable_acc": 0.0,
            "balanced_acc": 0.0,
        }
        train_metrics["loss"] = train_loss_sum / max(train_n, 1)

        do_validate = (epoch % validate_every == 0) or (epoch == epochs)
        val_eval_count = 0
        if do_validate:
            val_loader, val_eval_count = make_val_loader(
                val_dataset=val_dataset,
                train_shapes=train_shapes,
                samples_per_shape_per_batch=samples_per_shape_per_batch,
                device=device,
                num_workers=0,
                val_eval_fraction=val_eval_fraction,
                seed=seed,
                epoch=epoch,
            )
            if val_loader is None:
                val_metrics = {"loss": 0.0, "acc": 0.0, "stable_acc": 0.0, "unstable_acc": 0.0, "balanced_acc": 0.0}
            else:
                val_metrics = evaluate_model(model, val_loader, device, criterion)
        else:
            val_metrics = None

        epoch_time = time.perf_counter() - epoch_t0

        history.append({
            "epoch": epoch,
            "epoch_time": epoch_time,
            "train": train_metrics,
            "val": val_metrics,
            "validated": do_validate,
            "val_eval_count": val_eval_count,
        })
        current_lr = optimizer.param_groups[0]["lr"]

        if do_validate:
            logger.info(
                f"Epoch {epoch}/{epochs} | Loss: {train_metrics['loss']:.4f} | "
                f"Balanced accuracy: {train_metrics['balanced_acc']:.1%} | "
                f"Val loss: {val_metrics['loss']:.4f} | "
                f"Val balanced accuracy: {val_metrics['balanced_acc']:.1%} | "
                f"Time: {epoch_time:.2f}s"
            )
        else:
            logger.info(
                f"Epoch {epoch}/{epochs} | Loss: {train_metrics['loss']:.4f} | "
                f"Balanced accuracy: {train_metrics['balanced_acc']:.1%} | "
                f"Time: {epoch_time:.2f}s"
            )

        epoch_path = save_dir / f"stability_checker_epoch_{epoch:03d}.pkl"
        save_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_balanced_acc": best_val,
        }
        torch.save(save_payload, epoch_path)
        logger.debug("Saved epoch checkpoint: %s", epoch_path)

        if do_validate and val_metrics["balanced_acc"] > best_val:
            best_val = val_metrics["balanced_acc"]
            best_path = save_dir / "stability_checker_best.pkl"
            save_payload["best_val_balanced_acc"] = best_val
            torch.save(save_payload, best_path)
            logger.info("Saved best checkpoint: %s", best_path)

        scheduler.step()
        for param_group in optimizer.param_groups:
            if param_group["lr"] < min_lr:
                param_group["lr"] = min_lr

    return model, history
