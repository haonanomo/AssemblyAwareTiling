"""JAX stability features and inference network for sequence scenes.

Callers supply converted checker parameters. This module performs no checkpoint
loading or training; batched MCTS helpers operate on already-sized boards.
"""

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk


float_type = jnp.float32


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


def get_adjacency_map_jax(board: jnp.ndarray) -> jnp.ndarray:
    """
    board: (B, H, W) int32
    return: (B, H, W, 4) float32
    channels: up, down, left, right
    """
    b, h, w = board.shape

    up_diff = board[:, 1:, :] != board[:, :-1, :]
    down_diff = board[:, :-1, :] != board[:, 1:, :]
    left_diff = board[:, :, 1:] != board[:, :, :-1]
    right_diff = board[:, :, :-1] != board[:, :, 1:]

    up_boundary = (board[:, 0, :] != -1)[:, None, :]
    up = jnp.concatenate([up_boundary, up_diff], axis=1)

    down_boundary = (board[:, h - 1, :] != -1)[:, None, :]
    down = jnp.concatenate([down_diff, down_boundary], axis=1)

    left_boundary = (board[:, :, 0] != -1)[:, :, None]
    left = jnp.concatenate([left_boundary, left_diff], axis=2)

    right_boundary = (board[:, :, w - 1] != -1)[:, :, None]
    right = jnp.concatenate([right_diff, right_boundary], axis=2)

    out = jnp.stack([up, down, left, right], axis=-1)
    return out.astype(float_type)


def compute_piece_size_map_jax(board: jnp.ndarray) -> jnp.ndarray:
    """
    board: (B, H, W) int32
    return: (B, H, W, 1)
    """
    board_np = np.asarray(board, dtype=np.int32)
    b, h, w = board_np.shape
    out = np.zeros((b, h, w, 1), dtype=np.float32)
    norm = float(max(h * w, 1))

    for bi in range(b):
        ids = np.unique(board_np[bi])
        ids = ids[ids >= 0]
        for pid in ids:
            mask = board_np[bi] == pid
            size = float(mask.sum()) / norm
            out[bi, mask, 0] = size

    return jnp.asarray(out, dtype=float_type)


def compute_bottom_support_map_jax(board: jnp.ndarray) -> jnp.ndarray:
    """
    board: (B, H, W) int32
    return: (B, H, W, 1)
    """
    b, h, w = board.shape
    occupied = board >= 0

    support = jnp.zeros((b, h, w), dtype=jnp.bool_)
    support = support.at[:, h - 1, :].set(occupied[:, h - 1, :])

    if h > 1:
        below = board[:, 1:, :]
        curr = board[:, :-1, :]
        has_support = (curr >= 0) & (below >= 0)
        support = support.at[:, :-1, :].set(has_support)

    return support[..., None].astype(float_type)


def encode_board_features_2d_jax(board, hold_mask=None, target_h=16, target_w=16):
    """
    input:
        board: np.ndarray or jnp.ndarray, shape (H,W) or (B,H,W)
        hold_mask: None, np.ndarray, or jnp.ndarray, shape (H,W) or (B,H,W)
    output:
        feats: (B, H, W, 8) float32, NHWC
    """
    if isinstance(board, jnp.ndarray):
        board_np = np.asarray(board, dtype=np.int32)
    else:
        board_np = np.asarray(board, dtype=np.int32)

    board_np = pad_board_top_left(board_np, target_h=target_h, target_w=target_w)
    if board_np.ndim == 2:
        board_np = board_np[None, ...]

    if hold_mask is None:
        hold_mask_np = np.zeros_like(board_np, dtype=np.float32)
    else:
        if isinstance(hold_mask, jnp.ndarray):
            hold_mask_np = np.asarray(hold_mask, dtype=np.float32)
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

    virtual_hold = (board_np == 0).astype(np.float32)
    hold_mask_np = np.maximum(hold_mask_np, virtual_hold)
    hold_mask_np = np.where(board_np >= 0, hold_mask_np, 0.0).astype(np.float32)

    board_j = jnp.asarray(board_np, dtype=jnp.int32)
    hold_j = jnp.asarray(hold_mask_np, dtype=float_type)[..., None]

    adjacency = get_adjacency_map_jax(board_j)
    blocked = (board_j == -1)[..., None].astype(float_type)
    piece_size = compute_piece_size_map_jax(board_j)
    support = compute_bottom_support_map_jax(board_j)

    feats = jnp.concatenate([adjacency, blocked, piece_size, support, hold_j], axis=-1)
    return feats


def group_norm_nhwc(x, groups, name=None, eps=1e-5):
    """
    x: NHWC
    Match PyTorch nn.GroupNorm(num_groups, num_channels) behavior as closely as possible.
    Normalize over per-group channels and spatial dims.
    """
    n, h, w, c = x.shape
    if c % groups != 0:
        raise ValueError(f"channels={c} must be divisible by groups={groups}")

    g = groups
    x = jnp.reshape(x, (n, h, w, g, c // g))

    mean = jnp.mean(x, axis=(1, 2, 4), keepdims=True)
    var = jnp.var(x, axis=(1, 2, 4), keepdims=True)

    x = (x - mean) / jnp.sqrt(var + eps)
    x = jnp.reshape(x, (n, h, w, c))

    scale = hk.get_parameter(
        f"{name}_scale",
        shape=(c,),
        dtype=x.dtype,
        init=jnp.ones,
    )
    offset = hk.get_parameter(
        f"{name}_offset",
        shape=(c,),
        dtype=x.dtype,
        init=jnp.zeros,
    )

    x = x * scale[None, None, None, :] + offset[None, None, None, :]
    return x


def maybe_dropout(x, rate, is_training):
    if rate <= 0.0:
        return x
    if is_training:
        return hk.dropout(hk.next_rng_key(), rate, x)
    return x


class ResidualLayer2D(hk.Module):
    def __init__(self, out_channels, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels

    def __call__(self, x):
        in_channels = x.shape[-1]
        mid_channels = max(self.out_channels // 4, 8)

        h = hk.Conv2D(
            output_channels=mid_channels,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=False,
            data_format="NHWC",
            name="conv1",
        )(x)
        g1 = 4 if mid_channels % 4 == 0 else 1
        h = group_norm_nhwc(h, g1, name="norm1")
        h = jax.nn.relu(h)

        h = hk.Conv2D(
            output_channels=mid_channels,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=False,
            data_format="NHWC",
            name="conv2",
        )(h)
        g2 = 4 if mid_channels % 4 == 0 else 1
        h = group_norm_nhwc(h, g2, name="norm2")
        h = jax.nn.relu(h)

        h = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=False,
            data_format="NHWC",
            name="conv3",
        )(h)
        g3 = 8 if self.out_channels % 8 == 0 else 1
        h = group_norm_nhwc(h, g3, name="norm3")

        if in_channels != self.out_channels:
            x = hk.Conv2D(
                output_channels=self.out_channels,
                kernel_shape=1,
                stride=1,
                padding="SAME",
                with_bias=False,
                data_format="NHWC",
                name="shortcut",
            )(x)

        return jax.nn.relu(x + h)


class Downsample2D(hk.Module):
    def __init__(self, out_channels, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels

    def __call__(self, x):
        # Match PyTorch Conv2d(kernel=4, stride=2, padding=1)
        x = jnp.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        return hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=4,
            stride=2,
            padding="VALID",
            with_bias=True,
            data_format="NHWC",
            name="conv",
        )(x)


class Upsample2D(hk.Module):
    def __init__(self, out_channels, name=None):
        super().__init__(name=name)
        self.out_channels = out_channels

    def __call__(self, x):
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="bilinear")
        x = hk.Conv2D(
            output_channels=self.out_channels,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=True,
            data_format="NHWC",
            name="conv",
        )(x)
        return x


class StabilityUNet2DJAX(hk.Module):
    def __init__(
        self,
        in_channels=8,
        base_channels=64,
        scale=3,
        num_blocks=1,
        feat_channels=64,
        head_hidden=64,
        dropout=0.10,
        name=None,
    ):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.scale = scale
        self.num_blocks = num_blocks
        self.feat_channels = feat_channels
        self.head_hidden = head_hidden
        self.dropout = dropout

    def __call__(self, x, is_training: bool):
        stem_groups = 8 if self.base_channels % 8 == 0 else (4 if self.base_channels % 4 == 0 else 1)

        x = hk.Conv2D(
            output_channels=self.base_channels,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=True,
            data_format="NHWC",
            name="input_proj",
        )(x)
        x = group_norm_nhwc(x, stem_groups, name="stem_norm")
        x = jax.nn.relu(x)

        ch = self.base_channels
        hs = []
        for i in range(self.scale):
            out_ch = ch if i == 0 else 2 * ch
            x = Downsample2D(out_ch, name=f"down_{i}_down")(x)
            x = jax.nn.relu(x)
            for j in range(self.num_blocks):
                x = ResidualLayer2D(out_ch, name=f"down_{i}_res_{j}")(x)
            hs.append(x)
            ch = out_ch

        x = ResidualLayer2D(ch, name="mid_block1")(x)
        x = ResidualLayer2D(ch, name="mid_block2")(x)
        x = maybe_dropout(x, self.dropout, is_training)

        for i in range(self.scale):
            skip = hs.pop()
            x = jnp.concatenate([x, skip], axis=-1)
            x = ResidualLayer2D(ch, name=f"up_{i}_merge")(x)
            for j in range(self.num_blocks):
                x = ResidualLayer2D(ch, name=f"up_{i}_res_{j}")(x)
            if i < self.scale - 1:
                x = Upsample2D(ch // 2, name=f"up_{i}_up")(x)
                x = jax.nn.relu(x)
                ch = ch // 2
            else:
                x = Upsample2D(ch, name=f"up_{i}_up")(x)
                x = jax.nn.relu(x)

        feat = hk.Conv2D(
            output_channels=self.feat_channels,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=True,
            data_format="NHWC",
            name="conv_out_feat",
        )(x)

        feat = jnp.mean(feat, axis=(1, 2))
        h = hk.Linear(self.head_hidden, name="head_linear1")(feat)
        h = jax.nn.relu(h)
        logits = hk.Linear(1, name="head_linear2")(h)
        return logits[:, 0]


def checker_forward(feats, is_training: bool):
    model = StabilityUNet2DJAX()
    return model(feats, is_training=is_training)


def build_checker_model():
    """Construct the Haiku transform; checkpoint parameters are supplied by callers."""
    return hk.transform(checker_forward)


def init_checker_params(rng_key, batch_size=1, h=16, w=16, c=8):
    model = build_checker_model()
    dummy = jnp.zeros((batch_size, h, w, c), dtype=jnp.float32)
    params = model.init(rng_key, dummy, is_training=False)
    return model, params


def checker_predict_proba(model, params, feats):
    logits = model.apply(params, None, feats, is_training=False)
    return jax.nn.sigmoid(logits)


def checker_predict(model, params, feats, threshold=0.5):
    probs = checker_predict_proba(model, params, feats)
    return (probs > threshold).astype(jnp.float32)


MAX_CHECKER_ID = 512
CHECKER_MODEL = build_checker_model()


@jax.jit
def relabel_zero_components_for_checker_jax_single(board: jnp.ndarray) -> jnp.ndarray:
    """
    board: (H, W), int32
    Replace each 0-connected component with a unique positive id appended after current max piece id.
    Pure JAX version for MCTS checker usage.
    """
    h, w = board.shape
    zero_mask = (board == 0)

    init_labels = jnp.where(
        zero_mask,
        jnp.arange(h * w, dtype=jnp.int32).reshape(h, w) + 1,
        0,
    )

    big = jnp.int32(h * w + 1)

    def body_fn(_, labels):
        up = jnp.pad(labels[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        down = jnp.pad(labels[1:, :], ((0, 1), (0, 0)), constant_values=0)
        left = jnp.pad(labels[:, :-1], ((0, 0), (1, 0)), constant_values=0)
        right = jnp.pad(labels[:, 1:], ((0, 0), (0, 1)), constant_values=0)

        c0 = jnp.where(zero_mask & (labels > 0), labels, big)
        c1 = jnp.where(zero_mask & (up > 0), up, big)
        c2 = jnp.where(zero_mask & (down > 0), down, big)
        c3 = jnp.where(zero_mask & (left > 0), left, big)
        c4 = jnp.where(zero_mask & (right > 0), right, big)

        new_labels = jnp.minimum(jnp.minimum(jnp.minimum(c0, c1), jnp.minimum(c2, c3)), c4)
        new_labels = jnp.where(zero_mask, new_labels, 0)
        return new_labels

    labels = jax.lax.fori_loop(0, h * w, body_fn, init_labels)

    max_piece_id = jnp.max(jnp.where(board > 0, board, 0))
    relabeled = jnp.where(zero_mask, max_piece_id + labels, board)
    return relabeled


@jax.jit
def compute_piece_size_map_jax_pure(board: jnp.ndarray) -> jnp.ndarray:
    """
    board: (B, H, W), int32
    return: (B, H, W, 1)
    Pure JAX piece size map with fixed MAX_CHECKER_ID.
    """
    b, h, w = board.shape
    norm = jnp.float32(max(h * w, 1))

    ids = jnp.arange(MAX_CHECKER_ID + 1, dtype=jnp.int32)  # fixed compile-time constant
    counts = jnp.sum(board[..., None] == ids[None, None, None, :], axis=(1, 2)).astype(jnp.float32) / norm

    idx = jnp.clip(board, 0, MAX_CHECKER_ID)
    size_map = jnp.take_along_axis(counts[:, None, None, :], idx[..., None], axis=-1)
    size_map = jnp.where(board[..., None] >= 0, size_map, 0.0)
    return size_map.astype(float_type)


@jax.jit
def encode_board_features_2d_jax_mcts(boards: jnp.ndarray, hold_masks: jnp.ndarray) -> jnp.ndarray:
    """
    boards: (B, H, W), int32
    hold_masks: (B, H, W)
    Assumes boards already have target size (e.g. 16x16).
    Pure JAX feature encoder for MCTS.
    Returns NHWC: (B, H, W, 8)
    """
    hold_masks = hold_masks.astype(float_type)
    virtual_hold = (boards == 0).astype(float_type)
    hold_masks = jnp.maximum(hold_masks, virtual_hold)
    hold_masks = jnp.where(boards >= 0, hold_masks, 0.0)
    adjacency = get_adjacency_map_jax(boards)
    blocked = (boards == -1)[..., None].astype(float_type)
    piece_size = compute_piece_size_map_jax_pure(boards)
    support = compute_bottom_support_map_jax(boards)
    hold = hold_masks[..., None]

    feats = jnp.concatenate([adjacency, blocked, piece_size, support, hold], axis=-1)
    return feats


@jax.jit
def checker_predict_batch_jax(boards: jnp.ndarray, hold_masks: jnp.ndarray, checker_params) -> jnp.ndarray:
    """
    boards: (B, H, W), int32
    hold_masks: (B, H, W)

    Encode the supplied board with held pieces and virtual zero-region support.
    Zero regions keep their IDs; no component relabeling occurs in this path.
    """
    boards = boards.astype(jnp.int32)
    hold_masks = hold_masks.astype(float_type)
    feats = encode_board_features_2d_jax_mcts(boards, hold_masks)

    logits = CHECKER_MODEL.apply(checker_params, None, feats, is_training=False)
    probs = jax.nn.sigmoid(logits)
    return probs


@jax.jit
def checker_predict_batch_no_hold_jax(boards: jnp.ndarray, checker_params) -> jnp.ndarray:
    hold_masks = jnp.zeros_like(boards, dtype=float_type)
    return checker_predict_batch_jax(boards, hold_masks, checker_params)
