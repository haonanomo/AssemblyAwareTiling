"""Attention policy and value network for polyomino tiling."""

import haiku as hk
import jax
import jax.numpy as jnp


@jax.jit
def get_adjacency_map(board):
    """Return four directional boundary features for each board cell."""
    B, N, M = board.shape
    up_match = (board[:, 1:, :] != board[:, :-1, :])
    down_match = (board[:, :-1, :] != board[:, 1:, :])
    left_match = (board[:, :, 1:] != board[:, :, :-1])
    right_match = (board[:, :, :-1] != board[:, :, 1:])

    up_board = (board[:, 0, :] != -1)[:, jnp.newaxis, :]
    up = jnp.concatenate([up_board, up_match], axis=1)
    down_board = (board[:, N - 1, :] != -1)[:, jnp.newaxis, :]
    down = jnp.concatenate([down_match, down_board], axis=1)
    left_board = (board[:, :, 0] != -1)[:, :, jnp.newaxis]
    left = jnp.concatenate([left_board, left_match], axis=2)
    right_board = (board[:, :, M - 1] != -1)[:, :, jnp.newaxis]
    right = jnp.concatenate([right_match, right_board], axis=2)
    return jnp.stack([up, down, left, right], axis=-1)


def get_2d_sin_cos(height, width, d_model):
    """Build sinusoidal position features with shape (height, width, d_model)."""
    half_dim = d_model // 2
    pe = jnp.zeros((d_model, height, width), dtype=jnp.float32)
    div_term = jnp.exp(jnp.arange(0., half_dim, 2) * -(jnp.log(10000.0) / half_dim))

    pos_w = jnp.arange(width)[:, None]
    sin_w = jnp.sin(pos_w * div_term)
    cos_w = jnp.cos(pos_w * div_term)
    sin_w = sin_w.T[:, None, :]
    cos_w = cos_w.T[:, None, :]
    sin_w = jnp.tile(sin_w, (1, height, 1))
    cos_w = jnp.tile(cos_w, (1, height, 1))

    pos_h = jnp.arange(height)[:, None]
    sin_h = jnp.sin(pos_h * div_term)
    cos_h = jnp.cos(pos_h * div_term)
    sin_h = sin_h.T[:, :, None]
    cos_h = cos_h.T[:, :, None]
    sin_h = jnp.tile(sin_h, (1, 1, width))
    cos_h = jnp.tile(cos_h, (1, 1, width))

    pe = pe.at[0:half_dim:2].set(sin_w)
    pe = pe.at[1:half_dim:2].set(cos_w)
    pe = pe.at[half_dim::2].set(sin_h)
    pe = pe.at[half_dim + 1::2].set(cos_h)

    pe = jnp.transpose(pe, (1, 2, 0))
    return pe


def pack_bits(board_feats_bin):
    assert board_feats_bin.shape[-1] <= 7
    bits = jnp.array([1 << i for i in range(board_feats_bin.shape[-1])], dtype=jnp.int32)
    return jnp.sum(board_feats_bin.astype(jnp.int32) * bits, axis=-1)


class TransformerBlock(hk.Module):
    """Residual attention and feed-forward block with layer normalization."""

    def __init__(self, num_heads: int, model_size: int, name=None):
        super().__init__(name=name)
        self.mha = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=model_size // num_heads,
            model_size=model_size,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        )
        self.ln1 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.ln2 = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        self.mlp = hk.Sequential([
            hk.Linear(2 * model_size),
            jax.nn.gelu,
            hk.Linear(model_size)
        ])

    def __call__(self, query, key=None, value=None, mask=None):
        k = key if key is not None else query
        v = value if value is not None else query
        x = query + self.mha(query, k, v, mask=mask)
        x = self.ln1(x)
        x = x + self.mlp(x)
        return self.ln2(x)


def get_local_patch(board_tokens, board_valid, pos, h, w, k):
    """
    board_tokens: (B, HW, C)
    board_valid: (B, HW)
    pos: (B, 2)
    returns:
        patch_tokens: (B, K*K, C)
        patch_valid: (B, K*K)
    """
    B, HW, C = board_tokens.shape
    pad = k // 2

    board_grid = board_tokens.reshape(B, h, w, C)
    valid_grid = board_valid.reshape(B, h, w)

    board_grid = jnp.pad(
        board_grid,
        ((0,0),(pad,pad),(pad,pad),(0,0)),
        mode="constant"
    )
    valid_grid = jnp.pad(
        valid_grid,
        ((0,0),(pad,pad),(pad,pad)),
        constant_values=0
    )

    pos_padded = pos + pad

    def extract_single(b, v, p):
        y, x = p
        patch = jax.lax.dynamic_slice(b, (y - pad, x - pad, 0), (k, k, C))
        patch_v = jax.lax.dynamic_slice(v, (y - pad, x - pad), (k, k))
        return patch, patch_v

    patch, patch_valid = jax.vmap(extract_single)(board_grid, valid_grid, pos_padded)

    patch_tokens = patch.reshape(B, k*k, C)
    patch_valid = patch_valid.reshape(B, k*k)

    return patch_tokens, patch_valid


class AttentionPolicyPatch(hk.Module):
    """Encode board and piece tokens with local cross-attention.

    Boards use -1 for blocked cells, 0 for empty cells, and positive piece IDs.
    Returns action logits and a value in [0, 1] for each batch element.
    """

    def __init__(self, hidden_dim=128, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_indicator = hk.get_parameter(
            "target_indicator",
            [hidden_dim],
            init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        self.board_embed = hk.Embed(vocab_size=128, embed_dim=hidden_dim)
        self.initial_conv = hk.Conv2D(hidden_dim, kernel_shape=3)
        self.inventory_embedder = hk.Embed(vocab_size=100, embed_dim=hidden_dim)

        self.res_blocks = [
            (hk.Conv2D(hidden_dim, 3, name=f"res_{i}_1"),
             hk.Conv2D(hidden_dim, 3, name=f"res_{i}_2"))
            for i in range(4)
        ]

        self.pos_proj = hk.Linear(hidden_dim)
        self.piece_proj = hk.Linear(hidden_dim)

        self.board_self_attn_layers = [
            TransformerBlock(num_heads, hidden_dim, name=f"board_self_attn_{i}")
            for i in range(3)
        ]

        self.piece_self_attn_layers = [
            TransformerBlock(num_heads, hidden_dim, name=f"piece_self_attn_{i}")
            for i in range(3)
        ]

        self.cross_attn_layers = [
            TransformerBlock(num_heads, hidden_dim, name=f"cross_attn_{i}")
            for i in range(2)
        ]

        self.policy_head = hk.Linear(1)

        self.v_head = hk.Sequential([
            hk.Linear(hidden_dim * 2),
            jax.nn.relu,
            hk.Linear(1),
            jax.nn.sigmoid
        ])

    def __call__(self, board, inventory, pos, pieces, is_sampling: bool):
        batch_size, h, w = board.shape

        def masked_mean(x, mask, axis):
            mask = mask.astype(jnp.float32)
            x = x * mask[..., None]
            return jnp.sum(x, axis=axis) / (jnp.sum(mask, axis=axis, keepdims=True) + 1e-6)

        # Board features.
        adjacency = get_adjacency_map(board)
        board_feats = jnp.concatenate([
            adjacency,
            (board == -1)[..., None],
            (board == 0)[..., None],
            (board > 0)[..., None],
        ], axis=-1).astype(jnp.float32)
        board_feats_int = pack_bits(board_feats)
        board_embeds = self.board_embed(board_feats_int)

        # Positional encoding.
        grid_enc = get_2d_sin_cos(h, w, self.hidden_dim)
        grid_enc = self.pos_proj(grid_enc)
        grid_enc = jnp.broadcast_to(grid_enc[None, ...],(batch_size, h, w, self.hidden_dim))
        board_tokens = jnp.reshape(board_embeds, (batch_size, h * w, self.hidden_dim))
        board_tokens = board_tokens + jnp.reshape(grid_enc, (batch_size, h * w, self.hidden_dim))

        board_valid = (board != -1).reshape(batch_size, -1)
        mask = board_valid[:, None, :, None] * board_valid[:, None, None, :]
        for layer in self.board_self_attn_layers:
            board_tokens = layer(board_tokens, mask=mask.astype(bool))


        # Pieces.
        num_pieces = pieces.shape[1]
        inventory_embeds = self.inventory_embedder(inventory)
        inventory_embeds = inventory_embeds * (inventory != 0)[..., None]
        piece_tokens = jnp.reshape(pieces, (batch_size, num_pieces, -1)).astype(jnp.float32)
        piece_tokens = self.piece_proj(piece_tokens)
        piece_tokens = piece_tokens + inventory_embeds

        piece_valid = (inventory != 0)
        mask = piece_valid[:, None, :, None] * piece_valid[:, None, None, :]

        for layer in self.piece_self_attn_layers:
            piece_tokens = layer(piece_tokens, mask=mask.astype(bool))
            piece_tokens = piece_tokens * piece_valid[..., None]

        # Cross-attention with local patch.
        patch_tokens, patch_valid = get_local_patch(
            board_tokens, board_valid, pos, h, w, k=5
        )

        mask = (piece_valid[:, None, :, None] *
                patch_valid[:, None, None, :])

        for layer in self.cross_attn_layers:
            piece_tokens = layer(
                query=piece_tokens,
                key=patch_tokens,
                value=patch_tokens,
                mask=mask.astype(bool),
            )
            piece_tokens = piece_tokens * piece_valid[..., None]

        # Policy.
        logits = self.policy_head(piece_tokens).squeeze(-1)
        logits = jnp.where(inventory == 0, -1e9, logits)

        # Value.
        global_avg = masked_mean(board_tokens, board_valid, axis=1)
        avg_piece_context = masked_mean(piece_tokens, piece_valid, axis=1)
        global_feats = jnp.concatenate(
            [global_avg, avg_piece_context], axis=-1
        )
        value = self.v_head(global_feats).squeeze(-1)

        # Sampling.
        sample_logits = jnp.where(inventory == 0, -1e9, 1.0)

        return (
            jnp.where(is_sampling, sample_logits, logits),
            jnp.where(is_sampling, 0.0, value),
        )
