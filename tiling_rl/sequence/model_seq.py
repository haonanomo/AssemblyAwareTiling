"""Piece-ID-invariant attention policy and value network for disassembly."""

import haiku as hk
import jax
import jax.numpy as jnp


def get_2d_sin_cos(height, width, d_model):
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


class TransformerBlock(hk.Module):
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


def pack_bits_seq(board_feats_bin):
    n_bits = board_feats_bin.shape[-1]
    assert n_bits <= 9
    bits = jnp.array([1 << i for i in range(n_bits)], dtype=jnp.int32)
    return jnp.sum(board_feats_bin.astype(jnp.int32) * bits, axis=-1)


def _shift_bool(mask, direction):
    """Align neighbors of a [B,H,W] mask to each cell.
    Directions 0/1/2/3 are up/down/left/right.
    """
    if direction == 0:
        pad = jnp.zeros_like(mask[:, :1, :])
        return jnp.concatenate([pad, mask[:, :-1, :]], axis=1)
    if direction == 1:
        pad = jnp.zeros_like(mask[:, :1, :])
        return jnp.concatenate([mask[:, 1:, :], pad], axis=1)
    if direction == 2:
        pad = jnp.zeros_like(mask[:, :, :1])
        return jnp.concatenate([pad, mask[:, :, :-1]], axis=2)
    if direction == 3:
        pad = jnp.zeros_like(mask[:, :, :1])
        return jnp.concatenate([mask[:, :, 1:], pad], axis=2)
    raise ValueError("direction must be 0, 1, 2, or 3")


def _shift_values(values, direction, fill_value=-1):
    """
    values: [B, H, W]
    Returns neighbor values aligned to the current cell.
    """
    fill = jnp.full_like(values[:, :1, :], fill_value)
    fill_col = jnp.full_like(values[:, :, :1], fill_value)

    if direction == 0:
        return jnp.concatenate([fill, values[:, :-1, :]], axis=1)
    if direction == 1:
        return jnp.concatenate([values[:, 1:, :], fill], axis=1)
    if direction == 2:
        return jnp.concatenate([fill_col, values[:, :, :-1]], axis=2)
    if direction == 3:
        return jnp.concatenate([values[:, :, 1:], fill_col], axis=2)
    raise ValueError("direction must be 0, 1, 2, or 3")


def get_id_invariant_adjacency_map(board):
    """Return [B,H,W,4] contacts between distinct positive piece IDs.
    Channels are up/down/left/right; features are invariant to ID relabeling.
    """
    board = jnp.asarray(board)
    cur_real = board > 0
    feats = []
    for d in range(4):
        nb = _shift_values(board, d, fill_value=-1)
        nb_real = nb > 0
        diff_real_piece = cur_real & nb_real & (board != nb)
        feats.append(diff_real_piece[..., None])
    return jnp.concatenate(feats, axis=-1).astype(jnp.float32)


def hold_mask_from_piece_masks_and_states(piece_masks, part_states):
    """Return a [B,H,W] mask for held pieces (part_states == 2).
    piece_masks is [B,P,H,W]; part_states is [B,P]. IDs need not be contiguous.
    """
    held = (part_states == 2).astype(piece_masks.dtype)
    held_masks = piece_masks * held[:, :, None, None]
    return jnp.any(held_masks > 0, axis=1)


def virtual_contact_dir_masks_from_board_and_piece_masks(board, piece_masks):
    """Return [B,P,H,W,4] contacts between real pieces and board == 0.
    Inputs: board [B,H,W], piece_masks [B,P,H,W].
    Channels are up/down/left/right; absolute piece IDs are unused.
    """
    zero = board == 0
    piece = piece_masks > 0

    dirs = []
    for d in range(4):
        zero_neighbor = _shift_bool(zero, d)
        dirs.append((piece & zero_neighbor[:, None, :, :])[..., None])

    return jnp.concatenate(dirs, axis=-1).astype(jnp.float32)


def masked_board_pool_per_part(board_tokens, piece_masks, eps=1e-6):
    """
    board_tokens: [B, H*W, C]
    piece_masks:  [B, P, H, W], aligned with board coordinates
    returns:      [B, P, C]
    """
    batch_size, hw, hidden_dim = board_tokens.shape
    _, num_parts, h, w = piece_masks.shape

    masks = piece_masks.reshape(batch_size, num_parts, h * w).astype(jnp.float32)
    denom = jnp.sum(masks, axis=-1, keepdims=True) + eps
    context = jnp.einsum("bph,bhc->bpc", masks, board_tokens) / denom
    return context


class AttentionPolicyPatchSeq(hk.Module):
    """Predict hold/remove logits and a normalized value from board/part features.

    IK and stability constraints are applied by the environment and search;
    IK tables and joint configurations are not network inputs.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_heads=4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.board_embed = hk.Embed(vocab_size=512, embed_dim=hidden_dim)
        self.pos_proj = hk.Linear(hidden_dim)

        self.state_embed = hk.Embed(vocab_size=4, embed_dim=hidden_dim)
        self.board_context_proj = hk.Linear(hidden_dim)
        self.virtual_contact_proj = hk.Linear(hidden_dim)
        self.global_proj = hk.Linear(hidden_dim)

        self.board_self_attn_layers = [
            TransformerBlock(num_heads, hidden_dim, name=f"board_self_attn_{i}")
            for i in range(3)
        ]

        self.part_self_attn_layers = [
            TransformerBlock(num_heads, hidden_dim, name=f"part_self_attn_{i}")
            for i in range(4)
        ]

        self.global_cross_attn_layers = [
            TransformerBlock(num_heads, hidden_dim, name=f"global_cross_attn_{i}")
            for i in range(2)
        ]

        self.hold_head = hk.Linear(1, name="hold_head")
        self.remove_head = hk.Linear(1, name="remove_head")

        self.v_head = hk.Sequential([
            hk.Linear(hidden_dim * 3),
            jax.nn.relu,
            hk.Linear(hidden_dim),
            jax.nn.relu,
            hk.Linear(1),
            jax.nn.sigmoid,
        ])

    def __call__(
        self,
        board,
        part_states,
        contact_matrix,
        piece_masks,
        is_sampling: bool = False,
    ):
        """Compute ID-invariant sequence logits [B,2P] and values [B].

        board [B,H,W]: -1 outside, 0 virtual support, positive IDs real pieces.
        part_states [B,P]: 0 removed, 1 free, 2 held.
        contact_matrix [B,P,P]: contacts between real pieces.
        piece_masks [B,P,H,W]: real-piece masks; P defines action order.
        Logits [:P] select hold actions; [P:] select removal actions.
        The virtual support is excluded from action parts.
        """
        batch_size, h, w = board.shape
        _, num_parts = part_states.shape

        contact_matrix = contact_matrix.astype(jnp.float32)
        piece_masks = piece_masks.astype(jnp.float32)

        part_exists = part_states > 0
        part_free = part_states == 1
        part_held = part_states == 2

        adjacency = get_id_invariant_adjacency_map(board)
        held_mask = hold_mask_from_piece_masks_and_states(piece_masks, part_states)

        board_feats = jnp.concatenate([
            adjacency,
            (board == -1)[..., None],
            (board == 0)[..., None],
            (board > 0)[..., None],
            held_mask[..., None],
        ], axis=-1).astype(jnp.float32)

        board_feats_int = pack_bits_seq(board_feats)
        board_tokens = self.board_embed(board_feats_int)

        grid_enc = get_2d_sin_cos(h, w, self.hidden_dim)
        grid_enc = self.pos_proj(grid_enc)
        grid_enc = jnp.broadcast_to(
            grid_enc[None, ...],
            (batch_size, h, w, self.hidden_dim),
        )

        board_tokens = board_tokens + grid_enc
        board_tokens = board_tokens.reshape(batch_size, h * w, self.hidden_dim)

        board_valid = (board != -1).reshape(batch_size, h * w)
        board_attn_mask = board_valid[:, None, :, None] * board_valid[:, None, None, :]

        for layer in self.board_self_attn_layers:
            board_tokens = layer(board_tokens, mask=board_attn_mask.astype(bool))
            board_tokens = board_tokens * board_valid[..., None]

        part_board_context = masked_board_pool_per_part(
            board_tokens=board_tokens,
            piece_masks=piece_masks,
        )
        part_board_tokens = self.board_context_proj(part_board_context)

        virtual_contact_masks = virtual_contact_dir_masks_from_board_and_piece_masks(
            board=board,
            piece_masks=piece_masks,
        )
        virtual_contact_flat = virtual_contact_masks.reshape(batch_size, num_parts, h * w * 4)
        virtual_contact_tokens = self.virtual_contact_proj(virtual_contact_flat)

        state_tokens = self.state_embed(jnp.clip(part_states, 0, 3))

        part_tokens = (
            state_tokens
            + part_board_tokens
            + virtual_contact_tokens
        )
        part_tokens = part_tokens * part_exists[..., None]

        part_valid = part_exists
        self_mask = part_valid[:, None, :, None] * part_valid[:, None, None, :]

        contact_mask = contact_matrix > 0
        eye = jnp.eye(num_parts, dtype=bool)[None, :, :]
        contact_or_self = contact_mask | eye
        part_attn_mask = self_mask & contact_or_self[:, None, :, :]

        for layer in self.part_self_attn_layers:
            part_tokens = layer(part_tokens, mask=part_attn_mask.astype(bool))
            part_tokens = part_tokens * part_valid[..., None]

        global_token = self._masked_mean(part_tokens, part_valid, axis=1)
        global_token = self.global_proj(global_token)[:, None, :]

        global_mask = part_valid[:, None, None, :]
        for layer in self.global_cross_attn_layers:
            global_token = layer(
                query=global_token,
                key=part_tokens,
                value=part_tokens,
                mask=global_mask.astype(bool),
            )

        global_context = jnp.broadcast_to(
            global_token,
            (batch_size, num_parts, self.hidden_dim),
        )

        action_tokens = part_tokens + global_context

        hold_logits = self.hold_head(action_tokens).squeeze(-1)
        remove_logits = self.remove_head(action_tokens).squeeze(-1)
        logits = jnp.concatenate([hold_logits, remove_logits], axis=-1)

        valid_actions = jnp.concatenate([part_free, part_held], axis=-1)
        logits = jnp.where(valid_actions, logits, -1e9)

        sample_logits = jnp.where(valid_actions, 1.0, -1e9)

        board_context = self._masked_mean(board_tokens, board_valid, axis=1)
        existing_context = self._masked_mean(part_tokens, part_exists, axis=1)
        held_context = self._masked_mean(part_tokens, part_held, axis=1)

        value_feats = jnp.concatenate([
            board_context,
            existing_context,
            held_context,
        ], axis=-1)

        value = self.v_head(value_feats).squeeze(-1)

        return (
            jnp.where(is_sampling, sample_logits, logits),
            jnp.where(is_sampling, 0.0, value),
        )

    def _masked_mean(self, x, mask, axis):
        mask = mask.astype(jnp.float32)
        x = x * mask[..., None]
        denom = jnp.sum(mask, axis=axis, keepdims=True) + 1e-6
        return jnp.sum(x, axis=axis) / denom


AttentionPolicy = AttentionPolicyPatchSeq
