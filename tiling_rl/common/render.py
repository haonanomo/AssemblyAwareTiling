"""Render tiled boards with shared colors for symmetry-equivalent pieces."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def get_canonical_shape(board, piece_id):
    """Return a rotation/reflection-invariant shape key, or None for an absent ID."""
    coords = np.argwhere(board == piece_id)
    if len(coords) == 0: return None

    coords -= coords.min(axis=0)

    max_r, max_c = coords.max(axis=0)
    shape_grid = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in coords:
        shape_grid[r, c] = 1

    symmetries = []
    curr = shape_grid
    for _ in range(4):
        curr = np.rot90(curr)
        symmetries.append(curr)
        symmetries.append(np.flipud(curr))

    forms = []
    for s in symmetries:
        # Trim empty rows/cols that might appear after rotation
        r_mask = np.any(s, axis=1)
        c_mask = np.any(s, axis=0)
        s_trimmed = s[np.ix_(r_mask, c_mask)]
        forms.append(tuple(map(tuple, s_trimmed)))

    return min(forms)


def render_equivalent_sets(board, pos=None, save_path=None, held_ids=None):
    """Save or display a board, optionally marking a cell and held pieces.

    Positive values identify pieces; -1 is blocked, 0 is empty, and -2 marks
    holes. pos is a (row, column) pair; held_ids contains positive piece IDs.
    """
    board = np.array(board)
    held_ids = set([] if held_ids is None else held_ids)
    height, width = board.shape

    # Group pieces by shape.
    unique_ids = [i for i in np.unique(board) if i > 0]
    shape_to_group_id = {}
    id_to_group_id = {}
    next_group_id = 0

    for pid in unique_ids:
        canonical = get_canonical_shape(board, pid)
        if canonical not in shape_to_group_id:
            shape_to_group_id[canonical] = next_group_id
            next_group_id += 1
        id_to_group_id[pid] = shape_to_group_id[canonical]

    fig, ax = plt.subplots(figsize=(width * 0.8, height * 0.8))

    color_blocked = "#ffffff"
    color_empty = "#ecf0f1"
    color_hole = "#f6e58d"
    line_color = "#2c3e50"
    voxel_line_color = "#bdc3c7"

    lw = 2
    voxel_lw = 0.5
    voxel_alpha = 0.6

    num_sets = next_group_id
    colors = sns.color_palette("muted", num_sets) if num_sets > 0 else []

    for r in range(height):
        for c in range(width):
            seq = int(board[r, c])

            if seq > 0:
                color = colors[id_to_group_id[seq]]
            elif seq == -1:
                color = color_blocked
            elif seq == -2:
                color = color_hole
            else:
                color = color_empty

            rect = patches.Rectangle((c, height - r - 1), 1, 1, linewidth=0, facecolor=color)
            ax.add_patch(rect)

            # Outline held pieces.
            if seq > 0 and seq in held_ids:
                hold_rect = patches.Rectangle(
                    (c + 0.12, height - r - 1 + 0.12),
                    0.76,
                    0.76,
                    linewidth=2.0,
                    edgecolor="#e74c3c",
                    facecolor="none",
                    zorder=4,
                )
                ax.add_patch(hold_rect)

            # Draw the top and left grid edges for each cell.
            grid_params = dict(color=voxel_line_color, linewidth=voxel_lw, alpha=voxel_alpha, zorder=2)
            ax.plot([c, c + 1], [height - r, height - r], **grid_params)
            ax.plot([c, c], [height - r - 1, height - r], **grid_params)

            # Close the grid on the far right and bottom edges
            if c == width - 1:
                ax.plot([c + 1, c + 1], [height - r - 1, height - r], **grid_params)
            if r == height - 1:
                ax.plot([c, c + 1], [height - r - 1, height - r - 1], **grid_params)

            def draw_bold(x, y):
                is_on_boundary = (
                        (np.all(np.equal(x, 0))) or
                        (np.all(np.equal(x, width))) or
                        (np.all(np.equal(y, 0))) or
                        (np.all(np.equal(y, height)))
                )

                # Use 2x width for the board perimeter, 1x for piece boundaries
                current_lw = lw * 2 if is_on_boundary else lw
                ax.plot(x, y, color=line_color, linewidth=current_lw,
                        solid_capstyle='butt', zorder=3)

            target_l = board[r, c - 1] if c > 0 else -1
            if target_l != seq and ((seq > 0) or (seq == -2) or (target_l > 0) or (target_l == -2)):
                draw_bold([c, c], [height - r - 1, height - r])

            target_r = board[r, c + 1] if c + 1 < width else -1
            if target_r != seq and ((seq > 0) or (seq == -2) or (target_r > 0) or (target_r == -2)):
                draw_bold([c + 1, c + 1], [height - r - 1, height - r])

            target_t = board[r - 1, c] if r > 0 else -1
            if target_t != seq and ((seq > 0) or (seq == -2) or (target_t > 0) or (target_t == -2)):
                draw_bold([c, c + 1], [height - r, height - r])

            target_b = board[r + 1, c] if r + 1 < height else -1
            if target_b != seq and ((seq > 0) or (seq == -2) or (target_b > 0) or (target_b == -2)):
                draw_bold([c, c + 1], [height - r - 1, height - r - 1])

    # Mark the selected cell.
    if pos is not None:
        r_pos, c_pos = pos
        cross_color = "#e74c3c"
        ax.plot([c_pos, c_pos + 1], [height - r_pos - 1, height - r_pos], color=cross_color, linewidth=lw + 1, zorder=4)
        ax.plot([c_pos, c_pos + 1], [height - r_pos, height - r_pos - 1], color=cross_color, linewidth=lw + 1, zorder=4)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close(fig)
