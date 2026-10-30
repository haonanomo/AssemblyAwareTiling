"""Generate polyomino orientations and the piece IDs used by the policies.

Enumeration order and ID offsets are retained for checkpoint compatibility.
"""

import jax.numpy as jnp

def get_symmetries(polyomino):
    """Return distinct normalized rotations and reflections as coordinate tuples."""
    def normalize(polyomino):
        """Shifts the polyomino so its minimum coordinates are (0,0)."""
        min_x = min(p[0] for p in polyomino)
        min_y = min(p[1] for p in polyomino)
        return tuple(sorted((x - min_x, y - min_y) for x, y in polyomino))

    symmetries = set()
    current = polyomino
    for _ in range(4):
        current = [(-y, x) for x, y in current]
        symmetries.add(normalize(current))
        reflected = [(-x, y) for x, y in current]
        symmetries.add(normalize(reflected))
    return symmetries


def generate_polyominoes(n, grid_size=4):
    """Generates all unique polyominoes of size n that fit in grid_size."""
    if n <= 0:
        raise ValueError("Piece size must be positive")
    if grid_size < 1 or n > grid_size * grid_size:
        raise ValueError("Piece size must fit in a positive grid")

    current_set = {((0, 0),)}
    def is_within_bounds(polyomino, max_dim=5):
        """Checks if the polyomino fits within a max_dim x max_dim grid."""
        if not polyomino: return True
        xs = [p[0] for p in polyomino]
        ys = [p[1] for p in polyomino]
        return (max(xs) - min(xs) < max_dim) and (max(ys) - min(ys) < max_dim)

    def to_board(polyomino, size=4):
        """Convert coordinates into a square binary grid of the requested size."""
        board = [[0 for _ in range(size)] for _ in range(size)]
        for x, y in polyomino:
            board[y][x] = 1
        return board

    for _ in range(n - 1):
        next_set = set()
        for poly in current_set:
            for x, y in poly:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    neighbor = (x + dx, y + dy)
                    if neighbor not in poly:
                        new_poly = poly + (neighbor,)
                        if is_within_bounds(new_poly, grid_size):
                            syms = get_symmetries(new_poly)
                            next_set.add(min(syms))
        current_set = next_set

    pieces = []
    pieces_id = []
    sorted_canonical = sorted(list(current_set))
    for canonical_id, canonical_poly in enumerate(sorted_canonical):
        # Keep the existing orientation order for policy checkpoint compatibility.
        unique_orientations = get_symmetries(canonical_poly)

        for orientation in unique_orientations:
            pieces.append(to_board(orientation, grid_size))
            pieces_id.append(canonical_id)

    return jnp.array(pieces, dtype=jnp.int32), jnp.array(pieces_id, dtype=jnp.int32)


def generate_polyominoes_from_list(n_piece_list, grid_size=4):
    """Combine requested sizes, preserving orientation order and legacy ID offsets."""
    pieces = []
    pieces_id = []
    global_id = 0
    for n_pieces in n_piece_list:
        p, p_id = generate_polyominoes(n_pieces, grid_size)
        pieces.extend(p)
        pieces_id.extend(p_id + global_id)
        global_id += p_id.shape[0]
    if not pieces:
        raise ValueError("At least one piece size is required")
    return jnp.stack(pieces, dtype=jnp.int32), jnp.stack(pieces_id, dtype=jnp.int32)
