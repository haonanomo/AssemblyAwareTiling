"""Load PNG silhouettes as blocked masks: 1 is blocked and 0 is a target cell.

Pixels brighter than the threshold are target cells. Loading preserves image
dimensions; the tiling environment crops and aligns masks to its board size.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from PIL import Image


def load_mask_from_png(path, threshold=127):
    """Decode a PNG to a two-dimensional int32 JAX blocked mask.

    Images are converted to grayscale without resizing or alpha compositing.
    """
    with Image.open(path) as image:
        arr = np.array(image.convert("L"))
    mask = np.where(arr > threshold, 0, 1).astype(np.int32)
    return jnp.array(mask)


def load_masks_from_png_dir(folder):
    """Return name/mask dictionaries for PNG files sorted by filename.

    Only files directly inside folder are loaded. An absent directory or a
    directory without PNG files raises FileNotFoundError.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"PNG directory does not exist: {folder}")
    png_files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in: {folder}")
    return [{"name": path.stem, "mask": load_mask_from_png(path)} for path in png_files]
