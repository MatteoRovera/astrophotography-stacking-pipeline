"""stretch the stacked image so faint detail is visible, then save it.
percentile (linear) or asinh (linear near zero, log for bright values).
see readme for why asinh is the default."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("astro_stack.postprocess")

STRETCH_METHODS = ("percentile", "asinh", "none")


def stretch_percentile(image: np.ndarray, low_pct: float = 0.25, high_pct: float = 99.75) -> np.ndarray:
    """linear stretch between low and high percentile cutpoints."""
    lo, hi = np.percentile(image, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = (image - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def stretch_asinh(image: np.ndarray, black_point_pct: float = 1.0, scale: float = 10.0) -> np.ndarray:
    """asinh stretch: linear near zero, logarithmic for bright values."""
    black = np.percentile(image, black_point_pct)
    shifted = np.clip(image - black, 0.0, None)
    max_val = float(shifted.max())
    if max_val <= 0:
        max_val = 1e-6
    stretched = np.arcsinh(scale * shifted) / np.arcsinh(scale * max_val)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def stretch_image(image: np.ndarray, method: str = "asinh", **kwargs) -> np.ndarray:
    """apply a stretch method to remap intensity. 'none', 'percentile', or 'asinh'."""
    if method == "none":
        lo, hi = float(image.min()), float(image.max())
        if hi <= lo:
            hi = lo + 1e-6
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if method == "percentile":
        return stretch_percentile(image, **kwargs)
    if method == "asinh":
        return stretch_asinh(image, **kwargs)
    raise ValueError(f"Unknown stretch method {method!r}; choose from {STRETCH_METHODS}")


def save_image(image: np.ndarray, output_path: Path, bit_depth: int = 16) -> None:
    """save a [0, 1] float image. 16-bit for tiff, 8-bit for everything else."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    ext = output_path.suffix.lower()

    if ext in {".tif", ".tiff"} and bit_depth == 16:
        import tifffile

        data = (clipped * 65535).astype(np.uint16)
        tifffile.imwrite(output_path, data)
    else:
        import imageio.v3 as iio

        data = (clipped * 255).astype(np.uint8)
        iio.imwrite(output_path, data)

    logger.info("Saved result to %s (shape=%s)", output_path, image.shape)
