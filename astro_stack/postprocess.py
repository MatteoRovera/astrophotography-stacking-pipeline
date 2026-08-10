"""Stretch the stacked image so faint detail becomes visible, then save it.

A stacked astro image is linear data with an enormous dynamic range: the
sky background and faint nebulosity sit just above zero, while stars are
orders of magnitude brighter. Displayed linearly on an 8-bit screen, that
looks like a black frame with a handful of white dots -- almost everything
interesting is crushed into the bottom few intensity levels. "Stretching"
remaps intensity so the faint stuff becomes visible without just blowing
out the bright stuff to solid white.

Two stretch options:

- PERCENTILE (linear) stretch: pick a low/high percentile of the pixel
  values as black/white point and linearly rescale between them. Simple
  and predictable, but a linear map can't simultaneously reveal faint
  detail *and* avoid clipping bright cores when the dynamic range is huge.
- ASINH stretch: asinh(x) is ~linear for small x and ~logarithmic for large
  x. That matches what we want: faint, near-background signal is stretched
  close to proportionally (so noise doesn't get exaggerated), while bright
  signal (stars, a bright core) gets compressed logarithmically instead of
  slamming into a hard clip. This is the same idea behind the "Lupton
  stretch" used for SDSS survey images, and is generally the better default
  for astrophotography.

Both operate on shared black/white points computed across all channels at
once (not per-channel), so color balance isn't skewed by stretching.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("astro_stack.postprocess")

STRETCH_METHODS = ("percentile", "asinh", "none")


def stretch_percentile(image: np.ndarray, low_pct: float = 0.25, high_pct: float = 99.75) -> np.ndarray:
    lo, hi = np.percentile(image, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1e-6
    stretched = (image - lo) / (hi - lo)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def stretch_asinh(image: np.ndarray, black_point_pct: float = 1.0, scale: float = 10.0) -> np.ndarray:
    black = np.percentile(image, black_point_pct)
    shifted = np.clip(image - black, 0.0, None)
    max_val = float(shifted.max())
    if max_val <= 0:
        max_val = 1e-6
    stretched = np.arcsinh(scale * shifted) / np.arcsinh(scale * max_val)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def stretch_image(image: np.ndarray, method: str = "asinh", **kwargs) -> np.ndarray:
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
    """Save a [0, 1] float image. 16-bit for TIFF, 8-bit for everything else."""
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
