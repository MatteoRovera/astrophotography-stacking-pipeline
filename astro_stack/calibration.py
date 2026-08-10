"""Build master calibration frames and apply them to light frames.

The three corrections mirror the three defects each calibration frame type
isolates (see loader.py docstring):

    calibrated = (light - master_dark) / normalized_master_flat

- Subtracting the master DARK removes the additive thermal/read-noise
  pattern baked into every pixel at that exposure time.
- Dividing by the normalized master FLAT removes multiplicative optical
  defects (vignetting, dust shadows) without changing overall brightness,
  because the flat is normalized to a mean of 1 before it's used.
- BIAS is not subtracted from the light directly when a matching dark is
  available -- the dark already contains that same read-noise offset (a
  dark exposure is "bias + thermal noise"), so subtracting bias too would
  remove it twice. Instead bias is used to clean up the flat (a flat's
  own bias offset would otherwise skew the normalization), or as a
  fallback stand-in for dark subtraction when no matching dark exists
  (this only removes the fixed offset, not thermal buildup, so it's an
  approximation -- fine for short exposures, less accurate for long ones).

Master frames are built with a per-pixel MEDIAN across the set, not a mean.
A median rejects one-off outliers (cosmic ray hits, hot pixels that spiked
in a single sub-frame) that a mean would blend into the result.

If no dark/flat/bias frames are present, calibration is skipped entirely and
the light frames pass through untouched -- this is what makes the pipeline
work on phone photos with no calibration set.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .loader import Frame

logger = logging.getLogger("astro_stack.calibration")


def _stack_matching_shape(frames: list[Frame], label: str) -> np.ndarray:
    """Stack frames of identical shape into (N, H, W[, C]); drop mismatches."""
    ref_shape = frames[0].data.shape
    good = [f for f in frames if f.data.shape == ref_shape]
    if len(good) < len(frames):
        dropped = len(frames) - len(good)
        logger.warning(
            "Dropped %d %s frame(s) with mismatched shape (expected %s)",
            dropped, label, ref_shape,
        )
    return np.stack([f.data for f in good], axis=0)


def median_combine(frames: list[Frame], label: str) -> np.ndarray:
    stack = _stack_matching_shape(frames, label)
    return np.median(stack, axis=0).astype(np.float32)


def build_master_bias(bias_frames: list[Frame]) -> Optional[np.ndarray]:
    if not bias_frames:
        return None
    master = median_combine(bias_frames, "bias")
    logger.info("Built master BIAS from %d frame(s)", len(bias_frames))
    return master


def build_master_dark(dark_frames: list[Frame]) -> Optional[np.ndarray]:
    if not dark_frames:
        return None
    master = median_combine(dark_frames, "dark")
    logger.info("Built master DARK from %d frame(s)", len(dark_frames))
    return master


def normalize_flat(
    combined: np.ndarray, master_bias: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """Bias-correct (if possible) and normalize a combined flat to mean=1.

    Split out of build_master_flat() so the streaming pipeline can reuse the
    exact same normalization on a flat it combined via chunked processing.
    """
    if master_bias is not None and master_bias.shape == combined.shape:
        combined = combined - master_bias
    # Normalize to a mean of 1 so dividing by the flat corrects *relative*
    # brightness (vignetting/dust) without darkening or brightening the image.
    mean_val = float(np.mean(combined))
    if mean_val <= 0:
        logger.warning("Master flat has non-positive mean; skipping flat correction")
        return None
    return combined / mean_val


def build_master_flat(
    flat_frames: list[Frame], master_bias: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    if not flat_frames:
        return None
    combined = median_combine(flat_frames, "flat")
    normalized = normalize_flat(combined, master_bias)
    if normalized is not None:
        logger.info("Built master FLAT from %d frame(s), normalized to mean=1", len(flat_frames))
    return normalized


def calibrate_light(
    light: Frame,
    master_dark: Optional[np.ndarray],
    master_bias: Optional[np.ndarray],
    master_flat: Optional[np.ndarray],
) -> Frame:
    data = light.data.astype(np.float32).copy()

    if master_dark is not None and master_dark.shape == data.shape:
        data -= master_dark
    elif master_bias is not None and master_bias.shape == data.shape:
        data -= master_bias

    if master_flat is not None and master_flat.shape == data.shape:
        # Guard against divide-by-near-zero at dead/vignetted pixel edges.
        safe_flat = np.clip(master_flat, 1e-3, None)
        data /= safe_flat

    return Frame(
        data=data,
        path=light.path,
        frame_type="light",
        exposure_s=light.exposure_s,
        header=light.header,
    )


def calibrate_lights(buckets: dict[str, list[Frame]]) -> list[Frame]:
    """Calibrate all light frames, skipping cleanly if no cal frames exist."""
    lights = buckets.get("light", [])
    darks = buckets.get("dark", [])
    flats = buckets.get("flat", [])
    biases = buckets.get("bias", [])

    if not darks and not flats and not biases:
        logger.info("No dark/flat/bias frames found -- skipping calibration.")
        return lights

    master_bias = build_master_bias(biases)
    master_dark = build_master_dark(darks)
    master_flat = build_master_flat(flats, master_bias)

    if master_dark is None and master_bias is None:
        logger.info("No dark or bias frames -- lights will not be offset-corrected.")
    if master_flat is None:
        logger.info("No flat frames -- lights will not be flat-fielded.")

    calibrated = [
        calibrate_light(light, master_dark, master_bias, master_flat) for light in lights
    ]
    logger.info("Calibrated %d light frame(s)", len(calibrated))
    return calibrated
