"""build master dark/flat/bias frames and calibrate lights against them.
calibrated = (light - master_dark) / normalized_master_flat. see readme
for the reasoning. skipped entirely if no cal frames exist."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .loader import Frame

logger = logging.getLogger("astro_stack.calibration")


def _stack_matching_shape(frames: list[Frame], label: str) -> np.ndarray:
    """stack frames of identical shape, dropping mismatches."""
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
    """combine bias frames into a master bias using median."""
    if not bias_frames:
        return None
    master = median_combine(bias_frames, "bias")
    logger.info("Built master BIAS from %d frame(s)", len(bias_frames))
    return master


def build_master_dark(dark_frames: list[Frame]) -> Optional[np.ndarray]:
    """combine dark frames into a master dark using median."""
    if not dark_frames:
        return None
    master = median_combine(dark_frames, "dark")
    logger.info("Built master DARK from %d frame(s)", len(dark_frames))
    return master


def normalize_flat(
    combined: np.ndarray, master_bias: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """bias-correct and normalize a combined flat to mean=1. split out so
    streaming.py can reuse it."""
    if master_bias is not None and master_bias.shape == combined.shape:
        combined = combined - master_bias
    # normalize to mean=1 so dividing corrects relative brightness only
    mean_val = float(np.mean(combined))
    if mean_val <= 0:
        logger.warning("Master flat has non-positive mean; skipping flat correction")
        return None
    return combined / mean_val


def build_master_flat(
    flat_frames: list[Frame], master_bias: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """combine flat frames into a normalized master flat using median."""
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
        # avoid divide-by-near-zero at dead/vignetted pixels
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
    """calibrate all light frames, skipping cleanly if no cal frames exist."""
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
