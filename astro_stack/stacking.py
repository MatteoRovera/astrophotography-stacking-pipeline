"""combine aligned frames into one image and measure the noise reduction.
mean, median, or sigma-clipped mean. see readme for when to use each."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .alignment import to_luminance
from .loader import Frame

logger = logging.getLogger("astro_stack.stacking")

STACK_METHODS = ("mean", "median", "sigma")


def stack_mean(data: np.ndarray) -> np.ndarray:
    """average frames along axis 0. optimal snr for gaussian noise."""
    return data.mean(axis=0).astype(np.float32)


def stack_median(data: np.ndarray) -> np.ndarray:
    """median combine frames along axis 0. immune to outliers."""
    return np.median(data, axis=0).astype(np.float32)


def stack_sigma_clip(data: np.ndarray, sigma: float = 3.0, maxiters: int = 5) -> np.ndarray:
    from astropy.stats import sigma_clip

    clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
    combined = np.ma.mean(clipped, axis=0)

    mask = np.ma.getmaskarray(combined)
    if mask.any():
        # rare: every frame clipped away at this pixel. fall back to plain mean.
        fallback = data.mean(axis=0)
        combined = np.where(mask, fallback, np.ma.filled(combined, 0.0))
    else:
        combined = np.ma.filled(combined, 0.0)

    return np.asarray(combined, dtype=np.float32)


def stack_frames(
    frames: list[Frame], method: str = "sigma", sigma: float = 3.0, maxiters: int = 5
) -> np.ndarray:
    if not frames:
        raise ValueError("No frames to stack")
    if method not in STACK_METHODS:
        raise ValueError(f"Unknown stack method {method!r}; choose from {STACK_METHODS}")

    data = np.stack([f.data for f in frames], axis=0)
    logger.info("Stacking %d frame(s) with method=%s", len(frames), method)

    if method == "mean":
        return stack_mean(data)
    if method == "median":
        return stack_median(data)
    return stack_sigma_clip(data, sigma=sigma, maxiters=maxiters)


@dataclass
class NoiseStats:
    background_median: float
    noise_std: float
    peak_signal: float
    snr: float


def measure_noise_snr(image: np.ndarray) -> NoiseStats:
    """estimate background noise and snr with sigma-clipped stats, no
    hand-picked empty patch needed."""
    from astropy.stats import sigma_clipped_stats

    gray = to_luminance(image)
    _mean, median, std = sigma_clipped_stats(gray, sigma=3.0, maxiters=5)
    noise = max(float(std), 1e-8)
    peak = float(np.percentile(gray, 99.9))
    signal = max(peak - float(median), 0.0)
    snr = signal / noise
    return NoiseStats(background_median=float(median), noise_std=noise, peak_signal=peak, snr=snr)


def compare_before_after(single_frame: np.ndarray, stacked: np.ndarray, n_frames: int) -> dict:
    """noise/snr for one raw frame vs the final stack, plus the theoretical prediction."""
    before = measure_noise_snr(single_frame)
    after = measure_noise_snr(stacked)
    observed_noise_ratio = before.noise_std / after.noise_std if after.noise_std > 0 else float("inf")
    theoretical_ratio = np.sqrt(n_frames)
    return {
        "before": before,
        "after": after,
        "observed_noise_reduction": observed_noise_ratio,
        "theoretical_noise_reduction_sqrt_n": float(theoretical_ratio),
        "snr_improvement": (after.snr / before.snr) if before.snr > 0 else float("inf"),
    }
