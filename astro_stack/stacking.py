"""Combine aligned frames into one image, and quantify the noise reduction.

Why stacking reduces noise at all: assume each pixel's noise (photon shot
noise + sensor read noise) is independent from frame to frame. Averaging N
frames makes the *signal* add linearly (N x), while independent noise adds
in quadrature (sqrt(N) x). The ratio signal/noise therefore improves by
sqrt(N) -- this is the entire reason deep-sky imaging stacks dozens of
sub-exposures instead of taking one long one.

Three combine methods, in order of when to reach for them:

- MEAN: the statistically optimal estimator for independent Gaussian noise
  -- it gets the full sqrt(N) improvement. Its weakness is that it has no
  concept of "this pixel is wrong": a satellite trail, plane, or cosmic ray
  hit in even one frame gets divided by N and blended into every pixel it
  touched, leaving a faint but real streak in the result.
- MEDIAN: the middle value at each pixel is immune to a minority of frames
  being contaminated (as long as fewer than half the frames are bad at that
  pixel, outliers vanish completely). The cost is statistical efficiency --
  for large N, the median's noise is about 1.25x the mean's (a well known
  result for Gaussian statistics), so it needs ~1.57x as many frames to
  match the mean's noise reduction.
- SIGMA-CLIPPED MEAN: the practical default, and what tools like
  DeepSkyStacker/PixInsight use. Per pixel, compute the mean and std across
  the stack, mask any value more than `sigma` std devs away, then average
  what's left. This rejects outliers like the median does, but because it
  only throws away the actual bad pixels (not smooshing everything to the
  middle value), it keeps noise performance close to the plain mean. It
  needs enough frames (rule of thumb: 5+) for the per-pixel statistics to
  be meaningful.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .alignment import to_luminance
from .loader import Frame

logger = logging.getLogger("astro_stack.stacking")

STACK_METHODS = ("mean", "median", "sigma")


def stack_mean(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=0).astype(np.float32)


def stack_median(data: np.ndarray) -> np.ndarray:
    return np.median(data, axis=0).astype(np.float32)


def stack_sigma_clip(data: np.ndarray, sigma: float = 3.0, maxiters: int = 5) -> np.ndarray:
    from astropy.stats import sigma_clip

    clipped = sigma_clip(data, sigma=sigma, maxiters=maxiters, axis=0, masked=True)
    combined = np.ma.mean(clipped, axis=0)

    mask = np.ma.getmaskarray(combined)
    if mask.any():
        # A pixel where every frame got clipped away (all N values were
        # flagged as outliers of each other -- rare, but possible with very
        # few frames). Fall back to the plain mean there instead of NaN.
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
    """Estimate background noise and SNR of a single image.

    Background level/noise use sigma-clipped statistics over the *whole*
    image: stars and the target are a small minority of pixels, so an
    iterative sigma-clip converges on the sky background's mean/median and
    std without needing a hand-picked empty patch. "Signal" is taken as the
    99.9th-percentile brightness above that background (a robust stand-in
    for the target's peak, without letting a single hot pixel dominate).
    """
    from astropy.stats import sigma_clipped_stats

    gray = to_luminance(image)
    _mean, median, std = sigma_clipped_stats(gray, sigma=3.0, maxiters=5)
    noise = max(float(std), 1e-8)
    peak = float(np.percentile(gray, 99.9))
    signal = max(peak - float(median), 0.0)
    snr = signal / noise
    return NoiseStats(background_median=float(median), noise_std=noise, peak_signal=peak, snr=snr)


def compare_before_after(single_frame: np.ndarray, stacked: np.ndarray, n_frames: int) -> dict:
    """Report noise/SNR for one raw frame vs. the final stack, plus theory."""
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
