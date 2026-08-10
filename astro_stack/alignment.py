"""Align (register) light frames to a common reference frame.

Between exposures, the target drifts across the sensor -- tracking error,
polar-alignment drift, or just a handheld phone moving slightly. Stacking
misaligned frames blurs everything instead of reinforcing it, so every frame
must be resampled onto the same pixel grid as a chosen reference frame
before stacking.

Two methods are used, chosen automatically per-frame:

1. astroalign (primary). Detects star-like point sources, forms triangles
   from triplets of them, and matches triangles between the source and
   reference image by their shape (side-length ratios), which stays the
   same regardless of rotation, scale or translation. Matched triangles
   give point correspondences, from which a similarity transform (rotation
   + scale + translation) is solved. This is robust and handles field
   rotation, but it needs a handful of well-detected stars -- it has
   nothing to work with on a sparse frame (a few stars, or a single bright
   disk like the Moon).

2. Phase correlation (fallback). Computes the pixel shift between two
   images from the phase of their FFT cross-power spectrum -- no point
   detection required. It only recovers a pure translation (no rotation),
   but that's exactly what's needed for a short, sparse, or single-object
   sequence (e.g. handheld Moon shots) where astroalign has too few stars
   to form triangles from.

astroalign is tried first; if it raises (too few matched stars) or isn't
installed, the frame falls back to phase correlation automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .loader import Frame

logger = logging.getLogger("astro_stack.alignment")

_LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass
class AlignmentInfo:
    filename: str
    method: str  # 'reference' | 'astroalign' | 'phase_correlation' | 'failed'
    success: bool
    detail: str = ""


def to_luminance(data: np.ndarray) -> np.ndarray:
    """Collapse a color frame to a single 2D array for star/feature detection."""
    if data.ndim == 2:
        return data.astype(np.float32)
    channels = data.shape[-1]
    if channels >= 3:
        return (data[..., :3] * _LUMINANCE_WEIGHTS).sum(axis=-1).astype(np.float32)
    return data.mean(axis=-1).astype(np.float32)


def _warp_with_transform(data: np.ndarray, transform, output_shape: tuple) -> np.ndarray:
    from skimage.transform import warp

    kwargs = dict(order=3, mode="constant", cval=0.0, preserve_range=True)
    if data.ndim == 2:
        return warp(data, transform.inverse, output_shape=output_shape, **kwargs).astype(np.float32)
    channels = [
        warp(data[..., c], transform.inverse, output_shape=output_shape, **kwargs)
        for c in range(data.shape[-1])
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def _phase_correlation_shift(ref_gray: np.ndarray, src_gray: np.ndarray) -> np.ndarray:
    from skimage.registration import phase_cross_correlation

    shift, _error, _diffphase = phase_cross_correlation(ref_gray, src_gray, upsample_factor=10)
    return shift  # (dy, dx), sub-pixel


def _shift_image(data: np.ndarray, shift: np.ndarray) -> np.ndarray:
    from scipy.ndimage import shift as nd_shift

    full_shift = tuple(shift) + (0,) * (data.ndim - len(shift))
    return nd_shift(data, shift=full_shift, order=3, mode="constant", cval=0.0).astype(np.float32)


def check_astroalign_available() -> bool:
    """Import-check astroalign once; log the fallback-everywhere case clearly."""
    try:
        import astroalign  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "astroalign not installed -- using translation-only phase "
            "correlation for every frame"
        )
        return False


def align_single(
    frame: Frame, ref_gray: np.ndarray, have_astroalign: bool
) -> tuple[Optional[np.ndarray], AlignmentInfo]:
    """Align one frame onto a reference luminance image (astroalign, else phase correlation).

    Factored out of align_frames() so the memory-bounded streaming pipeline
    (streaming.py) can align one frame at a time without needing every frame
    resident in memory at once.
    """
    src_gray = to_luminance(frame.data)

    if have_astroalign:
        import astroalign

        try:
            transform, (src_pts, _tgt_pts) = astroalign.find_transform(src_gray, ref_gray)
            registered = _warp_with_transform(frame.data, transform, ref_gray.shape)
            return registered, AlignmentInfo(
                frame.path.name, "astroalign", True, detail=f"{len(src_pts)} matched stars"
            )
        except Exception as exc:  # noqa: BLE001 - too few stars, etc.
            logger.info(
                "astroalign failed for %s (%s); falling back to phase correlation",
                frame.path.name, exc,
            )

    try:
        shift = _phase_correlation_shift(ref_gray, src_gray)
        registered = _shift_image(frame.data, shift)
        return registered, AlignmentInfo(
            frame.path.name, "phase_correlation", True,
            detail=f"shift=({shift[0]:.2f}, {shift[1]:.2f}) px",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Alignment failed entirely for %s: %s", frame.path.name, exc)
        return None, AlignmentInfo(frame.path.name, "failed", False, detail=str(exc))


def align_frames(
    frames: list[Frame], reference_index: int = 0
) -> tuple[list[Frame], list[AlignmentInfo]]:
    """Align every frame in ``frames`` onto ``frames[reference_index]``."""
    if not frames:
        raise ValueError("No frames to align")

    have_astroalign = check_astroalign_available()

    reference = frames[reference_index]
    ref_gray = to_luminance(reference.data)

    aligned: list[Frame] = []
    infos: list[AlignmentInfo] = []

    for i, frame in enumerate(frames):
        if i == reference_index:
            aligned.append(frame)
            infos.append(AlignmentInfo(frame.path.name, "reference", True))
            continue

        registered, info = align_single(frame, ref_gray, have_astroalign)
        infos.append(info)
        if registered is None:
            continue

        aligned.append(
            Frame(
                data=registered,
                path=frame.path,
                frame_type=frame.frame_type,
                exposure_s=frame.exposure_s,
                header=frame.header,
            )
        )

    for info in infos:
        logger.info("Align %-28s method=%-17s ok=%s %s", info.filename, info.method, info.success, info.detail)

    return aligned, infos
