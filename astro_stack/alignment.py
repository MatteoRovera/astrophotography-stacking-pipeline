"""align light frames to a common reference frame.
tries astroalign (star triangle matching) first, falls back to phase
correlation (fft shift, translation only) if there aren't enough stars.
see readme for the full tradeoff writeup."""

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
    """collapse a color frame to grayscale for star detection."""
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
    return shift  # (dy, dx), sub-pixel accurate


def _shift_image(data: np.ndarray, shift: np.ndarray) -> np.ndarray:
    from scipy.ndimage import shift as nd_shift

    full_shift = tuple(shift) + (0,) * (data.ndim - len(shift))
    return nd_shift(data, shift=full_shift, order=3, mode="constant", cval=0.0).astype(np.float32)


def check_astroalign_available() -> bool:
    """check once whether astroalign is installed."""
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
    """align one frame onto a reference. shared by align_frames() and
    streaming.py so streaming can align one frame at a time."""
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
    """align every frame onto frames[reference_index]."""
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
