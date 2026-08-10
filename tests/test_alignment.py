from pathlib import Path

import numpy as np
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation

from astro_stack.alignment import align_frames
from astro_stack.loader import Frame


def _synthetic_star_field(size=200, n_stars=40, seed=0) -> np.ndarray:
    """A background-noise field with several Gaussian 'stars' -- enough
    point sources for astroalign to form triangles from."""
    rng = np.random.default_rng(seed)
    field = rng.normal(0.02, 0.005, size=(size, size)).astype(np.float32)

    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_stars):
        cy, cx = rng.uniform(20, size - 20, size=2)
        amp = rng.uniform(0.3, 1.0)
        sigma = rng.uniform(1.2, 2.0)
        field += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))

    return np.clip(field, 0, None).astype(np.float32)


def test_alignment_recovers_known_injected_shift():
    reference = _synthetic_star_field()
    injected_shift = (5.3, -3.7)  # (dy, dx)
    shifted = nd_shift(reference, shift=injected_shift, order=3, mode="constant", cval=0.02)

    frames = [
        Frame(data=reference, path=Path("ref.fits"), frame_type="light"),
        Frame(data=shifted.astype(np.float32), path=Path("shifted.fits"), frame_type="light"),
    ]

    aligned, infos = align_frames(frames, reference_index=0)

    assert infos[1].success
    assert infos[1].method in ("astroalign", "phase_correlation")

    # After alignment, the residual shift between the aligned frame and the
    # reference should be close to zero -- i.e. the injected shift was undone.
    residual_shift, _error, _phase = phase_cross_correlation(
        reference, aligned[1].data, upsample_factor=10
    )
    assert np.allclose(residual_shift, (0.0, 0.0), atol=0.5)


def test_alignment_falls_back_to_phase_correlation_for_sparse_frame():
    """A single bright disk (like a Moon shot) has too few point sources for
    astroalign's triangle matching, so this should hit the phase-correlation
    fallback -- and still recover the shift correctly. The disk needs a
    sharp edge (like the Moon's limb) rather than a smooth Gaussian glow:
    phase correlation locates a shift from high-frequency content, and a
    perfectly smooth blob doesn't have enough of it to pin down a precise
    peak."""
    size = 100
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.sqrt((yy - 50) ** 2 + (xx - 50) ** 2)
    reference = (radius < 20).astype(np.float32)
    reference += rng.normal(0, 0.02, size=reference.shape).astype(np.float32)  # surface texture
    reference = np.clip(reference, 0, None).astype(np.float32)

    injected_shift = (4.0, 6.0)
    shifted = nd_shift(reference, shift=injected_shift, order=3, mode="constant", cval=0.0)

    frames = [
        Frame(data=reference, path=Path("moon_ref.fits"), frame_type="light"),
        Frame(data=shifted.astype(np.float32), path=Path("moon_shifted.fits"), frame_type="light"),
    ]

    aligned, infos = align_frames(frames, reference_index=0)

    assert infos[1].success
    residual_shift, _error, _phase = phase_cross_correlation(
        reference, aligned[1].data, upsample_factor=10
    )
    assert np.allclose(residual_shift, (0.0, 0.0), atol=0.5)
