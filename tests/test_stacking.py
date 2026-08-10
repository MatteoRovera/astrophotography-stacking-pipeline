from pathlib import Path

import numpy as np

from astro_stack.loader import Frame
from astro_stack.stacking import stack_frames


def _make_frames(image: np.ndarray, n: int) -> list[Frame]:
    return [
        Frame(data=image.copy(), path=Path(f"frame_{i}.fits"), frame_type="light")
        for i in range(n)
    ]


def test_stack_mean_of_identical_frames_returns_same_frame():
    rng = np.random.default_rng(0)
    image = rng.uniform(0.1, 0.9, size=(32, 32)).astype(np.float32)
    frames = _make_frames(image, 5)

    result = stack_frames(frames, method="mean")

    assert np.allclose(result, image, atol=1e-6)


def test_stack_median_of_identical_frames_returns_same_frame():
    rng = np.random.default_rng(1)
    image = rng.uniform(0.1, 0.9, size=(32, 32)).astype(np.float32)
    frames = _make_frames(image, 5)

    result = stack_frames(frames, method="median")

    assert np.allclose(result, image, atol=1e-6)


def test_stack_sigma_clip_of_identical_frames_returns_same_frame():
    rng = np.random.default_rng(2)
    image = rng.uniform(0.1, 0.9, size=(32, 32)).astype(np.float32)
    frames = _make_frames(image, 8)

    result = stack_frames(frames, method="sigma", sigma=3.0)

    assert np.allclose(result, image, atol=1e-6)


def test_stack_identical_frames_works_for_color_images():
    rng = np.random.default_rng(3)
    image = rng.uniform(0.1, 0.9, size=(16, 16, 3)).astype(np.float32)
    frames = _make_frames(image, 4)

    result = stack_frames(frames, method="mean")

    assert result.shape == image.shape
    assert np.allclose(result, image, atol=1e-6)
