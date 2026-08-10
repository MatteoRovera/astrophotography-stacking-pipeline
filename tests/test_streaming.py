from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astro_stack.pipeline import _run_in_memory_pipeline
from astro_stack.stacking import stack_mean, stack_median, stack_sigma_clip
from astro_stack.streaming import (
    _open_memmap,
    _write_array_to_disk,
    chunked_combine,
    estimate_chunk_rows,
    run_streaming_pipeline,
)


def test_estimate_chunk_rows_respects_budget():
    # A tiny budget for many frames of a wide image should force chunking
    # (fewer rows per chunk than the image actually has).
    rows = estimate_chunk_rows(n_frames=50, width=4000, channels=3, budget_bytes=1024 * 1024)
    assert 1 <= rows < 200

    # A generous budget for a small image should allow the whole thing at once.
    rows_big = estimate_chunk_rows(n_frames=5, width=10, channels=1, budget_bytes=1024 * 1024 * 1024)
    assert rows_big >= 100


def test_chunked_combine_matches_eager_combine(tmp_path):
    rng = np.random.default_rng(0)
    n, h, w = 7, 23, 17  # deliberately not a multiple of the chunk size
    stack = rng.uniform(0, 1, size=(n, h, w)).astype(np.float32)
    # Inject an outlier in one frame so sigma-clip rejection actually engages.
    stack[3, 5, 5] = 50.0

    for i in range(n):
        _write_array_to_disk(stack[i], tmp_path / f"src_{i}.dat")
    memmaps = [_open_memmap(tmp_path / f"src_{i}.dat", (h, w)) for i in range(n)]

    for method in ("mean", "median", "sigma"):
        expected = {
            "mean": stack_mean(stack),
            "median": stack_median(stack),
            "sigma": stack_sigma_clip(stack, sigma=3.0, maxiters=5),
        }[method]

        out_mm = np.memmap(tmp_path / f"out_{method}.dat", mode="w+", dtype=np.float32, shape=(h, w))
        # chunk_rows=4 forces several row-bands so this actually exercises chunking.
        chunked_combine(memmaps, method, sigma=3.0, maxiters=5, out=out_mm, chunk_rows=4)

        assert np.allclose(np.array(out_mm), expected, atol=1e-5), f"mismatch for method={method}"


def _write_fits(path: Path, data: np.ndarray) -> None:
    fits.writeto(path, data.astype(np.float32), overwrite=True)


def _make_dataset(root: Path, size: int = 40) -> Path:
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:size, 0:size]
    dark_offset = np.full((size, size), 5.0, dtype=np.float32)

    def star_field(shift=(0, 0), seed=0):
        r = np.random.default_rng(seed)
        field = r.normal(20, 1.0, size=(size, size)).astype(np.float32)
        for cy, cx, amp in [(15, 15, 150), (25, 28, 90)]:
            field += amp * np.exp(
                -(((yy - cy - shift[0]) ** 2 + (xx - cx - shift[1]) ** 2) / (2 * 1.8**2))
            )
        return field

    for i in range(4):
        shift = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
        light = star_field(shift, seed=100 + i) + dark_offset
        _write_fits(root / f"Light_{i:03d}.fits", light)

    for i in range(3):
        d = dark_offset + rng.normal(0, 0.3, size=(size, size)).astype(np.float32)
        _write_fits(root / f"Dark_{i:03d}.fits", d)

    return root


def test_streaming_pipeline_matches_in_memory_pipeline(tmp_path):
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    _make_dataset(dataset_dir)

    streaming_out = tmp_path / "streaming_result.tiff"
    in_memory_out = tmp_path / "in_memory_result.tiff"

    streaming_summary = run_streaming_pipeline(
        {
            "light": sorted(dataset_dir.glob("Light_*.fits")),
            "dark": sorted(dataset_dir.glob("Dark_*.fits")),
            "flat": [],
            "bias": [],
        },
        streaming_out,
        stack_method="mean",
        stretch_method="none",
        reference_index=0,
        bit_depth=16,
        memory_budget_mb=1,  # force tiny chunks so chunking logic is exercised
    )

    in_memory_summary = _run_in_memory_pipeline(
        dataset_dir,
        in_memory_out,
        stack_method="mean",
        stretch_method="none",
        reference_index=0,
        bit_depth=16,
    )

    assert streaming_out.exists()
    assert streaming_summary["mode"] == "streaming"
    assert in_memory_summary["mode"] == "in-memory"

    # Same math, same data -- noise reduction should agree closely between
    # the two orchestration paths (small differences are possible since
    # alignment can independently pick astroalign vs. phase-correlation).
    streaming_noise = streaming_summary["noise_report"]["after"].noise_std
    in_memory_noise = in_memory_summary["noise_report"]["after"].noise_std
    assert streaming_noise == pytest.approx(in_memory_noise, rel=0.2)
