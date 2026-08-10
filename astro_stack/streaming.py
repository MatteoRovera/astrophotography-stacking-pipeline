"""Memory-bounded version of the pipeline, for datasets too big to hold in RAM.

The rest of this package (loader.py's ``load_folder``, calibration.py,
alignment.py, stacking.py as called by ``pipeline._run_in_memory_pipeline``)
takes the simple approach: decode every file into a normal in-RAM numpy
array and keep all of them around at once. That's easy to reason about and
is what the unit tests exercise directly, but it doesn't scale -- N frames
at H x W x C x 4 bytes (float32) each means N x H x W x C x 4 bytes of RAM
simultaneously. A session of 80 full-resolution DSLR frames can need tens
of gigabytes that way, far more than a typical machine has free.

This module does the same calibrate -> align -> stack work, but:

1. Decodes one source file at a time (reusing loader.load_frame, the exact
   same decoders), immediately writes the result to a temporary
   memory-mapped file on disk, and drops the in-RAM copy. Peak RAM for
   *loading* is therefore one frame, not N.
2. Any operation that genuinely needs to see all N frames at a pixel (the
   masters' median-combine, and the final stack) is done in horizontal row
   chunks: read a chunk of rows from every frame's memmap (cheap, only
   touches that slice), combine just that chunk with the *same*
   stack_mean/stack_median/stack_sigma_clip functions used everywhere else
   in the package, write the chunk to an output memmap, move to the next
   chunk. Peak RAM for combining is then N x chunk_rows x W x C x 4 bytes,
   which is tuned to a fixed memory budget regardless of image size or N.
3. Calibration and alignment, which only ever look at one frame plus the
   already-tiny master arrays, are done in the same single pass as decoding
   -- so a light frame's temp file already holds the calibrated, aligned
   result, and intermediate per-frame temp files are deleted as soon as
   they're no longer needed rather than piling up for the whole run.

Everything here is orchestration around code that already exists elsewhere
(calibrate_light, align_single, stack_mean/median/sigma_clip) -- the actual
calibration/alignment/stacking math is not duplicated.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .alignment import AlignmentInfo, align_single, check_astroalign_available, to_luminance
from .calibration import calibrate_light, normalize_flat
from .loader import Frame, load_frame
from .stacking import compare_before_after, stack_mean, stack_median, stack_sigma_clip

logger = logging.getLogger("astro_stack.streaming")

DEFAULT_MEMORY_BUDGET_MB = 512


def estimate_chunk_rows(
    n_frames: int, width: int, channels: int, budget_bytes: int, safety_factor: float = 4.0
) -> int:
    """How many rows can be combined at once and stay under ``budget_bytes``.

    ``safety_factor`` gives headroom for the temporary arrays sigma-clipping
    allocates internally (mask, clipped copy, ...) on top of the raw chunk.
    """
    per_row_bytes = max(1, n_frames * width * max(channels, 1) * 4)
    rows = int(budget_bytes / (per_row_bytes * safety_factor))
    return max(1, rows)


def _close_memmap(mm) -> None:
    """Explicitly release the OS file handle behind a memmap.

    On Windows, relying on ``del`` + refcounting to close a memmap's
    underlying file handle is not reliable enough to then immediately
    unlink() that same file -- the delete can fail with a PermissionError
    because the mapping is still technically open. Closing the private
    ``_mmap`` object directly is the standard workaround.
    """
    underlying = getattr(mm, "_mmap", None)
    if underlying is not None:
        try:
            underlying.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def _write_array_to_disk(array: np.ndarray, path: Path) -> None:
    """Write ``array`` to a new memmap file at ``path`` and fully close it."""
    mm = np.memmap(path, mode="w+", dtype=np.float32, shape=array.shape)
    mm[:] = array
    mm.flush()
    _close_memmap(mm)
    del mm


def _open_memmap(path: Path, shape: tuple) -> np.memmap:
    return np.memmap(path, mode="r", dtype=np.float32, shape=shape)


def _combine_chunk(chunk_stack: np.ndarray, method: str, sigma: float, maxiters: int) -> np.ndarray:
    if method == "mean":
        return stack_mean(chunk_stack)
    if method == "median":
        return stack_median(chunk_stack)
    return stack_sigma_clip(chunk_stack, sigma=sigma, maxiters=maxiters)


def chunked_combine(
    sources: list[np.memmap],
    method: str,
    sigma: float,
    maxiters: int,
    out: np.memmap,
    chunk_rows: int,
) -> None:
    """Combine ``sources`` along axis 0 into ``out``, one row-band at a time."""
    height = out.shape[0]
    for r0 in range(0, height, chunk_rows):
        r1 = min(r0 + chunk_rows, height)
        chunk_stack = np.stack([np.array(s[r0:r1]) for s in sources], axis=0)
        out[r0:r1] = _combine_chunk(chunk_stack, method, sigma, maxiters)
    out.flush()


def _build_master_streaming(
    paths: list[Path], label: str, tmp_dir: Path, budget_bytes: int
) -> Optional[np.ndarray]:
    """Median-combine a bucket of calibration frames without holding them all in RAM."""
    if not paths:
        return None

    per_frame_paths: list[Path] = []
    memmaps: list[np.memmap] = []
    shape: Optional[tuple] = None

    for i, p in enumerate(paths):
        frame = load_frame(p)
        if shape is None:
            shape = frame.data.shape
        elif frame.data.shape != shape:
            logger.warning("Dropped %s frame %s with mismatched shape (expected %s)", label, p.name, shape)
            continue
        mm_path = tmp_dir / f"{label}_{i:04d}.dat"
        _write_array_to_disk(frame.data, mm_path)
        per_frame_paths.append(mm_path)
        memmaps.append(_open_memmap(mm_path, shape))
        del frame

    if not memmaps:
        return None

    width = shape[1] if len(shape) >= 2 else 1
    channels = shape[2] if len(shape) == 3 else 1
    chunk_rows = estimate_chunk_rows(len(memmaps), width, channels, budget_bytes)

    master_path = tmp_dir / f"master_{label}.dat"
    master_mm = np.memmap(master_path, mode="w+", dtype=np.float32, shape=shape)
    chunked_combine(memmaps, "median", 3.0, 5, master_mm, chunk_rows)
    result = np.array(master_mm)

    for mm in memmaps:
        _close_memmap(mm)
    _close_memmap(master_mm)
    del memmaps, master_mm
    for p in per_frame_paths:
        p.unlink(missing_ok=True)
    master_path.unlink(missing_ok=True)

    logger.info("Built master %s from %d frame(s) [streaming]", label.upper(), len(per_frame_paths))
    return result


def run_streaming_pipeline(
    paths_by_type: dict[str, list[Path]],
    output_path: Path,
    stack_method: str = "sigma",
    sigma: float = 3.0,
    maxiters: int = 5,
    stretch_method: str = "asinh",
    reference_index: int = 0,
    bit_depth: int = 16,
    memory_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB,
) -> dict:
    from .postprocess import save_image, stretch_image  # local import: avoid cycle at module load

    output_path = Path(output_path)
    budget_bytes = int(max(memory_budget_mb, 64) * 1024 * 1024)

    light_paths = paths_by_type.get("light", [])
    dark_paths = paths_by_type.get("dark", [])
    flat_paths = paths_by_type.get("flat", [])
    bias_paths = paths_by_type.get("bias", [])
    if not light_paths:
        raise ValueError("No light frames found")

    reference_index = max(0, min(reference_index, len(light_paths) - 1))

    with tempfile.TemporaryDirectory(prefix="astro_stack_") as tmp:
        tmp_dir = Path(tmp)

        logger.info("=== Calibration (streaming) ===")
        master_bias = _build_master_streaming(bias_paths, "bias", tmp_dir, budget_bytes)
        master_dark = _build_master_streaming(dark_paths, "dark", tmp_dir, budget_bytes)
        master_flat_raw = _build_master_streaming(flat_paths, "flat", tmp_dir, budget_bytes)
        master_flat = normalize_flat(master_flat_raw, master_bias) if master_flat_raw is not None else None
        if not dark_paths and not flat_paths and not bias_paths:
            logger.info("No dark/flat/bias frames found -- skipping calibration.")

        have_astroalign = check_astroalign_available()

        logger.info("=== Calibrating + aligning light frames (streaming) ===")
        ref_path = light_paths[reference_index]
        ref_frame = calibrate_light(load_frame(ref_path), master_dark, master_bias, master_flat)
        ref_gray = to_luminance(ref_frame.data)
        shape = ref_frame.data.shape

        aligned_paths: list[Optional[Path]] = [None] * len(light_paths)
        infos: list[Optional[AlignmentInfo]] = [None] * len(light_paths)

        ref_mm_path = tmp_dir / "aligned_ref.dat"
        _write_array_to_disk(ref_frame.data, ref_mm_path)
        aligned_paths[reference_index] = ref_mm_path
        infos[reference_index] = AlignmentInfo(ref_path.name, "reference", True)
        del ref_frame

        for i, p in enumerate(light_paths):
            if i == reference_index:
                continue
            frame = calibrate_light(load_frame(p), master_dark, master_bias, master_flat)
            registered, info = align_single(frame, ref_gray, have_astroalign)
            infos[i] = info
            del frame
            if registered is None:
                continue
            mm_path = tmp_dir / f"aligned_{i:04d}.dat"
            _write_array_to_disk(registered, mm_path)
            aligned_paths[i] = mm_path
            del registered

        for info in infos:
            if info is not None:
                logger.info(
                    "Align %-28s method=%-17s ok=%s %s",
                    info.filename, info.method, info.success, info.detail,
                )

        used_paths = [p for p in aligned_paths if p is not None]
        if not used_paths:
            raise RuntimeError("All light frames failed to align")

        logger.info("=== Stacking (%s, streaming) ===", stack_method)
        memmaps = [_open_memmap(p, shape) for p in used_paths]
        width = shape[1] if len(shape) >= 2 else 1
        channels = shape[2] if len(shape) == 3 else 1
        chunk_rows = estimate_chunk_rows(len(memmaps), width, channels, budget_bytes)

        stacked_path = tmp_dir / "stacked.dat"
        stacked_mm = np.memmap(stacked_path, mode="w+", dtype=np.float32, shape=shape)
        chunked_combine(memmaps, stack_method, sigma, maxiters, stacked_mm, chunk_rows)
        stacked = np.array(stacked_mm)
        for mm in memmaps:
            _close_memmap(mm)
        _close_memmap(stacked_mm)
        del memmaps, stacked_mm

        logger.info("=== Noise / SNR ===")
        ref_mm = _open_memmap(ref_mm_path, shape)
        ref_data = np.array(ref_mm)
        _close_memmap(ref_mm)
        del ref_mm
        noise_report = compare_before_after(ref_data, stacked, n_frames=len(used_paths))
        before, after = noise_report["before"], noise_report["after"]
        logger.info(
            "Background noise: %.6f -> %.6f (%.2fx reduction, theoretical sqrt(N)=%.2fx)",
            before.noise_std, after.noise_std,
            noise_report["observed_noise_reduction"],
            noise_report["theoretical_noise_reduction_sqrt_n"],
        )
        logger.info(
            "SNR: %.2f -> %.2f (%.2fx improvement)",
            before.snr, after.snr, noise_report["snr_improvement"],
        )

        logger.info("=== Post-processing (%s stretch) ===", stretch_method)
        stretched = stretch_image(stacked, method=stretch_method)
        save_image(stretched, output_path, bit_depth=bit_depth)

    return {
        "n_lights": len(light_paths),
        "stack_method": stack_method,
        "align_infos": infos,
        "noise_report": noise_report,
        "output_path": str(output_path),
        "mode": "streaming",
    }
