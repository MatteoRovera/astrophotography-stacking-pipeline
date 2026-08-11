"""wires loading, calibration, alignment, stacking, and post-processing
together. picks in-memory or streaming mode based on estimated size."""

from __future__ import annotations

import logging
from pathlib import Path

from .alignment import align_frames
from .calibration import calibrate_lights
from .loader import estimate_decoded_bytes, load_folder, scan_folder
from .postprocess import save_image, stretch_image
from .stacking import compare_before_after, stack_frames
from .streaming import DEFAULT_MEMORY_BUDGET_MB, run_streaming_pipeline

logger = logging.getLogger("astro_stack.pipeline")

# sessions estimated to need more ram than this switch to streaming mode
DEFAULT_IN_MEMORY_BUDGET_MB = 3072


def run_pipeline(
    input_dir: Path,
    output_path: Path,
    stack_method: str = "sigma",
    sigma: float = 3.0,
    maxiters: int = 5,
    stretch_method: str = "asinh",
    reference_index: int = 0,
    recursive: bool = False,
    bit_depth: int = 16,
    memory_budget_mb: float = DEFAULT_IN_MEMORY_BUDGET_MB,
) -> dict:
    """Run the complete astrophotography processing pipeline from input folder to output image.

    Automatically selects in-memory or streaming mode based on estimated memory usage.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    paths_by_type = scan_folder(input_dir, recursive=recursive)
    total_files = sum(len(v) for v in paths_by_type.values())
    total_bytes = sum(estimate_decoded_bytes(p) for paths in paths_by_type.values() for p in paths)
    budget_bytes = memory_budget_mb * 1024 * 1024 if memory_budget_mb and memory_budget_mb > 0 else 0
    use_streaming = budget_bytes <= 0 or total_bytes > budget_bytes

    logger.info(
        "Found %d file(s), estimated decoded size ~%.2f GB (budget %s) -> %s mode",
        total_files,
        total_bytes / 1e9,
        f"{budget_bytes / 1e9:.2f} GB" if budget_bytes > 0 else "streaming forced",
        "streaming" if use_streaming else "in-memory",
    )

    if use_streaming:
        stream_budget = memory_budget_mb if memory_budget_mb and memory_budget_mb > 0 else DEFAULT_MEMORY_BUDGET_MB
        return run_streaming_pipeline(
            paths_by_type,
            output_path,
            stack_method=stack_method,
            sigma=sigma,
            maxiters=maxiters,
            stretch_method=stretch_method,
            reference_index=reference_index,
            bit_depth=bit_depth,
            memory_budget_mb=stream_budget,
        )

    return _run_in_memory_pipeline(
        input_dir,
        output_path,
        stack_method=stack_method,
        sigma=sigma,
        maxiters=maxiters,
        stretch_method=stretch_method,
        reference_index=reference_index,
        recursive=recursive,
        bit_depth=bit_depth,
    )


def _run_in_memory_pipeline(
    input_dir: Path,
    output_path: Path,
    stack_method: str = "sigma",
    sigma: float = 3.0,
    maxiters: int = 5,
    stretch_method: str = "asinh",
    reference_index: int = 0,
    recursive: bool = False,
    bit_depth: int = 16,
) -> dict:
    """Run pipeline with all frames loaded into memory at once."""
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    logger.info("=== Loading frames from %s ===", input_dir)
    buckets = load_folder(input_dir, recursive=recursive)

    logger.info("=== Calibration ===")
    calibrated_lights = calibrate_lights(buckets)
    if len(calibrated_lights) == 1:
        logger.warning("Only one light frame -- alignment/stacking will be a no-op")

    logger.info("=== Alignment ===")
    reference_index = max(0, min(reference_index, len(calibrated_lights) - 1))
    aligned, align_infos = align_frames(calibrated_lights, reference_index=reference_index)

    logger.info("=== Stacking (%s) ===", stack_method)
    stacked = stack_frames(aligned, method=stack_method, sigma=sigma, maxiters=maxiters)

    logger.info("=== Noise / SNR ===")
    noise_report = compare_before_after(
        aligned[reference_index].data, stacked, n_frames=len(aligned)
    )
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
        "n_lights": len(calibrated_lights),
        "stack_method": stack_method,
        "align_infos": align_infos,
        "noise_report": noise_report,
        "output_path": str(output_path),
        "mode": "in-memory",
    }
