#!/usr/bin/env python
"""CLI for the astrophotography stacking pipeline.

Example:
    python stack.py --input ./my_photos --output result.tiff --stack sigma
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from astro_stack.pipeline import DEFAULT_IN_MEMORY_BUDGET_MB, run_pipeline
from astro_stack.postprocess import STRETCH_METHODS
from astro_stack.stacking import STACK_METHODS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate, align, and stack astrophotography light frames."
    )
    parser.add_argument("--input", required=True, type=Path, help="Folder of input images")
    parser.add_argument("--output", required=True, type=Path, help="Output image path (.tiff/.png/.jpg)")
    parser.add_argument("--stack", choices=STACK_METHODS, default="sigma", help="Combine method (default: sigma)")
    parser.add_argument("--sigma-thresh", type=float, default=3.0, help="Sigma-clip threshold (default: 3.0)")
    parser.add_argument("--sigma-maxiters", type=int, default=5, help="Sigma-clip max iterations (default: 5)")
    parser.add_argument("--stretch", choices=STRETCH_METHODS, default="asinh", help="Post-process stretch (default: asinh)")
    parser.add_argument("--reference", type=int, default=0, help="Index of the reference light frame (default: 0)")
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=16, help="Output bit depth for TIFF (default: 16)")
    parser.add_argument(
        "--memory-budget-mb", type=float, default=DEFAULT_IN_MEMORY_BUDGET_MB,
        help=(
            "If the input folder's estimated decoded size exceeds this, switch "
            "to a slower but memory-bounded streaming mode that processes one "
            f"frame at a time via temp files on disk (default: {DEFAULT_IN_MEMORY_BUDGET_MB} MB). "
            "Pass 0 to force streaming mode always."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        summary = run_pipeline(
            input_dir=args.input,
            output_path=args.output,
            stack_method=args.stack,
            sigma=args.sigma_thresh,
            maxiters=args.sigma_maxiters,
            stretch_method=args.stretch,
            reference_index=args.reference,
            recursive=args.recursive,
            bit_depth=args.bit_depth,
            memory_budget_mb=args.memory_budget_mb,
        )
    except Exception as exc:  # noqa: BLE001 - top-level CLI error boundary
        logging.getLogger("astro_stack").error("Pipeline failed: %s", exc)
        return 1

    noise = summary["noise_report"]
    print(f"\nStacked {summary['n_lights']} light frame(s) with '{summary['stack_method']}' method")
    print(
        f"Noise:  {noise['before'].noise_std:.6f} -> {noise['after'].noise_std:.6f} "
        f"({noise['observed_noise_reduction']:.2f}x reduction, "
        f"theory sqrt(N) = {noise['theoretical_noise_reduction_sqrt_n']:.2f}x)"
    )
    print(
        f"SNR:    {noise['before'].snr:.2f} -> {noise['after'].snr:.2f} "
        f"({noise['snr_improvement']:.2f}x improvement)"
    )
    print(f"Mode:   {summary['mode']}")
    print(f"Saved -> {summary['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
