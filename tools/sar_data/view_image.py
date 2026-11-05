#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.24", "matplotlib>=3.8"]
# ///

"""Save a log-magnitude PNG from a SAR NumPy image; optionally compare or show.

Run from the repository root: uv run tools/sar_data/view_image.py image.npy
Plotting dependencies are isolated by uv; applications do not import this tool.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_magnitude(path: Path) -> np.ndarray:
    """Load numeric image samples without pickle and preserve amplitude range."""
    image = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(image, np.ndarray):
        image.close()
        raise ValueError(f"{path}: expected a .npy array, not an archive")
    if image.ndim != 2 or not image.size:
        raise ValueError(f"{path}: expected a nonempty 2-D image, got shape {image.shape}")
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"{path}: expected real or complex numeric samples, got {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: image contains non-finite samples")
    # Promote before abs: integer minima and large complex64 values otherwise
    # overflow despite containing finite input samples.
    dtype = np.complex128 if np.iscomplexobj(image) else np.float64
    magnitude = np.abs(image.astype(dtype))
    if not np.isfinite(magnitude).all():
        raise ValueError(f"{path}: magnitude exceeds the supported floating-point range")
    if magnitude.max() <= 0:
        raise ValueError(f"{path}: image is all zero; there is no signal to display")
    return magnitude


def decibels(magnitude: np.ndarray, reference: float, range_db: float) -> np.ndarray:
    """Use one peak reference across panels; zeros map to the display floor."""
    log = np.full(magnitude.shape, -np.inf, dtype=np.float64)
    np.log10(magnitude, out=log, where=magnitude > 0)
    return np.clip(20 * (log - np.log10(reference)), -range_db, 0)


def render(paths: list[Path], output: Path, range_db: float, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    magnitudes = [load_magnitude(path) for path in paths]
    if len({image.shape for image in magnitudes}) != 1:
        raise ValueError(
            "comparison requires matching array shapes; no image registration is performed"
        )
    reference = max(float(image.max()) for image in magnitudes)
    figure, axes = plt.subplots(
        1,
        len(paths),
        figsize=(6 * len(paths), 6),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    try:
        for axis, path, magnitude in zip(axes[0], paths, magnitudes, strict=True):
            plotted = axis.imshow(
                decibels(magnitude, reference, range_db),
                cmap="gray",
                vmin=-range_db,
                vmax=0,
                origin="upper",
                interpolation="nearest",
            )
            axis.set_title(path.name)
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")
        label = (
            "dB relative to shared image peak" if len(paths) > 1 else "dB relative to image peak"
        )
        figure.colorbar(plotted, ax=list(axes[0]), label=label)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150)
        print(f"Saved {output.resolve()}", flush=True)
        if show:
            if matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
                raise RuntimeError(
                    "PNG saved, but --show needs an interactive Matplotlib backend and a display; "
                    "open the PNG on your desktop instead"
                )
            plt.show()
    finally:
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="saved 2-D .npy sample arrays")
    parser.add_argument(
        "--compare", action="store_true", help="compare images on one intensity scale"
    )
    parser.add_argument("--show", action="store_true", help="also open an interactive plot")
    parser.add_argument(
        "--range-db", type=float, default=60, help="display range below peak (default: 60)"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="PNG path (default: beside the first input)"
    )
    args = parser.parse_args(argv)
    if args.compare != (len(args.images) > 1):
        parser.error("use one image, or pass --compare with two or more images")
    if not np.isfinite(args.range_db) or args.range_db <= 0:
        parser.error("--range-db must be finite and greater than zero")
    first = args.images[0]
    output = args.output or first.with_name(
        first.stem + ("_comparison.png" if args.compare else ".png")
    )
    if output.suffix.lower() != ".png":
        parser.error("--output must end in .png; input arrays are never overwritten")
    if any(output.resolve() == image.resolve() for image in args.images):
        parser.error("output must differ from every input image")
    try:
        render(args.images, output, args.range_db, args.show)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"view_image: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
