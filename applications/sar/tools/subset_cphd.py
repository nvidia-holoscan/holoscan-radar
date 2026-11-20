#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subset a CPHD file: pulse window and/or a contiguous sample subband.

Windows are a recipe/CLI knob, resolved against the source header. Give an
explicit [start, stop), or a count / bandwidth / duration placed at origin
(center, start, or end). Omit an axis to keep it whole.

uv run applications/sar/tools/subset_cphd.py applications/sar/tools/config/houston_1m.yaml -o data/sar
python3 subset_cphd.py --source in.cphd --sample-count 128 --pulse-count 40 -o out.cphd
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["lxml>=6.1", "numpy>=1.24", "pyyaml>=6.0", "sarkit>=1.10"]
# ///

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sar_data.recipe import validate_recipe_name  # noqa: E402
from sar_data.subset import (  # noqa: E402
    DEFAULT_IO_CHUNK_BYTES,
    AxisWindow,
    load_subset_recipe,
    parse_axis_window,
    subset_cphd,
)


def _output_path(output: Path, name: str) -> Path:
    name = validate_recipe_name(name)
    if output.suffix.lower() == ".cphd":
        return output
    return output / f"{name}.cphd"


def _cli_window(
    *,
    axis: str,
    span: str | None,
    count: int | None,
    origin: str | None,
    bandwidth_hz: float | None,
    duration_s: float | None,
) -> AxisWindow | None:
    """Build a window from CLI flags, or None if the recipe should stand."""
    flags = (span is not None, count is not None, bandwidth_hz is not None, duration_s is not None)
    if sum(flags) > 1:
        raise SystemExit(
            f"only one of --{axis}, --{axis.rstrip('s')}-count, "
            f"--bandwidth-hz / --duration-s may be set for {axis}"
        )
    if not any(flags) and origin is None:
        return None
    if span is not None:
        return parse_axis_window(span, axis=axis)
    if not any(flags):
        raise SystemExit(f"--{axis}-origin needs --{axis}-count (or bandwidth/duration)")
    mapping: dict[str, float | int | str] = {"origin": origin or "center"}
    if count is not None:
        mapping["count"] = count
    if bandwidth_hz is not None:
        mapping["bandwidth_hz"] = bandwidth_hz
    if duration_s is not None:
        mapping["duration_s"] = duration_s
    return parse_axis_window(mapping, axis=axis)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "recipe",
        nargs="?",
        type=Path,
        help="subset recipe YAML (source + window knobs)",
    )
    parser.add_argument(
        "--allow-source-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="explicitly trust an additional HTTP(S) server (scheme://host[:port]); repeatable",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="local CPHD path or HTTPS URL (overrides the recipe source)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        help="output .cphd, or a directory to write <name>.cphd into",
    )
    parser.add_argument(
        "--pulses",
        type=str,
        help="half-open pulse window start:stop (overrides the recipe)",
    )
    parser.add_argument(
        "--pulse-count",
        type=int,
        help="keep this many pulses (placed by --pulse-origin, default center)",
    )
    parser.add_argument(
        "--pulse-origin",
        choices=("start", "center", "end"),
        help="where to place --pulse-count / --duration-s",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        help="pulse window length in seconds (uses median PRI from the file)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        help="half-open sample window start:stop (overrides the recipe)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        help="keep this many samples (placed by --sample-origin, default center)",
    )
    parser.add_argument(
        "--sample-origin",
        choices=("start", "center", "end"),
        help="where to place --sample-count / --bandwidth-hz",
    )
    parser.add_argument(
        "--bandwidth-hz",
        type=float,
        help="sample-window bandwidth; count = round(bandwidth / SCSS)",
    )
    parser.add_argument(
        "--io-chunk-bytes",
        type=int,
        default=None,
        help="signal staging/transfer bytes (default 67108864; recipe may set)",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=None,
        help="parallel keep-alive Range connections (default 128, recipe may set)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="restart a partial download and rewrite even an existing matching destination",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse header/XML and print resolved windows; do not fetch signal or write",
    )
    parser.add_argument("--progress-label", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.io_chunk_bytes is not None and (
        args.io_chunk_bytes < 8 or args.io_chunk_bytes % 8 or args.io_chunk_bytes > sys.maxsize
    ):
        parser.error("--io-chunk-bytes must be at least 8, a multiple of 8, and addressable")

    if args.recipe is None and args.source is None:
        parser.error("a recipe or --source is required")

    try:
        pulse_override = _cli_window(
            axis="pulses",
            span=args.pulses,
            count=args.pulse_count,
            origin=args.pulse_origin,
            bandwidth_hz=None,
            duration_s=args.duration_s,
        )
        sample_override = _cli_window(
            axis="samples",
            span=args.samples,
            count=args.sample_count,
            origin=args.sample_origin,
            bandwidth_hz=args.bandwidth_hz,
            duration_s=None,
        )
    except ValueError as exc:
        parser.error(str(exc))

    name = "subset"
    source = args.source
    pulses = pulse_override
    samples = sample_override
    connections = args.connections if args.connections is not None else 128
    io_chunk_bytes = (
        args.io_chunk_bytes if args.io_chunk_bytes is not None else DEFAULT_IO_CHUNK_BYTES
    )

    if args.recipe is not None:
        recipe = load_subset_recipe(args.recipe)
        name = recipe.name
        if source is None:
            source = recipe.source
        if pulses is None:
            pulses = recipe.pulses
        if samples is None:
            samples = recipe.samples
        if args.connections is None:
            connections = recipe.connections
        if args.io_chunk_bytes is None:
            io_chunk_bytes = recipe.io_chunk_bytes

    dest = _output_path(args.output, name)
    result = subset_cphd(
        source,
        dest,
        pulses=pulses,
        samples=samples,
        connections=connections,
        io_chunk_bytes=io_chunk_bytes,
        force=args.force,
        dry_run=args.dry_run,
        progress=True,
        progress_label=args.progress_label or str(dest),
        allowed_source_origins=tuple(args.allow_source_origin),
    )
    if result.dry_run:
        print(f"dry-run: would write {dest} ({result.num_vectors} x {result.num_samples})")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
