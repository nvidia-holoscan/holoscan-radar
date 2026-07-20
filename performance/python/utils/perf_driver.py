# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable CLI driver for Python perf_<operator> scripts."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Sequence
from typing import TypeVar

import cupy as cp
from utils.perf_report import PerfReport, RunSettings, SizeDef, write_json, write_text
from utils.platform import capture_platform, capture_provenance

ScenarioT = TypeVar("ScenarioT")

_DEFAULT_SIZE_CHOICES = ["xs", "sm", "md", "lg", "xl"]


def ensure_cuda_device(tool_name: str) -> bool:
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception:
        count = 0
    if count <= 0:
        print(
            f"error: no CUDA device available; {tool_name} requires a GPU",
            file=sys.stderr,
        )
        return False
    return True


def default_log_level() -> None:
    if os.environ.get("HOLOSCAN_LOG_LEVEL") is None:
        os.environ["HOLOSCAN_LOG_LEVEL"] = "ERROR"


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    size_choices: Sequence[str] | None = None,
    full_help: str = "Run the full size ladder (default).",
    full_default: bool = True,
) -> None:
    choices = list(size_choices) if size_choices is not None else list(_DEFAULT_SIZE_CHOICES)
    parser.add_argument(
        "--full",
        action=argparse.BooleanOptionalAction,
        default=full_default,
        help=full_help,
    )
    parser.add_argument(
        "--size",
        default=None,
        choices=choices,
        help="Run a single size id (CA size-sweep / overrides --full)",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated size ids, or 'all' (CA size-sweep; overrides --full)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup frames (default: 5)")
    parser.add_argument(
        "--measured",
        type=int,
        default=20,
        help="Measured frames per scenario (default: 20)",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Scenario repeats (default: 1)")
    parser.add_argument("--output", help="Write machine-readable JSON report")
    parser.add_argument("--text", help="Also write the text summary to a file")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout text summary")


def select_size_ids(
    *,
    full: bool,
    size: str | None,
    sizes: str | None,
    size_ladder_defs: list[SizeDef],
) -> list[str]:
    """Resolve which ladder ids to run for size-ladder consumers."""
    ladder_ids = [item.id for item in size_ladder_defs]
    if sizes is not None:
        if sizes.strip() == "all":
            return list(ladder_ids)
        requested = [tok.strip() for tok in sizes.split(",") if tok.strip()]
        unknown = [sid for sid in requested if sid not in ladder_ids]
        if unknown:
            raise ValueError(f"unknown size id(s): {', '.join(unknown)}")
        return requested
    if size is not None:
        if size not in ladder_ids:
            raise ValueError(f"unknown size id '{size}'")
        return [size]
    if full:
        return list(ladder_ids)
    if "sm" in ladder_ids:
        return ["sm"]
    return ladder_ids[:1]


def _write_outputs(args: argparse.Namespace, report: PerfReport) -> None:
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            write_json(handle, report)
        print(f"wrote JSON report: {args.output}", file=sys.stderr)
    if args.text:
        with open(args.text, "w", encoding="utf-8") as handle:
            write_text(handle, report)
        print(f"wrote text report: {args.text}", file=sys.stderr)
    if not args.quiet:
        write_text(sys.stdout, report)


def run_perf_main(
    *,
    tool_name: str,
    description: str,
    size_ladder_defs: list[SizeDef],
    run_scenario: Callable[[str, RunSettings, PerfReport], None],
    process_warmup: Callable[[RunSettings], None] | None = None,
    full_help: str = ("Run the full size ladder (default). Use --no-full with --size / --sizes."),
    full_default: bool = True,
) -> int:
    """Size-ladder driver (one scenario family per size id)."""
    parser = argparse.ArgumentParser(prog=tool_name, description=description)
    add_common_args(
        parser,
        size_choices=[item.id for item in size_ladder_defs],
        full_help=full_help,
        full_default=full_default,
    )
    args = parser.parse_args()

    if not ensure_cuda_device(tool_name):
        return 1
    default_log_level()

    rs = RunSettings(
        warmup_frames=args.warmup,
        measured_frames=args.measured,
        repeats=args.repeats,
    )

    try:
        size_ids = select_size_ids(
            full=args.full,
            size=args.size,
            sizes=args.sizes,
            size_ladder_defs=size_ladder_defs,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    size_map = {item.id: item for item in size_ladder_defs}
    report = PerfReport(
        provenance=capture_provenance(),
        platform=capture_platform(),
        run_settings=rs,
    )
    report.run_settings.scenarios_planned = len(size_ids)
    report.sizes = [size_map[sid] for sid in size_ids]

    if process_warmup is not None:
        process_warmup(rs)

    n = len(size_ids)
    for i, size_id in enumerate(size_ids, start=1):
        print(f"[{i}/{n}] running size={size_id}", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        run_scenario(size_id, rs, report)
        elapsed_s = time.perf_counter() - t0
        print(f"  ({elapsed_s:.3f} s)", file=sys.stderr, flush=True)
        report.run_settings.scenarios_completed = i

    _write_outputs(args, report)
    return 0


def run_perf_matrix_main(
    *,
    tool_name: str,
    description: str,
    size_ladder_defs: list[SizeDef],
    build_full_matrix: Callable[[], list[ScenarioT]],
    build_default_matrix: Callable[[], list[ScenarioT]],
    build_size_sweep: Callable[[list[str]], list[ScenarioT]],
    run_scenario: Callable[[ScenarioT, RunSettings, PerfReport], None],
    scenario_label: Callable[[ScenarioT], str],
    exercised_size_ids: Callable[[ScenarioT], str],
    process_warmup: Callable[[RunSettings], None] | None = None,
    isolate_failures: bool = False,
    full_help: str = "Full scenario matrix.",
    full_default: bool = False,
    post_run: Callable[[PerfReport], None] | None = None,
) -> int:
    """Scenario-matrix driver mirroring C++ ``perf_driver.hpp`` selection."""
    parser = argparse.ArgumentParser(prog=tool_name, description=description)
    add_common_args(
        parser,
        size_choices=[item.id for item in size_ladder_defs],
        full_help=full_help,
        full_default=full_default,
    )
    args = parser.parse_args()

    if not ensure_cuda_device(tool_name):
        return 1
    default_log_level()

    rs = RunSettings(
        warmup_frames=args.warmup,
        measured_frames=args.measured,
        repeats=args.repeats,
    )

    ladder_ids = [item.id for item in size_ladder_defs]
    try:
        if args.sizes is not None:
            size_ids = select_size_ids(
                full=False,
                size=None,
                sizes=args.sizes,
                size_ladder_defs=size_ladder_defs,
            )
            scenarios = build_size_sweep(size_ids)
        elif args.size is not None:
            if args.size not in ladder_ids:
                raise ValueError(f"unknown size id '{args.size}'")
            scenarios = build_size_sweep([args.size])
        elif args.full:
            scenarios = build_full_matrix()
        else:
            scenarios = build_default_matrix()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    used_ids: list[str] = []
    for scen in scenarios:
        sid = exercised_size_ids(scen)
        if sid in ladder_ids and sid not in used_ids:
            used_ids.append(sid)
    size_map = {item.id: item for item in size_ladder_defs}

    report = PerfReport(
        provenance=capture_provenance(),
        platform=capture_platform(),
        run_settings=rs,
    )
    report.run_settings.scenarios_planned = len(scenarios)
    report.sizes = [size_map[sid] for sid in used_ids if sid in size_map]

    if process_warmup is not None:
        process_warmup(rs)

    n = len(scenarios)
    for i, scen in enumerate(scenarios, start=1):
        label = scenario_label(scen)
        print(f"[{i}/{n}] {label}", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        try:
            run_scenario(scen, rs, report)
        except Exception as exc:  # noqa: BLE001 — mirror C++ isolate_failures
            if not isolate_failures:
                raise
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            elapsed_s = time.perf_counter() - t0
            print(f"  ({elapsed_s:.3f} s)", file=sys.stderr, flush=True)
            report.run_settings.scenarios_completed = i
            continue
        elapsed_s = time.perf_counter() - t0
        print(f"  ({elapsed_s:.3f} s)", file=sys.stderr, flush=True)
        report.run_settings.scenarios_completed = i

    if post_run is not None:
        post_run(report)

    _write_outputs(args, report)
    return 0
