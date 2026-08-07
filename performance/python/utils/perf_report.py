# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified performance report for the Python harness."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import IO, Any

from utils.perf_stats import LatencyStats
from utils.platform import Platform, Provenance


@dataclass
class RunSettings:
    warmup_frames: int = 5
    measured_frames: int = 20
    repeats: int = 1
    sync_policy: str = "device_sync_at_terminal_sink"
    stop_after_s: float = 0.0
    stopped_early: bool = False
    scenarios_planned: int = 0
    scenarios_completed: int = 0


@dataclass
class SizeDef:
    id: str
    shape: list[int]
    cells: int
    role: str


@dataclass
class ResultRow:
    scenario_id: str
    op_name: str = "CFARDetectorOp"
    bucket: str = ""
    variant: str = ""
    precision: str = "float"
    size_id: str = ""
    map_shape: list[int] = field(default_factory=list)
    input_shape: list[int] = field(default_factory=list)
    out_batch: int = 1
    guard: list[int] = field(default_factory=list)
    training: list[int] = field(default_factory=list)
    k: int = 1
    setup_ms: float = math.nan
    cold_ms: float = math.nan
    latency: LatencyStats = field(default_factory=LatencyStats)
    override_ms: float = math.nan
    override_delta_ms: float = math.nan
    cells_per_second: float = math.nan
    slices_per_second: float = math.nan
    estimated_effective_gbps: float = math.nan
    status: str = "ok"
    notes: str = ""
    calibration_pair: str = ""
    calibration_mode: str = ""


@dataclass
class AlphaCalibrationRow:
    pair_id: str = ""
    variant: str = ""
    size_id: str = ""
    k: int = 1
    slow_cold_ms: float = math.nan
    fast_cold_ms: float = math.nan
    cold_speedup: float = math.nan
    slow_ovr_ms: float = math.nan
    fast_ovr_ms: float = math.nan
    ovr_speedup: float = math.nan


@dataclass
class PerfReport:
    provenance: Provenance = field(default_factory=Provenance)
    platform: Platform = field(default_factory=Platform)
    run_settings: RunSettings = field(default_factory=RunSettings)
    sizes: list[SizeDef] = field(default_factory=list)
    results: list[ResultRow] = field(default_factory=list)
    alpha_calibration: list[AlphaCalibrationRow] = field(default_factory=list)


def _is_num(value: float) -> bool:
    return not math.isnan(value)


def _shape_text(shape: list[int]) -> str:
    return "x".join(str(d) for d in shape)


def _col(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) == width:
        return text
    if len(text) < width:
        return text + " " * (width - len(text))
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _fixed(value: float, prec: int) -> str:
    if not _is_num(value):
        return "-"
    return f"{value:.{prec}f}"


def _sci(value: float) -> str:
    if not _is_num(value):
        return "-"
    return f"{value:.3g}"


def _shape_column_width(report: PerfReport, min_width: int = 14) -> int:
    width = min_width
    for size in report.sizes:
        width = max(width, len(_shape_text(size.shape)) + 2)
    for row in report.results:
        width = max(width, len(_shape_text(row.input_shape)) + 2)
    return min(width, 26)


def _row_to_json(row: ResultRow) -> dict[str, Any]:
    return {
        "scenario_id": row.scenario_id,
        "operator": row.op_name,
        "bucket": row.bucket,
        "variant": row.variant,
        "precision": row.precision,
        "size_id": row.size_id,
        "map_shape": row.map_shape,
        "input_shape": row.input_shape,
        "out_batch": row.out_batch,
        "guard": row.guard,
        "training": row.training,
        "k": row.k,
        "setup_ms": row.setup_ms if _is_num(row.setup_ms) else None,
        "cold_ms": row.cold_ms if _is_num(row.cold_ms) else None,
        "min_ms": row.latency.min_ms if _is_num(row.latency.min_ms) else None,
        "median_ms": row.latency.median_ms if _is_num(row.latency.median_ms) else None,
        "p90_ms": row.latency.p90_ms if _is_num(row.latency.p90_ms) else None,
        "p99_ms": row.latency.p99_ms if _is_num(row.latency.p99_ms) else None,
        "max_ms": row.latency.max_ms if _is_num(row.latency.max_ms) else None,
        "override_ms": row.override_ms if _is_num(row.override_ms) else None,
        "override_delta_ms": (row.override_delta_ms if _is_num(row.override_delta_ms) else None),
        "cells_per_second": (row.cells_per_second if _is_num(row.cells_per_second) else None),
        "slices_per_second": (row.slices_per_second if _is_num(row.slices_per_second) else None),
        "estimated_effective_gbps": (
            row.estimated_effective_gbps if _is_num(row.estimated_effective_gbps) else None
        ),
        "status": row.status,
        "notes": row.notes,
        "calibration_pair": row.calibration_pair,
        "calibration_mode": row.calibration_mode,
    }


def write_json(stream: IO[str], report: PerfReport) -> None:
    payload = {
        "schema_version": 2,
        "provenance": {
            "git_commit": report.provenance.git_commit,
            "git_dirty": report.provenance.git_dirty,
            "timestamp_utc": report.provenance.timestamp_utc,
        },
        "platform": {
            "gpu_name": report.platform.gpu_name,
            "gpu_driver": report.platform.gpu_driver,
            "cuda_runtime": report.platform.cuda_runtime,
            "holoscan_version": report.platform.holoscan_version,
            "cpu_model": report.platform.cpu_model,
            "os": report.platform.os_name,
        },
        "run_settings": {
            "warmup_frames": report.run_settings.warmup_frames,
            "measured_frames": report.run_settings.measured_frames,
            "repeats": report.run_settings.repeats,
            "sync_policy": report.run_settings.sync_policy,
            "stop_after_s": report.run_settings.stop_after_s,
            "stopped_early": report.run_settings.stopped_early,
            "scenarios_planned": report.run_settings.scenarios_planned,
            "scenarios_completed": report.run_settings.scenarios_completed,
        },
        "sizes": [
            {
                "id": size.id,
                "shape": size.shape,
                "cells": size.cells,
                "role": size.role,
            }
            for size in report.sizes
        ],
        "results": [_row_to_json(row) for row in report.results],
        "alpha_calibration": [
            {
                "pair_id": row.pair_id,
                "variant": row.variant,
                "size_id": row.size_id,
                "k": row.k,
                "slow_cold_ms": row.slow_cold_ms if _is_num(row.slow_cold_ms) else None,
                "fast_cold_ms": row.fast_cold_ms if _is_num(row.fast_cold_ms) else None,
                "cold_speedup": row.cold_speedup if _is_num(row.cold_speedup) else None,
                "slow_ovr_ms": row.slow_ovr_ms if _is_num(row.slow_ovr_ms) else None,
                "fast_ovr_ms": row.fast_ovr_ms if _is_num(row.fast_ovr_ms) else None,
                "ovr_speedup": row.ovr_speedup if _is_num(row.ovr_speedup) else None,
            }
            for row in report.alpha_calibration
        ],
    }
    json.dump(payload, stream, indent=2)
    stream.write("\n")


def write_text(stream: IO[str], report: PerfReport) -> None:
    stream.write("=== holoscan-radar performance report ===\n")
    dirty = " (dirty)" if report.provenance.git_dirty else " (clean)"
    stream.write(
        f"commit: {report.provenance.git_commit}{dirty}  " f"{report.provenance.timestamp_utc}\n"
    )
    stream.write(
        f"platform: {report.platform.gpu_name} | driver "
        f"{report.platform.gpu_driver} | CUDA {report.platform.cuda_runtime} | "
        f"Holoscan {report.platform.holoscan_version}\n"
    )
    stream.write(f"          {report.platform.cpu_model} | {report.platform.os_name}\n")
    stream.write(
        f"run: warmup={report.run_settings.warmup_frames} "
        f"measured={report.run_settings.measured_frames} "
        f"repeats={report.run_settings.repeats}\n"
    )
    stream.write("note: latencies are valid only on the machine/GPU they were captured on.\n")

    shape_w = _shape_column_width(report)
    if report.sizes:
        stream.write("\n--- sizes ---\n")
        stream.write(_col("id", 6) + _col("shape (DxR)", shape_w) + _col("cells", 10) + "role\n")
        for size in report.sizes:
            stream.write(
                _col(size.id, 6)
                + _col(_shape_text(size.shape), shape_w)
                + _col(str(size.cells), 10)
                + size.role
                + "\n"
            )

    stream.write("\n--- setup ---\n")
    stream.write(_col("scenario", 40) + _col("size", 6) + _col("setup_ms", 10) + "\n")
    for row in report.results:
        if row.bucket != "setup":
            continue
        stream.write(
            _col(row.scenario_id, 40)
            + _col(row.size_id, 6)
            + _col(_fixed(row.setup_ms, 2), 10)
            + "\n"
        )

    stream.write("\n--- compute ---\n")
    stream.write(
        _col("scenario", 40)
        + _col("size", 6)
        + _col("shape", shape_w)
        + _col("cold_ms", 10)
        + _col("median_ms", 10)
        + _col("p90_ms", 10)
        + _col("ovr_ms", 10)
        + _col("ovr_d_ms", 10)
        + _col("cells/s", 11)
        + _col("GB/s", 8)
        + "\n"
    )
    for row in report.results:
        if row.bucket != "compute":
            continue
        stream.write(
            _col(row.scenario_id, 40)
            + _col(row.size_id, 6)
            + _col(_shape_text(row.input_shape), shape_w)
            + _col(_fixed(row.cold_ms, 3), 10)
            + _col(_fixed(row.latency.median_ms, 3), 10)
            + _col(_fixed(row.latency.p90_ms, 3), 10)
            + _col(_fixed(row.override_ms, 3), 10)
            + _col(_fixed(row.override_delta_ms, 3), 10)
            + _col(_sci(row.cells_per_second), 11)
            + _col(_fixed(row.estimated_effective_gbps, 1), 8)
            + "\n"
        )

    if report.alpha_calibration:
        stream.write("\n--- alpha_lut calibration (pfa vs alpha_lut) ---\n")
        stream.write(
            _col("pair", 14)
            + _col("var", 4)
            + _col("size", 6)
            + _col("k", 4)
            + _col("slow_c", 10)
            + _col("fast_c", 10)
            + _col("c_x", 8)
            + _col("slow_o", 10)
            + _col("fast_o", 10)
            + _col("o_x", 8)
            + "\n"
        )
        for row in report.alpha_calibration:
            stream.write(
                _col(row.pair_id, 14)
                + _col(row.variant, 4)
                + _col(row.size_id, 6)
                + _col(str(row.k), 4)
                + _col(_fixed(row.slow_cold_ms, 3), 10)
                + _col(_fixed(row.fast_cold_ms, 3), 10)
                + _col(_fixed(row.cold_speedup, 2), 8)
                + _col(_fixed(row.slow_ovr_ms, 3), 10)
                + _col(_fixed(row.fast_ovr_ms, 3), 10)
                + _col(_fixed(row.ovr_speedup, 2), 8)
                + "\n"
            )
    stream.write("\n")
