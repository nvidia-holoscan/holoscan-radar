# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared per-scenario run skeleton for the Python performance harness."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, TypeVar

from utils.perf_app import TimingCollector
from utils.perf_report import PerfReport, ResultRow, RunSettings
from utils.perf_stats import summarize

T = TypeVar("T")


@dataclass
class RunPlan:
    total_frames: int = 0
    override_at: int = -1
    override_spike_frames: int = 0

    def has_override(self) -> bool:
        return self.override_at >= 0


def make_run_plan(rs: RunSettings, has_override: bool, spike_frames: int = 3) -> RunPlan:
    plan = RunPlan()
    if has_override:
        plan.override_at = rs.warmup_frames + max(2, rs.measured_frames // 2)
        plan.override_spike_frames = spike_frames
        plan.total_frames = max(
            rs.warmup_frames + rs.measured_frames,
            plan.override_at + spike_frames,
        )
    else:
        plan.override_at = -1
        plan.override_spike_frames = 0
        plan.total_frames = rs.warmup_frames + rs.measured_frames
    return plan


@dataclass
class ScenarioSamples:
    setup: list[float]
    cold: list[float]
    steady: list[float]
    override_ms_best: float = math.nan


def collect_scenario_samples(
    rs: RunSettings,
    plan: RunPlan,
    run_rep: Callable[[int, RunPlan], TimingCollector],
) -> ScenarioSamples:
    samples = ScenarioSamples(setup=[], cold=[], steady=[])
    for rep in range(rs.repeats):
        collector = run_rep(rep, plan)
        samples.setup.append(collector.setup_ms())
        latencies = collector.frame_latencies_ms()
        if latencies:
            samples.cold.append(latencies[0])
        for i in range(rs.warmup_frames, len(latencies)):
            frame = i
            if not plan.has_override() or frame < plan.override_at:
                samples.steady.append(latencies[i])
            elif frame < plan.override_at + plan.override_spike_frames:
                value = latencies[i]
                if math.isnan(samples.override_ms_best) or value > samples.override_ms_best:
                    samples.override_ms_best = value
    return samples


def append_setup_row(row: ResultRow, samples: ScenarioSamples, report: PerfReport) -> None:
    setup_row = ResultRow(
        scenario_id=row.scenario_id,
        op_name=row.op_name,
        bucket="setup",
        variant=row.variant,
        precision=row.precision,
        size_id=row.size_id,
        map_shape=list(row.map_shape),
        input_shape=list(row.input_shape),
        out_batch=row.out_batch,
        guard=list(row.guard),
        training=list(row.training),
        k=row.k,
        notes=row.notes,
    )
    setup_row.setup_ms = summarize(samples.setup).median_ms
    report.results.append(setup_row)


def fill_compute_row(row: ResultRow, samples: ScenarioSamples) -> None:
    row.bucket = "compute"
    row.cold_ms = summarize(samples.cold).median_ms
    row.latency = summarize(samples.steady)
    row.override_ms = samples.override_ms_best
    if not math.isnan(samples.override_ms_best) and not math.isnan(row.latency.median_ms):
        row.override_delta_ms = samples.override_ms_best - row.latency.median_ms
