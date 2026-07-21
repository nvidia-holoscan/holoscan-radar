# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for ClutterFilterOp (Python).

Measurement tool only: no timing assertions. Scenario matrices mirror
``performance/cpp/perf_clutter_filter.cpp`` (default / --full / --sizes).

Sweeps the (chirps x range_bins) size ladder, the suppression method, and a
multi-channel batch. Every compute run inlines a mid-session [live] method
override so the method-switch cost (and output chirp-count change) is
measured. Default CLI matches C++ (default matrix, not ``--full``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_PERF_DIR = os.path.dirname(os.path.abspath(__file__))
if _PERF_DIR not in sys.path:
    sys.path.insert(0, _PERF_DIR)

from utils.bootstrap import bootstrap_radar_package  # noqa: E402

bootstrap_radar_package()

from utils.perf_app import TimingCollector, now  # noqa: E402
from utils.perf_driver import run_perf_matrix_main  # noqa: E402
from utils.perf_report import PerfReport, ResultRow, RunSettings, SizeDef  # noqa: E402
from utils.perf_runner import append_setup_row  # noqa: E402
from utils.perf_runner import (
    RunPlan,
    collect_scenario_samples,
    fill_compute_row,
    make_run_plan,
)
from utils.perf_sink import PerfTerminalSinkOp  # noqa: E402
from utils.perf_source import PerfOverrideSourceOp, PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import ClutterFilterMethod, ClutterFilterOpF32  # noqa: E402

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_clutter_filter.cpp
# ---------------------------------------------------------------------------

SIZE_LADDER = [
    SizeDef("xs", [16, 128], 16 * 128, "small map; fast smoke"),
    SizeDef("sm", [32, 256], 32 * 256, "compact sensor / few chirps"),
    SizeDef("md", [64, 512], 64 * 512, "common dev size"),
    SizeDef("lg", [128, 1024], 128 * 1024, "moderate production-scale"),
    SizeDef("xl", [256, 2048], 256 * 2048, "range-heavy moderate (v1 upper bound)"),
]

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeDef:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


@dataclass
class ClutterScenario:
    id: str
    size_id: str
    num_chirps: int
    num_range_bins: int
    channels: int = 1
    method: ClutterFilterMethod = ClutterFilterMethod.MEAN_SUBTRACTION


# Short tags so the report's variant column (width 6) stays aligned.
def _method_tag(method: ClutterFilterMethod) -> str:
    if method == ClutterFilterMethod.MEAN_SUBTRACTION:
        return "msub"
    if method == ClutterFilterMethod.SINGLE_DELAY:
        return "sdel"
    if method == ClutterFilterMethod.DOUBLE_DELAY:
        return "ddel"
    return "none"


# Flip to a different valid method so a live override always changes the
# kernel path (and the output chirp count). NONE is invalid for this operator.
def _flip_method(method: ClutterFilterMethod) -> ClutterFilterMethod:
    if method == ClutterFilterMethod.MEAN_SUBTRACTION:
        return ClutterFilterMethod.SINGLE_DELAY
    return ClutterFilterMethod.MEAN_SUBTRACTION


def _profiles_shape(scen: ClutterScenario) -> tuple[int, ...]:
    if scen.channels > 1:
        return (scen.channels, scen.num_chirps, scen.num_range_bins)
    return (scen.num_chirps, scen.num_range_bins)


def make_scenario(method: ClutterFilterMethod, size: SizeDef, channels: int = 1) -> ClutterScenario:
    num_chirps, num_range_bins = size.shape
    sid = f"clutter_{_method_tag(method)}_{size.id}"
    if channels > 1:
        sid += f"_ch{channels}"
    sid += "_compute"
    return ClutterScenario(
        id=sid,
        size_id=size.id,
        num_chirps=num_chirps,
        num_range_bins=num_range_bins,
        channels=channels,
        method=method,
    )


def build_default_matrix() -> list[ClutterScenario]:
    out: list[ClutterScenario] = []
    for sid in ("xs", "md", "lg", "xl"):
        out.append(make_scenario(ClutterFilterMethod.MEAN_SUBTRACTION, _must_find_size(sid)))
    md = _must_find_size("md")
    out.append(make_scenario(ClutterFilterMethod.SINGLE_DELAY, md))
    out.append(make_scenario(ClutterFilterMethod.DOUBLE_DELAY, md))
    out.append(make_scenario(ClutterFilterMethod.MEAN_SUBTRACTION, md, 4))
    return out


def build_size_sweep(size_ids: list[str]) -> list[ClutterScenario]:
    out: list[ClutterScenario] = []
    for sid in size_ids:
        out.append(make_scenario(ClutterFilterMethod.MEAN_SUBTRACTION, _must_find_size(sid)))
    return out


def build_full_matrix() -> list[ClutterScenario]:
    out: list[ClutterScenario] = []
    methods = (
        ClutterFilterMethod.MEAN_SUBTRACTION,
        ClutterFilterMethod.SINGLE_DELAY,
        ClutterFilterMethod.DOUBLE_DELAY,
    )
    for size in SIZE_LADDER:
        for method in methods:
            out.append(make_scenario(method, size))
    for sid in ("md", "lg"):
        size = _must_find_size(sid)
        for channels in (4, 16):
            out.append(make_scenario(ClutterFilterMethod.MEAN_SUBTRACTION, size, channels))
    return out


def _base_row(scen: ClutterScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="ClutterFilterOp",
        variant=_method_tag(scen.method),
        size_id=scen.size_id,
        map_shape=[scen.num_chirps, scen.num_range_bins],
        input_shape=list(_profiles_shape(scen)),
        out_batch=scen.channels,
        k=scen.num_chirps,
        notes="live_override=method",
    )


def _fill_throughput(row: ResultRow, scen: ClutterScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    batch = float(scen.channels)
    chirps = float(scen.num_chirps)
    bins = float(scen.num_range_bins)
    row.cells_per_second = batch * chirps * bins / sec  # complex cells/s
    row.slices_per_second = batch * bins / sec  # slow-time columns/s
    csz = 8.0  # complex64
    # Output chirp count varies by method (msub: N, sdel: N-1, ddel: N-2); use
    # N for a stable order-of-magnitude bandwidth estimate.
    bytes_total = 2.0 * batch * chirps * bins * csz  # in + out
    row.estimated_effective_gbps = bytes_total / sec / 1e9


def _run_graph(
    *,
    scen: ClutterScenario,
    collector: TimingCollector,
    frame: dict,
    override_method: ClutterFilterMethod,
    override_at: int,
    total_frames: int,
) -> None:
    class ClutterPerfApp(Application):
        def compose(self) -> None:
            op = ClutterFilterOpF32(self, method=scen.method, name="clutter")
            src = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="range_profiles",
                frame=frame,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("range_profiles", "range_profiles")})

            sink = PerfTerminalSinkOp(self, collector=collector, name="out_sink")
            self.add_flow(op, sink, {("range_profiles", "input")})

            override = PerfOverrideSourceOp(
                self,
                CountCondition(self, count=total_frames),
                target_tick=override_at,
                value=override_method,
                name="ovr",
            )
            self.add_flow(override, op, {("output", "method")})

    ClutterPerfApp().run()


def run_scenario(scen: ClutterScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        frame = single_map(make_iq_cube_device(_profiles_shape(scen)), "range_profiles")
        collector.t_construct = now()
        _run_graph(
            scen=scen,
            collector=collector,
            frame=frame,
            override_method=_flip_method(scen.method),
            override_at=run_plan.override_at,
            total_frames=run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(rs, plan, run_rep)
    base = _base_row(scen)
    append_setup_row(base, samples, report)
    compute = _base_row(scen)
    fill_compute_row(compute, samples)
    _fill_throughput(compute, scen)
    compute.notes = (
        "cold_ms is first-frame latency; median/p90 are pre-override steady "
        "state; override_ms is peak latency over the 3-frame live method "
        "switch (changes the kernel path and output chirp count); cells/s is "
        "complex cells/s"
    )
    report.results.append(compute)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(
        make_scenario(ClutterFilterMethod.MEAN_SUBTRACTION, _must_find_size("xs")),
        warm,
        discarded,
    )


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_clutter_filter",
        description="ClutterFilterOp performance characterization (Python)",
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        full_default=False,
        full_help="Full matrix (methods x sizes x channel batch).",
    )


if __name__ == "__main__":
    sys.exit(main())
