# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for SnapshotGatherOp (Python).

The sparse-path bridge that gathers a per-detection channel-vector
neighborhood out of per-channel RD maps. Sweeps channel count, RD-map size,
detection count, and snapshot window, and inlines a mid-session ``[live]``
``snapshot_window`` override (the interesting cost: the window grows, so
the output re-shapes and the gather re-plans). Measurement tool only: no
timing assertions.

Scenario matrix mirrors ``performance/cpp/perf_snapshot_gather.cpp``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

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
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.perf_source import PerfOverrideSourceOp  # noqa: E402
from utils.tensor_factory import make_detection_idx_device  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application, Operator, OperatorSpec  # noqa: E402
from holoscan.radar import SnapshotEdgeMode, SnapshotGatherOpF32  # noqa: E402

# v1 characterizes single precision; complex64 sample size / int64 index size.
_COMPLEX_BYTES = 8.0
_INT64_BYTES = 8.0

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_snapshot_gather.cpp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeSpec:
    id: str
    channels: int
    doppler: int
    range: int
    detections: int
    role: str


SIZE_LADDER = [
    SizeSpec("xs", 4, 64, 256, 64, "small map; fast smoke"),
    SizeSpec("sm", 8, 128, 512, 256, "compact SIMO array"),
    SizeSpec("md", 16, 256, 512, 1024, "common dev size"),
    SizeSpec("lg", 32, 512, 1024, 4096, "dense virtual array"),
    SizeSpec("xl", 64, 512, 2048, 16384, "upper characterization bound"),
]
_SIZE_BY_ID = {s.id: s for s in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeSpec:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


# A snapshot window is a (doppler, range) half-extent; the gather produces
# (2*wd + 1) * (2*wr + 1) snapshots per channel per detection.
@dataclass(frozen=True)
class Window:
    wd: int = 0
    wr: int = 0


def snapshots_for(w: Window) -> int:
    return (2 * w.wd + 1) * (2 * w.wr + 1)


def window_tag(w: Window) -> str:
    return f"w{w.wd}x{w.wr}"


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------


@dataclass
class SnapScenario:
    id: str
    size_id: str
    channels: int
    doppler: int
    range: int
    detections: int
    window: Window
    # Filled per run by run_scenario() for inline live-override measurement:
    # the window grows one cell on each axis, forcing an output re-shape +
    # re-plan.
    override_at: int = 0
    override_window: Window = Window(0, 0)


def make_scenario(s: SizeSpec, w: Window) -> SnapScenario:
    sid = f"snap_gather_{s.id}_c{s.channels}_n{s.detections}_{window_tag(w)}"
    return SnapScenario(
        id=sid,
        size_id=s.id,
        channels=s.channels,
        doppler=s.doppler,
        range=s.range,
        detections=s.detections,
        window=w,
    )


def build_default_matrix() -> list[SnapScenario]:
    out: list[SnapScenario] = []
    md = _must_find_size("md")
    for w in (Window(0, 0), Window(1, 1), Window(2, 2)):
        out.append(make_scenario(md, w))
    for sid in ("xs", "sm", "md", "lg"):
        out.append(make_scenario(_must_find_size(sid), Window(1, 1)))
    return out


def build_size_sweep(size_ids: list[str]) -> list[SnapScenario]:
    out: list[SnapScenario] = []
    for sid in size_ids:
        spec = _SIZE_BY_ID.get(sid)
        if spec is None:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(make_scenario(spec, Window(1, 1)))
    return out


def build_full_matrix() -> list[SnapScenario]:
    out: list[SnapScenario] = []
    for s in SIZE_LADDER:
        for w in (Window(0, 0), Window(1, 1), Window(2, 2)):
            out.append(make_scenario(s, w))
    return out


def _base_row(scen: SnapScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="SnapshotGatherOp",
        variant=window_tag(scen.window),
        size_id=scen.size_id,
        map_shape=[scen.doppler, scen.range],
        input_shape=[scen.detections, 2],
        out_batch=scen.detections,
        k=scen.channels,
    )


def _fill_throughput(row: ResultRow, scen: SnapScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    n = float(scen.detections)
    snaps = float(snapshots_for(scen.window))
    gathered = n * float(scen.channels) * snaps

    row.cells_per_second = n / sec  # detections/s (sparse axis)
    row.slices_per_second = n * snaps / sec  # gathered snapshots/s

    # Effective traffic: read each gathered complex sample from the RD maps,
    # write it to the snapshot matrix, write the bool validity flag, and read
    # the detection index rows.
    idx_bytes = n * 2.0 * _INT64_BYTES
    valid_bytes = n * snaps  # bool (1 byte/elem)
    row.estimated_effective_gbps = (
        (2.0 * gathered * _COMPLEX_BYTES + valid_bytes + idx_bytes) / sec / 1e9
    )


# Multi-port source: emits the same preallocated TensorMaps every tick so
# measured latency is dominated by SnapshotGatherOp, not H2D.
class SnapGatherPerfSourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        rd_maps: dict,
        detection_idx: dict,
        collector: TimingCollector,
        **kwargs,
    ):
        self._rd_maps = rd_maps
        self._detection_idx = detection_idx
        self._collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("rd_maps")
        spec.output("detection_idx")

    def compute(self, op_input, op_output, context) -> None:
        if not self._collector.first_tick_set:
            self._collector.t_first_tick = now()
            self._collector.first_tick_set = True
        op_output.emit(dict(self._rd_maps), "rd_maps")
        op_output.emit(dict(self._detection_idx), "detection_idx")
        self._collector.starts.append(now())


def _run_graph(
    *,
    scen: SnapScenario,
    collector: TimingCollector,
    rd_maps: dict,
    detection_idx: dict,
    total_frames: int,
) -> None:
    class SnapGatherPerfApp(Application):
        def compose(self) -> None:
            op = SnapshotGatherOpF32(
                self,
                snapshot_window=[scen.window.wd, scen.window.wr],
                doppler_edge=SnapshotEdgeMode.WRAP,
                range_edge=SnapshotEdgeMode.SHIFT,
                name="gather",
            )

            src = SnapGatherPerfSourceOp(
                self,
                CountCondition(self, count=total_frames),
                rd_maps=rd_maps,
                detection_idx=detection_idx,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("rd_maps", "rd_maps")})
            self.add_flow(src, op, {("detection_idx", "detection_idx")})

            # Sparse output goes to the timed terminal sink; is_valid (bool,
            # same transport caveat as CFAR detection_mask) drains through a
            # null sink.
            term = PerfTerminalSinkOp(self, collector=collector, name="snapshot_sink")
            self.add_flow(op, term, {("snapshot", "input")})

            valid_drain = make_null_sink(self, "valid_drain")
            self.add_flow(op, valid_drain, {("is_valid", "input")})

            override = PerfOverrideSourceOp(
                self,
                CountCondition(self, count=total_frames),
                target_tick=scen.override_at,
                value=[scen.override_window.wd, scen.override_window.wr],
                name="ovr",
            )
            self.add_flow(override, op, {("output", "snapshot_window")})

    SnapGatherPerfApp().run()


def run_scenario(scen: SnapScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        run_scen = replace(
            scen,
            override_at=run_plan.override_at,
            override_window=Window(scen.window.wd + 1, scen.window.wr + 1),
        )

        collector = TimingCollector()
        rd_maps_t = make_iq_cube_device([scen.channels, scen.doppler, scen.range])
        idx_t = make_detection_idx_device(scen.detections, scen.doppler, scen.range)
        rd_maps = single_map(rd_maps_t, "rd_maps")
        detection_idx = single_map(idx_t, "detection_idx")

        collector.t_construct = now()
        _run_graph(
            scen=run_scen,
            collector=collector,
            rd_maps=rd_maps,
            detection_idx=detection_idx,
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
        "state; override_ms is peak latency over the 3-frame live-"
        "snapshot_window (window +1/axis) window; cells/s is detections/s"
    )
    report.results.append(compute)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario(_must_find_size("xs"), Window(1, 1)), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_snapshot_gather",
        description="SnapshotGatherOp performance characterization (Python)",
        size_ladder_defs=[
            SizeDef(s.id, [s.doppler, s.range], s.doppler * s.range, s.role) for s in SIZE_LADDER
        ],
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        full_default=False,
        full_help="Full matrix (sizes x windows {0x0, 1x1, 2x2})",
    )


if __name__ == "__main__":
    sys.exit(main())
