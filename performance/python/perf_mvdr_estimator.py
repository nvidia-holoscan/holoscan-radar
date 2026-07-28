# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for MVDREstimatorOp (Python).

Adaptive (Capon) sparse-path DOA. Per item it forms a snapshot covariance,
diagonal-loads and inverts it (batched), then scans the (az, el) grid for
the Capon spectrum. Sweeps detection count, channel count, snapshot count,
and scan-grid size, and inlines a mid-session ``[live]`` ``az_angles``
override (finer grid -> steering rebuild). Measurement tool only: no timing
assertions.

Scenario matrix mirrors ``performance/cpp/perf_mvdr_estimator.cpp``.
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

# Import order matters for MatX/CuPy CUDA teardown: load holoscan.radar's
# compiled modules before CuPy (CuPy-first makes MatX eig cache Clear() abort
# at interpreter exit), then register MatX cache retirement after CuPy so it
# runs first on exit (atexit is LIFO) while CUDA is still alive. Mirrors
# tests/python/conftest.py. isort would otherwise alphabetize cupy back
# ahead of holoscan.radar (that regression is what caused the crash this
# guard fixes) - keep this block isort-off.
# isort: off
import holoscan.radar  # noqa: E402,F401

import cupy  # noqa: E402,F401

# isort: on

holoscan.radar.install_matx_cache_atexit()

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
from utils.perf_source import PerfOverrideSourceOp, PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import AngleConvention  # noqa: E402
from holoscan.radar import (
    AngleConventionPreset,
    ArrayGeometry,
    ArrayGeometryType,
    MVDREstimatorOpF32,
)

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_mvdr_estimator.cpp
#
# Covariance + inverse scratch is O(detections · channels²); the ladder keeps
# that bounded rather than chasing the wide dense-array sizes Beamscan uses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeSpec:
    id: str
    channels: int
    detections: int
    snapshots: int
    num_az: int
    num_el: int
    role: str


SIZE_LADDER = [
    SizeSpec("xs", 4, 64, 16, 21, 11, "small array; fast smoke"),
    SizeSpec("sm", 8, 128, 32, 41, 21, "compact SIMO array"),
    SizeSpec("md", 16, 256, 48, 61, 31, "common dev size"),
    SizeSpec("lg", 32, 512, 64, 81, 41, "wide aperture, many looks"),
    SizeSpec("xl", 32, 1024, 96, 101, 51, "upper characterization bound"),
]
_SIZE_BY_ID = {s.id: s for s in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeSpec:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


def linspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (stop - start) / float(n - 1)
    return [start + step * float(i) for i in range(n)]


def make_ula_geometry(num_elements: int, spacing: float, wavelength: float) -> ArrayGeometry:
    geom = ArrayGeometry()
    geom.wavelength = wavelength
    geom.geometry_type = ArrayGeometryType.ULA
    positions: list[float] = []
    for m in range(num_elements):
        positions.extend([0.0, float(m) * spacing, 0.0])
    geom.element_positions = positions
    return geom


def radar_boresight_convention() -> AngleConvention:
    conv = AngleConvention()
    conv.preset = AngleConventionPreset.RADAR_BORESIGHT
    return conv


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------


@dataclass
class MvdrScenario:
    id: str
    size_id: str
    channels: int
    detections: int
    snapshots: int
    num_az: int
    num_el: int
    # Filled per run by run_scenario() for inline live-override measurement.
    override_at: int = 0
    override_num_az: int = 0


def make_scenario(s: SizeSpec) -> MvdrScenario:
    sid = f"mvdr_{s.id}_c{s.channels}_n{s.detections}_s{s.snapshots}" f"_g{s.num_az}x{s.num_el}"
    return MvdrScenario(
        id=sid,
        size_id=s.id,
        channels=s.channels,
        detections=s.detections,
        snapshots=s.snapshots,
        num_az=s.num_az,
        num_el=s.num_el,
        override_num_az=s.num_az + 20,
    )


def build_default_matrix() -> list[MvdrScenario]:
    return [make_scenario(_must_find_size(sid)) for sid in ("xs", "sm", "md")]


def build_size_sweep(size_ids: list[str]) -> list[MvdrScenario]:
    out: list[MvdrScenario] = []
    for sid in size_ids:
        spec = _SIZE_BY_ID.get(sid)
        if spec is None:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(make_scenario(spec))
    return out


def build_full_matrix() -> list[MvdrScenario]:
    return [make_scenario(s) for s in SIZE_LADDER]


def _base_row(scen: MvdrScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="MVDREstimatorOp",
        variant="capon",
        size_id=scen.size_id,
        map_shape=[scen.num_az, scen.num_el],
        input_shape=[scen.detections, scen.channels, scen.snapshots],
        out_batch=scen.detections,
        k=scen.channels,
    )


def _fill_throughput(row: ResultRow, scen: MvdrScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    n = float(scen.detections)
    beams = float(scen.num_az * scen.num_el)
    row.cells_per_second = n / sec
    row.slices_per_second = n * beams / sec


def _run_graph(
    *,
    scen: MvdrScenario,
    collector: TimingCollector,
    frame: dict,
    geometry: ArrayGeometry,
    convention: AngleConvention,
    az_grid: list[float],
    el_grid: list[float],
    total_frames: int,
) -> None:
    class MvdrPerfApp(Application):
        def compose(self) -> None:
            op = MVDREstimatorOpF32(
                self,
                array_geometry=geometry,
                angle_convention=convention,
                az_angles=list(az_grid),
                el_angles=list(el_grid),
                name="mvdr",
            )
            src = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="snapshot",
                frame=frame,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("snapshot", "snapshot")})

            aoa_sink = PerfTerminalSinkOp(self, collector=collector, name="aoa_sink")
            self.add_flow(op, aoa_sink, {("aoa_estimates", "input")})

            spec_drain = make_null_sink(self, "spectrum_drain")
            self.add_flow(op, spec_drain, {("angle_spectrum", "input")})

            override = PerfOverrideSourceOp(
                self,
                CountCondition(self, count=total_frames),
                target_tick=scen.override_at,
                value=linspace(-0.5, 0.5, scen.override_num_az),
                name="ovr",
            )
            self.add_flow(override, op, {("output", "az_angles")})

    MvdrPerfApp().run()


def run_scenario(scen: MvdrScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        run_scen = replace(scen, override_at=run_plan.override_at)
        collector = TimingCollector()
        snap = make_iq_cube_device([scen.detections, scen.channels, scen.snapshots])
        frame = single_map(snap, "snapshot")
        geometry = make_ula_geometry(scen.channels, 0.5, 1.0)
        convention = radar_boresight_convention()
        az_grid = linspace(-0.5, 0.5, scen.num_az)
        el_grid = linspace(-0.3, 0.3, scen.num_el)
        collector.t_construct = now()
        _run_graph(
            scen=run_scen,
            collector=collector,
            frame=frame,
            geometry=geometry,
            convention=convention,
            az_grid=az_grid,
            el_grid=el_grid,
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
        "state; override_ms is peak latency over the live-az_angles (finer "
        "grid) window; cells/s is detections/s. The Capon scan dominates per "
        "tick (~97%); covariance + batched inverse are minor"
    )
    report.results.append(compute)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario(_must_find_size("xs")), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_mvdr_estimator",
        description="MVDREstimatorOp performance characterization (Python)",
        size_ladder_defs=[
            SizeDef(s.id, [s.num_az, s.num_el], s.num_az * s.num_el, s.role) for s in SIZE_LADDER
        ],
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        full_default=False,
        full_help="Full size ladder",
    )


if __name__ == "__main__":
    sys.exit(main())
