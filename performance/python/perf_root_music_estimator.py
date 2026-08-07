# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for RootMUSICEstimatorOp (Python).

Subspace, gridless (polynomial rooting) sparse-path DOA on a ULA. Per item it
forms a snapshot covariance, eigendecomposes it (batched matx::eig,
EigenMode::VECTOR), resolves a model order, builds the noise-subspace
projector, then roots the conjugate-reciprocal polynomial via companion-matrix
eigensolves (one ``cusolverDnXgeev`` call per item). Sweeps
detection count, channel count, and snapshot count (no scan grid). Inlines a
mid-session bundle of ``[live]`` overrides:
``array_geometry`` (slightly different spacing), ``num_sources`` (fixed 1 →
auto/EFT), ``source_count_pfa``, ``max_sources``, and ``spatial_smoothing``
(``[2]`` on the ULA). Measurement tool only: no timing assertions.

Scenario matrix mirrors ``performance/cpp/perf_root_music_estimator.cpp``.
``num_sources`` is fixed (=1) across the steady matrix — see that file's
docstring for why.
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
# ahead of holoscan.radar - keep this block isort-off.
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
    RootMUSICEstimatorOpF32,
)

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_root_music_estimator.cpp
#
# Same channel/detection/snapshot counts as MVDR; covariance + eig +
# projector + rooting scratch is O(detections · channels²).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeSpec:
    id: str
    channels: int
    detections: int
    snapshots: int
    role: str


SIZE_LADDER = [
    SizeSpec("xs", 4, 64, 16, "small array; fast smoke"),
    SizeSpec("sm", 8, 128, 32, "compact SIMO array"),
    SizeSpec("md", 16, 256, 48, "common dev size"),
    SizeSpec("lg", 32, 512, 64, "wide aperture, many looks"),
    SizeSpec("xl", 32, 1024, 96, "upper characterization bound"),
]
_SIZE_BY_ID = {s.id: s for s in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeSpec:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


def make_ula_geometry(num_elements: int, spacing: float, wavelength: float) -> ArrayGeometry:
    geom = ArrayGeometry()
    geom.wavelength = wavelength
    geom.geometry_type = ArrayGeometryType.ULA
    positions: list[float] = []
    for m in range(num_elements):
        positions.extend([0.0, float(m) * spacing, 0.0])
    geom.element_positions = positions
    return geom


def make_ura_geometry(ny: int, nz: int, spacing: float, wavelength: float) -> ArrayGeometry:
    geom = ArrayGeometry()
    geom.wavelength = wavelength
    geom.geometry_type = ArrayGeometryType.URA
    positions: list[float] = []
    for iz in range(nz):
        for iy in range(ny):
            positions.extend([0.0, float(iy) * spacing, float(iz) * spacing])
    geom.element_positions = positions
    return geom


def scenario_geometry(scen: RootMusicScenario, spacing: float, wavelength: float) -> ArrayGeometry:
    if scen.is_ura:
        return make_ura_geometry(scen.ny, scen.nz, spacing, wavelength)
    return make_ula_geometry(scen.channels, spacing, wavelength)


def radar_boresight_convention() -> AngleConvention:
    conv = AngleConvention()
    conv.preset = AngleConventionPreset.RADAR_BORESIGHT
    return conv


# ---------------------------------------------------------------------------
# Scenario matrix
# ---------------------------------------------------------------------------


@dataclass
class RootMusicScenario:
    id: str
    size_id: str
    channels: int
    detections: int
    snapshots: int
    # Filled per run by run_scenario() for inline live-override measurement.
    override_at: int = 0
    # URA exercises the native 2-D az+el rooting path.
    is_ura: bool = False
    ny: int = 0
    nz: int = 0


def make_scenario(s: SizeSpec) -> RootMusicScenario:
    sid = f"root_music_{s.id}_c{s.channels}_n{s.detections}_s{s.snapshots}"
    return RootMusicScenario(
        id=sid,
        size_id=s.id,
        channels=s.channels,
        detections=s.detections,
        snapshots=s.snapshots,
    )


def make_ura_scenario(
    size_id: str, ny: int, nz: int, detections: int, snapshots: int
) -> RootMusicScenario:
    sid = f"root_music_ura_{ny}x{nz}_n{detections}_s{snapshots}"
    return RootMusicScenario(
        id=sid,
        size_id=size_id,
        channels=ny * nz,
        detections=detections,
        snapshots=snapshots,
        is_ura=True,
        ny=ny,
        nz=nz,
    )


def build_default_matrix() -> list[RootMusicScenario]:
    scenarios = [make_scenario(_must_find_size(sid)) for sid in ("xs", "sm", "md")]
    # Native 2-D az+el path: a compact filled URA (per-axis rooting + pairing).
    scenarios.append(make_ura_scenario("ura_sm", 6, 6, 128, 96))
    return scenarios


def build_size_sweep(size_ids: list[str]) -> list[RootMusicScenario]:
    out: list[RootMusicScenario] = []
    for sid in size_ids:
        spec = _SIZE_BY_ID.get(sid)
        if spec is None:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(make_scenario(spec))
    return out


def build_full_matrix() -> list[RootMusicScenario]:
    scenarios = [make_scenario(s) for s in SIZE_LADDER]
    # URA 2-D scenarios: per-axis rooting doubles the per-item geev calls, and
    # channel counts grow as ny * nz — characterize, don't gate.
    scenarios.append(make_ura_scenario("ura_sm", 6, 6, 128, 96))
    scenarios.append(make_ura_scenario("ura_lg", 8, 8, 256, 128))
    return scenarios


def _base_row(scen: RootMusicScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="RootMUSICEstimatorOp",
        variant="root_music",
        size_id=scen.size_id,
        input_shape=[scen.detections, scen.channels, scen.snapshots],
        out_batch=scen.detections,
        k=scen.channels,
    )


def _fill_throughput(row: ResultRow, scen: RootMusicScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    row.cells_per_second = float(scen.detections) / sec


def _run_graph(
    *,
    scen: RootMusicScenario,
    collector: TimingCollector,
    frame: dict,
    geometry: ArrayGeometry,
    convention: AngleConvention,
    total_frames: int,
) -> None:
    class RootMusicPerfApp(Application):
        def compose(self) -> None:
            op = RootMUSICEstimatorOpF32(
                self,
                array_geometry=geometry,
                angle_convention=convention,
                num_sources=1,
                source_count_pfa=1e-4,
                max_sources=4,
                name="root_music",
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

            doa_drain = make_null_sink(self, "doa_multi_drain")
            self.add_flow(op, doa_drain, {("doa_multi", "input")})
            count_drain = make_null_sink(self, "source_count_drain")
            self.add_flow(op, count_drain, {("source_count", "input")})
            eig_drain = make_null_sink(self, "eigenvalues_drain")
            self.add_flow(op, eig_drain, {("eigenvalues", "input")})

            override_values = {
                "array_geometry": scenario_geometry(scen, 0.55, geometry.wavelength),
                "num_sources": -1,
                "source_count_pfa": 0.5,
                "max_sources": 3,
            }
            # Spatial smoothing collapses the URA's (iz, iy) lattice out of the
            # separable order the native 2-D path relies on; ULA-only.
            if not scen.is_ura:
                override_values["spatial_smoothing"] = [2]
            for port, value in override_values.items():
                source = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=value,
                    name=f"{port}_ovr",
                )
                self.add_flow(source, op, {("output", port)})

    RootMusicPerfApp().run()


def run_scenario(scen: RootMusicScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        run_scen = replace(scen, override_at=run_plan.override_at)
        collector = TimingCollector()
        snap = make_iq_cube_device([scen.detections, scen.channels, scen.snapshots])
        frame = single_map(snap, "snapshot")
        geometry = scenario_geometry(scen, 0.5, 1.0)
        convention = radar_boresight_convention()
        collector.t_construct = now()
        _run_graph(
            scen=run_scen,
            collector=collector,
            frame=frame,
            geometry=geometry,
            convention=convention,
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
        "num_sources fixed=1 (deterministic model order, isolates the "
        "covariance/eig(VECTOR)/projector/root path from EFT); cold_ms is "
        "first-frame latency; median/p90 are pre-override steady state; "
        "override_ms is peak latency over the live bundle (geometry spacing, "
        "num_sources fixed->auto+EFT, source_count_pfa, max_sources, "
        "spatial_smoothing); cells/s is detections/s"
    )
    report.results.append(compute)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario(_must_find_size("xs")), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_root_music_estimator",
        description="RootMUSICEstimatorOp performance characterization (Python)",
        size_ladder_defs=[
            SizeDef(s.id, [s.channels, s.detections, s.snapshots], s.detections, s.role)
            for s in SIZE_LADDER
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
