# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python performance characterization for ESPRITEstimatorOp."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

_PERF_DIR = os.path.dirname(os.path.abspath(__file__))
if _PERF_DIR not in sys.path:
    sys.path.insert(0, _PERF_DIR)

from utils.bootstrap import bootstrap_radar_package  # noqa: E402

bootstrap_radar_package()

# isort: off
import holoscan.radar  # noqa: E402,F401
import cupy  # noqa: E402,F401

# isort: on

holoscan.radar.install_matx_cache_atexit()

from utils.perf_app import TimingCollector, now  # noqa: E402
from utils.perf_driver import run_perf_matrix_main  # noqa: E402
from utils.perf_report import PerfReport, ResultRow, RunSettings, SizeDef  # noqa: E402
from utils.perf_runner import (  # noqa: E402
    RunPlan,
    append_setup_row,
    collect_scenario_samples,
    fill_compute_row,
    make_run_plan,
)
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.perf_source import PerfOverrideSourceOp, PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import (  # noqa: E402
    AngleConvention,
    AngleConventionPreset,
    ArrayGeometry,
    ArrayGeometryType,
    ESPRITEstimatorOpF32,
)


@dataclass(frozen=True)
class SizeSpec:
    id: str
    channels: int
    detections: int
    snapshots: int
    role: str


@dataclass
class Scenario:
    id: str
    size_id: str
    channels: int
    detections: int
    snapshots: int
    override_at: int = 0
    # URA exercises the native 2-D dual-invariance path.
    is_ura: bool = False
    ny: int = 0
    nz: int = 0


SIZE_LADDER = [
    SizeSpec("xs", 4, 64, 16, "small array; fast smoke"),
    SizeSpec("sm", 8, 128, 32, "compact SIMO array"),
    SizeSpec("md", 16, 256, 48, "common dev size"),
    SizeSpec("lg", 32, 512, 64, "wide aperture, many looks"),
    SizeSpec("xl", 32, 1024, 96, "upper characterization bound"),
]
_SIZE_BY_ID = {size.id: size for size in SIZE_LADDER}


def make_ula(channels: int, spacing: float) -> ArrayGeometry:
    geometry = ArrayGeometry()
    geometry.wavelength = 1.0
    geometry.geometry_type = ArrayGeometryType.ULA
    geometry.element_positions = [
        coordinate for channel in range(channels) for coordinate in (0.0, spacing * channel, 0.0)
    ]
    return geometry


def make_ura(ny: int, nz: int, spacing: float) -> ArrayGeometry:
    geometry = ArrayGeometry()
    geometry.wavelength = 1.0
    geometry.geometry_type = ArrayGeometryType.URA
    geometry.element_positions = [
        coordinate
        for iz in range(nz)
        for iy in range(ny)
        for coordinate in (0.0, spacing * iy, spacing * iz)
    ]
    return geometry


def scenario_geometry(scenario: Scenario, spacing: float) -> ArrayGeometry:
    if scenario.is_ura:
        return make_ura(scenario.ny, scenario.nz, spacing)
    return make_ula(scenario.channels, spacing)


def make_scenario(size: SizeSpec) -> Scenario:
    return Scenario(
        id=(f"esprit_{size.id}_c{size.channels}_n{size.detections}" f"_s{size.snapshots}"),
        size_id=size.id,
        channels=size.channels,
        detections=size.detections,
        snapshots=size.snapshots,
    )


def make_ura_scenario(size_id: str, ny: int, nz: int, detections: int, snapshots: int) -> Scenario:
    return Scenario(
        id=f"esprit_ura_{ny}x{nz}_n{detections}_s{snapshots}",
        size_id=size_id,
        channels=ny * nz,
        detections=detections,
        snapshots=snapshots,
        is_ura=True,
        ny=ny,
        nz=nz,
    )


def build_default_matrix() -> list[Scenario]:
    scenarios = [make_scenario(_SIZE_BY_ID[key]) for key in ("xs", "sm", "md")]
    # Native 2-D az+el path: a compact filled URA (dual invariance).
    scenarios.append(make_ura_scenario("ura_sm", 6, 6, 128, 96))
    return scenarios


def build_size_sweep(size_ids: list[str]) -> list[Scenario]:
    return [make_scenario(_SIZE_BY_ID[size_id]) for size_id in size_ids if size_id in _SIZE_BY_ID]


def build_full_matrix() -> list[Scenario]:
    scenarios = [make_scenario(size) for size in SIZE_LADDER]
    # URA 2-D scenarios: dual-invariance LS + joint-diagonalization pairing over
    # an ny * nz channel count — characterize, don't gate.
    scenarios.append(make_ura_scenario("ura_sm", 6, 6, 128, 96))
    scenarios.append(make_ura_scenario("ura_lg", 8, 8, 256, 128))
    return scenarios


def base_row(scenario: Scenario) -> ResultRow:
    return ResultRow(
        scenario_id=scenario.id,
        op_name="ESPRITEstimatorOp",
        variant="esprit_ls",
        size_id=scenario.size_id,
        input_shape=[
            scenario.detections,
            scenario.channels,
            scenario.snapshots,
        ],
        out_batch=scenario.detections,
        k=scenario.channels,
    )


def run_graph(
    scenario: Scenario,
    collector: TimingCollector,
    frame: dict,
    total_frames: int,
) -> None:
    geometry = scenario_geometry(scenario, 0.5)
    convention = AngleConvention()
    convention.preset = AngleConventionPreset.RADAR_BORESIGHT

    class ESPRITPerfApp(Application):
        def compose(self) -> None:
            estimator = ESPRITEstimatorOpF32(
                self,
                array_geometry=geometry,
                angle_convention=convention,
                num_sources=1,
                source_count_pfa=1e-4,
                max_sources=4,
                name="esprit",
            )
            source = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="snapshot",
                frame=frame,
                collector=collector,
                name="source",
            )
            self.add_flow(source, estimator, {("snapshot", "snapshot")})
            sink = PerfTerminalSinkOp(self, collector=collector, name="sink")
            self.add_flow(estimator, sink, {("aoa_estimates", "input")})
            for port in ("doa_multi", "source_count", "eigenvalues"):
                drain = make_null_sink(self, f"{port}_drain")
                self.add_flow(estimator, drain, {(port, "input")})

            overrides = {
                "array_geometry": scenario_geometry(scenario, 0.55),
                "num_sources": -1,
                "source_count_pfa": 0.5,
                "max_sources": 3,
            }
            # Spatial smoothing reshapes the aperture out of the URA's (iz, iy)
            # lattice the native 2-D derivation depends on; ULA-only.
            if not scenario.is_ura:
                overrides["spatial_smoothing"] = [2]
            for port, value in overrides.items():
                override = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scenario.override_at,
                    value=value,
                    name=f"{port}_override",
                )
                self.add_flow(override, estimator, {("output", port)})

    ESPRITPerfApp().run()


def run_scenario(scenario: Scenario, settings: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(settings, has_override=True)

    def run_repeat(_repeat: int, run_plan: RunPlan) -> TimingCollector:
        run = replace(scenario, override_at=run_plan.override_at)
        collector = TimingCollector()
        tensor = make_iq_cube_device([scenario.detections, scenario.channels, scenario.snapshots])
        collector.t_construct = now()
        run_graph(
            run,
            collector,
            single_map(tensor, "snapshot"),
            run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(settings, plan, run_repeat)
    append_setup_row(base_row(scenario), samples, report)
    row = base_row(scenario)
    fill_compute_row(row, samples)
    if row.latency.median_ms > 0:
        row.cells_per_second = scenario.detections / (row.latency.median_ms / 1000.0)
    row.notes = (
        "fixed one-source steady path; live bundle switches geometry, EFT, "
        "Pfa, capacity, and smoothing"
    )
    report.results.append(row)


def process_warmup(_settings: RunSettings) -> None:
    discarded = PerfReport()
    warmup = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario(_SIZE_BY_ID["xs"]), warmup, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_esprit_estimator",
        description="ESPRITEstimatorOp performance characterization (Python)",
        size_ladder_defs=[
            SizeDef(
                size.id,
                [size.channels, size.detections, size.snapshots],
                size.detections,
                size.role,
            )
            for size in SIZE_LADDER
        ],
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scenario: scenario.id,
        exercised_size_ids=lambda scenario: scenario.size_id,
        process_warmup=process_warmup,
        full_default=False,
        full_help="Full size ladder",
    )


if __name__ == "__main__":
    sys.exit(main())
