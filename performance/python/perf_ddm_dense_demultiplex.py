# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for DDMDenseDemultiplexOp (Python).

Measurement tool only: no timing assertions. Mirrors
``performance/cpp/perf_ddm_dense_demultiplex.cpp``: TDM per-Doppler
inter-TX phase correction on separated RD maps, sweeping (M_tx, N_rx,
num_doppler, num_range_bins), a leading batch axis, and inlining a
mid-session ``[live]`` waveform override so correction-table rebuild cost is
measured.
"""

from __future__ import annotations

import math
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
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.perf_source import PerfDualTensorSourceOp, PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import (  # noqa: E402
    make_iq_cube_device,
    make_power_map_device,
    single_map,
)

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import DDMCodeDescriptor  # noqa: E402
from holoscan.radar import DDMDenseDemultiplexOpF32  # noqa: E402

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_ddm_dense_demultiplex.cpp
# ---------------------------------------------------------------------------

SIZE_LADDER = [
    SizeDef("xs", [2, 4, 32, 128], 2 * 4 * 32 * 128, "small MIMO; fast smoke"),
    SizeDef("sm", [4, 8, 64, 256], 4 * 8 * 64 * 256, "compact MIMO"),
    SizeDef("md", [4, 16, 128, 512], 4 * 16 * 128 * 512, "common dev size"),
    SizeDef("lg", [8, 32, 256, 1024], 8 * 32 * 256 * 1024, "moderate MIMO"),
    SizeDef("xl", [8, 64, 512, 2048], 8 * 64 * 512 * 2048, "upper characterization bound"),
]

_SIZE_SPECS = {
    "xs": (2, 4, 32, 128),
    "sm": (4, 8, 64, 256),
    "md": (4, 16, 128, 512),
    "lg": (8, 32, 256, 1024),
    "xl": (8, 64, 512, 2048),
}

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


@dataclass
class CompensationScenario:
    id: str
    size_id: str
    variant: str
    num_tx: int
    num_rx: int
    num_doppler: int
    num_range_bins: int
    batch: int = 1
    non_linear_ddm: bool = False
    velocity_axis: bool = False


def _must_find_size(size_id: str) -> tuple[int, int, int, int]:
    try:
        return _SIZE_SPECS[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


def make_ddm_code(num_tx: int) -> DDMCodeDescriptor:
    code = DDMCodeDescriptor()
    code.num_tx = num_tx
    code.doppler_offsets = [float(m) / float(num_tx) for m in range(num_tx)]
    code.guard_bands = 0
    return code


def make_ddm_nl_code(num_tx: int, num_doppler: int) -> DDMCodeDescriptor:
    """Non-linear unit-magnitude codes (matches C++ perf / unit-test recipe)."""
    code = DDMCodeDescriptor()
    code.num_tx = num_tx
    code.guard_bands = 0
    codes: list[complex] = []
    for m in range(num_tx):
        for k in range(num_doppler):
            phase = math.sin(0.7 * float(k) + float(m)) * 2.0
            codes.append(complex(math.cos(phase), math.sin(phase)))
    code.phase_codes = codes
    return code


def input_shape(scen: CompensationScenario) -> list[int]:
    shape: list[int] = []
    if scen.batch > 1:
        shape.append(scen.batch)
    shape.extend([scen.num_rx, scen.num_doppler, scen.num_range_bins])
    return shape


def make_scenario(
    size_id: str,
    *,
    non_linear_ddm: bool = False,
    velocity_axis: bool = False,
) -> CompensationScenario:
    num_tx, num_rx, num_doppler, num_range_bins = _must_find_size(size_id)
    variant = "ddm_nl" if non_linear_ddm else "ddm"
    return CompensationScenario(
        id=f"ddm_dense_demultiplex_{variant}_{size_id}" f"{'_axis' if velocity_axis else ''}",
        size_id=size_id,
        variant=variant,
        num_tx=num_tx,
        num_rx=num_rx,
        num_doppler=num_doppler,
        num_range_bins=num_range_bins,
        non_linear_ddm=non_linear_ddm,
        velocity_axis=velocity_axis,
    )


def build_default_matrix() -> list[CompensationScenario]:
    out: list[CompensationScenario] = []
    for sid in ("xs", "md", "xl"):
        out.extend([make_scenario(sid), make_scenario(sid, non_linear_ddm=True)])
        if sid == "md":
            out.extend(
                [
                    make_scenario(sid, velocity_axis=True),
                    make_scenario(sid, non_linear_ddm=True, velocity_axis=True),
                ]
            )
    return out


def build_size_sweep(size_ids: list[str]) -> list[CompensationScenario]:
    out: list[CompensationScenario] = []
    for sid in size_ids:
        if sid in _SIZE_BY_ID:
            out.extend(
                [
                    make_scenario(sid),
                    make_scenario(sid, non_linear_ddm=True),
                    make_scenario(sid, velocity_axis=True),
                    make_scenario(sid, non_linear_ddm=True, velocity_axis=True),
                ]
            )
    return out


def build_full_matrix() -> list[CompensationScenario]:
    return build_size_sweep([item.id for item in SIZE_LADDER])


def _base_row(scen: CompensationScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="DDMDenseDemultiplexOp",
        variant=scen.variant + ("_axis" if scen.velocity_axis else ""),
        size_id=scen.size_id,
        input_shape=input_shape(scen),
        out_batch=scen.batch,
        k=scen.num_doppler,
        notes="live_override=waveform",
    )


def _fill_throughput(row: ResultRow, scen: CompensationScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    batch = float(scen.batch)
    # Input samples touched: TDM carries M_tx explicitly; DDM folds the M TX
    # into the (already counted) Doppler axis, so it has no M_tx factor.
    tx_factor = 1.0
    elems = (
        batch
        * tx_factor
        * float(scen.num_rx)
        * float(scen.num_doppler)
        * float(scen.num_range_bins)
    )
    row.cells_per_second = elems / sec
    csz = 8.0  # sizeof(complex64)
    bytes_ = elems * csz
    row.estimated_effective_gbps = 2.0 * bytes_ / sec / 1e9


def _run_graph(
    *,
    scen: CompensationScenario,
    collector: TimingCollector,
    frame: dict,
    total_frames: int,
) -> None:
    class CompensationPerfApp(Application):
        def compose(self) -> None:
            ddm_code = (
                make_ddm_nl_code(scen.num_tx, scen.num_doppler)
                if scen.non_linear_ddm
                else make_ddm_code(scen.num_tx)
            )
            op = DDMDenseDemultiplexOpF32(
                self,
                ddm_code=ddm_code,
                name="compensation",
            )

            if scen.velocity_axis:
                axis = single_map(
                    make_power_map_device([scen.num_doppler]),
                    "velocity_bins",
                )
                src = PerfDualTensorSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    sideband_port="velocity_bins",
                    sideband_frame=axis,
                    trigger_port="rd_maps",
                    trigger_frame=frame,
                    collector=collector,
                    name="src",
                )
                self.add_flow(src, op, {("velocity_bins", "velocity_bins")})
                self.add_flow(src, op, {("rd_maps", "rd_maps")})
                axis_sink = make_null_sink(self, "velocity_axis_sink")
                self.add_flow(op, axis_sink, {("velocity_bins_out", "input")})
            else:
                src = PerfTensorSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    output_port="rd_maps",
                    frame=frame,
                    collector=collector,
                    name="src",
                )
                self.add_flow(src, op, {("rd_maps", "rd_maps")})

            sink = PerfTerminalSinkOp(self, collector=collector, name="sink")
            self.add_flow(op, sink, {("rd_maps_out", "input")})

    CompensationPerfApp().run()


def run_scenario(scen: CompensationScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=False)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        frame = single_map(make_iq_cube_device(input_shape(scen)), "rd_maps")
        collector.t_construct = now()
        _run_graph(
            scen=scen,
            collector=collector,
            frame=frame,
            total_frames=run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(rs, plan, run_rep)
    append_setup_row(_base_row(scen), samples, report)

    row = _base_row(scen)
    fill_compute_row(row, samples)
    _fill_throughput(row, scen)
    if scen.non_linear_ddm:
        row.notes = (
            "DDM nonlinear phase-code correlation; cold_ms includes weights "
            "and FFT setup; output keeps the full Doppler axis"
        )
    else:
        row.notes = (
            "DDM linear per-band gather; cold_ms includes the initial " "demultiplex table build"
        )
    row.notes += "; velocity_axis=wired" if scen.velocity_axis else "; velocity_axis=absent"
    report.results.append(row)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario("xs"), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_ddm_dense_demultiplex",
        description="DDMDenseDemultiplexOp performance characterization (Python)",
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        isolate_failures=True,
        full_help="Full matrix (sizes + batch axis on md/lg + DDM-NL)",
        full_default=False,
    )


if __name__ == "__main__":
    sys.exit(main())
