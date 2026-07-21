# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for VirtualArrayAssemblyOp (Python).

Measurement tool only: no timing assertions. Mirrors
``performance/cpp/perf_virtual_array_assembly.cpp``: flatten separated
(tx, rx) RD maps into a virtual channel axis and emit runtime
``array_geometry``, sweeping (M_tx, N_rx, num_doppler, num_range_bins), a
leading batch axis, and inlining a mid-session ``[live]`` wavelength
override.
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
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.perf_source import PerfOverrideSourceOp, PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import OverlapPolicy, VirtualArrayAssemblyOpF32  # noqa: E402

_DEFAULT_WAVELENGTH = 0.03
_OVERRIDE_WAVELENGTH = 0.025

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_virtual_array_assembly.cpp
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

_POLICY_TAG = {
    OverlapPolicy.DEDUP: "dedup",
    OverlapPolicy.AVERAGE: "avg",
    OverlapPolicy.KEEP_ALL: "keep",
}


@dataclass
class AssemblyScenario:
    id: str
    size_id: str
    num_tx: int
    num_rx: int
    num_doppler: int
    num_range_bins: int
    batch: int = 1
    policy: OverlapPolicy = OverlapPolicy.DEDUP


def _must_find_size(size_id: str) -> tuple[int, int, int, int]:
    try:
        return _SIZE_SPECS[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


def linear_positions(count: int, spacing: float) -> list[float]:
    out: list[float] = []
    for i in range(count):
        out.append(0.0)
        out.append(float(i) * spacing)
        out.append(0.0)
    return out


def policy_tag(policy: OverlapPolicy) -> str:
    return _POLICY_TAG.get(policy, "unk")


def separated_shape(scen: AssemblyScenario) -> list[int]:
    if scen.batch > 1:
        return [
            scen.batch,
            scen.num_tx,
            scen.num_rx,
            scen.num_doppler,
            scen.num_range_bins,
        ]
    return [scen.num_tx, scen.num_rx, scen.num_doppler, scen.num_range_bins]


def make_scenario(
    size_id: str,
    batch: int = 1,
    policy: OverlapPolicy = OverlapPolicy.DEDUP,
) -> AssemblyScenario:
    num_tx, num_rx, num_doppler, num_range_bins = _must_find_size(size_id)
    scen_id = f"virtual_{policy_tag(policy)}_{size_id}"
    if batch > 1:
        scen_id += f"_b{batch}"
    return AssemblyScenario(
        id=scen_id,
        size_id=size_id,
        num_tx=num_tx,
        num_rx=num_rx,
        num_doppler=num_doppler,
        num_range_bins=num_range_bins,
        batch=batch,
        policy=policy,
    )


def build_default_matrix() -> list[AssemblyScenario]:
    out: list[AssemblyScenario] = []
    for sid in ("xs", "md", "lg", "xl"):
        out.append(make_scenario(sid))
    out.append(make_scenario("md", 4))
    return out


def build_size_sweep(size_ids: list[str]) -> list[AssemblyScenario]:
    out: list[AssemblyScenario] = []
    for sid in size_ids:
        if sid not in _SIZE_BY_ID:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(make_scenario(sid))
    return out


def build_full_matrix() -> list[AssemblyScenario]:
    out: list[AssemblyScenario] = []
    for item in SIZE_LADDER:
        out.append(make_scenario(item.id))
    out.append(make_scenario("md", 1, OverlapPolicy.AVERAGE))
    out.append(make_scenario("md", 1, OverlapPolicy.KEEP_ALL))
    for sid in ("md", "lg"):
        out.append(make_scenario(sid, 4))
    return out


def _base_row(scen: AssemblyScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="VirtualArrayAssemblyOp",
        variant=policy_tag(scen.policy),
        size_id=scen.size_id,
        input_shape=separated_shape(scen),
        out_batch=scen.batch,
        k=scen.num_doppler,
        notes="live_override=wavelength",
    )


def _fill_throughput(row: ResultRow, scen: AssemblyScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    batch = float(scen.batch)
    elems = (
        batch
        * float(scen.num_tx)
        * float(scen.num_rx)
        * float(scen.num_doppler)
        * float(scen.num_range_bins)
    )
    row.cells_per_second = elems / sec
    csz = 8.0  # sizeof(complex64)
    out_virtual = float(scen.num_tx * scen.num_rx)  # dedup upper bound
    in_bytes = elems * csz
    out_bytes = batch * out_virtual * float(scen.num_doppler) * float(scen.num_range_bins) * csz
    row.estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9


def _run_graph(
    *,
    scen: AssemblyScenario,
    collector: TimingCollector,
    frame: dict,
    override_at: int,
    total_frames: int,
) -> None:
    tx_positions = linear_positions(scen.num_tx, 0.5)
    rx_positions = linear_positions(scen.num_rx, 0.5)

    class VirtualAssemblyPerfApp(Application):
        def compose(self) -> None:
            op = VirtualArrayAssemblyOpF32(
                self,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                wavelength=_DEFAULT_WAVELENGTH,
                overlap_policy=scen.policy,
                name="virtual_array",
            )

            src = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="rd_maps",
                frame=frame,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("rd_maps", "rd_maps")})

            rd_sink = PerfTerminalSinkOp(self, collector=collector, name="rd_sink")
            self.add_flow(op, rd_sink, {("rd_maps", "input")})

            geom_drain = make_null_sink(self, "geom_drain")
            self.add_flow(op, geom_drain, {("array_geometry", "input")})

            ovr = PerfOverrideSourceOp(
                self,
                CountCondition(self, count=total_frames),
                target_tick=override_at,
                value=_OVERRIDE_WAVELENGTH,
                name="ovr",
            )
            self.add_flow(ovr, op, {("output", "wavelength")})

    VirtualAssemblyPerfApp().run()


def run_scenario(scen: AssemblyScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        frame = single_map(make_iq_cube_device(separated_shape(scen)), "rd_maps")
        collector.t_construct = now()
        _run_graph(
            scen=scen,
            collector=collector,
            frame=frame,
            override_at=run_plan.override_at,
            total_frames=run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(rs, plan, run_rep)
    append_setup_row(_base_row(scen), samples, report)

    row = _base_row(scen)
    fill_compute_row(row, samples)
    _fill_throughput(row, scen)
    row.notes = (
        "virtual channel flatten + geometry emit; cold_ms is first-frame "
        "latency; median/p90 are pre-override steady state; override_ms is "
        "peak latency over the 3-frame live wavelength switch; cells/s counts "
        "input complex samples"
    )
    report.results.append(row)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario("xs"), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_virtual_array_assembly",
        description="VirtualArrayAssemblyOp performance characterization (Python)",
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        isolate_failures=True,
        full_help="Full matrix (sizes + overlap policies + batch axis on md/lg)",
        full_default=False,
    )


if __name__ == "__main__":
    sys.exit(main())
