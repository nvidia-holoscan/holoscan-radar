# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for MIMODeinterleaveOp (Python).

Measurement tool only: no timing assertions. Mirrors
``performance/cpp/perf_mimo_deinterleave.cpp``: TDM slow-time gather into an
explicit TX axis, sweeping (M_tx, N_rx, L, samples) and a leading batch axis.
No ``[live]`` parameters.
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
from utils.perf_source import PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import MIMODeinterleaveOpF32, TXSchedule  # noqa: E402

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_mimo_deinterleave.cpp
# ---------------------------------------------------------------------------

SIZE_LADDER = [
    SizeDef("xs", [4, 16, 64], 4 * 16 * 64, "small TDM; fast smoke"),
    SizeDef("sm", [8, 64, 128], 8 * 64 * 128, "compact MIMO"),
    SizeDef("md", [16, 128, 256], 16 * 128 * 256, "common dev size"),
    SizeDef("lg", [32, 512, 512], 32 * 512 * 512, "moderate MIMO"),
    SizeDef("xl", [64, 1024, 1024], 64 * 1024 * 1024, "upper characterization bound"),
]

_SIZE_SPECS = {
    "xs": (2, 4, 8, 64),
    "sm": (4, 8, 16, 128),
    "md": (4, 16, 32, 256),
    "lg": (8, 32, 64, 512),
    "xl": (8, 64, 128, 1024),
}

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


@dataclass
class DeinterleaveScenario:
    id: str
    size_id: str
    num_tx: int
    num_rx: int
    chirps_per_tx: int
    samples: int
    batch: int = 1


def _must_find_size(size_id: str) -> tuple[int, int, int, int]:
    try:
        return _SIZE_SPECS[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


def interleaved_shape(scen: DeinterleaveScenario) -> list[int]:
    slow = scen.num_tx * scen.chirps_per_tx
    if scen.batch > 1:
        return [scen.batch, scen.num_rx, slow, scen.samples]
    return [scen.num_rx, slow, scen.samples]


def make_scenario(size_id: str, batch: int = 1) -> DeinterleaveScenario:
    num_tx, num_rx, chirps_per_tx, samples = _must_find_size(size_id)
    scen_id = f"deinterleave_rr_{size_id}"
    if batch > 1:
        scen_id += f"_b{batch}"
    return DeinterleaveScenario(
        id=scen_id,
        size_id=size_id,
        num_tx=num_tx,
        num_rx=num_rx,
        chirps_per_tx=chirps_per_tx,
        samples=samples,
        batch=batch,
    )


def build_default_matrix() -> list[DeinterleaveScenario]:
    out: list[DeinterleaveScenario] = []
    for sid in ("xs", "md", "lg", "xl"):
        out.append(make_scenario(sid))
    out.append(make_scenario("md", 4))
    return out


def build_size_sweep(size_ids: list[str]) -> list[DeinterleaveScenario]:
    out: list[DeinterleaveScenario] = []
    for sid in size_ids:
        if sid not in _SIZE_BY_ID:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(make_scenario(sid))
    return out


def build_full_matrix() -> list[DeinterleaveScenario]:
    out: list[DeinterleaveScenario] = []
    for item in SIZE_LADDER:
        out.append(make_scenario(item.id))
    for sid in ("md", "lg"):
        out.append(make_scenario(sid, 4))
    return out


def _base_row(scen: DeinterleaveScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="MIMODeinterleaveOp",
        variant="batch" if scen.batch > 1 else "rr",
        size_id=scen.size_id,
        input_shape=interleaved_shape(scen),
        out_batch=scen.batch,
        k=scen.chirps_per_tx,
    )


def _fill_throughput(row: ResultRow, scen: DeinterleaveScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    batch = float(scen.batch)
    elems = (
        batch
        * float(scen.num_tx)
        * float(scen.num_rx)
        * float(scen.chirps_per_tx)
        * float(scen.samples)
    )
    row.cells_per_second = elems / sec
    csz = 8.0  # sizeof(complex64)
    slow = float(scen.num_tx * scen.chirps_per_tx)
    in_bytes = batch * float(scen.num_rx) * slow * float(scen.samples) * csz
    out_bytes = elems * csz
    row.estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9


def _run_graph(
    *,
    scen: DeinterleaveScenario,
    collector: TimingCollector,
    frame: dict,
    total_frames: int,
) -> None:
    class DeinterleavePerfApp(Application):
        def compose(self) -> None:
            schedule = TXSchedule()
            schedule.num_tx = scen.num_tx
            op = MIMODeinterleaveOpF32(self, tx_schedule=schedule, name="deinterleave")

            src = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="adc_cube",
                frame=frame,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("adc_cube", "adc_cube")})

            sink = PerfTerminalSinkOp(self, collector=collector, name="sink")
            self.add_flow(op, sink, {("adc_cube", "input")})

    DeinterleavePerfApp().run()


def run_scenario(scen: DeinterleaveScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=False)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        frame = single_map(make_iq_cube_device(interleaved_shape(scen)), "adc_cube")
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
    row.notes = (
        "pure gather marshaling; cold_ms is first-frame latency; "
        "median/p90 are steady state; cells/s counts complex samples moved"
    )
    report.results.append(row)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario("xs"), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_mimo_deinterleave",
        description="MIMODeinterleaveOp performance characterization (Python)",
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        isolate_failures=True,
        full_help="Full matrix (sizes + batch axis on md/lg)",
        full_default=False,
    )


if __name__ == "__main__":
    sys.exit(main())
