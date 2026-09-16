# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for DopplerNotchOp (Python)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_PERF_DIR = os.path.dirname(os.path.abspath(__file__))
if _PERF_DIR not in sys.path:
    sys.path.insert(0, _PERF_DIR)

from utils.bootstrap import bootstrap_radar_package  # noqa: E402

bootstrap_radar_package()

from utils.perf_app import TimingCollector, now  # noqa: E402
from utils.perf_driver import run_perf_matrix_main  # noqa: E402
from utils.perf_report import PerfReport, ResultRow, RunSettings, SizeDef  # noqa: E402
from utils.perf_runner import append_setup_row  # noqa: E402
from utils.perf_runner import (  # noqa: E402
    RunPlan,
    collect_scenario_samples,
    fill_compute_row,
    make_run_plan,
)
from utils.perf_sink import PerfTerminalSinkOp  # noqa: E402
from utils.perf_source import (  # noqa: E402
    PerfDualTensorSourceOp,
    PerfOverrideSourceOp,
)
from utils.tensor_factory import (  # noqa: E402
    make_iq_cube_device,
    single_map,
    to_device_array,
)

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import DopplerNotchOpF32  # noqa: E402

SIZE_LADDER = [
    SizeDef("xs", [32, 128], 32 * 128, "small map; fast smoke"),
    SizeDef("md", [128, 512], 128 * 512, "common development size"),
    SizeDef("xl", [256, 2048], 256 * 2048, "range-heavy upper characterization"),
]
_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


@dataclass
class NotchScenario:
    id: str
    size_id: str
    num_doppler: int
    num_range: int
    rank4: bool
    cosine: bool
    live_override: bool = False


def make_scenario(
    size_id: str,
    *,
    rank4: bool,
    cosine: bool,
    live_override: bool = False,
) -> NotchScenario:
    size = _SIZE_BY_ID[size_id]
    variant = ("cosine" if cosine else "hard") + (
        "_mimo" if rank4 else "_siso"
    )
    if live_override:
        variant += "_live"
    return NotchScenario(
        id=f"doppler_notch_{variant}_{size_id}",
        size_id=size_id,
        num_doppler=size.shape[0],
        num_range=size.shape[1],
        rank4=rank4,
        cosine=cosine,
        live_override=live_override,
    )


def input_shape(scenario: NotchScenario) -> list[int]:
    if scenario.rank4:
        return [4, 4, scenario.num_doppler, scenario.num_range]
    return [scenario.num_doppler, scenario.num_range]


def build_size_sweep(size_ids: list[str]) -> list[NotchScenario]:
    result = []
    for size_id in size_ids:
        if size_id not in _SIZE_BY_ID:
            continue
        for cosine in (False, True):
            result.append(make_scenario(size_id, rank4=False, cosine=cosine))
            result.append(make_scenario(size_id, rank4=True, cosine=cosine))
    return result


def build_default_matrix() -> list[NotchScenario]:
    result = build_size_sweep(["xs", "md", "xl"])
    result.append(
        make_scenario("md", rank4=True, cosine=True, live_override=True)
    )
    return result


def build_full_matrix() -> list[NotchScenario]:
    return build_size_sweep([item.id for item in SIZE_LADDER])


def _base_row(scenario: NotchScenario) -> ResultRow:
    variant = "cosine" if scenario.cosine else "hard"
    variant += "_mimo" if scenario.rank4 else "_siso"
    return ResultRow(
        scenario_id=scenario.id,
        op_name="DopplerNotchOp",
        variant=variant,
        size_id=scenario.size_id,
        input_shape=input_shape(scenario),
        k=scenario.num_doppler,
        notes=(
            "center_velocity live override" if scenario.live_override else ""
        ),
    )


def _fill_throughput(row: ResultRow, scenario: NotchScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    elements = int(np.prod(input_shape(scenario)))
    seconds = row.latency.median_ms / 1000.0
    row.cells_per_second = elements / seconds
    row.estimated_effective_gbps = 2.0 * elements * 8.0 / seconds / 1e9


def _run_graph(
    scenario: NotchScenario,
    collector: TimingCollector,
    total_frames: int,
    override_at: int,
) -> None:
    class NotchPerfApp(Application):
        def compose(self) -> None:
            maps = single_map(
                make_iq_cube_device(input_shape(scenario)), "rd_maps"
            )
            axis_host = np.linspace(
                -20.0,
                20.0,
                scenario.num_doppler,
                endpoint=False,
                dtype=np.float32,
            )
            axis = single_map(to_device_array(axis_host), "velocity_bins")
            source = PerfDualTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                sideband_port="velocity_bins",
                sideband_frame=axis,
                trigger_port="rd_maps",
                trigger_frame=maps,
                collector=collector,
                name="src",
            )
            op = DopplerNotchOpF32(
                self,
                stop_half_width=0.5,
                transition_width=1.0 if scenario.cosine else 0.0,
                max_doppler_bins=scenario.num_doppler,
                max_range_bins=scenario.num_range,
                max_rx=4 if scenario.rank4 else 1,
                max_tx=4 if scenario.rank4 else 0,
                name="notch",
            )
            self.add_flow(source, op, {("velocity_bins", "velocity_bins")})
            self.add_flow(source, op, {("rd_maps", "rd_maps")})
            if scenario.live_override:
                override = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=override_at,
                    value=1.0,
                    name="center_override",
                )
                self.add_flow(
                    override, op, {("output", "center_velocity")}
                )
            sink = PerfTerminalSinkOp(self, collector=collector, name="sink")
            self.add_flow(op, sink, {("rd_maps_out", "input")})

    NotchPerfApp().run()


def run_scenario(
    scenario: NotchScenario,
    settings: RunSettings,
    report: PerfReport,
) -> None:
    plan = make_run_plan(settings, has_override=scenario.live_override)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        collector.t_construct = now()
        _run_graph(
            scenario, collector, run_plan.total_frames, run_plan.override_at
        )
        return collector

    samples = collect_scenario_samples(settings, plan, run_rep)
    append_setup_row(_base_row(scenario), samples, report)
    row = _base_row(scenario)
    fill_compute_row(row, samples)
    _fill_throughput(row, scenario)
    report.results.append(row)


def _process_warmup(_settings: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(
        make_scenario("xs", rank4=False, cosine=False),
        warm,
        discarded,
    )


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_doppler_notch",
        description="DopplerNotchOp performance characterization (Python)",
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scenario: scenario.id,
        exercised_size_ids=lambda scenario: scenario.size_id,
        process_warmup=_process_warmup,
        isolate_failures=True,
        full_help="Full hard/cosine and rank-2/rank-4 matrix",
        full_default=False,
    )


if __name__ == "__main__":
    sys.exit(main())
