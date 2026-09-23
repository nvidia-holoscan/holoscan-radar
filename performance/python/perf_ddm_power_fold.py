# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse DDM power-fold characterization."""

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
from utils.perf_runner import (  # noqa: E402
    collect_scenario_samples,
    fill_compute_row,
    make_run_plan,
)
from utils.perf_sink import PerfTerminalSinkOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application, Operator, OperatorSpec  # noqa: E402
from holoscan.radar import DDMCodeDescriptor, DDMPowerFoldOpF32  # noqa: E402


@dataclass(frozen=True)
class SizeSpec:
    id: str
    tx: int
    rx: int
    doppler: int
    range: int
    role: str


SIZES = [
    SizeSpec("xs", 2, 4, 64, 256, "smoke"),
    SizeSpec("sm", 4, 4, 128, 512, "sparse automotive"),
    SizeSpec("md", 4, 8, 256, 512, "common MIMO"),
    SizeSpec("lg", 4, 16, 512, 1024, "large array"),
    SizeSpec("xl", 8, 16, 512, 2048, "upper bound"),
]
SIZE_BY_ID = {size.id: size for size in SIZES}


class Source(Operator):
    def __init__(self, fragment, *args, payload, collector, **kwargs):
        self.payload = payload
        self.collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("rd_maps")

    def compute(self, op_input, op_output, context):
        if not self.collector.first_tick_set:
            self.collector.t_first_tick = now()
            self.collector.first_tick_set = True
        self.collector.starts.append(now())
        op_output.emit(dict(self.payload), "rd_maps")


def code(tx: int) -> DDMCodeDescriptor:
    result = DDMCodeDescriptor()
    result.num_tx = tx
    result.doppler_offsets = [index / tx for index in range(tx)]
    return result


def run_graph(size, collector, payload, frames):
    class App(Application):
        def compose(self):
            source = Source(
                self,
                CountCondition(self, count=frames),
                payload=payload,
                collector=collector,
                name="source",
            )
            op = DDMPowerFoldOpF32(
                self,
                max_rx=size.rx,
                max_doppler_bins=size.doppler,
                max_range_bins=size.range,
                ddm_code=code(size.tx),
                name="fold",
            )
            sink = PerfTerminalSinkOp(self, collector=collector, name="sink")
            self.add_flow(source, op, {("rd_maps", "rd_maps")})
            self.add_flow(op, sink, {("power_map", "input")})

    App().run()


def scenario(size):
    return {"id": f"ddm_fold_{size.id}", "size_id": size.id, "size": size}


def base_row(item):
    size = item["size"]
    return ResultRow(
        scenario_id=item["id"],
        op_name="DDMPowerFoldOp",
        variant="linear_exact",
        size_id=item["size_id"],
        map_shape=[size.doppler, size.range],
        input_shape=[size.rx, size.doppler, size.range],
        k=size.tx * size.rx,
        notes="honest pre-demux input and folded output extents",
    )


def run_scenario(item, settings, report):
    plan = make_run_plan(settings, has_override=False)

    def run_rep(_rep, current):
        size = item["size"]
        collector = TimingCollector()
        payload = single_map(make_iq_cube_device([size.rx, size.doppler, size.range]), "rd_maps")
        collector.t_construct = now()
        run_graph(size, collector, payload, current.total_frames)
        return collector

    samples = collect_scenario_samples(settings, plan, run_rep)
    append_setup_row(base_row(item), samples, report)
    row = base_row(item)
    fill_compute_row(row, samples)
    report.results.append(row)


def build_size_sweep(ids):
    return [scenario(SIZE_BY_ID[item]) for item in ids if item in SIZE_BY_ID]


def process_warmup(_settings):
    report = PerfReport()
    settings = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(scenario(SIZES[0]), settings, report)


def main():
    return run_perf_matrix_main(
        tool_name="perf_ddm_power_fold",
        description="Sparse DDM power-fold characterization",
        size_ladder_defs=[
            SizeDef(s.id, [s.doppler, s.range], s.doppler * s.range, s.role) for s in SIZES
        ],
        build_full_matrix=lambda: [scenario(s) for s in SIZES],
        build_default_matrix=lambda: build_size_sweep(["xs", "sm", "md"]),
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda item: item["id"],
        exercised_size_ids=lambda item: item["size_id"],
        process_warmup=process_warmup,
        full_default=False,
        full_help="Full sparse DDM power-fold size ladder",
    )


if __name__ == "__main__":
    raise SystemExit(main())
