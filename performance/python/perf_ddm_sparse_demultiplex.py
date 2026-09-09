# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse DDM demultiplex characterization."""

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
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.tensor_factory import make_detection_idx_device  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application, Operator, OperatorSpec  # noqa: E402
from holoscan.radar import (  # noqa: E402
    DDMCodeDescriptor,
    DDMSparseDemultiplexOpF32,
    OverlapPolicy,
)


@dataclass(frozen=True)
class SizeSpec:
    id: str
    tx: int
    rx: int
    doppler: int
    range: int
    detections: int
    role: str


SIZES = [
    SizeSpec("xs", 2, 4, 64, 256, 16, "smoke"),
    SizeSpec("sm", 4, 4, 128, 512, 64, "sparse automotive"),
    SizeSpec("md", 4, 8, 256, 512, 256, "common MIMO"),
    SizeSpec("lg", 4, 16, 512, 1024, 1024, "large array"),
    SizeSpec("xl", 8, 16, 512, 2048, 4096, "upper bound"),
]
SIZE_BY_ID = {size.id: size for size in SIZES}


def positions(count, spacing):
    result = [0.0] * (count * 3)
    for index in range(count):
        result[3 * index] = index * spacing
    return result


def code(tx):
    result = DDMCodeDescriptor()
    result.num_tx = tx
    result.doppler_offsets = [index / tx for index in range(tx)]
    return result


class Source(Operator):
    def __init__(self, fragment, *args, rd, idx, collector, **kwargs):
        self.rd = rd
        self.idx = idx
        self.collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("rd_maps")
        spec.output("detection_idx")

    def compute(self, op_input, op_output, context):
        if not self.collector.first_tick_set:
            self.collector.t_first_tick = now()
            self.collector.first_tick_set = True
        self.collector.starts.append(now())
        op_output.emit(dict(self.rd), "rd_maps")
        op_output.emit(dict(self.idx), "detection_idx")


def run_graph(size, collector, rd, idx, frames):
    class App(Application):
        def compose(self):
            source = Source(
                self,
                CountCondition(self, count=frames),
                rd=rd,
                idx=idx,
                collector=collector,
                name="source",
            )
            op = DDMSparseDemultiplexOpF32(
                self,
                max_tx=size.tx,
                max_rx=size.rx,
                max_doppler_bins=size.doppler,
                max_range_bins=size.range,
                max_detections=size.detections,
                max_virtual_channels=size.tx * size.rx,
                max_snapshots=1,
                ddm_code=code(size.tx),
                tx_positions=positions(size.tx, 0.02),
                rx_positions=positions(size.rx, 0.002),
                overlap_policy=OverlapPolicy.KEEP_ALL,
                wavelength=0.0039,
                name="sparse",
            )
            sink = PerfTerminalSinkOp(self, collector=collector, name="sink")
            self.add_flow(source, op, {("rd_maps", "rd_maps")})
            self.add_flow(source, op, {("detection_idx", "detection_idx")})
            self.add_flow(op, sink, {("snapshot", "input")})
            for port in ("is_valid", "array_geometry"):
                drain = make_null_sink(self, f"{port}_sink")
                self.add_flow(op, drain, {(port, "input")})

    App().run()


def scenario(size):
    return {"id": f"ddm_sparse_{size.id}", "size_id": size.id, "size": size}


def base_row(item):
    size = item["size"]
    return ResultRow(
        scenario_id=item["id"],
        op_name="DDMSparseDemultiplexOp",
        variant="keep_all_k1",
        size_id=item["size_id"],
        map_shape=[size.doppler, size.range],
        input_shape=[size.rx, size.doppler, size.range],
        out_batch=size.detections,
        k=size.tx * size.rx,
        notes="honest pre-demux input and detection-sized snapshot output",
    )


def run_scenario(item, settings, report):
    plan = make_run_plan(settings, has_override=False)

    def run_rep(_rep, current):
        size = item["size"]
        collector = TimingCollector()
        rd = single_map(make_iq_cube_device([size.rx, size.doppler, size.range]), "rd_maps")
        idx = single_map(
            make_detection_idx_device(size.detections, size.doppler // size.tx, size.range),
            "detection_idx",
        )
        collector.t_construct = now()
        run_graph(size, collector, rd, idx, current.total_frames)
        return collector

    samples = collect_scenario_samples(settings, plan, run_rep)
    append_setup_row(base_row(item), samples, report)
    row = base_row(item)
    fill_compute_row(row, samples)
    report.results.append(row)


def build_size_sweep(ids):
    return [scenario(SIZE_BY_ID[item]) for item in ids if item in SIZE_BY_ID]


def process_warmup(_settings):
    run_scenario(
        scenario(SIZES[0]),
        RunSettings(warmup_frames=0, measured_frames=2, repeats=1),
        PerfReport(),
    )


def main():
    return run_perf_matrix_main(
        tool_name="perf_ddm_sparse_demultiplex",
        description="Sparse DDM demultiplex characterization",
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
        full_help="Full sparse DDM size ladder",
    )


if __name__ == "__main__":
    sys.exit(main())
