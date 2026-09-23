# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detection-count characterization for TDMSparseCompensationOp."""

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
    RunPlan,
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
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    OverlapPolicy,
    TDMSparseCompensationOpF32,
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


SIZE_LADDER = [
    SizeSpec("xs", 2, 4, 64, 256, 16, "smoke"),
    SizeSpec("sm", 3, 4, 128, 512, 64, "sparse automotive"),
    SizeSpec("md", 4, 8, 256, 512, 256, "common MIMO"),
    SizeSpec("lg", 4, 16, 512, 1024, 1024, "large array"),
    SizeSpec("xl", 8, 16, 512, 2048, 4096, "upper bound"),
]
_SIZE_BY_ID = {size.id: size for size in SIZE_LADDER}


@dataclass(frozen=True)
class Scenario:
    id: str
    size_id: str
    size: SizeSpec


def make_waveform() -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = 77.0e9
    spec.bandwidth = 1.0e9
    spec.chirp_duration = 40.0e-6
    spec.sample_rate = 20.0e6
    spec.chirp_repetition_interval = 1.0e-3
    return FMCWWaveformDescriptor(spec)


def ula_positions(count: int, spacing: float) -> list[float]:
    result = [0.0] * (count * 3)
    for index in range(count):
        result[index * 3] = index * spacing
    return result


def make_scenario(size: SizeSpec) -> Scenario:
    return Scenario(f"tdm_sparse_{size.id}", size.id, size)


class SparseSourceOp(Operator):
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


def run_graph(
    scenario: Scenario,
    collector: TimingCollector,
    rd_maps: dict,
    detection_idx: dict,
    total_frames: int,
) -> None:
    size = scenario.size

    class SparseApp(Application):
        def compose(self) -> None:
            source = SparseSourceOp(
                self,
                CountCondition(self, count=total_frames),
                rd_maps=rd_maps,
                detection_idx=detection_idx,
                collector=collector,
                name="source",
            )
            op = TDMSparseCompensationOpF32(
                self,
                max_tx=size.tx,
                max_rx=size.rx,
                max_virtual_channels=size.tx * size.rx,
                max_doppler_bins=size.doppler,
                max_range_bins=size.range,
                max_detections=size.detections,
                max_snapshots=1,
                waveform=make_waveform(),
                tx_positions=ula_positions(size.tx, 0.02),
                rx_positions=ula_positions(size.rx, 0.002),
                overlap_policy=OverlapPolicy.KEEP_ALL,
                name="sparse",
            )
            self.add_flow(source, op, {("rd_maps", "rd_maps")})
            self.add_flow(source, op, {("detection_idx", "detection_idx")})

            terminal = PerfTerminalSinkOp(self, collector=collector, name="terminal")
            self.add_flow(op, terminal, {("snapshot", "input")})
            for port in ("is_valid", "array_geometry"):
                sink = make_null_sink(self, f"{port}_sink")
                self.add_flow(op, sink, {(port, "input")})

    SparseApp().run()


def base_row(scenario: Scenario) -> ResultRow:
    size = scenario.size
    return ResultRow(
        scenario_id=scenario.id,
        op_name="TDMSparseCompensationOp",
        variant="keep_all_k1",
        size_id=scenario.size_id,
        map_shape=[size.doppler, size.range],
        input_shape=[size.tx, size.rx, size.doppler, size.range],
        out_batch=size.detections,
        k=size.tx * size.rx,
        notes=(
            "compare with dense MIMO compensation and virtual assembly "
            "reports at the same cube shape"
        ),
    )


def run_scenario(scenario: Scenario, settings: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(settings, has_override=False)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        size = scenario.size
        collector = TimingCollector()
        rd_maps = single_map(
            make_iq_cube_device([size.tx, size.rx, size.doppler, size.range]),
            "rd_maps",
        )
        detection_idx = single_map(
            make_detection_idx_device(size.detections, size.doppler, size.range),
            "detection_idx",
        )
        collector.t_construct = now()
        run_graph(
            scenario,
            collector,
            rd_maps,
            detection_idx,
            run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(settings, plan, run_rep)
    append_setup_row(base_row(scenario), samples, report)
    row = base_row(scenario)
    fill_compute_row(row, samples)
    if row.latency.median_ms > 0:
        row.cells_per_second = scenario.size.detections / (row.latency.median_ms / 1000.0)
    report.results.append(row)


def build_size_sweep(size_ids: list[str]) -> list[Scenario]:
    return [make_scenario(_SIZE_BY_ID[size_id]) for size_id in size_ids if size_id in _SIZE_BY_ID]


def process_warmup(_settings: RunSettings) -> None:
    report = PerfReport()
    settings = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario(_SIZE_BY_ID["xs"]), settings, report)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_tdm_sparse_compensation",
        description="TDMSparseCompensationOp characterization (Python)",
        size_ladder_defs=[
            SizeDef(
                size.id,
                [size.doppler, size.range],
                size.doppler * size.range,
                size.role,
            )
            for size in SIZE_LADDER
        ],
        build_full_matrix=lambda: [make_scenario(size) for size in SIZE_LADDER],
        build_default_matrix=lambda: build_size_sweep(["xs", "sm", "md"]),
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scenario: scenario.id,
        exercised_size_ids=lambda scenario: scenario.size_id,
        process_warmup=process_warmup,
        full_default=False,
        full_help="Full sparse TDM size ladder",
    )


if __name__ == "__main__":
    sys.exit(main())
