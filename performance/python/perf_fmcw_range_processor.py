# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for FMCWRangeProcessorOp (Python).

Measurement tool only: no timing assertions. Scenario matrices mirror
``performance/cpp/perf_fmcw_range_processor.cpp`` (default / --full / --sizes).

Sweeps the (chirps x samples) size ladder, the range window (rectangular /
Hann), and a multi-channel batch. Every compute run inlines a mid-session
[live] range_window override so the window-coefficient rebuild (warn-once-
and-cache) cost is measured. Default CLI matches C++ (default matrix, not
``--full``).
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
from holoscan.radar import FMCWRangeProcessorOpF32  # noqa: E402
from holoscan.radar import (
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    WindowSpec,
    WindowType,
)

# ---------------------------------------------------------------------------
# Size ladder — FFT-friendly powers of two so cuFFT takes its fast path.
# Keep aligned with perf_fmcw_range_processor.cpp.
# ---------------------------------------------------------------------------

SIZE_LADDER = [
    SizeDef("xs", [16, 128], 16 * 128, "small cube; fast smoke"),
    SizeDef("sm", [32, 256], 32 * 256, "compact sensor"),
    SizeDef("md", [64, 512], 64 * 512, "common dev size"),
    SizeDef("lg", [128, 1024], 128 * 1024, "moderate production-scale"),
    SizeDef("xl", [256, 2048], 256 * 2048, "range-heavy moderate (v1 upper bound)"),
]

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeDef:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


@dataclass
class RangeScenario:
    id: str
    size_id: str
    num_chirps: int
    samples_per_chirp: int
    channels: int = 1  # leading batch axis (e.g. rx elements)
    window: WindowType = WindowType.NONE  # build-time range window


def _window_tag(window: WindowType) -> str:
    return "hann" if window == WindowType.HANN else "rect"


def _window_of(window_type: WindowType) -> WindowSpec:
    spec = WindowSpec()
    spec.type = window_type
    return spec


# Flip to the other window so a live override always forces a coefficient
# rebuild (rect <-> Hann).
def _flip_window(window: WindowType) -> WindowType:
    return WindowType.NONE if window == WindowType.HANN else WindowType.HANN


# A fixed, valid waveform descriptor so range_bins is emitted every tick. The
# exact values do not affect timing; they only need to pass descriptor
# validation (all finite and > 0, bandwidth defined).
def _make_waveform() -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = 77.0e9
    spec.bandwidth = 150.0e6
    spec.chirp_duration = 20.0e-6
    spec.sample_rate = 12.8e6
    spec.chirp_repetition_interval = 25.0e-6
    return FMCWWaveformDescriptor(spec)


def _cube_shape(scen: RangeScenario) -> tuple[int, ...]:
    if scen.channels > 1:
        return (scen.channels, scen.num_chirps, scen.samples_per_chirp)
    return (scen.num_chirps, scen.samples_per_chirp)


def make_scenario(window: WindowType, size: SizeDef, channels: int = 1) -> RangeScenario:
    num_chirps, samples_per_chirp = size.shape
    sid = f"range_{_window_tag(window)}_{size.id}"
    if channels > 1:
        sid += f"_ch{channels}"
    sid += "_compute"
    return RangeScenario(
        id=sid,
        size_id=size.id,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp,
        channels=channels,
        window=window,
    )


def build_default_matrix() -> list[RangeScenario]:
    out: list[RangeScenario] = []
    for sid in ("xs", "md", "lg", "xl"):
        out.append(make_scenario(WindowType.NONE, _must_find_size(sid)))
    md = _must_find_size("md")
    out.append(make_scenario(WindowType.HANN, md))
    out.append(make_scenario(WindowType.NONE, md, 4))
    return out


def build_size_sweep(size_ids: list[str]) -> list[RangeScenario]:
    out: list[RangeScenario] = []
    for sid in size_ids:
        out.append(make_scenario(WindowType.NONE, _must_find_size(sid)))
    return out


def build_full_matrix() -> list[RangeScenario]:
    out: list[RangeScenario] = []
    for size in SIZE_LADDER:
        for window in (WindowType.NONE, WindowType.HANN):
            out.append(make_scenario(window, size))
    for sid in ("md", "lg"):
        size = _must_find_size(sid)
        for channels in (4, 16):
            out.append(make_scenario(WindowType.NONE, size, channels))
    return out


def _base_row(scen: RangeScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="FMCWRangeProcessorOp",
        variant=_window_tag(scen.window),
        size_id=scen.size_id,
        map_shape=[scen.num_chirps, scen.samples_per_chirp],
        input_shape=list(_cube_shape(scen)),
        out_batch=scen.channels,
        k=scen.samples_per_chirp,  # transform length
        notes="live_override=range_window",
    )


def _fill_throughput(row: ResultRow, scen: RangeScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    batch = float(scen.channels)
    chirps = float(scen.num_chirps)
    n = float(scen.samples_per_chirp)
    row.cells_per_second = batch * chirps * n / sec  # complex bins/s
    row.slices_per_second = batch * chirps / sec  # range FFTs/s
    csz = 8.0  # complex64
    in_bytes = batch * chirps * n * csz
    out_bytes = batch * chirps * n * csz + n * 4.0  # profiles + bins (float32)
    row.estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9


def _run_graph(
    *,
    scen: RangeScenario,
    collector: TimingCollector,
    frame: dict,
    waveform: FMCWWaveformDescriptor,
    override_window: WindowSpec,
    override_at: int,
    total_frames: int,
) -> None:
    class RangePerfApp(Application):
        def compose(self) -> None:
            op = FMCWRangeProcessorOpF32(
                self,
                waveform=waveform,
                range_window=_window_of(scen.window),
                name="range",
            )
            src = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="adc_cube",
                frame=frame,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("adc_cube", "adc_cube")})

            profiles_sink = PerfTerminalSinkOp(self, collector=collector, name="profiles_sink")
            self.add_flow(op, profiles_sink, {("range_profiles", "input")})

            # range_bins is optional but emitted (waveform is set); drain it
            # so the operator ticks and the axis path is exercised.
            bins_sink = make_null_sink(self, "bins_sink")
            self.add_flow(op, bins_sink, {("range_bins", "input")})

            override = PerfOverrideSourceOp(
                self,
                CountCondition(self, count=total_frames),
                target_tick=override_at,
                value=override_window,
                name="ovr",
            )
            self.add_flow(override, op, {("output", "range_window")})

    RangePerfApp().run()


def run_scenario(scen: RangeScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        # Preallocate the input before the setup clock so the H2D upload is
        # not billed as operator setup.
        frame = single_map(make_iq_cube_device(_cube_shape(scen)), "adc_cube")
        collector.t_construct = now()
        _run_graph(
            scen=scen,
            collector=collector,
            frame=frame,
            waveform=_make_waveform(),
            override_window=_window_of(_flip_window(scen.window)),
            override_at=run_plan.override_at,
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
        "cold_ms is first-frame latency including initial window/FFT plan "
        "build; median/p90 are pre-override steady state; override_ms is peak "
        "latency over the 3-frame live range_window rebuild window; cells/s is "
        "complex range bins/s"
    )
    report.results.append(compute)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario(WindowType.NONE, _must_find_size("xs")), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_fmcw_range_processor",
        description="FMCWRangeProcessorOp performance characterization (Python)",
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        full_default=False,
        full_help="Full matrix (windows x sizes x channel batch).",
    )


if __name__ == "__main__":
    sys.exit(main())
