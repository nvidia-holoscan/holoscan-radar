# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Holoscan graph comparison: split vs fused (Python).

Split:
  range → [clutter] → Doppler
Fused:
  FMCWRangeDopplerProcessorOp

Measurement tool only — no timing assertions and no keep/kill verdict.
Summing the three per-operator perf tools overstates split cost (extra
Holoscan nodes / intermediate tensors are missing from a naive sum); this
tool times both graphs with identical I/O so the team can fill the
decision table in
operators/fmcw/fmcw_range_doppler_processor_op/OPEN_ITEMS.md §1.

Scenario matrices mirror ``performance/cpp/perf_fmcw_split_vs_fused.cpp``
(default / --full / --sizes). No [live] overrides.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum

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
from utils.perf_source import PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import ClutterFilterMethod  # noqa: E402
from holoscan.radar import (
    ClutterFilterOpF32,
    FMCWDopplerProcessorOpF32,
    FMCWRangeDopplerProcessorOpF32,
    FMCWRangeProcessorOpF32,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    WindowSpec,
    WindowType,
)

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_fmcw_split_vs_fused.cpp
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


class GraphKind(str, Enum):
    SPLIT = "split"
    FUSED = "fused"


@dataclass
class CompareScenario:
    id: str
    size_id: str
    pair_id: str
    num_chirps: int
    samples_per_chirp: int
    channels: int = 1
    clutter: ClutterFilterMethod = ClutterFilterMethod.NONE
    doppler_window: WindowType = WindowType.NONE
    graph: GraphKind = GraphKind.FUSED


def _clutter_tag(method: ClutterFilterMethod) -> str:
    if method == ClutterFilterMethod.NONE:
        return "noclt"
    if method == ClutterFilterMethod.MEAN_SUBTRACTION:
        return "msub"
    if method == ClutterFilterMethod.SINGLE_DELAY:
        return "sdel"
    if method == ClutterFilterMethod.DOUBLE_DELAY:
        return "ddel"
    return "unk"


def _make_waveform() -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = 77.0e9
    spec.bandwidth = 150.0e6
    spec.chirp_duration = 20.0e-6
    spec.sample_rate = 12.8e6
    spec.chirp_repetition_interval = 25.0e-6
    return FMCWWaveformDescriptor(spec)


def _cube_shape(scen: CompareScenario) -> tuple[int, ...]:
    if scen.channels > 1:
        return (scen.channels, scen.num_chirps, scen.samples_per_chirp)
    return (scen.num_chirps, scen.samples_per_chirp)


def make_scenario(
    clutter: ClutterFilterMethod,
    size: SizeDef,
    graph: GraphKind,
    channels: int = 1,
    doppler_window: WindowType = WindowType.NONE,
) -> CompareScenario:
    num_chirps, samples_per_chirp = size.shape
    pair_id = f"svf_{_clutter_tag(clutter)}_{doppler_window.name.lower()}_{size.id}"
    if channels > 1:
        pair_id += f"_ch{channels}"
    return CompareScenario(
        id=f"{pair_id}_{graph.value}",
        size_id=size.id,
        pair_id=pair_id,
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp,
        channels=channels,
        clutter=clutter,
        doppler_window=doppler_window,
        graph=graph,
    )


def _append_pair(
    out: list[CompareScenario],
    clutter: ClutterFilterMethod,
    size: SizeDef,
    channels: int = 1,
    doppler_window: WindowType = WindowType.NONE,
) -> None:
    out.append(make_scenario(clutter, size, GraphKind.SPLIT, channels, doppler_window))
    out.append(make_scenario(clutter, size, GraphKind.FUSED, channels, doppler_window))


def build_default_matrix() -> list[CompareScenario]:
    out: list[CompareScenario] = []
    for sid in ("xs", "md", "lg", "xl"):
        size = _must_find_size(sid)
        _append_pair(out, ClutterFilterMethod.NONE, size)
        _append_pair(out, ClutterFilterMethod.MEAN_SUBTRACTION, size)
    _append_pair(out, ClutterFilterMethod.NONE, _must_find_size("md"), 4)
    _append_pair(
        out,
        ClutterFilterMethod.MEAN_SUBTRACTION,
        _must_find_size("md"),
        doppler_window=WindowType.HANN,
    )
    return out


def build_size_sweep(size_ids: list[str]) -> list[CompareScenario]:
    out: list[CompareScenario] = []
    for sid in size_ids:
        size = _must_find_size(sid)
        _append_pair(out, ClutterFilterMethod.NONE, size)
        _append_pair(out, ClutterFilterMethod.MEAN_SUBTRACTION, size)
    return out


def build_full_matrix() -> list[CompareScenario]:
    out: list[CompareScenario] = []
    clutters = (
        ClutterFilterMethod.NONE,
        ClutterFilterMethod.MEAN_SUBTRACTION,
        ClutterFilterMethod.SINGLE_DELAY,
    )
    for size in SIZE_LADDER:
        for clutter in clutters:
            _append_pair(out, clutter, size)
        _append_pair(
            out,
            ClutterFilterMethod.MEAN_SUBTRACTION,
            size,
            doppler_window=WindowType.HANN,
        )
    for sid in ("md", "lg"):
        size = _must_find_size(sid)
        for channels in (4, 16):
            _append_pair(out, ClutterFilterMethod.NONE, size, channels)
    return out


def _base_row(scen: CompareScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="FMCWRangeDoppler_split_vs_fused",
        variant=(
            f"{_clutter_tag(scen.clutter)}_"
            f"{scen.doppler_window.name.lower()}_{scen.graph.value}"
        ),
        size_id=scen.size_id,
        map_shape=[scen.num_chirps, scen.samples_per_chirp],
        input_shape=list(_cube_shape(scen)),
        out_batch=scen.channels,
        k=scen.samples_per_chirp,
        calibration_pair=scen.pair_id,
        calibration_mode=scen.graph.value,
        notes="e2e Holoscan graph; no live override; pair_id joins split/fused rows",
    )


def _fill_throughput(row: ResultRow, scen: CompareScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    batch = float(scen.channels)
    chirps = float(scen.num_chirps)
    n = float(scen.samples_per_chirp)
    row.cells_per_second = batch * chirps * n / sec
    row.slices_per_second = batch / sec
    csz = 8.0  # complex64
    bytes_ = 2.0 * batch * chirps * n * csz
    row.estimated_effective_gbps = bytes_ / sec / 1e9


def _run_fused_graph(
    *,
    scen: CompareScenario,
    collector: TimingCollector,
    frame: dict,
    waveform: FMCWWaveformDescriptor,
    total_frames: int,
) -> None:
    class FusedCompareApp(Application):
        def compose(self) -> None:
            doppler_window = WindowSpec()
            doppler_window.type = scen.doppler_window
            op = FMCWRangeDopplerProcessorOpF32(
                self,
                waveform=waveform,
                clutter_filter=scen.clutter,
                doppler_window=doppler_window,
                name="fused",
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

            rd_sink = PerfTerminalSinkOp(self, collector=collector, name="rd_sink")
            self.add_flow(op, rd_sink, {("rd_maps", "input")})

            for port in ("range_bins", "velocity_bins"):
                null_sink = make_null_sink(self, f"{port}_sink")
                self.add_flow(op, null_sink, {(port, "input")})

    FusedCompareApp().run()


def _run_split_graph(
    *,
    scen: CompareScenario,
    collector: TimingCollector,
    frame: dict,
    waveform: FMCWWaveformDescriptor,
    total_frames: int,
) -> None:
    class SplitCompareApp(Application):
        def compose(self) -> None:
            range_op = FMCWRangeProcessorOpF32(self, waveform=waveform, name="range")
            doppler_window = WindowSpec()
            doppler_window.type = scen.doppler_window
            doppler = FMCWDopplerProcessorOpF32(
                self,
                waveform=waveform,
                doppler_window=doppler_window,
                name="doppler",
            )

            src = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="adc_cube",
                frame=frame,
                collector=collector,
                name="src",
            )
            self.add_flow(src, range_op, {("adc_cube", "adc_cube")})

            if scen.clutter == ClutterFilterMethod.NONE:
                self.add_flow(range_op, doppler, {("range_profiles", "range_profiles")})
            else:
                clutter_kwargs = {"method": scen.clutter, "name": "clutter"}
                if scen.clutter == ClutterFilterMethod.MEAN_SUBTRACTION:
                    clutter_kwargs["doppler_window"] = doppler_window
                clutter = ClutterFilterOpF32(self, **clutter_kwargs)
                self.add_flow(range_op, clutter, {("range_profiles", "range_profiles")})
                self.add_flow(clutter, doppler, {("range_profiles_out", "range_profiles")})

            rd_sink = PerfTerminalSinkOp(self, collector=collector, name="rd_sink")
            self.add_flow(doppler, rd_sink, {("rd_maps", "input")})

            range_bins_sink = make_null_sink(self, "range_bins_sink")
            self.add_flow(range_op, range_bins_sink, {("range_bins", "input")})
            vel_bins_sink = make_null_sink(self, "velocity_bins_sink")
            self.add_flow(doppler, vel_bins_sink, {("velocity_bins", "input")})

    SplitCompareApp().run()


def run_scenario(scen: CompareScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=False)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        frame = single_map(make_iq_cube_device(_cube_shape(scen)), "adc_cube")
        collector.t_construct = now()
        kwargs = dict(
            scen=scen,
            collector=collector,
            frame=frame,
            waveform=_make_waveform(),
            total_frames=run_plan.total_frames,
        )
        if scen.graph == GraphKind.FUSED:
            _run_fused_graph(**kwargs)
        else:
            _run_split_graph(**kwargs)
        return collector

    samples = collect_scenario_samples(rs, plan, run_rep)
    base = _base_row(scen)
    append_setup_row(base, samples, report)
    compute = _base_row(scen)
    fill_compute_row(compute, samples)
    _fill_throughput(compute, scen)
    report.results.append(compute)


def print_comparison_table(report: PerfReport) -> None:
    @dataclass
    class PairStats:
        pair_id: str = ""
        size_id: str = ""
        clutter: str = ""
        split_ms: float = 0.0
        fused_ms: float = 0.0
        have_split: bool = False
        have_fused: bool = False

    pairs: list[PairStats] = []
    by_id: dict[str, PairStats] = {}

    for row in report.results:
        if row.bucket != "compute" or not row.calibration_pair:
            continue
        p = by_id.get(row.calibration_pair)
        if p is None:
            p = PairStats(pair_id=row.calibration_pair)
            by_id[row.calibration_pair] = p
            pairs.append(p)
        p.size_id = row.size_id
        us = row.variant.find("_")
        p.clutter = row.variant if us < 0 else row.variant[:us]
        if row.calibration_mode == "split":
            p.split_ms = row.latency.median_ms
            p.have_split = True
        elif row.calibration_mode == "fused":
            p.fused_ms = row.latency.median_ms
            p.have_fused = True

    print("\n=== split vs fused comparison (median_ms, team decision aid) ===")
    print("No keep/kill verdict is encoded. Fill OPEN_ITEMS §1 from this table.")
    header = (
        f"{'pair_id':<28}{'size':<8}{'clutter':<8}"
        f"{'split_ms':>12}{'fused_ms':>12}{'delta_ms':>12}"
        f"{'pct_saved':>12}{'speedup':>10}"
    )
    print(header)
    for p in pairs:
        if not p.have_split or not p.have_fused:
            continue
        delta = p.split_ms - p.fused_ms
        pct = 100.0 * (1.0 - p.fused_ms / p.split_ms) if p.split_ms > 0.0 else 0.0
        speedup = p.split_ms / p.fused_ms if p.fused_ms > 0.0 else 0.0
        print(
            f"{p.pair_id:<28}{p.size_id:<8}{p.clutter:<8}"
            f"{p.split_ms:12.3f}{p.fused_ms:12.3f}{delta:12.3f}"
            f"{pct:12.3f}{speedup:10.3f}"
        )
    print("(delta_ms = split − fused; pct_saved = 100·(1 − fused/split); " "speedup = split/fused)")
    print(
        "Note: do not compare against a naive sum of "
        "perf_fmcw_range + clutter + doppler medians — that omits "
        "inter-op Holoscan cost on the split path.\n"
    )


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    xs = _must_find_size("xs")
    run_scenario(make_scenario(ClutterFilterMethod.NONE, xs, GraphKind.FUSED), warm, discarded)
    run_scenario(make_scenario(ClutterFilterMethod.NONE, xs, GraphKind.SPLIT), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_fmcw_split_vs_fused",
        description=(
            "End-to-end split (range→[clutter]→Doppler) vs fused "
            "FMCWRangeDopplerProcessorOp comparison (no keep/kill verdict)"
        ),
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        post_run=print_comparison_table,
        full_default=False,
        full_help="Full matrix (noclt/msub/sdel × sizes × channel batch)",
    )


if __name__ == "__main__":
    sys.exit(main())
