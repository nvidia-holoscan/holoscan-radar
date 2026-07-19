# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for MIMOCoherentCompensationOp (Python).

Measurement tool only: no timing assertions. Mirrors
``performance/cpp/perf_mimo_coherent_compensation.cpp``: TDM per-Doppler
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

from utils.bootstrap import bootstrap_radar_package

bootstrap_radar_package()

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import (  # noqa: E402
    DDMCodeDescriptor,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    MIMOCoherentCompensationOpF32,
    MultiplexingScheme,
)

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
from utils.perf_sink import PerfTerminalSinkOp  # noqa: E402
from utils.perf_source import PerfOverrideSourceOp, PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import make_iq_cube_device, single_map  # noqa: E402

_DEFAULT_PRI = 25.0e-6
_OVERRIDE_PRI = 27.5e-6

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_mimo_coherent_compensation.cpp
# ---------------------------------------------------------------------------

SIZE_LADDER = [
    SizeDef("xs", [2, 4, 32, 128], 2 * 4 * 32 * 128, "small MIMO; fast smoke"),
    SizeDef("sm", [4, 8, 64, 256], 4 * 8 * 64 * 256, "compact MIMO"),
    SizeDef("md", [4, 16, 128, 512], 4 * 16 * 128 * 512, "common dev size"),
    SizeDef("lg", [8, 32, 256, 1024], 8 * 32 * 256 * 1024, "moderate MIMO"),
    SizeDef(
        "xl", [8, 64, 512, 2048], 8 * 64 * 512 * 2048, "upper characterization bound"
    ),
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
    scheme: MultiplexingScheme
    variant: str
    num_tx: int
    num_rx: int
    num_doppler: int
    num_range_bins: int
    batch: int = 1
    non_linear_ddm: bool = False


def _must_find_size(size_id: str) -> tuple[int, int, int, int]:
    try:
        return _SIZE_SPECS[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


def make_waveform(pri: float) -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = 77.0e9
    spec.bandwidth = 150.0e6
    spec.chirp_duration = 20.0e-6
    spec.sample_rate = 12.8e6
    spec.chirp_repetition_interval = pri
    return FMCWWaveformDescriptor(spec)


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
    ddm = scen.scheme == MultiplexingScheme.DDM
    shape: list[int] = []
    if scen.batch > 1:
        shape.append(scen.batch)
    if not ddm:
        shape.append(scen.num_tx)
    shape.append(scen.num_rx)
    shape.append(scen.num_doppler)
    shape.append(scen.num_range_bins)
    return shape


def make_scenario(
    size_id: str,
    scheme: MultiplexingScheme = MultiplexingScheme.TDM,
    batch: int = 1,
    *,
    non_linear_ddm: bool = False,
) -> CompensationScenario:
    num_tx, num_rx, num_doppler, num_range_bins = _must_find_size(size_id)
    if scheme == MultiplexingScheme.DDM:
        variant = "ddm_nl" if non_linear_ddm else "ddm"
    else:
        variant = "tdm"
    scen_id = f"compensation_{variant}_{size_id}"
    if batch > 1:
        scen_id += f"_b{batch}"
    return CompensationScenario(
        id=scen_id,
        size_id=size_id,
        scheme=scheme,
        variant=variant,
        num_tx=num_tx,
        num_rx=num_rx,
        num_doppler=num_doppler,
        num_range_bins=num_range_bins,
        batch=batch,
        non_linear_ddm=non_linear_ddm,
    )


def build_default_matrix() -> list[CompensationScenario]:
    out: list[CompensationScenario] = []
    for sid in ("xs", "md", "lg", "xl"):
        out.append(make_scenario(sid))
    out.append(make_scenario("md", MultiplexingScheme.TDM, 4))
    for sid in ("xs", "md", "xl"):
        out.append(make_scenario(sid, MultiplexingScheme.DDM))
        out.append(
            make_scenario(sid, MultiplexingScheme.DDM, non_linear_ddm=True)
        )
    return out


def build_size_sweep(size_ids: list[str]) -> list[CompensationScenario]:
    out: list[CompensationScenario] = []
    for sid in size_ids:
        if sid not in _SIZE_BY_ID:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(make_scenario(sid))
        out.append(make_scenario(sid, MultiplexingScheme.DDM))
        out.append(
            make_scenario(sid, MultiplexingScheme.DDM, non_linear_ddm=True)
        )
    return out


def build_full_matrix() -> list[CompensationScenario]:
    out: list[CompensationScenario] = []
    for item in SIZE_LADDER:
        out.append(make_scenario(item.id))
    for sid in ("md", "lg"):
        out.append(make_scenario(sid, MultiplexingScheme.TDM, 4))
    for item in SIZE_LADDER:
        out.append(make_scenario(item.id, MultiplexingScheme.DDM))
        out.append(
            make_scenario(item.id, MultiplexingScheme.DDM, non_linear_ddm=True)
        )
    return out


def _base_row(scen: CompensationScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="MIMOCoherentCompensationOp",
        variant=scen.variant,
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
    tx_factor = 1.0 if scen.scheme == MultiplexingScheme.DDM else float(scen.num_tx)
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
    override_at: int,
    total_frames: int,
) -> None:
    class CompensationPerfApp(Application):
        def compose(self) -> None:
            ddm_code = (
                make_ddm_nl_code(scen.num_tx, scen.num_doppler)
                if scen.non_linear_ddm
                else make_ddm_code(scen.num_tx)
            )
            op = MIMOCoherentCompensationOpF32(
                self,
                multiplexing_scheme=scen.scheme,
                waveform=make_waveform(_DEFAULT_PRI),
                ddm_code=ddm_code,
                name="compensation",
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

            sink = PerfTerminalSinkOp(self, collector=collector, name="sink")
            self.add_flow(op, sink, {("rd_maps", "input")})

            ovr = PerfOverrideSourceOp(
                self,
                CountCondition(self, count=total_frames),
                target_tick=override_at,
                value=make_waveform(_OVERRIDE_PRI),
                name="ovr",
            )
            self.add_flow(ovr, op, {("output", "waveform")})

    CompensationPerfApp().run()


def run_scenario(
    scen: CompensationScenario, rs: RunSettings, report: PerfReport
) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()
        frame = single_map(make_iq_cube_device(input_shape(scen)), "rd_maps")
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
    if scen.non_linear_ddm:
        row.notes = (
            "DDM non-linear phase_codes conjugate-correlation demux "
            "(waveform-independent); cold_ms includes initial code-spectrum "
            "build; median/p90 are steady state; [live] waveform override is "
            "inert; cells/s counts input complex samples read; output keeps "
            "full Doppler axis"
        )
    elif scen.scheme == MultiplexingScheme.DDM:
        row.notes = (
            "DDM per-band Doppler gather (waveform-independent); cold_ms is "
            "first-frame latency including initial demux-table build; "
            "median/p90 are steady state; the [live] waveform override is "
            "inert for DDM; cells/s counts input complex samples read"
        )
    else:
        row.notes = (
            "TDM per-bin complex multiply; cold_ms is first-frame latency "
            "including initial correction-table build; median/p90 are "
            "pre-override steady state; override_ms is peak latency over the "
            "3-frame live waveform rebuild window; cells/s counts complex "
            "samples touched"
        )
    report.results.append(row)


def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(make_scenario("xs"), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_mimo_coherent_compensation",
        description="MIMOCoherentCompensationOp performance characterization (Python)",
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
