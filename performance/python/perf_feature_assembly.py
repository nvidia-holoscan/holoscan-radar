# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for FeatureAssemblyOp (Python).

Measurement tool only: no timing assertions. Mirrors
``performance/cpp/perf_feature_assembly.cpp``: sweeps detection count,
score-map size, and feature packs (bins / +score / full clustering columns).
FeatureAssemblyOp has no [live] override ports, so this tool has neither a
process-warmup hook nor inline override measurement.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum

_PERF_DIR = os.path.dirname(os.path.abspath(__file__))
if _PERF_DIR not in sys.path:
    sys.path.insert(0, _PERF_DIR)

from utils.bootstrap import bootstrap_radar_package

bootstrap_radar_package()

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application, Operator, OperatorSpec  # noqa: E402
from holoscan.radar import AssemblyFeature, FeatureAssemblyOpF32  # noqa: E402

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
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.tensor_factory import (  # noqa: E402
    make_power_map_device,
    single_map,
    to_device_array,
)

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_feature_assembly.cpp
# ---------------------------------------------------------------------------

SIZE_LADDER = [
    SizeDef("xs", [64, 256], 64 * 256, "small map; fast smoke"),
    SizeDef("sm", [128, 512], 128 * 512, "compact sensor"),
    SizeDef("md", [256, 512], 256 * 512, "common dev size"),
    SizeDef("lg", [512, 1024], 512 * 1024, "moderate production-scale"),
    SizeDef("xlw", [1024, 2048], 1024 * 2048, "range-heavy moderate (v1 upper bound)"),
]

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeDef:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


class FeaturePack(str, Enum):
    BINS = "bins"
    SCORE = "score"
    FULL = "full"


ALL_PACKS = (FeaturePack.BINS, FeaturePack.SCORE, FeaturePack.FULL)


def _pack_features(pack: FeaturePack) -> list:
    if pack == FeaturePack.BINS:
        return [AssemblyFeature.RANGE, AssemblyFeature.VELOCITY]
    if pack == FeaturePack.SCORE:
        return [AssemblyFeature.RANGE, AssemblyFeature.VELOCITY, AssemblyFeature.SCORE]
    if pack == FeaturePack.FULL:
        return [
            AssemblyFeature.RANGE,
            AssemblyFeature.VELOCITY,
            AssemblyFeature.SCORE,
            AssemblyFeature.AZ,
            AssemblyFeature.EL,
        ]
    raise ValueError(f"unsupported feature pack {pack}")


def _pack_needs_score(pack: FeaturePack) -> bool:
    return pack in (FeaturePack.SCORE, FeaturePack.FULL)


def _pack_needs_aoa(pack: FeaturePack) -> bool:
    return pack == FeaturePack.FULL


@dataclass
class FeatScenario:
    id: str
    size_id: str
    pack: FeaturePack = FeaturePack.SCORE
    num_detections: int = 0
    doppler: int = 0
    range_: int = 0
    score_batch: int = 1  # leading dim on detection_score when > 1


# ---------------------------------------------------------------------------
# Deterministic device tensor factories (mirror the C++ host-fill formulas)
# ---------------------------------------------------------------------------


def _make_detection_idx(
    num_detections: int, doppler: int, range_: int, score_batch: int
) -> cp.ndarray:
    """Deterministic detection grid across the DxR plane (wraps if N > cells).

    When score_batch > 1, emits a preserved batch column `(batch, doppler,
    range)` so the assemble kernel's multi-plane score gather is exercised.
    """
    cols = 3 if score_batch > 1 else 2
    n = max(num_detections, 0)
    host = np.zeros((n, cols), dtype=np.int64)
    cells = max(doppler * range_, 1)
    batch = max(score_batch, 1)
    for i in range(num_detections):
        cell = (i * 997) % cells
        col = 0
        if cols == 3:
            host[i, col] = i % batch
            col += 1
        host[i, col] = cell // range_
        host[i, col + 1] = cell % range_
    return to_device_array(host)


def _make_axis_bins(n: int) -> cp.ndarray:
    return cp.asarray(np.arange(max(n, 0), dtype=np.float32))


def _make_aoa(num_detections: int) -> cp.ndarray:
    n = max(num_detections, 0)
    host = np.zeros((n, 2), dtype=np.float32)
    for i in range(num_detections):
        host[i, 0] = np.float32(0.01) * np.float32(i)
        host[i, 1] = np.float32(-0.02) * np.float32(i)
    return to_device_array(host)


# ---------------------------------------------------------------------------
# Multi-port source: emits the same preallocated TensorMaps every tick so
# measured latency is dominated by FeatureAssemblyOp, not H2D. All 5 output
# ports are always declared/emitted; compose() wires only the ports the
# active feature pack needs and null-drains the rest.
# ---------------------------------------------------------------------------


class FeatAsmPerfSourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        idx: dict,
        range_bins: dict,
        velocity_bins: dict,
        score: dict,
        aoa: dict,
        collector: TimingCollector,
        **kwargs,
    ):
        self._idx = idx
        self._range_bins = range_bins
        self._velocity_bins = velocity_bins
        self._score = score
        self._aoa = aoa
        self._collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("detection_idx")
        spec.output("range_bins")
        spec.output("velocity_bins")
        spec.output("detection_score")
        spec.output("aoa_estimates")

    def compute(self, op_input, op_output, context) -> None:
        if not self._collector.first_tick_set:
            self._collector.t_first_tick = now()
            self._collector.first_tick_set = True
        op_output.emit(self._idx, "detection_idx")
        op_output.emit(self._range_bins, "range_bins")
        op_output.emit(self._velocity_bins, "velocity_bins")
        op_output.emit(self._score, "detection_score")
        op_output.emit(self._aoa, "aoa_estimates")
        self._collector.starts.append(now())


def _run_graph(
    *,
    scen: FeatScenario,
    collector: TimingCollector,
    idx: dict,
    range_bins: dict,
    velocity_bins: dict,
    score: dict,
    aoa: dict,
    total_frames: int,
) -> None:
    class FeatAsmPerfApp(Application):
        def compose(self) -> None:
            features = _pack_features(scen.pack)
            op = FeatureAssemblyOpF32(self, features=features, name="feat")

            src = FeatAsmPerfSourceOp(
                self,
                CountCondition(self, count=total_frames),
                idx=idx,
                range_bins=range_bins,
                velocity_bins=velocity_bins,
                score=score,
                aoa=aoa,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("detection_idx", "detection_idx")})
            self.add_flow(src, op, {("range_bins", "range_bins")})
            self.add_flow(src, op, {("velocity_bins", "velocity_bins")})

            # Wire score/aoa into the op only when the pack needs them;
            # otherwise drain via a null sink so pack-to-pack latency stays
            # comparable.
            if _pack_needs_score(scen.pack):
                self.add_flow(src, op, {("detection_score", "detection_score")})
            else:
                score_drain = make_null_sink(self, "score_drain")
                self.add_flow(src, score_drain, {("detection_score", "input")})

            if _pack_needs_aoa(scen.pack):
                self.add_flow(src, op, {("aoa_estimates", "aoa_estimates")})
            else:
                aoa_drain = make_null_sink(self, "aoa_drain")
                self.add_flow(src, aoa_drain, {("aoa_estimates", "input")})

            term = PerfTerminalSinkOp(self, collector=collector, name="points_sink")
            self.add_flow(op, term, {("points", "input")})

    FeatAsmPerfApp().run()


# ---------------------------------------------------------------------------
# Scenario construction / matrices — mirror perf_feature_assembly.cpp exactly
# ---------------------------------------------------------------------------


def _make_scenario(
    pack: FeaturePack, size: SizeDef, num_detections: int, score_batch: int = 1
) -> FeatScenario:
    doppler, range_ = size.shape
    sid = f"feat_asm_{pack.value}_{size.id}_n{num_detections}"
    if score_batch > 1:
        sid += f"_b{score_batch}"
    sid += "_compute"
    return FeatScenario(
        id=sid,
        size_id=size.id,
        pack=pack,
        num_detections=num_detections,
        doppler=doppler,
        range_=range_,
        score_batch=score_batch,
    )


def build_default_matrix() -> list[FeatScenario]:
    out: list[FeatScenario] = []
    md = _must_find_size("md")
    counts = (64, 1024, 16384)
    for n in counts:
        for pack in ALL_PACKS:
            out.append(_make_scenario(pack, md, n))
    for sid in ("xs", "md", "lg", "xlw"):
        size = _must_find_size(sid)
        out.append(_make_scenario(FeaturePack.SCORE, size, 1024))
    out.append(_make_scenario(FeaturePack.SCORE, md, 1024, 4))
    out.append(_make_scenario(FeaturePack.BINS, md, 0))
    return out


def build_size_sweep(size_ids: list[str]) -> list[FeatScenario]:
    out: list[FeatScenario] = []
    for sid in size_ids:
        size = _SIZE_BY_ID.get(sid)
        if size is None:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(_make_scenario(FeaturePack.SCORE, size, 1024))
    return out


def build_full_matrix() -> list[FeatScenario]:
    out: list[FeatScenario] = []
    counts = (0, 64, 256, 1024, 4096, 16384, 65536)
    for size in SIZE_LADDER:
        for n in counts:
            for pack in ALL_PACKS:
                out.append(_make_scenario(pack, size, n))
    md = _must_find_size("md")
    for b in (4, 16):
        out.append(_make_scenario(FeaturePack.SCORE, md, 1024, b))
        out.append(_make_scenario(FeaturePack.FULL, md, 4096, b))
    return out


# ---------------------------------------------------------------------------
# Row construction / throughput
# ---------------------------------------------------------------------------


def _base_row(scen: FeatScenario) -> ResultRow:
    row = ResultRow(
        scenario_id=scen.id,
        op_name="FeatureAssemblyOp",
        variant=scen.pack.value,
        size_id=scen.size_id,
        map_shape=[scen.doppler, scen.range_],
        input_shape=[scen.num_detections, 3 if scen.score_batch > 1 else 2],
        out_batch=scen.num_detections,
        k=len(_pack_features(scen.pack)),
    )
    if scen.score_batch > 1:
        row.notes = f"score_batch={scen.score_batch}"
    return row


def _fill_throughput(row: ResultRow, scen: FeatScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    n = float(scen.num_detections)
    cells = float(scen.doppler) * float(scen.range_) * float(max(scen.score_batch, 1))
    row.cells_per_second = n / sec  # detections/s (sparse axis)
    row.slices_per_second = float(max(scen.score_batch, 1)) / sec

    tsz = 4.0
    idx_bytes = n * 2.0 * 8.0
    bins_bytes = float(scen.doppler + scen.range_) * tsz
    score_bytes = cells * tsz if _pack_needs_score(scen.pack) else 0.0
    aoa_bytes = n * 2.0 * tsz if _pack_needs_aoa(scen.pack) else 0.0
    points_bytes = n * float(len(_pack_features(scen.pack))) * tsz
    row.estimated_effective_gbps = (
        idx_bytes + bins_bytes + score_bytes + aoa_bytes + points_bytes
    ) / sec / 1e9


def run_scenario(scen: FeatScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=False)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        collector = TimingCollector()

        idx = single_map(
            _make_detection_idx(scen.num_detections, scen.doppler, scen.range_, scen.score_batch),
            "detection_idx",
        )
        range_bins = single_map(_make_axis_bins(scen.range_), "range_bins")
        velocity_bins = single_map(_make_axis_bins(scen.doppler), "velocity_bins")
        score_shape = (
            [scen.score_batch, scen.doppler, scen.range_]
            if scen.score_batch > 1
            else [scen.doppler, scen.range_]
        )
        score = single_map(make_power_map_device(score_shape), "detection_score")
        aoa = single_map(_make_aoa(scen.num_detections), "aoa_estimates")

        collector.t_construct = now()
        _run_graph(
            scen=scen,
            collector=collector,
            idx=idx,
            range_bins=range_bins,
            velocity_bins=velocity_bins,
            score=score,
            aoa=aoa,
            total_frames=run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(rs, plan, run_rep)
    append_setup_row(_base_row(scen), samples, report)

    compute = _base_row(scen)
    fill_compute_row(compute, samples)
    _fill_throughput(compute, scen)
    if not compute.notes:
        compute.notes = (
            "cold_ms is first-frame latency; median/p90 are post-warmup "
            "steady state; cells/s is detections/s"
        )
    else:
        compute.notes += "; cold_ms first-frame; median/p90 steady; cells/s detections/s"
    report.results.append(compute)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_feature_assembly",
        description=(
            "FeatureAssemblyOp performance characterization (Python). "
            "Mirrors performance/cpp/perf_feature_assembly.cpp."
        ),
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        full_default=False,
        full_help="Full matrix (packs x sizes x detection counts)",
    )


if __name__ == "__main__":
    sys.exit(main())
