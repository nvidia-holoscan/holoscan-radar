# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for DetectionReductionOp (Python).

Measurement tool only: no timing assertions. Mirrors
``performance/cpp/perf_detection_reduction.cpp``: group-by-label reduction
over (labels, detection_idx, points). Every compute run inlines a mid-session
[live] override (representative / feature_weights / score_method).
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
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
from holoscan.radar import (  # noqa: E402
    DetectionReductionOpF32,
    RepresentativePolicy,
    ScoreMethod,
)

from utils.perf_app import TimingCollector, now  # noqa: E402
from utils.perf_driver import run_perf_matrix_main  # noqa: E402
from utils.perf_report import PerfReport, ResultRow, RunSettings  # noqa: E402
from utils.perf_runner import (  # noqa: E402
    RunPlan,
    append_setup_row,
    collect_scenario_samples,
    fill_compute_row,
    make_run_plan,
)
from utils.perf_sink import PerfTerminalSinkOp  # noqa: E402
from utils.perf_source import PerfOverrideSourceOp  # noqa: E402
from utils.tensor_factory import single_map  # noqa: E402

# ---------------------------------------------------------------------------
# Size ladder — keep aligned with perf_detection_reduction.cpp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeSpec:
    id: str
    num_points: int
    num_clusters: int
    num_features: int
    role: str

    @property
    def shape(self) -> list[int]:
        return [self.num_points, self.num_features]

    @property
    def cells(self) -> int:
        return self.num_points


SIZE_LADDER = [
    SizeSpec("xs", 256, 32, 2, "small; fast smoke"),
    SizeSpec("sm", 1024, 64, 2, "compact detection set"),
    SizeSpec("md", 4096, 128, 2, "common mid-scale"),
    SizeSpec("lg", 16384, 256, 2, "dense cluster-then-angle"),
    SizeSpec("xl", 65536, 512, 2, "upper characterization bound"),
]

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeSpec:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


class OverrideKind(str, Enum):
    REPRESENTATIVE = "representative"
    FEATURE_WEIGHTS = "feature_weights"
    SCORE_METHOD = "score_method"


_OVERRIDE_TAG = {
    OverrideKind.REPRESENTATIVE: "rep",
    OverrideKind.FEATURE_WEIGHTS: "wts",
    OverrideKind.SCORE_METHOD: "score",
}

ALL_POLICIES = (
    RepresentativePolicy.FIRST,
    RepresentativePolicy.PEAK,
    RepresentativePolicy.MEDOID,
)


def _policy_tag(policy: RepresentativePolicy) -> str:
    return policy.name.lower()


@dataclass
class RedScenario:
    id: str
    size_id: str
    policy: RepresentativePolicy = RepresentativePolicy.PEAK
    score_method: ScoreMethod = ScoreMethod.WEIGHTED_SUM
    num_points: int = 0
    num_clusters: int = 0
    num_features: int = 2
    # Filled per run by run_scenario() for inline live-override measurement.
    override_kind: OverrideKind = OverrideKind.REPRESENTATIVE
    override_at: int = 0
    override_policy: RepresentativePolicy = RepresentativePolicy.FIRST
    override_score_method: ScoreMethod = ScoreMethod.WEIGHTED_L2
    override_weights: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic device tensor factories (mirror the C++ host-fill formulas)
# ---------------------------------------------------------------------------


def _make_labels(num_points: int, num_clusters: int) -> cp.ndarray:
    """~1/8 noise, remaining round-robin across clusters."""
    nclus = max(num_clusters, 1)
    host = np.zeros(max(num_points, 0), dtype=np.int64)
    for i in range(num_points):
        host[i] = -1 if i % 8 == 0 else i % nclus
    return cp.asarray(host)


def _make_detection_idx(num_points: int) -> cp.ndarray:
    host = np.zeros((max(num_points, 0), 2), dtype=np.int64)
    for i in range(num_points):
        host[i, 0] = i % 1024
        host[i, 1] = (i * 3) % 2048
    return cp.asarray(host)


def _make_points(num_points: int, num_features: int) -> cp.ndarray:
    n = max(num_points, 0)
    f = max(num_features, 0)
    host = np.zeros((n, f), dtype=np.float32)
    for i in range(num_points):
        for feat in range(num_features):
            host[i, feat] = ((i * 17 + feat * 3) % 100) * 0.01
    return cp.asarray(host)


# ---------------------------------------------------------------------------
# Multi-port source (labels, detection_idx, points): emits the same
# preallocated TensorMaps every tick and records start times.
# ---------------------------------------------------------------------------


class ReductionPerfSourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        labels: dict,
        idx: dict,
        points: dict,
        collector: TimingCollector,
        **kwargs,
    ):
        self._labels = labels
        self._idx = idx
        self._points = points
        self._collector = collector
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.output("labels")
        spec.output("detection_idx")
        spec.output("points")

    def compute(self, op_input, op_output, context) -> None:
        if not self._collector.first_tick_set:
            self._collector.t_first_tick = now()
            self._collector.first_tick_set = True
        op_output.emit(self._labels, "labels")
        op_output.emit(self._idx, "detection_idx")
        op_output.emit(self._points, "points")
        self._collector.starts.append(now())


def _run_graph(
    *,
    scen: RedScenario,
    collector: TimingCollector,
    labels: dict,
    idx: dict,
    points: dict,
    total_frames: int,
) -> None:
    class ReductionPerfApp(Application):
        def compose(self) -> None:
            # Always supply feature_weights + wire points so a mid-run
            # override to peak/medoid (or a weights/score_method flip) is
            # legal without rewiring.
            weights = [1.0] * scen.num_features
            op = DetectionReductionOpF32(
                self,
                representative=scen.policy,
                score_method=scen.score_method,
                feature_weights=weights,
                name="reduction",
            )

            src = ReductionPerfSourceOp(
                self,
                CountCondition(self, count=total_frames),
                labels=labels,
                idx=idx,
                points=points,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("labels", "labels")})
            self.add_flow(src, op, {("detection_idx", "detection_idx")})
            self.add_flow(src, op, {("points", "points")})

            term = PerfTerminalSinkOp(self, collector=collector, name="rep_sink")
            self.add_flow(op, term, {("rep_idx", "input")})

            if scen.override_kind == OverrideKind.REPRESENTATIVE:
                ovr = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=scen.override_policy,
                    name="ovr",
                )
                self.add_flow(ovr, op, {("output", "representative")})
            elif scen.override_kind == OverrideKind.FEATURE_WEIGHTS:
                ovr = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=scen.override_weights,
                    name="ovr",
                )
                self.add_flow(ovr, op, {("output", "feature_weights")})
            else:
                ovr = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=scen.override_score_method,
                    name="ovr",
                )
                self.add_flow(ovr, op, {("output", "score_method")})

    ReductionPerfApp().run()


def _flip_policy(policy: RepresentativePolicy) -> RepresentativePolicy:
    """Prefer a flip that changes the kernel path (first ↔ peak/medoid)."""
    if policy == RepresentativePolicy.FIRST:
        return RepresentativePolicy.PEAK
    return RepresentativePolicy.FIRST


# ---------------------------------------------------------------------------
# Row construction / throughput
# ---------------------------------------------------------------------------


def _fill_throughput(row: ResultRow, scen: RedScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    n = float(scen.num_points)
    row.cells_per_second = n / sec
    row.slices_per_second = float(max(scen.num_clusters, 1)) / sec

    tsz = 4.0
    labels_bytes = n * 8.0
    idx_bytes = n * 2.0 * 8.0
    # Points stay wired for every scenario (override may switch to
    # peak/medoid).
    points_bytes = n * float(scen.num_features) * tsz
    rep_bytes = float(scen.num_clusters) * 2.0 * 8.0
    row.estimated_effective_gbps = (
        labels_bytes + idx_bytes + points_bytes + rep_bytes
    ) / sec / 1e9


def _base_row(scen: RedScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="DetectionReductionOp",
        variant=_policy_tag(scen.policy),
        size_id=scen.size_id,
        map_shape=[scen.num_points, scen.num_features],
        input_shape=[scen.num_points, 2],
        out_batch=scen.num_clusters,
        k=scen.num_features,
        notes=f"live_override={scen.override_kind.value}",
    )


def _make_scenario(
    policy: RepresentativePolicy,
    size: SizeSpec,
    ovr: OverrideKind = OverrideKind.REPRESENTATIVE,
) -> RedScenario:
    # Weights / score_method overrides need a points-consuming policy.
    if ovr in (OverrideKind.FEATURE_WEIGHTS, OverrideKind.SCORE_METHOD):
        policy = RepresentativePolicy.PEAK
    # Keep ids short so the shared text-report scenario column (40) aligns.
    sid = f"red_{_policy_tag(policy)}_{size.id}_n{size.num_points}_{_OVERRIDE_TAG[ovr]}"
    return RedScenario(
        id=sid,
        size_id=size.id,
        policy=policy,
        num_points=size.num_points,
        num_clusters=size.num_clusters,
        num_features=size.num_features,
        override_kind=ovr,
    )


def build_default_matrix() -> list[RedScenario]:
    out: list[RedScenario] = []
    md = _must_find_size("md")
    for p in ALL_POLICIES:
        out.append(_make_scenario(p, md))
    for sid in ("xs", "sm", "md", "lg"):
        out.append(_make_scenario(RepresentativePolicy.PEAK, _must_find_size(sid)))
    # Distinct live params beyond the default representative flip.
    out.append(_make_scenario(RepresentativePolicy.PEAK, md, OverrideKind.FEATURE_WEIGHTS))
    out.append(_make_scenario(RepresentativePolicy.PEAK, md, OverrideKind.SCORE_METHOD))
    return out


def build_size_sweep(size_ids: list[str]) -> list[RedScenario]:
    out: list[RedScenario] = []
    for sid in size_ids:
        size = _SIZE_BY_ID.get(sid)
        if size is None:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(_make_scenario(RepresentativePolicy.PEAK, size))
    return out


def build_full_matrix() -> list[RedScenario]:
    out: list[RedScenario] = []
    for size in SIZE_LADDER:
        for p in ALL_POLICIES:
            out.append(_make_scenario(p, size))
        out.append(_make_scenario(RepresentativePolicy.PEAK, size, OverrideKind.FEATURE_WEIGHTS))
        out.append(_make_scenario(RepresentativePolicy.PEAK, size, OverrideKind.SCORE_METHOD))
    return out


# Run one scenario (all repeats). Each run measures setup, pre-override
# steady-state compute, and an inline [live] override spike in one session.
def run_scenario(scen: RedScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        run_scen = copy.deepcopy(scen)
        run_scen.override_at = run_plan.override_at
        if run_scen.override_kind == OverrideKind.REPRESENTATIVE:
            run_scen.override_policy = _flip_policy(run_scen.policy)
        elif run_scen.override_kind == OverrideKind.FEATURE_WEIGHTS:
            weights = [0.0] * run_scen.num_features
            if weights:
                weights[0] = 1.0
            run_scen.override_weights = weights
        else:
            run_scen.override_score_method = ScoreMethod.WEIGHTED_L2

        collector = TimingCollector()
        labels = single_map(_make_labels(scen.num_points, scen.num_clusters), "labels")
        idx = single_map(_make_detection_idx(scen.num_points), "detection_idx")
        points = single_map(_make_points(scen.num_points, scen.num_features), "points")

        collector.t_construct = now()
        _run_graph(
            scen=run_scen,
            collector=collector,
            labels=labels,
            idx=idx,
            points=points,
            total_frames=run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(rs, plan, run_rep)
    append_setup_row(_base_row(scen), samples, report)

    compute = _base_row(scen)
    fill_compute_row(compute, samples)
    _fill_throughput(compute, scen)
    compute.notes = (
        "cold_ms is first-frame latency; median/p90 are pre-override "
        "steady state; override_ms is peak latency over the 3-frame "
        f"live-{scen.override_kind.value} window; cells/s is points/s"
    )
    report.results.append(compute)


# Pay CUDA + Holoscan first-Application costs (and first-touch each policy
# kernel) so scenario #1 cold_ms is comparable to later rows.
def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    xs = _must_find_size("xs")
    for p in ALL_POLICIES:
        run_scenario(_make_scenario(p, xs), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_detection_reduction",
        description=(
            "DetectionReductionOp performance characterization (Python). "
            "Mirrors performance/cpp/perf_detection_reduction.cpp."
        ),
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        process_warmup=_process_warmup,
        full_default=False,
        full_help=(
            "Full matrix (policies x sizes + feature_weights/score_method overrides)"
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
