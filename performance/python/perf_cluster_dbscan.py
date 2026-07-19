# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for ClusterDBSCANOp (Python).

Measurement tool only: no timing assertions. Mirrors
``performance/cpp/perf_cluster_dbscan.cpp``: brute-force O(N²·F) DBSCAN over
a (num_points, num_features) feature matrix. Every compute run inlines a
mid-session [live] override (epsilon / min_points / max_clusters). The
workload is compact, well-separated blobs so the connected-components loop
converges in a few iterations; N and F both drive cost (FeatureAssembly
layouts are typically F∈{2,3,5}).
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
from holoscan.core import Application  # noqa: E402
from holoscan.radar import ClusterDBSCANOpF32  # noqa: E402

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
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.perf_source import PerfOverrideSourceOp, PerfTensorSourceOp  # noqa: E402
from utils.tensor_factory import single_map  # noqa: E402

# ---------------------------------------------------------------------------
# Size ladder — O(N² · F) sweeps. N spans compact→dense detection clouds; F
# spans the FeatureAssembly layouts that actually feed this op (2 = R+V,
# 3 = +score pre-angle, 5 = +az+el post-angle). Keep aligned with
# perf_cluster_dbscan.cpp.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeSpec:
    id: str
    num_points: int
    num_features: int
    num_clusters: int
    role: str

    @property
    def shape(self) -> list[int]:
        return [self.num_points, self.num_features]

    @property
    def cells(self) -> int:
        return self.num_points


SIZE_LADDER = [
    SizeSpec("xs", 256, 2, 8, "small; fast smoke (R+V)"),
    SizeSpec("sm", 1024, 2, 32, "compact 2-feature (R+V)"),
    SizeSpec("sm3", 1024, 3, 32, "compact 3-feature (R+V+score)"),
    SizeSpec("md", 2048, 2, 64, "mid-scale 2-feature"),
    SizeSpec("md3", 2048, 3, 64, "mid-scale 3-feature (pre-angle)"),
    SizeSpec("md5", 2048, 5, 64, "mid-scale 5-feature (post-angle)"),
    SizeSpec("lg", 4096, 2, 128, "dense 2-feature cloud"),
    SizeSpec("lg5", 4096, 5, 128, "dense 5-feature cloud"),
    SizeSpec("xl", 8192, 2, 256, "upper N bound (O(N²), 2-feature)"),
    SizeSpec("xl5", 8192, 5, 256, "upper N×F bound (O(N²F))"),
]

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}


def _must_find_size(size_id: str) -> SizeSpec:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


class OverrideKind(str, Enum):
    EPSILON = "epsilon"
    MIN_POINTS = "min_points"
    MAX_CLUSTERS = "max_clusters"


_OVERRIDE_TAG = {
    OverrideKind.EPSILON: "eps",
    OverrideKind.MIN_POINTS: "minpts",
    OverrideKind.MAX_CLUSTERS: "maxclus",
}


@dataclass
class DbscanScenario:
    id: str
    size_id: str
    num_points: int = 0
    num_features: int = 2
    num_clusters: int = 1
    epsilon: list[float] = field(default_factory=lambda: [0.5])
    min_points: int = 0
    max_clusters: int = 0
    # Filled per run for the inline live-override measurement.
    override_kind: OverrideKind = OverrideKind.EPSILON
    override_at: int = 0
    override_epsilon: list[float] = field(default_factory=lambda: [0.6])
    override_min_points: int = 0
    override_max_clusters: int = 0


# ---------------------------------------------------------------------------
# Deterministic compact blobs: cluster c centered at (c·100, 0, …) with a
# small (±0.1) per-axis jitter so within-blob points are eps-neighbors and
# separate blobs are not. No RNG (keeps runs reproducible).
# ---------------------------------------------------------------------------


def _make_points(num_points: int, num_features: int, num_clusters: int) -> cp.ndarray:
    nclus = max(num_clusters, 1)
    spacing = 100.0
    n = max(num_points, 0)
    f = max(num_features, 0)
    host = np.zeros((n, f), dtype=np.float32)
    for i in range(num_points):
        c = i % nclus
        for feat in range(num_features):
            base = float(c) * spacing if feat == 0 else 0.0
            jitter = (((i * 131 + feat * 17) % 21) - 10) / 100.0
            host[i, feat] = base + jitter
    return cp.asarray(host)


def _run_graph(
    *,
    scen: DbscanScenario,
    collector: TimingCollector,
    points: dict,
    total_frames: int,
) -> None:
    class DbscanPerfApp(Application):
        def compose(self) -> None:
            op = ClusterDBSCANOpF32(
                self,
                epsilon=scen.epsilon,
                min_points=scen.min_points,
                max_clusters=scen.max_clusters,
                name="dbscan",
            )

            src = PerfTensorSourceOp(
                self,
                CountCondition(self, count=total_frames),
                output_port="points",
                frame=points,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("points", "points")})

            term = PerfTerminalSinkOp(self, collector=collector, name="labels_sink")
            self.add_flow(op, term, {("labels", "input")})

            # sizes must be wired for the op to tick; it is not the timed
            # result.
            sizes_drain = make_null_sink(self, "sizes_drain")
            self.add_flow(op, sizes_drain, {("sizes", "input")})

            self._wire_override(op, total_frames)

        def _wire_override(self, op, total_frames: int) -> None:
            if scen.override_kind == OverrideKind.EPSILON:
                ovr = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=scen.override_epsilon,
                    name="ovr",
                )
                self.add_flow(ovr, op, {("output", "epsilon")})
            elif scen.override_kind == OverrideKind.MIN_POINTS:
                ovr = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=scen.override_min_points,
                    name="ovr",
                )
                self.add_flow(ovr, op, {("output", "min_points")})
            else:
                ovr = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=scen.override_max_clusters,
                    name="ovr",
                )
                self.add_flow(ovr, op, {("output", "max_clusters")})

    DbscanPerfApp().run()


# ---------------------------------------------------------------------------
# Row construction / throughput
# ---------------------------------------------------------------------------


def _fill_throughput(row: ResultRow, scen: DbscanScenario) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    n = float(scen.num_points)
    row.cells_per_second = n / sec  # points/s
    row.slices_per_second = float(max(scen.num_clusters, 1)) / sec

    tsz = 4.0
    points_bytes = n * float(scen.num_features) * tsz
    labels_bytes = n * 8.0
    row.estimated_effective_gbps = (points_bytes + labels_bytes) / sec / 1e9


def _base_row(scen: DbscanScenario) -> ResultRow:
    return ResultRow(
        scenario_id=scen.id,
        op_name="ClusterDBSCANOp",
        variant="dbscan",
        size_id=scen.size_id,
        map_shape=[scen.num_points, scen.num_features],
        input_shape=[scen.num_points, scen.num_features],
        out_batch=scen.num_clusters,
        k=scen.num_features,
        notes=f"live_override={scen.override_kind.value}",
    )


def _make_scenario(size: SizeSpec, ovr: OverrideKind = OverrideKind.EPSILON) -> DbscanScenario:
    sid = (
        f"dbscan_{size.id}_n{size.num_points}_f{size.num_features}_"
        f"{_OVERRIDE_TAG[ovr]}"
    )
    return DbscanScenario(
        id=sid,
        size_id=size.id,
        num_points=size.num_points,
        num_features=size.num_features,
        num_clusters=size.num_clusters,
        epsilon=[0.5],
        min_points=2 * size.num_features,
        max_clusters=0,
        override_kind=ovr,
    )


def build_default_matrix() -> list[DbscanScenario]:
    out: list[DbscanScenario] = []
    # N ladder at F=2, plus mid-scale F=3/5 and dense F=5 so the default
    # report covers both axes without the full --sizes all matrix.
    for sid in ("xs", "sm", "sm3", "md", "md3", "md5", "lg", "lg5"):
        out.append(_make_scenario(_must_find_size(sid)))
    # Distinct live params beyond the default epsilon override.
    md = _must_find_size("md")
    out.append(_make_scenario(md, OverrideKind.MIN_POINTS))
    out.append(_make_scenario(md, OverrideKind.MAX_CLUSTERS))
    return out


def build_size_sweep(size_ids: list[str]) -> list[DbscanScenario]:
    out: list[DbscanScenario] = []
    for sid in size_ids:
        size = _SIZE_BY_ID.get(sid)
        if size is None:
            print(f"warning: unknown size id '{sid}', skipping", file=sys.stderr)
            continue
        out.append(_make_scenario(size))
    return out


def build_full_matrix() -> list[DbscanScenario]:
    out: list[DbscanScenario] = []
    for size in SIZE_LADDER:
        out.append(_make_scenario(size, OverrideKind.EPSILON))
        out.append(_make_scenario(size, OverrideKind.MIN_POINTS))
        out.append(_make_scenario(size, OverrideKind.MAX_CLUSTERS))
    return out


# Run one scenario (all repeats). Each run measures setup, pre-override
# steady-state compute, and an inline [live] override spike in one session.
def run_scenario(scen: DbscanScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        run_scen = copy.deepcopy(scen)
        run_scen.override_at = run_plan.override_at
        if run_scen.override_kind == OverrideKind.EPSILON:
            run_scen.override_epsilon = [0.6]  # re-resolve vector override
        elif run_scen.override_kind == OverrideKind.MIN_POINTS:
            run_scen.override_min_points = run_scen.min_points + 1
        else:
            # Cap below the blob count to exercise the drop-surplus path.
            run_scen.override_max_clusters = max(1, run_scen.num_clusters // 2)

        collector = TimingCollector()
        points = single_map(
            _make_points(scen.num_points, scen.num_features, scen.num_clusters), "points"
        )

        collector.t_construct = now()
        _run_graph(
            scen=run_scen,
            collector=collector,
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


# Pay CUDA + Holoscan first-Application costs so scenario #1 cold_ms is
# comparable to later rows.
def _process_warmup(_rs: RunSettings) -> None:
    discarded = PerfReport()
    warm = RunSettings(warmup_frames=0, measured_frames=2, repeats=1)
    run_scenario(_make_scenario(_must_find_size("xs")), warm, discarded)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_cluster_dbscan",
        description=(
            "ClusterDBSCANOp performance characterization (Python). "
            "Mirrors performance/cpp/perf_cluster_dbscan.cpp."
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
        full_help="Full matrix (sizes x {epsilon, min_points, max_clusters} overrides)",
    )


if __name__ == "__main__":
    sys.exit(main())
