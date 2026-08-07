# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Performance characterization for CFARDetectorOp (Python).

Measurement tool only: no timing assertions.

Scenario matrices intentionally mirror
``performance/cpp/perf_cfar_detector.cpp`` (default / --full / --sizes).
C++/Python matrix sync is an open question — see
``performance/OPEN_ITEMS.md``.

Default CLI matches C++ (default matrix, not ``--full``): mid/large GO/SO/OS
footprints are Monte Carlo–heavy (~tens of seconds cold) and are only in
``--full``. Curated pfa↔alpha_lut calibration pairs are included in both.
Default also includes ``peak_only`` NMS rows (CA × default sizes ×
``{square8, cross4}``); ``--full`` extends those across the size ladder and
adds GO/SO/OS at ``md``.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass, field

_PERF_DIR = os.path.dirname(os.path.abspath(__file__))
if _PERF_DIR not in sys.path:
    sys.path.insert(0, _PERF_DIR)

from utils.bootstrap import bootstrap_radar_package  # noqa: E402

bootstrap_radar_package()

from utils.perf_app import TimingCollector, now  # noqa: E402
from utils.perf_driver import run_perf_matrix_main  # noqa: E402
from utils.perf_report import PerfReport  # noqa: E402
from utils.perf_report import AlphaCalibrationRow, ResultRow, RunSettings, SizeDef
from utils.perf_runner import append_setup_row  # noqa: E402
from utils.perf_runner import (
    RunPlan,
    collect_scenario_samples,
    fill_compute_row,
    make_run_plan,
)
from utils.perf_sink import PerfTerminalSinkOp, make_null_sink  # noqa: E402
from utils.perf_source import PerfDataSourceOp, PerfOverrideSourceOp  # noqa: E402
from utils.tensor_factory import make_power_map_device, single_map  # noqa: E402

from holoscan.conditions import CountCondition  # noqa: E402
from holoscan.core import Application  # noqa: E402
from holoscan.radar import (  # noqa: E402
    CFARDetectorOpF32,
    CfarPeakNeighborhood,
    CfarVariant,
    build_ca_alpha_lut,
    build_go_alpha_lut,
    build_os_alpha_lut,
    build_so_alpha_lut,
)

# ---------------------------------------------------------------------------
# Size ladder / footprints — keep aligned with perf_cfar_detector.cpp
# ---------------------------------------------------------------------------

SIZE_LADDER = [
    SizeDef("xs", [64, 256], 64 * 256, "small real map; fast smoke; border-heavy"),
    SizeDef("sm", [128, 512], 128 * 512, "compact sensor / few chirps"),
    SizeDef("md", [256, 512], 256 * 512, "common dev size"),
    SizeDef("mdw", [128, 1024], 128 * 1024, "range-heavy, same cell count as md"),
    SizeDef("lgs", [512, 512], 512 * 512, "square mid-scale"),
    SizeDef("lg", [512, 1024], 512 * 1024, "moderate production-scale"),
    SizeDef("xl", [1024, 1024], 1024 * 1024, "large moderate"),
    SizeDef("xlw", [1024, 2048], 1024 * 2048, "range-heavy moderate (v1 upper bound)"),
]

_SIZE_BY_ID = {item.id: item for item in SIZE_LADDER}

ALL_VARIANTS = (CfarVariant.CA, CfarVariant.GO, CfarVariant.SO, CfarVariant.OS)

ALL_PEAK_NEIGHBORHOODS = (
    CfarPeakNeighborhood.SQUARE8,
    CfarPeakNeighborhood.CROSS4,
)

CALIBRATION_PAIR_SPECS = (
    ("cfar_GO_xlw_small_k1_compute", "GO_xlw_k1"),
    ("cfar_SO_xlw_small_k1_compute", "SO_xlw_k1"),
    ("cfar_GO_md_small_k8_compute", "GO_md_k8"),
    ("cfar_SO_md_small_k8_compute", "SO_md_k8"),
    ("cfar_OS_md_small_k8_compute", "OS_md_k8"),
    ("cfar_OS_lg_small_k8_compute", "OS_lg_k8"),
)

_LUT_CACHE: dict[tuple, list[float]] = {}


@dataclass(frozen=True)
class Footprint:
    name: str
    guard: tuple[int, int]
    training: tuple[int, int]


FOOTPRINT_SMALL = Footprint("small", (1, 1), (2, 2))
FOOTPRINT_MID = Footprint("mid", (2, 2), (4, 4))
FOOTPRINT_LARGE = Footprint("large", (2, 2), (8, 8))


@dataclass
class CfarScenario:
    id: str
    size_id: str
    variant: CfarVariant
    input_shape: list[int]
    footprint: Footprint
    nci_axes: list[int] = field(default_factory=list)
    os_rank: int = 0
    pfa: float = 1e-4
    use_alpha_lut: bool = False
    alpha_lut: list[float] = field(default_factory=list)
    calibration_pair: str = ""
    # Optional peak-grouping (NMS) stage — static; off by default.
    peak_only: bool = False
    peak_neighborhood: CfarPeakNeighborhood = CfarPeakNeighborhood.SQUARE8
    # Filled per run for live-override wiring.
    override_at: int = 0
    override_pfa: float = 1e-6
    override_alpha_lut: list[float] = field(default_factory=list)
    blocking_rebuild: bool = False


def _variant_tag(variant: CfarVariant) -> str:
    name = getattr(variant, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(variant).rsplit(".", maxsplit=1)[-1]


def _interior_training(fp: Footprint) -> int:
    hd = fp.guard[0] + fp.training[0]
    hr = fp.guard[1] + fp.training[1]
    window = (2 * hd + 1) * (2 * hr + 1)
    guard_box = (2 * fp.guard[0] + 1) * (2 * fp.guard[1] + 1)
    return window - guard_box


def _batch_and_k(shape: list[int], nci_axes: list[int]) -> tuple[int, int]:
    out_batch = 1
    k = 1
    lead = len(shape) - 2
    nci = set(nci_axes)
    for axis in range(lead):
        if axis in nci:
            k *= shape[axis]
        else:
            out_batch *= shape[axis]
    return out_batch, k


def _must_find_size(size_id: str) -> SizeDef:
    try:
        return _SIZE_BY_ID[size_id]
    except KeyError as exc:
        raise ValueError(f"unknown size id '{size_id}'") from exc


def make_compute_scenario(
    variant: CfarVariant, size: SizeDef, fp: Footprint, out_batch: int
) -> CfarScenario:
    doppler, range_bins = size.shape
    if out_batch > 1:
        input_shape = [out_batch, doppler, range_bins]
    else:
        input_shape = [doppler, range_bins]
    os_rank = 0
    if variant == CfarVariant.OS:
        os_rank = (_interior_training(fp) * 3) // 4
    sid = f"cfar_{_variant_tag(variant)}_{size.id}_{fp.name}"
    if out_batch > 1:
        sid += f"_b{out_batch}"
    sid += "_k1_compute"
    return CfarScenario(
        id=sid,
        size_id=size.id,
        variant=variant,
        input_shape=input_shape,
        footprint=fp,
        os_rank=os_rank,
    )


def _peak_neighborhood_tag(nb: CfarPeakNeighborhood) -> str:
    name = getattr(nb, "name", None)
    if isinstance(name, str) and name:
        # CROSS4 / SQUARE8 → cross4 / square8 (match C++ to_string)
        return name.lower()
    return str(nb).rsplit(".", maxsplit=1)[-1].lower()


def make_peak_scenario(
    variant: CfarVariant,
    size: SizeDef,
    fp: Footprint,
    neighborhood: CfarPeakNeighborhood,
) -> CfarScenario:
    """Compute scenario with peak_only NMS; id inserts ``_peak_<nb>``."""
    sc = make_compute_scenario(variant, size, fp, out_batch=1)
    sc.peak_only = True
    sc.peak_neighborhood = neighborhood
    insert = f"_peak_{_peak_neighborhood_tag(neighborhood)}"
    if sc.id.endswith("_compute"):
        sc.id = sc.id[: -len("_compute")] + insert + "_compute"
    else:
        sc.id += insert
    return sc


def append_peak_only_scenarios(
    scenarios: list[CfarScenario],
    size_ids: list[str],
    *,
    all_variants_at_md: bool,
) -> None:
    """Mirror C++ append_peak_only_scenarios."""
    fp = FOOTPRINT_SMALL
    for sid in size_ids:
        size = _must_find_size(sid)
        for nb in ALL_PEAK_NEIGHBORHOODS:
            scenarios.append(make_peak_scenario(CfarVariant.CA, size, fp, nb))
    if not all_variants_at_md:
        return
    md = _must_find_size("md")
    for variant in ALL_VARIANTS:
        if variant == CfarVariant.CA:
            continue
        for nb in ALL_PEAK_NEIGHBORHOODS:
            scenarios.append(make_peak_scenario(variant, md, fp, nb))


def make_nci_scenario(variant: CfarVariant, size: SizeDef, channels: int) -> CfarScenario:
    doppler, range_bins = size.shape
    fp = FOOTPRINT_SMALL
    os_rank = 0
    if variant == CfarVariant.OS:
        os_rank = (_interior_training(fp) * 3) // 4
    return CfarScenario(
        id=f"cfar_{_variant_tag(variant)}_{size.id}_small_k{channels}_compute",
        size_id=size.id,
        variant=variant,
        input_shape=[channels, doppler, range_bins],
        footprint=fp,
        nci_axes=[0],
        os_rank=os_rank,
    )


def precompute_alpha_lut(scen: CfarScenario) -> list[float]:
    """Offline LUT build (may MC for GO/SO/OS); cached across scenarios."""
    _, fold = _batch_and_k(scen.input_shape, scen.nci_axes)
    n = len(scen.input_shape)
    num_doppler = scen.input_shape[n - 2]
    num_range = scen.input_shape[n - 1]
    gd, gr = scen.footprint.guard
    td, tr = scen.footprint.training
    key = (
        _variant_tag(scen.variant),
        scen.pfa,
        fold,
        scen.os_rank,
        num_doppler,
        num_range,
        gd,
        gr,
        td,
        tr,
    )
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return list(cached)

    if scen.variant == CfarVariant.CA:
        lut = build_ca_alpha_lut(scen.pfa, fold, num_doppler, num_range, gd, gr, td, tr)
    elif scen.variant == CfarVariant.GO:
        lut = build_go_alpha_lut(scen.pfa, fold, num_doppler, num_range, gd, gr, td, tr)
    elif scen.variant == CfarVariant.SO:
        lut = build_so_alpha_lut(scen.pfa, fold, num_doppler, num_range, gd, gr, td, tr)
    elif scen.variant == CfarVariant.OS:
        lut = build_os_alpha_lut(
            scen.pfa, fold, scen.os_rank, num_doppler, num_range, gd, gr, td, tr
        )
    else:
        raise ValueError(f"unsupported CFAR variant {scen.variant}")
    lut_list = [float(x) for x in lut]
    _LUT_CACHE[key] = lut_list
    return list(lut_list)


def tag_and_append_alpha_lut_pairs(scenarios: list[CfarScenario]) -> list[CfarScenario]:
    """Mirror C++ tag_and_append_alpha_lut_pairs."""
    by_id = {sc.id: sc for sc in scenarios}
    for slow_id, pair_id in CALIBRATION_PAIR_SPECS:
        slow = by_id.get(slow_id)
        if slow is None:
            continue
        for sc in scenarios:
            if sc.id == slow_id:
                sc.calibration_pair = pair_id
                sc.use_alpha_lut = False
        fast = copy.deepcopy(slow)
        fast.calibration_pair = pair_id
        fast.use_alpha_lut = True
        fast.alpha_lut = []
        if fast.id.endswith("_compute"):
            fast.id = fast.id[: -len("_compute")] + "_alphalut_compute"
        else:
            fast.id += "_alphalut"
        scenarios.append(fast)
        by_id[fast.id] = fast
    return scenarios


def build_default_matrix() -> list[CfarScenario]:
    out: list[CfarScenario] = []
    fp = FOOTPRINT_SMALL
    for sid in ("xs", "md", "lg", "xlw"):
        size = _must_find_size(sid)
        for variant in ALL_VARIANTS:
            out.append(make_compute_scenario(variant, size, fp, 1))
    md = _must_find_size("md")
    for variant in ALL_VARIANTS:
        out.append(make_compute_scenario(variant, md, fp, 4))
    for sid in ("md", "lg"):
        size = _must_find_size(sid)
        for variant in ALL_VARIANTS:
            out.append(make_nci_scenario(variant, size, 8))
    tag_and_append_alpha_lut_pairs(out)
    # peak_only NMS characterization (CA × default sizes × both neighborhoods).
    append_peak_only_scenarios(out, ["xs", "md", "lg", "xlw"], all_variants_at_md=False)
    return out


def build_size_sweep(size_ids: list[str]) -> list[CfarScenario]:
    out: list[CfarScenario] = []
    fp = FOOTPRINT_SMALL
    known: list[str] = []
    for sid in size_ids:
        size = _must_find_size(sid)
        out.append(make_compute_scenario(CfarVariant.CA, size, fp, 1))
        known.append(sid)
    # Include peak_only NMS on the same CA×sizes subset so --size(s) / smoke
    # exercises the new stage (both neighborhoods).
    append_peak_only_scenarios(out, known, all_variants_at_md=False)
    return out


def build_full_matrix() -> list[CfarScenario]:
    out: list[CfarScenario] = []
    fps = (FOOTPRINT_SMALL, FOOTPRINT_MID, FOOTPRINT_LARGE)
    for size in SIZE_LADDER:
        for variant in ALL_VARIANTS:
            for fp in fps:
                out.append(make_compute_scenario(variant, size, fp, 1))
    for sid in ("md", "lg"):
        size = _must_find_size(sid)
        for variant in ALL_VARIANTS:
            out.append(make_compute_scenario(variant, size, FOOTPRINT_SMALL, 4))
            out.append(make_compute_scenario(variant, size, FOOTPRINT_SMALL, 16))
    for sid in ("md", "lg"):
        size = _must_find_size(sid)
        for variant in ALL_VARIANTS:
            out.append(make_nci_scenario(variant, size, 8))
    tag_and_append_alpha_lut_pairs(out)
    # peak_only: CA × full ladder × both neighborhoods; GO/SO/OS at md.
    append_peak_only_scenarios(
        out,
        [item.id for item in SIZE_LADDER],
        all_variants_at_md=True,
    )
    return out


def _base_row(scen: CfarScenario) -> ResultRow:
    out_batch, k = _batch_and_k(scen.input_shape, scen.nci_axes)
    n = len(scen.input_shape)
    return ResultRow(
        scenario_id=scen.id,
        op_name="CFARDetectorOp",
        variant=_variant_tag(scen.variant),
        size_id=scen.size_id,
        map_shape=list(scen.input_shape[n - 2 :]),
        input_shape=list(scen.input_shape),
        out_batch=out_batch,
        guard=list(scen.footprint.guard),
        training=list(scen.footprint.training),
        k=k,
        calibration_pair=scen.calibration_pair,
        calibration_mode="alpha_lut" if scen.use_alpha_lut else "pfa",
    )


def _fill_throughput(row: ResultRow) -> None:
    if row.latency.median_ms <= 0:
        return
    sec = row.latency.median_ms / 1000.0
    doppler, range_bins = row.map_shape
    cells = float(row.out_batch) * float(doppler) * float(range_bins)
    row.cells_per_second = cells / sec
    row.slices_per_second = float(row.out_batch) / sec
    tsz = 4.0
    in_bytes = cells * float(row.k) * tsz
    out_bytes = cells * (2.0 * tsz + 1.0)
    row.estimated_effective_gbps = (in_bytes + out_bytes) / sec / 1e9


def _run_graph(
    *,
    scen: CfarScenario,
    collector: TimingCollector,
    frame: dict,
    total_frames: int,
) -> None:
    class CfarPerfApp(Application):
        def compose(self) -> None:
            op = CFARDetectorOpF32(
                self,
                cfar_variant=scen.variant,
                guard_cells=list(scen.footprint.guard),
                training_cells=list(scen.footprint.training),
                pfa=scen.pfa,
                alpha_lut=list(scen.alpha_lut) if scen.use_alpha_lut else [],
                threshold_scale=1.0,
                nci_axes=list(scen.nci_axes),
                os_rank=scen.os_rank,
                max_detections=0,
                blocking_alpha_lut_rebuild=scen.blocking_rebuild,
                peak_only=scen.peak_only,
                peak_neighborhood=scen.peak_neighborhood,
                name="cfar",
            )
            src = PerfDataSourceOp(
                self,
                CountCondition(self, count=total_frames),
                frame=frame,
                collector=collector,
                name="src",
            )
            self.add_flow(src, op, {("rd_maps", "rd_maps")})

            idx_sink = PerfTerminalSinkOp(self, collector=collector, name="idx_sink")
            self.add_flow(op, idx_sink, {("detection_idx", "input")})

            for port in ("detection_mask", "detection_score", "noise_estimate"):
                null_sink = make_null_sink(self, f"{port}_sink")
                self.add_flow(op, null_sink, {(port, "input")})

            if scen.use_alpha_lut:
                override = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=list(scen.override_alpha_lut),
                    name="ovr",
                )
                self.add_flow(override, op, {("output", "alpha_lut")})
            else:
                override = PerfOverrideSourceOp(
                    self,
                    CountCondition(self, count=total_frames),
                    target_tick=scen.override_at,
                    value=scen.override_pfa,
                    name="ovr",
                )
                self.add_flow(override, op, {("output", "pfa")})

    CfarPerfApp().run()


def run_scenario(scen: CfarScenario, rs: RunSettings, report: PerfReport) -> None:
    plan = make_run_plan(rs, has_override=True)

    def run_rep(_rep: int, run_plan: RunPlan) -> TimingCollector:
        run_scen = copy.deepcopy(scen)
        if run_scen.use_alpha_lut:
            if not run_scen.alpha_lut:
                run_scen.alpha_lut = precompute_alpha_lut(run_scen)
            override_scen = copy.deepcopy(run_scen)
            override_scen.pfa = 1e-6
            run_scen.override_alpha_lut = precompute_alpha_lut(override_scen)
            run_scen.blocking_rebuild = False
        else:
            run_scen.override_pfa = 1e-6
            run_scen.blocking_rebuild = True
        run_scen.override_at = run_plan.override_at

        collector = TimingCollector()
        frame = single_map(make_power_map_device(run_scen.input_shape), "rd_maps")
        collector.t_construct = now()
        _run_graph(
            scen=run_scen,
            collector=collector,
            frame=frame,
            total_frames=run_plan.total_frames,
        )
        return collector

    samples = collect_scenario_samples(rs, plan, run_rep)
    base = _base_row(scen)
    append_setup_row(base, samples, report)
    compute = _base_row(scen)
    fill_compute_row(compute, samples)
    _fill_throughput(compute)
    if scen.use_alpha_lut:
        compute.notes = (
            "cold_ms is first-frame latency including initial alpha_lut H2D; "
            "median/p90 are pre-override steady state; override_ms is peak "
            "latency over the alpha_lut live-update window (pfa=1e-6 LUT)"
        )
    else:
        compute.notes = (
            "cold_ms is first-frame latency including initial LUT calibration; "
            "median/p90 are pre-override steady state; override_ms is peak "
            "latency over the blocking pfa reconfig window"
        )
    if scen.peak_only:
        nb = _peak_neighborhood_tag(scen.peak_neighborhood)
        compute.notes += f"; peak_only NMS enabled (neighborhood={nb})"
    report.results.append(compute)


def build_alpha_calibration_summary(report: PerfReport) -> None:
    rows: dict[str, AlphaCalibrationRow] = {
        pair_id: AlphaCalibrationRow(pair_id=pair_id) for _, pair_id in CALIBRATION_PAIR_SPECS
    }
    for row in report.results:
        if row.bucket != "compute" or not row.calibration_pair:
            continue
        out = rows.get(row.calibration_pair)
        if out is None:
            continue
        out.variant = row.variant
        out.size_id = row.size_id
        out.k = row.k
        if row.calibration_mode == "pfa":
            out.slow_cold_ms = row.cold_ms
            out.slow_ovr_ms = row.override_ms
        elif row.calibration_mode == "alpha_lut":
            out.fast_cold_ms = row.cold_ms
            out.fast_ovr_ms = row.override_ms
    for out in rows.values():
        if (
            not math.isnan(out.slow_cold_ms)
            and not math.isnan(out.fast_cold_ms)
            and out.fast_cold_ms > 0.0
        ):
            out.cold_speedup = out.slow_cold_ms / out.fast_cold_ms
        if (
            not math.isnan(out.slow_ovr_ms)
            and not math.isnan(out.fast_ovr_ms)
            and out.fast_ovr_ms > 0.0
        ):
            out.ovr_speedup = out.slow_ovr_ms / out.fast_ovr_ms
        if not math.isnan(out.slow_cold_ms) or not math.isnan(out.fast_cold_ms):
            report.alpha_calibration.append(out)
    report.alpha_calibration.sort(key=lambda r: r.pair_id)


def main() -> int:
    return run_perf_matrix_main(
        tool_name="perf_cfar_detector",
        description=(
            "CFARDetectorOp performance characterization (Python). "
            "Default matrix matches C++; pass --full for mid/large footprints."
        ),
        size_ladder_defs=SIZE_LADDER,
        build_full_matrix=build_full_matrix,
        build_default_matrix=build_default_matrix,
        build_size_sweep=build_size_sweep,
        run_scenario=run_scenario,
        scenario_label=lambda scen: scen.id,
        exercised_size_ids=lambda scen: scen.size_id,
        isolate_failures=True,
        post_run=build_alpha_calibration_summary,
        full_default=False,
        full_help=(
            "Full matrix: all variants/footprints/batch/NCI (includes mid/large "
            "GO/SO/OS Monte Carlo cold builds — same cost class as C++ --full) "
            "plus peak_only NMS rows (CA×ladder×{square8,cross4}; GO/SO/OS at md)."
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
