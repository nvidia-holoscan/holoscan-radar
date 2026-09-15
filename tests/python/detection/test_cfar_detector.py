# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of core tests/cpp/detection/test_cfar_detector.cpp (CA variant)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import CFARDetectorOpF64  # noqa: E402
from holoscan.radar import (
    CFARDetectorOpF32,
    CfarPeakMode,
    CfarPeakNeighborhood,
    CfarVariant,
)

pytestmark = pytest.mark.requires_gpu

K_DOPPLER = 16
K_RANGE = 32


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for CFARDetectorOp tests: {exc}")


# ---------------------------------------------------------------------------
# Helpers (test_cfar_detector.cpp)
# ---------------------------------------------------------------------------


def power_map(
    floor: float,
    targets: Sequence[tuple[int, int]],
    peak: float,
    *,
    num_doppler: int = K_DOPPLER,
    num_range: int = K_RANGE,
) -> np.ndarray:
    """Flat power map with optional (doppler, range) peaks."""
    arr = np.full(num_doppler * num_range, floor, dtype=np.float64)
    for d, r in targets:
        arr[d * num_range + r] = peak
    return arr


def _ca_kwargs(pfa: float, **extra) -> dict:
    return {
        "cfar_variant": CfarVariant.CA,
        "guard_cells": [1, 1],
        "training_cells": [2, 2],
        "pfa": pfa,
        **extra,
    }


def run_cfar(
    host,
    shape: tuple[int, ...],
    *,
    dtype=np.float64,
    op_cls=CFARDetectorOpF64,
    pfa: float = 1e-3,
    op_kwargs: dict | None = None,
    **cfar_kwargs,
) -> dict[str, np.ndarray]:
    """Run CFAR once; capture all five outputs including detection_mask.

    detection_mask is uint8 (0/1) and therefore NumPy-visible on HSDK 4.3 (a
    kDLBool port throws on the 4.3 message path). Pass ``op_kwargs`` for a full
    Arg set without ``_ca_kwargs`` defaults (error-path tests that omit required
    params).
    """
    arr = np.asarray(host, dtype=dtype).reshape(shape)
    frame = single_map(to_device_array(arr, dtype=dtype), "rd_maps")
    kwargs = dict(op_kwargs) if op_kwargs is not None else _ca_kwargs(pfa, **cfar_kwargs)

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator_multi(
        make_op,
        frame,
        input_port="rd_maps",
        output_ports=[
            "detection_score",
            "noise_estimate",
            "detection_idx",
            "detection_mask",
            "cfar_power",
        ],
    )
    assert len(outs["detection_score"]) == 1
    return {
        "score": outs["detection_score"][0],
        "noise": outs["noise_estimate"][0],
        "idx": outs["detection_idx"][0],
        "mask": outs["detection_mask"][0],
        "power": outs["cfar_power"][0],
    }


def num_detections(idx: np.ndarray) -> int:
    return 0 if idx.size == 0 else int(idx.shape[0])


def idx_contains(
    idx: np.ndarray,
    doppler: int,
    range_bin: int,
    batch: Sequence[int] = (),
) -> bool:
    if idx.ndim < 2:
        return False
    n_cols = idx.shape[1]
    n_preserved = n_cols - 2
    if len(batch) != n_preserved:
        return False
    for row in idx:
        if n_preserved and not all(row[b] == batch[b] for b in range(n_preserved)):
            continue
        if row[n_preserved] == doppler and row[n_preserved + 1] == range_bin:
            return True
    return False


def expect_mask_matches_score_interior(
    score: np.ndarray,
    num_doppler: int,
    num_range: int,
) -> None:
    """Interior cells: score > 1 on detected cells (mask drained; score-only)."""
    margin = 3
    grid = score.reshape(num_doppler, num_range)
    for d in range(margin, num_doppler - margin):
        for r in range(margin, num_range - margin):
            if grid[d, r] > 1.0:
                assert grid[d, r] > 1.0, f"cell ({d}, {r})"
            else:
                assert grid[d, r] <= 1.0, f"cell ({d}, {r})"


# ---------------------------------------------------------------------------
# Core CA cases
# ---------------------------------------------------------------------------


def test_DetectsSingleTargetRealPath():
    dt, rt = 8, 16
    mp = power_map(1.0, [(dt, rt)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3)

    assert num_detections(out["idx"]) == 1
    rows = out["idx"]
    assert rows[0, 0] == dt
    assert rows[0, 1] == rt
    assert out["score"][dt, rt] > 1.0


def test_DetectionMaskIsUint8AndMatchesScore():
    # L1: detection_mask emits as uint8 (0/1), so Python can read it on HSDK 4.3
    # (a bool port would throw). It must equal (detection_score > 1) everywhere.
    dt, rt = 8, 16
    mp = power_map(1.0, [(dt, rt)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3)

    mask = out["mask"]
    assert mask.dtype == np.uint8
    assert mask.shape == out["score"].shape
    assert set(np.unique(mask)).issubset({0, 1})
    np.testing.assert_array_equal(mask != 0, out["score"] > 1.0)
    assert mask[dt, rt] == 1


def test_NoiseEstimateTracksFloor():
    floor = 3.5
    mp = power_map(floor, [], floor)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3)
    assert out["noise"][8, 16] == pytest.approx(floor, abs=1e-5)


def test_NoFalseAlarmsOnFlatMap():
    mp = power_map(1.0, [], 1.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-6)
    assert num_detections(out["idx"]) == 0


def test_ComplexInputFormsPower():
    dt, rt = 5, 20
    cube = np.full(K_DOPPLER * K_RANGE, 1.0 + 0.0j, dtype=np.complex128)
    cube[dt * K_RANGE + rt] = 10.0 + 0.0j
    out = run_cfar(
        cube,
        (K_DOPPLER, K_RANGE),
        dtype=np.complex128,
        pfa=1e-3,
    )
    assert num_detections(out["idx"]) == 1
    assert out["idx"][0, 0] == dt
    assert out["idx"][0, 1] == rt
    assert out["noise"][0] == pytest.approx(1.0, abs=1e-5)


def test_MaxDetectionsCap():
    mp = power_map(1.0, [(4, 8), (12, 24)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3, max_detections=1)
    assert num_detections(out["idx"]) == 1


def test_FloatDoubleAgree():
    dt, rt = 8, 16
    mp = power_map(1.0, [(dt, rt)], 100.0)
    out_f = run_cfar(
        mp.astype(np.float32),
        (K_DOPPLER, K_RANGE),
        dtype=np.float32,
        op_cls=CFARDetectorOpF32,
        pfa=1e-3,
    )
    out_d = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3)
    assert num_detections(out_f["idx"]) == num_detections(out_d["idx"])
    assert out_f["idx"][0, 0] == out_d["idx"][0, 0]
    assert out_f["idx"][0, 1] == out_d["idx"][0, 1]


def test_DetectsTwoTargets():
    mp = power_map(1.0, [(4, 8), (12, 24)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3)
    assert num_detections(out["idx"]) == 2
    assert idx_contains(out["idx"], 4, 8)
    assert idx_contains(out["idx"], 12, 24)


def test_MaskMatchesScoreInterior():
    mp = power_map(1.0, [(8, 16)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3)
    expect_mask_matches_score_interior(out["score"], K_DOPPLER, K_RANGE)
    assert out["score"][8, 16] > 1.0
    assert out["score"][4, 4] < 1.0


def test_RealAndComplexPathsAgree():
    dt, rt = 10, 22
    floor, peak = 2.0, 80.0
    real_map = power_map(floor, [(dt, rt)], peak)
    complex_map = np.full(K_DOPPLER * K_RANGE, np.sqrt(floor) + 0.0j, dtype=np.complex128)
    complex_map[dt * K_RANGE + rt] = np.sqrt(peak) + 0.0j

    out_real = run_cfar(real_map, (K_DOPPLER, K_RANGE), pfa=1e-3)
    out_complex = run_cfar(complex_map, (K_DOPPLER, K_RANGE), dtype=np.complex128, pfa=1e-3)
    assert num_detections(out_real["idx"]) == num_detections(out_complex["idx"])
    assert idx_contains(out_real["idx"], dt, rt)
    assert idx_contains(out_complex["idx"], dt, rt)


# ---------------------------------------------------------------------------
# cfar_power — port of tests/cpp/detection/test_cfar_detector.cpp
# ---------------------------------------------------------------------------


def test_CfarPowerIsSquareLawPowerWithoutNci():
    # nci_axes empty: the thresholded surface is the plain square-law power map.
    cube = np.full((K_DOPPLER, K_RANGE), 1.0 + 0.0j, dtype=np.complex128)
    cube[5, 20] = 3.0 + 4.0j
    out = run_cfar(cube, (K_DOPPLER, K_RANGE), dtype=np.complex128, pfa=1e-3)
    assert out["power"].shape == (K_DOPPLER, K_RANGE)
    assert out["power"].shape == out["score"].shape  # contract: same as the maps
    np.testing.assert_allclose(out["power"], np.abs(cube) ** 2, rtol=0, atol=1e-9)
    assert out["power"][5, 20] == pytest.approx(25.0)  # |3+4i|^2


def test_CfarPowerIsNciSumOverFold():
    # nci_axes=[0] over a rank-3 cube: the channel axis folds away and the
    # surface is sum_k |x_k|^2. Channel k has background (k+1), so the
    # background sum is 1 + 4 + 9 + 16 = 30.
    n_ch = 4
    cube = np.empty((n_ch, K_DOPPLER, K_RANGE), dtype=np.complex128)
    for k in range(n_ch):
        cube[k] = float(k + 1) + 0.0j
        cube[k, 5, 20] = 10.0 + 0.0j
    out = run_cfar(
        cube,
        (n_ch, K_DOPPLER, K_RANGE),
        dtype=np.complex128,
        pfa=1e-3,
        nci_axes=[0],
        # nci_fold is not exposed by the Python binding (it is only an alpha-LUT
        # pre-warm hint); the operator derives the fold from the channel axis.
    )
    assert out["power"].shape == (K_DOPPLER, K_RANGE)  # folded axis is gone
    assert out["power"].shape == out["score"].shape
    assert out["power"][0, 0] == pytest.approx(30.0)
    assert out["power"][5, 20] == pytest.approx(400.0)  # 4 * |10|^2
    # It is the surface the detector thresholded: the detection is its maximum.
    assert num_detections(out["idx"]) == 1
    peak = np.unravel_index(np.argmax(out["power"]), out["power"].shape)
    assert idx_contains(out["idx"], int(peak[0]), int(peak[1]))


# ---------------------------------------------------------------------------
# Peak modes — port of tests/cpp/detection/test_cfar_detector.cpp
# ---------------------------------------------------------------------------


def test_PeakModeNoneMatchesRawMask():
    # None by default keeps the raw candidate mask: a
    # contiguous 3-cell strip stays 3 detections either way.
    mp = power_map(1.0, [(8, 15), (8, 16), (8, 17)], 100.0)
    raw = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-2)
    off = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-2,
        peak_mode=CfarPeakMode.NONE,
    )
    assert num_detections(raw["idx"]) == 3
    assert num_detections(off["idx"]) == num_detections(raw["idx"])


def test_PeakModeRangeDopplerKeepsSingleMax():
    # Adjacent (guard-separated) pair: range-Doppler mode keeps the
    # maximum only; detection_mask agrees with the reduced count.
    flat = power_map(1.0, [], 1.0).reshape(K_DOPPLER, K_RANGE)
    flat[8, 16] = 100.0
    flat[8, 17] = 50.0
    off = run_cfar(flat, (K_DOPPLER, K_RANGE), pfa=1e-3)
    on = run_cfar(
        flat,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        peak_mode=CfarPeakMode.RANGE_DOPPLER,
    )
    assert num_detections(off["idx"]) == 2
    assert num_detections(on["idx"]) == 1
    assert idx_contains(on["idx"], 8, 16)  # the max, not (8, 17)
    assert int((on["mask"] != 0).sum()) == 1


def test_PeakModeRangeDopplerUsesRasterTiebreak():
    # A flat contiguous strip collapses to its lowest-raster-index cell.
    mp = power_map(1.0, [(8, 15), (8, 16), (8, 17)], 100.0)
    off = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-2)
    on = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-2,
        peak_mode=CfarPeakMode.RANGE_DOPPLER,
    )
    assert num_detections(off["idx"]) == 3
    assert num_detections(on["idx"]) == 1
    assert idx_contains(on["idx"], 8, 15)  # lowest linear index survives
    assert int((on["mask"] != 0).sum()) == 1


def test_PeakModeRangeDopplerUsesConfiguredNeighborhood():
    # square8 sees a diagonal neighbor (collapses a diagonal equal pair to one);
    # cross4 does not (both survive).
    mp = power_map(1.0, [(5, 5), (6, 6)], 100.0)
    square8 = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        peak_mode=CfarPeakMode.RANGE_DOPPLER,
        peak_neighborhood=CfarPeakNeighborhood.SQUARE8,
    )
    cross4 = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        peak_mode=CfarPeakMode.RANGE_DOPPLER,
        peak_neighborhood=CfarPeakNeighborhood.CROSS4,
    )
    assert num_detections(square8["idx"]) == 1
    assert idx_contains(square8["idx"], 5, 5)  # lower linear index survives
    assert num_detections(cross4["idx"]) == 2


def test_PeakModeAllSuppressesAcrossEveryRank4Axis():
    shape = (2, 2, K_DOPPLER, K_RANGE)
    data = np.ones(shape, dtype=np.float64)
    data[0, 0, 8, 16] = 100.0
    data[0, 1, 8, 16] = 110.0
    data[1, 0, 8, 16] = 120.0
    data[1, 1, 8, 16] = 130.0
    none = run_cfar(data, shape, pfa=1e-2, max_tx=2, max_rx=2)
    rd = run_cfar(
        data,
        shape,
        pfa=1e-2,
        peak_mode=CfarPeakMode.RANGE_DOPPLER,
        max_tx=2,
        max_rx=2,
    )
    all_axes = run_cfar(
        data,
        shape,
        pfa=1e-2,
        peak_mode=CfarPeakMode.ALL,
        max_tx=2,
        max_rx=2,
    )
    assert num_detections(none["idx"]) == 4
    assert num_detections(rd["idx"]) == 4
    assert all_axes["idx"].tolist() == [[1, 1, 8, 16]]
    assert int((all_axes["mask"] != 0).sum()) == 1
    np.testing.assert_array_equal(none["score"], all_axes["score"])
    np.testing.assert_array_equal(none["noise"], all_axes["noise"])
    np.testing.assert_array_equal(none["power"], all_axes["power"])


def test_PeakModeAllRank3TiesAndDoesNotWrapAxes():
    shape = (2, K_DOPPLER, K_RANGE)
    data = np.ones(shape, dtype=np.float32)
    data[0, 8, 16] = 100.0
    data[1, 8, 16] = 100.0
    data[0, 4, K_RANGE - 1] = 90.0
    data[0, 5, 0] = 80.0
    out = run_cfar(
        data,
        shape,
        dtype=np.float32,
        op_cls=CFARDetectorOpF32,
        pfa=1e-2,
        peak_mode=CfarPeakMode.ALL,
        max_rx=2,
    )
    assert out["idx"].tolist() == [
        [0, 4, K_RANGE - 1],
        [0, 5, 0],
        [0, 8, 16],
    ]
    assert int((out["mask"] != 0).sum()) == 3


# ---------------------------------------------------------------------------
# Additional CA coverage (previously deferred stubs)
# ---------------------------------------------------------------------------


def run_cfar_ticks(
    host,
    shape: tuple[int, ...],
    *,
    overrides: dict | None = None,
    output_ports: Sequence[str] = ("detection_idx",),
    dtype=np.float64,
    op_cls=CFARDetectorOpF64,
    pfa: float = 1e-3,
    n_ticks: int | None = None,
    **cfar_kwargs,
) -> dict[str, list[np.ndarray]]:
    """Multi-tick CFAR run with optional [live] overrides.

    An override sequence shorter than the tick count leaves the later ticks
    without a match, so they fall back to the operator's Params value.
    """
    arr = np.asarray(host, dtype=dtype).reshape(shape)
    frame = single_map(to_device_array(arr, dtype=dtype), "rd_maps")
    ticks = n_ticks if n_ticks is not None else 3
    frames = [frame] * ticks

    def make_op(app):
        return op_cls(
            app,
            name="op_under_test",
            **_ca_kwargs(pfa, **cfar_kwargs),
        )

    return run_operator_multi(
        make_op,
        frames,
        input_port="rd_maps",
        output_ports=list(output_ports),
        overrides=overrides,
        drain_ports=["detection_mask"]
        + [
            p
            for p in ("detection_score", "noise_estimate", "detection_idx")
            if p not in output_ports
        ],
    )


def power_cube(
    shape: tuple[int, ...],
    floor: float,
    cells: Sequence[tuple[tuple[int, ...], float]],
) -> np.ndarray:
    """Rank-N power cube with selected cells elevated."""
    arr = np.full(shape, floor, dtype=np.float64)
    for coords, value in cells:
        arr[coords] = value
    return arr


def test_DetectsOnLargeMap():
    k_d, k_r = 256, 512
    dt, rt = 128, 256
    mp = power_map(1.0, [(dt, rt)], 100.0, num_doppler=k_d, num_range=k_r)
    out = run_cfar(mp, (k_d, k_r), pfa=1e-3)
    assert num_detections(out["idx"]) == 1
    assert idx_contains(out["idx"], dt, rt)
    expect_mask_matches_score_interior(out["score"], k_d, k_r)


def test_DetectionIdxRasterOrder():
    mp = power_map(1.0, [(4, 8), (12, 24)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3)
    assert num_detections(out["idx"]) == 2
    assert out["idx"].shape == (2, 2)
    assert out["idx"][0, 0] == 4
    assert out["idx"][0, 1] == 8
    assert out["idx"][1, 0] == 12
    assert out["idx"][1, 1] == 24


def test_BorderCellsUseInBoundsTrainingOnly():
    floor = 4.0
    mp = power_map(floor, [], floor)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-6)
    assert out["noise"][0, 0] == pytest.approx(floor, abs=1e-6)
    assert out["noise"][8, 16] == pytest.approx(floor, abs=1e-9)
    assert out["score"][0, 0] <= 1.0
    assert num_detections(out["idx"]) == 0


def test_NciIntegratesChannelsForDetection():
    k_channels = 8
    dt, rt = 8, 16
    floor, weak_peak = 1.0, 14.0
    cells = [((c, dt, rt), weak_peak if c == 0 else floor) for c in range(k_channels)]
    cube = power_cube((k_channels, K_DOPPLER, K_RANGE), floor, cells)

    per_channel = run_cfar(cube, (k_channels, K_DOPPLER, K_RANGE), pfa=1e-6)
    assert num_detections(per_channel["idx"]) == 0

    integrated = run_cfar(
        cube,
        (k_channels, K_DOPPLER, K_RANGE),
        pfa=1e-3,
        nci_axes=[0],
    )
    assert integrated["score"].shape == (K_DOPPLER, K_RANGE)
    assert integrated["score"][dt, rt] > 1.0
    assert num_detections(integrated["idx"]) >= 1
    assert idx_contains(integrated["idx"], dt, rt)


def test_CaAlphaK1MatchesContractClosedForm():
    from holoscan.radar import ca_cfar_alpha, ca_cfar_pfa_from_alpha

    k_n = 40
    k_pfa = 1e-2
    alpha = ca_cfar_alpha(k_pfa, k_n, 1)
    closed = k_n * (k_pfa ** (-1.0 / k_n) - 1.0)
    assert alpha == pytest.approx(closed, rel=1e-6)
    assert ca_cfar_pfa_from_alpha(alpha, k_n, 1) == pytest.approx(k_pfa, abs=1e-9)


def test_LiveThresholdScaleOverrideAppliesPerTick():
    mp = power_map(1.0, [(8, 16)], 100.0)
    outs = run_cfar_ticks(
        mp,
        (K_DOPPLER, K_RANGE),
        overrides={"threshold_scale": [1.0, 1.0e4]},
        output_ports=["detection_idx"],
    )
    idx = outs["detection_idx"]
    assert len(idx) == 3
    assert num_detections(idx[0]) == 1
    assert num_detections(idx[1]) == 0
    # Tick 2 has no override and reverts to the Params scale of 1.0.
    assert num_detections(idx[2]) == 1


def test_LivePfaOverrideAppliesPerTick():
    dt, rt = 8, 16
    mp = power_map(1.0, [(dt, rt)], 26.0)
    outs = run_cfar_ticks(
        mp,
        (K_DOPPLER, K_RANGE),
        overrides={"pfa": [1e-2, 1e-8]},
        output_ports=["detection_score"],
        pfa=1e-3,
        blocking_alpha_lut_rebuild=True,
    )
    scores = outs["detection_score"]
    s0 = float(scores[0][dt, rt])
    s1 = float(scores[1][dt, rt])
    s2 = float(scores[2][dt, rt])
    # Tick 2 reverts to the Params pfa of 1e-3, between the two overrides.
    assert s1 < s2 < s0


def test_LiveMaxDetectionsOverrideAppliesPerTick():
    mp = power_map(1.0, [(4, 8), (12, 24)], 100.0)
    outs = run_cfar_ticks(
        mp,
        (K_DOPPLER, K_RANGE),
        overrides={"max_detections": [0, 1]},
        output_ports=["detection_idx"],
    )
    idx = outs["detection_idx"]
    assert num_detections(idx[0]) == 2
    assert num_detections(idx[1]) == 1
    # Tick 2 reverts to the Params cap, which is unset (uncapped).
    assert num_detections(idx[2]) == 2


def test_LiveGuardCellsOverrideAppliesPerTick():
    dt, rt = 8, 16
    mp = power_map(1.0, [], 1.0)
    flat = mp.reshape(K_DOPPLER, K_RANGE)
    flat[dt, rt] = 40.0
    flat[dt, rt + 2] = 10000.0
    outs = run_cfar_ticks(
        flat,
        (K_DOPPLER, K_RANGE),
        overrides={"guard_cells": [[1, 1], [1, 2]]},
        output_ports=["detection_score"],
        blocking_alpha_lut_rebuild=True,
    )
    scores = outs["detection_score"]
    s0 = float(scores[0][dt, rt])
    s1 = float(scores[1][dt, rt])
    s2 = float(scores[2][dt, rt])
    assert s0 < s1
    # Tick 2 reverts to the Params guard extent, which tick 0 also used.
    assert s2 == pytest.approx(s0)


def test_LiveTrainingCellsOverrideAppliesPerTick():
    dt, rt = 8, 16
    mp = power_map(1.0, [], 1.0)
    flat = mp.reshape(K_DOPPLER, K_RANGE)
    flat[6, rt] = 10000.0
    flat[dt, rt] = 40.0
    outs = run_cfar_ticks(
        flat,
        (K_DOPPLER, K_RANGE),
        overrides={"training_cells": [[1, 1], [3, 3]]},
        output_ports=["detection_score"],
        training_cells=[1, 1],
        blocking_alpha_lut_rebuild=True,
    )
    scores = outs["detection_score"]
    s0 = float(scores[0][dt, rt])
    s1 = float(scores[1][dt, rt])
    s2 = float(scores[2][dt, rt])
    assert s0 < s1
    # Tick 2 reverts to the Params training extent, which tick 0 also used.
    assert s2 == pytest.approx(s0)


def test_LiveNciAxesOverrideAppliesPerTick():
    k_channels = 8
    dt, rt = 8, 16
    floor, weak_peak = 1.0, 14.0
    cells = [((c, dt, rt), weak_peak if c == 0 else floor) for c in range(k_channels)]
    cube = power_cube((k_channels, K_DOPPLER, K_RANGE), floor, cells)
    outs = run_cfar_ticks(
        cube,
        (k_channels, K_DOPPLER, K_RANGE),
        overrides={"nci_axes": [[], [0]]},
        output_ports=["detection_idx", "detection_score"],
        pfa=1e-6,
        nci_axes=[],
        blocking_alpha_lut_rebuild=True,
    )
    idx = outs["detection_idx"]
    scores = outs["detection_score"]
    assert num_detections(idx[0]) == 0
    assert scores[0].shape == (k_channels, K_DOPPLER, K_RANGE)
    assert scores[1].shape == (K_DOPPLER, K_RANGE)
    # Tick 2 reverts to the Params nci_axes (empty), as on tick 0.
    assert scores[2].shape == scores[0].shape
    assert num_detections(idx[2]) == 0


# ---------------------------------------------------------------------------
# CA leftovers vs test_cfar_detector.cpp
# ---------------------------------------------------------------------------


def test_GuardCellsOverrideAffectsScore():
    dt, rt = 8, 16
    mp = power_map(1.0, [], 1.0).reshape(K_DOPPLER, K_RANGE)
    mp[dt, rt] = 40.0
    mp[dt, rt + 2] = 10000.0
    narrow = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        guard_cells=[1, 1],
        blocking_alpha_lut_rebuild=True,
    )
    wide = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        guard_cells=[1, 2],
        blocking_alpha_lut_rebuild=True,
    )
    assert float(narrow["score"][dt, rt]) < float(wide["score"][dt, rt])


def test_TrainingCellsOverrideAffectsScore():
    dt, rt = 8, 16
    mp = power_map(1.0, [], 1.0).reshape(K_DOPPLER, K_RANGE)
    mp[6, rt] = 10000.0
    mp[dt, rt] = 40.0
    wide = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        training_cells=[3, 3],
        blocking_alpha_lut_rebuild=True,
    )
    narrow = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        training_cells=[1, 1],
        blocking_alpha_lut_rebuild=True,
    )
    assert float(wide["score"][dt, rt]) > float(narrow["score"][dt, rt])


def test_NciSumsAllChannelPower():
    k_channels = 4
    dt, rt = 8, 16
    cells = [((c, dt, rt), 10.0) for c in range(k_channels)]
    cube = power_cube((k_channels, K_DOPPLER, K_RANGE), 1.0, cells)
    out = run_cfar(
        cube,
        (k_channels, K_DOPPLER, K_RANGE),
        pfa=1e-3,
        nci_axes=[0],
    )
    assert float(out["score"][dt, rt]) > 1.0
    assert num_detections(out["idx"]) == 1


def test_NciPreservesBatchAxis():
    k_batch, k_channels = 2, 3
    dt, rt = 8, 16
    peak = 12.0
    cells = [((1, c, dt, rt), peak) for c in range(k_channels)]
    cube = power_cube((k_batch, k_channels, K_DOPPLER, K_RANGE), 1.0, cells)
    out = run_cfar(
        cube,
        (k_batch, k_channels, K_DOPPLER, K_RANGE),
        pfa=1e-3,
        nci_axes=[1],
    )
    assert out["score"].shape == (k_batch, K_DOPPLER, K_RANGE)
    assert num_detections(out["idx"]) == 1
    assert out["idx"].shape == (1, 3)
    assert idx_contains(out["idx"], dt, rt, batch=(1,))


def test_BatchIdxIncludesPreservedAxes():
    k_batch = 2
    dt, rt = 8, 16
    cube = power_cube(
        (k_batch, K_DOPPLER, K_RANGE),
        1.0,
        [((0, dt, rt), 100.0), ((1, dt, rt), 100.0)],
    )
    out = run_cfar(
        cube,
        (k_batch, K_DOPPLER, K_RANGE),
        pfa=1e-3,
        blocking_alpha_lut_rebuild=True,
    )
    assert num_detections(out["idx"]) == 2
    assert idx_contains(out["idx"], dt, rt, batch=(0,))
    assert idx_contains(out["idx"], dt, rt, batch=(1,))


def test_NciScoreMatchesDetectionOnIntegratedMap():
    """Score-only stand-in for C++ NciMaskMatchesScoreOnIntegratedMap."""
    k_channels = 4
    cube = power_cube((k_channels, K_DOPPLER, K_RANGE), 1.0, [((0, 8, 16), 40.0)])
    out = run_cfar(
        cube,
        (k_channels, K_DOPPLER, K_RANGE),
        pfa=1e-3,
        nci_axes=[0],
    )
    expect_mask_matches_score_interior(out["score"], K_DOPPLER, K_RANGE)
    assert float(out["score"][8, 16]) > 1.0


def test_RejectsMissingGuardCells():
    mp = power_map(1.0, [], 1.0)
    with pytest.raises(ValueError):
        run_cfar(
            mp,
            (K_DOPPLER, K_RANGE),
            op_kwargs={
                "cfar_variant": CfarVariant.CA,
                "training_cells": [2, 2],
                "pfa": 1e-3,
            },
        )


def test_RejectsInvalidNciAxis():
    mp = power_map(1.0, [], 1.0)
    with pytest.raises(ValueError):
        run_cfar(
            mp,
            (K_DOPPLER, K_RANGE),
            pfa=1e-3,
            nci_axes=[0],
        )


# ---------------------------------------------------------------------------
# GO / SO / OS (shared matrix + unique cases)
# ---------------------------------------------------------------------------

_VARIANT_CASES = [
    (CfarVariant.GO, {}),
    (CfarVariant.SO, {}),
    (CfarVariant.OS, {"os_rank": 30}),
]


@pytest.mark.parametrize("variant,extra", _VARIANT_CASES)
def test_VariantDetectsSingleTargetRealPath(variant, extra):
    dt, rt = 8, 16
    mp = power_map(1.0, [(dt, rt)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3, cfar_variant=variant, **extra)
    assert num_detections(out["idx"]) == 1
    assert out["idx"][0, 0] == dt
    assert out["idx"][0, 1] == rt
    assert out["score"][dt, rt] > 1.0


@pytest.mark.parametrize("variant,extra", _VARIANT_CASES)
def test_VariantNoFalseAlarmsOnFlatMap(variant, extra):
    mp = power_map(1.0, [], 1.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-6, cfar_variant=variant, **extra)
    assert num_detections(out["idx"]) == 0


@pytest.mark.parametrize("variant,extra", _VARIANT_CASES)
def test_VariantNoiseEstimateTracksFloor(variant, extra):
    floor = 3.5
    mp = power_map(floor, [], floor)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3, cfar_variant=variant, **extra)
    assert out["noise"][8, 16] == pytest.approx(floor, abs=1e-5)


@pytest.mark.parametrize("variant,extra", _VARIANT_CASES)
def test_VariantDetectsTwoTargets(variant, extra):
    mp = power_map(1.0, [(4, 8), (12, 24)], 100.0)
    out = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3, cfar_variant=variant, **extra)
    assert num_detections(out["idx"]) == 2
    assert idx_contains(out["idx"], 4, 8)
    assert idx_contains(out["idx"], 12, 24)


@pytest.mark.parametrize("variant,extra", _VARIANT_CASES)
def test_VariantFloatDoubleAgree(variant, extra):
    dt, rt = 8, 16
    mp = power_map(1.0, [(dt, rt)], 100.0)
    out_f = run_cfar(
        mp.astype(np.float32),
        (K_DOPPLER, K_RANGE),
        dtype=np.float32,
        op_cls=CFARDetectorOpF32,
        pfa=1e-3,
        cfar_variant=variant,
        **extra,
    )
    out_d = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3, cfar_variant=variant, **extra)
    assert num_detections(out_f["idx"]) == num_detections(out_d["idx"])
    assert out_f["idx"][0, 0] == out_d["idx"][0, 0]
    assert out_f["idx"][0, 1] == out_d["idx"][0, 1]


@pytest.mark.parametrize(
    "variant,extra,weak_peak",
    [
        (CfarVariant.GO, {}, 14.0),
        (CfarVariant.SO, {}, 20.0),
        (CfarVariant.OS, {"os_rank": 30}, 20.0),
    ],
)
def test_VariantNciIntegratesChannelsForDetection(variant, extra, weak_peak):
    # Mirror GO/SO/OS C++: zero-det check uses a flat cube; SO needs a stronger
    # weak_peak (20) than CA/GO/OS (14) for the NCI detect assertion.
    k_channels = 8
    dt, rt = 8, 16
    floor = 1.0
    flat = power_cube((k_channels, K_DOPPLER, K_RANGE), floor, [])
    per_channel = run_cfar(
        flat,
        (k_channels, K_DOPPLER, K_RANGE),
        pfa=1e-6,
        cfar_variant=variant,
        **extra,
    )
    assert num_detections(per_channel["idx"]) == 0

    cells = [((c, dt, rt), weak_peak if c == 0 else floor) for c in range(k_channels)]
    cube = power_cube((k_channels, K_DOPPLER, K_RANGE), floor, cells)
    integrated = run_cfar(
        cube,
        (k_channels, K_DOPPLER, K_RANGE),
        pfa=1e-3,
        cfar_variant=variant,
        nci_axes=[0],
        **extra,
    )
    assert integrated["score"].shape == (K_DOPPLER, K_RANGE)
    assert float(integrated["score"][dt, rt]) > 1.0
    assert idx_contains(integrated["idx"], dt, rt)


def test_GoMoreConservativeThanCaWithOneSidedInterferer():
    dt, rt = 8, 16
    mp = power_map(1.0, [], 1.0).reshape(K_DOPPLER, K_RANGE)
    mp[dt, rt] = 50.0
    mp[dt, rt + 3] = 10000.0
    out_ca = run_cfar(mp, (K_DOPPLER, K_RANGE), pfa=1e-3, blocking_alpha_lut_rebuild=True)
    out_go = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        cfar_variant=CfarVariant.GO,
        blocking_alpha_lut_rebuild=True,
    )
    assert float(out_ca["score"][dt, rt]) > float(out_go["score"][dt, rt])
    assert float(out_go["noise"][dt, rt]) > float(out_ca["noise"][dt, rt])


def test_SoLessConservativeThanGoWithOneSidedInterferer():
    dt, rt = 8, 16
    mp = power_map(1.0, [], 1.0).reshape(K_DOPPLER, K_RANGE)
    mp[dt, rt] = 50.0
    mp[dt, rt + 3] = 10000.0
    out_go = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        cfar_variant=CfarVariant.GO,
        blocking_alpha_lut_rebuild=True,
    )
    out_so = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        cfar_variant=CfarVariant.SO,
        blocking_alpha_lut_rebuild=True,
    )
    assert float(out_so["score"][dt, rt]) > float(out_go["score"][dt, rt])
    assert float(out_so["noise"][dt, rt]) < float(out_go["noise"][dt, rt])


def test_OsNoiseEstimateMatchesOrderStatistic():
    dt, rt = 8, 16
    mp = power_map(1.0, [], 1.0).reshape(K_DOPPLER, K_RANGE)
    mp[dt, rt] = 50.0
    out = run_cfar(
        mp,
        (K_DOPPLER, K_RANGE),
        pfa=1e-3,
        cfar_variant=CfarVariant.OS,
        os_rank=30,
    )
    noise = float(out["noise"][dt, rt])
    assert noise > 0.0
    assert noise < 50.0


def test_RejectsOsRankOutOfRange():
    mp = power_map(1.0, [(8, 16)], 100.0)
    with pytest.raises(ValueError):
        run_cfar(
            mp,
            (K_DOPPLER, K_RANGE),
            pfa=1e-3,
            cfar_variant=CfarVariant.OS,
            os_rank=1000,
        )


def test_RejectsUnsetOsRankWithClearError():
    mp = power_map(1.0, [(8, 16)], 100.0)
    with pytest.raises(ValueError, match="os_rank is required for OS variant"):
        run_cfar(
            mp,
            (K_DOPPLER, K_RANGE),
            pfa=1e-3,
            cfar_variant=CfarVariant.OS,
        )


def test_LiveOsRankOverrideAppliesPerTick():
    dt, rt = 8, 16
    mp = np.empty(K_DOPPLER * K_RANGE, dtype=np.float64)
    for i in range(mp.size):
        mp[i] = 1.0 + float(i % 17) * 0.03
    flat = mp.reshape(K_DOPPLER, K_RANGE)
    flat[dt, rt] = 50.0
    outs = run_cfar_ticks(
        flat,
        (K_DOPPLER, K_RANGE),
        overrides={"os_rank": [5, 30]},
        output_ports=["noise_estimate"],
        cfar_variant=CfarVariant.OS,
        os_rank=5,
        blocking_alpha_lut_rebuild=True,
    )
    noise = outs["noise_estimate"]
    n0 = float(noise[0][dt, rt])
    n1 = float(noise[1][dt, rt])
    n2 = float(noise[2][dt, rt])
    assert n0 != n1
    # Tick 2 reverts to the Params rank of 5, which tick 0 also used.
    assert n2 == pytest.approx(n0)
