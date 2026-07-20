# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/detection/test_feature_assembly.cpp (happy + reject)."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402

from holoscan.radar import (  # noqa: E402
    AssemblyFeature,
    FeatureAssemblyOpF64,
)

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for FeatureAssemblyOp tests: {exc}")


def run_assembly(
    idx: np.ndarray,
    features: list,
    *,
    range_bins=None,
    velocity_bins=None,
    detection_score=None,
    aoa_estimates=None,
    op_cls=FeatureAssemblyOpF64,
) -> np.ndarray:
    idx_arr = np.asarray(idx, dtype=np.int64)
    if idx_arr.ndim == 1:
        idx_arr = idx_arr.reshape(-1, 2)
    idx_frame = single_map(to_device_array(idx_arr.astype(np.int64)), "detection_idx")

    extra: dict[str, list] = {}
    if range_bins is not None:
        rb = np.asarray(range_bins, dtype=np.float64 if op_cls is FeatureAssemblyOpF64 else np.float32)
        extra["range_bins"] = [single_map(to_device_array(rb), "range_bins")]
    if velocity_bins is not None:
        vb = np.asarray(
            velocity_bins, dtype=np.float64 if op_cls is FeatureAssemblyOpF64 else np.float32
        )
        extra["velocity_bins"] = [single_map(to_device_array(vb), "velocity_bins")]
    if detection_score is not None:
        sc = np.asarray(
            detection_score,
            dtype=np.float64 if op_cls is FeatureAssemblyOpF64 else np.float32,
        )
        extra["detection_score"] = [
            single_map(to_device_array(sc), "detection_score")
        ]
    if aoa_estimates is not None:
        aoa = np.asarray(aoa_estimates, dtype=np.float64)
        extra["aoa_estimates"] = [
            single_map(to_device_array(aoa), "aoa_estimates")
        ]

    def make_op(app):
        return op_cls(app, features=features, name="op_under_test")

    outs = run_operator(
        make_op,
        idx_frame,
        input_port="detection_idx",
        output_port="points",
        extra_inputs=extra or None,
    )
    assert len(outs) == 1
    return outs[0]


def test_AssemblesRawBinEscapeHatches():
    idx = np.array([3, 5, 7, 2], dtype=np.int64)
    points = run_assembly(
        idx,
        [AssemblyFeature.DOPPLER_BIN, AssemblyFeature.RANGE_BIN],
    )
    assert points.shape == (2, 2)
    assert points[0, 0] == pytest.approx(3.0)
    assert points[0, 1] == pytest.approx(5.0)
    assert points[1, 0] == pytest.approx(7.0)
    assert points[1, 1] == pytest.approx(2.0)


def test_LooksUpPhysicalRangeVelocityAndScore():
    k_d, k_r = 8, 16
    idx = np.array([2, 4], dtype=np.int64)
    range_bins = 10.0 * np.arange(k_r, dtype=np.float64)
    vel_bins = -1.5 * np.arange(k_d, dtype=np.float64)
    score = np.zeros(k_d * k_r, dtype=np.float64)
    score[2 * k_r + 4] = 12.5

    points = run_assembly(
        idx,
        [AssemblyFeature.RANGE, AssemblyFeature.VELOCITY, AssemblyFeature.SCORE],
        range_bins=range_bins,
        velocity_bins=vel_bins,
        detection_score=score.reshape(k_d, k_r),
    )
    assert points.shape == (1, 3)
    assert points[0, 0] == pytest.approx(40.0)
    assert points[0, 1] == pytest.approx(-3.0)
    assert points[0, 2] == pytest.approx(12.5)


def test_EmptyDetectionIdxEmitsEmptyPoints():
    idx = np.empty((0, 2), dtype=np.int64)
    points = run_assembly(
        idx, [AssemblyFeature.DOPPLER_BIN, AssemblyFeature.RANGE_BIN]
    )
    assert points.shape == (0, 2)


def test_AppendsAoaEstimates():
    idx = np.array([[0, 1], [2, 3]], dtype=np.int64)
    aoa = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float64)
    points = run_assembly(
        idx,
        [AssemblyFeature.DOPPLER_BIN, AssemblyFeature.AZ, AssemblyFeature.EL],
        aoa_estimates=aoa,
    )
    assert points.shape == (2, 3)
    assert points[0, 0] == pytest.approx(0.0)
    assert points[0, 1] == pytest.approx(0.1)
    assert points[0, 2] == pytest.approx(-0.2)
    assert points[1, 0] == pytest.approx(2.0)
    assert points[1, 1] == pytest.approx(0.3)
    assert points[1, 2] == pytest.approx(0.4)


def test_RejectsEmptyFeatures():
    idx = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError):
        run_assembly(idx, [])


def test_RejectsScoreFeatureWithoutScoreInput():
    idx = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError):
        run_assembly(idx, [AssemblyFeature.SCORE])
