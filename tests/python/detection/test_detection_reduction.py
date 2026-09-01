# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/detection/test_detection_reduction.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402

from holoscan.radar import DetectionReductionOpF32  # noqa: E402
from holoscan.radar import DetectionReductionOpF64, RepresentativePolicy, ScoreMethod

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for DetectionReductionOp tests: {exc}")


def run_reduction(
    labels: np.ndarray,
    idx: np.ndarray,
    points: np.ndarray | None = None,
    *,
    op_cls=DetectionReductionOpF32,
    **kwargs,
) -> np.ndarray:
    labels_t = to_device_array(np.asarray(labels, dtype=np.int64))
    idx_t = to_device_array(np.asarray(idx, dtype=np.int64))
    extra = {
        "detection_idx": [single_map(idx_t, "detection_idx")],
    }
    if points is not None:
        dtype = np.float32 if op_cls is DetectionReductionOpF32 else np.float64
        pts_t = to_device_array(np.asarray(points, dtype=dtype))
        extra["points"] = [single_map(pts_t, "points")]

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator(
        make_op,
        single_map(labels_t, "labels"),
        input_port="labels",
        output_port="rep_idx",
        extra_inputs=extra,
    )
    assert len(outs) == 1
    return outs[0]


def test_FirstSelectsLowestIndexPerCluster():
    labels = np.array([0, 1, 0, -1], dtype=np.int64)
    idx = np.array(
        [[10, 1], [20, 2], [11, 3], [99, 9]],
        dtype=np.int64,
    )
    rep = run_reduction(labels, idx, representative=RepresentativePolicy.FIRST)
    assert rep.shape == (2, 2)
    assert rep[0, 0] == 10
    assert rep[0, 1] == 1
    assert rep[1, 0] == 20
    assert rep[1, 1] == 2


def test_PeakSelectsMaxWeightedSum():
    labels = np.array([0, 0, 1], dtype=np.int64)
    idx = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    points = np.array([[1.0, 0.0], [5.0, 0.0], [3.0, 0.0]], dtype=np.float64)
    rep = run_reduction(
        labels,
        idx,
        points,
        op_cls=DetectionReductionOpF64,
        representative=RepresentativePolicy.PEAK,
        score_method=ScoreMethod.WEIGHTED_SUM,
        feature_weights=[1.0, 0.0],
    )
    assert rep.shape == (2, 2)
    assert rep[0, 0] == 1
    assert rep[0, 1] == 1
    assert rep[1, 0] == 2
    assert rep[1, 1] == 2


def test_PeakSelectsMaxWeightedL2():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[10, 1], [20, 2]], dtype=np.int64)
    points = np.array([[4.0, 3.0], [6.0, 0.0]], dtype=np.float64)
    rep = run_reduction(
        labels,
        idx,
        points,
        op_cls=DetectionReductionOpF64,
        representative=RepresentativePolicy.PEAK,
        score_method=ScoreMethod.WEIGHTED_L2,
        feature_weights=[1.0, 1.0],
    )
    assert rep.shape == (1, 2)
    assert rep[0, 0] == 20
    assert rep[0, 1] == 2


def test_PeakSelectsMaxWeightedMax():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[10, 1], [20, 2]], dtype=np.int64)
    points = np.array([[10.0, 1.0], [9.0, 8.0]], dtype=np.float64)
    rep = run_reduction(
        labels,
        idx,
        points,
        op_cls=DetectionReductionOpF64,
        representative=RepresentativePolicy.PEAK,
        score_method=ScoreMethod.WEIGHTED_MAX,
        feature_weights=[1.0, 1.0],
    )
    assert rep.shape == (1, 2)
    assert rep[0, 0] == 10
    assert rep[0, 1] == 1


def test_MedoidRespectsFeatureWeights():
    labels = np.array([0, 0, 0], dtype=np.int64)
    idx = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.int64)
    points = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]], dtype=np.float64)
    rep = run_reduction(
        labels,
        idx,
        points,
        op_cls=DetectionReductionOpF64,
        representative=RepresentativePolicy.MEDOID,
        feature_weights=[1.0, 0.0],
    )
    assert rep.shape == (1, 2)
    assert rep[0, 0] == 1
    assert rep[0, 1] == 0


def test_ExcludesNoiseAndEmitsOnePerCluster():
    labels = np.array([-1, 0, -1, 1], dtype=np.int64)
    idx = np.array([[9, 9], [0, 1], [8, 8], [2, 3]], dtype=np.int64)
    rep = run_reduction(labels, idx, representative=RepresentativePolicy.FIRST)
    assert rep.shape == (2, 2)
    assert rep[0, 0] == 0
    assert rep[0, 1] == 1
    assert rep[1, 0] == 2
    assert rep[1, 1] == 3


def test_RejectsPeakWithoutPoints():
    labels = np.array([0], dtype=np.int64)
    idx = np.array([[0, 0]], dtype=np.int64)
    with pytest.raises(ValueError):
        run_reduction(
            labels,
            idx,
            representative=RepresentativePolicy.PEAK,
            feature_weights=[1.0],
        )


def test_RejectsMismatchedFeatureWeights():
    labels = np.array([0], dtype=np.int64)
    idx = np.array([[0, 0]], dtype=np.int64)
    points = np.array([[1.0, 2.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        run_reduction(
            labels,
            idx,
            points,
            op_cls=DetectionReductionOpF64,
            representative=RepresentativePolicy.PEAK,
            feature_weights=[1.0],
        )


def test_FloatDoubleAgreeOnPeak():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[0, 0], [1, 1]], dtype=np.int64)
    pd = np.array([[1.0], [4.0]], dtype=np.float64)
    pf = np.array([[1.0], [4.0]], dtype=np.float32)
    rd = run_reduction(
        labels,
        idx,
        pd,
        op_cls=DetectionReductionOpF64,
        representative=RepresentativePolicy.PEAK,
        feature_weights=[1.0],
    )
    rf = run_reduction(
        labels,
        idx,
        pf,
        representative=RepresentativePolicy.PEAK,
        feature_weights=[1.0],
    )
    assert rd[0, 0] == rf[0, 0]
    assert rd[0, 1] == rf[0, 1]


def test_EmptyInputEmitsZeroClusters():
    labels = np.empty((0,), dtype=np.int64)
    idx = np.empty((0, 2), dtype=np.int64)
    rep = run_reduction(labels, idx, representative=RepresentativePolicy.FIRST)
    assert rep.shape == (0, 2)


def test_AllNoiseEmitsZeroClusters():
    labels = np.array([-1, -1], dtype=np.int64)
    idx = np.array([[1, 2], [3, 4]], dtype=np.int64)
    rep = run_reduction(labels, idx, representative=RepresentativePolicy.FIRST)
    assert rep.shape == (0, 2)


def test_SparseLabelGapsSkipped():
    labels = np.array([0, 2], dtype=np.int64)
    idx = np.array([[10, 1], [30, 3]], dtype=np.int64)
    rep = run_reduction(labels, idx, representative=RepresentativePolicy.FIRST)
    assert rep.shape == (2, 2)
    assert rep[0, 0] == 10
    assert rep[0, 1] == 1
    assert rep[1, 0] == 30
    assert rep[1, 1] == 3


def test_WidenedDetectionIdxCols():
    labels = np.array([0], dtype=np.int64)
    idx = np.array([[7, 8, 9]], dtype=np.int64)
    rep = run_reduction(labels, idx, representative=RepresentativePolicy.FIRST)
    assert rep.shape == (1, 3)
    assert rep[0, 0] == 7
    assert rep[0, 1] == 8
    assert rep[0, 2] == 9


def test_PeakEqualScorePicksLowestIndex():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[10, 1], [20, 2]], dtype=np.int64)
    points = np.array([[5.0], [5.0]], dtype=np.float64)
    rep = run_reduction(
        labels,
        idx,
        points,
        op_cls=DetectionReductionOpF64,
        representative=RepresentativePolicy.PEAK,
        feature_weights=[1.0],
    )
    assert rep.shape == (1, 2)
    assert rep[0, 0] == 10
    assert rep[0, 1] == 1


def test_CustomScoreMethodIgnoredForFirst():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[10, 1], [20, 2]], dtype=np.int64)
    rep = run_reduction(
        labels,
        idx,
        representative=RepresentativePolicy.FIRST,
        score_method=ScoreMethod.CUSTOM,
    )
    assert rep.shape == (1, 2)
    assert rep[0, 0] == 10
    assert rep[0, 1] == 1


def test_LiveRepresentativeOverrideAppliesPerTick():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[0, 0], [1, 1]], dtype=np.int64)
    points = np.array([[1.0], [5.0]], dtype=np.float32)
    labels_t = to_device_array(labels)
    idx_t = to_device_array(idx)
    pts_t = to_device_array(points)
    frames = [single_map(labels_t, "labels")] * 3

    def make_op(app):
        return DetectionReductionOpF32(
            app,
            representative=RepresentativePolicy.PEAK,
            feature_weights=[1.0],
            name="op_under_test",
        )

    outs = run_operator(
        make_op,
        frames,
        input_port="labels",
        output_port="rep_idx",
        extra_inputs={
            "detection_idx": [single_map(idx_t, "detection_idx")] * 3,
            "points": [single_map(pts_t, "points")] * 3,
        },
        overrides={
            "representative": [
                RepresentativePolicy.PEAK,
                RepresentativePolicy.FIRST,
            ]
        },
    )
    assert len(outs) == 3
    assert outs[0].shape == (1, 2)
    assert outs[0][0, 0] == 1 and outs[0][0, 1] == 1
    assert outs[1][0, 0] == 0 and outs[1][0, 1] == 0
    # Tick 2 reverts to the Params policy (PEAK), as on tick 0.
    assert outs[2][0, 0] == 1 and outs[2][0, 1] == 1


def test_LiveScoreMethodOverrideAppliesPerTick():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[10, 1], [20, 2]], dtype=np.int64)
    points = np.array([[4.0, 3.0], [6.0, 0.0]], dtype=np.float32)
    labels_t = to_device_array(labels)
    idx_t = to_device_array(idx)
    pts_t = to_device_array(points)
    frames = [single_map(labels_t, "labels")] * 3

    def make_op(app):
        return DetectionReductionOpF32(
            app,
            representative=RepresentativePolicy.PEAK,
            score_method=ScoreMethod.WEIGHTED_SUM,
            feature_weights=[1.0, 1.0],
            name="op_under_test",
        )

    outs = run_operator(
        make_op,
        frames,
        input_port="labels",
        output_port="rep_idx",
        extra_inputs={
            "detection_idx": [single_map(idx_t, "detection_idx")] * 3,
            "points": [single_map(pts_t, "points")] * 3,
        },
        overrides={"score_method": [ScoreMethod.WEIGHTED_SUM, ScoreMethod.WEIGHTED_L2]},
    )
    assert outs[0][0, 0] == 10 and outs[0][0, 1] == 1
    assert outs[1][0, 0] == 20 and outs[1][0, 1] == 2
    # Tick 2 reverts to the Params method (WEIGHTED_SUM), as on tick 0.
    assert outs[2][0, 0] == 10 and outs[2][0, 1] == 1


def test_LiveFeatureWeightsOverrideAppliesPerTick():
    labels = np.array([0, 0], dtype=np.int64)
    idx = np.array([[10, 1], [20, 2]], dtype=np.int64)
    points = np.array([[5.0, 0.0], [1.0, 10.0]], dtype=np.float32)
    labels_t = to_device_array(labels)
    idx_t = to_device_array(idx)
    pts_t = to_device_array(points)
    frames = [single_map(labels_t, "labels")] * 3

    def make_op(app):
        return DetectionReductionOpF32(
            app,
            representative=RepresentativePolicy.PEAK,
            score_method=ScoreMethod.WEIGHTED_SUM,
            feature_weights=[1.0, 0.0],
            name="op_under_test",
        )

    outs = run_operator(
        make_op,
        frames,
        input_port="labels",
        output_port="rep_idx",
        extra_inputs={
            "detection_idx": [single_map(idx_t, "detection_idx")] * 3,
            "points": [single_map(pts_t, "points")] * 3,
        },
        overrides={"feature_weights": [[1.0, 0.0], [0.0, 1.0]]},
    )
    assert outs[0][0, 0] == 10 and outs[0][0, 1] == 1
    assert outs[1][0, 0] == 20 and outs[1][0, 1] == 2
    # Tick 2 reverts to the Params weights [1.0, 0.0], as on tick 0.
    assert outs[2][0, 0] == 10 and outs[2][0, 1] == 1
