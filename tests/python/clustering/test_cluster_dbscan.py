# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/clustering/test_cluster_dbscan.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import ClusterDBSCANOpF32, ClusterDBSCANOpF64  # noqa: E402

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for ClusterDBSCANOp tests: {exc}")


def run_dbscan(
    host: list[float] | np.ndarray,
    rows: int,
    cols: int,
    *,
    op_cls=ClusterDBSCANOpF64,
    **kwargs,
) -> dict[str, np.ndarray]:
    dtype = np.float32 if op_cls is ClusterDBSCANOpF32 else np.float64
    if rows == 0:
        arr = np.empty((0, cols), dtype=dtype)
    else:
        arr = np.asarray(host, dtype=dtype).reshape(rows, cols)
    frame = single_map(to_device_array(arr), "points")

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator_multi(
        make_op,
        frame,
        input_port="points",
        output_ports=["labels", "sizes"],
    )
    assert len(outs["labels"]) == 1
    return {"labels": outs["labels"][0], "sizes": outs["sizes"][0]}


def test_TwoBlobsAndNoise():
    pts = [
        0.0,
        0.0,
        0.1,
        0.0,
        0.0,
        0.1,
        0.1,
        0.1,
        5.0,
        5.0,
        10.0,
        10.0,
        10.1,
        10.0,
        10.0,
        10.1,
    ]
    out = run_dbscan(pts, 8, 2, epsilon=[0.5], min_points=3)
    lab = out["labels"]
    assert lab.shape == (8,)
    assert lab[0] == 0
    assert lab[1] == 0
    assert lab[2] == 0
    assert lab[3] == 0
    assert lab[4] == -1
    assert lab[5] == 1
    assert lab[6] == 1
    assert lab[7] == 1
    sz = out["sizes"]
    assert sz.shape == (2,)
    assert sz[0] == 4
    assert sz[1] == 3


def test_AnisotropicPerAxisEpsilon():
    pts = [0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 5.0]
    out = run_dbscan(pts, 4, 2, epsilon=[3.0, 1.0], min_points=2)
    lab = out["labels"]
    assert lab[0] == 0
    assert lab[1] == 0
    assert lab[2] == 0
    assert lab[3] == -1
    assert out["sizes"].shape == (1,)
    assert out["sizes"][0] == 3


def test_MinPointsSeparatesCoreFromNoise():
    pts = [0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 10.0, 0.0, 10.1, 0.0]

    out2 = run_dbscan(pts, 5, 2, epsilon=[0.5], min_points=2)
    lab = out2["labels"]
    assert list(lab) == [0, 0, 0, 1, 1]
    assert list(out2["sizes"]) == [3, 2]

    out3 = run_dbscan(pts, 5, 2, epsilon=[0.5], min_points=3)
    lab = out3["labels"]
    assert list(lab) == [0, 0, 0, -1, -1]
    assert out3["sizes"].shape == (1,)
    assert out3["sizes"][0] == 3


def test_DefaultMinPointsIsTwiceNumFeatures():
    pts = [0.0, 0.0, 0.1, 0.0, 0.2, 0.0]
    out = run_dbscan(pts, 3, 2, epsilon=[0.5])
    assert list(out["labels"]) == [-1, -1, -1]
    assert out["sizes"].shape == (0,)


def test_MaxClustersCapsAndDropsSurplus():
    pts = [
        0.0,
        0.0,
        0.1,
        0.0,
        10.0,
        0.0,
        10.1,
        0.0,
        20.0,
        0.0,
        20.1,
        0.0,
    ]
    out = run_dbscan(pts, 6, 2, epsilon=[0.5], min_points=2, max_clusters=2)
    lab = out["labels"]
    assert list(lab) == [0, 0, 1, 1, -1, -1]
    assert list(out["sizes"]) == [2, 2]


def test_NumbersClustersByDescendingSize():
    # Smaller pair at early rows, larger triple later: descending-size numbering
    # labels the triple 0, so `sizes` is descending.
    pts = [0.0, 0.0, 0.1, 0.0, 10.0, 0.0, 10.1, 0.0, 10.2, 0.0]
    out = run_dbscan(pts, 5, 2, epsilon=[0.5], min_points=2)
    assert list(out["labels"]) == [1, 1, 0, 0, 0]
    assert list(out["sizes"]) == [3, 2]


def test_MaxClustersKeepsLargest():
    # Cap to 1 keeps the larger cluster (the triple) and drops the pair.
    pts = [0.0, 0.0, 0.1, 0.0, 10.0, 0.0, 10.1, 0.0, 10.2, 0.0]
    out = run_dbscan(pts, 5, 2, epsilon=[0.5], min_points=2, max_clusters=1)
    assert list(out["labels"]) == [-1, -1, 0, 0, 0]
    assert list(out["sizes"]) == [3]


def test_EmptyPointsEmitsEmptyOutputs():
    out = run_dbscan([], 0, 2, epsilon=[0.5], min_points=2)
    assert out["labels"].shape == (0,)
    assert out["sizes"].shape == (0,)


def test_FloatAndDoubleAgree():
    pts = [
        0.0,
        0.0,
        0.1,
        0.0,
        0.0,
        0.1,
        10.0,
        10.0,
        10.1,
        10.0,
        10.0,
        10.1,
        5.0,
        5.0,
    ]
    of = run_dbscan(pts, 7, 2, op_cls=ClusterDBSCANOpF32, epsilon=[0.5], min_points=3)
    od = run_dbscan(pts, 7, 2, epsilon=[0.5], min_points=3)
    assert of["labels"].shape == od["labels"].shape
    assert list(of["labels"]) == list(od["labels"])
    assert of["sizes"].shape == od["sizes"].shape


def test_RejectsEmptyEpsilon():
    pts = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        run_dbscan(pts, 2, 2, epsilon=[])


def test_LiveEpsilonOverrideAppliesPerTick():
    # Two nearby pairs. epsilon=0.9 keeps them separate (2 clusters);
    # epsilon=3.0 links them (1 cluster).
    pts = [0.0, 0.0, 0.1, 0.0, 2.0, 0.0, 2.1, 0.0]
    arr = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    frame = single_map(to_device_array(arr), "points")
    frames = [frame, frame, frame]

    def make_op(app):
        return ClusterDBSCANOpF64(app, epsilon=[0.9], min_points=2, name="op_under_test")

    outs = run_operator_multi(
        make_op,
        frames,
        input_port="points",
        output_ports=["labels", "sizes"],
        overrides={"epsilon": [[3.0], [0.9]]},
    )
    assert len(outs["sizes"]) == 3
    assert outs["sizes"][0].shape[0] == 1  # eps=3.0 → merged
    assert outs["sizes"][1].shape[0] == 2  # eps=0.9 → split
    assert outs["sizes"][2].shape[0] == 2  # no override → Params 0.9 → split
    assert int(outs["sizes"][0][0]) == 4
