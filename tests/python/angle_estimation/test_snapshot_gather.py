# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of tests/cpp/angle_estimation/test_snapshot_gather.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import (  # noqa: E402
    run_operator,
    run_operator_multi,
    single_map,
    to_device_array,
)

from holoscan.radar import SnapshotEdgeMode  # noqa: E402
from holoscan.radar import SnapshotGatherOpF32, SnapshotGatherOpF64

pytestmark = pytest.mark.requires_gpu

K_C = 3
K_D = 8
K_R = 16


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for SnapshotGatherOp tests: {exc}")


def cell_value(c: int, d: int, r: int) -> complex:
    return complex(c * 100 + d * 10 + r, c + d + r)


def make_rd_maps(*, dtype=np.complex128) -> np.ndarray:
    host = np.empty((K_C, K_D, K_R), dtype=dtype)
    for c in range(K_C):
        for d in range(K_D):
            for r in range(K_R):
                host[c, d, r] = np.complex128(cell_value(c, d, r))
    return host


def run_gather(
    rd_maps: np.ndarray,
    detection_idx: np.ndarray,
    *,
    op_cls=SnapshotGatherOpF64,
    **kwargs,
) -> np.ndarray:
    dtype = np.complex64 if op_cls is SnapshotGatherOpF32 else np.complex128
    rd = to_device_array(np.asarray(rd_maps, dtype=dtype), dtype=dtype)
    idx = to_device_array(np.asarray(detection_idx, dtype=np.int64), dtype=np.int64)
    frame = single_map(rd, "rd_maps")

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outs = run_operator(
        make_op,
        frame,
        input_port="rd_maps",
        output_port="snapshot",
        extra_inputs={"detection_idx": [single_map(idx, "detection_idx")]},
        drain_ports=["is_valid"],
    )
    assert len(outs) == 1
    return outs[0]


def snap_at(snap: np.ndarray, item: int, slot: int, channel: int) -> complex:
    """Index snapshot with the contract layout (item, channel, slot).

    snapshot is (num_detections, num_channels, num_snapshots) row-major
    (channel-major, snapshot-minor), matching the estimators' covariance
    indexing xi[c * num_snapshots + s].
    """
    num_snapshots = int(snap.shape[2])
    flat = snap.reshape(snap.shape[0], -1)
    return flat[item, channel * num_snapshots + slot]


def test_SingleCellGather():
    idx = np.array([[3, 5]], dtype=np.int64)
    snap = run_gather(make_rd_maps(), idx)
    assert snap.shape == (1, K_C, 1)
    for c in range(K_C):
        exp = cell_value(c, 3, 5)
        got = snap_at(snap, 0, 0, c)
        assert got.real == pytest.approx(exp.real, abs=1e-6)
        assert got.imag == pytest.approx(exp.imag, abs=1e-6)


def test_WindowGatherPreservesChannelAxis():
    idx = np.array([[3, 5]], dtype=np.int64)
    snap = run_gather(
        make_rd_maps(),
        idx,
        snapshot_window=[1, 0],
    )
    assert snap.shape == (1, K_C, 3)
    d_bins = [2, 3, 4]
    for slot, d in enumerate(d_bins):
        for c in range(K_C):
            exp = cell_value(c, d, 5)
            got = snap_at(snap, 0, slot, c)
            assert got.real == pytest.approx(exp.real, abs=1e-6)
            assert got.imag == pytest.approx(exp.imag, abs=1e-6)


def test_EmptyDetections():
    idx = np.empty((0, 2), dtype=np.int64)
    snap = run_gather(make_rd_maps(), idx)
    assert snap.shape[0] == 0


def test_FloatInstantiationSmoke():
    idx = np.array([[3, 5]], dtype=np.int64)
    snap = run_gather(make_rd_maps(dtype=np.complex64), idx, op_cls=SnapshotGatherOpF32)
    assert snap.shape == (1, K_C, 1)
    exp = cell_value(0, 3, 5)
    assert snap_at(snap, 0, 0, 0).real == pytest.approx(float(exp.real), abs=1e-5)


def test_RejectsNegativeSnapshotWindow():
    idx = np.array([[3, 5]], dtype=np.int64)
    with pytest.raises(ValueError):
        run_gather(make_rd_maps(), idx, snapshot_window=[-1, 0])


def test_ClampIsValidMaskUint8Visible():
    # L1: is_valid emits as uint8 (0/1), so Python can read it on HSDK 4.3 (a
    # bool port would throw). Under a corner clamp only slots 4,5,7,8 are valid.
    idx = np.array([[0, 0]], dtype=np.int64)
    rd = to_device_array(make_rd_maps(), dtype=np.complex128)
    id_t = to_device_array(np.asarray(idx, dtype=np.int64), dtype=np.int64)

    def make_op(app):
        return SnapshotGatherOpF64(
            app,
            name="op_under_test",
            snapshot_window=[1, 1],
            doppler_edge=SnapshotEdgeMode.CLAMP,
            range_edge=SnapshotEdgeMode.CLAMP,
        )

    outs = run_operator_multi(
        make_op,
        single_map(rd, "rd_maps"),
        input_port="rd_maps",
        output_ports=["snapshot", "is_valid"],
        extra_inputs={"detection_idx": [single_map(id_t, "detection_idx")]},
    )
    valid = outs["is_valid"][0]
    assert valid.dtype == np.uint8
    assert valid.shape == (1, 9)
    assert set(np.unique(valid)).issubset({0, 1})
    expected = np.zeros((1, 9), dtype=np.uint8)
    expected[0, [4, 5, 7, 8]] = 1
    np.testing.assert_array_equal(valid, expected)


def test_ClampKeepsInBoundSlots():
    # Corner detection under clamp: only the four in-bound (d,r) slots are
    # written (buffer is zero-initialized); is_valid marks them (see
    # test_ClampIsValidMaskUint8Visible).
    idx = np.array([[0, 0]], dtype=np.int64)
    snap = run_gather(
        make_rd_maps(),
        idx,
        snapshot_window=[1, 1],
        doppler_edge=SnapshotEdgeMode.CLAMP,
        range_edge=SnapshotEdgeMode.CLAMP,
    )
    assert snap.shape == (1, K_C, 9)
    expected = {
        4: (0, 0),
        5: (0, 1),
        7: (1, 0),
        8: (1, 1),
    }
    invalid_slots = [0, 1, 2, 3, 6]
    for slot, (d, r) in expected.items():
        for c in range(K_C):
            exp = cell_value(c, d, r)
            got = snap_at(snap, 0, slot, c)
            assert got.real == pytest.approx(exp.real, abs=1e-6)
            assert got.imag == pytest.approx(exp.imag, abs=1e-6)
    for slot in invalid_slots:
        for c in range(K_C):
            got = snap_at(snap, 0, slot, c)
            assert got.real == pytest.approx(0.0, abs=1e-6)
            assert got.imag == pytest.approx(0.0, abs=1e-6)


def test_ShiftKeepsSnapshotCount():
    idx = np.array([[0, 0]], dtype=np.int64)
    snap = run_gather(
        make_rd_maps(),
        idx,
        snapshot_window=[1, 1],
        doppler_edge=SnapshotEdgeMode.SHIFT,
        range_edge=SnapshotEdgeMode.SHIFT,
    )
    assert snap.shape == (1, K_C, 9)
    for c in range(K_C):
        exp0 = cell_value(c, 0, 0)
        got0 = snap_at(snap, 0, 0, c)
        assert got0.real == pytest.approx(exp0.real, abs=1e-6)
        exp8 = cell_value(c, 2, 2)
        got8 = snap_at(snap, 0, 8, c)
        assert got8.real == pytest.approx(exp8.real, abs=1e-6)


def test_DopplerWrapAtBoundary():
    # Center at doppler bin 0; half=1 under wrap yields bins {7, 0, 1}.
    idx = np.array([[0, 5]], dtype=np.int64)
    snap = run_gather(
        make_rd_maps(),
        idx,
        snapshot_window=[1, 0],
        doppler_edge=SnapshotEdgeMode.WRAP,
    )
    assert snap.shape == (1, K_C, 3)
    d_bins = [7, 0, 1]
    for slot, d in enumerate(d_bins):
        for c in range(K_C):
            exp = cell_value(c, d, 5)
            got = snap_at(snap, 0, slot, c)
            assert got.real == pytest.approx(exp.real, abs=1e-6)
            assert got.imag == pytest.approx(exp.imag, abs=1e-6)
