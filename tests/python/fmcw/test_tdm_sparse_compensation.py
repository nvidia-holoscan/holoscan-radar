# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python coverage for TDMSparseCompensationOp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402
from utils.mimo_helpers import (  # noqa: E402
    build_misaligned_rd_host,
    make_test_waveform,
)

from holoscan.radar import (  # noqa: E402
    OverlapPolicy,
    TDMSparseCompensationOpF32,
    TDMSparseCompensationOpF64,
)

pytestmark = pytest.mark.requires_gpu

K_TX = 2
K_RX = 2
K_DOPPLER = 8
K_RANGE = 4


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for sparse TDM tests: {exc}")


def run_sparse(op_cls, *, window=(0, 0), policy=OverlapPolicy.KEEP_ALL):
    waveform = make_test_waveform()
    dtype = np.complex64 if op_cls is TDMSparseCompensationOpF32 else np.complex128
    host = build_misaligned_rd_host(waveform, K_TX, K_RX, K_DOPPLER, K_RANGE, dtype=dtype)
    rd = to_device_array(host, dtype=dtype)
    idx = to_device_array(np.array([[3, 1]], dtype=np.int64), dtype=np.int64)

    def make_op(app):
        return op_cls(
            app,
            max_tx=K_TX,
            max_rx=K_RX,
            max_virtual_channels=K_TX * K_RX,
            max_doppler_bins=K_DOPPLER,
            max_range_bins=K_RANGE,
            max_detections=4,
            max_snapshots=9,
            waveform=waveform,
            tx_positions=[0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            rx_positions=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            overlap_policy=policy,
            snapshot_window=list(window),
            name="sparse",
        )

    return run_operator(
        make_op,
        single_map(rd, "rd_maps"),
        input_port="rd_maps",
        output_port="snapshot",
        extra_inputs={"detection_idx": [single_map(idx, "detection_idx")]},
        drain_ports=["is_valid"],
    )[0]


@pytest.mark.parametrize(
    "op_cls,tolerance",
    [(TDMSparseCompensationOpF32, 1e-5), (TDMSparseCompensationOpF64, 1e-12)],
)
def test_sparse_compensation_aligns_tx_lanes(op_cls, tolerance):
    snapshot = run_sparse(op_cls)
    assert snapshot.shape == (1, K_TX * K_RX, 1)
    for rx in range(K_RX):
        np.testing.assert_allclose(
            snapshot[0, rx, 0],
            snapshot[0, K_RX + rx, 0],
            atol=tolerance,
            rtol=tolerance,
        )


def test_window_compensates_each_doppler_cell():
    snapshot = run_sparse(TDMSparseCompensationOpF64, window=(1, 0))
    assert snapshot.shape == (1, K_TX * K_RX, 3)
    for rx in range(K_RX):
        np.testing.assert_allclose(
            snapshot[0, rx, :],
            snapshot[0, K_RX + rx, :],
            atol=1e-12,
            rtol=1e-12,
        )
