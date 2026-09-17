# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python parity coverage for DopplerNotchOp."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402

from holoscan.radar import (  # noqa: E402
    DopplerNotchOpF32,
    DopplerNotchOpF64,
)

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for DopplerNotchOp tests: {exc}")


def run_notch(host, axis, *, op_cls=DopplerNotchOpF32, **kwargs):
    complex_dtype = np.complex64 if op_cls is DopplerNotchOpF32 else np.complex128
    real_dtype = np.float32 if op_cls is DopplerNotchOpF32 else np.float64
    frame = single_map(to_device_array(host, dtype=complex_dtype), "rd_maps")
    axis_frame = single_map(to_device_array(axis, dtype=real_dtype), "velocity_bins")

    def make_op(app):
        return op_cls(
            app,
            max_doppler_bins=host.shape[-2],
            max_range_bins=host.shape[-1],
            max_rx=host.shape[-3] if host.ndim >= 3 else 1,
            max_tx=host.shape[-4] if host.ndim == 4 else 0,
            name="op_under_test",
            **kwargs,
        )

    outputs = run_operator(
        make_op,
        frame,
        input_port="rd_maps",
        output_port="rd_maps_out",
        extra_inputs={"velocity_bins": [axis_frame]},
    )
    assert len(outputs) == 1
    return outputs[0]


@pytest.mark.parametrize(
    ("op_cls", "dtype"),
    [(DopplerNotchOpF32, np.complex64), (DopplerNotchOpF64, np.complex128)],
)
def test_hard_notch_float_double(op_cls, dtype):
    host = np.full((5, 2), 2 - 1j, dtype=dtype)
    axis = np.arange(-2, 3)
    output = run_notch(host, axis, op_cls=op_cls)
    expected = host.copy()
    expected[2] = 0
    np.testing.assert_allclose(output, expected)


def test_cosine_transition_broadcasts_over_channels():
    host = np.full((2, 2, 5, 2), 2 - 1j, dtype=np.complex64)
    axis = np.arange(-2, 3, dtype=np.float32)
    output = run_notch(host, axis, transition_width=2.0)
    gain = np.asarray([1.0, 0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    np.testing.assert_allclose(output, host * gain[None, None, :, None])
    nonzero = output != 0
    np.testing.assert_allclose(output.imag[nonzero] / output.real[nonzero], -0.5)


def test_cyclic_edge_transition():
    host = np.ones((5, 1), dtype=np.complex64)
    axis = np.arange(-2, 3, dtype=np.float32)
    output = run_notch(host, axis, center_velocity=2.0, transition_width=2.0)
    np.testing.assert_allclose(output[:, 0], [0.5, 1.0, 1.0, 0.5, 0.0])
