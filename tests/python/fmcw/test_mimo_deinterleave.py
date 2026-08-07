# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of core tests/cpp/fmcw/test_mimo_deinterleave.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402
from utils.mimo_helpers import K_NUM_RX  # noqa: E402
from utils.mimo_helpers import (
    K_CHIRPS_PER_TX,
    K_NUM_TX,
    K_SAMPLES,
    K_SLOW_TIME,
    build_interleaved_cube,
    custom_order_schedule,
    expand_chirp_tx_index,
    explicit_chirp_index_schedule,
    marker_value,
    round_robin_schedule,
)

from holoscan.radar import MIMODeinterleaveOpF32, MIMODeinterleaveOpF64  # noqa: E402

pytestmark = pytest.mark.requires_gpu


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for MIMODeinterleaveOp tests: {exc}")


def run_deinterleave(
    host: np.ndarray,
    shape: tuple[int, ...],
    schedule,
    *,
    op_cls=MIMODeinterleaveOpF32,
    real: bool = False,
) -> np.ndarray:
    dtype = np.float32 if real else np.complex64
    if op_cls is MIMODeinterleaveOpF64:
        dtype = np.float64 if real else np.complex128
    arr = to_device_array(np.asarray(host, dtype=dtype), dtype=dtype)
    frame = single_map(arr, "adc_cube")

    def make_op(app):
        return op_cls(app, tx_schedule=schedule, name="op_under_test")

    outs = run_operator(
        make_op,
        frame,
        input_port="adc_cube",
        output_port="adc_cube",
    )
    assert len(outs) == 1
    return outs[0]


def expect_deinterleaved(out: np.ndarray, shape: tuple[int, ...], *, real: bool = False) -> None:
    num_tx = shape[-4]
    num_rx = shape[-3]
    chirps_per_tx = shape[-2]
    samples = shape[-1]
    batch = 1
    for i in range(len(shape) - 4):
        batch *= shape[i]
    flat = out.reshape(-1)
    slice_elems = num_tx * num_rx * chirps_per_tx * samples
    for b in range(batch):
        for tx in range(num_tx):
            for rx in range(num_rx):
                for chirp_of_tx in range(chirps_per_tx):
                    for s in range(samples):
                        idx = b * slice_elems + (
                            ((tx * num_rx + rx) * chirps_per_tx + chirp_of_tx) * samples + s
                        )
                        expected = marker_value(b, tx, chirp_of_tx, rx, s)
                        if real:
                            assert flat[idx] == pytest.approx(expected, rel=1e-4, abs=1e-4)
                        else:
                            assert flat[idx].real == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_RoundRobinComplexFloat():
    shape = (K_NUM_RX, K_SLOW_TIME, K_SAMPLES)
    schedule = round_robin_schedule()
    pattern = expand_chirp_tx_index(schedule, K_SLOW_TIME)
    host = build_interleaved_cube(pattern, shape)
    out = run_deinterleave(host, (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES), schedule)
    expect_deinterleaved(out, (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES))


def test_CustomTxOrderComplexFloat():
    shape = (K_NUM_RX, K_SLOW_TIME, K_SAMPLES)
    schedule = custom_order_schedule()
    pattern = expand_chirp_tx_index(schedule, K_SLOW_TIME)
    host = build_interleaved_cube(pattern, shape)
    out = run_deinterleave(host, (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES), schedule)
    expect_deinterleaved(out, (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES))


def test_ExplicitChirpTxIndexUniformPattern():
    shape = (K_NUM_RX, K_SLOW_TIME, K_SAMPLES)
    schedule = explicit_chirp_index_schedule()
    pattern = expand_chirp_tx_index(schedule, K_SLOW_TIME)
    host = build_interleaved_cube(pattern, shape)
    out = run_deinterleave(host, (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES), schedule)
    expect_deinterleaved(out, (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES))


def test_RejectsNonUniformSchedule():
    schedule = round_robin_schedule()
    schedule.chirp_tx_index = [0, 1, 0, 1, 0, 0]
    shape = (K_NUM_RX, K_SLOW_TIME, K_SAMPLES)
    host = np.full(shape, complex(1.0, 0.0), dtype=np.complex64)
    with pytest.raises(ValueError):
        expand_chirp_tx_index(schedule, K_SLOW_TIME)
    with pytest.raises(ValueError):
        run_deinterleave(host, (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES), schedule)


def test_LeadingBatchAxesPreservedAndIndependent():
    shape = (2, K_NUM_RX, K_SLOW_TIME, K_SAMPLES)
    schedule = round_robin_schedule()
    pattern = expand_chirp_tx_index(schedule, K_SLOW_TIME)
    host = build_interleaved_cube(pattern, shape)
    out = run_deinterleave(host, (2, K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES), schedule)
    assert out.shape[0] == 2
    expect_deinterleaved(out, (2, K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES))


def test_FloatDoubleAgreement():
    shape = (K_NUM_RX, K_SLOW_TIME, K_SAMPLES)
    schedule = round_robin_schedule()
    pattern = expand_chirp_tx_index(schedule, K_SLOW_TIME)
    host_f = build_interleaved_cube(pattern, shape, dtype=np.complex64)
    host_d = build_interleaved_cube(pattern, shape, dtype=np.complex128)
    out_f = run_deinterleave(
        host_f,
        (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES),
        schedule,
        op_cls=MIMODeinterleaveOpF32,
    )
    out_d = run_deinterleave(
        host_d,
        (K_NUM_TX, K_NUM_RX, K_CHIRPS_PER_TX, K_SAMPLES),
        schedule,
        op_cls=MIMODeinterleaveOpF64,
    )
    assert out_f.shape == out_d.shape
    for i in range(out_f.size):
        assert out_f.reshape(-1)[i].real == pytest.approx(
            float(out_d.reshape(-1)[i].real), abs=1e-4
        )
        assert out_f.reshape(-1)[i].imag == pytest.approx(
            float(out_d.reshape(-1)[i].imag), abs=1e-4
        )
