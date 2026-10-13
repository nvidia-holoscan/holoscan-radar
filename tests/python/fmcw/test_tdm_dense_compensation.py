# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of core tests/cpp/fmcw/test_tdm_dense_compensation.cpp.

Covers TDM correction-table layout/invariants, multi-TX alignment, float/double
agreement, batch axes, and basic DDM demux. C++ retains the full reject-path
and live-override matrix.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import run_operator, single_map, to_device_array  # noqa: E402
from utils.mimo_helpers import (
    build_misaligned_rd_host,
    build_tdm_correction_table,
    expect_tx_lanes_aligned,
    make_test_waveform,
)

from holoscan.radar import TDMDenseCompensationOpF32  # noqa: E402
from holoscan.radar import TDMDenseCompensationOpF64

pytestmark = pytest.mark.requires_gpu

K_NUM_TX = 2
K_NUM_RX = 2
K_DOPPLER = 8
K_RANGE = 4


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for TDMDenseCompensationOp tests: {exc}")


def run_compensation(
    host: np.ndarray,
    waveform,
    *,
    op_cls=TDMDenseCompensationOpF32,
) -> np.ndarray:
    dtype = np.complex64 if op_cls is TDMDenseCompensationOpF32 else np.complex128
    arr = to_device_array(np.asarray(host, dtype=dtype), dtype=dtype)
    frame = single_map(arr, "rd_maps")
    kwargs = dict(waveform=waveform, name="op_under_test")

    def make_op(app):
        return op_cls(app, **kwargs)

    outs = run_operator(
        make_op,
        frame,
        input_port="rd_maps",
        output_port="rd_maps",
    )
    assert len(outs) == 1
    return outs[0]


def test_SingleTxIsIdentity():
    waveform = make_test_waveform()
    host = build_misaligned_rd_host(waveform, 1, K_NUM_RX, K_DOPPLER, K_RANGE)
    out = run_compensation(host, waveform)
    assert np.allclose(out, host, atol=1e-5)


def test_TdmCorrectionTableIsUnitMagnitude():
    table = build_tdm_correction_table(make_test_waveform(), K_NUM_TX, K_DOPPLER)
    for factor in table:
        assert abs(factor) == pytest.approx(1.0, abs=1e-12)


def test_TdmCorrectionTableTxZeroIsIdentity():
    """tx=0 occupies indices [0, num_doppler) — must be 1+0j (layout check)."""
    table = build_tdm_correction_table(make_test_waveform(), K_NUM_TX, K_DOPPLER)
    for d in range(K_DOPPLER):
        assert table[d].real == pytest.approx(1.0, abs=1e-12), f"d={d}"
        assert table[d].imag == pytest.approx(0.0, abs=1e-12), f"d={d}"


def test_TdmCorrectionTableLayoutMatchesFormula():
    """Index must be tx * num_doppler + d (catches append-order layout bugs)."""
    waveform = make_test_waveform()
    num_tx, num_doppler = 3, K_DOPPLER
    table = build_tdm_correction_table(waveform, num_tx, num_doppler)
    pri = waveform.chirp_repetition_interval()
    prf = 1.0 / pri
    t_inter = pri / float(num_tx)
    for d in range(num_doppler):
        idx = (d - num_doppler // 2 + num_doppler) % num_doppler
        f_d = float(idx) / float(num_doppler) * prf
        if f_d >= prf / 2.0:
            f_d -= prf
        for tx in range(num_tx):
            phase = -2.0 * math.pi * f_d * float(tx) * t_inter
            expect = complex(math.cos(phase), math.sin(phase))
            got = table[tx * num_doppler + d]
            assert got.real == pytest.approx(expect.real, abs=1e-12), (tx, d)
            assert got.imag == pytest.approx(expect.imag, abs=1e-12), (tx, d)
            # Non-zero TX at a non-DC bin must not look like identity.
            if tx > 0 and abs(f_d) > 1e-9:
                assert abs(got - 1.0) > 1e-6, (tx, d, got)


def test_TdmCorrectionTableUsesFiringSlot():
    waveform = make_test_waveform()
    slot_by_tx = [1, 0]
    table = build_tdm_correction_table(waveform, K_NUM_TX, K_DOPPLER, slot_by_tx)
    identity = build_tdm_correction_table(waveform, K_NUM_TX, K_DOPPLER)
    for d in range(K_DOPPLER):
        first = table[1 * K_DOPPLER + d]
        assert first.real == pytest.approx(1.0, abs=1e-12)
        assert first.imag == pytest.approx(0.0, abs=1e-12)
        swapped = table[d]
        expect = identity[1 * K_DOPPLER + d]
        assert swapped.real == pytest.approx(expect.real, abs=1e-12)
        assert swapped.imag == pytest.approx(expect.imag, abs=1e-12)


def test_TdmRemovesInterTxPhaseAcrossDopplerBins():
    waveform = make_test_waveform()
    host = build_misaligned_rd_host(waveform, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    out = run_compensation(host, waveform)
    expect_tx_lanes_aligned(out, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)


def test_FourTxLanesAlignAfterCompensation():
    num_tx = 4
    waveform = make_test_waveform()
    host = build_misaligned_rd_host(waveform, num_tx, K_NUM_RX, K_DOPPLER, K_RANGE)
    out = run_compensation(host, waveform)
    expect_tx_lanes_aligned(out, num_tx, K_NUM_RX, K_DOPPLER, K_RANGE)


def test_ComplexFloatDtypePreserved():
    waveform = make_test_waveform()
    host = build_misaligned_rd_host(waveform, 1, K_NUM_RX, K_DOPPLER, K_RANGE)
    out = run_compensation(host, waveform)
    assert np.iscomplexobj(out)


def test_FloatDoubleAgreement():
    waveform = make_test_waveform()
    host_f = build_misaligned_rd_host(
        waveform, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE, dtype=np.complex64
    )
    host_d = build_misaligned_rd_host(
        waveform, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE, dtype=np.complex128
    )
    out_f = run_compensation(host_f, waveform, op_cls=TDMDenseCompensationOpF32)
    out_d = run_compensation(host_d, waveform, op_cls=TDMDenseCompensationOpF64)
    assert out_f.shape == out_d.shape
    np.testing.assert_allclose(out_f.real, out_d.real, atol=1e-4)
    np.testing.assert_allclose(out_f.imag, out_d.imag, atol=1e-4)


def test_LeadingBatchAxesPreservedAndIndependent():
    waveform = make_test_waveform()
    shape = (2, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    host = build_misaligned_rd_host(waveform, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE, batch=2)
    out = run_compensation(host, waveform)
    assert out.shape == shape
    expect_tx_lanes_aligned(out, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE, batch=2)


def test_RejectsInvalidWaveform():
    from holoscan.radar import FMCWWaveformDescriptor

    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    host = np.full(shape, complex(1.0, 0.0), dtype=np.complex64)
    with pytest.raises(ValueError):
        run_compensation(host, FMCWWaveformDescriptor())


def test_RejectsRealInput():
    waveform = make_test_waveform()
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    host = np.ones(shape, dtype=np.float32)

    def make_op(app):
        return TDMDenseCompensationOpF32(
            app,
            waveform=waveform,
            name="op_under_test",
        )

    frame = single_map(to_device_array(host, dtype=np.float32), "rd_maps")
    with pytest.raises(ValueError):
        run_operator(
            make_op,
            frame,
            input_port="rd_maps",
            output_port="rd_maps",
        )
