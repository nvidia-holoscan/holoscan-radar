# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of core tests/cpp/fmcw/test_mimo_coherent_compensation.cpp.

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
from utils.mimo_helpers import (  # noqa: E402
    K_DDM_DOPPLER,
    K_DDM_RANGE,
    K_DDM_RX,
    K_DDM_TX,
    build_ddm_input_host,
    build_misaligned_rd_host,
    build_tdm_correction_table,
    expect_tx_lanes_aligned,
    fft_center,
    make_offset_code,
    make_test_waveform,
    uniform_ddm_offsets,
)

from holoscan.radar import (  # noqa: E402
    MIMOCoherentCompensationOpF32,
    MIMOCoherentCompensationOpF64,
    MultiplexingScheme,
)

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
        pytest.skip(f"GPU required for MIMOCoherentCompensationOp tests: {exc}")


def run_compensation(
    host: np.ndarray,
    waveform,
    *,
    op_cls=MIMOCoherentCompensationOpF32,
    scheme=MultiplexingScheme.TDM,
    ddm_code=None,
) -> np.ndarray:
    dtype = np.complex64 if op_cls is MIMOCoherentCompensationOpF32 else np.complex128
    arr = to_device_array(np.asarray(host, dtype=dtype), dtype=dtype)
    frame = single_map(arr, "rd_maps")
    kwargs = dict(multiplexing_scheme=scheme, waveform=waveform, name="op_under_test")
    if ddm_code is not None:
        kwargs["ddm_code"] = ddm_code

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
    out_f = run_compensation(host_f, waveform, op_cls=MIMOCoherentCompensationOpF32)
    out_d = run_compensation(host_d, waveform, op_cls=MIMOCoherentCompensationOpF64)
    assert out_f.shape == out_d.shape
    np.testing.assert_allclose(out_f.real, out_d.real, atol=1e-4)
    np.testing.assert_allclose(out_f.imag, out_d.imag, atol=1e-4)


def test_LeadingBatchAxesPreservedAndIndependent():
    waveform = make_test_waveform()
    shape = (2, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    host = build_misaligned_rd_host(
        waveform, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE, batch=2
    )
    out = run_compensation(host, waveform)
    assert out.shape == shape
    expect_tx_lanes_aligned(out, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE, batch=2)


def test_RejectsUnspecifiedMultiplexingScheme():
    waveform = make_test_waveform()
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    host = np.full(shape, complex(1.0, 0.0), dtype=np.complex64)

    def make_op(app):
        return MIMOCoherentCompensationOpF32(
            app, waveform=waveform, name="op_under_test"
        )

    frame = single_map(to_device_array(host), "rd_maps")
    with pytest.raises(ValueError):
        run_operator(
            make_op,
            frame,
            input_port="rd_maps",
            output_port="rd_maps",
        )


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
        return MIMOCoherentCompensationOpF32(
            app,
            multiplexing_scheme=MultiplexingScheme.TDM,
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


def test_DdmDemuxRecoversEachTxIntoBand():
    waveform = make_test_waveform()
    code = make_offset_code(K_DDM_TX)
    band = K_DDM_DOPPLER // K_DDM_TX
    p0 = 1
    offsets = uniform_ddm_offsets(K_DDM_TX)
    host = build_ddm_input_host(
        offsets, K_DDM_TX, K_DDM_RX, K_DDM_DOPPLER, K_DDM_RANGE, p0
    )
    out = run_compensation(
        host,
        waveform,
        scheme=MultiplexingScheme.DDM,
        ddm_code=code,
    )
    assert out.shape == (K_DDM_TX, K_DDM_RX, band, K_DDM_RANGE)
    c_w = fft_center(band)
    o0 = (p0 - c_w) % band
    for m in range(K_DDM_TX):
        for rx in range(K_DDM_RX):
            for o in range(band):
                for r in range(K_DDM_RANGE):
                    expected = (
                        complex(1.0 + 0.1 * m + 0.01 * rx, 0.05 * r - 0.02 * m)
                        if o == o0
                        else 0.0 + 0.0j
                    )
                    got = out[m, rx, o, r]
                    assert got.real == pytest.approx(expected.real, abs=1e-4)
                    assert got.imag == pytest.approx(expected.imag, abs=1e-4)


def test_DdmGuardBandsReduceWidth():
    waveform = make_test_waveform()
    code = make_offset_code(K_DDM_TX, guard=1)
    band = K_DDM_DOPPLER // K_DDM_TX - 1  # W = 3
    p0 = 0
    offsets = uniform_ddm_offsets(K_DDM_TX)
    host = build_ddm_input_host(
        offsets, K_DDM_TX, K_DDM_RX, K_DDM_DOPPLER, K_DDM_RANGE, p0
    )
    out = run_compensation(
        host,
        waveform,
        scheme=MultiplexingScheme.DDM,
        ddm_code=code,
    )
    assert out.shape == (K_DDM_TX, K_DDM_RX, band, K_DDM_RANGE)
    c_w = fft_center(band)
    o0 = (p0 - c_w) % band
    for m in range(K_DDM_TX):
        for rx in range(K_DDM_RX):
            for o in range(band):
                for r in range(K_DDM_RANGE):
                    expected = (
                        complex(1.0 + 0.1 * m + 0.01 * rx, 0.05 * r - 0.02 * m)
                        if o == o0
                        else 0.0 + 0.0j
                    )
                    got = out[m, rx, o, r]
                    assert got.real == pytest.approx(expected.real, abs=1e-4)
                    assert got.imag == pytest.approx(expected.imag, abs=1e-4)


def test_DdmGuardBandDiscardsEdgeBin():
    # Target parked in the discarded top fft-order bin of each band must vanish.
    waveform = make_test_waveform()
    code = make_offset_code(K_DDM_TX, guard=1)
    band = K_DDM_DOPPLER // K_DDM_TX - 1  # W = 3
    edge = K_DDM_DOPPLER // K_DDM_TX - 1  # fft bin 3 (guarded)
    offsets = uniform_ddm_offsets(K_DDM_TX)
    host = build_ddm_input_host(
        offsets, K_DDM_TX, K_DDM_RX, K_DDM_DOPPLER, K_DDM_RANGE, edge
    )
    out = run_compensation(
        host,
        waveform,
        scheme=MultiplexingScheme.DDM,
        ddm_code=code,
    )
    assert out.shape == (K_DDM_TX, K_DDM_RX, band, K_DDM_RANGE)
    for m in range(K_DDM_TX):
        for rx in range(K_DDM_RX):
            for o in range(band):
                for r in range(K_DDM_RANGE):
                    got = out[m, rx, o, r]
                    assert got.real == pytest.approx(0.0, abs=1e-5)
                    assert got.imag == pytest.approx(0.0, abs=1e-5)


def _make_nl_phase_codes(num_tx: int, num_chirps: int):
    from holoscan.radar import DDMCodeDescriptor

    code = DDMCodeDescriptor()
    code.num_tx = num_tx
    code.guard_bands = 0
    codes = []
    for m in range(num_tx):
        for k in range(num_chirps):
            phase = math.sin(0.7 * float(k) + float(m)) * 2.0
            codes.append(complex(math.cos(phase), math.sin(phase)))
    code.phase_codes = codes
    return code


def test_DdmNonLinearPhaseCodesCorrelateRecovers():
    # Single-TX fft-order model, fed as center-zero (fftshift) to match the
    # Doppler processor contract. Peak lands at the fftshifted bin of d0.
    waveform = make_test_waveform()
    num_tx, num_rx, d, num_range = 1, 2, 16, 3
    d0 = 3
    d0_cz = (d0 + d // 2) % d
    val = complex(1.5, -0.3)
    code = _make_nl_phase_codes(num_tx, d)
    c_m = np.asarray(code.phase_codes, dtype=np.complex128)
    spectrum = np.fft.fft(c_m)  # matches unnormalized DFT in C++ host builder

    host_fft = np.zeros((num_rx, d, num_range), dtype=np.complex64)
    for rx in range(num_rx):
        for r in range(num_range):
            for k in range(d):
                host_fft[rx, k, r] = val * spectrum[(k - d0) % d]
    host = np.fft.fftshift(host_fft, axes=1)

    out = run_compensation(
        host,
        waveform,
        scheme=MultiplexingScheme.DDM,
        ddm_code=code,
    )
    assert out.shape == (num_tx, num_rx, d, num_range)
    expected = val * float(d)
    tol = 0.05 * abs(expected)
    for rx in range(num_rx):
        for r in range(num_range):
            peak = 0.0
            other = 0.0
            at_d0 = out[0, rx, d0_cz, r]
            for dop in range(d):
                mag = abs(out[0, rx, dop, r])
                if dop == d0_cz:
                    peak = mag
                else:
                    other = max(other, mag)
            assert at_d0.real == pytest.approx(expected.real, abs=tol)
            assert at_d0.imag == pytest.approx(expected.imag, abs=tol)
            assert peak > 5.0 * other


def test_DdmNonLinearRejectsNonZeroGuardBands():
    waveform = make_test_waveform()
    num_tx, num_rx, d, num_range = 2, 2, 16, 3
    code = _make_nl_phase_codes(num_tx, d)
    code.guard_bands = 1
    host = np.ones((num_rx, d, num_range), dtype=np.complex64)
    with pytest.raises(ValueError):
        run_compensation(
            host,
            waveform,
            scheme=MultiplexingScheme.DDM,
            ddm_code=code,
        )


def test_DdmNonLinearRejectsChirpDopplerMismatch():
    waveform = make_test_waveform()
    num_tx, num_rx, d, num_range = 2, 2, 16, 3
    code = _make_nl_phase_codes(num_tx, 8)  # chirps != doppler
    host = np.ones((num_rx, d, num_range), dtype=np.complex64)
    with pytest.raises(ValueError):
        run_compensation(
            host,
            waveform,
            scheme=MultiplexingScheme.DDM,
            ddm_code=code,
        )
