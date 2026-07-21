# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python ports of core tests/cpp/fmcw/test_virtual_array_assembly.cpp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.harness import single_map  # noqa: E402
from utils.harness import (
    TensorMapCapture,
    to_device_array,
    wire_capture_sink,
    wire_data_source,
)

from holoscan.core import Application, Operator, OperatorSpec  # noqa: E402
from holoscan.radar import OverlapPolicy  # noqa: E402
from holoscan.radar import (
    ArrayGeometry,
    VirtualArrayAssemblyOpF32,
    VirtualArrayAssemblyOpF64,
)

pytestmark = pytest.mark.requires_gpu

K_NUM_TX = 2
K_NUM_RX = 2
K_DOPPLER = 2
K_RANGE = 2
K_WAVELENGTH = 0.03


@pytest.fixture(scope="module", autouse=True)
def _require_cuda_device():
    try:
        cupy.cuda.Device(0).compute_capability
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU required for VirtualArrayAssemblyOp tests: {exc}")


def separated_tx_positions() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def overlapping_tx_positions() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def default_rx_positions() -> list[float]:
    return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def analytic_virtual_positions(
    tx_positions: list[float], rx_positions: list[float], num_tx: int, num_rx: int
) -> list[float]:
    out: list[float] = []
    for tx in range(num_tx):
        for rx in range(num_rx):
            for i in range(3):
                out.append(tx_positions[tx * 3 + i] + rx_positions[rx * 3 + i])
    return out


def channel_marker(batch: int, tx: int, rx: int, doppler: int, range_bin: int) -> float:
    return (
        10000.0 * float(batch)
        + 100.0 * float(tx)
        + 10.0 * float(rx)
        + float(doppler)
        + 0.1 * float(range_bin)
    )


def build_rd_host(shape: tuple[int, ...], *, dtype=np.complex64) -> np.ndarray:
    num_tx = shape[-4]
    num_rx = shape[-3]
    doppler = shape[-2]
    range_bins = shape[-1]
    batch = 1
    for i in range(len(shape) - 4):
        batch *= shape[i]
    host = np.empty(int(np.prod(shape)), dtype=dtype)
    slice_size = num_tx * num_rx * doppler * range_bins
    for b in range(batch):
        for tx in range(num_tx):
            for rx in range(num_rx):
                for d in range(doppler):
                    for r in range(range_bins):
                        idx = b * slice_size + (((tx * num_rx + rx) * doppler) + d) * range_bins + r
                        m = channel_marker(b, tx, rx, d, r)
                        host[idx] = complex(m, 0.0)
    return host.reshape(shape)


class _ArrayGeometryCaptureOp(Operator):
    def __init__(self, fragment, *, store: list[ArrayGeometry], **kwargs):
        self._store = store
        super().__init__(fragment, **kwargs)

    def setup(self, spec: OperatorSpec) -> None:
        spec.input("input")

    def compute(self, op_input, op_output, context) -> None:
        self._store.append(op_input.receive("input"))


class _GeometryCapture:
    def __init__(self) -> None:
        self.frames: list[ArrayGeometry] = []


def run_assembly(
    host: np.ndarray,
    shape: tuple[int, ...],
    tx_positions: list[float],
    rx_positions: list[float],
    wavelength: float,
    policy: OverlapPolicy,
    *,
    op_cls=VirtualArrayAssemblyOpF32,
) -> tuple[np.ndarray, ArrayGeometry]:
    dtype = np.complex64 if op_cls is VirtualArrayAssemblyOpF32 else np.complex128
    rd = to_device_array(np.asarray(host, dtype=dtype), dtype=dtype)
    frame = single_map(rd, "rd_maps")
    rd_capture = TensorMapCapture()
    geom_capture = _GeometryCapture()

    class _App(Application):
        def compose(self) -> None:
            op = op_cls(
                self,
                tx_positions=tx_positions,
                rx_positions=rx_positions,
                wavelength=wavelength,
                overlap_policy=policy,
                name="op_under_test",
            )
            wire_data_source(self, op, "rd_maps", [frame])
            wire_capture_sink(self, op, "rd_maps", rd_capture)
            sink = _ArrayGeometryCaptureOp(self, store=geom_capture.frames, name="geom_sink")
            self.add_flow(op, sink, {("array_geometry", "input")})

    _App().run()
    assert len(rd_capture.frames) == 1
    assert len(geom_capture.frames) == 1
    return rd_capture.frames[0], geom_capture.frames[0]


def expect_geometry_positions(
    geom: ArrayGeometry, expected: list[float], wavelength: float = K_WAVELENGTH
):
    assert len(geom.element_positions) == len(expected)
    for got, exp in zip(geom.element_positions, expected):
        assert got == pytest.approx(exp, abs=1e-9)
    assert geom.wavelength == pytest.approx(wavelength, abs=1e-12)


def test_EmittedGeometryMatchesAnalyticLayout():
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    rd, geom = run_assembly(
        build_rd_host(shape),
        shape,
        separated_tx_positions(),
        default_rx_positions(),
        K_WAVELENGTH,
        OverlapPolicy.KEEP_ALL,
    )
    expected = analytic_virtual_positions(
        separated_tx_positions(), default_rx_positions(), K_NUM_TX, K_NUM_RX
    )
    expect_geometry_positions(geom, expected)
    assert rd.shape == (K_NUM_TX * K_NUM_RX, K_DOPPLER, K_RANGE)


def test_KeepAllPreservesChannels():
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    rd, geom = run_assembly(
        build_rd_host(shape),
        shape,
        separated_tx_positions(),
        default_rx_positions(),
        K_WAVELENGTH,
        OverlapPolicy.KEEP_ALL,
    )
    assert rd.shape[0] == K_NUM_TX * K_NUM_RX
    assert geom.num_elements() == K_NUM_TX * K_NUM_RX


def test_DedupCollapsesCoincidentPositions():
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    rd, geom = run_assembly(
        build_rd_host(shape),
        shape,
        overlapping_tx_positions(),
        default_rx_positions(),
        K_WAVELENGTH,
        OverlapPolicy.DEDUP,
    )
    assert rd.shape[0] == 2
    assert geom.num_elements() == 2
    expect_geometry_positions(geom, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def test_AverageCombinesCoincidentChannels():
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    rd, _ = run_assembly(
        build_rd_host(shape),
        shape,
        overlapping_tx_positions(),
        default_rx_positions(),
        K_WAVELENGTH,
        OverlapPolicy.AVERAGE,
    )
    assert rd.shape[0] == 2
    for d in range(K_DOPPLER):
        for r in range(K_RANGE):
            expected = 0.5 * (channel_marker(0, 0, 0, d, r) + channel_marker(0, 1, 0, d, r))
            assert rd[0, d, r].real == pytest.approx(expected, abs=1e-5)


def test_AverageCombinesAllCollidingChannels():
    # All TX/RX virtual positions collapse to the origin → one averaged channel.
    origin_rx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    rd, geom = run_assembly(
        build_rd_host(shape),
        shape,
        overlapping_tx_positions(),
        origin_rx,
        K_WAVELENGTH,
        OverlapPolicy.AVERAGE,
    )
    assert rd.shape[0] == 1
    assert geom.num_elements() == 1
    for d in range(K_DOPPLER):
        for r in range(K_RANGE):
            expected = 0.0
            for tx in range(K_NUM_TX):
                for rx in range(K_NUM_RX):
                    expected += channel_marker(0, tx, rx, d, r)
            expected /= float(K_NUM_TX * K_NUM_RX)
            assert rd[0, d, r].real == pytest.approx(expected, abs=1e-5)


def test_ComplexDoublePreservesValues():
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    rd, _ = run_assembly(
        build_rd_host(shape, dtype=np.complex128),
        shape,
        separated_tx_positions(),
        default_rx_positions(),
        K_WAVELENGTH,
        OverlapPolicy.KEEP_ALL,
        op_cls=VirtualArrayAssemblyOpF64,
    )
    assert rd.dtype == np.complex128
    for tx in range(K_NUM_TX):
        for rx in range(K_NUM_RX):
            v = tx * K_NUM_RX + rx
            for d in range(K_DOPPLER):
                for r in range(K_RANGE):
                    assert rd[v, d, r].real == pytest.approx(
                        channel_marker(0, tx, rx, d, r), abs=1e-9
                    )


def test_LeadingBatchAxesPreservedAndIndependent():
    shape = (2, K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    rd, geom = run_assembly(
        build_rd_host(shape),
        shape,
        separated_tx_positions(),
        default_rx_positions(),
        K_WAVELENGTH,
        OverlapPolicy.KEEP_ALL,
    )
    assert rd.shape[0] == 2
    assert rd.shape[1] == K_NUM_TX * K_NUM_RX
    assert geom.num_elements() == K_NUM_TX * K_NUM_RX


def test_RejectsInvalidWavelength():
    shape = (K_NUM_TX, K_NUM_RX, K_DOPPLER, K_RANGE)
    with pytest.raises(ValueError):
        run_assembly(
            build_rd_host(shape),
            shape,
            separated_tx_positions(),
            default_rx_positions(),
            0.0,
            OverlapPolicy.KEEP_ALL,
        )
