# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared MIMO test fixtures (TX schedule, TDM waveform, marker cubes)."""

from __future__ import annotations

import math

import numpy as np

from holoscan.radar import (
    DDMCodeDescriptor,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    TXSchedule,
)

K_PRI = 1.0e-3
K_NUM_TX = 2
K_NUM_RX = 2
K_CHIRPS_PER_TX = 3
K_SAMPLES = 4
K_SLOW_TIME = K_NUM_TX * K_CHIRPS_PER_TX

K_DDM_TX = 4
K_DDM_RX = 2
K_DDM_DOPPLER = 16
K_DDM_RANGE = 3


def make_test_waveform(pri: float = K_PRI) -> FMCWWaveformDescriptor:
    spec = FMCWWaveformSpec()
    spec.carrier_frequency = 77.0e9
    spec.bandwidth = 1.0e6
    spec.chirp_duration = 50.0e-6
    spec.sample_rate = 2.0e6
    spec.chirp_repetition_interval = pri
    return FMCWWaveformDescriptor(spec)


def expand_chirp_tx_index(schedule: TXSchedule, slow_time: int) -> list[int]:
    """Python port of TXSchedule::expand_chirp_tx_index (not bound in pybind)."""
    schedule.validate()
    num_tx = schedule.num_tx
    if slow_time <= 0 or slow_time % num_tx != 0:
        raise ValueError("TXSchedule: slow_time must be a positive multiple of num_tx")

    if schedule.chirp_tx_index:
        if len(schedule.chirp_tx_index) != slow_time:
            raise ValueError("TXSchedule: chirp_tx_index length must equal slow_time")
        pattern = list(schedule.chirp_tx_index)
    else:
        cycle = list(range(num_tx)) if not schedule.tx_order else list(schedule.tx_order)
        pattern = [cycle[k % num_tx] for k in range(slow_time)]

    for tx in pattern:
        if tx < 0 or tx >= num_tx:
            raise ValueError("TXSchedule: chirp TX index out of range")

    for k in range(num_tx, slow_time):
        if pattern[k] != pattern[k % num_tx]:
            raise ValueError("TXSchedule: non-uniform / non-repeating schedule rejected")

    chirps_per_tx = slow_time // num_tx
    counts = [0] * num_tx
    for tx in pattern:
        counts[tx] += 1
    for tx in range(num_tx):
        if counts[tx] != chirps_per_tx:
            raise ValueError("TXSchedule: each TX must appear exactly chirps_per_tx times")
    return pattern


def marker_value(batch: int, tx: int, chirp_of_tx: int, rx: int, sample: int) -> float:
    return (
        10000.0 * float(batch)
        + 1000.0 * float(tx)
        + 10.0 * float(chirp_of_tx)
        + 0.1 * float(rx)
        + 0.01 * float(sample)
    )


def input_leading_batch(shape: tuple[int, ...]) -> int:
    batch = 1
    for i in range(len(shape) - 3):
        batch *= shape[i]
    return batch


def output_leading_batch(shape: tuple[int, ...]) -> int:
    batch = 1
    for i in range(len(shape) - 4):
        batch *= shape[i]
    return batch


def build_interleaved_cube(
    pattern: list[int],
    shape: tuple[int, ...],
    *,
    real: bool = False,
    dtype=np.complex64,
) -> np.ndarray:
    num_rx = shape[-3]
    slow_time = shape[-2]
    samples = shape[-1]
    batch = input_leading_batch(shape)
    slice_elems = num_rx * slow_time * samples
    data = np.empty(batch * slice_elems, dtype=dtype)
    seen = [0] * K_NUM_TX
    for b in range(batch):
        seen = [0] * K_NUM_TX
        for k in range(slow_time):
            tx = pattern[k]
            chirp_of_tx = seen[tx]
            seen[tx] += 1
            for rx in range(num_rx):
                for s in range(samples):
                    idx = b * slice_elems + (rx * slow_time + k) * samples + s
                    m = marker_value(b, tx, chirp_of_tx, rx, s)
                    if real:
                        data[idx] = m
                    else:
                        data[idx] = complex(m, 0.0)
    return data.reshape(shape)


def _doppler_frequency_hz(shifted_bin: int, num_doppler: int, prf: float) -> float:
    idx = (shifted_bin - num_doppler // 2 + num_doppler) % num_doppler
    f = float(idx) / float(num_doppler) * prf
    if f >= prf / 2.0:
        f -= prf
    return f


def build_tdm_correction_table(
    waveform: FMCWWaveformDescriptor, num_tx: int, num_doppler: int
) -> list[complex]:
    if not waveform.is_valid():
        raise ValueError("waveform must be valid")
    pri = waveform.chirp_repetition_interval()
    prf = 1.0 / pri
    t_inter_chirp = pri / float(num_tx)
    # Layout matches C++ build_tdm_correction_table: index = tx * num_doppler + d.
    table = [0j] * (num_tx * num_doppler)
    for d in range(num_doppler):
        f_d = _doppler_frequency_hz(d, num_doppler, prf)
        for tx in range(num_tx):
            phase = -2.0 * math.pi * f_d * float(tx) * t_inter_chirp
            table[tx * num_doppler + d] = complex(math.cos(phase), math.sin(phase))
    return table


def build_misaligned_rd_host(
    waveform: FMCWWaveformDescriptor,
    num_tx: int,
    num_rx: int,
    num_doppler: int,
    num_range: int,
    batch: int = 1,
    *,
    dtype=np.complex64,
) -> np.ndarray:
    table = build_tdm_correction_table(waveform, num_tx, num_doppler)
    slice_size = num_tx * num_rx * num_doppler * num_range
    host = np.empty(batch * slice_size, dtype=dtype)
    for b in range(batch):
        for tx in range(num_tx):
            for rx in range(num_rx):
                for d in range(num_doppler):
                    for r in range(num_range):
                        idx = b * slice_size + (
                            ((tx * num_rx + rx) * num_doppler + d) * num_range + r
                        )
                        corr = table[tx * num_doppler + d]
                        mag = 1.0 + 0.01 * float(rx) + 0.001 * float(r)
                        misaligned = np.conj(corr) * mag
                        host[idx] = misaligned
    shape = (
        (num_tx, num_rx, num_doppler, num_range)
        if batch == 1
        else (batch, num_tx, num_rx, num_doppler, num_range)
    )
    return host.reshape(shape)


def expect_tx_lanes_aligned(
    out: np.ndarray,
    num_tx: int,
    num_rx: int,
    num_doppler: int,
    num_range: int,
    batch: int = 1,
    tol: float = 1e-4,
) -> None:
    import pytest

    slice_size = num_tx * num_rx * num_doppler * num_range
    flat = out.reshape(-1)
    for b in range(batch):
        for rx in range(num_rx):
            for d in range(num_doppler):
                for r in range(num_range):
                    ref = b * slice_size + ((0 * num_rx + rx) * num_doppler + d) * num_range + r
                    for tx in range(1, num_tx):
                        other = (
                            b * slice_size + ((tx * num_rx + rx) * num_doppler + d) * num_range + r
                        )
                        assert flat[other].real == pytest.approx(flat[ref].real, abs=tol)
                        assert flat[other].imag == pytest.approx(flat[ref].imag, abs=tol)


def fft_center(n: int) -> int:
    return (n + 1) // 2


def round_robin_schedule() -> TXSchedule:
    sched = TXSchedule()
    sched.num_tx = K_NUM_TX
    return sched


def custom_order_schedule() -> TXSchedule:
    sched = TXSchedule()
    sched.num_tx = K_NUM_TX
    sched.tx_order = [1, 0]
    return sched


def explicit_chirp_index_schedule() -> TXSchedule:
    """Uniform alternating chirp_tx_index (same coverage as round-robin)."""
    sched = TXSchedule()
    sched.num_tx = K_NUM_TX
    sched.chirp_tx_index = [0, 1, 0, 1, 0, 1]
    return sched


def uniform_ddm_offsets(num_tx: int) -> list[float]:
    return [float(m) / float(num_tx) for m in range(num_tx)]


def make_offset_code(num_tx: int, guard: int = 0) -> DDMCodeDescriptor:
    code = DDMCodeDescriptor()
    code.num_tx = num_tx
    code.doppler_offsets = uniform_ddm_offsets(num_tx)
    code.guard_bands = guard
    return code


def ddm_cell_value(tx: int, rx: int, r: int) -> complex:
    return complex(1.0 + 0.1 * float(tx) + 0.01 * float(rx), 0.05 * float(r) - 0.02 * float(tx))


def build_ddm_input_host(
    offsets: list[float],
    num_tx: int,
    num_rx: int,
    num_doppler: int,
    num_range: int,
    p0_reduced: int,
    batch: int = 1,
    *,
    dtype=np.complex64,
) -> np.ndarray:
    d = num_doppler
    c = fft_center(d)
    slice_size = num_rx * d * num_range
    host = np.zeros(batch * slice_size, dtype=dtype)
    for b in range(batch):
        for m in range(num_tx):
            start = int(round(offsets[m] * float(d))) % d
            j = (start + p0_reduced) % d
            in_cz = ((j - c) % d + d) % d
            for rx in range(num_rx):
                for r in range(num_range):
                    idx = b * slice_size + (rx * d + in_cz) * num_range + r
                    host[idx] = ddm_cell_value(m, rx, r)
    shape = (
        (batch, num_rx, num_doppler, num_range) if batch > 1 else (num_rx, num_doppler, num_range)
    )
    return host.reshape(shape)
