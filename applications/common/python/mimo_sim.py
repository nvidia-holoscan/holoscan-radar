# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MIMO TDM/DDM scene + analytic simulators for the fmcw_mimo example app.

Python port of ``radar_common/sim/mimo_sim.hpp``: reuses the shared
point-target primitives from ``sim.py`` (``SimWaveform``, ``PointTarget``,
``SimResult``) and adds the MIMO-specific scene (multiple TX elements), the
TDM slow-time-interleaved and DDM (linear-offset and non-linear phase-coded)
IF-cube simulators, and the DDM code helpers (linear-phase classification,
offset resolution, demo-only code generators). One consumer today
(fmcw_mimo), co-located at this layer to mirror the C++ file placement
(mimo_sim.hpp lives beside point_target_sim.hpp, not inside the app).
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field

import cupy as cp
from holoscan.radar import TXSchedule

from common.python.sim import K_PI, PointTarget, SimResult, SimWaveform

K_C = 299792458.0


@dataclass
class TargetTruth:
    range: float
    radial_velocity: float
    look_direction: tuple[float, float, float]


@dataclass
class MimoScene:
    targets: list[PointTarget] = field(default_factory=list)
    waveform: SimWaveform = field(default_factory=SimWaveform)
    num_tx: int = 2
    num_chirps: int = 64
    samples_per_chirp: int = 256
    tx_positions: list[tuple[float, float, float]] = field(default_factory=list)
    rx_positions: list[tuple[float, float, float]] = field(default_factory=list)
    propagation_speed: float = K_C
    snr_db: float | None = None
    # Fixed seed, reseeded every frame — same demo contract as C++
    # (frozen noise realization for reproducibility).
    seed: int = 0x5EED5EED


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def derive_truth(target: PointTarget) -> TargetTruth:
    rng = _norm(target.position)
    if rng <= 0.0:
        raise ValueError("point_target_sim: target at the origin")
    look = (
        target.position[0] / rng,
        target.position[1] / rng,
        target.position[2] / rng,
    )
    return TargetTruth(
        range=rng,
        radial_velocity=_dot(target.velocity, look),
        look_direction=look,
    )


def expand_chirp_tx_index(schedule: TXSchedule, slow_time: int) -> list[int]:
    schedule.validate()
    if slow_time <= 0 or slow_time % schedule.num_tx != 0:
        raise ValueError("TXSchedule: slow_time must be a positive multiple of num_tx")
    if schedule.chirp_tx_index:
        if len(schedule.chirp_tx_index) != slow_time:
            raise ValueError("TXSchedule: chirp_tx_index length must equal slow_time")
        pattern = list(schedule.chirp_tx_index)
    else:
        cycle = list(range(schedule.num_tx)) if not schedule.tx_order else list(schedule.tx_order)
        pattern = [cycle[k % schedule.num_tx] for k in range(slow_time)]
    for tx in pattern:
        if tx < 0 or tx >= schedule.num_tx:
            raise ValueError("TXSchedule: chirp TX index out of range")
    return pattern


def uniform_ddm_offsets(num_tx: int) -> list[float]:
    return [float(m) / float(num_tx) for m in range(num_tx)]


def fill_sin_phase_codes(code, num_chirps: int) -> None:
    """High-crosstalk NL codes (unit-test / perf recipe). Not for clean demos."""
    if code.num_tx < 1 or num_chirps < 1:
        raise ValueError("fill_sin_phase_codes: invalid args")
    code.doppler_offsets = []
    code.guard_bands = 0
    codes: list[complex] = []
    for m in range(code.num_tx):
        for k in range(num_chirps):
            phase = math.sin(0.7 * float(k) + float(m)) * 2.0
            codes.append(complex(math.cos(phase), math.sin(phase)))
    code.phase_codes = codes


def fill_zadoff_chu_phase_codes(code, num_chirps: int) -> None:
    """Low-crosstalk non-linear Zadoff-Chu roots (coprime to N).

    Quadratic phase fails the linear-phase classifier so the correlate path
    is used; distinct roots keep periodic cross-correlation low.
    """
    if code.num_tx < 1 or num_chirps < 1:
        raise ValueError("fill_zadoff_chu_phase_codes: invalid args")
    roots: list[int] = []
    for u in range(1, num_chirps):
        if math.gcd(u, num_chirps) == 1:
            roots.append(u)
            if len(roots) >= code.num_tx:
                break
    if len(roots) < code.num_tx:
        raise ValueError(
            "fill_zadoff_chu_phase_codes: need num_tx distinct roots coprime "
            "to num_chirps (try a longer cube.num_chirps)"
        )
    code.doppler_offsets = []
    code.guard_bands = 0
    odd_n = (num_chirps % 2) != 0
    codes: list[complex] = []
    for m in range(code.num_tx):
        u = float(roots[m])
        for k in range(num_chirps):
            kd = float(k)
            if odd_n:
                arg = -math.pi * u * kd * (kd + 1.0) / float(num_chirps)
            else:
                arg = -math.pi * u * kd * kd / float(num_chirps)
            codes.append(complex(math.cos(arg), math.sin(arg)))
    code.phase_codes = codes


def ddm_codes_are_linear(code) -> bool:
    """Mirror C++ ddm_classify_codes (1e-3 rad constant-step test)."""
    code.validate()
    if code.doppler_offsets:
        return True
    if not code.phase_codes:
        return True
    num_tx = int(code.num_tx)
    num_chirps = len(code.phase_codes) // num_tx
    if num_chirps < 2:
        return False
    two_pi = 2.0 * math.pi
    for m in range(num_tx):
        row = code.phase_codes[m * num_chirps : (m + 1) * num_chirps]
        delta = cmath.phase(row[1] * row[0].conjugate())
        for k in range(num_chirps - 1):
            step = cmath.phase(row[k + 1] * row[k].conjugate())
            d = step - delta
            d -= two_pi * round(d / two_pi)
            if abs(d) > 1e-3:
                return False
    return True


def resolve_ddm_offsets(code) -> list[float]:
    """Resolve linear-phase DDM offsets (raises if codes are non-linear)."""
    code.validate()
    if code.doppler_offsets:
        return [o - math.floor(o) for o in code.doppler_offsets]
    if not code.phase_codes:
        return uniform_ddm_offsets(code.num_tx)
    if not ddm_codes_are_linear(code):
        raise ValueError("resolve_ddm_offsets: phase_codes are non-linear; use correlate path")
    num_tx = int(code.num_tx)
    num_chirps = len(code.phase_codes) // num_tx
    offsets: list[float] = []
    for m in range(num_tx):
        row = code.phase_codes[m * num_chirps : (m + 1) * num_chirps]
        delta = cmath.phase(row[1] * row[0].conjugate())
        o = delta / (2.0 * math.pi)
        o -= math.floor(o)
        offsets.append(o)
    return offsets


def simulate_mimo_tdm_interleaved(scene: MimoScene, schedule: TXSchedule) -> SimResult:
    """Analytic TDM-interleaved IF cube on GPU (CuPy)."""
    if not scene.tx_positions or not scene.rx_positions:
        raise ValueError("simulate_mimo_tdm_interleaved: need TX and RX positions")
    num_rx = len(scene.rx_positions)
    num_tx = scene.num_tx
    slow_time = num_tx * scene.num_chirps
    samples = scene.samples_per_chirp
    physical_pri = scene.waveform.chirp_repetition_interval
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    c = scene.propagation_speed

    pattern = cp.asarray(expand_chirp_tx_index(schedule, slow_time), dtype=cp.int64)
    tx_pos = cp.asarray(scene.tx_positions, dtype=cp.float64)
    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    if tx_pos.shape != (num_tx, 3) or rx_pos.shape != (num_rx, 3):
        raise ValueError("simulate_mimo_tdm_interleaved: tx/rx positions must be (N, 3)")

    t = cp.arange(samples, dtype=cp.float64) / fs
    k = cp.arange(slow_time, dtype=cp.float64)
    cube = cp.zeros((num_rx, slow_time, samples), dtype=cp.complex128)

    for target in scene.targets:
        truth = derive_truth(target)
        look = cp.asarray(truth.look_direction, dtype=cp.float64)
        range_k = truth.range + truth.radial_velocity * k * physical_pri
        apert_tx_k = (tx_pos @ look)[pattern]
        apert_rx = rx_pos @ look
        tau = (
            2.0 * range_k[cp.newaxis, :] - apert_tx_k[cp.newaxis, :] - apert_rx[:, cp.newaxis]
        ) / c
        beat = slope * tau
        phase = (2.0 * K_PI) * (
            beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :] - f_c * tau[:, :, cp.newaxis]
        )
        cube += target.amplitude * cp.exp(1j * phase)

    if scene.snr_db is not None:
        noise_power = 10.0 ** (-scene.snr_db / 10.0)
        sigma = math.sqrt(noise_power / 2.0)
        rng = cp.random.default_rng(scene.seed)
        cube += sigma * (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape))

    cube64 = cube.astype(cp.complex64, copy=False)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def simulate_mimo_ddm_coded(scene: MimoScene, offsets: list[float]) -> SimResult:
    """Analytic DDM-coded IF cube on GPU (CuPy)."""
    if not scene.tx_positions or not scene.rx_positions:
        raise ValueError("simulate_mimo_ddm_coded: need TX and RX positions")
    num_rx = len(scene.rx_positions)
    num_tx = scene.num_tx
    num_chirps = scene.num_chirps
    samples = scene.samples_per_chirp
    pri = scene.waveform.chirp_repetition_interval
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    c = scene.propagation_speed

    if len(offsets) != num_tx:
        raise ValueError("simulate_mimo_ddm_coded: offsets length must equal num_tx")

    tx_pos = cp.asarray(scene.tx_positions, dtype=cp.float64)
    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    if tx_pos.shape != (num_tx, 3) or rx_pos.shape != (num_rx, 3):
        raise ValueError("simulate_mimo_ddm_coded: tx/rx positions must be (N, 3)")
    offsets_arr = cp.asarray(offsets, dtype=cp.float64)

    t = cp.arange(samples, dtype=cp.float64) / fs
    k = cp.arange(num_chirps, dtype=cp.float64)
    cube = cp.zeros((num_rx, num_chirps, samples), dtype=cp.complex128)

    for target in scene.targets:
        truth = derive_truth(target)
        look = cp.asarray(truth.look_direction, dtype=cp.float64)
        range_k = truth.range + truth.radial_velocity * k * pri
        apert_tx = tx_pos @ look
        apert_rx = rx_pos @ look
        for tx in range(num_tx):
            tau = (2.0 * range_k[cp.newaxis, :] - apert_tx[tx] - apert_rx[:, cp.newaxis]) / c
            beat = slope * tau
            code_phase = (2.0 * K_PI) * offsets_arr[tx] * k
            phase = (2.0 * K_PI) * (
                beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :] - f_c * tau[:, :, cp.newaxis]
            ) + code_phase[cp.newaxis, :, cp.newaxis]
            cube += target.amplitude * cp.exp(1j * phase)

    if scene.snr_db is not None:
        noise_power = 10.0 ** (-scene.snr_db / 10.0)
        sigma = math.sqrt(noise_power / 2.0)
        rng = cp.random.default_rng(scene.seed)
        cube += sigma * (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape))

    cube64 = cube.astype(cp.complex64, copy=False)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def simulate_mimo_ddm_phase_coded(scene: MimoScene, phase_codes: list[complex]) -> SimResult:
    """Analytic DDM phase-coded IF cube on GPU (CuPy) — non-linear codes."""
    if not scene.tx_positions or not scene.rx_positions:
        raise ValueError("simulate_mimo_ddm_phase_coded: need TX and RX positions")
    num_rx = len(scene.rx_positions)
    num_tx = scene.num_tx
    num_chirps = scene.num_chirps
    samples = scene.samples_per_chirp
    pri = scene.waveform.chirp_repetition_interval
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    c = scene.propagation_speed

    if len(phase_codes) != num_tx * num_chirps:
        raise ValueError(
            "simulate_mimo_ddm_phase_coded: phase_codes length must be " "num_tx * num_chirps"
        )

    tx_pos = cp.asarray(scene.tx_positions, dtype=cp.float64)
    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    if tx_pos.shape != (num_tx, 3) or rx_pos.shape != (num_rx, 3):
        raise ValueError("simulate_mimo_ddm_phase_coded: tx/rx positions must be (N, 3)")
    codes = cp.asarray(phase_codes, dtype=cp.complex128).reshape(num_tx, num_chirps)

    t = cp.arange(samples, dtype=cp.float64) / fs
    k = cp.arange(num_chirps, dtype=cp.float64)
    cube = cp.zeros((num_rx, num_chirps, samples), dtype=cp.complex128)

    for target in scene.targets:
        truth = derive_truth(target)
        look = cp.asarray(truth.look_direction, dtype=cp.float64)
        range_k = truth.range + truth.radial_velocity * k * pri
        apert_tx = tx_pos @ look
        apert_rx = rx_pos @ look
        for tx in range(num_tx):
            tau = (2.0 * range_k[cp.newaxis, :] - apert_tx[tx] - apert_rx[:, cp.newaxis]) / c
            beat = slope * tau
            phase = (2.0 * K_PI) * (
                beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :] - f_c * tau[:, :, cp.newaxis]
            )
            cube += target.amplitude * codes[tx][cp.newaxis, :, cp.newaxis] * cp.exp(1j * phase)

    if scene.snr_db is not None:
        noise_power = 10.0 ** (-scene.snr_db / 10.0)
        sigma = math.sqrt(noise_power / 2.0)
        rng = cp.random.default_rng(scene.seed)
        cube += sigma * (rng.standard_normal(cube.shape) + 1j * rng.standard_normal(cube.shape))

    cube64 = cube.astype(cp.complex64, copy=False)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def advance_scene(scene: MimoScene, dt: float) -> None:
    for target in scene.targets:
        target.position = (
            target.position[0] + target.velocity[0] * dt,
            target.position[1] + target.velocity[1] * dt,
            target.position[2] + target.velocity[2] * dt,
        )
