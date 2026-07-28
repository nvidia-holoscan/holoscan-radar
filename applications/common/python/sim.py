# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared analytic point-target simulator for the FMCW example apps.

Python port of ``radar_common/sim/point_target_sim.hpp``: synthesizes a
complex-IF ADC cube by summing each point target's closed-form contribution
over (rx, chirp, fast-time sample), then adds noise. A single-element RX
array (the default) reduces to the SISO cube; more RX elements produce the
SIMO / dense-path cube - one ``simulate()`` for both, exactly like the C++
``point_target_sim.hpp::simulate()`` it mirrors. Used by
``ops.SimAdcSourceOp`` so fmcw_siso / fmcw_simo / fmcw_dense_simo share one
simulator instance instead of each keeping its own copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cupy as cp

from holoscan.radar import K_PI
from holoscan.radar import K_SPEED_OF_LIGHT as K_C


@dataclass
class SimWaveform:
    carrier_frequency: float = 77e9
    bandwidth: float = 150e6
    chirp_duration: float = 20e-6
    sample_rate: float = 12.8e6
    chirp_repetition_interval: float = 25e-6

    def slope(self) -> float:
        return self.bandwidth / self.chirp_duration

    def wavelength(self, c: float = K_C) -> float:
        return c / self.carrier_frequency


@dataclass
class PointTarget:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: complex = 1.0 + 0.0j


@dataclass
class SimScene:
    targets: list[PointTarget] = field(default_factory=list)
    waveform: SimWaveform = field(default_factory=SimWaveform)
    num_chirps: int = 64
    samples_per_chirp: int = 256
    rx_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)]
    )
    propagation_speed: float = K_C
    snr_db: float | None = None
    # Fixed seed, reseeded every frame — same demo contract as C++
    # point_target_sim (frozen noise realization for reproducibility).
    seed: int = 0x5EED5EED


@dataclass
class SimResult:
    # Device (CuPy) complex64 IF cube, already shaped for emit: rank-2
    # (num_chirps, samples_per_chirp) for a single RX element, rank-3
    # (num_rx, num_chirps, samples_per_chirp) otherwise - mirrors
    # point_target_sim.hpp::SimResult::shape.
    cube: object
    shape: tuple[int, ...]


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def simulate(scene: SimScene) -> SimResult:
    """Simulate one CPI on GPU (CuPy); returns a rank-agnostic complex64 cube."""
    if not scene.rx_positions:
        raise ValueError("point_target_sim: need >= 1 RX element")
    num_rx = len(scene.rx_positions)
    nc = scene.num_chirps
    ns = scene.samples_per_chirp
    c = scene.propagation_speed
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    pri = scene.waveform.chirp_repetition_interval
    tx_pos = (0.0, 0.0, 0.0)

    rx_pos = cp.asarray(scene.rx_positions, dtype=cp.float64)
    t = cp.arange(ns, dtype=cp.float64) / fs
    k = cp.arange(nc, dtype=cp.float64)
    cube = cp.zeros((num_rx, nc, ns), dtype=cp.complex128)
    for target in scene.targets:
        rng = _norm(target.position)
        if rng <= 0.0:
            raise ValueError("point_target_sim: target at the origin")
        look = (
            target.position[0] / rng,
            target.position[1] / rng,
            target.position[2] / rng,
        )
        look_v = cp.asarray(look, dtype=cp.float64)
        v_r = _dot(target.velocity, look)
        apert_rx = rx_pos @ look_v + _dot(tx_pos, look)
        range_k = rng + v_r * k * pri
        tau = (2.0 * range_k[cp.newaxis, :] - apert_rx[:, cp.newaxis]) / c
        beat = slope * tau
        phase = (2.0 * K_PI) * (
            beat[:, :, cp.newaxis] * t[cp.newaxis, cp.newaxis, :] - f_c * tau[:, :, cp.newaxis]
        )
        cube += target.amplitude * cp.exp(1j * phase)

    if scene.snr_db is not None:
        signal_power = 1.0
        noise_power = signal_power / (10.0 ** (scene.snr_db / 10.0))
        sigma = math.sqrt(noise_power / 2.0)
        rng_gen = cp.random.default_rng(scene.seed)
        cube += sigma * (
            rng_gen.standard_normal(cube.shape) + 1j * rng_gen.standard_normal(cube.shape)
        )

    cube64 = cube.astype(cp.complex64, copy=False)
    if num_rx == 1:
        cube64 = cube64.reshape(nc, ns)
    return SimResult(cube=cube64, shape=tuple(int(d) for d in cube64.shape))


def advance_scene(scene: SimScene, dt: float) -> None:
    for target in scene.targets:
        target.position = (
            target.position[0] + target.velocity[0] * dt,
            target.position[1] + target.velocity[1] * dt,
            target.position[2] + target.velocity[2] * dt,
        )
