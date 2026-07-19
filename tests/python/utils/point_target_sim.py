# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal Python port of ``radar_common/sim/point_target_sim.hpp``.

Used by pytest operator ports that need analytic ADC cubes. Matches the C++
simulator's dechirped / RF paths for SISO and multi-RX scenes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

K_C = 299792458.0
K_PI = math.pi


@dataclass
class SimWaveform:
    carrier_frequency: float = 77e9
    bandwidth: float = 150e6
    chirp_duration: float = 20e-6
    sample_rate: float = 12.8e6
    chirp_repetition_interval: float = 25e-6

    def slope(self) -> float:
        return self.bandwidth / self.chirp_duration

    def wavelength(self) -> float:
        return K_C / self.carrier_frequency


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
    tx_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_positions: list[tuple[float, float, float]] = field(
        default_factory=lambda: [(0.0, 0.0, 0.0)]
    )
    propagation_speed: float = K_C
    dechirped: bool = True
    snr_db: float | None = None


@dataclass
class SimResult:
    cube: np.ndarray  # complex128, row-major flat or already shaped
    shape: tuple[int, ...]


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def simulate(scene: SimScene) -> SimResult:
    """Simulate one CPI; returns flat ``cube`` and ``shape`` like the C++ API."""
    num_rx = len(scene.rx_positions)
    nc = scene.num_chirps
    ns = scene.samples_per_chirp
    c = scene.propagation_speed
    fs = scene.waveform.sample_rate
    slope = scene.waveform.slope()
    f_c = scene.waveform.carrier_frequency
    pri = scene.waveform.chirp_repetition_interval

    cube = np.zeros((num_rx, nc, ns), dtype=np.complex128)
    for target in scene.targets:
        rng = _norm(target.position)
        if rng <= 0.0:
            raise ValueError("point_target_sim: target at the origin")
        look = (
            target.position[0] / rng,
            target.position[1] / rng,
            target.position[2] / rng,
        )
        v_r = _dot(target.velocity, look)
        for rx, rx_pos in enumerate(scene.rx_positions):
            aperture = _dot(scene.tx_position, look) + _dot(rx_pos, look)
            for k in range(nc):
                range_k = rng + v_r * k * pri
                tau = (2.0 * range_k - aperture) / c
                beat = slope * tau
                t = np.arange(ns, dtype=np.float64) / fs
                phase = 2.0 * K_PI * (beat * t) - 2.0 * K_PI * f_c * tau
                cube[rx, k, :] += target.amplitude * np.exp(1j * phase)

    if not scene.dechirped:
        t = np.arange(ns, dtype=np.float64) / fs
        ref = np.exp(1j * K_PI * slope * t * t)
        cube *= ref[np.newaxis, np.newaxis, :]

    if scene.snr_db is not None:
        # Unit-amplitude reference power; complex AWGN on real/imag.
        signal_power = 1.0
        noise_power = signal_power / (10.0 ** (scene.snr_db / 10.0))
        sigma = math.sqrt(noise_power / 2.0)
        noise = sigma * (
            np.random.randn(*cube.shape) + 1j * np.random.randn(*cube.shape)
        )
        cube = cube + noise

    if num_rx == 1:
        flat = cube[0]
        return SimResult(cube=flat.reshape(-1), shape=flat.shape)
    return SimResult(cube=cube.reshape(-1), shape=cube.shape)


def moving_target(waveform: SimWaveform, f: float) -> PointTarget:
    """Point target whose slow-time Doppler tone frequency is ``f`` cycles/CPI."""
    f_d = f / waveform.chirp_repetition_interval
    v_r = -f_d * waveform.wavelength() / 2.0
    return PointTarget(position=(45.0, 0.0, 0.0), velocity=(v_r, 0.0, 0.0))
