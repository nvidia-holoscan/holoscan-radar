# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python port of ``tests/cpp/utils/bin_model.hpp``."""

from __future__ import annotations

import math

K_C = 299792458.0


def range_of_bin(
    bin_idx: int, slope: float, sample_rate: float, nfft: int, c: float = K_C
) -> float:
    f = float(bin_idx) * sample_rate / float(nfft)
    return c * f / (2.0 * slope)


def expected_range_bin(
    range_m: float, slope: float, sample_rate: float, nfft: int, c: float = K_C
) -> int:
    f_b = slope * 2.0 * range_m / c
    return int(round(f_b * float(nfft) / sample_rate))


def expected_range_axis(
    num_bins: int, slope: float, sample_rate: float, nfft: int, c: float = K_C
) -> list[float]:
    return [range_of_bin(i, slope, sample_rate, nfft, c) for i in range(num_bins)]


def doppler_frequency(radial_velocity: float, wavelength: float) -> float:
    return -2.0 * radial_velocity / wavelength


def unshifted_doppler_bin(f_d: float, pri: float, num_doppler: int) -> int:
    n = float(num_doppler)
    bin_val = round(f_d * pri * n)
    bin_val = math.fmod(bin_val, n)
    if bin_val < 0.0:
        bin_val += n
    return int(bin_val)


def fftshift_position(unshifted_bin: int, n: int) -> int:
    return (unshifted_bin + n // 2) % n


def expected_doppler_bin(
    radial_velocity: float,
    wavelength: float,
    pri: float,
    num_doppler: int,
    center_zero_doppler: bool,
) -> int:
    f_d = doppler_frequency(radial_velocity, wavelength)
    unshifted = unshifted_doppler_bin(f_d, pri, num_doppler)
    if center_zero_doppler:
        return fftshift_position(unshifted, num_doppler)
    return unshifted


def expected_velocity_axis(
    num_doppler: int, wavelength: float, pri: float, center_zero_doppler: bool
) -> list[float]:
    prf = 1.0 / pri
    axis: list[float] = []
    for i in range(num_doppler):
        idx = i
        if center_zero_doppler:
            idx = (i - num_doppler // 2 + num_doppler) % num_doppler
        f = float(idx) / float(num_doppler) * prf
        if f >= prf / 2.0:
            f -= prf
        axis.append(-f * wavelength / 2.0)
    return axis
