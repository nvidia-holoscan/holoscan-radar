# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared geometry / steering helpers for angle-estimation pytest ports."""

from __future__ import annotations

import math

import numpy as np

from holoscan.radar import (
    K_PI,
    AngleConvention,
    AngleConventionPreset,
    ArrayGeometry,
    ArrayGeometryType,
)


def linspace(start: float, stop: float, n: int) -> list[float]:
    if n == 1:
        return [start]
    step = (stop - start) / float(n - 1)
    return [start + step * float(i) for i in range(n)]


def make_ula_geometry(num_elements: int, spacing: float, wavelength: float) -> ArrayGeometry:
    geom = ArrayGeometry()
    geom.wavelength = wavelength
    geom.geometry_type = ArrayGeometryType.ULA
    positions: list[float] = []
    for m in range(num_elements):
        positions.extend([0.0, float(m) * spacing, 0.0])
    geom.element_positions = positions
    return geom


def make_planar_ura_geometry(ny: int, nz: int, spacing: float, wavelength: float) -> ArrayGeometry:
    geom = ArrayGeometry()
    geom.wavelength = wavelength
    geom.geometry_type = ArrayGeometryType.URA
    positions: list[float] = []
    for iz in range(nz):
        for iy in range(ny):
            positions.extend([0.0, float(iy) * spacing, float(iz) * spacing])
    geom.element_positions = positions
    return geom


def _from_canonical_angles(az: float, el: float) -> tuple[float, float, float]:
    cel = math.cos(el)
    return cel * math.cos(az), cel * math.sin(az), math.sin(el)


def _rotate_z(x: float, y: float, z: float, delta: float) -> tuple[float, float, float]:
    c = math.cos(delta)
    s = math.sin(delta)
    return c * x - s * y, s * x + c * y, z


def boundary_to_canonical(
    conv: AngleConvention, az: float, el: float
) -> tuple[float, float, float]:
    """Mirror of radar_common/dsp/angle_convention.cpp (test fixtures only)."""
    if conv.uses_direction_cosine():
        u, v = az, el
        if (u * u + v * v) > 1.0 + 1e-12:
            return 0.0, 0.0, 0.0
        w = math.sqrt(max(0.0, 1.0 - u * u - v * v))
        return w, u, v

    preset = conv.preset
    if preset in (
        AngleConventionPreset.RADAR_BORESIGHT,
        AngleConventionPreset.ROS_REP103,
        AngleConventionPreset.ISO8855_FLU,
    ):
        return _from_canonical_angles(az, el)
    if preset in (
        AngleConventionPreset.AEROSPACE_NED,
        AngleConventionPreset.GEODETIC_NED,
    ):
        return _from_canonical_angles(-az, -el)
    if preset == AngleConventionPreset.GEODETIC_ENU:
        return _rotate_z(*_from_canonical_angles(az, el), -K_PI / 2.0)
    if preset == AngleConventionPreset.PHYSICS_SPHERICAL:
        horizon_el = K_PI / 2.0 - el
        return _from_canonical_angles(az, horizon_el)
    raise ValueError(f"AngleConvention: unsupported preset {preset}")


def make_steering_snapshot(
    geom: ArrayGeometry,
    conv: AngleConvention,
    az: float,
    el: float,
) -> np.ndarray:
    dx, dy, dz = boundary_to_canonical(conv, az, el)
    k = 2.0 * K_PI / geom.wavelength
    n = geom.num_elements()
    snap = np.empty(n, dtype=np.complex128)
    pos = geom.element_positions
    for m in range(n):
        base = m * 3
        phase = k * (pos[base] * dx + pos[base + 1] * dy + pos[base + 2] * dz)
        snap[m] = complex(math.cos(phase), math.sin(phase))
    return snap


def radar_boresight_convention() -> AngleConvention:
    conv = AngleConvention()
    conv.preset = AngleConventionPreset.RADAR_BORESIGHT
    return conv


def physics_spherical_convention() -> AngleConvention:
    conv = AngleConvention()
    conv.preset = AngleConventionPreset.PHYSICS_SPHERICAL
    return conv


def aerospace_ned_convention() -> AngleConvention:
    conv = AngleConvention()
    conv.preset = AngleConventionPreset.AEROSPACE_NED
    return conv


def direction_cosine_convention() -> AngleConvention:
    conv = AngleConvention()
    conv.preset = AngleConventionPreset.DIRECTION_COSINE
    return conv
