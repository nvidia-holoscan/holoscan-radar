# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WGS84 conversions and the collection geometry a recipe describes.

Everything here is ECEF-facing, because that is the frame CPHD PVP positions
are written in. Scene content is authored in a local ENU frame at the image
area reference point (IARP) — metres east/north/up are the only units a
recipe author should have to think in — and converted here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# WGS84 defining parameters.
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

SPEED_OF_LIGHT = 299792458.0


def llh_to_ecef(lat_deg: float, lon_deg: float, hae_m: float) -> np.ndarray:
    """Geodetic latitude/longitude/height above ellipsoid to ECEF metres."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + hae_m) * math.cos(lat) * math.cos(lon),
            (n + hae_m) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - WGS84_E2) + hae_m) * sin_lat,
        ]
    )


def ecef_to_llh(ecef: np.ndarray) -> tuple[float, float, float]:
    """ECEF metres to geodetic (latitude deg, longitude deg, height m).

    Ferrari's closed-form solution, which is exact rather than iterative and
    so gives the same answer every run — the sidecar's corner points are
    compared between platforms.
    """
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    ep2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    lat = math.atan2(
        z + ep2 * WGS84_B * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat) ** 2)
    hae = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-12 else abs(z) - WGS84_B
    return math.degrees(lat), math.degrees(lon), hae


def enu_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit east, north and up vectors at a geodetic point, in ECEF."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    north = np.array(
        [
            -math.sin(lat) * math.cos(lon),
            -math.sin(lat) * math.sin(lon),
            math.cos(lat),
        ]
    )
    up = np.array(
        [
            math.cos(lat) * math.cos(lon),
            math.cos(lat) * math.sin(lon),
            math.sin(lat),
        ]
    )
    return east, north, up


@dataclass(frozen=True)
class CollectionGeometry:
    """The platform track and scene frame a recipe resolves to.

    Broadside, monostatic, straight and level: the platform flies a constant
    velocity line and the antenna stares at the IARP, which is what makes the
    SRP fixed for the whole collection. That is the simplest geometry that is
    still a real spotlight collection, and every quantity a test wants to
    predict analytically stays in closed form.
    """

    iarp_ecef: np.ndarray
    iarp_llh: tuple[float, float, float]
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray
    # Horizontal unit vector from the IARP toward the platform's ground
    # track, and the along-track unit vector. Together with `up` these are
    # the scene's natural axes: ground range and azimuth.
    ground_range_axis: np.ndarray
    track_axis: np.ndarray
    apc_ecef: np.ndarray  # (num_pulses, 3)
    apc_velocity: np.ndarray  # (num_pulses, 3)
    tx_time: np.ndarray  # (num_pulses,) seconds from collection reference
    slant_range_at_closest_approach: float
    aperture_length: float

    @property
    def num_pulses(self) -> int:
        return int(self.apc_ecef.shape[0])

    def enu_to_ecef(self, enu: np.ndarray) -> np.ndarray:
        """Scene-frame (east, north, up) metres to ECEF, one point or many."""
        enu = np.atleast_2d(np.asarray(enu, dtype=float))
        basis = np.stack([self.east, self.north, self.up])
        return self.iarp_ecef + enu @ basis


def build_geometry(
    *,
    iarp_lat_deg: float,
    iarp_lon_deg: float,
    iarp_hae_m: float,
    slant_range_m: float,
    graze_deg: float,
    heading_deg: float,
    velocity_m_s: float,
    num_pulses: int,
    pri_s: float,
) -> CollectionGeometry:
    """Resolve a recipe's geometry block into an explicit platform track.

    `heading_deg` is the ground-track direction clockwise from north, and the
    platform looks to its right (`SideOfTrack = R`), which is the convention
    both pinned public collections use. Closest approach is the aperture
    centre, so the collection is broadside with zero squint.
    """
    if num_pulses < 2:
        raise ValueError("num_pulses must be at least 2")
    if not slant_range_m > 0.0:
        raise ValueError("slant_range_m must be positive")
    if not 0.0 < graze_deg < 90.0:
        raise ValueError("graze_deg must be within (0, 90)")

    iarp_ecef = llh_to_ecef(iarp_lat_deg, iarp_lon_deg, iarp_hae_m)
    east, north, up = enu_basis(iarp_lat_deg, iarp_lon_deg)

    heading = math.radians(heading_deg)
    track_axis = math.sin(heading) * east + math.cos(heading) * north
    # Right-looking: the platform sits 90 degrees clockwise from the heading,
    # so the IARP-to-track direction is the heading rotated by +90.
    ground_range_axis = math.cos(heading) * east - math.sin(heading) * north

    graze = math.radians(graze_deg)
    closest_approach = iarp_ecef + slant_range_m * (
        math.cos(graze) * ground_range_axis + math.sin(graze) * up
    )

    # Pulse times start at zero because CPHD requires non-negative TxTime,
    # and the aperture is centred on closest approach so the collection is
    # broadside. Those are two different origins, so positions are placed
    # against time *relative to closest approach* and only the published
    # timestamps are shifted.
    tx_time = np.arange(num_pulses, dtype=float) * pri_s
    time_from_closest_approach = tx_time - 0.5 * (num_pulses - 1) * pri_s
    apc_ecef = closest_approach + np.outer(time_from_closest_approach * velocity_m_s, track_axis)
    apc_velocity = np.tile(velocity_m_s * track_axis, (num_pulses, 1))

    return CollectionGeometry(
        iarp_ecef=iarp_ecef,
        iarp_llh=(iarp_lat_deg, iarp_lon_deg, iarp_hae_m),
        east=east,
        north=north,
        up=up,
        ground_range_axis=ground_range_axis,
        track_axis=track_axis,
        apc_ecef=apc_ecef,
        apc_velocity=apc_velocity,
        tx_time=tx_time,
        slant_range_at_closest_approach=slant_range_m,
        aperture_length=velocity_m_s * pri_s * (num_pulses - 1),
    )
