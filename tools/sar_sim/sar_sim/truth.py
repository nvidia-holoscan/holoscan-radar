# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ground-truth sidecar written beside every generated CPHD.

This is why the simulator exists. A real collection can be imaged but not
*checked*: there is no independent statement of where a scatterer was. Here
there is, so the sidecar records it in the form a test consumes — including
the image lattice the fixture expects to be formed on, so that the recipe,
the application YAML and the assertion cannot drift apart.

Recording the lattice here rather than duplicating it in the YAML is the
point: a suggested `SampleGeometry` computed from the same geometry that
produced the signal is a lattice the predicted pixel indices are actually
valid for.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .cphd import FOURIER_SIGN, image_area_axes, slant_range_swath
from .geometry import CollectionGeometry
from .recipe import Recipe
from .signal import frequency_axis


def suggested_sample_geometry(recipe: Recipe, geometry: CollectionGeometry) -> dict[str, Any]:
    """A `SampleGeometry` descriptor covering the scene, IARP-centred.

    Rows (`v`) run in ground range and columns (`u`) in azimuth, matching
    SICD's Row/Col convention. Both counts are forced odd so the IARP sits
    exactly on the centre pixel rather than between four of them, which is
    what lets a test name an expected index instead of a neighbourhood.
    """
    iax, iay = image_area_axes(geometry)
    graze = math.radians(recipe.geometry.graze_deg)

    # The lattice is on the ground plane, so the slant-range resolution
    # projects by 1/cos(graze). Ignoring that is the classic way to get an
    # image that is correct but the wrong shape.
    ground_range_spacing = recipe.waveform.range_resolution / (
        math.cos(graze) * recipe.image_oversample
    )
    azimuth_spacing = recipe.azimuth_resolution / recipe.image_oversample

    def odd_count(spacing: float) -> int:
        half = int(math.floor(recipe.image_half_extent_m / spacing))
        return 2 * max(half, 1) + 1

    num_v = odd_count(ground_range_spacing)
    num_u = odd_count(azimuth_spacing)
    origin = (
        geometry.iarp_ecef
        - (num_v // 2) * ground_range_spacing * iax
        - (num_u // 2) * azimuth_spacing * iay
    )
    return {
        "origin": [float(value) for value in origin],
        "axis_v": [float(value) for value in iax],
        "axis_u": [float(value) for value in iay],
        "spacing": [float(ground_range_spacing), float(azimuth_spacing)],
        "counts": [int(num_v), int(num_u)],
    }


def target_truth(
    recipe: Recipe, geometry: CollectionGeometry, lattice: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-target positions and the pixel each should focus to."""
    iax = np.array(lattice["axis_v"])
    iay = np.array(lattice["axis_u"])
    origin = np.array(lattice["origin"])
    spacing_v, spacing_u = lattice["spacing"]

    records = []
    for target in recipe.targets:
        enu = np.array([target.east_m, target.north_m, target.up_m])
        ecef = geometry.enu_to_ecef(enu)[0]
        offset = ecef - origin
        index_v = float(np.dot(offset, iax) / spacing_v)
        index_u = float(np.dot(offset, iay) / spacing_u)
        records.append(
            {
                "name": target.name,
                "enu_m": [target.east_m, target.north_m, target.up_m],
                "ecef_m": [float(value) for value in ecef],
                "amplitude": target.amplitude,
                "phase_deg": target.phase_deg,
                # Fractional index is the honest prediction; the rounded pair
                # is what a peak-location assertion compares against. A
                # target placed off a pixel centre has no integer answer, and
                # rounding here rather than in the test keeps every consumer
                # rounding the same way.
                "image_index_vu": [index_v, index_u],
                "image_index_vu_rounded": [int(round(index_v)), int(round(index_u))],
            }
        )
    return records


def build_sidecar(
    recipe: Recipe,
    geometry: CollectionGeometry,
    *,
    cphd_path: Path,
    verification: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assemble the full sidecar document."""
    lattice = suggested_sample_geometry(recipe, geometry)
    freqs = frequency_axis(recipe)
    return {
        "schema": "holoscan-radar/sar-sim-truth/1",
        "recipe_name": recipe.name,
        "description": recipe.description,
        "cphd_file": cphd_path.name,
        "collection": {
            "num_pulses": geometry.num_pulses,
            "num_samples": recipe.collection.num_samples,
            "signal_array_format": "CF8",
            "domain_type": "FX",
            "fourier_sign": FOURIER_SIGN,
            "fx_min_hz": float(freqs[0]),
            "fx_max_hz": float(freqs[-1]),
            "fx_spacing_hz": float(freqs[1] - freqs[0]),
        },
        "geometry": {
            "iarp_llh": list(geometry.iarp_llh),
            "iarp_ecef": [float(value) for value in geometry.iarp_ecef],
            "graze_deg": recipe.geometry.graze_deg,
            "heading_deg": recipe.geometry.heading_deg,
            "slant_range_m": recipe.geometry.slant_range_m,
            "velocity_m_s": recipe.geometry.velocity_m_s,
            "aperture_length_m": float(geometry.aperture_length),
            "integration_angle_deg": math.degrees(
                geometry.aperture_length / recipe.geometry.slant_range_m
            ),
        },
        "resolution": {
            "slant_range_m": recipe.waveform.range_resolution,
            "azimuth_m": recipe.azimuth_resolution,
            "wavelength_m": recipe.waveform.wavelength,
            "unambiguous_slant_range_swath_m": slant_range_swath(recipe),
        },
        "sample_geometry": lattice,
        "targets": target_truth(recipe, geometry, lattice),
        "self_check": verification or {},
    }


def write_sidecar(path: str | Path, sidecar: dict[str, Any]) -> Path:
    """Write the sidecar as indented JSON, newline-terminated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return path
