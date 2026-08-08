# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe to CPHD-plus-sidecar, in one call."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cphd import write_cphd
from .geometry import build_geometry
from .recipe import Recipe, load_recipe
from .signal import synthesize, verify_range_compression
from .truth import build_sidecar, write_sidecar


@dataclass(frozen=True)
class GenerateResult:
    recipe: Recipe
    cphd_path: Path
    sidecar_path: Path
    self_check: dict[str, float]


def generate(
    recipe: Recipe | str | Path,
    output_dir: str | Path,
    *,
    self_check: bool = True,
) -> GenerateResult:
    """Generate a CPHD fixture and its ground-truth sidecar.

    `self_check` compresses one pulse and confirms the targets land where the
    geometry says they should. It is on by default and costs milliseconds:
    the failure it catches — a phase-sign or frequency-axis convention error —
    produces a file that is perfectly well-formed and quietly wrong, which is
    the most expensive kind of fixture to debug downstream.
    """
    if not isinstance(recipe, Recipe):
        recipe = load_recipe(recipe)

    geometry = build_geometry(
        iarp_lat_deg=recipe.geometry.iarp_lat_deg,
        iarp_lon_deg=recipe.geometry.iarp_lon_deg,
        iarp_hae_m=recipe.geometry.iarp_hae_m,
        slant_range_m=recipe.geometry.slant_range_m,
        graze_deg=recipe.geometry.graze_deg,
        heading_deg=recipe.geometry.heading_deg,
        velocity_m_s=recipe.geometry.velocity_m_s,
        num_pulses=recipe.collection.num_pulses,
        pri_s=recipe.waveform.chirp_repetition_interval,
    )

    signal = synthesize(recipe, geometry)

    checks: dict[str, float] = {}
    if self_check:
        checks = verify_range_compression(recipe, geometry, signal)
        if checks["worst_range_error_m"] > checks["range_resolution_m"]:
            raise RuntimeError(
                "simulator self-check failed: a target compressed "
                f"{checks['worst_range_error_m']:.3f} m from its predicted "
                f"differential range, more than the {checks['range_resolution_m']:.3f} m "
                "range resolution. The phase model and Global/SGN disagree."
            )

    output_dir = Path(output_dir)
    cphd_path = write_cphd(output_dir / f"{recipe.name}.cphd", recipe, geometry, signal)
    sidecar = build_sidecar(recipe, geometry, cphd_path=cphd_path, verification=checks)
    sidecar_path = write_sidecar(output_dir / f"{recipe.name}_truth.json", sidecar)

    return GenerateResult(
        recipe=recipe,
        cphd_path=cphd_path,
        sidecar_path=sidecar_path,
        self_check=checks,
    )
