# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe model: the committed description a CPHD fixture is generated from.

No CPHD binaries live in git, so the recipe *is* the fixture. It must
therefore be complete enough to regenerate the same collection, and readable
enough that a reviewer can tell what scene it describes without running it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .geometry import SPEED_OF_LIGHT


@dataclass(frozen=True)
class Waveform:
    """The transmitted linear-FM chirp.

    Mirrors the `LFMWaveformDescriptor` fields so a recipe and the operator
    parameters it feeds are read in the same vocabulary.
    """

    carrier_frequency: float
    bandwidth: float
    chirp_duration: float
    sample_rate: float
    chirp_repetition_interval: float

    @property
    def wavelength(self) -> float:
        return SPEED_OF_LIGHT / self.carrier_frequency

    @property
    def range_resolution(self) -> float:
        """Slant-range resolution, c / (2 B), before any taper broadening."""
        return SPEED_OF_LIGHT / (2.0 * self.bandwidth)

    def validate(self) -> None:
        for name in (
            "carrier_frequency",
            "bandwidth",
            "chirp_duration",
            "sample_rate",
            "chirp_repetition_interval",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"waveform.{name} must be finite and > 0")


@dataclass(frozen=True)
class Geometry:
    """Where the platform flies and what it stares at."""

    iarp_lat_deg: float
    iarp_lon_deg: float
    iarp_hae_m: float
    slant_range_m: float
    graze_deg: float
    heading_deg: float
    velocity_m_s: float

    def validate(self) -> None:
        if not -90.0 <= self.iarp_lat_deg <= 90.0:
            raise ValueError("geometry.iarp_lat_deg must be within [-90, 90]")
        if not -180.0 <= self.iarp_lon_deg <= 180.0:
            raise ValueError("geometry.iarp_lon_deg must be within [-180, 180]")
        if not self.slant_range_m > 0.0:
            raise ValueError("geometry.slant_range_m must be > 0")
        if not 0.0 < self.graze_deg < 90.0:
            raise ValueError("geometry.graze_deg must be within (0, 90)")
        if not self.velocity_m_s > 0.0:
            raise ValueError("geometry.velocity_m_s must be > 0")


@dataclass(frozen=True)
class PointTarget:
    """One ideal point scatterer, placed in the scene's local ENU frame.

    `amplitude` and `phase_deg` are the complex reflectivity. A test asserting
    on relative brightness needs amplitudes it chose; a test asserting on
    phase (autofocus, coherence) needs a non-zero phase somewhere in the
    scene, so both are authored rather than fixed at 1 and 0.
    """

    east_m: float
    north_m: float
    up_m: float = 0.0
    amplitude: float = 1.0
    phase_deg: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class Collection:
    """Sampling extent: how many pulses and how many frequency samples."""

    num_pulses: int
    num_samples: int

    def validate(self) -> None:
        if self.num_pulses < 2:
            raise ValueError("collection.num_pulses must be at least 2")
        if self.num_samples < 2:
            raise ValueError("collection.num_samples must be at least 2")


@dataclass(frozen=True)
class Recipe:
    """A complete, regenerable description of one synthetic collection."""

    name: str
    description: str
    waveform: Waveform
    geometry: Geometry
    collection: Collection
    targets: list[PointTarget] = field(default_factory=list)
    # Oversampling of the formed image relative to the intrinsic resolution.
    # Drives only the suggested SampleGeometry in the sidecar, never the
    # signal: a metric assertion wants a few pixels across the mainlobe.
    image_oversample: float = 2.0
    # Half-extent of the suggested image lattice, in metres from the IARP.
    image_half_extent_m: float = 50.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("recipe.name is required")
        self.waveform.validate()
        self.geometry.validate()
        self.collection.validate()
        if not self.targets:
            raise ValueError("recipe.targets must name at least one point target")
        if not self.image_oversample > 0.0:
            raise ValueError("recipe.image_oversample must be > 0")
        if not self.image_half_extent_m > 0.0:
            raise ValueError("recipe.image_half_extent_m must be > 0")

    @property
    def azimuth_resolution(self) -> float:
        """Broadside azimuth resolution, lambda / (2 * integration angle).

        Small-angle: the integration angle is the aperture length over the
        slant range at closest approach.
        """
        aperture = (
            self.geometry.velocity_m_s
            * self.waveform.chirp_repetition_interval
            * (self.collection.num_pulses - 1)
        )
        integration_angle = aperture / self.geometry.slant_range_m
        return self.waveform.wavelength / (2.0 * integration_angle)


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{section}: '{key}' is required")
    return mapping[key]


def load_recipe(path: str | Path) -> Recipe:
    """Load and validate a recipe YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: recipe must be a YAML mapping")
    return parse_recipe(raw, source=str(path))


def parse_recipe(raw: dict[str, Any], *, source: str = "<recipe>") -> Recipe:
    """Build a Recipe from an already-loaded mapping."""
    waveform_raw = _require(raw, "waveform", source)
    geometry_raw = _require(raw, "geometry", source)
    collection_raw = _require(raw, "collection", source)
    targets_raw = _require(raw, "targets", source)

    recipe = Recipe(
        name=str(_require(raw, "name", source)),
        description=str(raw.get("description", "")),
        waveform=Waveform(
            carrier_frequency=float(_require(waveform_raw, "carrier_frequency", "waveform")),
            bandwidth=float(_require(waveform_raw, "bandwidth", "waveform")),
            chirp_duration=float(_require(waveform_raw, "chirp_duration", "waveform")),
            sample_rate=float(_require(waveform_raw, "sample_rate", "waveform")),
            chirp_repetition_interval=float(
                _require(waveform_raw, "chirp_repetition_interval", "waveform")
            ),
        ),
        geometry=Geometry(
            iarp_lat_deg=float(_require(geometry_raw, "iarp_lat_deg", "geometry")),
            iarp_lon_deg=float(_require(geometry_raw, "iarp_lon_deg", "geometry")),
            iarp_hae_m=float(geometry_raw.get("iarp_hae_m", 0.0)),
            slant_range_m=float(_require(geometry_raw, "slant_range_m", "geometry")),
            graze_deg=float(_require(geometry_raw, "graze_deg", "geometry")),
            heading_deg=float(geometry_raw.get("heading_deg", 0.0)),
            velocity_m_s=float(_require(geometry_raw, "velocity_m_s", "geometry")),
        ),
        collection=Collection(
            num_pulses=int(_require(collection_raw, "num_pulses", "collection")),
            num_samples=int(_require(collection_raw, "num_samples", "collection")),
        ),
        targets=[
            PointTarget(
                east_m=float(_require(target, "east_m", "targets")),
                north_m=float(_require(target, "north_m", "targets")),
                up_m=float(target.get("up_m", 0.0)),
                amplitude=float(target.get("amplitude", 1.0)),
                phase_deg=float(target.get("phase_deg", 0.0)),
                name=str(target.get("name", f"target_{index}")),
            )
            for index, target in enumerate(targets_raw)
        ],
        image_oversample=float(raw.get("image_oversample", 2.0)),
        image_half_extent_m=float(raw.get("image_half_extent_m", 50.0)),
    )
    recipe.validate()
    return recipe
