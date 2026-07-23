# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# App-owned config for fmcw_aoa_estimators: the AppConfig type and its YAML
# loader (a filled y-z URA shared by all five estimators plus the
# estimator-router control policy). Colocated with the app; the graph wiring
# lives in
# fmcw_aoa_estimators.py. Operator blocks are NOT parsed here — they flow
# through Holoscan's from_config().
#
# Imported bare (``import config``) from fmcw_aoa_estimators.py: each app is
# invoked as a script, which puts this file's own directory at sys.path[0] -
# the Python analogue of C++'s colocated ``#include "config.hpp"``. Do not
# import this module by its bare name from another process.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from common.python.demo_config import (
    CommonRunConfig,
    grid_extent,
    make_ura_geometry,
    parse_common_run_config,
    parse_cube_dims,
    parse_noise,
    parse_target,
    parse_waveform,
)
from common.python.sim import PointTarget, SimScene, SimWaveform


@dataclass
class RouterConfig:
    mode: str = "rotate"
    rotate_period: int = 40
    hold: str = "beamscan"
    control_path: str = ""
    log_every: int = 0


@dataclass
class AppConfig(CommonRunConfig):
    scene: SimScene = field(default_factory=SimScene)
    array_geometry: object = None
    router: RouterConfig = field(default_factory=RouterConfig)
    # Boresight-view axis extents only (read-only mirror of the grid estimator's
    # az_angles/el_angles; the operators still get them via from_config()).
    az_lo: float = -0.35
    az_hi: float = 0.35
    el_lo: float = -0.35
    el_hi: float = 0.35


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    root = yaml.safe_load(path.read_text())

    waveform = parse_waveform(root["waveform"])
    wavelength = waveform.wavelength()
    num_chirps, samples_per_chirp = parse_cube_dims(root)
    scene = SimScene(
        waveform=SimWaveform(
            carrier_frequency=waveform.carrier_frequency(),
            bandwidth=waveform.bandwidth(),
            chirp_duration=waveform.chirp_duration(),
            sample_rate=waveform.sample_rate(),
            chirp_repetition_interval=waveform.chirp_repetition_interval(),
        ),
        num_chirps=num_chirps,
        samples_per_chirp=samples_per_chirp,
    )

    rx = root.get("rx_array")
    if not rx:
        raise ValueError("rx_array section is required")
    az_elements = int(rx["az_elements"])
    el_elements = int(rx["el_elements"])
    spacing = float(rx["spacing"])
    if az_elements < 2 or el_elements < 2:
        raise ValueError(
            "rx_array.az_elements and rx_array.el_elements must both be >= 2 for "
            "a filled URA the gridless estimators resolve in 2-D"
        )
    array_geometry = make_ura_geometry(az_elements, el_elements, spacing, wavelength)
    # (iz outer, iy inner) channel order, matching make_ura_geometry.
    scene.rx_positions = [
        (0.0, float(iy) * spacing * wavelength, float(iz) * spacing * wavelength)
        for iz in range(el_elements)
        for iy in range(az_elements)
    ]

    for t in root.get("targets", []):
        position, velocity, amplitude = parse_target(t, app_name="fmcw_aoa_estimators")
        scene.targets.append(PointTarget(position=position, velocity=velocity, amplitude=amplitude))

    parse_noise(root, scene)

    er = root.get("estimator_router", {}) or {}
    router = RouterConfig(
        mode=str(er.get("mode", "rotate")),
        rotate_period=int(er.get("rotate_period", 40)),
        hold=str(er.get("hold", "beamscan")),
        control_path=str(er.get("control_path", "")),
        log_every=int(er.get("log_every", 0)),
    )
    if router.rotate_period <= 0:
        raise ValueError("estimator_router.rotate_period must be > 0")
    if router.log_every < 0:
        raise ValueError("estimator_router.log_every must be >= 0")

    # Read-only mirror: az_angles/el_angles are the grid estimator's own
    # from_config() parameters; this only sizes the boresight-view axes.
    be = root.get("beamscan_estimator", {})
    az_lo, az_hi = grid_extent([float(x) for x in be.get("az_angles", [])] or None, -0.35, 0.35)
    el_lo, el_hi = grid_extent([float(x) for x in be.get("el_angles", [])] or None, -0.35, 0.35)

    common = parse_common_run_config(root, config_path=path.resolve())
    return AppConfig(
        **vars(common),
        scene=scene,
        array_geometry=array_geometry,
        router=router,
        az_lo=az_lo,
        az_hi=az_hi,
        el_lo=el_lo,
        el_hi=el_hi,
    )
