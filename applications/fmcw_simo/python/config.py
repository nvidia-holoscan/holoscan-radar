# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# App-owned config for fmcw_simo: the AppConfig type and its YAML loader.
# Colocated with the app; the graph wiring lives in fmcw_simo.py. Operator
# blocks are NOT parsed here — they flow through Holoscan's from_config().
#
# Imported bare (``import config``) from fmcw_simo.py: each app is invoked as
# a script, which puts this file's own directory at sys.path[0] - the Python
# analogue of C++'s colocated ``#include "config.hpp"  // NOLINT(...)``. Do
# not import this module by its bare name from another process: `config` is
# not a qualified package name, so importing two apps' config.py in the same
# interpreter would collide in sys.modules.

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
class AppConfig(CommonRunConfig):
    scene: SimScene = field(default_factory=SimScene)
    array_geometry: object = None
    # Boresight-view axis extents only (BeamscanEstimatorOp's az_angles /
    # el_angles are read-only mirrored here to size the plot, not to
    # configure the operator - that flows through from_config()).
    az_lo: float = -0.35
    az_hi: float = 0.35
    el_lo: float = -0.1
    el_hi: float = 0.1


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
        raise ValueError("rx_array section is required for SIMO")
    az_elements = int(rx["az_elements"])
    el_elements = int(rx["el_elements"])
    spacing = float(rx["spacing"])
    if az_elements < 2 or el_elements < 2:
        raise ValueError("SIMO URA requires rx_array.az_elements >= 2 and el_elements >= 2")
    array_geometry = make_ura_geometry(az_elements, el_elements, spacing, wavelength)
    scene.rx_positions = []
    for iz in range(el_elements):
        for iy in range(az_elements):
            scene.rx_positions.append(
                (
                    0.0,
                    float(iy) * spacing * wavelength,
                    float(iz) * spacing * wavelength,
                )
            )

    for t in root.get("targets", []):
        position, velocity, amplitude = parse_target(t, app_name="fmcw_simo")
        scene.targets.append(PointTarget(position=position, velocity=velocity, amplitude=amplitude))

    parse_noise(root, scene)

    # Read-only mirror: az_angles/el_angles are BeamscanEstimatorOp's own
    # from_config() parameters; this only sizes the boresight-view axes.
    be = root.get("beamscan_estimator", {})
    az_lo, az_hi = grid_extent([float(x) for x in be.get("az_angles", [])] or None, -0.35, 0.35)
    el_lo, el_hi = grid_extent([float(x) for x in be.get("el_angles", [])] or None, -0.1, 0.1)

    common = parse_common_run_config(root, config_path=path.resolve())
    cfg = AppConfig(
        **vars(common),
        scene=scene,
        array_geometry=array_geometry,
        az_lo=az_lo,
        az_hi=az_hi,
        el_lo=el_lo,
        el_hi=el_hi,
    )
    # READ-ONLY MIRROR of the cfar_detector operator block: peak_only stays a
    # native CFAR param (from_config); the app reads it only to gate the graph.
    cfg.peak_only = bool(root.get("cfar_detector", {}).get("peak_only", False))
    return cfg
