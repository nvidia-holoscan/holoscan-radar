# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# App-owned config for fmcw_siso: the AppConfig type and its YAML loader.
# Colocated with the app; the graph wiring lives in fmcw_siso.py. Operator
# blocks are NOT parsed here — they flow through Holoscan's from_config().
#
# Imported bare (``import config``) from fmcw_siso.py: each app is invoked as
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
    parse_common_run_config,
    parse_cube_dims,
    parse_noise,
    parse_targets_range1d,
    parse_waveform,
)
from common.python.sim import SimScene, SimWaveform


@dataclass
class AppConfig(CommonRunConfig):
    scene: SimScene = field(default_factory=SimScene)


def load_config(path: Path) -> AppConfig:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    root = yaml.safe_load(path.read_text())

    waveform = parse_waveform(root["waveform"])
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
        targets=parse_targets_range1d(root),
    )
    parse_noise(root, scene)

    common = parse_common_run_config(root, config_path=path.resolve())
    cfg = AppConfig(**vars(common), scene=scene)
    # READ-ONLY MIRROR of the cfar_detector operator block: peak_only stays a
    # native CFAR param (from_config); the app reads it only to gate the graph.
    cfg.peak_only = bool((root.get("cfar_detector") or {}).get("peak_only", False))
    return cfg
