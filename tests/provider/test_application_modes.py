# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application mode manifests and inherited algorithm presets."""

import json
import shlex
from importlib import import_module
from pathlib import Path

import pytest

import holoscan as hs

_ROOT = Path(__file__).resolve().parents[2] / "applications" / "fmcw"
_APPS = (
    "siso",
    "simo",
    "dense_beamforming",
    "mimo_tdm",
    "mimo_ddm",
    "adaptive_aoa",
    "multi_waveform",
)


@pytest.mark.parametrize("name", _APPS)
def test_mode_manifests_match_languages_and_presets_exist(name):
    manifests = [
        json.loads((_ROOT / name / language / "metadata.json").read_text())["application"]
        for language in ("cpp", "python")
    ]
    expected = {"default", "smoke"}
    if name == "mimo_ddm":
        expected |= {"nonlinear_ddm", "nonlinear_ddm_smoke"}
    if name == "adaptive_aoa":
        expected |= {"root_music", "esprit", "mvdr"}
    for manifest in manifests:
        assert manifest["default_mode"] == "default"
        assert set(manifest["modes"]) == expected
    for mode in expected - {"default"}:
        commands = [
            shlex.split(manifest["modes"][mode]["run"]["command"]) for manifest in manifests
        ]
        assert commands[0][-1] == commands[1][-1]
        assert (_ROOT / name / "config" / commands[0][-1]).is_file()
        python_mode = manifests[1]["modes"][mode]
        default = manifests[1]["modes"]["default"]
        assert python_mode["build"] == default["build"]
        assert python_mode["run"]["env"] == default["run"]["env"]


@pytest.mark.parametrize("algorithm", ("root_music", "esprit", "mvdr"))
def test_inherited_aoa_algorithm_presets_compile(algorithm):
    module = import_module("holoscan_radar.applications.adaptive_aoa")
    preset = _ROOT / "adaptive_aoa" / "config" / f"adaptive_aoa_{algorithm}.yaml"
    workflow = module.build_graph(preset)
    assert hs.compile(workflow.graph, workflow.options).ok
