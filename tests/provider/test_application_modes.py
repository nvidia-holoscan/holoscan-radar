# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application mode manifests and inherited algorithm presets."""

import json
import shlex
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
from holoscan_radar import _radar

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
    for mode in expected:
        assert (
            manifests[0]["modes"][mode]["description"] == manifests[1]["modes"][mode]["description"]
        )
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
def test_inherited_aoa_algorithm_presets_compile(algorithm, monkeypatch):
    # Compile-only coverage: a real window is only needed when running the graph.
    monkeypatch.setattr(_radar, "_resolve_x11_presentation", lambda *_: True)
    select = _radar._select_application_device
    monkeypatch.setattr(_radar, "_select_application_device", lambda *_: select(False, False, True))
    monkeypatch.setattr(_radar, "_application_display_device", lambda: select(False, False, True))
    monkeypatch.setattr(_radar, "_application_composer_device", lambda: select(False, False, True))
    module = import_module("holoscan_radar.applications.adaptive_aoa")
    preset = _ROOT / "adaptive_aoa" / "config" / f"adaptive_aoa_{algorithm}.yaml"
    workflow = module.build_graph(preset)
    assert hs.compile(workflow.graph, workflow.options).ok


@pytest.mark.parametrize("choose", (False, True))
def test_cli_forwards_choose_presenter(choose, monkeypatch, tmp_path):
    from holoscan_radar.applications import cli

    path = tmp_path / "app.yaml"
    path.write_text("app: {}\n")
    calls = []

    def build(config, *, choose_presenter, presenter):
        calls.append((config, choose_presenter, presenter))
        return SimpleNamespace(run=lambda: calls.append("run"))

    monkeypatch.setattr(cli, "import_module", lambda _: SimpleNamespace(build_graph=build))
    args = ["siso", str(path)] + (["--choose-presenter"] if choose else [])
    assert cli.main(args) == 0
    assert calls == [(path, choose, None), "run"]
