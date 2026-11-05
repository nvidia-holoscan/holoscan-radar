# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke checks for the seven Python FMCW Graph entry points."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from holoscan_radar import _radar
from holoscan_radar.applications import (
    adaptive_aoa,
    dense_beamforming,
    mimo_ddm,
    mimo_tdm,
    multi_waveform,
    simo,
    siso,
)

import holoscan as hs

_CASES = [
    (siso, "siso", "siso"),
    (simo, "simo", "simo"),
    (dense_beamforming, "dense_beamforming", "dense_beamforming"),
    (mimo_tdm, "mimo_tdm", "mimo_tdm"),
    (mimo_ddm, "mimo_ddm", "mimo_ddm_linear"),
    (mimo_ddm, "mimo_ddm", "mimo_ddm_nonlinear"),
    (adaptive_aoa, "adaptive_aoa", "adaptive_aoa"),
    (multi_waveform, "multi_waveform", "multi_waveform"),
]
_ROOT = Path(__file__).resolve().parents[2] / "applications" / "fmcw"


@pytest.fixture(autouse=True)
def isolate_presenter_choice(monkeypatch):
    monkeypatch.setattr(
        _radar,
        "_resolve_x11_presentation",
        lambda window, configured, choose_again=False, config_path="": (
            True if configured is None else configured
        ),
    )


@pytest.mark.parametrize(
    "preset,num_angles",
    [("dense_beamforming", 13), ("dense_beamforming_viz_smoke", 5)],
)
def test_dense_frontend_and_cfar_extents_are_independent(preset, num_angles):
    path = _ROOT / "dense_beamforming" / "config" / f"{preset}.yaml"
    cfg = _radar._load_dense(str(path))
    frontend = cfg["extents"]
    cfar = cfg["cfar_extents"]
    assert frontend is not cfar
    # The ADC has 8x8 receivers, independent of the az/el scan dimensions.
    assert frontend["max_rx"] == 64
    assert frontend["max_channels"] == num_angles * num_angles
    assert "max_tx" not in frontend
    assert cfar["max_tx"] == num_angles
    assert cfar["max_rx"] == num_angles
    assert "max_channels" not in cfar
    cfar["max_rx"] = 1
    assert frontend["max_rx"] == 64


@pytest.mark.parametrize("module,directory,preset", _CASES, ids=[item[2] for item in _CASES])
def test_smoke_preset_reaches_all_report_frames(module, directory, preset, capfd):
    path = _ROOT / directory / "config" / f"{preset}_smoke.yaml"
    workflow = module.build_graph(path)
    assert workflow.num_frames == 2
    session = workflow.run()
    assert not session.diagnostics
    if preset == "siso":
        reports = re.findall(
            r"centroid range ([0-9.]+) m, velocity (-?[0-9.]+) m/s",
            capfd.readouterr().err,
        )
        assert len(reports) == 2
        assert all(abs(float(distance) - 245.0) < 2.0 for distance, _ in reports)
        assert all(abs(float(speed) + 6.5006) < 1.0 for _, speed in reports)


@pytest.mark.parametrize("module,directory,preset", _CASES, ids=[item[2] for item in _CASES])
def test_visualization_preset_compiles(module, directory, preset, monkeypatch):
    # This test checks Graph admission without creating a native display surface.
    select = _radar._select_application_device
    monkeypatch.setattr(_radar, "_select_application_device", lambda *_: select(False, False, True))
    monkeypatch.setattr(_radar, "_application_display_device", lambda: select(False, False, True))
    monkeypatch.setattr(_radar, "_application_composer_device", lambda: select(False, False, True))
    path = _ROOT / directory / "config" / f"{preset}.yaml"
    workflow = module.build_graph(path)
    assert hs.compile(workflow.graph, workflow.options).ok


@pytest.mark.parametrize("module,directory,preset", _CASES, ids=[item[2] for item in _CASES])
@pytest.mark.parametrize("visualize", (False, True))
def test_all_tensor_placements_follow_selected_nonzero_device(
    module, directory, preset, visualize, monkeypatch
):
    # Exercise real Graph composition with a renderer whose CUDA ordinal is 3.
    # Record bindings rather than compiling against the test host's GPU count.
    placements = []
    queries = []

    def select_device(window, cuda_sink, x11_presentation):
        queries.append((window, cuda_sink, x11_presentation))
        return 3

    def bind_output(**placement):
        placements.append(placement)

    options = SimpleNamespace(deployment=SimpleNamespace(bind_tensor_output_device=bind_output))
    monkeypatch.setattr(_radar, "_select_application_device", select_device)
    monkeypatch.setattr(_radar, "_application_display_device", lambda: 3)
    monkeypatch.setattr(_radar, "_application_composer_device", lambda: 3)
    monkeypatch.setattr(hs, "CompileOptions", lambda: options)
    suffix = "" if visualize else "_smoke"
    path = _ROOT / directory / "config" / f"{preset}{suffix}.yaml"
    workflow = module.build_graph(path)
    assert workflow.options is options
    assert queries == [(visualize, visualize, True)]
    assert placements
    assert all(placement["device"].value == 3 for placement in placements)
    if visualize:
        assert {
            ("display_compositor", "rgba"),
            ("dashboard_image", "image"),
            ("dashboard_frame", "output"),
        } <= {(item["operator_path"], item["output_port"]) for item in placements}


@pytest.mark.parametrize("module,directory,preset", _CASES, ids=[item[2] for item in _CASES])
@pytest.mark.parametrize("composer_device", (0, 2))
@pytest.mark.parametrize("x11", (False, True))
@pytest.mark.parametrize("x11_enabled", (False, True))
def test_distinct_display_device_adds_explicit_image_transfer(
    module, directory, preset, composer_device, x11, x11_enabled, monkeypatch, tmp_path
):
    placements = []
    options = SimpleNamespace(
        deployment=SimpleNamespace(bind_tensor_output_device=lambda **item: placements.append(item))
    )
    monkeypatch.setattr(hs, "CompileOptions", lambda: options)

    def select_device(window, cuda_sink, enabled):
        assert window and cuda_sink
        assert enabled is x11_enabled
        return 0

    monkeypatch.setattr(_radar, "_select_application_device", select_device)
    monkeypatch.setattr(_radar, "_application_display_device", lambda: 3)
    monkeypatch.setattr(_radar, "_application_composer_device", lambda: composer_device)
    monkeypatch.setattr(_radar, "_fmcw_uses_x11_presentation", lambda enabled: enabled and x11)
    uses_x11 = x11 and x11_enabled
    if uses_x11:

        def unexpected_display_query():
            raise AssertionError("X11 transport must not depend on a window presentation GPU")

        monkeypatch.setattr(_radar, "_application_display_device", unexpected_display_query)
    config_path = tmp_path / "presentation.yaml"
    base = _ROOT / directory / "config" / f"{preset}.yaml"
    config_path.write_text(
        f'extends: "{base}"\napp:\n  visualization:\n    x11_presentation: {str(x11_enabled).lower()}\n'
    )
    module.build_graph(config_path)
    upload = [item for item in placements if item["operator_path"] == "dashboard_upload"]
    expected_devices = {"dashboard_frame": composer_device}
    if uses_x11:
        assert not upload
    else:
        assert len(upload) == 1
        assert upload[0]["output_port"] == "image_out"
        assert upload[0]["device"].value == 3
        expected_devices["dashboard_upload"] = 3
    if composer_device != 0:
        expected_devices["dashboard_composer_upload"] = composer_device
    assert {item["operator_path"] for item in placements if item["output_port"] == "image_out"} == {
        name for name in expected_devices if name.endswith("_upload")
    }
    assert all(
        item["device"].value == expected_devices.get(item["operator_path"], 0)
        for item in placements
    )


@pytest.mark.parametrize("display", (None, "", ":99"))
def test_disabled_x11_presentation_ignores_display(display, monkeypatch):
    if display is None:
        monkeypatch.delenv("DISPLAY", raising=False)
    else:
        monkeypatch.setenv("DISPLAY", display)
    assert not _radar._fmcw_uses_x11_presentation(False)


@pytest.mark.parametrize("module,directory,preset", _CASES, ids=[item[2] for item in _CASES])
def test_choose_presenter_reaches_shared_selector(module, directory, preset, monkeypatch):
    calls = []

    def choose(window, configured, choose_again, config_path):
        calls.append((window, configured, choose_again, config_path))
        return False

    monkeypatch.setattr(_radar, "_resolve_x11_presentation", choose)
    monkeypatch.setattr(_radar, "_select_application_device", lambda *_: 0)
    monkeypatch.setattr(_radar, "_application_display_device", lambda: 0)
    monkeypatch.setattr(_radar, "_application_composer_device", lambda: 0)
    module.build_graph(_ROOT / directory / "config" / f"{preset}.yaml", choose_presenter=True)
    assert calls == [(True, None, True, str(_ROOT / directory / "config" / f"{preset}.yaml"))]
