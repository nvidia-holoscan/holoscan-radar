# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical and execution parity for all six Python-authored SAR graphs."""

import importlib
import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

import holoscan as hs

_ROOT = Path(__file__).resolve().parents[2]
_APPS = (
    "spotlight",
    "video_sar",
    "tiled_strip",
    "continuous_strip",
    "independent_images",
    "autofocus",
)
_CASES = [(app, f"{app}_generated_smoke.yaml") for app in _APPS] + [
    ("video_sar", "video_sar_retention_generated_smoke.yaml"),
    ("tiled_strip", "tiled_strip_seam_viz_generated_smoke.yaml"),
    ("independent_images", "independent_images_varying_viz_generated_smoke.yaml"),
    ("autofocus", "autofocus_generated_smoke_lie.yaml"),
    ("autofocus", "autofocus_generated_smoke_oracle.yaml"),
    ("autofocus", "autofocus_generated_smoke_polar.yaml"),
]


def _build_root():
    # CTest supplies this for both source and CLI container builds. An explicit
    # value also supports focused pytest invocation in a development container.
    return Path(os.environ["HOLOSCAN_RADAR_TEST_BUILD_DIR"])


def _preset(app, name):
    return _build_root() / "applications" / "sar" / app / "cpp" / name


def _builder(app):
    return importlib.import_module(f"holoscan_radar.applications.{app}").build_graph


def _write_config(source, destination, *, precision="single", dump=None, visualize=False):
    config = yaml.safe_load(source.read_text())
    config["sample_precision"] = precision
    config["visualize"] = visualize
    config["headless"] = True
    config["visualize_size"] = 32
    config["dump"] = dump is not None
    if dump is not None:
        config["output"] = {"path": str(dump / "image_{index}.npy")}
    destination.write_text(yaml.safe_dump(config))
    return destination


@pytest.mark.parametrize("app,preset", _CASES)
@pytest.mark.parametrize("precision", ("single", "double"))
def test_sar_complex_products_match_cpp(app, preset, precision, tmp_path):
    source = _preset(app, preset)
    reference_dir, python_dir = tmp_path / "cpp", tmp_path / "python"
    reference_dir.mkdir()
    python_dir.mkdir()
    reference = _write_config(
        source, tmp_path / "cpp.yaml", precision=precision, dump=reference_dir
    )
    candidate = _write_config(
        source, tmp_path / "python.yaml", precision=precision, dump=python_dir
    )
    executable = _build_root() / "applications" / "sar" / app / "cpp" / app
    result = subprocess.run(
        [str(executable), str(reference)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected_count = int(
        re.search(r"stopped after (\d+) image\(s\)", result.stderr + result.stdout)[1]
    )
    workflow = _builder(app)(candidate)
    assert workflow.pools["ingest_pulses"] <= workflow.pools["aperture_pulses"]
    assert workflow.run() == expected_count
    expected, actual = sorted(reference_dir.glob("image_*.npy")), sorted(
        python_dir.glob("image_*.npy")
    )
    assert len(actual) == len(expected) == expected_count
    for reference_path, actual_path in zip(expected, actual, strict=True):
        reference_image, image = np.load(reference_path), np.load(actual_path)
        assert (
            image.dtype
            == reference_image.dtype
            == np.dtype("complex64" if precision == "single" else "complex128")
        )
        assert np.isfinite(image).all()
        # Identical native kernels and data, allowing float accumulation order.
        np.testing.assert_allclose(
            image,
            reference_image,
            rtol=2e-5 if precision == "single" else 1e-10,
            atol=1e-7 if precision == "single" else 1e-12,
        )


@pytest.mark.parametrize("app", _APPS)
def test_sar_headless_visualization_executes(app, tmp_path):
    source = _preset(app, f"{app}_generated_smoke.yaml")
    path = _write_config(source, tmp_path / "viz.yaml", visualize=True)
    workflow = _builder(app)(path)
    assert hs.compile(workflow.graph, workflow.options).ok
    assert workflow.run() > 0


def test_sar_runtime_data_root_resolves_relocated_collection(tmp_path, monkeypatch):
    data_root = tmp_path / "runtime_data"
    cphd = data_root / "sar" / "relocated.cphd"
    cphd.parent.mkdir(parents=True)
    shutil.copyfile(_build_root() / "tests/cpp/generated_sar_smoke.cphd", cphd)
    path = _write_config(
        _preset("spotlight", "spotlight_generated_smoke.yaml"), tmp_path / "config.yaml"
    )
    config = yaml.safe_load(path.read_text())
    config["cphd"]["path"] = "sar/relocated.cphd"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HOLOHUB_DATA_DIR", str(data_root))
    assert _builder("spotlight")(path).run() == 1


def test_sar_contexts_have_independent_lifetimes_and_reject_second_run(tmp_path):
    source = _preset("spotlight", "spotlight_generated_smoke.yaml")
    path = _write_config(source, tmp_path / "config.yaml")
    first, second = _builder("spotlight")(path), _builder("spotlight")(path)
    assert first.graph is not second.graph
    assert first.run() == second.run() == 1
    with pytest.raises(RuntimeError, match="run once"):
        first.run()


@pytest.mark.parametrize("app", _APPS)
def test_sar_language_manifests_preserve_modes(app):
    root = _ROOT / "applications" / "sar" / app
    cpp, python = [
        json.loads((root / lang / "metadata.json").read_text())["application"]
        for lang in ("cpp", "python")
    ]
    assert cpp["modes"].keys() == python["modes"].keys()
    for mode, definition in cpp["modes"].items():
        assert definition["description"] == python["modes"][mode]["description"]
        assert f"applications.cli {app}" in python["modes"][mode]["run"]["command"]


def test_sar_cli_passes_cphd_override(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from holoscan_radar.applications import cli

    path = tmp_path / "app.yaml"
    path.write_text("{}")
    calls = []

    def build(config, *, cphd_override):
        calls.append((config, cphd_override))
        return SimpleNamespace(run=lambda: calls.append("run"))

    monkeypatch.setattr(cli, "import_module", lambda _: SimpleNamespace(build_graph=build))
    assert cli.main(["spotlight", str(path), "--cphd", "override.cphd"]) == 0
    assert calls == [(path, "override.cphd"), "run"]


@pytest.mark.parametrize("signum", (signal.SIGINT, signal.SIGTERM), ids=("sigint", "sigterm"))
def test_sar_looping_cli_stops_cooperatively(signum, tmp_path):
    import sys
    import time

    config = yaml.safe_load(_preset("spotlight", "spotlight_generated_smoke.yaml").read_text())
    config["cphd_reader"].update(loop=True, playback_rate=1.0, wait_for_downstream=True)
    path = tmp_path / "loop.yaml"
    path.write_text(yaml.safe_dump(config))
    log_path = tmp_path / "run.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "holoscan_radar.applications.cli", "spotlight", str(path)],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 15
            while "image 1 formed" not in log_path.read_text():
                assert process.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, log_path.read_text()
                time.sleep(0.05)
            process.send_signal(signum)
            assert process.wait(timeout=10) == 130, log_path.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert "RunSession failed" not in log_path.read_text()
