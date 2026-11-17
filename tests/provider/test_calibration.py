# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline CFAR calibration, portable LUTs, progress and cancellation."""

import json

import pytest
from holoscan_radar import calibrate_cfar, plan_cfar_calibration


def config(variant="CA"):
    return {"cfar_variant": variant, "pfa": 0.1, "guard_cells": [1, 1], "training_cells": [1, 1]}


def test_analytic_and_external_round_trip():
    result = calibrate_cfar(config(), [16, 16])
    assert result["method"] == "analytic"
    assert result["total_trials"] == 0
    assert result["coverage"] is None
    saved = json.loads(json.dumps(result))
    p = {**config("GO"), "pfa": 1e-9, "alpha_lut": saved["alpha_lut"]}
    # GO indexes a smaller table than CA; all supplied entries remain valid.
    reused = calibrate_cfar(p, [16, 16])
    assert reused["method"] == "external"
    assert reused["coverage"] is None
    assert reused["alpha_lut"] == saved["alpha_lut"]
    assert plan_cfar_calibration(p, [16, 16])["total_trials"] == 0


def test_go_progress_and_reproducibility():
    events = []
    first = calibrate_cfar(config("GO"), [8, 8], progress=events.append)
    second = calibrate_cfar(config("GO"), [8, 8])
    assert first["alpha_lut"] == second["alpha_lut"]
    assert events[0]["status"] == "started"
    assert events[-1]["status"] == "completed"
    assert events[-1]["completed_trials"] == first["total_trials"]
    assert first["coverage"] >= 0.95
    assert first["seed"] == second["seed"]


def test_rare_pfa_plan_and_keyboard_interrupt():
    p = {**config("GO"), "pfa": 1e-9}
    plan = plan_cfar_calibration(p, [16, 16])
    assert plan["trials_per_geometry"] > 380_000_000_000
    assert plan["estimated_seconds"] > 0

    def interrupt(event):
        if event["status"] == "running":
            raise KeyboardInterrupt("cancel test")

    with pytest.raises(KeyboardInterrupt, match="cancel test"):
        calibrate_cfar(p, [16, 16], progress=interrupt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pfa", 0),
        ("calibration_relative_error", 0),
        ("calibration_confidence", 1),
        ("calibration_trials", -1),
    ],
)
def test_invalid_accuracy(field, value):
    with pytest.raises((ValueError, RuntimeError)):
        plan_cfar_calibration({**config(), field: value}, [16, 16])


def test_python_interrupt_is_serviced_between_native_batches():
    import subprocess
    import sys

    code = """
import os
import signal
import threading
from holoscan_radar import calibrate_cfar
p = {"cfar_variant": "GO", "pfa": 1e-9, "guard_cells": [1, 1], "training_cells": [1, 1]}
timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT))
timer.start()
try:
    calibrate_cfar(p, [16, 16])
except KeyboardInterrupt:
    print("interrupted between batches")
else:
    raise AssertionError("rare-event job unexpectedly completed")
finally:
    timer.join()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert "interrupted between batches" in result.stdout


def test_graph_stop_cancels_runtime_calibration(tmp_path, capfd):
    import time
    from pathlib import Path

    import yaml
    from holoscan_radar.applications import dense_beamforming

    import holoscan as hs

    path = (
        Path(__file__).resolve().parents[2]
        / "applications/fmcw/dense_beamforming/config/dense_beamforming_smoke.yaml"
    )
    config_data = {
        "extends": str(path),
        "cfar_detector": {"cfar_variant": "GO", "pfa": 1e-9},
    }
    preset = tmp_path / "cancel.yaml"
    preset.write_text(yaml.safe_dump(config_data))
    workflow = dense_beamforming.build_graph(preset)
    plan = hs.compile(workflow.graph, workflow.options)
    assert plan.ok
    session = hs.run_async(plan)
    time.sleep(1)
    started = time.monotonic()
    session.request_stop()
    session.wait()
    assert time.monotonic() - started < 3
    assert not session.diagnostics, session.diagnostics
    assert "CFAR calibration GO start=" in capfd.readouterr().err


@pytest.mark.parametrize("variant", ["GO", "SO", "OS"])
def test_explicit_trials_override_rare_pfa_and_report_coverage(variant):
    p = {
        **config(variant),
        "pfa": 1e-30,
        "calibration_trials": 17,
        "preintegrated_looks": 2,
        "os_rank": 10,
    }
    events = []
    result = calibrate_cfar(p, [8, 8], progress=events.append)
    assert result["trials_per_geometry"] == 17
    assert result["tail_rank"] == 1
    assert result["coverage"] < p.get("calibration_confidence", 0.95)
    assert result["params"]["calibration_trials"] == 17
    assert result["total_trials"] == 17 * len(result["geometries"])
    assert events[-1]["completed_trials"] == result["total_trials"]
    assert plan_cfar_calibration(p, [8, 8])["trials_per_geometry"] == 17


def test_default_trials_and_analytic_bypass():
    default = plan_cfar_calibration(config("GO"), [8, 8])
    explicit_zero = plan_cfar_calibration({**config("GO"), "calibration_trials": 0}, [8, 8])
    assert default["params"]["calibration_trials"] == 0
    assert default["trials_per_geometry"] == explicit_zero["trials_per_geometry"]
    assert default["coverage"] >= 0.95
    analytic = plan_cfar_calibration({**config(), "calibration_trials": 17}, [8, 8])
    assert analytic["total_trials"] == 0
    assert analytic["coverage"] is None


def test_graph_runtime_trial_override(tmp_path, capfd):
    from pathlib import Path

    import yaml
    from holoscan_radar.applications import dense_beamforming

    path = (
        Path(__file__).resolve().parents[2]
        / "applications/fmcw/dense_beamforming/config/dense_beamforming_smoke.yaml"
    )
    config_data = {
        "extends": str(path),
        "cfar_detector": {"cfar_variant": "GO", "pfa": 1e-9, "calibration_trials": 1024},
    }
    preset = tmp_path / "override.yaml"
    preset.write_text(yaml.safe_dump(config_data))
    workflow = dense_beamforming.build_graph(preset)
    session = workflow.run()
    assert not session.diagnostics, session.diagnostics
    logs = capfd.readouterr().err
    assert "trials_per_geometry=1024" in logs
    assert "budget_mode=override" in logs
    assert "Not enough Monte Carlo trials to achieve the requested CFAR confidence" in logs
    assert "Requested coverage: 95%" in logs
    assert "Trials requested per geometry: 1024" in logs
    assert "Continuing with the requested trial count" in logs
