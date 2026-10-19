# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke checks for the seven Python FMCW Graph entry points."""

import re
from pathlib import Path

import pytest
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
def test_visualization_preset_compiles(module, directory, preset):
    path = _ROOT / directory / "config" / f"{preset}.yaml"
    workflow = module.build_graph(path)
    assert hs.compile(workflow.graph, workflow.options).ok
