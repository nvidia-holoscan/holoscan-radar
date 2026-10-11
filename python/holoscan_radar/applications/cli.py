# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an FMCW Graph workflow from a Holoscan CLI application manifest."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path

_DEFAULT_CONFIG = {
    "siso": "siso.yaml",
    "simo": "simo.yaml",
    "dense_beamforming": "dense_beamforming.yaml",
    "mimo_tdm": "mimo_tdm.yaml",
    "mimo_ddm": "mimo_ddm_linear.yaml",
    "adaptive_aoa": "adaptive_aoa.yaml",
    "multi_waveform": "multi_waveform.yaml",
}


def _config_path(application: str, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / path
        if candidate.is_file():
            return candidate
        shipped = base / "applications" / "fmcw" / application / "config" / path
        if shipped.is_file():
            return shipped
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Holoscan Radar FMCW Graph application")
    parser.add_argument("application", choices=tuple(_DEFAULT_CONFIG))
    parser.add_argument("config", nargs="?", help="YAML preset or path to an application config")
    args = parser.parse_args(argv)
    config = _config_path(args.application, args.config or _DEFAULT_CONFIG[args.application])
    if not config.is_file():
        parser.error(f"configuration file not found: {config}")
    workflow = import_module(f"holoscan_radar.applications.{args.application}").build_graph(config)
    workflow.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
