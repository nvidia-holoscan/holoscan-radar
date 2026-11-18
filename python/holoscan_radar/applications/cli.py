# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a Radar Graph workflow from a Holoscan CLI application manifest."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

_SAR_APPLICATIONS = (
    "spotlight",
    "video_sar",
    "tiled_strip",
    "continuous_strip",
    "independent_images",
    "autofocus",
)

_DEFAULT_CONFIG = {
    "spotlight": "spotlight.yaml",
    "video_sar": "video_sar.yaml",
    "tiled_strip": "tiled_strip.yaml",
    "continuous_strip": "continuous_strip.yaml",
    "independent_images": "independent_images.yaml",
    "autofocus": "autofocus.yaml",
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
        shipped = (
            base
            / "applications"
            / ("sar" if application in _SAR_APPLICATIONS else "fmcw")
            / application
            / "config"
            / path
        )
        if shipped.is_file():
            return shipped
    return path


def _print_sar_input_error(error: RuntimeError) -> None:
    """Style actionable native input diagnostics only at the terminal boundary."""
    color = (
        sys.stderr.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
    )
    for index, line in enumerate(str(error).splitlines()):
        style = ""
        if index == 0:
            line = f"Error: {line}"
            style = "\033[1;31m"
        elif line.startswith("Next step:"):
            style = "\033[1;36m"
        elif line == "More information":
            style = "\033[1m"
        if color and style:
            line = f"{style}{line}\033[0m"
        print(line, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Holoscan Radar Graph application")
    parser.add_argument("application", choices=tuple(_DEFAULT_CONFIG))
    parser.add_argument("config", nargs="?", help="YAML preset or path to an application config")
    parser.add_argument("cphd_override", nargs="?", help="SAR CPHD input override")
    parser.add_argument("--cphd", help="Override the SAR CPHD input path")
    parser.add_argument(
        "--choose-presenter",
        action="store_true",
        help="Forget the saved presenter for this display setup and ask again (explicit YAML wins)",
    )
    args = parser.parse_args(argv)
    cphd_override = args.cphd or args.cphd_override
    if (
        args.application in _SAR_APPLICATIONS
        and args.config
        and Path(args.config).suffix.lower() == ".cphd"
    ):
        if cphd_override:
            parser.error("supply only one CPHD override")
        cphd_override, args.config = args.config, None
    config = _config_path(args.application, args.config or _DEFAULT_CONFIG[args.application])
    if not config.is_file():
        parser.error(f"configuration file not found: {config}")
    builder = import_module(f"holoscan_radar.applications.{args.application}").build_graph
    if args.application in _SAR_APPLICATIONS:
        if args.choose_presenter:
            parser.error("--choose-presenter is available for FMCW applications only")
        if args.cphd and args.cphd_override:
            parser.error("supply only one CPHD override")
        try:
            workflow = builder(config, cphd_override=cphd_override)
        except RuntimeError as error:
            # Other build and execution failures retain their traceback for diagnosis.
            if not str(error).startswith("CPHD input file not found:\n"):
                raise
            _print_sar_input_error(error)
            return 1
    else:
        if args.cphd or args.cphd_override:
            parser.error("--cphd is available for SAR applications only")
        workflow = builder(config, choose_presenter=args.choose_presenter)
    try:
        workflow.run()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
