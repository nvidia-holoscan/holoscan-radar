# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the build-tree ``holoscan.radar`` package importable."""

from __future__ import annotations

import os
import sys

_PERF_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF_ROOT not in sys.path:
    sys.path.insert(0, _PERF_ROOT)


def bootstrap_radar_package() -> str:
    """Extend ``holoscan.__path__`` with the CMake build tree; return build dir."""
    build_dir = os.environ.get(
        "HOLOSCAN_RADAR_BUILD_DIR",
        os.path.join(
            os.path.dirname(_PERF_ROOT),
            "..",
            "build",
        ),
    )
    build_dir = os.path.abspath(build_dir)
    build_holoscan_path = os.path.join(build_dir, "python", "lib", "holoscan")
    if not os.path.isdir(build_holoscan_path):
        raise RuntimeError(
            f"holoscan.radar build tree not found at {build_holoscan_path}; "
            "set HOLOSCAN_RADAR_BUILD_DIR or build the project first"
        )

    import holoscan  # noqa: WPS433 — intentional late import after path setup

    if build_holoscan_path not in holoscan.__path__:
        holoscan.__path__.insert(0, build_holoscan_path)
    return build_dir
