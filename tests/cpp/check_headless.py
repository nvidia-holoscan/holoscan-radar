# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the actual C++ mode override without any window-system environment."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check(executable, config):
    with tempfile.TemporaryDirectory(prefix="radar-headless-") as directory:
        path = Path(directory) / "headless.yaml"
        path.write_text(
            f"extends: {json.dumps(str(Path(config).resolve()))}\n"
            "app:\n  run: {num_frames: 2}\n  visualization:\n"
            "    enabled: false\n    final_image_hold_s: -1\n    max_side: 32\n"
        )
        environment = dict(os.environ, XDG_CACHE_HOME=str(Path(directory) / "cache"))
        for name in ("DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "XDG_RUNTIME_DIR"):
            environment.pop(name, None)
        result = subprocess.run(
            [executable, str(path), "--presenter", "headless"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            env=environment,
            timeout=45,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output[-10000:]
        assert "headless GPU rendering" in output
        assert "Choose how to display" not in output
        assert "X11 finished-image transport" not in output
        assert not list((Path(directory) / "cache").rglob("*.json"))
        print("PASS: headless overrides disabled YAML, completes, and ignores indefinite GUI hold")


if __name__ == "__main__":
    check(*sys.argv[1:])
