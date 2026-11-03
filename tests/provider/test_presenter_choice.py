# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the native presenter selector through real process/terminal boundaries."""

import json
import os
import pty
import select
import subprocess
import sys
import time

import pytest

_CODE = (
    "from holoscan_radar import _radar; "
    'print(_radar._resolve_x11_presentation(True, None, FORCE, "custom_preset.yaml"))'
)


def _choose(env, replies, *, force=False):
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-c", _CODE.replace("FORCE", repr(force))],
        stdin=slave,
        stderr=slave,
        stdout=subprocess.PIPE,
        env=env,
    )
    os.close(slave)
    transcript = b""
    try:
        for marker, reply in zip((b"Presenter [", b"Remember ["), replies, strict=True):
            deadline = time.monotonic() + 15
            while marker not in transcript:
                if time.monotonic() >= deadline:
                    pytest.fail(f"Missing {marker!r}: {transcript!r}")
                if select.select([master], [], [], 0.1)[0]:
                    transcript += os.read(master, 4096)
            os.write(master, reply)
        stdout, _ = process.communicate(timeout=15)
        assert process.returncode == 0, transcript
        assert b'in "custom_preset.yaml".' in transcript
        return stdout.strip()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


def _unattended(env, *, configured="None", force=False):
    code = _CODE.replace("None", configured).replace("FORCE", repr(force))
    return subprocess.run(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=env,
        timeout=15,
        check=True,
    )


@pytest.mark.parametrize("remember", (b"\n", b"2\n", b"3\n"), ids=("once", "week", "forever"))
def test_remember_lifetime_across_processes(remember, tmp_path):
    env = dict(os.environ, DISPLAY=":99", XDG_CACHE_HOME=str(tmp_path))
    assert _choose(env, (b"n\n", remember)) == b"False"
    files = list(tmp_path.glob("holoscan-radar/presentation-v2/*.json"))
    saved = _unattended(env)
    if remember == b"\n":
        assert not files
        assert saved.stdout.strip() == b"True"
        return
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record["version"] == 2 and record["presenter"] == "native"
    assert record["saved_at"] > 0
    if remember == b"2\n":
        assert record["expires_at"] - record["saved_at"] == 7 * 24 * 60 * 60
        assert b"expires" in saved.stderr
    else:
        assert record["expires_at"] is None
        assert b"until changed" in saved.stderr
    assert saved.stdout.strip() == b"False"
    assert b"--choose-presenter" in saved.stderr
    assert _unattended(env, configured="True", force=True).stdout.strip() == b"True"
    assert _unattended(env).stdout.strip() == b"False"
    assert _choose(env, (b"x\n", b"1\n"), force=True) == b"True"
    assert not files[0].exists()
    assert _unattended(env).stdout.strip() == b"True"
