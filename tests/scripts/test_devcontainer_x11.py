# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the host startup command without changing a real X server's ACL."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DevcontainerX11Test(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / ".devcontainer/devcontainer.json"
        text = "\n".join(
            line for line in path.read_text().splitlines() if not line.lstrip().startswith("//")
        )
        self.config = json.loads(text)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "requests"
        (self.root / "bash").symlink_to("/bin/bash")
        self.script("id", 'printf "%s\\n" "$MOCK_USER"; exit "${ID_EXIT:-0}"')
        self.script("xhost", 'printf "%s\\n" "$@" >> "$REQUEST_LOG"; exit "${XHOST_EXIT:-0}"')

    def script(self, name, body):
        path = self.root / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o700)

    def run_startup(self, **overrides):
        env = (
            os.environ
            | {
                "PATH": str(self.root),
                "REQUEST_LOG": str(self.log),
                "MOCK_USER": "test-user",
                "DISPLAY": ":42",
                "XAUTHORITY": "/existing/forwarding-cookie",
            }
            | overrides
        )
        result = subprocess.run(
            ["/bin/bash", "-c", self.config["initializeCommand"]],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_local_display_uses_only_the_uid_matched_account(self):
        self.assertEqual(self.config["remoteUser"], "dev")
        self.assertTrue(self.config["updateRemoteUserUID"])
        self.assertEqual(self.run_startup(), ["+SI:localuser:test-user"])
        self.assertEqual(self.config["containerEnv"]["DISPLAY"], "${localEnv:DISPLAY}")

    def test_failed_scoped_authorization_never_falls_back_to_local_or_root(self):
        self.assertEqual(self.run_startup(XHOST_EXIT="1"), ["+SI:localuser:test-user"])

    def test_missing_xhost_is_a_soft_startup(self):
        (self.root / "xhost").unlink()
        self.assertEqual(self.run_startup(), [])

    def test_failed_identity_lookup_does_not_add_an_empty_grant(self):
        self.assertEqual(self.run_startup(ID_EXIT="1"), [])

    def test_host_username_remains_one_literal_argument(self):
        name = "example user; $(must-not-run)"
        self.assertEqual(self.run_startup(MOCK_USER=name), ["+SI:localuser:" + name])

    def test_headless_failure_is_soft(self):
        self.assertEqual(self.run_startup(DISPLAY="", XHOST_EXIT="1"), ["+SI:localuser:test-user"])

    def test_forwarded_display_keeps_its_existing_authentication_environment(self):
        self.script(
            "xhost",
            'printf "%s\\n" "$@" "$DISPLAY" "$XAUTHORITY" >> "$REQUEST_LOG"',
        )
        self.assertEqual(
            self.run_startup(DISPLAY="localhost:10.0"),
            ["+SI:localuser:test-user", "localhost:10.0", "/existing/forwarding-cookie"],
        )


if __name__ == "__main__":
    unittest.main()
