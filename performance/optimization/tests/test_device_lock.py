# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-lock isolation, contention, and hostile filesystem entries."""

import hashlib
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from measurement import device_lock


class DeviceLockTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / f"radar-operator-perf-locks-{os.getuid()}"
        self.name = hashlib.sha256(b"test-gpu").hexdigest() + ".lock"
        self.path = self.folder / self.name
        override = patch("measurement.tempfile.gettempdir", return_value=str(self.root))
        override.start()
        self.addCleanup(override.stop)

    def assert_rejected(self):
        with self.assertRaises((OSError, ValueError)):
            with device_lock("test-gpu"):
                self.fail("unsafe lock acquired")

    def test_private_modes_stable_inode_and_release_after_exception(self):
        with self.assertRaisesRegex(RuntimeError, "work failed"):
            with device_lock("test-gpu"):
                self.assertEqual(stat.S_IMODE(self.folder.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
                inode = self.path.stat().st_ino
                raise RuntimeError("work failed")
        with device_lock("test-gpu"):
            self.assertEqual(self.path.stat().st_ino, inode)

    def test_serializes_separate_processes_and_working_directories(self):
        child = """from measurement import device_lock
import sys
try:
    with device_lock('test-gpu'):
        sys.exit(0)
except ValueError:
    sys.exit(23)
"""
        env = os.environ | {
            "TMPDIR": str(self.root),
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        }
        elsewhere = self.root / "other-checkout"
        elsewhere.mkdir()
        with device_lock("test-gpu"):
            result = subprocess.run(
                [sys.executable, "-c", child], env=env, cwd=elsewhere, check=False, timeout=10
            )
            self.assertEqual(result.returncode, 23)
            with device_lock("different-gpu"):
                pass
        result = subprocess.run(
            [sys.executable, "-c", child], env=env, cwd=elsewhere, check=False, timeout=10
        )
        self.assertEqual(result.returncode, 0)

    def test_rejects_symlink_directory(self):
        target = self.root / "attacker"
        target.mkdir(mode=0o700)
        self.folder.symlink_to(target, target_is_directory=True)
        self.assert_rejected()
        self.assertEqual(list(target.iterdir()), [])

    def test_rejects_directory_with_shared_permissions(self):
        self.folder.mkdir(mode=0o700)
        self.folder.chmod(0o755)
        self.assert_rejected()
        self.assertFalse(self.path.exists())

    def test_rejects_foreign_owned_directory(self):
        # Stand in for another uid without requiring root or changing ownership.
        with patch("measurement.os.getuid", return_value=os.getuid() + 1):
            foreign = self.root / f"radar-operator-perf-locks-{os.getuid()}"
            foreign.mkdir(mode=0o700)
            self.assert_rejected()
            self.assertEqual(list(foreign.iterdir()), [])

    def test_rejects_symlink_file_without_creating_target(self):
        self.folder.mkdir(mode=0o700)
        target = self.root / "must-not-create"
        self.path.symlink_to(target)
        self.assert_rejected()
        self.assertFalse(target.exists())

    def test_rejects_hardlink_file_without_modifying_target(self):
        self.folder.mkdir(mode=0o700)
        target = self.root / "untouched"
        target.write_text("preserve")
        target.chmod(0o600)
        self.path.hardlink_to(target)
        self.assert_rejected()
        self.assertEqual(target.read_text(), "preserve")

    def test_rejects_fifo_without_blocking(self):
        self.folder.mkdir(mode=0o700)
        os.mkfifo(self.path, mode=0o600)
        self.assert_rejected()

    def test_rejects_shared_lock_file(self):
        self.folder.mkdir(mode=0o700)
        self.path.touch(mode=0o600)
        self.path.chmod(0o666)
        self.assert_rejected()

    def test_rejects_foreign_owned_lock_file(self):
        real_fstat = os.fstat

        def foreign_file(fd):
            info = real_fstat(fd)
            if stat.S_ISREG(info.st_mode):
                return SimpleNamespace(
                    st_mode=info.st_mode, st_uid=info.st_uid + 1, st_nlink=info.st_nlink
                )
            return info

        with patch("measurement.os.fstat", side_effect=foreign_file):
            self.assert_rejected()

    def test_ignores_legacy_shared_directory(self):
        target = self.root / "missing-target"
        (self.root / "radar-operator-perf-locks").symlink_to(target)
        with device_lock("test-gpu"):
            self.assertTrue(self.path.is_file())
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
