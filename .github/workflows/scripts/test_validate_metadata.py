"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import tempfile
import unittest
from pathlib import Path

import validate_metadata


class ValidateMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name) / "repo"
        self.repo_root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_reads_regular_file_within_repository(self):
        path = self.repo_root / "metadata.json"
        path.write_text('{"application": {}}')

        self.assertEqual(
            validate_metadata.read_metadata_file(path, self.repo_root),
            '{"application": {}}',
        )

    def test_rejects_symlink(self):
        target = self.repo_root / "target.json"
        target.write_text("{}")
        path = self.repo_root / "metadata.json"
        path.symlink_to(target)

        with self.assertRaisesRegex(ValueError, "regular file"):
            validate_metadata.read_metadata_file(path, self.repo_root)

    def test_rejects_parent_directory_symlink(self):
        target_dir = self.repo_root / "target"
        target_dir.mkdir()
        (target_dir / "metadata.json").write_text("{}")
        link = self.repo_root / "linked"
        link.symlink_to(target_dir, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "symlink"):
            validate_metadata.read_metadata_file(link / "metadata.json", self.repo_root)

    def test_rejects_oversized_file(self):
        path = self.repo_root / "metadata.json"
        path.write_bytes(b" " * (validate_metadata._MAX_METADATA_BYTES + 1))

        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_metadata.read_metadata_file(path, self.repo_root)

    def test_rejects_path_outside_repository(self):
        path = Path(self.temp_dir.name) / "metadata.json"
        path.write_text("{}")

        with self.assertRaisesRegex(ValueError, "outside"):
            validate_metadata.read_metadata_file(path, self.repo_root)

    def test_rejects_invalid_utf8(self):
        path = self.repo_root / "metadata.json"
        path.write_bytes(b"\xff")

        with self.assertRaisesRegex(ValueError, "not UTF-8"):
            validate_metadata.read_metadata_file(path, self.repo_root)


if __name__ == "__main__":
    unittest.main()
