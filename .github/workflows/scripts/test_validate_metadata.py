"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path

import validate_metadata


def _read_metadata_in_child(path, repo_root, result_pipe):
    try:
        validate_metadata.read_metadata_file(path, repo_root)
    except ValueError as exc:
        result_pipe.send(str(exc))
    else:
        result_pipe.send(None)
    finally:
        result_pipe.close()


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

    def test_rejects_fifo_without_blocking(self):
        path = self.repo_root / "metadata.json"
        os.mkfifo(path)

        context = multiprocessing.get_context("fork")
        result_pipe, child_pipe = context.Pipe(duplex=False)
        process = context.Process(
            target=_read_metadata_in_child,
            args=(path, self.repo_root, child_pipe),
        )
        process.start()
        child_pipe.close()
        try:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join()
                self.fail("reading a FIFO metadata.json blocked")

            self.assertEqual(process.exitcode, 0)
            self.assertTrue(result_pipe.poll(1))
            self.assertRegex(result_pipe.recv(), "regular file")
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            result_pipe.close()

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

    def test_skips_virtual_environments(self):
        for name in (".venv", "venv"):
            template = self.repo_root / name / "templates" / "{{cookiecutter.name}}"
            template.mkdir(parents=True)
            (template / "metadata.json").write_text("{{ cookiecutter }}")
        (self.repo_root / "metadata.json").write_text("{}")

        self.assertEqual(
            list(validate_metadata.iter_metadata_files(self.repo_root)),
            [self.repo_root / "metadata.json"],
        )


if __name__ == "__main__":
    unittest.main()
