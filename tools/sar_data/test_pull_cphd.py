# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline downloader checks: python3 -m unittest discover -s tools/sar_data."""

import contextlib
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import pull_cphd


class DownloadCliTest(unittest.TestCase):
    def test_autofocus_uses_original_catalog_object_and_preset_path(self):
        output = io.StringIO()
        with (
            mock.patch("sys.argv", ["pull_cphd.py", "--app", "autofocus", "--list"]),
            mock.patch.object(pull_cphd, "pull_one") as pull,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(pull_cphd.main(), 0)
        pull.assert_not_called()
        self.assertIn("sar-data/tasks/Kourou%2C%20French%20Guiana/", output.getvalue())
        self.assertTrue(
            output.getvalue()
            .rstrip()
            .endswith(" -> data/sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd")
        )

    def test_all_custom_root_keeps_shared_collection_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sar"
            jobs = pull_cphd.jobs_for("all", root)
            destinations = [job.dest_dir / Path(job.key).name for job in jobs]
            self.assertEqual(len(destinations), 8)
            self.assertEqual(len(set(destinations)), 8)
            self.assertIn(root / "2023-06-24-13-22-56_UMBRA-05_CPHD.cphd", destinations)
            self.assertEqual(sum(p.parent == root / "imperial" for p in destinations), 1)
            self.assertEqual(sum(p.parent == root / "mode5" for p in destinations), 6)
            self.assertFalse(root.exists())

    def test_all_rejects_collection_subdirectory_before_download(self):
        for dest in (
            "data/sar/imperial",
            "data/sar/mode5",
            str(Path("data/sar/imperial").resolve()),
        ):
            with self.subTest(dest=dest):
                output = io.StringIO()
                with (
                    mock.patch("sys.argv", ["pull_cphd.py", "--app", "all", "--dest", dest]),
                    mock.patch.object(pull_cphd, "pull_one") as pull,
                    contextlib.redirect_stderr(output),
                    self.assertRaises(SystemExit) as caught,
                ):
                    pull_cphd.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--app all --dest data/sar", output.getvalue())
                pull.assert_not_called()

    def test_http_404_reports_source_and_cleans_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
            output = io.StringIO()
            with (
                mock.patch("sys.argv", ["pull_cphd.py", "--app", "autofocus", "--dest", temp]),
                mock.patch.object(pull_cphd.shutil, "which", return_value=None),
                mock.patch.object(pull_cphd, "https_size", return_value=None),
                mock.patch.object(
                    pull_cphd.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.HTTPError(
                        "https://example.test/missing", 404, "Not Found", {}, None
                    ),
                ),
                contextlib.redirect_stderr(output),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(pull_cphd.main(), 1)
            self.assertIn("HTTP Error 404", output.getvalue())
            self.assertIn("Source: https://umbra-open-data-catalog.", output.getvalue())
            self.assertIn(f"Destination: {dest}", output.getvalue())
            self.assertNotIn("Traceback", output.getvalue())
            self.assertFalse(dest.exists())
            self.assertFalse(pull_cphd.partial_path(dest).exists())


if __name__ == "__main__":
    unittest.main()
