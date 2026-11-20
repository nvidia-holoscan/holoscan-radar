# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline reporting checks, including the real subprocess output boundary."""

import contextlib
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unittest import mock

import pull_cphd
from sar_data.progress import DownloadProgress


class ProgressTests(unittest.TestCase):
    def test_terminal_and_redirected_metrics_match(self):
        outputs = []
        for tty in (False, True):
            stream = io.StringIO()
            with (
                contextlib.redirect_stderr(stream),
                mock.patch.object(stream, "isatty", return_value=tty),
                mock.patch("sar_data.progress.time.monotonic", return_value=100),
            ):
                meter = DownloadProgress("[1/2] scene", 1024)
                meter.set(512)
                with mock.patch("sar_data.progress.time.monotonic", return_value=102):
                    meter.finish()
            text = stream.getvalue()
            self.assertIn("512.0 B / 1.0 KiB | 50.0%", text)
            self.assertIn("256.0 B/s | elapsed 00:02 | ETA 00:02 | Complete", text)
            self.assertEqual("\r" in text, tty)
            self.assertNotIn("\x1b", text)
            self.assertTrue(text.endswith("\n"))
            outputs.append(text.splitlines()[-1])
        self.assertEqual(*outputs)

    def test_redirected_updates_are_throttled_and_still_show_waits(self):
        stream = io.StringIO()
        with (
            contextlib.redirect_stderr(stream),
            mock.patch("sar_data.progress.time.monotonic", return_value=100) as clock,
        ):
            meter = DownloadProgress("scene", 100)
            meter._draw()
            for _ in range(100):
                meter.add(1)
            self.assertEqual(len(stream.getvalue().splitlines()), 1)
            clock.return_value = 102
            meter._draw()  # Same callback used by the background heartbeat.
            self.assertEqual(len(stream.getvalue().splitlines()), 2)
            meter.finish()
        self.assertNotIn("\\r", stream.getvalue())
        self.assertIn("elapsed 00:02", stream.getvalue())

    def test_unknown_size_and_resume_rate(self):
        stream = io.StringIO()
        with (
            contextlib.redirect_stderr(stream),
            mock.patch("sar_data.progress.time.monotonic", return_value=100),
        ):
            meter = DownloadProgress("scene", None)
            meter.set(20)
            self.assertIn("20.0 B / ? | ?%", stream.getvalue())
            self.assertIn("ETA --:--", stream.getvalue())
            meter.phase("Subset preparation", 1000, initial=800)
            with mock.patch("sar_data.progress.time.monotonic", return_value=102):
                meter.set(900)
                meter.finish()
        self.assertIn("50.0 B/s | elapsed 00:02 | ETA 00:02", stream.getvalue())

    def test_heartbeat_during_blocked_work_and_shutdown(self):
        updated = threading.Event()

        class Terminal(io.StringIO):
            def isatty(self):
                return True

            def write(self, text):
                if self.getvalue().count("elapsed") >= 1 and "elapsed" in text:
                    updated.set()
                return super().write(text)

        stream = Terminal()
        with contextlib.redirect_stderr(stream):
            with DownloadProgress("waiting", None) as meter:
                self.assertTrue(updated.wait(2), "no heartbeat while transfer is blocked")
            self.assertFalse(meter._thread.is_alive())
        self.assertEqual(stream.getvalue().count("Complete"), 1)

    def test_failure_stops_heartbeat_and_terminates_terminal_line(self):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), self.assertRaisesRegex(RuntimeError, "broken"):
            with DownloadProgress("scene", 10) as meter:
                raise RuntimeError("broken")
        self.assertFalse(meter._thread.is_alive())
        self.assertIn("Failed\n", stream.getvalue())
        self.assertNotIn("Complete", stream.getvalue())

    def test_aws_progress_is_translated_and_destination_is_published(self):
        real_popen = subprocess.Popen
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "collect.cphd"
            partial = pull_cphd.partial_path(dest)
            script = (
                "import pathlib, sys; "
                "sys.stdout.write('Completed 2.0 Bytes/4.0 Bytes (2 Bytes/s)\\r'); "
                "sys.stdout.flush(); "
                f"pathlib.Path({str(partial)!r}).write_bytes(b'data'); "
                "print('Completed 4.0 Bytes/4.0 Bytes (2 Bytes/s)')"
            )

            def launch(command, **kwargs):
                self.assertEqual(command[:3], ["aws", "s3", "cp"])
                self.assertIn("--no-sign-request", command)
                self.assertNotIn("--no-progress", command)
                return real_popen([sys.executable, "-c", script], **kwargs)

            with (
                contextlib.redirect_stderr(output),
                mock.patch.object(pull_cphd.subprocess, "Popen", side_effect=launch),
            ):
                meter = DownloadProgress("[1/1] scene", 4)
                pull_cphd.download_aws("bucket", "collect", dest, meter)
            self.assertEqual(meter.done, 4)
            self.assertEqual(dest.read_bytes(), b"data")
            self.assertFalse(partial.exists())
        self.assertNotIn("Completed 2.0 Bytes", output.getvalue())
        self.assertIn("4.0 B / 4.0 B | 100.0%", output.getvalue())
        self.assertEqual(output.getvalue().count("Complete"), 1)

    def test_aws_failure_still_falls_back_to_https(self):
        with tempfile.TemporaryDirectory() as temp:
            job = pull_cphd.PullJob(
                "scene", "test.cphd", "bucket", pull_cphd.UMBRA_HTTPS, Path(temp), False
            )
            response = io.BytesIO(b"data")
            response.headers = {"Content-Length": "4"}
            output = io.StringIO()
            with (
                contextlib.redirect_stderr(output),
                mock.patch.object(pull_cphd.shutil, "which", return_value="aws"),
                mock.patch.object(pull_cphd, "aws_size", return_value=4),
                mock.patch.object(pull_cphd, "https_size", return_value=4),
                mock.patch.object(
                    pull_cphd, "download_aws", side_effect=subprocess.CalledProcessError(1, "aws")
                ),
                mock.patch.object(pull_cphd, "open_source_url", return_value=response),
            ):
                self.assertEqual(pull_cphd.pull_one(job, False, 1, 1).read_bytes(), b"data")
            self.assertIn("falling back to HTTPS", output.getvalue())
            self.assertIn("Failed", output.getvalue())
            self.assertIn("Complete", output.getvalue())


if __name__ == "__main__":
    unittest.main()
