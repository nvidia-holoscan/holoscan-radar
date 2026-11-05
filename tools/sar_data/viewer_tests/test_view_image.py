#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.24", "matplotlib>=3.8"]
# ///

"""Offline viewer checks: uv run tools/sar_data/viewer_tests/test_view_image.py."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import view_image  # noqa: E402


class ImageViewerTest(unittest.TestCase):
    def test_known_complex_amplitudes_and_shared_decibel_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "phase.npy"
            np.save(path, np.array([[10j, -1, 0.1j, 0]], dtype=np.complex64))
            magnitude = view_image.load_magnitude(path)
            np.testing.assert_allclose(
                view_image.decibels(magnitude, 10, 60), [[0, -20, -40, -60]], atol=1e-5
            )
            # A dimmer comparison must stay 20 dB dimmer, not normalize itself.
            np.testing.assert_allclose(
                view_image.decibels(magnitude / 10, 10, 60), [[-20, -40, -60, -60]], atol=1e-5
            )

    def test_integer_minimum_and_large_complex_values_do_not_overflow(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.npy"
            for image, expected in (
                (np.array([[-32768]], dtype=np.int16), 32768),
                (np.array([[3e38 + 3e38j]], dtype=np.complex64), np.hypot(3e38, 3e38)),
            ):
                np.save(path, image)
                self.assertAlmostEqual(
                    float(view_image.load_magnitude(path)[0, 0]) / expected, 1, places=6
                )

    def test_invalid_images_fail_without_writing_a_png(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "invalid.npy"
            for image in (
                np.zeros((2, 2)),
                np.array([[np.nan]]),
                np.array([[np.inf * 1j]]),
                np.ones(3),
                np.ones((1, 2, 3)),
                np.empty((0, 2)),
                np.array([["text"]]),
                np.array([[object()]]),
            ):
                with self.subTest(shape=image.shape, dtype=image.dtype):
                    np.save(path, image)
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(view_image.main([str(path)]), 1)
                    self.assertFalse(path.with_suffix(".png").exists())

    def test_missing_corrupt_and_archive_inputs_report_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corrupt = root / "corrupt.npy"
            corrupt.write_bytes(b"not a NumPy image")
            archive = root / "archive.npz"
            np.savez(archive, image=np.ones((2, 2)))
            for path in (root / "missing.npy", corrupt, archive):
                with self.subTest(path=path):
                    with contextlib.redirect_stderr(io.StringIO()) as error:
                        self.assertEqual(view_image.main([str(path)]), 1)
                    self.assertIn("view_image:", error.getvalue())
                    self.assertFalse(path.with_suffix(".png").exists())

    def test_single_and_comparison_pngs_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "clean's image.npy", Path(temp) / "corrected.npy"
            np.save(first, np.array([[1j, 10], [-1, 0.1j]]))
            np.save(second, np.array([[0.1j, 1], [-0.1, 0.01j]]))
            original = first.read_bytes(), second.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(view_image.main([str(first)]), 0)
                self.assertEqual(view_image.main([str(first), str(second), "--compare"]), 0)
                custom = Path(temp) / "figures" / "custom.png"
                self.assertEqual(
                    view_image.main([str(first), "--range-db", "80", "--output", str(custom)]), 0
                )
                self.assertTrue(custom.is_file())
            for png in (first.with_suffix(".png"), first.with_name(first.stem + "_comparison.png")):
                self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertGreater(png.stat().st_size, 1000)
            self.assertEqual((first.read_bytes(), second.read_bytes()), original)

    def test_comparison_rejects_different_shapes(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first.npy", Path(temp) / "second.npy"
            np.save(first, np.ones((2, 2)))
            np.save(second, np.ones((3, 3)))
            with contextlib.redirect_stderr(io.StringIO()) as error:
                self.assertEqual(view_image.main([str(first), str(second), "--compare"]), 1)
            self.assertIn("matching array shapes", error.getvalue())

    def test_invalid_cli_options_are_rejected(self):
        for options in (
            ["--range-db", "0"],
            ["--range-db", "nan"],
            ["--range-db", "inf"],
            ["--range-db=-1"],
            ["--compare"],
            ["second.npy"],
            ["-o", "image.npy"],
        ):
            with self.subTest(options=options):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    view_image.main(["image.npy", *options])
                self.assertEqual(caught.exception.code, 2)

    def test_headless_show_reports_saved_png_and_display_requirement(self):
        import matplotlib

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.npy"
            np.save(path, np.ones((2, 2)))
            with (
                mock.patch.object(matplotlib, "get_backend", return_value="Agg"),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as error,
            ):
                self.assertEqual(view_image.main([str(path), "--show"]), 1)
            self.assertTrue(path.with_suffix(".png").exists())
            self.assertIn("PNG saved", error.getvalue())


if __name__ == "__main__":
    unittest.main()
