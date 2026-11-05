# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline report checks: python3 -m unittest discover -s tools/sar_autofocus."""

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sweep


def report(reason="converged", accepted=2, residual=0.1, baseline=2.0):
    return (
        f"AutofocusScoreOp: affine-removed RMS {residual} rad (1 mm LOS) "
        f"baseline {baseline} rad RMS\n"
        "AutofocusSharpnessReportOp: sharpness 1 -> 2 (2.000x over 8 iteration(s))\n"
        f"AutofocusSharpnessReportOp: termination {reason}, {accepted} iteration(s) accepted\n"
    )


class SweepReportTest(unittest.TestCase):
    def run_report(self, text, returncode=0):
        completed = subprocess.CompletedProcess(["autofocus"], returncode, "", text)
        with (
            mock.patch.object(sweep.subprocess, "run", return_value=completed),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return sweep.run_one(Path("autofocus"), {}, "scene", "case")

    def test_failure_after_good_score_is_not_a_result(self):
        self.assertIsNone(self.run_report(report() + "product formation failed", returncode=1))

    def test_accepted_no_progress_is_not_a_refusal(self):
        result = self.run_report(report(residual=2.0))
        self.assertIsNotNone(result)
        self.assertFalse(result.refused)
        self.assertEqual(result.verdict, "FAIL")
        self.assertEqual(result.iterations, 2)

    def test_explicit_refusals_include_zero_and_small_baselines(self):
        for reason in sweep.REFUSALS:
            for baseline in (0.0, 0.1, 2.0):
                with self.subTest(reason=reason, baseline=baseline):
                    result = self.run_report(report(reason, 0, baseline, baseline))
                    self.assertEqual(result.verdict, "refused")
                    self.assertEqual(result.iterations, 0)

    def test_legacy_refusal_warning_is_explicit_provenance(self):
        text = report().split("AutofocusSharpnessReportOp: termination")[0]
        text += "autofocus REFUSED (refused_scatterers); image returned uncorrected\n"
        self.assertEqual(self.run_report(text).termination, "refused_scatterers")

    def test_missing_score_or_provenance_is_not_a_result(self):
        self.assertIsNone(self.run_report(""))
        self.assertIsNone(self.run_report(report().split("termination")[0]))

    def test_nonfinite_score_or_unknown_termination_is_not_a_result(self):
        self.assertIsNone(self.run_report(report(residual="1e999")))
        self.assertIsNone(self.run_report(report(reason="unknown")))

    def test_last_image_supplies_score_and_status(self):
        result = self.run_report(report() + report("budget_exhausted", 8, 0.2))
        self.assertEqual(result.residual_rad, 0.2)
        self.assertEqual(result.iterations, 8)

    def test_timeout_is_not_a_result_and_removes_temp_config(self):
        configs = []

        def timeout(args, **_kwargs):
            configs.append(Path(args[1]))
            raise subprocess.TimeoutExpired(args, 1800)

        with (
            mock.patch.object(sweep.subprocess, "run", side_effect=timeout),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertIsNone(sweep.run_one(Path("autofocus"), {}, "scene", "case"))
        self.assertFalse(configs[0].exists())

    def test_incomplete_campaign_continues_but_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            binary, config = root / "autofocus", root / "scene.yaml"
            binary.touch()
            config.write_text("{}\n")
            result = sweep.Result("scene", "good", 2.0, 0.1, 2.0, 2, "converged")
            with (
                mock.patch(
                    "sys.argv", ["sweep.py", "--binary", str(binary), "--base", str(config)]
                ),
                mock.patch.dict(
                    sweep.GRIDS, {"shapes": lambda: [sweep.Case("bad", {}), sweep.Case("good", {})]}
                ),
                mock.patch.object(sweep, "run_one", side_effect=[None, result]) as run,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(sweep.main(), 1)
                self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
