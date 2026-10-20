# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Distribution estimands and chronology checks; synthetic data are not qualification."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distributions import analyze, window_pairs, window_runs
from frame_count import summarize


class DistributionTest(unittest.TestCase):
    def pairs(self):
        return [[[[1.0] * 50 + [2.0] * 50 for _ in range(3)] for _ in range(2)]] * 30

    def spec(self):
        return {
            "windows": [
                {"id": "early", "start": 0, "frames": 50},
                {"id": "late", "start": 50, "frames": 50},
            ],
            "contrasts": [{"before": "early", "after": "late"}],
            "datasets": [{"id": "synthetic", "pairs": {"path": "unused"}}],
            "dataset_family_bound": 2,
            "prior_family_slots": 100,
        }

    def analyze(self, spec):
        sampling = {"frames": 100, "repeats": 3}
        data = {p: self.pairs() for p in ("float", "double")}
        with patch("distributions.load_pairs", return_value=(data, sampling, ("Op", "c", "s"))):
            return analyze(spec)

    def test_temporal_shift_preserves_independent_pair_count(self):
        spec = self.spec()
        rows = summarize(window_pairs(self.pairs(), *spec["windows"]), 50, [0.5, 0.99], 100)
        for row in rows:
            self.assertEqual(row["pairs"], 30)
            self.assertEqual(row["repeats_per_side"], 6)
            self.assertAlmostEqual(row["median_change_pct"], 100)
            for bound in row["simultaneous_ci95_pct"]:
                self.assertAlmostEqual(bound, 100)

    def test_between_sides_keeps_original_repeats_and_null(self):
        rows = summarize(window_pairs(self.pairs(), self.spec()["windows"][1]), 50, [0.9], 100)
        self.assertEqual(rows[0]["repeats_per_side"], 3)
        self.assertEqual(rows[0]["median_change_pct"], 0)

    def test_windows_are_exact_and_do_not_mutate_or_extrapolate(self):
        runs = [[1, 2, 3, 4]]
        self.assertEqual(window_runs(runs, 1, 2), [[2, 3]])
        self.assertEqual(runs, [[1, 2, 3, 4]])
        for start, frames in [(-1, 2), (0, 0), (3, 2), (True, 2)]:
            with self.assertRaises(ValueError):
                window_runs(runs, start, frames)

    def test_family_covers_windows_contrasts_precisions_and_prior(self):
        result = self.analyze(self.spec())
        self.assertEqual(result["family_size"], 100 + 2 * 3 * 3 * 2)
        self.assertEqual(len(result["results"]), 18)
        self.assertEqual(result["unavailable"], [])
        self.assertEqual(result["status"], "exploration_only_not_qualification_or_acceptance")

    def test_unavailable_windows_count_in_family_but_not_results(self):
        spec = self.spec()
        spec["windows"].append({"id": "unseen", "start": 0, "frames": 200})
        result = self.analyze(spec)
        self.assertEqual(result["family_size"], 100 + 2 * 4 * 3 * 2)
        self.assertEqual(result["unavailable"], [{"dataset": "synthetic", "comparison": "unseen"}])

    def test_invalid_selection_and_unequal_temporal_windows_rejected(self):
        original = self.spec()
        mutations = [
            ("windows", [original["windows"][0]] * 2),
            ("dataset_family_bound", 0),
            ("quantiles", [0.5, 0.5]),
            ("quantiles", [1]),
            ("datasets", original["datasets"] * 2),
            ("contrasts", original["contrasts"] * 2),
            ("contrasts", [{"before": "early", "after": "unknown"}]),
        ]
        for key, value in mutations:
            spec = copy.deepcopy(original)
            spec[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.analyze(spec)
        spec = copy.deepcopy(original)
        spec["windows"][1]["frames"] = 25
        with self.assertRaisesRegex(ValueError, "equal-length"):
            self.analyze(spec)

    def test_tail_change_is_not_hidden_by_median(self):
        pairs = [[[[1.0] * 100], [[1.0] * 98 + [10.0] * 2]]] * 30
        median, tail = summarize(pairs, 100, [0.5, 0.99], 100)
        self.assertEqual(median["median_change_pct"], 0)
        self.assertGreater(tail["median_change_pct"], 0)

    def test_directional_evidence_keeps_null_distinct_from_unresolved(self):
        result = self.analyze(self.spec())
        for row in result["results"]:
            expected = "detected_slowdown" if row["kind"] == "temporal" else "inconclusive"
            self.assertEqual(row["directional_evidence"], expected)
        spec = self.spec()
        spec["prior_family_slots"] = 2**40
        for row in self.analyze(spec)["results"]:
            self.assertEqual(row["directional_evidence"], "insufficient_interval_resolution")


if __name__ == "__main__":
    unittest.main()
