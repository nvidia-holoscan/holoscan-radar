# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic statistical/tool checks; not deployment measurement evidence."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import digest, write_json
from frame_count import analyze, load_pairs, run_curve, summarize


class FrameCountTest(unittest.TestCase):
    def test_prefix_preserves_early_regime_and_does_not_extrapolate(self):
        runs = [[1.0] * 50 + [2.0] * 150]
        self.assertEqual(run_curve(runs, 50, [0.5]), [1.0])
        self.assertEqual(run_curve(runs, 200, [0.5]), [2.0])
        with self.assertRaises(ValueError):
            run_curve(runs, 500, [0.5])

    def test_equal_run_distribution_is_not_median_of_run_quantiles(self):
        runs = [[1.0] * 50, [1.0] * 50, [100.0] * 50]
        self.assertEqual(run_curve(runs, 50, [0.9]), [100.0])

    def test_scale_independent_effect_and_simultaneous_interval(self):
        pairs = [[[[scale] * 50], [[scale * 0.95] * 50]] for scale in range(1, 31)]
        rows = summarize(pairs, 50, [0.5, 0.99], 100)
        for row in rows:
            self.assertAlmostEqual(row["median_change_pct"], -5.0)
            for bound in row["simultaneous_ci95_pct"]:
                self.assertAlmostEqual(bound, -5.0)

    def test_frames_cannot_manufacture_pair_resolution(self):
        pairs = [[[[1.0] * 200], [[1.01] * 200]]] * 10
        for frames in (50, 200):
            row = summarize(pairs, frames, [0.99], 6240)[0]
            self.assertEqual(row["pairs"], 10)
            self.assertEqual(row["simultaneous_ci95_pct"], [None, None])

    def test_invalid_timings_rejected(self):
        for value in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                run_curve([[value]], 1, [0.5])

    def archive(self, folder):
        def save(name, value):
            path = folder / name
            write_json(path, value)
            return {"path": str(path), "sha256": digest(path)}

        pair = {}
        for side in ("before", "after"):
            rows = [
                {
                    "operator": "SyntheticOp",
                    "scenario_id": "static",
                    "size_id": "small",
                    "precision": precision,
                    "bucket": "compute",
                    "status": "ok",
                    "raw_steady_ms": [1.0] * 50,
                    "raw_steady_run_offsets": [0, 50],
                }
                for precision in ("float", "double")
            ]
            child = save(side + "-run.json", {"results": rows})
            pair[side] = save(side + ".json", {"isolation": {"children": [child]}})
        index = {
            "measurement_policy_version": 3,
            "sampling": {"frames": 50, "repeats": 1, "process_mode": "isolated"},
            "requested_pairs": 1,
            "pairs": [pair],
        }
        return index, save

    def test_hashes_reuse_and_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            index, save = self.archive(Path(directory))
            pairs, _, _ = load_pairs(save("valid.json", index))
            self.assertEqual(len(pairs["float"]), 1)
            duplicated = copy.deepcopy(index)
            duplicated["pairs"] *= 2
            duplicated["requested_pairs"] = 2
            with self.assertRaisesRegex(ValueError, "reused run"):
                load_pairs(save("duplicated.json", duplicated))
            index["requested_pairs"] = 2
            with self.assertRaisesRegex(ValueError, "complete isolated"):
                load_pairs(save("incomplete.json", index))
            (Path(directory) / "before-run.json").write_text("{}")
            index["requested_pairs"] = 1
            with self.assertRaises(ValueError):
                load_pairs(save("tampered.json", index))

    def test_family_and_missing_windows_reported_without_invention(self):
        with tempfile.TemporaryDirectory() as directory:
            index, save = self.archive(Path(directory))
            spec = {
                "quantiles": [0.5, 0.99],
                "frame_prefixes": [20, 50, 200, 500],
                "dataset_family_bound": 156,
                "datasets": [{"id": "synthetic", "pairs": save("pairs.json", index)}],
            }
            result = analyze(spec)
            self.assertEqual(result["family_size"], 156 * 2 * 4 * 2)
            self.assertEqual(result["unmeasured_frame_prefixes"], [200, 500])
            self.assertEqual(len(result["results"]), 8)
            self.assertFalse(result["results"][0]["unchanged_early_window_available"])
            spec["dataset_family_bound"] = 0
            with self.assertRaises(ValueError):
                analyze(spec)


if __name__ == "__main__":
    unittest.main()
