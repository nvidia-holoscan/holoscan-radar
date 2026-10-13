# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent semantic oracle and gate integration, including adversarial outputs."""

import base64
import copy
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection_reduction import VERSION, Arithmetic, Unsupported, validate_frame
from equivalence import compare_snapshots, validate_policy


def context(policy="medoid"):
    return {
        "labels": [2, 2, 2, -1],
        "points": [[0.0], [1.0], [8.0], [100.0]],
        "detection_idx": [[0, 0], [1, 0], [8, 0], [100, 0]],
        "representative": policy,
        "score_method": "weighted_sum",
        "feature_weights": [1.0],
    }


def tensor(rows, dtype):
    shape = [len(rows), len(rows[0])] if rows and isinstance(rows[0], list) else [len(rows)]
    flat = [x for row in rows for x in row] if len(shape) == 2 else rows
    fmt = {"float32": "f", "float64": "d", "int64": "q"}[dtype]
    return {
        "kind": "tensor",
        "sequence": 1,
        "capture_time_ns": 100,
        "clock_id": 1,
        "memory_kind": "device",
        "shape": shape,
        "dtype": dtype,
        "data_base64": base64.b64encode(struct.pack("<" + fmt * len(flat), *flat)).decode(),
    }


def export():
    cases = []
    for precision, dtype in (("float", "float32"), ("double", "float64")):
        c = context()
        cases.append(
            {
                "id": "fixture",
                "precision": precision,
                "coverage_gaps": [],
                "termination": "stopped",
                "diagnostics": [],
                "semantic_inputs": [
                    {
                        "parameters": {
                            k: c[k] for k in ("representative", "score_method", "feature_weights")
                        },
                        "inputs": {
                            k: tensor(c[k], dtype if k == "points" else "int64")
                            for k in ("points", "labels", "detection_idx")
                        },
                    }
                ],
                "outputs": {
                    "rep_idx": {"wired": True, "emissions": [tensor([[1, 0]], "int64")]},
                    "reduced_points": {"wired": True, "emissions": [tensor([[1.0]], dtype)]},
                },
            }
        )
    return {
        "schema_version": 1,
        "byte_order": "little",
        "fixture_id": "semantic-test",
        "operator": "detection_reduction",
        "source_id": "test",
        "cases": cases,
    }


def policies():
    return {
        port: {
            precision: {"mode": VERSION, "rationale": "input-derived semantic policy"}
            for precision in ("float", "double")
        }
        for port in ("rep_idx", "reduced_points")
    }


class SemanticTests(unittest.TestCase):
    def test_separated_winner_and_gather(self):
        for precision in ("float", "double"):
            r = validate_frame(context(), precision, [[1, 0]], [[1.0]])
            self.assertEqual(r[0]["admissible_rows"], [1])
            with self.assertRaisesRegex(ValueError, "admissible"):
                validate_frame(context(), precision, [[8, 0]], [[8.0]])
            with self.assertRaisesRegex(ValueError, "gathered"):
                validate_frame(context(), precision, [[1, 0]], [[0.0]])

    def test_near_tie_is_not_allclose(self):
        c = context()
        c["labels"] = [0, 0, 0]
        c["points"] = [[-1.0], [1.0], [0.00000003]]
        c["detection_idx"] = [[0, 0], [1, 0], [2, 0]]
        # Middle point is an unequivocal winner despite symmetric outer rows.
        with self.assertRaises(ValueError):
            validate_frame(c, "float", [[0, 0]], [[-1.0]])
        c["labels"] = [0, 0]
        c["points"] = c["points"][:2]
        c["detection_idx"] = c["detection_idx"][:2]
        validate_frame(c, "float", [[0, 0]], [[-1.0]])
        with self.assertRaises(ValueError):
            validate_frame(c, "float", [[1, 0]], [[1.0]])
        c["labels"] = [0, 0, 0, 0]
        c["points"] = [[-1.0], [1.0], [-100.0], [100.0000152587890625]]
        c["detection_idx"] = [[0, 0], [1, 0], [2, 0], [3, 0]]
        validate_frame(c, "float", [[0, 0]], [[-1.0]])
        validate_frame(c, "float", [[1, 0]], [[1.0]])

    def test_precision_and_preserved_index_columns(self):
        c = context()
        c["labels"] = [0, 0, 0, 0]
        c["points"] = [[-1.0], [1.0], [-100.0], [100.0000152587890625]]
        c["detection_idx"] = [[0, 0], [1, 0], [2, 0], [3, 0]]
        self.assertEqual(validate_frame(c, "float", [[0, 0]])[0]["admissible_rows"], [0, 1])
        with self.assertRaises(ValueError):
            validate_frame(c, "double", [[0, 0]])
        c = context("weighted_mean")
        c["feature_weights"] = []
        c["labels"] = [0, 0, 0]
        c["points"] = [[1.0], [2.0], [3.0]]
        c["detection_idx"] = [[0, 0, 0], [0, 0, 100], [0, 0, 101]]
        validate_frame(c, "double", [[0, 0, 100]], [[2.0]])
        with self.assertRaises(ValueError):
            validate_frame(c, "double", [[0, 0, 0]], [[2.0]])

    def test_identical_location_tie_requires_first_row(self):
        c = context()
        c["points"] = [[1.0], [1.0], [1.0], [100.0]]
        with self.assertRaisesRegex(ValueError, "admissible"):
            validate_frame(c, "float", [[1, 0]], [[1.0]])

    def test_noise_and_cluster_order(self):
        with self.assertRaisesRegex(ValueError, "member"):
            validate_frame(context(), "float", [[100, 0]], [[100.0]])
        c = context()
        c["labels"] = [5, 2, 5, -1]
        with self.assertRaises(ValueError):
            validate_frame(c, "double", [[0, 0], [1, 0]], [[0.0], [1.0]])

    def test_weighted_uniform_and_nonuniform_means(self):
        for precision in ("float", "double"):
            c = context("weighted_mean")
            c["feature_weights"] = []
            validate_frame(c, precision, [[1, 0]], [[3.0]])
            with self.assertRaisesRegex(ValueError, "weighted mean"):
                validate_frame(c, precision, [[1, 0]], [[3.01]])
            c["feature_weights"] = [1.0]
            validate_frame(c, precision, [[8, 0]], [[65 / 9]])
            for method in ("weighted_max", "weighted_l2"):
                c["score_method"] = method
                validate_frame(c, precision, [[8, 0]], [[65 / 9]])

    def test_first_exact_gather_and_unwired_features(self):
        validate_frame(context("first"), "float", [[0, 0]], None)
        with self.assertRaises(ValueError):
            validate_frame(context("first"), "float", [[1, 0]], None)

    def test_invalid_and_unresolved_scopes(self):
        c = context("weighted_mean")
        c["score_method"] = "custom"
        with self.assertRaises(Unsupported):
            validate_frame(c, "float", [[1, 0]], None)
        c["feature_weights"] = [-1.0]
        c["score_method"] = "weighted_sum"
        with self.assertRaises(Unsupported):
            validate_frame(c, "float", [[1, 0]], None)

    def test_policy_bounds_are_not_candidate_configurable(self):
        with self.assertRaises(ValueError):
            validate_policy({"mode": VERSION, "rationale": "test", "rtol": 0.1})

    def test_integration_and_metadata(self):
        a = export()
        b = copy.deepcopy(a)
        self.assertTrue(compare_snapshots(a, b, policies())["passed"])
        b["cases"][0]["outputs"]["rep_idx"]["emissions"][0]["clock_id"] = 2
        self.assertFalse(compare_snapshots(a, b, policies())["passed"])

    def test_coupled_alternative_member_passes_only_semantic_policy(self):
        a = export()
        c = context()
        c["labels"] = [0, 0, 0, 0]
        c["points"] = [[-1.0], [1.0], [-100.0], [100.0000152587890625]]
        c["detection_idx"] = [[0, 0], [1, 0], [2, 0], [3, 0]]
        witness = a["cases"][0]["semantic_inputs"][0]
        witness["inputs"] = {
            k: tensor(c[k], "float32" if k == "points" else "int64")
            for k in ("labels", "points", "detection_idx")
        }
        a["cases"][0]["outputs"]["rep_idx"]["emissions"] = [tensor([[0, 0]], "int64")]
        a["cases"][0]["outputs"]["reduced_points"]["emissions"] = [tensor([[-1.0]], "float32")]
        b = copy.deepcopy(a)
        b["cases"][0]["outputs"]["rep_idx"]["emissions"] = [tensor([[1, 0]], "int64")]
        b["cases"][0]["outputs"]["reduced_points"]["emissions"] = [tensor([[1.0]], "float32")]
        self.assertTrue(compare_snapshots(a, b, policies())["passed"])
        exact = {port: {p: {"mode": "exact"} for p in ("float", "double")} for port in policies()}
        self.assertFalse(compare_snapshots(a, b, exact)["passed"])
        b["cases"][0]["outputs"]["reduced_points"]["emissions"] = [tensor([[-1.0]], "float32")]
        self.assertFalse(compare_snapshots(a, b, policies())["passed"])

    def test_unwired_optional_output(self):
        a = export()
        for c in a["cases"]:
            c["outputs"]["reduced_points"] = {"wired": False, "emissions": []}
        self.assertTrue(compare_snapshots(a, copy.deepcopy(a), policies())["passed"])

    def test_missing_witness_and_changed_inputs(self):
        a = export()
        b = copy.deepcopy(a)
        del a["cases"][0]["semantic_inputs"]
        r = compare_snapshots(a, b, policies())
        self.assertFalse(r["passed"])
        self.assertEqual(r["failures"][0]["classification"], "blocked")
        a = export()
        b = copy.deepcopy(a)
        b["cases"][0]["semantic_inputs"][0]["parameters"]["feature_weights"] = [2.0]
        self.assertFalse(compare_snapshots(a, b, policies())["passed"])

    def test_bad_baseline_is_rejected_too(self):
        a = export()
        a["cases"][0]["outputs"]["reduced_points"]["emissions"] = [tensor([[7.0]], "float32")]
        self.assertFalse(compare_snapshots(a, copy.deepcopy(a), policies())["passed"])

    def test_gamma_bound_covers_cancellation_and_scale(self):
        from fractions import Fraction

        ar = Arithmetic("float")
        lo, hi = ar.sum([(Fraction(x), Fraction(x)) for x in (1e8, 1, -1e8)])
        self.assertLessEqual(lo, 0)
        self.assertGreaterEqual(hi, 1)


if __name__ == "__main__":
    unittest.main()
