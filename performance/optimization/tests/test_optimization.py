# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable host-tool invariants; synthetic data never counts as GPU evidence."""

import base64
import contextlib
import copy
import io
import os
import sqlite3
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import checkpoint, digest, file_state, read_json, restore, write_json
from diagnostics import phase_breakdown, preflight
from equivalence import compare_snapshots
from qualify_native import check_inventory
from radar_perf import (
    DEFAULTS,
    GATES,
    add_candidate,
    config_hash,
    load_session,
    main,
    normalize_config,
    profile_artifact,
    render_report,
    stop_reason,
    timing_comparison,
)
from timing import budget_proposal, compare_pairs, median_interval, run_pairs
from workspace import audit_archive, scratch_begin, scratch_restore, verify_scratch_rebuild


def snapshots(value=1.0):
    cases = []
    for precision, dtype, fmt in (("float", "float32", "f"), ("double", "float64", "d")):
        emission = {
            "kind": "tensor",
            "sequence": 0,
            "capture_time_ns": 100,
            "clock_id": 1,
            "shape": [1],
            "dtype": dtype,
            "memory_kind": "device",
            "data_base64": base64.b64encode(struct.pack("<" + fmt, value)).decode(),
        }
        cases.append(
            {
                "id": "baseline",
                "precision": precision,
                "termination": "stopped",
                "diagnostics": [],
                "coverage_gaps": [],
                "outputs": {"out": {"wired": True, "emissions": [emission]}},
            }
        )
    return {
        "schema_version": 1,
        "byte_order": "little",
        "fixture_id": "test",
        "operator": "test",
        "source_id": "before",
        "cases": cases,
    }


def config():
    return normalize_config(
        {
            "operator": "test",
            "editable_scope": ["operators/test"],
            "policies": {"out": {"float": {"mode": "exact"}, "double": {"mode": "exact"}}},
            "pairs": 10,
            "timing_precisions": {"test": ["float"], "consumer": ["float"]},
            "verification_cases": {"test": ["baseline"]},
            "sampling": {"scope": "overrides", "warmup": 5, "frames": 50, "repeats": 3},
        }
    )


def session():
    settings = config()
    return {
        "config": settings,
        "confirmed_config_sha256": config_hash(settings),
        "started_at": time.time(),
        "candidates": [],
        "bugs": [],
        "coverage_gaps": [],
        "remaining_findings": [],
    }


def perf_report(latency, commit="base"):
    return {
        "schema_version": 3,
        "provenance": {"git_commit": commit},
        "platform": {"gpu": "Thor"},
        "run_settings": {
            "warmup_frames": 5,
            "measured_frames": 50,
            "repeats": 3,
            "scope": "overrides",
            "stopped_early": False,
            "scenarios_planned": 1,
            "scenarios_completed": 1,
        },
        "results": [
            {
                "operator": "test",
                "scenario_id": "baseline",
                "precision": "float",
                "size_id": "small",
                "bucket": "compute",
                "status": "ok",
                "median_ms": latency,
                "cold_ms": 100.0,
                "samples": 150,
            }
        ],
    }


class OptimizationToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def json_file(self, name, value):
        path = self.root / name
        write_json(path, value)
        return str(path)

    def timing_index(self, after=8.0, before=10.0):
        builds = {
            side: {
                "operator": "test",
                "operator_class": "test",
                "source_id": side,
                "git_commit": side,
                "recipe_sha256": "recipe",
                "sdk_image": "sdk5-pin",
                "build_settings": {"cuda": "13"},
            }
            for side in ("before", "after")
        }
        pairs = []
        for pair in range(10):
            record = {}
            for side, value in (("before", before), ("after", after)):
                latency = value[pair % len(value)] if isinstance(value, list) else value
                path = self.json_file(f"{pair}-{side}.json", perf_report(latency, side))
                record[side] = {"path": path, "sha256": digest(path)}
            pairs.append(record)
        return builds | {"pairs": pairs, "sampling": config()["sampling"]}

    def compare(self, index):
        return compare_pairs(index)

    def calibration_index(self, index, prefix="aa"):
        aa = copy.deepcopy(index)
        aa["after"] = copy.deepcopy(aa["before"])
        for number, pair in enumerate(aa["pairs"]):
            for side in ("before", "after"):
                original = read_json(index["pairs"][number]["before"]["path"])
                path = self.json_file(f"{prefix}-{number}-{side}.json", original)
                pair[side] = {"path": path, "sha256": digest(path)}
        return aa

    def test_exact_outputs_and_each_precision(self):
        policies = config()["policies"]
        self.assertTrue(compare_snapshots(snapshots(), snapshots(), policies)["passed"])
        self.assertFalse(compare_snapshots(snapshots(), snapshots(1.001), policies)["passed"])
        missing = snapshots()
        missing["cases"].pop()
        with self.assertRaisesRegex(ValueError, "both float and double"):
            compare_snapshots(missing, missing, policies)

    def test_per_port_tolerance_does_not_waive_temporal_metadata(self):
        policies = {
            "out": {
                p: {
                    "mode": "tolerance",
                    "atol": 0.01,
                    "rtol": 0.0,
                    "rationale": "independent test envelope",
                }
                for p in ("float", "double")
            }
        }
        self.assertTrue(compare_snapshots(snapshots(), snapshots(1.001), policies)["passed"])
        candidate = snapshots()
        candidate["cases"][0]["outputs"]["out"]["emissions"][0]["capture_time_ns"] = 101
        self.assertFalse(compare_snapshots(snapshots(), candidate, policies)["passed"])

    def test_absent_ports_missing_coordinates_and_coverage_fail(self):
        for mutation in ("ports", "coordinates", "coverage"):
            export = snapshots()
            for case in export["cases"]:
                if mutation == "ports":
                    case["outputs"] = {}
                elif mutation == "coordinates":
                    case["outputs"]["out"]["emissions"][0]["capture_time_ns"] = None
                else:
                    case["coverage_gaps"] = ["missing live fallback"]
            if mutation == "ports":
                with self.assertRaises(ValueError):
                    compare_snapshots(export, export, config()["policies"])
            else:
                self.assertFalse(compare_snapshots(export, export, config()["policies"])["passed"])

    def test_nonfinite_and_invalid_snapshot_payload(self):
        policies = {
            "out": {
                p: {"mode": "tolerance", "atol": 1.0, "rtol": 0.0, "rationale": "test"}
                for p in ("float", "double")
            }
        }
        self.assertFalse(
            compare_snapshots(snapshots(float("nan")), snapshots(float("nan")), policies)["passed"]
        )
        export = snapshots()
        export["cases"][0]["outputs"]["out"]["emissions"][0]["shape"] = [2]
        with self.assertRaisesRegex(ValueError, "byte count"):
            compare_snapshots(export, export, config()["policies"])

    def test_descriptor_change_and_unwired_output(self):
        baseline = snapshots()
        for case in baseline["cases"]:
            case["outputs"]["out"]["emissions"][0] = {
                "kind": "descriptor",
                "sequence": 0,
                "capture_time_ns": 1,
                "clock_id": 7,
                "value": {"positions": [[0, 0, 0]]},
            }
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0]["outputs"]["out"]["emissions"][0]["value"]["positions"][0][0] = 1
        self.assertFalse(compare_snapshots(baseline, candidate, config()["policies"])["passed"])
        for case in baseline["cases"]:
            case["outputs"]["out"] = {"wired": False, "emissions": []}
        self.assertTrue(compare_snapshots(baseline, baseline, config()["policies"])["passed"])

    def test_improvement_regression_and_noise(self):
        self.assertTrue(self.compare(self.timing_index())["passed"])
        self.assertFalse(self.compare(self.timing_index(after=12.0))["passed"])
        self.assertFalse(self.compare(self.timing_index(after=10.0))["passed"])
        self.assertFalse(self.compare(self.timing_index(after=[8.0, 12.0]))["passed"])

    def test_stale_commit_changed_hash_and_case_set(self):
        for mutation in ("commit", "hash", "cases", "platform", "sampling", "missing", "context"):
            index = self.timing_index()
            path = index["pairs"][0]["after"]["path"]
            report = read_json(path)
            if mutation == "commit":
                report["provenance"]["git_commit"] = "stale"
            elif mutation == "cases":
                report["results"][0]["scenario_id"] = "other"
            elif mutation == "platform":
                report["platform"]["gpu"] = "other"
            elif mutation == "sampling":
                report["run_settings"]["measured_frames"] = 3
            elif mutation == "missing":
                report["results"][0]["median_ms"] = None
            elif mutation == "context":
                report["results"][0]["override_context"] = "cached_configuration"
            else:
                report["results"][0]["median_ms"] = 1.0
            write_json(path, report)
            if mutation != "hash":
                index["pairs"][0]["after"]["sha256"] = digest(path)
            with self.assertRaises(ValueError, msg=mutation):
                self.compare(index)

    def test_pair_budget_and_confirmed_sampling(self):
        index = self.timing_index()
        with self.assertRaisesRegex(ValueError, "budget"):
            compare_pairs(index, expected_pairs=2)
        index["sampling"]["frames"] = 3
        with self.assertRaisesRegex(ValueError, "sampling"):
            self.compare(index)

    def test_candidate_rollback_preserves_kept_work_and_refuses_intervening_edits(self):
        path = self.root / "operator.cpp"
        path.write_text("accepted optimization\n")
        saved = checkpoint(self.root, ["operator.cpp", "new.hpp"])
        path.write_text("candidate optimization\n")
        (self.root / "new.hpp").write_text("candidate helper\n")
        sealed = {"root": str(self.root), "states": file_state(self.root, saved["states"])}
        path.write_text("user edit\n")
        with self.assertRaisesRegex(ValueError, "intervening"):
            restore(saved, sealed)
        path.write_text("candidate optimization\n")
        restore(saved, sealed)
        self.assertEqual(path.read_text(), "accepted optimization\n")
        self.assertFalse((self.root / "new.hpp").exists())

    def test_checkpoint_rejects_escape_symlink_and_corruption(self):
        with self.assertRaises(ValueError):
            checkpoint(self.root, ["../escape"])
        (self.root / "link").symlink_to(self.root / "target")
        with self.assertRaises(ValueError):
            checkpoint(self.root, ["link"])

    def test_confirmation_and_limit_integrity(self):
        self.assertEqual(DEFAULTS["max_candidates"], 20)
        self.assertEqual(DEFAULTS["max_unsuccessful"], 3)
        path = self.json_file("session.json", session())
        self.assertEqual(load_session(path)["config"]["max_candidates"], DEFAULTS["max_candidates"])
        data = read_json(path)
        data["config"]["max_candidates"] = 100
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "confirmation"):
            load_session(path)
        data = session()
        data["candidates"] = [{"kept": False}] * 3
        self.assertEqual(stop_reason(data), "consecutive unsuccessful limit")
        data["candidates"] = [{"kept": False}, {"kept": False}]
        self.assertIsNone(stop_reason(data))
        data["candidates"] = [{"kept": True}] * 20
        self.assertEqual(stop_reason(data), "candidate limit")

    def test_unrelated_gates_cannot_certify_candidate(self):
        state = session()
        self.valid_candidate(state)
        gate = self.json_file("old.json", {"passed": True, "evidence": "unrelated"})
        record = {
            "id": "one",
            "summary": "pool output",
            "files": ["operator.cpp"],
            "affected_operators": ["test"],
            "before_source_id": "before",
            "cases": ["baseline"],
            "after_source_id": "after",
            "gates": {name: gate for name in GATES},
        }
        with self.assertRaisesRegex(ValueError, "another candidate"):
            add_candidate(state, record)

    def test_cli_reports_preexisting_bug_without_stopping_independent_work(self):
        state = self.json_file("session.json", session())
        bug = self.json_file(
            "bug.json",
            {
                "title": "Suspected baseline error",
                "expected": "correct axis",
                "observed": "wrong axis",
                "baseline_reproduction": "yes",
                "affected_cases": ["axis"],
                "blocks_validation": True,
                "evidence": "baseline log",
            },
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["bug", "--session", state, "--record", bug]), 0)
        self.assertIsNone(stop_reason(read_json(state)))
        self.assertEqual(main(["finish", "--session", state, "--reason", "dry"]), 0)
        report = render_report(read_json(state))
        self.assertIn("Suspected baseline error", report)
        self.assertIn("wrong axis", report)
        self.assertIn("no cumulative speedup", report)

    def test_profiler_empty_stale_and_valid_exports(self):
        capture, table = self.root / "capture.nsys-rep", self.root / "api.csv"
        capture.write_bytes(b"capture")
        table.write_text("Name,Num Calls\ncudaMalloc,3\n")
        started = capture.stat().st_mtime - 1
        os.utime(table, (started + 2, started + 2))
        report = profile_artifact(capture, [table], started, "source")
        self.assertEqual(report["findings"][0]["name"], "cudaMalloc")
        with self.assertRaisesRegex(ValueError, "stale"):
            profile_artifact(capture, [table], started + 10, "source")
        table.write_text("Name,Num Calls\n")
        with self.assertRaises(ValueError):
            profile_artifact(capture, [table], started, "source")

    def test_invalid_json(self):
        for content in ('{"x": NaN}', '{"x": 1, "x": 2}'):
            path = self.root / "invalid.json"
            path.write_text(content)
            with self.assertRaises(ValueError):
                read_json(path)

    def valid_candidate(self, state, candidate_id="one"):
        plan = {
            "id": candidate_id,
            "mechanism": "reduce redundant launch",
            "before_source_id": "before",
            "target_cases": ["baseline"],
            "expected_gain_pct": 10.0,
        }
        plan_path = self.json_file(candidate_id + "-plan.json", plan)
        state.setdefault("plans", {})[candidate_id] = {
            "path": plan_path,
            "sha256": digest(plan_path),
            "declared_at": time.time(),
        }
        context = {
            "candidate_id": candidate_id,
            "confirmed_config_sha256": state["confirmed_config_sha256"],
            "before_source_id": "before",
            "after_source_id": "after",
            "affected_operators": ["test"],
            "candidate_plan_sha256": digest(plan_path),
        }
        prefix = candidate_id + "-"
        context_path = self.json_file(prefix + "context.json", context)
        before = self.json_file(prefix + "before.json", snapshots())
        after_export = snapshots()
        after_export["source_id"] = "after"
        after = self.json_file(prefix + "after.json", after_export)
        policies = self.json_file(prefix + "policies.json", state["config"]["policies"])
        equivalence = str(self.root / (prefix + "equivalence.json"))
        self.assertEqual(
            main(
                [
                    "compare-outputs",
                    "--before",
                    before,
                    "--after",
                    after,
                    "--policies",
                    policies,
                    "--context",
                    context_path,
                    "--out",
                    equivalence,
                ]
            ),
            0,
        )
        index = self.timing_index()
        pairs = self.json_file(prefix + "pairs.json", index)
        calibration = self.json_file(
            prefix + "calibration.json", self.calibration_index(index, prefix)
        )
        settings = self.json_file(prefix + "config.json", state["config"])
        speed = str(self.root / (prefix + "speed.json"))
        self.assertEqual(
            main(
                [
                    "compare-timing",
                    "--pairs",
                    pairs,
                    "--calibration",
                    calibration,
                    "--plan",
                    plan_path,
                    "--config",
                    settings,
                    "--context",
                    context_path,
                    "--out",
                    speed,
                ]
            ),
            0,
        )
        gates = {"equivalence": equivalence, "speed": speed}
        for kind in ("contract", "coverage", "tests"):
            gates[kind] = self.json_file(
                prefix + kind + ".json",
                {
                    "kind": kind,
                    "context": context,
                    "passed": True,
                    "operators": ["test"],
                    "evidence": {"reviewed": "synthetic behavioral test evidence"},
                },
            )
        return {
            "id": candidate_id,
            "summary": "synthetic candidate",
            "files": ["operator.cpp"],
            "cases": ["baseline"],
            "affected_operators": ["test"],
            "before_source_id": "before",
            "after_source_id": "after",
            "gates": gates,
        }

    def test_candidate_acceptance_recomputes_raw_evidence_and_binds_sources(self):
        state = session()
        record = self.valid_candidate(state)
        self.assertTrue(add_candidate(state, record)["kept"])
        self.assertIn("**kept**", render_report(state))
        state = session()
        record = self.valid_candidate(state, "two")
        gate = read_json(record["gates"]["equivalence"])
        before = gate["inputs"]["before"]
        gate["inputs"]["after"] = before
        write_json(record["gates"]["equivalence"], gate)
        with self.assertRaisesRegex(ValueError, "source identity"):
            add_candidate(state, record)

    def test_shared_consumer_evidence_and_case_inventory_are_required(self):
        state = session()
        record = self.valid_candidate(state)
        record["affected_operators"].append("consumer")
        for path in record["gates"].values():
            gate = read_json(path)
            gate["context"]["affected_operators"] = record["affected_operators"]
            gate["operators"] = record["affected_operators"]
            write_json(path, gate)
        with self.assertRaisesRegex(ValueError, "consumer evidence"):
            add_candidate(state, record)
        record["cases"] = []
        with self.assertRaisesRegex(ValueError, "cases"):
            add_candidate(state, record)

    def test_bug_blocks_actual_cases_even_when_candidate_declares_unrelated_cases(self):
        state = session()
        record = self.valid_candidate(state)
        record["cases"] = ["independent"]
        state["bugs"] = [
            {
                "title": "baseline defect",
                "expected": "correct",
                "observed": "incorrect",
                "baseline_reproduction": "yes",
                "affected_cases": ["baseline"],
                "blocks_validation": True,
                "evidence": "baseline log",
            }
        ]
        self.assertFalse(add_candidate(state, record)["kept"])
        self.assertIn("baseline defect", render_report(state))
        state = session()
        record = self.valid_candidate(state, "independent")
        state["bugs"] = [
            {"title": "other case", "affected_cases": ["unrelated"], "blocks_validation": True}
        ]
        self.assertTrue(add_candidate(state, record)["kept"])

    def test_single_fix_and_dry_session_limits(self):
        state = session()
        state["config"]["mode"] = "fix"
        state["confirmed_config_sha256"] = config_hash(state["config"])
        record = self.valid_candidate(state)
        add_candidate(state, record)
        self.assertEqual(stop_reason(state), "candidate limit")
        with self.assertRaisesRegex(ValueError, "exhausted"):
            add_candidate(state, record | {"id": "another"})
        state = session()
        for index in range(3):
            record = self.valid_candidate(state, f"rejected-{index}")
            gate = read_json(record["gates"]["tests"])
            gate["passed"] = False
            write_json(record["gates"]["tests"], gate)
            self.assertFalse(add_candidate(state, record)["kept"])
        self.assertEqual(stop_reason(state), "consecutive unsuccessful limit")

    def test_runner_alternates_order_and_preserves_raw_artifacts(self):
        manifests = self.timing_index()
        artifact = str(Path(__file__).resolve())
        order = []
        for side in ("before", "after"):
            manifests[side].update(
                command=["perf-test", side],
                artifacts={artifact: digest(artifact)},
                cwd=str(self.root),
            )

        def run(argv, **kwargs):
            self.assertEqual(kwargs["cwd"], str(self.root))
            side = argv[1]
            order.append(side)
            output = argv[argv.index("--output") + 1]
            write_json(output, perf_report(10.0 if side == "before" else 8.0, side))
            kwargs["stdout"].write(b"synthetic tool test\n")

        with patch("timing.subprocess.run", side_effect=run):
            result = run_pairs(
                manifests["before"],
                manifests["after"],
                self.root / "runs",
                pairs=2,
                sampling=config()["sampling"],
            )
        self.assertEqual(order, ["before", "after", "after", "before"])
        self.assertFalse(compare_pairs(result, expected_pairs=2)["passed"])
        with self.assertRaises(FileExistsError):
            run_pairs(manifests["before"], manifests["after"], self.root / "runs", pairs=2)

    def test_absolute_target_is_a_stop_condition(self):
        state = session()
        state["config"]["target_ms"] = 8.5
        state["confirmed_config_sha256"] = config_hash(state["config"])
        record = self.valid_candidate(state)
        self.assertTrue(add_candidate(state, record)["kept"])
        self.assertEqual(stop_reason(state), "target achieved")

    def test_consumer_only_needs_no_regression(self):
        index = self.timing_index(after=10.0)
        self.assertFalse(self.compare(index)["passed"])
        self.assertTrue(compare_pairs(index, require_improvement=False)["passed"])
        index = self.timing_index(after=12.0)
        self.assertFalse(compare_pairs(index, require_improvement=False)["passed"])

    def test_aggregate_accepts_complete_shared_consumer_evidence(self):
        state = session()
        state["config"]["consumer_policies"] = {"consumer": state["config"]["policies"]}
        state["config"]["target_cases"] = ["baseline"]
        state["config"]["verification_cases"]["consumer"] = ["baseline"]
        state["confirmed_config_sha256"] = config_hash(state["config"])
        record = self.valid_candidate(state)
        record["affected_operators"] = ["test", "consumer"]
        for path in record["gates"].values():
            gate = read_json(path)
            gate["context"]["affected_operators"] = record["affected_operators"]
            gate["operators"] = record["affected_operators"]
            write_json(path, gate)
        context = read_json(record["gates"]["equivalence"])["context"]
        context_path = self.json_file("shared-context.json", context)
        before, after = snapshots(), snapshots()
        before["operator"] = after["operator"] = "consumer"
        after["source_id"] = "after"
        before_path = self.json_file("consumer-before.json", before)
        after_path = self.json_file("consumer-after.json", after)
        policies = self.json_file("consumer-policies.json", state["config"]["policies"])
        equivalence = str(self.root / "consumer-equivalence.json")
        self.assertEqual(
            main(
                [
                    "compare-outputs",
                    "--before",
                    before_path,
                    "--after",
                    after_path,
                    "--policies",
                    policies,
                    "--context",
                    context_path,
                    "--out",
                    equivalence,
                ]
            ),
            0,
        )
        index = read_json(self.root / "one-pairs.json")
        for side in ("before", "after"):
            index[side]["operator"] = "consumer"
            index[side]["operator_class"] = "consumer"
        for pair_number, pair in enumerate(index["pairs"]):
            for side in ("before", "after"):
                report = perf_report(10.0, side)
                report["results"][0]["operator"] = "consumer"
                report["results"][0]["scenario_id"] = "consumer-only"
                path = self.json_file(f"consumer-{pair_number}-{side}.json", report)
                pair[side] = {"path": path, "sha256": digest(path)}
        pairs = self.json_file("consumer-pairs.json", index)
        calibration = self.json_file(
            "consumer-aa.json", self.calibration_index(index, "consumer-aa")
        )
        settings = self.json_file("shared-config.json", state["config"])
        speed = str(self.root / "consumer-speed.json")
        self.assertEqual(
            main(
                [
                    "compare-timing",
                    "--pairs",
                    pairs,
                    "--calibration",
                    calibration,
                    "--plan",
                    state["plans"]["one"]["path"],
                    "--config",
                    settings,
                    "--context",
                    context_path,
                    "--out",
                    speed,
                ]
            ),
            0,
        )
        for kind, consumer in (("equivalence", equivalence), ("speed", speed)):
            out = str(self.root / ("shared-" + kind + ".json"))
            self.assertEqual(
                main(
                    [
                        "aggregate",
                        "--kind",
                        kind,
                        "--context",
                        context_path,
                        "--checks",
                        record["gates"][kind],
                        consumer,
                        "--out",
                        out,
                    ]
                ),
                0,
            )
            record["gates"][kind] = out
        self.assertTrue(add_candidate(state, record)["kept"])

    def test_familywise_intervals_and_practical_gain(self):
        self.assertEqual(median_interval([20.0] * 10, 136), [None, None])
        self.assertEqual(median_interval([20.0] * 16, 136), [20.0, 20.0])
        index = self.timing_index()
        self.assertFalse(compare_pairs(index, min_effect_pct=25.0)["passed"])
        value = compare_pairs(index, family_multiplier=100)
        self.assertFalse(value["passed"])
        self.assertGreater(value["minimum_resolvable_pairs"], 10)
        self.assertTrue(all(r["verdict"] == "inconclusive" for r in value["results"]))
        # Alternating zero-centered noise must not establish a directional effect.
        index = self.timing_index(after=[9.9, 10.1])
        self.assertFalse(compare_pairs(index)["passed"])

    def test_calibration_blocks_bimodality_bias_and_stale_provenance(self):
        index = self.timing_index(before=[10.0, 20.0], after=[8.0, 16.0])
        aa = self.calibration_index(index)
        result = timing_comparison(config(), index, aa)
        self.assertEqual(result["status"], "validation_blocked")
        self.assertFalse(result["passed"])
        self.assertEqual(result["calibration"]["results"][1]["aa_max_delta_pct"], 0)
        aa["before"]["source_id"] = "old"
        with self.assertRaisesRegex(ValueError, "stale calibration"):
            timing_comparison(config(), index, aa)
        index = self.timing_index()
        aa = self.calibration_index(index, "biased")
        for pair in aa["pairs"]:
            report = read_json(pair["after"]["path"])
            report["results"][0]["median_ms"] = 8.0
            write_json(pair["after"]["path"], report)
            pair["after"]["sha256"] = digest(pair["after"]["path"])
        generous = config() | {"max_aa_delta_pct": 100.0, "max_baseline_spread_pct": 100.0}
        self.assertFalse(timing_comparison(generous, index, aa)["passed"])

    def test_calibration_hash_and_precision_coverage(self):
        index = self.timing_index()
        aa = self.calibration_index(index)
        settings = config() | {"timing_precisions": {}}
        with self.assertRaisesRegex(ValueError, "precision coverage"):
            timing_comparison(settings, index, aa)
        Path(aa["pairs"][0]["before"]["path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed after measurement"):
            timing_comparison(config(), index, aa)

    def test_predeclared_plan_is_immutable_and_cannot_change_targets(self):
        state = session()
        record = self.valid_candidate(state)
        path = state["plans"]["one"]["path"]
        plan = read_json(path)
        plan["target_cases"] = ["cherry-picked"]
        write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "unchanged predeclared"):
            add_candidate(state, record)

    @patch("campaign.validate_admission")
    def test_plan_command_checks_baseline_and_refuses_redeclaration(self, _admission):
        state = session()
        state["config"] = normalize_config(state["config"] | {"measurement_policy_version": 3})
        state["config"]["target_cases"] = ["baseline"]
        state["confirmed_config_sha256"] = config_hash(state["config"])
        state_path = self.json_file("plan-session.json", state)
        plan_path = self.json_file(
            "declaration.json",
            {
                "id": "one",
                "before_source_id": "before",
                "mechanism": "fewer launches",
                "target_cases": ["baseline"],
                "expected_gain_pct": 5.0,
            },
        )
        argv = ["plan", "--session", state_path, "--record", plan_path, "--root", str(self.root)]
        with patch("radar_perf.fingerprint", return_value="edited"):
            with self.assertRaisesRegex(ValueError, "before editing"):
                main(argv)
        with patch("radar_perf.fingerprint", return_value="before"):
            self.assertEqual(main(argv), 0)
            with self.assertRaisesRegex(ValueError, "already declared"):
                main(argv)
        self.assertEqual(read_json(state_path)["plans"]["one"]["sha256"], digest(plan_path))

    def test_underpowered_consumer_cannot_pass_regression_protection(self):
        index = self.timing_index(after=10.0)
        result = compare_pairs(index, require_improvement=False, family_multiplier=100)
        self.assertFalse(result["passed"])
        self.assertTrue(result["insufficient_pair_budget"])
        state = session()
        record = self.valid_candidate(state, "missing-plan")
        state["plans"] = {}
        with self.assertRaisesRegex(ValueError, "predeclared plan"):
            add_candidate(state, record)

    def test_blocked_measurement_is_not_a_failure_streak(self):
        state = session()
        state["candidates"] = [
            {"kept": False},
            {"kept": False},
            {"kept": False, "outcome": "validation_blocked"},
        ]
        self.assertIsNone(stop_reason(state))
        state["candidates"].append({"kept": False})
        self.assertEqual(stop_reason(state), "consecutive unsuccessful limit")

    def test_native_inventory_checks_ids_not_counts(self):
        inventory = {"test": ["baseline"]}
        export = snapshots()
        for case in export["cases"]:
            case["termination"] = "stopped"
            case["outputs"] = {}
        check_inventory(export, inventory)
        export["cases"][0]["id"] = "wrong-case-same-count"
        with self.assertRaisesRegex(AssertionError, "inventory changed"):
            check_inventory(export, inventory)

    def test_scoped_sqlite_phases_clip_and_union_cuda_calls(self):
        path = self.root / "capture.sqlite"
        with sqlite3.connect(path) as db:
            db.executescript(
                "CREATE TABLE StringIds(id INTEGER, value TEXT);"
                "INSERT INTO StringIds VALUES(1, 'op:compute'), (2, 'cudaMallocAsync');"
                "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT,"
                "textId INTEGER, globalTid INTEGER);"
                "INSERT INTO NVTX_EVENTS VALUES(1000, 11000, NULL, 1, 7);"
                "INSERT INTO NVTX_EVENTS VALUES(6000, NULL, 'op.fft', NULL, 7);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER,"
                "nameId INTEGER, globalTid INTEGER);"
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(0, 4000, 2, 7);"
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(2000, 6000, 2, 7);"
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(6000, 10000, 2, 99);"
            )
        started = path.stat().st_mtime - 1
        result = phase_breakdown(path, started, "source")
        self.assertEqual(result["compute_median_us"], 10)
        self.assertEqual(result["cuda_api_union_median_us"], 5)
        self.assertEqual(result["outside_cuda_api_median_us"], 5)
        self.assertEqual(result["phases"]["start -> op.fft"]["median_us"], 5)
        with self.assertRaisesRegex(ValueError, "stale"):
            phase_breakdown(path, started + 10, "source")
        with sqlite3.connect(path) as db:
            db.execute("INSERT INTO NVTX_EVENTS VALUES(1000, 11000, NULL, 1, 8)")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            phase_breakdown(path, started, "source")
        self.assertEqual(phase_breakdown(path, started, "source", global_tid=7)["frames"], 1)

    def test_environment_preflight_reports_missing_tools_without_provisioning(self):
        with (
            patch("diagnostics.shutil.which", return_value=None),
            patch("diagnostics.subprocess.run") as runner,
        ):
            result = preflight()
        runner.assert_not_called()
        self.assertIn("ncu unavailable", result["limitations"])

    def test_robust_calibration_retains_extreme_stalls(self):
        index = self.timing_index(before=[100.0] + [10.0] * 9, after=[100.0] + [10.0] * 9)
        index["after"] = copy.deepcopy(index["before"])
        for pair in index["pairs"]:
            pair["after"] = copy.deepcopy(pair["before"])
        value = compare_pairs(index, calibration=True)
        self.assertTrue(value["passed"])
        steady = next(r for r in value["results"] if r["metric"] == "median_ms")
        self.assertEqual(steady["baseline_iqr_pct"], 0)
        self.assertEqual(steady["stall_count"], 2)
        self.assertGreater(steady["baseline_spread_pct"], 500)

    def test_metric_specific_calibration_and_budget(self):
        index = self.timing_index(before=[10.0, 20.0])
        aa = self.calibration_index(index)
        self.assertFalse(compare_pairs(aa, calibration=True)["passed"])
        self.assertTrue(
            compare_pairs(
                aa, calibration=True, metric_policies={"median_ms": {"max_baseline_iqr_pct": 100.0}}
            )["passed"]
        )
        proposal = budget_proposal(perf_report(10), ["median_ms", "cold_ms"], "median_ms", 100)
        self.assertEqual(proposal["family_size"], 200)
        self.assertGreaterEqual(proposal["proposed_pairs"], 20)
        self.assertGreater(proposal["proposed_pairs"], proposal["minimum_resolvable_pairs"])
        incomplete = perf_report(10)
        incomplete["run_settings"]["stopped_early"] = True
        with self.assertRaisesRegex(ValueError, "complete successful"):
            budget_proposal(incomplete, ["median_ms"], "median_ms")

    def test_isolated_runner_replays_raw_repeat_reports(self):
        builds = self.timing_index()
        for side in ("before", "after"):
            builds[side].update(
                command=["fake", side],
                cwd=str(self.root),
                artifacts={str(Path(__file__)): digest(__file__)},
            )
        calls = []

        def run(argv, **kwargs):
            if "--list-cases" in argv:
                return type("Completed", (), {"stdout": "baseline\tsmall\n"})()
            calls.append(argv)
            self.assertEqual(argv[argv.index("--repeats") + 1], "1")
            self.assertEqual(argv[argv.index("--cases") + 1], "baseline")
            report = perf_report(10 if argv[1] == "before" else 8, argv[1])
            report["run_settings"]["repeats"] = 1
            write_json(argv[argv.index("--output") + 1], report)

        sampling = config()["sampling"] | {"process_mode": "isolated", "repeats": 2}
        with patch("timing.subprocess.run", side_effect=run):
            result = run_pairs(
                builds["before"],
                builds["after"],
                self.root / "isolated",
                pairs=2,
                sampling=sampling,
            )
        self.assertEqual(len(calls), 8)
        compare_pairs(result, expected_pairs=2)
        aggregate = read_json(result["pairs"][0]["before"]["path"])
        child = aggregate["isolation"]["children"][0]["path"]
        write_json(child, {})
        with self.assertRaisesRegex(ValueError, "isolated raw report changed"):
            compare_pairs(result, expected_pairs=2)

    def test_clock_configuration_changes_block_comparison(self):
        index = self.timing_index()
        index["clocks"] = {
            "expected_configuration": {"governor": "fixed"},
            "snapshots": [{"configuration": {"governor": "dynamic"}}],
        }
        with self.assertRaisesRegex(ValueError, "clock/governor"):
            compare_pairs(index)
        with self.assertRaisesRegex(ValueError, "confirmed clock"):
            compare_pairs(self.timing_index(), expected_clock_configuration={"governor": "fixed"})

    def test_archive_mapping_preserves_hash_bound_json(self):
        moved = self.root / "moved"
        moved.mkdir()
        child = moved / "gate.json"
        write_json(child, {"evidence": "retained"})
        old = str(self.root / "old")
        ledger = self.json_file(
            "archive.json", {"gate": {"path": old + "/gate.json", "sha256": digest(child)}}
        )
        identity = digest(ledger)
        self.assertFalse(audit_archive(ledger, {})["passed"])
        self.assertTrue(audit_archive(ledger, {old: str(moved)})["passed"])
        self.assertEqual(digest(ledger), identity)
        child.write_text("changed")
        self.assertFalse(audit_archive(ledger, {old: str(moved)})["passed"])

    def test_scratch_restore_preserves_best_and_requires_rebuild(self):
        source = self.root / "operator.cpp"
        source.write_text("accepted work")
        target = self.root / "program"
        target.write_bytes(b"build")
        best = digest(source)
        with patch("workspace.fingerprint", side_effect=lambda root: digest(source)):
            saved = scratch_begin(self.root, ["operator.cpp"], [str(target)], best)
            source.write_text("accepted work plus instrumentation")
            sealed = {"root": saved["root"], "states": file_state(self.root, ["operator.cpp"])}
            receipt = scratch_restore(saved, sealed)
            self.assertEqual(source.read_text(), "accepted work")
            manifest = self.timing_index()["before"] | {
                "source_id": best,
                "cwd": str(self.root),
                "command": ["fake"],
                "artifacts": {str(target): digest(target)},
            }
            with self.assertRaisesRegex(ValueError, "not rebuilt"):
                verify_scratch_rebuild(receipt, [manifest])
            os.utime(target, (receipt["restored_at"] + 1, receipt["restored_at"] + 1))
            self.assertTrue(verify_scratch_rebuild(receipt, [manifest])["passed"])

    def test_explicit_phase_session_labels_remove_warmup_ambiguity(self):
        path = self.root / "sessions.sqlite"
        with sqlite3.connect(path) as db:
            db.executescript(
                "CREATE TABLE StringIds(id INTEGER,value TEXT);"
                "INSERT INTO StringIds VALUES(1,'op:compute'),(2,'cudaLaunchKernel');"
                "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);"
                "INSERT INTO NVTX_EVENTS VALUES(0,100,NULL,1,7),(200,300,NULL,1,8);"
                "INSERT INTO NVTX_EVENTS VALUES(150,350,'radar-perf:case:baseline:repeat=0',NULL,1);"
                "INSERT INTO NVTX_EVENTS VALUES(250,NULL,'out:emit',NULL,8);"
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,globalTid INTEGER);"
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(210,220,2,8);"
            )
        result = phase_breakdown(
            path, path.stat().st_mtime - 1, "source", case_id="baseline", timed_output="out"
        )
        self.assertEqual(result["frames"], 1)
        self.assertEqual(result["after_timed_emit_median_us"], 0.05)
        with self.assertRaisesRegex(ValueError, "session label"):
            phase_breakdown(path, path.stat().st_mtime - 1, "source", case_id="unknown")

    def test_old_policy_cannot_accept_new_candidate(self):
        state = session()
        state["config"].pop("measurement_policy_version")
        with self.assertRaisesRegex(ValueError, "freshly confirmed"):
            add_candidate(state, {})

    def test_directional_calibration_not_hidden_by_gain_threshold(self):
        index = self.timing_index()
        index["after"]["source_id"] = index["before"]["source_id"]
        result = compare_pairs(
            index,
            calibration=True,
            min_effect_pct=90,
            max_aa_delta_pct=100,
            max_baseline_spread_pct=100,
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
