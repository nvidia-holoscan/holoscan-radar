# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable host-tool invariants; synthetic data never counts as GPU evidence."""

import base64
import copy
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import checkpoint, digest, file_state, read_json, restore, write_json
from equivalence import compare_snapshots
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
)
from timing import compare_pairs, run_pairs


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
            "pairs": 2,
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
        for pair in range(2):
            record = {}
            for side, value in (("before", before), ("after", after)):
                latency = value[pair] if isinstance(value, list) else value
                path = self.json_file(f"{pair}-{side}.json", perf_report(latency, side))
                record[side] = {"path": path, "sha256": digest(path)}
            pairs.append(record)
        return builds | {"pairs": pairs, "sampling": config()["sampling"]}

    def compare(self, index):
        return compare_pairs(index, expected_pairs=2)

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
        for mutation in ("commit", "hash", "cases", "platform", "sampling", "missing"):
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
            compare_pairs(index)
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
        path = self.json_file("session.json", session())
        self.assertEqual(load_session(path)["config"]["max_candidates"], DEFAULTS["max_candidates"])
        data = read_json(path)
        data["config"]["max_candidates"] = 100
        write_json(path, data)
        with self.assertRaisesRegex(ValueError, "confirmation"):
            load_session(path)
        data = session()
        data["candidates"] = [{"kept": False}, {"kept": False}]
        self.assertEqual(stop_reason(data), "consecutive unsuccessful limit")

    def test_unrelated_gates_cannot_certify_candidate(self):
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
            add_candidate(session(), record)

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
        context = {
            "candidate_id": candidate_id,
            "confirmed_config_sha256": state["confirmed_config_sha256"],
            "before_source_id": "before",
            "after_source_id": "after",
            "affected_operators": ["test"],
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
        pairs = self.json_file(prefix + "pairs.json", self.timing_index())
        settings = self.json_file(prefix + "config.json", state["config"])
        speed = str(self.root / (prefix + "speed.json"))
        self.assertEqual(
            main(
                [
                    "compare-timing",
                    "--pairs",
                    pairs,
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
        for index in range(2):
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
                command=["perf-test", side], artifacts={artifact: digest(artifact)}
            )

        def run(argv, **kwargs):
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
        self.assertTrue(self.compare(result)["passed"])
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
        self.assertTrue(compare_pairs(index, expected_pairs=2, require_improvement=False)["passed"])
        index = self.timing_index(after=12.0)
        self.assertFalse(
            compare_pairs(index, expected_pairs=2, require_improvement=False)["passed"]
        )

    def test_aggregate_accepts_complete_shared_consumer_evidence(self):
        state = session()
        state["config"]["consumer_policies"] = {"consumer": state["config"]["policies"]}
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
                path = self.json_file(f"consumer-{pair_number}-{side}.json", report)
                pair[side] = {"path": path, "sha256": digest(path)}
        pairs = self.json_file("consumer-pairs.json", index)
        settings = self.json_file("shared-config.json", state["config"])
        speed = str(self.root / "consumer-speed.json")
        self.assertEqual(
            main(
                [
                    "compare-timing",
                    "--pairs",
                    pairs,
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


if __name__ == "__main__":
    unittest.main()
