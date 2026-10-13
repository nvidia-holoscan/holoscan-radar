# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host protocol invariants. Synthetic timings are never deployment evidence."""

import copy
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_optimization as legacy
from artifacts import digest, read_json, write_json
from campaign import (
    CorrectnessFailure,
    create,
    discover,
    identity,
    load,
    native_baseline,
    qualify_campaign,
    report,
    reuse_build,
    validate_admission,
    validate_consumer_baselines,
)
from measurement import derive_raw, device_lock, effective, execute, protocol
from qualification import checked, forecast, qualify, timing_table
from radar_perf import main, normalize_config
from test_optimization import config, perf_report, snapshots
from timing import aggregate_isolated, budget_proposal, compare_pairs, quantile, run_pairs


def raw_report(values, case="baseline"):
    value = perf_report(quantile(values, 0.5))
    value["run_settings"].update(repeats=1, measured_frames=len(values))
    row = value["results"][0]
    row.update(
        scenario_id=case,
        p90_ms=quantile(values, 0.9),
        p99_ms=quantile(values, 0.99),
        max_ms=max(values),
        samples=len(values),
        raw_steady_ms=values,
        raw_steady_run_offsets=[0, len(values)],
    )
    return value


class BenchmarkToolsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def settings(self):
        return normalize_config(config() | {"measurement_policy_version": 3, "pairs": 2})

    def build(self):
        script = self.root / "synthetic.py"
        script.write_text("""import json,sys
from pathlib import Path
args=sys.argv[1:]
if '--list-cases' in args:
 print('baseline\\tsmall');sys.exit(0)
def arg(key):return args[args.index(key)+1]
n=int(arg('--frames'))
report={'schema_version':3,'provenance':{'git_commit':'base'},'platform':{'gpu':'synthetic-host-fixture'},
'run_settings':{'warmup_frames':int(arg('--warmup')),'measured_frames':n,'repeats':1,'scope':arg('--scope'),
'stopped_early':False,'scenarios_planned':1,'scenarios_completed':1},
'results':[{'operator':'test','scenario_id':arg('--cases'),'precision':'float','size_id':'small','bucket':'compute',
'status':'ok','median_ms':10.,'p90_ms':10.,'p99_ms':10.,'cold_ms':100.,'max_ms':10.,'samples':n,
'raw_steady_ms':[10.]*n,'raw_steady_run_offsets':[0,n]}]}
Path(arg('--output')).write_text(json.dumps(report))
""")
        return {
            "operator": "test",
            "operator_class": "test",
            "source_id": "same",
            "git_commit": "base",
            "recipe_sha256": "recipe",
            "sdk_image": "sdk5-pin",
            "build_settings": {"cuda": "13"},
            "cwd": str(self.root),
            "command": [sys.executable, str(script)],
            "artifacts": {str(script): digest(script)},
        }

    def index(self, directory="pairs", sampling=None):
        build = self.build()
        with patch(
            "measurement.clock_state", return_value={"configuration": {}, "observations": {}}
        ):
            return run_pairs(
                build,
                build,
                self.root / directory,
                pairs=2,
                sampling=sampling or protocol({"repeats": 3}),
            )

    def test_sampling_overrides_and_early_floor(self):
        cfg = protocol(
            {
                "frames": 500,
                "repeats": 7,
                "case_overrides": {"baseline": {"frames": 50, "repeats": 3}},
            }
        )
        self.assertEqual(
            (effective(cfg, "baseline")["frames"], effective(cfg, "baseline")["repeats"]), (50, 3)
        )
        with self.assertRaises(ValueError):
            protocol({"case_overrides": {"baseline": {"frames": 49}}})
        with self.assertRaises(ValueError):
            protocol({"warmup": True})
        with self.assertRaises(ValueError):
            protocol({"unrecognized": 1})

    def test_early_window_uses_first_fifty_without_discard(self):
        values = [20.0] * 50 + [10.0] * 450
        report = derive_raw(raw_report(values), protocol({"frames": 500}))
        row = report["results"][0]
        self.assertEqual(row["early_median_ms"], 20.0)
        self.assertEqual(row["median_ms"], 10.0)
        self.assertEqual(row["frame_diagnostics"]["early_bias_pct"], 100.0)
        self.assertEqual(report["run_settings"]["warmup_frames"], 5)
        self.assertEqual(row["raw_steady_ms"], values)

    def test_raw_reconstruction_rejects_invented_summary_and_boundaries(self):
        report = raw_report([10.0] * 50)
        report["results"][0]["median_ms"] = 9.0
        with self.assertRaises(ValueError):
            derive_raw(report, protocol({}))
        report = raw_report([10.0] * 50)
        report["results"][0]["raw_steady_run_offsets"] = [1, 50]
        with self.assertRaises(ValueError):
            derive_raw(report, protocol({}))

    def test_median_default_and_mean_exploratory_aggregation(self):
        reports = [raw_report([value] * 50) for value in (1.0, 2.0, 9.0)]
        median = aggregate_isolated(reports, protocol({}), ["baseline"])
        mean = aggregate_isolated(reports, protocol({"aggregation": "mean"}), ["baseline"])
        self.assertEqual(median["results"][0]["early_median_ms"], 2.0)
        self.assertEqual(mean["results"][0]["early_median_ms"], 4.0)
        self.assertEqual(mean["results"][0]["max_ms"], 9.0)
        self.assertNotIn("raw_steady_ms", median["results"][0])

    def test_early_metrics_increase_family_budget(self):
        report = derive_raw(raw_report([10.0] * 50), protocol({}))
        base = budget_proposal(report, ["p90_ms", "p99_ms"], "median_ms", 2)
        early = budget_proposal(report, self.settings()["protected_metrics"], "median_ms", 2)
        self.assertGreater(early["family_size"], base["family_size"])
        self.assertEqual(early["family_size"], 14)

    def test_adjacent_order_resume_and_tamper_rejection(self):
        index = self.index()
        paths = list((self.root / "pairs").glob("000-*-000-*.process.json"))
        order = [p.name for p in sorted(paths, key=lambda p: read_json(p)["started_at"])]
        self.assertIn("before", order[0])
        self.assertIn("after", order[1])
        self.assertIn("after", order[2])
        self.assertIn("before", order[3])
        before_times = {str(p): read_json(p)["started_at"] for p in paths}
        self.index()
        self.assertEqual(before_times, {str(p): read_json(p)["started_at"] for p in paths})
        child = Path(
            read_json(index["pairs"][0]["before"]["path"])["isolation"]["children"][0]["path"]
        )
        child.write_text(child.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.index()

    def test_reused_observations_cannot_inflate_pair_budget(self):
        index = self.index()
        index["pairs"][1] = copy.deepcopy(index["pairs"][0])
        with self.assertRaisesRegex(ValueError, "repeated isolated measurement"):
            compare_pairs(index, expected_pairs=2)

    def test_unsealed_units_rejected(self):
        index = self.index()
        child = Path(
            read_json(index["pairs"][0]["before"]["path"])["isolation"]["children"][0]["path"]
        )
        child.with_suffix(".receipt.json").unlink()
        with self.assertRaisesRegex(ValueError, "unsealed"):
            self.index()

    def test_case_sampling_resolution_bound_to_aggregate(self):
        cfg = protocol(
            {
                "frames": 500,
                "repeats": 7,
                "case_overrides": {"baseline": {"frames": 50, "repeats": 3}},
            }
        )
        index = self.index(sampling=cfg)
        report = checked(index["pairs"][0]["before"])
        self.assertEqual(
            report["run_settings"]["sampling_by_case"]["baseline"], {"frames": 50, "repeats": 3}
        )
        result = compare_pairs(
            index,
            expected_pairs=2,
            protected_metrics=self.settings()["protected_metrics"],
            calibration=True,
        )
        self.assertTrue(result["insufficient_pair_budget"])
        self.assertIn(
            "insufficient_interval_resolution", [x["kind"] for x in result["blocker_reasons"]]
        )

    def test_interruption_cleanup_only_owned_process(self):
        receipt = self.root / "process.json"
        with self.assertRaises(subprocess.TimeoutExpired):
            execute(
                [sys.executable, "-c", "import time;time.sleep(10)"],
                self.root,
                self.root / "process.log",
                receipt,
                0.05,
            )
        self.assertEqual(read_json(receipt)["state"], "finished")
        self.assertLess(read_json(receipt)["returncode"], 0)

    def test_failed_launch_restores_signal_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(FileNotFoundError):
            execute(
                ["/nonexistent/operator-perf-fixture"],
                self.root,
                self.root / "failed.log",
                self.root / "failed.json",
                1,
            )
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_device_jobs_serialized_across_directories(self):
        with device_lock("synthetic-test-device"):
            with self.assertRaisesRegex(ValueError, "already in use"):
                with device_lock("synthetic-test-device"):
                    self.fail("concurrent job acquired same device")

    def test_positive_explicit_budget_required(self):
        for budget in (None, 0, -1, True):
            with self.assertRaises(ValueError):
                qualify({}, self.settings(), self.root, budget)

    def test_frozen_disjoint_validation_and_no_implicit_migration(self):
        settings = self.settings()
        settings["sampling"] = protocol({"repeats": 1})
        result = qualify(
            self.build(),
            settings,
            self.root / "qualify",
            60,
            {"frames": [50], "repeats": [1], "calibration_pairs": 2, "validation_pairs": 2},
        )
        self.assertFalse(result["passed"])
        frozen = checked(result["frozen"])
        calibration = checked(frozen["calibration_evidence"][0])
        validation = checked(result["validation_pairs"])
        self.assertTrue(
            set(x["before"]["path"] for x in calibration["pairs"]).isdisjoint(
                x["before"]["path"] for x in validation["pairs"]
            )
        )
        changed = copy.deepcopy(settings)
        changed["sampling"]["warmup"] = 6
        with self.assertRaisesRegex(ValueError, "intent changed"):
            qualify(self.build(), changed, self.root / "qualify", 60)
        with self.assertRaisesRegex(ValueError, "admission"):
            validate_admission(settings)

    def test_forecasts_account_for_precision_and_sampling(self):
        value = forecast(protocol({"frames": 500, "repeats": 7}), 10, case_count=2, precisions=2)
        self.assertEqual(value["processes"], 280)
        self.assertEqual(value["raw_frames"], 280000)
        self.assertGreater(value["estimated_seconds"], 280)

    def test_campaign_inventory_missing_harness_and_budget(self):
        root = Path(__file__).resolve().parents[3]
        inventory = discover(root)
        self.assertEqual(len(inventory), 32)
        self.assertEqual(sum(x["recipe"] is not None for x in inventory), 26)
        self.assertTrue(all(x["status"] == "blocked" for x in inventory if not x["recipe"]))
        state = {"operators": [], "config": {}}
        write_json(self.root / "campaign.json", state)
        with self.assertRaises(ValueError):
            qualify_campaign(self.root)

    def test_discovery_loads_yaml_recipes_and_detects_changes(self):
        root = Path(__file__).resolve().parents[3]
        with patch("campaign.preflight", return_value={}):
            state = create({"root": str(root), "operators": ["clutter_filter"]}, self.root)
        self.assertEqual(load(self.root)["operators"], state["operators"])
        state["operators"][0]["recipe"]["sha256"] = "stale"
        write_json(self.root / "campaign.json", state)
        with self.assertRaisesRegex(ValueError, "recipe hash changed"):
            load(self.root)

    def test_shared_consumer_source_invalidation(self):
        build = self.build()
        common = self.root / "shared.hpp"
        common.write_text("original")
        build["measurement_files"] = {str(common): digest(common)}
        identity(build)
        common.write_text("changed")
        with self.assertRaisesRegex(ValueError, "measurement source changed"):
            identity(build)

    def test_reuse_receipt_preserves_build_and_rejects_changed_sources(self):
        build = self.build()
        source = self.root / "operator.cpp"
        source.write_text("same production")
        build["measurement_files"] = {str(source): digest(source)}
        path = self.root / "build.json"
        write_json(path, build)
        with patch("campaign.fingerprint", return_value="new-docs-provenance"):
            rebound = reuse_build(path, self.root / "reuse")
        self.assertEqual(read_json(path), build)
        self.assertEqual(rebound["compiled_source_id"], "same")
        self.assertEqual(rebound["source_id"], "new-docs-provenance")
        self.assertTrue(checked(rebound["reuse_receipt"])["passed"])
        source.write_text("changed production")
        with self.assertRaisesRegex(ValueError, "measurement source changed"):
            reuse_build(path, self.root / "invalid-reuse")

    def test_campaign_rejects_changed_configuration(self):
        path = self.root / "config.json"
        write_json(path, {})
        write_json(
            self.root / "campaign.json",
            {
                "operators": [],
                "configuration_files": {"test": {"path": str(path), "sha256": digest(path)}},
            },
        )
        path.write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "hash changed"):
            load(self.root)

    def test_v2_historical_reporting_and_new_production_rejection(self):
        legacy_settings = config()
        path = self.root / "config.json"
        write_json(path, legacy_settings)
        with self.assertRaisesRegex(ValueError, "historical"):
            main(
                [
                    "init",
                    "--config",
                    str(path),
                    "--confirmed",
                    "--out",
                    str(self.root / "session.json"),
                ]
            )
        # Existing historical paired evidence remains replayable through the old public API.
        fixture = legacy.OptimizationToolsTest()
        fixture.setUp()
        try:
            replay = compare_pairs(fixture.timing_index(), expected_pairs=10)
            self.assertIn("results", replay)
        finally:
            fixture.doCleanups()

    def test_correctness_mismatch_keeps_separate_hash_bound_evidence(self):
        build = self.build()
        coverage = {"source_id": "same", "passed": True, "cases": ["baseline"]} | {
            name: True
            for name in ("ownership", "boundaries", "analytic_reference", "errors", "live_fallback")
        }
        path = self.root / "coverage.json"
        write_json(path, coverage)

        def capture(command, _cwd, _log, _receipt, _timeout):
            if "--verify-output" in command:
                destination = Path(command[command.index("--verify-output") + 1])
                write_json(destination, snapshots(1.0 if "native-0" in destination.name else 2.0))

        with patch("campaign.execute", side_effect=capture):
            with self.assertRaises(CorrectnessFailure) as result:
                native_baseline(
                    build,
                    self.settings(),
                    self.root,
                    [sys.executable, "fixture"],
                    {"path": str(path), "sha256": digest(path)},
                )
        self.assertFalse(checked(result.exception.evidence)["passed"])

    def test_missing_consumer_evidence_blocks_admission(self):
        cfg = self.settings()
        cfg["consumer_policies"] = {"consumer": cfg["policies"]}
        with self.assertRaisesRegex(ValueError, "affected consumer"):
            validate_consumer_baselines(cfg, {})

    def test_reports_refuse_active_measurement_sessions(self):
        write_json(self.root / "campaign.json", {"phase": "qualifying", "operators": []})
        with self.assertRaisesRegex(ValueError, "sealed measurement"):
            report(self.root, self.root / "report.md")

    def test_report_fixture_absolute_limits_uncertainty_and_partial_coverage(self):
        result = {
            "results": [
                {
                    "case": ["test", "baseline", "double", "large", "compute"],
                    "metric": "p99_ms",
                    "before_median": 0.71,
                    "after_median": 0.68,
                    "aa_p90_delta_pct": 6.3,
                    "baseline_iqr_pct": 2.2,
                    "familywise_ci95_pct": [None, None],
                    "limits": {"aa_p90_delta_pct": 5, "baseline_iqr_pct": 10},
                }
            ]
        }
        path = self.root / "aa.json"
        write_json(path, result)
        state = {
            "phase": "qualification_complete",
            "config": {"root": str(Path(__file__).resolve().parents[3])},
            "events": [],
            "sessions": {},
            "operators": [
                {
                    "operator": "test",
                    "class": "TestOp",
                    "metadata": {"path": str(path), "sha256": digest(path)},
                    "recipe": None,
                    "family": "irregular",
                    "status": "blocked",
                    "gaps": ["unresolved timing regime"],
                    "qualification": {"validation": {"path": str(path), "sha256": digest(path)}},
                }
            ],
        }
        write_json(self.root / "campaign.json", state)
        report(self.root, self.root / "report.md")
        text = (self.root / "report.md").read_text()
        for expected in (
            "0.71",
            "0.68",
            "6.3% / 5",
            "None, None",
            "unresolved timing regime",
            "omitted or blocked",
        ):
            self.assertIn(expected, text)
        self.assertIn("p99_ms", "\n".join(timing_table(result)))


if __name__ == "__main__":
    unittest.main()
