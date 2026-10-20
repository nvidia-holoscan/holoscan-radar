# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real host subprocess fixtures validate tooling; never GPU performance evidence."""

import copy
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifacts import digest, read_json, write_json
from campaign import config_digest, validate_admission
from confirmation import collect, compare, protocol, render_table, replay, schedule
from qualification import reference
from radar_perf import config_hash, main, normalize_config, timing_comparison
from test_optimization import snapshots

SCRIPT = """import json,sys
from pathlib import Path
args=sys.argv[1:]
def arg(k):return args[args.index(k)+1]
if '--list-cases' in args:print('baseline\\tsmall');sys.exit(0)
path=Path(arg('--output'));n=int(arg('--frames'));side=arg('--side');block=0;occurrence=1
if (path.parent/'blocks.json').exists():
 block=int(path.name.split('-')[0]);pos=int(path.name.split('-')[-2]);index=json.loads((path.parent/'blocks.json').read_text());order=index['schedule'][block];occurrence=order[:pos+1].count(side)
v=10.*float(arg('--factor'))*occurrence
values=[v]*n
if arg('--mode')=='tail' and side=='B':values[-1]=v*3
if arg('--mode')=='mixed' and side=='B':values[-1]=v*(10 if block%2 else .75)
def q(p):
 x=sorted(values);z=(n-1)*p;i=int(z);j=min(i+1,n-1);return x[i]+(x[j]-x[i])*(z-i)
row={'operator':'test','scenario_id':arg('--cases'),'precision':'float','size_id':'small','bucket':'compute','status':'ok','median_ms':q(.5),'p90_ms':q(.9),'p99_ms':q(.99),'cold_ms':100.,'max_ms':max(values),'samples':n,'raw_steady_ms':values,'raw_steady_run_offsets':[0,n]}
setup={'operator':'test','scenario_id':arg('--cases'),'precision':'float','size_id':'small','bucket':'setup','status':'ok','setup_ms':.5}
r={'schema_version':3,'provenance':{'git_commit':'base'},'platform':{'gpu':'synthetic-host-fixture'},'run_settings':{'warmup_frames':int(arg('--warmup')),'measured_frames':n,'repeats':1,'scope':arg('--scope'),'stopped_early':False,'scenarios_planned':1,'scenarios_completed':1},'results':[row,setup]}
path.write_text(json.dumps(r))
"""


class ConfirmationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "synthetic.py"
        self.script.write_text(SCRIPT)
        self.config = normalize_config(
            {
                "measurement_policy_version": 4,
                "operator": "test",
                "editable_scope": ["operator"],
                "policies": {"out": {"float": {"mode": "exact"}, "double": {"mode": "exact"}}},
                "pairs": 16,
                "max_candidates": 1,
                "wall_time_s": 120,
                "timing_precisions": {"test": ["float"]},
                "verification_cases": {"test": ["baseline"]},
                "target_cases": ["baseline"],
                "sampling": {"cases": "baseline", "repeats": 2, "random_seed": 17},
            }
        )
        self.plan = self.root / "plan.json"
        write_json(
            self.plan,
            {
                "id": "candidate",
                "mechanism": "test",
                "before_source_id": "A",
                "target_cases": ["baseline"],
                "expected_gain_pct": 20,
            },
        )
        self.patch = patch(
            "confirmation.clock_state", return_value={"configuration": {}, "observations": {}}
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def build(self, side, factor=1.0, mode="base"):
        return {
            "operator": "test",
            "operator_class": "test",
            "source_id": side,
            "git_commit": "base",
            "recipe_sha256": "recipe",
            "sdk_image": "sdk5-pin",
            "build_settings": {"cuda": "13"},
            "cwd": str(self.root),
            "command": [
                sys.executable,
                str(self.script),
                "--side",
                side,
                "--factor",
                str(factor),
                "--mode",
                mode,
            ],
            "artifacts": {str(self.script): digest(self.script)},
        }

    def run_experiment(self, factor=0.8, mode="base", blocks=16, scale=1.0, suffix=""):
        a, b = self.build("A", scale), self.build("B", factor * scale, mode)
        screen = self.root / ("screen" + suffix)
        collect(a, b, screen, 2, self.config["sampling"], "screen", 20, 120, plan=self.plan)
        confirmation = self.root / ("confirmation" + suffix)
        index = collect(
            a,
            b,
            confirmation,
            blocks,
            self.config["sampling"],
            "confirmation",
            20,
            120,
            plan=self.plan,
            screening=[screen / "blocks.json"],
        )
        return a, b, index

    def review(self):
        index = self.root / "confirmation/blocks.json"
        return {
            "index": reference(index),
            "candidate_plan_sha256": digest(self.plan),
            "interpretable": True,
            "rationale": "Synthetic matched controls intentionally vary by 100%; no build/order confounding in this fixture.",
            "evidence": [reference(index)],
        }

    def test_high_control_dispersion_does_not_veto_demonstrated_gain(self):
        _, _, index = self.run_experiment()
        result = timing_comparison(
            self.config, index, None, read_json(self.plan), control_review=self.review()
        )
        self.assertTrue(result["passed"])
        self.assertNotIn("calibration", result)
        med = next(r for r in result["results"] if r["metric"] == "median_ms")
        self.assertAlmostEqual(med["effect_pct"], -20.0)
        self.assertTrue(
            all(abs(c["AA"] - 100) < 1e-8 and abs(c["BB"] - 100) < 1e-8 for c in med["controls"])
        )
        self.assertTrue(
            all(r["alias_of"] for r in result["results"] if r["metric"].startswith("early_"))
        )
        self.assertIn("AA / BB", "\n".join(render_table(result)))
        self.assertEqual(result["family_size"], 10)
        # Acceptance replays every raw report; no A/A threshold controls the result.
        without_review = timing_comparison(self.config, index, None, read_json(self.plan))
        self.assertTrue(without_review["statistical_passed"])
        self.assertFalse(without_review["passed"])
        review = self.review() | {"interpretable": False}
        self.assertFalse(
            timing_comparison(
                self.config, index, None, read_json(self.plan), control_review=review
            )["passed"]
        )

    def test_real_tail_regression_rejects_faster_median(self):
        _, _, index = self.run_experiment(mode="tail")
        result = timing_comparison(
            self.config, index, None, read_json(self.plan), control_review=self.review()
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(r["verdict"] == "regressed" and r["metric"] == "p99_ms" for r in result["results"])
        )

    def test_no_gain_and_inconclusive_protection_are_distinct(self):
        _, _, index = self.run_experiment(factor=1.0, suffix="-null")
        self.assertFalse(compare(index, self.config, ["baseline"])["passed"])
        _, _, index = self.run_experiment(factor=0.5, mode="mixed", suffix="-mixed")
        result = compare(index, self.config, ["baseline"])
        self.assertTrue(result["passed"])
        self.assertTrue(
            any(
                r["metric"] == "p99_ms" and r["verdict"] == "inconclusive"
                for r in result["results"]
            )
        )

    def test_unit_scaling_and_block_budget_match_actual_family(self):
        from confirmation import budget_proposal

        _, _, index = self.run_experiment()
        result = compare(index, self.config, ["baseline"])
        _, _, scaled = self.run_experiment(scale=1000.0, suffix="-scaled")
        larger = compare(scaled, self.config, ["baseline"])
        self.assertEqual(result["passed"], larger["passed"])
        for a, b in zip(result["results"], larger["results"]):
            self.assertAlmostEqual(a["effect_pct"], b["effect_pct"])
            self.assertEqual(a["verdict"], b["verdict"])
        raw = read_json(index["blocks"][0]["runs"][0]["report"]["path"])
        from confirmation import raw_sampling
        from measurement import derive_raw

        proposal = budget_proposal(
            derive_raw(raw, raw_sampling(self.config["sampling"])), self.config
        )
        self.assertEqual(proposal["family_size"], result["family_size"])
        self.assertEqual(
            proposal["confirmation_runs_per_case_precision"], 4 * proposal["proposed_pairs"]
        )

    def test_screening_and_insufficient_blocks_cannot_pass(self):
        _, _, index = self.run_experiment(blocks=2)
        cfg = self.config | {"pairs": 2}
        result = compare(index, cfg, ["baseline"])
        self.assertTrue(result["insufficient_pair_budget"])
        self.assertFalse(result["passed"])
        with self.assertRaisesRegex(ValueError, "not screening"):
            compare(read_json(self.root / "screen/blocks.json"), cfg, ["baseline"])

    def test_resume_preserves_runs_and_rejects_tamper_or_identity_changes(self):
        a, b, index = self.run_experiment(blocks=2)
        path = self.root / "confirmation"
        times = {p.name: read_json(p)["started_at"] for p in path.glob("*.process.json")}
        collect(
            a,
            b,
            path,
            2,
            self.config["sampling"],
            "confirmation",
            20,
            120,
            plan=self.plan,
            screening=[self.root / "screen/blocks.json"],
        )
        self.assertEqual(
            times, {p.name: read_json(p)["started_at"] for p in path.glob("*.process.json")}
        )
        sampling = self.config["sampling"] | {"random_seed": 3}
        with self.assertRaisesRegex(ValueError, "identity or schedule"):
            collect(
                a,
                b,
                path,
                2,
                sampling,
                "confirmation",
                20,
                120,
                plan=self.plan,
                screening=[self.root / "screen/blocks.json"],
            )
        changed = copy.deepcopy(index)
        changed["blocks"][1] = copy.deepcopy(changed["blocks"][0])
        changed["blocks"][1]["number"] = 1
        with self.assertRaises(ValueError):
            replay(changed)
        child = Path(index["blocks"][0]["runs"][0]["report"]["path"])
        child.write_text(child.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            replay(index)

    def test_precision_plan_context_and_stale_review_rejected(self):
        _, _, index = self.run_experiment(blocks=2)
        cfg = self.config | {"pairs": 2, "timing_precisions": {"test": ["float", "double"]}}
        with self.assertRaisesRegex(ValueError, "precision coverage"):
            compare(index, cfg, ["baseline"])
        review = self.review() | {"candidate_plan_sha256": "wrong"}
        with self.assertRaisesRegex(ValueError, "control review"):
            timing_comparison(
                self.config | {"pairs": 2}, index, None, read_json(self.plan), control_review=review
            )
        changed = copy.deepcopy(index)
        changed["blocks"][0]["runs"][0]["clock_after"]["configuration"] = {"clock": "changed"}
        with self.assertRaisesRegex(ValueError, "clock"):
            replay(changed)

    def test_interruption_resumes_only_sealed_units_and_rejects_screen_leakage(self):
        import confirmation

        a, b = self.build("A"), self.build("B", 0.8)
        screen = self.root / "screen"
        collect(a, b, screen, 2, self.config["sampling"], "screen", 20, 120, plan=self.plan)
        output = self.root / "confirmation"
        execute = confirmation.execute
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise TimeoutError("synthetic interruption before next launch")
            return execute(*args, **kwargs)

        with (
            patch("confirmation.execute", side_effect=interrupted),
            self.assertRaises(TimeoutError),
        ):
            collect(
                a,
                b,
                output,
                2,
                self.config["sampling"],
                "confirmation",
                20,
                120,
                plan=self.plan,
                screening=[screen / "blocks.json"],
            )
        first = next(output.glob("*.process.json"))
        first_time = read_json(first)["started_at"]
        index = collect(
            a,
            b,
            output,
            2,
            self.config["sampling"],
            "confirmation",
            20,
            120,
            plan=self.plan,
            screening=[screen / "blocks.json"],
        )
        self.assertEqual(first_time, read_json(first)["started_at"])
        changed = copy.deepcopy(index)
        screen_unit = read_json(screen / "blocks.json")["blocks"][0]["runs"][0]
        changed["blocks"][0]["runs"][0]["report"] = screen_unit["report"]
        with self.assertRaisesRegex(ValueError, "reused observations"):
            replay(changed)
        done = next(output.glob("*.receipt.json"))
        done.unlink()
        with self.assertRaisesRegex(ValueError, "unsealed"):
            collect(
                a,
                b,
                output,
                2,
                self.config["sampling"],
                "confirmation",
                20,
                120,
                plan=self.plan,
                screening=[screen / "blocks.json"],
            )

    def test_consumer_uses_its_distinct_v4_matrix_and_needs_no_gain(self):
        cfg = normalize_config(
            self.config
            | {
                "consumer_policies": {"consumer": self.config["policies"]},
                "consumer_sampling": {
                    "consumer": {"cases": "consumer-case", "random_seed": 9, "repeats": 2}
                },
                "timing_precisions": {"test": ["float"], "consumer": ["float"]},
            }
        )
        a, b = self.build("A"), self.build("B", 1.0)
        a["operator"] = b["operator"] = "consumer"
        sampling = cfg["consumer_sampling"]["consumer"]
        collect(a, b, self.root / "consumer-screen", 2, sampling, "screen", 20, 120, plan=self.plan)
        index = collect(
            a,
            b,
            self.root / "consumer-confirmation",
            16,
            sampling,
            "confirmation",
            20,
            120,
            plan=self.plan,
            screening=[self.root / "consumer-screen/blocks.json"],
        )
        path = self.root / "consumer-confirmation/blocks.json"
        review = {
            "index": reference(path),
            "candidate_plan_sha256": digest(self.plan),
            "interpretable": True,
            "rationale": "Unchanged synthetic consumer; within-build controls are matched.",
            "evidence": [reference(path)],
        }
        result = timing_comparison(cfg, index, None, read_json(self.plan), control_review=review)
        self.assertTrue(result["passed"])
        self.assertEqual(result["case_ids"], ["consumer-case"])
        self.assertEqual(result["operators"], ["consumer"])
        with self.assertRaisesRegex(ValueError, "sampling"):
            timing_comparison(
                self.config | {"consumer_policies": cfg["consumer_policies"]},
                index,
                None,
                read_json(self.plan),
                control_review=review,
            )

    def admission_config(self):
        a = self.build("A")
        a["measurement_files"] = {str(self.script): digest(self.script)}
        log = self.root / "fixture-evidence.json"
        write_json(log, {"synthetic_host_fixture": True, "passed": True})
        coverage = self.root / "coverage.json"
        write_json(
            coverage,
            {
                "source_id": "A",
                "cases": ["baseline"],
                "passed": True,
                "ownership": True,
                "boundaries": True,
                "analytic_reference": True,
                "errors": True,
                "live_fallback": True,
                "evidence": [reference(log)],
            },
        )
        exports = []
        for i in range(2):
            path = self.root / f"native-{i}.json"
            write_json(path, snapshots())
            exports.append(reference(path))
        native = self.root / "native.json"
        write_json(native, {"passed": True, "coverage": reference(coverage), "exports": exports})
        admission = self.root / "admission.json"
        write_json(
            admission,
            {
                "config_sha256": config_digest(self.config),
                "config": self.config,
                "build": a,
                "native": reference(native),
                "consumer_baselines": {},
            },
        )
        return self.config | {"benchmark_admission": reference(admission)}, a

    def test_native_admission_needs_no_aa_and_rejects_missing_coverage(self):
        cfg, _ = self.admission_config()
        validate_admission(cfg)
        path = self.root / "coverage.json"
        value = read_json(path)
        value["ownership"] = False
        write_json(path, value)
        with self.assertRaises(ValueError):
            validate_admission(cfg)

    def test_cli_full_session_acceptance_without_calibration(self):
        cfg, a = self.admission_config()
        b = self.build("B", 0.8)
        paths = {}
        for name, value in (("config", cfg), ("before", a), ("after", b)):
            paths[name] = self.root / (name + ".json")
            write_json(paths[name], value)
        session = self.root / "session.json"
        self.assertEqual(
            main(["init", "--config", str(paths["config"]), "--confirmed", "--out", str(session)]),
            0,
        )
        with patch("radar_perf.fingerprint", return_value="A"):
            self.assertEqual(
                main(
                    [
                        "plan",
                        "--session",
                        str(session),
                        "--record",
                        str(self.plan),
                        "--root",
                        str(self.root),
                    ]
                ),
                0,
            )
        common = [
            "measure",
            "--before",
            str(paths["before"]),
            "--after",
            str(paths["after"]),
            "--config",
            str(paths["config"]),
            "--session",
            str(session),
            "--plan",
            str(self.plan),
        ]
        self.assertEqual(
            main(common + ["--phase", "screen", "--out", str(self.root / "screen")]), 0
        )
        self.assertEqual(
            main(
                common
                + [
                    "--phase",
                    "confirmation",
                    "--screening",
                    str(self.root / "screen/blocks.json"),
                    "--out",
                    str(self.root / "confirmation"),
                ]
            ),
            0,
        )
        index = self.root / "confirmation/blocks.json"
        inspect = self.root / "inspect.json"
        self.assertEqual(
            main(
                [
                    "inspect-confirmation",
                    "--pairs",
                    str(index),
                    "--config",
                    str(paths["config"]),
                    "--plan",
                    str(self.plan),
                    "--out",
                    str(inspect),
                ]
            ),
            0,
        )
        self.assertFalse(read_json(inspect)["passed"])
        review = self.root / "control-review.json"
        write_json(review, self.review())
        context = {
            "candidate_id": "candidate",
            "confirmed_config_sha256": config_hash(cfg),
            "before_source_id": "A",
            "after_source_id": "B",
            "affected_operators": ["test"],
            "candidate_plan_sha256": digest(self.plan),
        }
        context_path = self.root / "context.json"
        write_json(context_path, context)
        speed = self.root / "speed.json"
        self.assertEqual(
            main(
                [
                    "compare-timing",
                    "--pairs",
                    str(index),
                    "--config",
                    str(paths["config"]),
                    "--context",
                    str(context_path),
                    "--plan",
                    str(self.plan),
                    "--control-review",
                    str(review),
                    "--out",
                    str(speed),
                ]
            ),
            0,
        )
        self.assertNotIn("calibration", read_json(speed)["inputs"])
        gates = {"speed": str(speed)}
        before, after = snapshots(), snapshots()
        before["source_id"], after["source_id"] = "A", "B"
        for name, value in (
            ("native-before", before),
            ("native-after", after),
            ("policies", cfg["policies"]),
        ):
            write_json(self.root / (name + ".json"), value)
        eq = self.root / "equivalence.json"
        self.assertEqual(
            main(
                [
                    "compare-outputs",
                    "--before",
                    str(self.root / "native-before.json"),
                    "--after",
                    str(self.root / "native-after.json"),
                    "--policies",
                    str(self.root / "policies.json"),
                    "--context",
                    str(context_path),
                    "--out",
                    str(eq),
                ]
            ),
            0,
        )
        gates["equivalence"] = str(eq)
        for kind in ("contract", "coverage", "tests"):
            path = self.root / (kind + "-gate.json")
            write_json(
                path,
                {
                    "kind": kind,
                    "context": context,
                    "passed": True,
                    "operators": ["test"],
                    "evidence": [reference(self.root / "fixture-evidence.json")],
                },
            )
            gates[kind] = str(path)
        record = self.root / "candidate.json"
        write_json(
            record,
            {
                "id": "candidate",
                "summary": "Synthetic validation",
                "files": ["operator"],
                "cases": ["baseline"],
                "affected_operators": ["test"],
                "before_source_id": "A",
                "after_source_id": "B",
                "gates": gates,
            },
        )
        self.assertEqual(main(["candidate", "--session", str(session), "--record", str(record)]), 0)
        self.assertTrue(read_json(session)["candidates"][0]["kept"])
        self.assertEqual(
            main(
                [
                    "finish",
                    "--session",
                    str(session),
                    "--reason",
                    "host validation complete",
                    "--cumulative-timing",
                    str(speed),
                    "--control-review",
                    str(review),
                ]
            ),
            0,
        )
        report = self.root / "report.md"
        self.assertEqual(main(["report", "--session", str(session), "--out", str(report)]), 0)
        self.assertIn("AA / BB", report.read_text())

    def test_campaign_discovers_baselines_and_prepares_without_aa(self):
        import campaign

        cfg, a = self.admission_config()
        cfg.pop("benchmark_admission")
        recipe = self.root / "recipe.yaml"
        recipe.write_text("synthetic recipe fixture")
        metadata = self.root / "metadata.json"
        write_json(metadata, {"synthetic": True})
        a["recipe_sha256"] = digest(recipe)
        cfg_path, build_path = self.root / "campaign-config.json", self.root / "build.json"
        write_json(cfg_path, cfg)
        write_json(build_path, a)
        item = {
            "operator": "test",
            "class": "test",
            "family": "reduction",
            "recipe": reference(recipe),
            "metadata": reference(metadata),
            "status": "discovered",
            "gaps": [],
        }
        output = self.root / "campaign"
        with (
            patch("campaign.discover", return_value=[item]),
            patch("campaign.preflight", return_value={}),
            patch("campaign.fingerprint", return_value="A"),
        ):
            campaign.create(
                {
                    "root": str(self.root),
                    "mode": "optimize",
                    "operators": ["test"],
                    "configs": {"test": str(cfg_path)},
                    "builds": {"test": str(build_path)},
                    "baseline_budget_seconds": 120,
                    "verification": {
                        "test": {
                            "correctness_command": [
                                sys.executable,
                                "-c",
                                "print('synthetic fixture')",
                            ],
                            "coverage": reference(self.root / "coverage.json"),
                        }
                    },
                },
                output,
            )
        with patch("campaign.qualify", side_effect=AssertionError("no standalone A/A allowed")):
            self.assertEqual(campaign.qualify_campaign(output)["phase"], "design_ready")
        execute = campaign.execute

        def capture(argv, *args, **kwargs):
            if "--verify-output" in argv:
                write_json(argv[argv.index("--verify-output") + 1], snapshots())
                return None
            return execute(argv, *args, **kwargs)

        with (
            patch("campaign.execute", side_effect=capture),
            patch("campaign.clock_state", return_value={"configuration": {}, "observations": {}}),
        ):
            state = campaign.baseline(output)
        self.assertEqual(
            state["operators"][0]["status"], "baseline_ready", state["operators"][0]["gaps"]
        )
        self.assertNotIn("calibration", state["operators"][0]["baseline"])
        with patch(
            "campaign.native_baseline", side_effect=AssertionError("completed evidence must resume")
        ):
            self.assertEqual(campaign.baseline(output)["operators"][0]["status"], "baseline_ready")
        state = campaign.prepare_sessions(output, confirmed=True)
        self.assertTrue(state["optimization_confirmed"])
        self.assertTrue((output / "sessions/test-session.json").exists())
        with patch("campaign.fingerprint", return_value="A"):
            campaign.report(output, self.root / "campaign-report.md")
        self.assertNotIn("full-case A/A", (self.root / "campaign-report.md").read_text())

    def test_simulated_null_coverage_with_correlated_frames_and_stalls(self):
        # Independent experiments/blocks; autocorrelated frames and shared run
        # shocks. This exercises uncertainty, not deployment qualification.
        from timing import median_interval, quantile

        rng = random.Random(20261020)
        trials, family, blocks, false_families = 200, 6, 20, 0
        for _ in range(trials):
            contrasts = [[] for _ in range(family)]
            for _block in range(blocks):
                shock = math.exp(rng.gauss(0.0, 0.2))
                for case, scale in enumerate((0.00013, 0.021)):
                    sides = []
                    for _side in range(2):
                        values = []
                        for _run in range(2):
                            run_scale = scale * shock * math.exp(rng.gauss(0.0, 0.1))
                            state = 0.0
                            for _frame in range(50):
                                state = 0.8 * state + rng.gauss(0.0, 0.025)
                                values.append(
                                    run_scale * math.exp(state) * (8 if rng.random() < 0.01 else 1)
                                )
                        sides.append(values)
                    for q, probability in enumerate((0.5, 0.9, 0.99)):
                        contrasts[case * 3 + q].append(
                            math.log(
                                quantile(sides[1], probability) / quantile(sides[0], probability)
                            )
                        )
            bounds = [median_interval(values, family) for values in contrasts]
            false_families += any(lo > 0 or hi < 0 for lo, hi in bounds)
        # 99.9% binomial bound at the nominal familywise 5% rate. Deterministic
        # seed; a regression can fail this without imposing a noise threshold.
        bound = next(
            k
            for k in range(trials + 1)
            if sum(math.comb(trials, j) * 0.05**j * 0.95 ** (trials - j) for j in range(k + 1))
            >= 0.999
        )
        self.assertLessEqual(false_families, bound)

    def test_balanced_schedule_and_invalid_migration(self):
        for phase in ("screen", "confirmation"):
            order = schedule(20, 19, phase)
            self.assertEqual(order, schedule(20, 19, phase))
            self.assertEqual(len(set(order)) > 1, True)
            self.assertTrue(
                all(sum(run[p] == "A" for run in order) == 10 for p in range(len(order[0])))
            )
        for update in (
            {"repeats": 3},
            {"random_seed": None},
            {"cases": "../escape"},
            {"case_overrides": {"missing": {"frames": 50}}},
        ):
            with self.assertRaises(ValueError):
                protocol(self.config["sampling"] | update)
        with self.assertRaisesRegex(ValueError, "no A/A"):
            normalize_config(self.config | {"max_aa_delta_pct": 5})
        with self.assertRaisesRegex(ValueError, "integrated controls"):
            timing_comparison(self.config, {"before": {"operator": "test"}}, None, calibrate=True)


if __name__ == "__main__":
    unittest.main()
