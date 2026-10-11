# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side CLI for operator-perf. No automatic production edits or issue filing."""

import argparse
import csv
import datetime
import hashlib
import io
import subprocess
import sys
import time
from pathlib import Path

from artifacts import (
    checkpoint,
    digest,
    file_state,
    fingerprint,
    finite,
    read_json,
    restore,
    write_json,
)
from equivalence import compare_snapshots, validate_policy
from timing import METRICS, compare_pairs, compatible, run_pairs, verify_build

DEFAULTS = {
    "mode": "optimize",
    "max_candidates": 20,
    "max_unsuccessful": 3,
    "wall_time_s": None,
    "target_ms": None,
    "pairs": 10,
    "objective": "median_ms",
    "target_cases": [],
    "consumer_policies": {},
    "consumer_targets": {},
    "verification_cases": {},
    "timing_precisions": {},
    "min_effect_pct": 0.0,
    "max_aa_delta_pct": 5.0,
    "max_baseline_spread_pct": 10.0,
    "protected_metrics": ["median_ms", "cold_ms", "setup_ms", "override_ms"],
    "sampling": {
        "scope": "overrides",
        "sizes": "small,medium,large",
        "warmup": 5,
        "frames": 50,
        "repeats": 3,
    },
}
GATES = ("contract", "coverage", "tests", "equivalence", "speed")


def normalize_config(config):
    result = DEFAULTS | config
    if not result.get("operator") or not result.get("policies") or not result.get("editable_scope"):
        raise ValueError("operator, policies, and editable_scope are required")
    if result["mode"] not in {"assess", "fix", "optimize"} or result["objective"] not in METRICS:
        raise ValueError("invalid mode or objective")
    for name in ("max_candidates", "max_unsuccessful", "pairs"):
        if type(result[name]) is not int or result[name] < (2 if name == "pairs" else 1):
            raise ValueError(f"invalid limit: {name}")
    for name in ("wall_time_s", "target_ms"):
        value = result[name]
        if value is not None:
            finite(value, name, positive=True)
    for name in ("min_effect_pct", "max_aa_delta_pct", "max_baseline_spread_pct"):
        if finite(result[name], name) < 0:
            raise ValueError(f"negative measurement threshold: {name}")
    for precisions in result["timing_precisions"].values():
        if not precisions or set(precisions) - {"float", "double"}:
            raise ValueError("timing precisions must explicitly name float and/or double")
    for port, precisions in result["policies"].items():
        if set(precisions) != {"float", "double"}:
            raise ValueError(f"{port}: propose separate float and double policies")
        for policy in precisions.values():
            validate_policy(policy)
    for policies in result["consumer_policies"].values():
        for precisions in policies.values():
            if set(precisions) != {"float", "double"}:
                raise ValueError("consumer policies require both precisions")
            for policy in precisions.values():
                validate_policy(policy)
    return result


def config_hash(config):
    import json

    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_session(path):
    session = read_json(path)
    if config_hash(session["config"]) != session["confirmed_config_sha256"]:
        raise ValueError("confirmed configuration changed; obtain new confirmation")
    return session


def stop_reason(session):
    config, candidates = session["config"], session["candidates"]
    if candidates and candidates[-1].get("kept") and candidates[-1].get("target_reached"):
        return "target achieved"
    if (
        config["wall_time_s"] is not None
        and time.time() - session["started_at"] >= config["wall_time_s"]
    ):
        return "wall-time limit"
    limit = 1 if config["mode"] == "fix" else config["max_candidates"]
    if len(candidates) >= limit:
        return "candidate limit"
    dry = 0
    for candidate in reversed(candidates):
        if candidate["kept"]:
            break
        if candidate.get("outcome") == "validation_blocked":
            continue
        dry += 1
    if dry >= config["max_unsuccessful"]:
        return "consecutive unsuccessful limit"
    return None


def validate_plan(plan, config):
    if not all(plan.get(key) for key in ("id", "mechanism", "before_source_id", "target_cases")):
        raise ValueError("plan requires id, mechanism, baseline source, and target cases")
    if (
        not isinstance(plan["target_cases"], list)
        or not all(isinstance(case, str) and case for case in plan["target_cases"])
        or len(set(plan["target_cases"])) != len(plan["target_cases"])
    ):
        raise ValueError("plan target cases must be unique nonempty strings")
    finite(plan.get("expected_gain_pct"), "expected_gain_pct", positive=True)
    if config["target_cases"] and set(plan["target_cases"]) - set(config["target_cases"]):
        raise ValueError("candidate targets are outside the confirmed allowable set")


def timing_comparison(config, pairs, calibration, plan=None, calibrate=False):
    """One policy path for CLI, acceptance replay, and cumulative reporting."""
    operator = pairs["before"]["operator"]
    primary = operator == config["operator"]
    if not primary and operator not in config["consumer_policies"]:
        raise ValueError("timing operator is not a confirmed consumer")
    targets = config["target_cases"] if primary else config["consumer_targets"].get(operator, [])
    if plan is not None:
        validate_plan(plan, config)
        if plan["before_source_id"] != pairs["before"]["source_id"]:
            raise ValueError("candidate plan baseline source mismatch")
        if primary:
            targets = plan["target_cases"]
    options = dict(
        metric=config["objective"],
        targets=targets,
        protected_metrics=config["protected_metrics"],
        expected_pairs=config["pairs"],
        expected_sampling=config["sampling"],
        require_improvement=primary,
        required_precisions=config["timing_precisions"].get(operator, ["float", "double"]),
        min_effect_pct=config["min_effect_pct"],
        max_aa_delta_pct=config["max_aa_delta_pct"],
        max_baseline_spread_pct=config["max_baseline_spread_pct"],
        family_multiplier=1 + len(config["consumer_policies"]),
    )
    if calibrate:
        return compare_pairs(pairs, **options, calibration=True)
    if calibration is None:
        raise ValueError("speed gate requires baseline-versus-baseline calibration")
    compatible(pairs["before"], calibration["before"])
    if any(
        calibration[side]["source_id"] != pairs["before"]["source_id"]
        or calibration[side]["git_commit"] != pairs["before"]["git_commit"]
        for side in ("before", "after")
    ):
        raise ValueError("stale calibration baseline provenance")
    calibrated = compare_pairs(calibration, **options, calibration=True)
    actual = compare_pairs(pairs, **options)
    for index in (pairs, calibration):
        first = read_json(index["pairs"][0]["before"]["path"])
        if index is pairs:
            platform = first["platform"]
            baseline_rows = first["results"]
        elif first["platform"] != platform:
            raise ValueError("calibration hardware differs from candidate measurement")
        elif [
            {k: row.get(k) for k in ("variant", "input_shape", "map_shape", "live_port")}
            for row in first["results"]
        ] != [
            {k: row.get(k) for k in ("variant", "input_shape", "map_shape", "live_port")}
            for row in baseline_rows
        ]:
            raise ValueError("calibration case configuration differs from baseline")
    if [(r["case"], r["metric"]) for r in actual["results"]] != [
        (r["case"], r["metric"]) for r in calibrated["results"]
    ]:
        raise ValueError("calibration case/metric inventory differs from candidate measurement")
    actual["calibration"] = calibrated
    if not calibrated["passed"]:
        actual.update(passed=False, status="validation_blocked")
    return actual


def comparison_evidence(value, config, context, plan=None):
    name = value["kind"]
    if value.get("context") != context or name not in {"equivalence", "speed"}:
        raise ValueError("comparison belongs to another candidate/configuration/source")
    if "checks" in value:
        operators, cases = set(), set()
        target_reached = False
        if not value["checks"]:
            raise ValueError("empty consumer checks")
        for artifact in value["checks"]:
            if digest(artifact["path"]) != artifact["sha256"]:
                raise ValueError("consumer comparison artifact changed")
            child = read_json(artifact["path"])
            if child.get("kind") != name or not child.get("passed"):
                raise ValueError("consumer comparison does not pass")
            covered, measured, hit_target = comparison_evidence(child, config, context, plan)
            operators |= covered
            cases |= measured
            target_reached = target_reached or hit_target
        return operators, cases, target_reached
    inputs = value.get("inputs", {})
    for artifact in inputs.values():
        if digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"{name} gate input artifact changed")
    if name == "equivalence":
        if set(inputs) != {"before", "after", "policies"}:
            raise ValueError("equivalence gate requires snapshot and policy artifacts")
        before, after = (read_json(inputs[side]["path"]) for side in ("before", "after"))
        if (
            before.get("source_id") != context["before_source_id"]
            or after.get("source_id") != context["after_source_id"]
        ):
            raise ValueError("equivalence gate source identity mismatch")
        operator = before["operator"]
        policies = read_json(inputs["policies"]["path"])
        expected = (
            config["policies"]
            if operator == config["operator"]
            else config["consumer_policies"].get(operator)
        )
        if policies != expected:
            raise ValueError("equivalence rules differ from confirmed consumer policies")
        actual = compare_snapshots(before, after, policies)
        inventory = config["verification_cases"].get(operator)
        if not inventory or set(actual["case_ids"]) != set(inventory):
            raise ValueError("verification cases differ from confirmed complete inventory")
    else:
        if set(inputs) != {"pairs", "calibration", "plan"}:
            raise ValueError("speed gate requires paired timing, calibration, and plan artifacts")
        if read_json(inputs["plan"]["path"]) != plan:
            raise ValueError("speed plan differs from predeclared candidate")
        pairs = read_json(inputs["pairs"]["path"])
        if any(
            pairs[side]["source_id"] != context[field]
            for side, field in (("before", "before_source_id"), ("after", "after_source_id"))
        ):
            raise ValueError("speed gate source identity mismatch")
        operator = pairs["before"]["operator"]
        actual = timing_comparison(config, pairs, read_json(inputs["calibration"]["path"]), plan)
    if not actual["passed"]:
        raise ValueError(f"{name} gate evidence does not pass")
    target_reached = False
    if name == "speed" and operator == config["operator"] and config["target_ms"] is not None:
        objective_rows = [
            row
            for row in actual["results"]
            if row["metric"] == "median_ms" and row["case"][1] in plan["target_cases"]
        ]
        target_reached = bool(objective_rows) and all(
            row["after_median"] <= config["target_ms"] for row in objective_rows
        )
    return set(actual["operators"]), set(actual["case_ids"]), target_reached


def add_candidate(session, record):
    if session.get("stop_reason") or stop_reason(session):
        raise ValueError("session stopped or budget exhausted")
    if session["config"]["mode"] == "assess":
        raise ValueError("assessment cannot record production candidates")
    if not all(record.get(field) for field in ("summary", "files", "cases", "affected_operators")):
        raise ValueError("candidate requires summary, files, cases, and affected operators")
    for field in ("before_source_id", "after_source_id"):
        if not record.get(field):
            raise ValueError(f"candidate requires {field}")
    if session["config"]["operator"] not in record["affected_operators"]:
        raise ValueError("affected consumers must include the selected operator")
    if any(c["id"] == record["id"] for c in session["candidates"]):
        raise ValueError("duplicate candidate ID")
    current_best = session.get("current_best_source_id")
    if current_best and record["before_source_id"] != current_best:
        raise ValueError("candidate baseline differs from the current best source")
    declaration = session.get("plans", {}).get(record["id"])
    if not declaration or digest(declaration["path"]) != declaration["sha256"]:
        raise ValueError("candidate requires an unchanged predeclared plan")
    plan = read_json(declaration["path"])
    validate_plan(plan, session["config"])
    if plan["before_source_id"] != record["before_source_id"]:
        raise ValueError("candidate differs from predeclared baseline")
    gates = record.get("gates", {})
    if set(gates) != set(GATES):
        raise ValueError("candidate requires all five gate artifacts")
    evidence = {}
    measured_cases = set(record["cases"])
    target_reached = False
    for name, path in gates.items():
        value = read_json(path)
        expected_context = {
            "candidate_id": record["id"],
            "confirmed_config_sha256": session["confirmed_config_sha256"],
            "before_source_id": record["before_source_id"],
            "after_source_id": record["after_source_id"],
            "affected_operators": record["affected_operators"],
            "candidate_plan_sha256": declaration["sha256"],
        }
        if value.get("kind") != name or value.get("context") != expected_context:
            raise ValueError(f"{name} gate belongs to another candidate/configuration/source")
        if type(value.get("passed")) is not bool:
            raise ValueError(f"{name} gate missing boolean passed")
        if name in {"contract", "coverage", "tests"} and not value.get("evidence"):
            raise ValueError(f"{name} gate requires review/test evidence")
        if (
            value["passed"]
            and name in {"contract", "coverage", "tests"}
            and set(value.get("operators", [])) != set(record["affected_operators"])
        ):
            raise ValueError(f"{name} evidence must cover every affected consumer")
        if value["passed"] and name in {"equivalence", "speed"}:
            operators, cases, hit_target = comparison_evidence(
                value, session["config"], expected_context, plan
            )
            if operators != set(record["affected_operators"]):
                raise ValueError(f"{name} gate lacks affected consumer evidence")
            measured_cases |= cases
            target_reached = target_reached or hit_target
        evidence[name] = {
            "path": str(Path(path).resolve()),
            "sha256": digest(path),
            "passed": value["passed"],
            "status": value.get("status"),
        }
    record = record | {
        "gates": evidence,
        "kept": all(g["passed"] for g in evidence.values()),
        "target_reached": target_reached,
    }
    for bug in session["bugs"]:
        if bug["blocks_validation"] and (
            not bug["affected_cases"] or set(bug["affected_cases"]) & measured_cases
        ):
            record["kept"] = False
            record["blocked_by_bug"] = bug["title"]
    record["outcome"] = (
        "kept"
        if record["kept"]
        else (
            "validation_blocked"
            if record.get("blocked_by_bug")
            or any(g.get("status") == "validation_blocked" for g in evidence.values())
            else "rejected"
        )
    )
    session["candidates"].append(record)
    session.setdefault("original_source_id", record["before_source_id"])
    session["current_best_source_id"] = (
        record["after_source_id"] if record["kept"] else record["before_source_id"]
    )
    session["next_stop_reason"] = stop_reason(session)
    return record


def profile_artifact(capture, tables, started_at, source_id):
    capture = Path(capture)
    if capture.stat().st_size == 0 or capture.stat().st_mtime < started_at:
        raise ValueError("empty or stale profiler capture")
    summaries, findings = [], []
    for path in tables:
        path = Path(path)
        if path.stat().st_mtime < capture.stat().st_mtime:
            raise ValueError(f"stale profiler export: {path}")
        text = path.read_text(encoding="utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
        name_keys = ("Name", "Operation", "Kernel Name")
        if not rows or not any(key in rows[0] for key in name_keys):
            raise ValueError(f"empty/unrecognized profiler table: {path}")
        summaries.append({"path": str(path.resolve()), "sha256": digest(path), "rows": rows})
        for row in rows:
            name = next((row[key] for key in name_keys if key in row), "") or ""
            if any(
                api in name
                for api in (
                    "cudaMalloc",
                    "cudaFree",
                    "cudaMemcpy",
                    "Synchronize",
                    "cudaLaunch",
                    "cudaGraphLaunch",
                    "cudaFuncSet",
                    "cudaStreamCreate",
                )
            ):
                findings.append(
                    {
                        "name": name,
                        "evidence": row,
                        "classification": "inspect steady recurrence; may be contract-required",
                    }
                )
    if not summaries or not source_id:
        raise ValueError("profiler needs tables and source fingerprint")
    return {
        "capture": str(capture.resolve()),
        "sha256": digest(capture),
        "source_id": source_id,
        "started_at": started_at,
        "tables": summaries,
        "findings": findings,
    }


def render_report(session):
    lines = [
        "# Operator Optimization Report",
        "",
        f"Operator: `{session['config']['operator']}`",
        f"Mode: `{session['config']['mode']}`",
        f"Stop reason: {session.get('stop_reason') or 'in progress'}",
        "",
        "## Unresolved Suspected Bugs",
        "",
    ]
    if not session["bugs"]:
        lines.append("No suspected bugs were found.")
    for bug in session["bugs"]:
        lines += [
            f"### {bug['title']}",
            "",
            f"Expected: {bug['expected']}",
            f"Observed: {bug['observed']}",
            f"Baseline reproduction: {bug['baseline_reproduction']}",
            f"Affected cases: {', '.join(bug['affected_cases']) or 'unknown'}",
            f"Blocks validation: {bug['blocks_validation']}",
            f"Evidence: {bug['evidence']}",
            "",
        ]
    lines += ["", "## Candidate Decisions", ""]
    if not session["candidates"]:
        lines.append("No candidates attempted.")
    for candidate in session["candidates"]:
        decision = candidate.get("outcome", "kept" if candidate["kept"] else "rejected")
        lines += [
            f"- {candidate['id']}: **{decision}**. {candidate['summary']}",
            f"  Affected operators: {', '.join(candidate['affected_operators'])}.",
            "  Gates: "
            + ", ".join(
                f"{k}={'pass' if v['passed'] else 'fail'}" for k, v in candidate["gates"].items()
            )
            + ".",
        ]
    lines += ["", "## Coverage And Remaining Findings", ""]
    for operator, precisions in session["config"].get("timing_precisions", {}).items():
        missing = {"float", "double"} - set(precisions)
        if missing:
            lines.append(
                f"- `{operator}`: timing does not protect {', '.join(sorted(missing))} precision."
            )
    lines += [f"- {item}" for item in session["coverage_gaps"] + session["remaining_findings"]] or [
        "None recorded."
    ]
    lines += ["", "## Cumulative Timing", ""]
    timing = session.get("cumulative_timing")
    if timing:
        lines += [
            "Compared final retained state against the original baseline.",
            "",
            "| Case | Metric | Improvement (%) | Familywise 95% Interval (%) | Verdict |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in timing["results"]:
            lo, hi = row["familywise_ci95_pct"]
            interval = "unresolved pair budget" if lo is None else f"{lo:.3f}, {hi:.3f}"
            lines.append(
                f"| {' / '.join(row['case'])} | {row['metric']} | "
                f"{row['improvement_pct']:.3f} | {interval} | {row['verdict']} |"
            )
    else:
        lines.append(
            "Final-versus-original timing has not been supplied; no cumulative speedup is claimed."
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"Confirmed configuration SHA256: `{session['confirmed_config_sha256']}`",
    ]
    for candidate in session["candidates"]:
        lines += [
            f"- {candidate['id']} {name}: `{gate['path']}` ({gate['sha256']})"
            for name, gate in candidate["gates"].items()
        ]
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--config", required=True)
    init.add_argument("--confirmed", action="store_true")
    init.add_argument("--out", required=True)
    source = commands.add_parser("fingerprint")
    source.add_argument("--root", default=".")
    capture = commands.add_parser("capture")
    capture.add_argument("--build", required=True)
    capture.add_argument("--out", required=True)
    capture.add_argument("--timeout-s", type=float, default=600)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--kind", choices=["speed", "equivalence"], required=True)
    aggregate.add_argument("--context", required=True)
    aggregate.add_argument("--checks", nargs="+", required=True)
    aggregate.add_argument("--out", required=True)
    for name in ("checkpoint", "seal"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--files", nargs="+", required=True)
        command.add_argument("--out", required=True)
    rollback = commands.add_parser("restore")
    rollback.add_argument("--checkpoint", required=True)
    rollback.add_argument("--candidate-state", required=True)
    verify = commands.add_parser("compare-outputs")
    verify.add_argument("--before", required=True)
    verify.add_argument("--after", required=True)
    verify.add_argument("--policies", required=True)
    verify.add_argument("--out", required=True)
    verify.add_argument("--context", required=True)
    measure = commands.add_parser("measure")
    measure.add_argument("--before", required=True)
    measure.add_argument("--after", required=True)
    measure.add_argument("--config", required=True)
    measure.add_argument("--out", required=True)
    measure.add_argument("--timeout-s", type=float, default=600)
    measure.add_argument("--session")
    compare = commands.add_parser("compare-timing")
    compare.add_argument("--pairs", required=True)
    compare.add_argument("--config", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--context", required=True)
    compare.add_argument("--calibration", required=True)
    compare.add_argument("--plan", required=True)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--pairs", required=True)
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--out", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--capture", required=True)
    profile.add_argument("--tables", nargs="+", required=True)
    profile.add_argument("--started-at", type=float, required=True)
    profile.add_argument("--source-id", required=True)
    profile.add_argument("--out", required=True)
    for name in ("candidate", "bug", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--session", required=True)
        command.add_argument("--record", required=True)
        if name == "plan":
            command.add_argument("--root", required=True)
    finish = commands.add_parser("finish")
    finish.add_argument("--session", required=True)
    finish.add_argument("--reason", required=True)
    finish.add_argument("--findings", nargs="*", default=[])
    finish.add_argument("--coverage-gaps", nargs="*", default=[])
    finish.add_argument("--cumulative-timing")
    finish.add_argument("--calibration")
    report = commands.add_parser("report")
    report.add_argument("--session", required=True)
    report.add_argument("--out", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command
    if command == "fingerprint":
        print(fingerprint(args.root))
        return 0
    if command == "capture":
        build = read_json(args.build)
        verify_build(build)
        path = Path(args.out).resolve()
        if path.exists():
            raise ValueError("snapshot export already exists")
        with path.with_suffix(".log").open("xb") as stream:
            subprocess.run(
                build["command"] + [str(path)],
                cwd=build["cwd"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_s,
                check=True,
            )
        value = read_json(path)
        if value.get("operator") != build["operator"]:
            raise ValueError("snapshot operator differs from build identity")
        value["source_id"] = build["source_id"]
        value["build_manifest_sha256"] = digest(args.build)
        write_json(path, value)
        return 0
    if command == "aggregate":
        context = read_json(args.context)
        checks = []
        passed = True
        blocked = False
        for path in args.checks:
            value = read_json(path)
            if value.get("context") != context or value.get("kind") != args.kind:
                raise ValueError("consumer gate context/kind mismatch")
            passed = passed and value["passed"]
            blocked = blocked or value.get("status") == "validation_blocked"
            checks.append({"path": str(Path(path).resolve()), "sha256": digest(path)})
        write_json(
            args.out,
            {
                "kind": args.kind,
                "context": context,
                "passed": passed,
                "checks": checks,
                "status": "validation_blocked" if blocked else "pass" if passed else "unsuccessful",
            },
        )
        return 0 if passed else 1
    if command == "init":
        config = normalize_config(read_json(args.config))
        if not args.confirmed:
            raise ValueError("obtain user confirmation of configuration before init --confirmed")
        if Path(args.out).exists():
            raise ValueError("session already exists")
        write_json(
            args.out,
            {
                "schema_version": 1,
                "config": config,
                "confirmed_config_sha256": config_hash(config),
                "started_at": time.time(),
                "confirmed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "candidates": [],
                "bugs": [],
                "coverage_gaps": [],
                "remaining_findings": [],
            },
        )
    elif command in {"checkpoint", "seal"}:
        if Path(args.out).exists():
            raise ValueError("checkpoint/state artifact already exists")
        value = (
            checkpoint(args.root, args.files)
            if command == "checkpoint"
            else {
                "root": str(Path(args.root).resolve()),
                "states": file_state(args.root, args.files),
            }
        )
        write_json(args.out, value)
    elif command == "restore":
        restore(read_json(args.checkpoint), read_json(args.candidate_state))
    elif command == "compare-outputs":
        value = compare_snapshots(
            read_json(args.before), read_json(args.after), read_json(args.policies)
        )
        value.update(
            kind="equivalence",
            context=read_json(args.context),
            inputs={
                name: {"path": str(Path(path).resolve()), "sha256": digest(path)}
                for name, path in (
                    ("before", args.before),
                    ("after", args.after),
                    ("policies", args.policies),
                )
            },
        )
        write_json(args.out, value)
        return 0 if value["passed"] else 1
    elif command == "measure":
        config = normalize_config(read_json(args.config))
        remaining = config["wall_time_s"]
        if args.session:
            session = load_session(args.session)
            if session["config"] != config or session.get("stop_reason") or stop_reason(session):
                raise ValueError("measurement configuration changed or session budget exhausted")
            if remaining is not None:
                remaining -= time.time() - session["started_at"]
        run_pairs(
            read_json(args.before),
            read_json(args.after),
            args.out,
            config["pairs"],
            args.timeout_s,
            config["sampling"],
            wall_time_s=remaining,
        )
    elif command == "compare-timing":
        config = normalize_config(read_json(args.config))
        pairs = read_json(args.pairs)
        plan = read_json(args.plan)
        context = read_json(args.context)
        if context.get("candidate_plan_sha256") != digest(args.plan):
            raise ValueError("comparison context differs from candidate plan")
        if context.get("candidate_id") != plan["id"]:
            raise ValueError("comparison candidate differs from declared plan")
        value = timing_comparison(config, pairs, read_json(args.calibration), plan)
        value.update(
            kind="speed",
            context=context,
            inputs={
                name: {"path": str(Path(path).resolve()), "sha256": digest(path)}
                for name, path in (
                    ("pairs", args.pairs),
                    ("calibration", args.calibration),
                    ("plan", args.plan),
                )
            },
        )
        write_json(args.out, value)
        return 0 if value["passed"] else 1
    elif command == "calibrate":
        value = timing_comparison(
            normalize_config(read_json(args.config)), read_json(args.pairs), None, calibrate=True
        )
        write_json(args.out, value)
        return 0 if value["passed"] else 1
    elif command == "profile":
        write_json(
            args.out, profile_artifact(args.capture, args.tables, args.started_at, args.source_id)
        )
    else:
        session = load_session(args.session)
        if command == "plan":
            plan = read_json(args.record)
            validate_plan(plan, session["config"])
            if fingerprint(args.root) != plan["before_source_id"]:
                raise ValueError("declare the plan on the baseline checkout before editing")
            if session.get("stop_reason") or stop_reason(session):
                raise ValueError("session budget exhausted")
            if (
                session.get("current_best_source_id", plan["before_source_id"])
                != plan["before_source_id"]
            ):
                raise ValueError("plan baseline is not current best")
            plans = session.setdefault("plans", {})
            if plan["id"] in plans:
                raise ValueError("candidate plan already declared")
            plans[plan["id"]] = {
                "path": str(Path(args.record).resolve()),
                "sha256": digest(args.record),
                "declared_at": time.time(),
            }
        elif command == "candidate":
            record = add_candidate(session, read_json(args.record))
            print(f"{record['id']}: {'kept' if record['kept'] else 'rejected'}")
        elif command == "bug":
            bug = read_json(args.record)
            required = {
                "title",
                "expected",
                "observed",
                "baseline_reproduction",
                "affected_cases",
                "blocks_validation",
                "evidence",
            }
            if not required <= bug.keys():
                raise ValueError("bug record missing reproduction/evidence fields")
            session["bugs"].append(bug)
            print(f"SUSPECTED BUG: {bug['title']}")
        elif command == "finish":
            session["stop_reason"] = args.reason
            session["remaining_findings"] = args.findings
            session["coverage_gaps"] = args.coverage_gaps
            if args.cumulative_timing:
                evidence = read_json(args.cumulative_timing)
                artifact = evidence.get("inputs", {}).get("pairs")
                if not artifact or digest(artifact["path"]) != artifact["sha256"]:
                    raise ValueError("cumulative timing requires unchanged raw paired evidence")
                pairs = read_json(artifact["path"])
                if pairs["before"]["source_id"] != session.get("original_source_id") or pairs[
                    "after"
                ]["source_id"] != session.get("current_best_source_id"):
                    raise ValueError(
                        "cumulative evidence is not original baseline versus retained state"
                    )
                config = session["config"]
                if not args.calibration:
                    raise ValueError("cumulative timing requires original-baseline calibration")
                session["cumulative_timing"] = timing_comparison(
                    config, pairs, read_json(args.calibration)
                )
        elif command == "report":
            Path(args.out).write_text(render_report(session), encoding="utf-8")
            return 0
        write_json(args.session, session)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
