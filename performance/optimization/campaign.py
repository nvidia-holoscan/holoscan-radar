# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment benchmark orchestration; production changes remain agent-owned."""

import argparse
import copy
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from artifacts import digest, fingerprint, finite, read_json, write_json
from diagnostics import clock_state, preflight
from measurement import device_lock, execute, protocol
from qualification import checked, qualify, reference, review, timing_table
from timing import run_pairs, verify_build

FAMILIES = {
    "clutter_filter": "reduction",
    "ddm_power_fold": "reduction",
    "fmcw_range_processor": "fft",
    "fmcw_doppler_processor": "fft",
    "fmcw_range_doppler_processor": "fft",
    "ddm_dense_demultiplex": "fft",
    "music_estimator": "eigensolver",
    "esprit_estimator": "eigensolver",
    "root_music_estimator": "eigensolver",
    "mvdr_estimator": "eigensolver",
}
PREFERRED = ("clutter_filter", "fmcw_range_processor", "music_estimator", "cfar_detector")


def identity(build):
    verify_build(build)
    files = build.get("measurement_files", {})
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError("measurement source changed: " + path)
    value = {
        k: build[k]
        for k in (
            "operator",
            "operator_class",
            "recipe_sha256",
            "sdk_image",
            "build_settings",
            "artifacts",
        )
    }
    value["measurement_files"] = files
    value["command"] = build["command"]
    value["device_id"] = build.get("device_id", "cuda-visible-device-0")
    if not files:
        value["whole_checkout_source_id"] = build["source_id"]
        if fingerprint(build["cwd"]) != build["source_id"]:
            raise ValueError("checkout changed; provide a verified measurement-source closure")
    return value


def reuse_build(build_path, output):
    """Create a new provenance binding; preserve the original manifest and archive."""
    build = read_json(build_path)
    if not build.get("measurement_files"):
        raise ValueError("build reuse requires a frozen measurement-source closure")
    value = identity(build)
    current = fingerprint(build["cwd"])
    directory = Path(output).resolve()
    if directory.exists():
        raise ValueError("reuse destination already exists")
    directory.mkdir(parents=True)
    receipt = {
        "passed": True,
        "original": reference(build_path),
        "compiled_source_id": build.get("compiled_source_id", build["source_id"]),
        "whole_checkout_source_id": current,
        "measurement_identity": value,
        "clock_configuration": clock_state()["configuration"],
        "note": "Unchanged reviewed source closure, recipe, command, image and archived runtime hashes; no historical evidence rewritten.",
    }
    write_json(directory / "reuse.json", receipt)
    rebound = build | {
        "source_id": current,
        "compiled_source_id": receipt["compiled_source_id"],
        "reuse_receipt": reference(directory / "reuse.json"),
    }
    write_json(directory / "build.json", rebound)
    return rebound


def discover(root, requested=None):
    root = Path(root).resolve()
    if requested is not None and (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(x, str) for x in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("requested operators must be a nonempty unique list")
    recipes = {
        p.parent.name: str(p) for p in (root / "performance/operators").glob("*/*/recipe.yaml")
    }
    values = []
    for path in sorted((root / "operators").glob("*/*/metadata.json")):
        data = read_json(path)
        name = path.parent.name.removesuffix("_op")
        if not data.get("operator"):
            continue
        if requested and name not in requested:
            continue
        values.append(
            {
                "operator": name,
                "class": data["operator"]["name"],
                "metadata": reference(path),
                "recipe": reference(recipes[name]) if name in recipes else None,
                "family": FAMILIES.get(name, "irregular"),
                "status": "discovered" if name in recipes else "blocked",
                "gaps": [] if name in recipes else ["missing performance harness"],
                "precision_coverage": "unverified until executable inventory and raw runs",
                "verification_coverage": "requires native coverage receipt and correctness fixtures",
                "case_patterns": (
                    re.findall(r"\bid: *[\"']?([^\"'\n,}]+)", Path(recipes[name]).read_text())
                    if name in recipes
                    else []
                ),
                "consumers": sorted(
                    {
                        p.parent.name.removesuffix("_op")
                        for p in (root / "operators").glob("*/*/*.*")
                        if p.suffix in {".cpp", ".hpp", ".cu"}
                        and p.parent != path.parent
                        and path.parent.name + "/" in p.read_text()
                    }
                ),
                "consumer_note": "Direct source includes only; agent must review transitive graph consumers.",
            }
        )
    if requested and set(requested) - {x["operator"] for x in values}:
        raise ValueError("unknown public operator requested")
    return values


def create(config, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    if path.exists():
        raise ValueError("campaign already exists")
    if config.get("mode", "assess") not in {"assess", "optimize"}:
        raise ValueError("invalid campaign mode")
    root = str(Path(config.get("root", ".")).resolve())
    items = discover(root, config.get("operators"))
    state = {
        "schema_version": 1,
        "config": config | {"root": root},
        "operators": items,
        "phase": "discovered",
        "started_at": time.time(),
        "events": [],
        "sessions": {},
        "whole_checkout_source_id": fingerprint(root),
        "optimization_confirmed": False,
    }
    state["configuration_files"] = {
        name: reference(value) for name, value in config.get("configs", {}).items()
    }
    write_json(path, state)
    write_json(output / "preflight.json", preflight(config.get("sdk_image")))
    return state


def load(output):
    state = read_json(Path(output) / "campaign.json")
    for ref in state.get("configuration_files", {}).values():
        checked(ref)
    for item in state["operators"]:
        checked(item["metadata"])
        if item["recipe"] and digest(item["recipe"]["path"]) != item["recipe"]["sha256"]:
            raise ValueError("recipe hash changed: " + item["recipe"]["path"])
    return state


def leaf_config(state, name):
    from radar_perf import normalize_config

    ref = state["config"].get("configs", {}).get(name)
    if not ref:
        raise ValueError("missing operator configuration: " + name)
    config = read_json(ref)
    if config.get("measurement_policy_version") not in {3, 4}:
        raise ValueError("campaign requires explicit version-3 or version-4 configurations")
    config["campaign_family_size"] = len([x for x in state["operators"] if x["recipe"]])
    return normalize_config(config)


def build_for(state, name):
    ref = state["config"].get("builds", {}).get(name)
    if not ref:
        raise ValueError("missing build manifest: " + name)
    build = read_json(ref)
    if build["operator"] != name:
        raise ValueError("build operator differs from campaign")
    item = next(x for x in state["operators"] if x["operator"] == name)
    if (
        build["operator_class"] != item["class"]
        or build["recipe_sha256"] != item["recipe"]["sha256"]
    ):
        raise ValueError("build class/recipe differs from discovered inventory")
    return build


def save(output, state):
    write_json(Path(output) / "campaign.json", state)


def qualify_campaign(output):
    state = load(output)
    eligible = [item for item in state["operators"] if item["recipe"]]
    if eligible and all(
        leaf_config(state, item["operator"])["measurement_policy_version"] == 4 for item in eligible
    ):
        for item in state["operators"]:
            if item["recipe"]:
                identity(build_for(state, item["operator"]))
                item["design"] = {"policy": 4, "standalone_aa_required": False}
        state["phase"] = "design_ready"
        save(output, state)
        return state

    budget = finite(
        state["config"].get("qualification_budget_seconds"),
        "explicit qualification budget",
        positive=True,
    )
    state["phase"] = "qualifying"
    save(output, state)
    candidates = [x for x in state["operators"] if x["recipe"]]
    names = state["config"].get("representatives")
    if names is None:
        names = []
        for family in ("reduction", "fft", "eigensolver", "irregular"):
            members = [x["operator"] for x in candidates if x["family"] == family]
            if members:
                names.append(next((x for x in PREFERRED if x in members), members[0]))
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - {x["operator"] for x in candidates}
    ):
        raise ValueError("invalid representative operators")
    for name in names:
        item = next(x for x in candidates if x["operator"] == name)
        started = time.time()
        try:
            cfg = leaf_config(state, name)
            build = build_for(state, name)
            current_identity = identity(build)
            if item.get("measurement_identity") not in (None, current_identity):
                raise ValueError("measurement identity changed; start a fresh campaign")
            item["measurement_identity"] = current_identity
            sampling = cfg["sampling"]
            selected_cases = (
                state["config"].get("representative_cases", {}).get(name, "baseline-small")
            )
            if not isinstance(selected_cases, str) or not selected_cases.strip():
                raise ValueError(
                    "representative_cases must contain nonempty comma-separated case IDs"
                )
            cfg["sampling"] = protocol(
                sampling
                | {
                    "cases": selected_cases,
                    "case_overrides": {
                        case: override
                        for case, override in sampling["case_overrides"].items()
                        if case in selected_cases.split(",")
                    },
                }
            )
            result = qualify(
                build,
                cfg,
                Path(output) / name / "qualification",
                budget / len(names),
                state["config"].get("exploration"),
            )
            item["qualification"] = result
            item["status"] = "protocol_recommended" if result.get("passed") else "blocked"
            if not result.get("passed"):
                item["gaps"] = [result.get("status", "qualification failed")]
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            item["status"] = "blocked"
            item["gaps"] = [str(error)]
            item["blocker_reasons"] = [
                {
                    "kind": (
                        "correctness"
                        if isinstance(error, CorrectnessFailure)
                        else "provenance_or_coverage"
                    ),
                    "detail": str(error),
                }
            ]
            if isinstance(error, CorrectnessFailure):
                item["blocker_reasons"][0]["evidence"] = error.evidence
        state["events"].append(
            {
                "phase": "qualification",
                "operator": name,
                "started_at": started,
                "finished_at": time.time(),
                "status": item["status"],
            }
        )
        save(output, state)
    state["phase"] = "qualification_complete"
    save(output, state)
    return state


class CorrectnessFailure(ValueError):
    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def _native_baseline(build, config, output, correctness_command, coverage_ref, deadline=None):
    """Existing native SDK fixtures only; hashes bind exact snapshots and test evidence."""
    from equivalence import compare_snapshots

    def limit():
        seconds = 600 if deadline is None else min(600, deadline - time.time())
        if seconds <= 0:
            raise TimeoutError("baseline budget exhausted")
        return seconds

    coverage = checked(coverage_ref)
    if coverage.get("source_id") != build["source_id"] or not coverage.get("passed"):
        raise ValueError("missing/stale native coverage receipt")
    for name in ("ownership", "boundaries", "analytic_reference", "errors", "live_fallback"):
        if coverage.get(name) is not True:
            raise ValueError("unverified native behavior: " + name)
    if not correctness_command or not all(isinstance(x, str) for x in correctness_command):
        raise ValueError("correctness command must be an argv list")
    execute(
        correctness_command,
        build["cwd"],
        output / "correctness.log",
        output / "correctness-process.json",
        limit(),
    )
    exports = []
    for repeat in range(2):
        path = output / f"native-{repeat}.json"
        command = build["command"] + ["--scope", config["sampling"]["scope"]]
        for option in ("cases", "sizes"):
            if option in config["sampling"]:
                command += ["--" + option, config["sampling"][option]]
        command += ["--verify-output", str(path)]
        receipt = path.with_suffix(".receipt.json")
        if receipt.exists():
            checked(read_json(receipt))
        else:
            if path.exists():
                raise ValueError("unsealed native snapshot; quarantine before retry")
            execute(
                command,
                build["cwd"],
                path.with_suffix(".log"),
                path.with_suffix(".process.json"),
                limit(),
            )
            write_json(receipt, reference(path))
        export = read_json(path)
        export["source_id"] = build["source_id"]  # In-memory context; immutable raw JSON unchanged.
        exports.append(export)
    required_cases = set(config["verification_cases"].get(build["operator"], []))
    if not required_cases or set(coverage.get("cases", [])) != required_cases:
        raise ValueError("native coverage receipt must list the complete confirmed case set")
    for export in exports:
        if {case["id"] for case in export["cases"]} != required_cases:
            raise ValueError("native snapshot case coverage differs")
    result = compare_snapshots(exports[0], exports[1], config["policies"])
    write_json(
        output / "native-review.json",
        result
        | {
            "exports": [reference(output / f"native-{i}.json") for i in range(2)],
            "coverage": coverage_ref,
        },
    )
    if not result["passed"]:
        raise CorrectnessFailure(
            "baseline native repeats failed exact comparison",
            reference(output / "native-review.json"),
        )
    return result


def native_baseline(build, config, output, correctness_command, coverage_ref, deadline=None):
    with device_lock(build.get("device_id", "cuda-visible-device-0")):
        return _native_baseline(build, config, output, correctness_command, coverage_ref, deadline)


def native_record(config, record, name, verify_current=True):
    """Version-four correctness admission has no timing repeatability prerequisite."""
    build = record["build"]
    identity(build) if verify_current else verify_build(build)
    policies = (
        config["policies"] if name == config["operator"] else config["consumer_policies"][name]
    )
    if record["config"]["policies"] != policies:
        raise ValueError("native output policies differ from confirmation")
    native = checked(record["native"])
    coverage = checked(native.get("coverage"))
    cases = set(config["verification_cases"].get(name, []))
    if (
        not cases
        or set(coverage.get("cases", [])) != cases
        or coverage.get("source_id") != build["source_id"]
    ):
        raise ValueError("native coverage identity/inventory differs")
    if (
        not native.get("passed")
        or not coverage.get("passed")
        or any(
            coverage.get(field) is not True
            for field in (
                "ownership",
                "boundaries",
                "analytic_reference",
                "errors",
                "live_fallback",
            )
        )
    ):
        raise ValueError("native behavioral coverage is incomplete")
    if not coverage.get("evidence"):
        raise ValueError("native coverage requires hashed fixture evidence")
    for ref in coverage["evidence"]:
        if digest(ref["path"]) != ref["sha256"] or not Path(ref["path"]).stat().st_size:
            raise ValueError("native fixture evidence changed or empty")
    exports = []
    for ref in native.get("exports", []):
        export = checked(ref)
        export["source_id"] = build["source_id"]
        if {(case["id"], case["precision"]) for case in export["cases"]} != {
            (case, precision) for case in cases for precision in ("float", "double")
        }:
            raise ValueError("native precision/case coverage incomplete")
        exports.append(export)
    from equivalence import compare_snapshots

    if len(exports) != 2 or not compare_snapshots(*exports, policies)["passed"]:
        raise ValueError("native exact repeats failed")


def baseline_v4(output, state):
    from confirmation import raw_sampling
    from measurement import derive_raw
    from timing import aggregate_isolated

    budget = finite(
        state["config"].get("baseline_budget_seconds"), "explicit baseline budget", positive=True
    )
    state.setdefault("baseline_deadline", time.time() + budget)
    save(output, state)
    for item in state["operators"]:
        if not item["recipe"]:
            continue
        name = item["operator"]
        started = time.time()
        try:
            cfg, build = leaf_config(state, name), build_for(state, name)
            if cfg["measurement_policy_version"] != 4:
                raise ValueError("do not mix legacy calibration and version-4 baseline policies")
            identity(build)
            if item.get("baseline"):
                native_record(cfg, item["baseline"], name)
                checked(item["baseline"]["screening"])
                continue
            directory = Path(output) / name / "baseline"
            directory.mkdir(parents=True, exist_ok=True)
            validation = state["config"].get("verification", {}).get(name, {})
            native_baseline(
                build,
                cfg,
                directory,
                validation.get("correctness_command"),
                validation.get("coverage"),
                state["baseline_deadline"],
            )
            sampling = raw_sampling(cfg["sampling"])
            sampling["repeats"] = 1
            for override in sampling["case_overrides"].values():
                override["repeats"] = 1
            reports, children = [], []
            with device_lock(build.get("device_id", "cuda-visible-device-0")):
                for case in sampling["cases"].split(","):
                    settings = sampling | sampling["case_overrides"].get(case, {})
                    path = directory / (case + "-screen.json")
                    receipt = path.with_suffix(".receipt.json")
                    if receipt.exists():
                        checked(read_json(receipt))
                    else:
                        if path.exists():
                            raise ValueError("unsealed baseline screen; quarantine before retry")
                        seconds = state["baseline_deadline"] - time.time()
                        if seconds <= 0:
                            raise TimeoutError("baseline budget exhausted")
                        clocks = clock_state()
                        if (
                            cfg["clock_configuration"] is not None
                            and clocks["configuration"] != cfg["clock_configuration"]
                        ):
                            raise ValueError("baseline deployment configuration changed")
                        execute(
                            build["command"]
                            + [
                                "--scope",
                                sampling["scope"],
                                "--cases",
                                case,
                                "--warmup",
                                str(settings["warmup"]),
                                "--frames",
                                str(settings["frames"]),
                                "--repeats",
                                "1",
                                "--raw-frame-times",
                                "--output",
                                str(path),
                            ],
                            build["cwd"],
                            path.with_suffix(".log"),
                            path.with_suffix(".process.json"),
                            min(600, seconds),
                        )
                        if clock_state()["configuration"] != clocks["configuration"]:
                            raise ValueError("baseline clock context changed during screen")
                        write_json(receipt, reference(path))
                    reports.append(derive_raw(read_json(path), sampling))
                    children.append(reference(path))
            preview = aggregate_isolated(reports, sampling, sampling["cases"].split(","))
            preview["isolation"] = {"children": children, "inventory": sampling["cases"].split(",")}
            write_json(directory / "baseline-screen.json", preview)
            item["baseline"] = {
                "config": cfg,
                "build": build,
                "native": reference(directory / "native-review.json"),
                "screening": reference(directory / "baseline-screen.json"),
            }
            native_record(cfg, item["baseline"], name)
            item.update(
                status="baseline_ready",
                gaps=["Baseline screen only; candidate confirmation pending"],
            )
        except (ValueError, OSError, subprocess.SubprocessError, TimeoutError) as error:
            item.update(status="blocked", gaps=[str(error)])
        state["events"].append(
            {
                "phase": "baseline_screen",
                "operator": name,
                "started_at": started,
                "finished_at": time.time(),
                "status": item["status"],
            }
        )
        save(output, state)
    state["phase"] = "baseline_complete"
    save(output, state)
    return state


def baseline(output):
    state = load(output)
    if any(
        leaf_config(state, item["operator"])["measurement_policy_version"] == 4
        for item in state["operators"]
        if item["recipe"]
    ):
        return baseline_v4(output, state)
    state["phase"] = "baseline_running"
    save(output, state)
    for item in state["operators"]:
        if not item["recipe"]:
            continue
        name = item["operator"]
        started = time.time()
        try:
            config = leaf_config(state, name)
            build = build_for(state, name)
            current = identity(build)
            if item.get("measurement_identity") not in (None, current):
                raise ValueError("measurement identity changed")
            item["measurement_identity"] = current
            recommendation = item.get("qualification")
            if not recommendation:
                peers = [
                    x
                    for x in state["operators"]
                    if x["family"] == item["family"] and x.get("qualification", {}).get("passed")
                ]
                recommendation = peers[0]["qualification"] if peers else None
            if not recommendation or not recommendation.get("passed"):
                raise ValueError("no validated representative protocol; baseline blocked")
            checked(recommendation["frozen"])
            # Preserve full case selection; only sampling levels are inherited.
            config["sampling"] = protocol(
                config["sampling"]
                | {k: recommendation["sampling"][k] for k in ("frames", "repeats", "aggregation")}
            )
            directory = Path(output) / name / "baseline"
            directory.mkdir(parents=True, exist_ok=True)
            if item.get("baseline"):
                record = item["baseline"]
                checked(record["native"])
                stored = checked(record["calibration"])
                replay = review(checked(record["pairs"]), config, config["pairs"])
                if stored != replay or record["config"] != config:
                    raise ValueError("stale baseline evidence")
                continue
            validation = state["config"].get("verification", {}).get(name, {})
            native_baseline(
                build,
                config,
                directory,
                validation.get("correctness_command"),
                validation.get("coverage"),
            )
            index = run_pairs(
                build,
                build,
                directory / "aa",
                pairs=config["pairs"],
                sampling=config["sampling"],
                wall_time_s=state["config"].get("baseline_budget_seconds"),
                expected_clock_configuration=config.get("clock_configuration"),
            )
            result = review(index, config, config["pairs"])
            write_json(directory / "calibration.json", result)
            item["baseline"] = {
                "config": config,
                "build": build,
                "pairs": reference(directory / "aa/pairs.json"),
                "calibration": reference(directory / "calibration.json"),
                "native": reference(directory / "native-review.json"),
            }
            item["status"] = (
                "assessed"
                if result["passed"] and not result["insufficient_pair_budget"]
                else "blocked"
            )
            item["gaps"] = (
                [] if item["status"] == "assessed" else ["full-matrix calibration blocked"]
            )
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            item["status"] = "blocked"
            item["gaps"] = [str(error)]
            item["blocker_reasons"] = [
                {
                    "kind": (
                        "correctness"
                        if isinstance(error, CorrectnessFailure)
                        else "provenance_or_coverage"
                    ),
                    "detail": str(error),
                }
            ]
            if isinstance(error, CorrectnessFailure):
                item["blocker_reasons"][0]["evidence"] = error.evidence
        state["events"].append(
            {
                "phase": "baseline",
                "operator": name,
                "started_at": started,
                "finished_at": time.time(),
                "status": item["status"],
            }
        )
        save(output, state)
    state["phase"] = "baseline_complete"
    save(output, state)
    return state


def validate_consumer_baselines(config, records, verify_current=True):
    if set(records) != set(config["consumer_policies"]):
        raise ValueError("admission missing affected consumer baselines")
    from radar_perf import timing_comparison

    for name, record in records.items():
        if config["measurement_policy_version"] == 4:
            native_record(config, record, name, verify_current)
            continue
        if verify_current:
            identity(record["build"])
        else:
            verify_build(record["build"])
        if record["config"]["policies"] != config["consumer_policies"][name]:
            raise ValueError("consumer output policies differ from admission")
        native = checked(record["native"])
        stored = checked(record["calibration"])
        if (
            not native.get("passed")
            or not stored.get("passed")
            or stored.get("insufficient_pair_budget")
        ):
            raise ValueError("consumer correctness/calibration is blocked")
        result = timing_comparison(config, checked(record["pairs"]), None, calibrate=True)
        if not result["passed"] or result["insufficient_pair_budget"]:
            raise ValueError("consumer calibration replay blocked")


def validate_admission(config, verify_current=True):
    admission = config.get("benchmark_admission")
    if not admission:
        raise ValueError("optimization requires hash-bound native benchmark admission")
    value = checked(admission)
    if value.get("config_sha256") != config_digest(config):
        raise ValueError("admission configuration changed")
    if verify_current:
        identity(value["build"])
    else:
        verify_build(value["build"])
    if config["measurement_policy_version"] == 4:
        native_record(config, value, config["operator"], verify_current)
        validate_consumer_baselines(config, value.get("consumer_baselines", {}), verify_current)
        return
    calibration = checked(value["calibration"])
    native = checked(value["native"])
    if (
        not native.get("passed")
        or not calibration.get("passed")
        or calibration.get("insufficient_pair_budget")
    ):
        raise ValueError("full-case correctness/calibration is blocked")
    from qualification import review as replay

    index = checked(value["pairs"])
    fresh = replay(index, config, config["pairs"])
    if not fresh["passed"] or fresh["insufficient_pair_budget"]:
        raise ValueError("baseline calibration replay blocked")
    validate_consumer_baselines(config, value.get("consumer_baselines", {}), verify_current)


def config_digest(config):
    value = copy.deepcopy(config)
    value.pop("benchmark_admission", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_sessions(output, confirmed=False):
    state = load(output)
    if state["config"].get("mode", "assess") != "optimize":
        raise ValueError("assessment campaign cannot prepare production sessions")
    directory = Path(output) / "sessions"
    directory.mkdir(exist_ok=True)
    for item in state["operators"]:
        if item["status"] not in {"assessed", "baseline_ready"}:
            continue
        name = item["operator"]
        record = item["baseline"]
        config = record["config"] | {"mode": "optimize"}
        consumers = {}
        for consumer in config["consumer_policies"]:
            peer = next((x for x in state["operators"] if x["operator"] == consumer), None)
            if not peer or peer["status"] not in {"assessed", "baseline_ready"}:
                raise ValueError("affected consumer is not fully assessed: " + consumer)
            consumers[consumer] = peer["baseline"]
        validate_consumer_baselines(config, consumers)
        admission_path = directory / (name + "-admission.json")
        admission = record | {
            "config_sha256": config_digest(config),
            "consumer_baselines": consumers,
        }
        if admission_path.exists() and read_json(admission_path) != admission:
            raise ValueError("reviewed admission changed; prepare a fresh session")
        write_json(admission_path, admission)
        config["benchmark_admission"] = reference(admission_path)
        path = directory / (name + "-config.json")
        if path.exists() and read_json(path) != config:
            raise ValueError("reviewed session configuration changed")
        write_json(path, config)
        state["sessions"][name] = reference(path)
    if confirmed:
        from radar_perf import main as leaf_main

        for name, ref in state["sessions"].items():
            checked(ref)
            session = directory / (name + "-session.json")
            if not session.exists():
                leaf_main(["init", "--config", ref["path"], "--confirmed", "--out", str(session)])
        state["optimization_confirmed"] = True
    state["phase"] = "sessions_prepared"
    save(output, state)
    return state


def disposition(output, name, session_path):
    from radar_perf import load_session

    state = load(output)
    session = load_session(session_path)
    item = next((x for x in state["operators"] if x["operator"] == name), None)
    if item is None or session["config"]["operator"] != name or not session.get("stop_reason"):
        raise ValueError("disposition requires a sealed matching operator session")
    for candidate in session["candidates"]:
        for ref in candidate["gates"].values():
            checked(ref)
    if any(x.get("kept") for x in session["candidates"]):
        if not session.get("cumulative_timing"):
            raise ValueError("improved disposition requires final-versus-original timing")
        item["status"] = "improved"
    else:
        item["status"] = (
            "blocked" if session["coverage_gaps"] else "assessed_no_validated_improvement"
        )
    if name not in state["sessions"] or session["config"] != checked(state["sessions"][name]):
        raise ValueError("disposition differs from reviewed session configuration")
    item["disposition"] = reference(session_path)
    item["gaps"] = session["coverage_gaps"]
    state["phase"] = (
        "complete"
        if all(
            x["status"] in {"improved", "blocked", "assessed_no_validated_improvement"}
            for x in state["operators"]
        )
        else "assessment_in_progress"
    )
    save(output, state)
    return state


def timing_summary(item):
    outcome = checked(item["disposition"]) if item.get("disposition") else None
    cumulative = outcome.get("cumulative_timing") if outcome else None
    measured = cumulative or (
        checked(item["baseline"]["calibration"])
        if item.get("baseline", {}).get("calibration")
        else {}
    )
    if not measured and item.get("baseline", {}).get("screening"):
        preview = checked(item["baseline"]["screening"])
        measured = {
            "results": [
                row
                | {
                    "metric": "median_ms",
                    "case": [
                        row["operator"],
                        row["scenario_id"],
                        row["precision"],
                        row["size_id"],
                        row["bucket"],
                    ],
                    "before_median": row.get("median_ms"),
                }
                for row in preview["results"]
                if row.get("median_ms") is not None
            ]
        }
    rows = [
        row
        for row in measured.get("results", [])
        if row["metric"] == "median_ms" and row["case"][-1] == "compute"
    ]

    def bounds(field):
        values = [row[field] for row in rows]
        return "not collected" if not values else f"{min(values):.6g}–{max(values):.6g}"

    changes = sum(x.get("kept", False) for x in outcome["candidates"]) if outcome else 0
    return (
        bounds("before_median"),
        bounds("after_median") if cumulative else "not collected",
        changes,
    )


def report(output, destination):
    state = load(output)
    if state["phase"] in {"qualifying", "baseline_running"}:
        raise ValueError("publish reports only between sealed measurement sessions")
    lines = [
        "# Operator benchmark campaign",
        "",
        f'Phase: {state["phase"]}. Mode: {state["config"].get("mode", "assess")}.',
        "",
        "Assessment timings do not establish adequate performance for omitted or blocked cases.",
        "",
        "Baseline/final columns show ranges of casewise compute medians in milliseconds, not pooled run statistics.",
        "",
        "| Operator | Family | Disposition | Baseline medians ms | Final medians ms | Retained changes | Coverage gaps |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in state["operators"]:
        before, after, changes = timing_summary(item)
        lines.append(
            f'| {item["operator"]} | {item["family"]} | {item["status"]} | {before} | {after} | {changes} | {"; ".join(item["gaps"]) or "None recorded"} |'
        )
    lines += [
        "",
        "## Phase durations",
        "",
        "Job durations include process launch and orchestration, and do not measure agent working time.",
        "",
    ]
    for event in state["events"]:
        lines.append(
            f'- {event["operator"]} / {event["phase"]}: {event["finished_at"] - event["started_at"]:.1f} seconds ({event["status"]}).'
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in state["sessions"]:
        session_path = Path(output) / "sessions" / (name + "-session.json")
        if session_path.exists():
            session = read_json(session_path)
            if (session.get("plans") or session["candidates"]) and not session.get("stop_reason"):
                raise ValueError("publish reports only between sealed sessions")
    for item in state["operators"]:
        qualification = item.get("qualification", {})
        if qualification.get("validation"):
            lines += ["", "## " + item["operator"] + " qualification A/A", ""] + timing_table(
                checked(qualification["validation"])
            )
        if item.get("baseline", {}).get("calibration"):
            lines += ["", "## " + item["operator"] + " full-case A/A", ""] + timing_table(
                checked(item["baseline"]["calibration"])
            )
    lines += [
        "",
        "## Optimization handoff",
        "",
        "Agent applies operator-perf candidate declarations and gates. No production edits are performed by this tool.",
        "",
        "Opportunity ranking requires attributed recurring work, removable fraction, affected cases and mechanism confidence; absolute latency alone is insufficient.",
        "",
        "Final gains require direct paired original/current-best measurements; intermediate percentages are never multiplied.",
        "",
    ]
    for item in state["operators"]:
        if item.get("disposition"):
            from radar_perf import render_report

            body = render_report(checked(item["disposition"]))
        else:
            body = "# " + item["class"] + " assessment\n\nDisposition: " + item["status"] + ".\n\n"
            body += "Coverage gaps: " + ("; ".join(item["gaps"]) or "None recorded") + ".\n\n"
            body += "No production candidates attempted. No cumulative gain claimed.\n\n"
            body += "Unresolved correctness findings: inspect recorded blockers; unverified scopes remain unknown.\n\n"
            for reason in item.get("blocker_reasons", []):
                body += f'- {reason["kind"]}: {reason["detail"]}. Evidence: `{reason.get("evidence", "none")}`.\n'
            body += "\n"
            body += "Artifacts: `" + str(Path(output) / item["operator"]) + "`.\n\n"
            for label, ref in (
                ("Representative A/A", item.get("qualification", {}).get("validation")),
                ("Full-case A/A", item.get("baseline", {}).get("calibration")),
            ):
                if ref:
                    body += "## " + label + "\n\n" + "\n".join(timing_table(checked(ref))) + "\n\n"
        (destination.parent / (item["operator"] + ".md")).write_text(body.rstrip() + "\n")
    destination.write_text("\n".join(lines), encoding="utf-8")
    write_json(
        Path(output) / "publication.json",
        {
            "report": reference(destination),
            "whole_checkout_source_id": fingerprint(state["config"]["root"]),
            "note": "Report publication does not rewrite preserved measurement manifests.",
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reuse_parser = commands.add_parser("reuse-build")
    reuse_parser.add_argument("--build", required=True)
    reuse_parser.add_argument("--out", required=True)
    discover_parser = commands.add_parser("discover")
    discover_parser.add_argument("--config", required=True)
    discover_parser.add_argument("--out", required=True)
    for name in (
        "qualify",
        "baseline",
        "prepare-sessions",
        "optimize",
        "status",
        "resume",
        "report",
        "disposition",
    ):
        command = commands.add_parser(name)
        command.add_argument("--campaign", required=True)
        if name in {"prepare-sessions", "optimize"}:
            command.add_argument("--confirmed", action="store_true")
        if name == "disposition":
            command.add_argument("--operator", required=True)
            command.add_argument("--session", required=True)
        if name == "resume":
            command.add_argument("--phase", choices=["qualify", "baseline"], required=True)
        if name == "report":
            command.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "reuse-build":
        value = reuse_build(args.build, args.out)
        print(
            json.dumps(
                {"source_id": value["source_id"], "receipt": value["reuse_receipt"]}, indent=2
            )
        )
        return 0
    if args.command == "discover":
        state = create(read_json(args.config), args.out)
    elif args.command == "qualify" or args.command == "resume" and args.phase == "qualify":
        state = qualify_campaign(args.campaign)
    elif args.command == "baseline" or args.command == "resume" and args.phase == "baseline":
        state = baseline(args.campaign)
    elif args.command in {"prepare-sessions", "optimize"}:
        state = prepare_sessions(args.campaign, args.confirmed)
    elif args.command == "disposition":
        state = disposition(args.campaign, args.operator, args.session)
    elif args.command == "report":
        report(args.campaign, args.out)
        return 0
    else:
        state = load(args.campaign)
    print(
        json.dumps(
            {
                "phase": state["phase"],
                "operators": [
                    {"operator": x["operator"], "status": x["status"], "gaps": x["gaps"]}
                    for x in state["operators"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    main()
