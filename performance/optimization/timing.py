# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unprofiled alternating A/B runs and deterministic paired bootstrap gates."""

import math
import random
import statistics
import subprocess
import time
from pathlib import Path

from artifacts import digest, finite, read_json, write_json
from diagnostics import clock_state

METRICS = {
    "median_ms",
    "cold_ms",
    "setup_ms",
    "override_ms",
    "achieved_frames_per_second",
    "p90_ms",
    "p99_ms",
    "max_ms",
}


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def budget_proposal(report, protected_metrics, objective, operators=1):
    settings = report.get("run_settings", {})
    if (
        report.get("schema_version") != 3
        or settings.get("stopped_early")
        or not report.get("results")
        or settings.get("scenarios_completed", 0) <= 0
        or settings.get("scenarios_completed") != settings.get("scenarios_planned")
        or any(row.get("status") != "ok" for row in report["results"])
    ):
        raise ValueError("budget requires a complete successful schema-v3 preview")
    family = operators * sum(
        row.get(metric) is not None
        for row in report["results"]
        for metric in set(protected_metrics) | {objective}
    )
    if family <= 0 or operators < 1:
        raise ValueError("budget requires a nonempty measurable matrix")
    minimum = math.ceil(math.log2(40 * family))
    proposed = max(20, minimum)
    # Avoid starting with only the extreme order statistics as finite bounds.
    while sum(math.comb(proposed, j) for j in range(3)) / 2**proposed > 0.025 / family:
        proposed += 1
    return {
        "family_size": family,
        "minimum_resolvable_pairs": minimum,
        "proposed_pairs": proposed,
        "note": "Third-order bounds or better; not a power guarantee. Confirm before measuring.",
    }


def verify_build(build):
    for field in (
        "operator",
        "operator_class",
        "source_id",
        "git_commit",
        "recipe_sha256",
        "sdk_image",
        "build_settings",
        "cwd",
    ):
        if not build.get(field):
            raise ValueError(f"build manifest missing {field}")
    if not build.get("artifacts"):
        raise ValueError("build manifest must hash executable and runtime dependencies")
    for path, expected in build["artifacts"].items():
        if digest(path) != expected:
            raise ValueError(f"stale build artifact: {path}")
    if not build.get("command") or not all(isinstance(x, str) for x in build["command"]):
        raise ValueError("command must be a nonempty argv list (no shell string)")
    if Path(build["command"][0]).name == "docker" and not any(
        arg in {"-w", "--workdir"} or arg.startswith("--workdir=") for arg in build["command"]
    ):
        raise ValueError("Docker manifests require an explicit container workdir")
    if not Path(build["cwd"]).is_absolute() or not Path(build["cwd"]).is_dir():
        raise ValueError("build cwd must be an existing absolute directory")


def report_rows(report, build):
    if report.get("schema_version") != 3:
        raise ValueError("expected performance report schema 3")
    if report.get("provenance", {}).get("git_commit") != build["git_commit"]:
        raise ValueError("report commit does not match build manifest")
    settings = report["run_settings"]
    for name, minimum in (("warmup_frames", 0), ("measured_frames", 2), ("repeats", 1)):
        if type(settings.get(name)) is not int or settings[name] < minimum:
            raise ValueError(f"invalid report sampling: {name}")
    if (
        settings.get("stopped_early")
        or settings.get("scenarios_completed", 0) <= 0
        or settings["scenarios_completed"] != settings["scenarios_planned"]
    ):
        raise ValueError("incomplete performance report")
    result = {}
    for row in report["results"]:
        if row["operator"] != build["operator_class"]:
            raise ValueError("report operator differs from build identity")
        if row.get("status") != "ok":
            raise ValueError(f"failed/unmeasured scenario: {row.get('scenario_id')}")
        key = (row["operator"], row["scenario_id"], row["precision"], row["size_id"], row["bucket"])
        if key in result:
            raise ValueError(f"duplicate performance row: {key}")
        for metric in METRICS:
            if row.get(metric) is not None:
                finite(row[metric], metric, positive=True)
        result[key] = row
    if not result:
        raise ValueError("empty performance report")
    return result


def compatible(a, b):
    for field in ("operator", "operator_class", "recipe_sha256", "sdk_image", "build_settings"):
        if a[field] != b[field]:
            raise ValueError(f"A/B {field} mismatch")


def run_pairs(
    before,
    after,
    output,
    pairs=10,
    timeout_s=600,
    sampling=None,
    wall_time_s=None,
    expected_clock_configuration=None,
):
    if pairs < 2 or timeout_s <= 0:
        raise ValueError("need at least two pairs and a positive process timeout")
    compatible(before, after)
    deadline = (
        None
        if wall_time_s is None
        else time.monotonic() + finite(wall_time_s, "wall_time_s", positive=True)
    )
    sampling = sampling or {"scope": "overrides", "warmup": 5, "frames": 50, "repeats": 3}
    if set(sampling) - {"scope", "warmup", "frames", "repeats", "sizes", "cases", "process_mode"}:
        raise ValueError("unsupported sampling option")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = {"before": before, "after": after, "pairs": [], "sampling": sampling}
    initial_clocks = clock_state()
    expected_clocks = (
        initial_clocks["configuration"]
        if expected_clock_configuration is None
        else expected_clock_configuration
    )
    index["clocks"] = {"expected_configuration": expected_clocks, "snapshots": []}

    def check_clocks():
        snapshot = clock_state()
        if snapshot["configuration"] != expected_clocks:
            raise ValueError("clock/governor configuration changed; measurement blocked")
        index["clocks"]["snapshots"].append(snapshot | {"timestamp": time.time()})

    check_clocks()
    write_json(output / "pairs.json", index)
    isolated = sampling.get("process_mode", "bulk") == "isolated"
    inventory = None
    if isolated:
        for build in (before, after):
            verify_build(build)
            argv = list(build["command"]) + ["--list-cases", "--scope", sampling["scope"]]
            for name in ("sizes", "cases"):
                if name in sampling:
                    argv += ["--" + name, sampling[name]]
            remaining = (
                timeout_s if deadline is None else min(timeout_s, deadline - time.monotonic())
            )
            if remaining <= 0:
                raise ValueError("paired measurement wall-time limit exhausted")
            result = subprocess.run(
                argv,
                cwd=build["cwd"],
                capture_output=True,
                text=True,
                check=True,
                timeout=remaining,
            )
            cases = [line.split("\t")[0] for line in result.stdout.splitlines() if line.strip()]
            if not cases or len(cases) != len(set(cases)):
                raise ValueError("invalid/duplicate case discovery")
            if inventory is not None and inventory != cases:
                raise ValueError("A/B selected case inventory differs")
            inventory = cases
    for pair in range(pairs):
        records = {}
        order = ("before", "after") if pair % 2 == 0 else ("after", "before")
        for side in order:
            build = index[side]
            verify_build(build)
            path = output / f"{pair:02d}-{side}.json"
            argv = list(build["command"])
            if isolated:
                children = []
                reports = []
                for case_number, case in enumerate(inventory):
                    for repeat in range(sampling["repeats"]):
                        child = output / f"{pair:02d}-{side}-{case_number:03d}-{repeat}.json"
                        child_sampling = {k: v for k, v in sampling.items() if k != "process_mode"}
                        child_sampling.update(cases=case, repeats=1)
                        command = list(build["command"])
                        for name, value in child_sampling.items():
                            command += ["--" + name, str(value)]
                        command += ["--quiet", "--output", str(child)]
                        remaining = (
                            timeout_s
                            if deadline is None
                            else min(timeout_s, deadline - time.monotonic())
                        )
                        if remaining <= 0:
                            raise ValueError("paired measurement wall-time limit exhausted")
                        with child.with_suffix(".log").open("wb") as stream:
                            check_clocks()
                            subprocess.run(
                                command,
                                cwd=build["cwd"],
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                                timeout=remaining,
                                check=True,
                            )
                            check_clocks()
                        report = read_json(child)
                        report_rows(report, build)
                        if {row["scenario_id"] for row in report["results"]} != {case}:
                            raise ValueError("isolated subprocess returned another case")
                        reports.append(report)
                        children.append({"path": str(child), "sha256": digest(child)})
                report = aggregate_isolated(reports, sampling, inventory)
                report["isolation"] = {"children": children, "inventory": inventory}
                write_json(path, report)
                records[side] = {"path": str(path), "sha256": digest(path)}
                continue
            for name, value in sampling.items():
                if name == "process_mode":
                    continue
                argv.extend(["--" + name, str(value)])
            argv.extend(["--quiet", "--output", str(path)])
            remaining = (
                timeout_s if deadline is None else min(timeout_s, deadline - time.monotonic())
            )
            if remaining <= 0:
                raise ValueError("paired measurement wall-time limit exhausted")
            with path.with_suffix(".log").open("wb") as stream:
                check_clocks()
                subprocess.run(
                    argv,
                    cwd=build["cwd"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                    check=True,
                )
                check_clocks()
            report = read_json(path)
            report_rows(report, build)
            records[side] = {"path": str(path), "sha256": digest(path)}
        index["pairs"].append(records)
        write_json(output / "pairs.json", index)
    return index


def aggregate_isolated(reports, sampling, inventory):
    import copy

    result = copy.deepcopy(reports[0])
    settings = result["run_settings"]
    settings.update(
        repeats=sampling["repeats"],
        scenarios_planned=len(inventory),
        scenarios_completed=len(inventory),
        selected_cases=list(inventory),
        process_mode="isolated",
        statistic="median_of_repeat_report_metrics",
        initialization_context="fresh_process_and_graph_per_case_repeat",
    )
    grouped = {}
    for report in reports:
        if (
            report["platform"] != result["platform"]
            or report["provenance"]["git_commit"] != result["provenance"]["git_commit"]
        ):
            raise ValueError("isolated process platform/provenance changed")
        for option, field in (
            ("warmup", "warmup_frames"),
            ("frames", "measured_frames"),
            ("scope", "scope"),
        ):
            if report["run_settings"].get(field) != sampling[option]:
                raise ValueError("isolated subprocess sampling changed")
        for field in ("sync_policy", "case_order", "stop_after_s"):
            if report["run_settings"].get(field) != reports[0]["run_settings"].get(field):
                raise ValueError("isolated subprocess synchronization/order settings changed")
        if (
            report["run_settings"].get("repeats") != 1
            or report["run_settings"].get("scenarios_completed") != 1
        ):
            raise ValueError("isolated subprocess must execute one case/repeat")
        for row in report["results"]:
            key = tuple(
                row[k] for k in ("operator", "scenario_id", "precision", "size_id", "bucket")
            )
            grouped.setdefault(key, []).append(row)
    result["results"] = []
    for rows in grouped.values():
        if len(rows) != sampling["repeats"]:
            raise ValueError("incomplete isolated repeat set")
        row = copy.deepcopy(rows[0])
        for entry in rows:
            for field in ("variant", "input_shape", "map_shape", "live_port", "override_context"):
                if entry.get(field) != row.get(field):
                    raise ValueError("isolated repeat case configuration changed")
        for metric in METRICS:
            values = [entry.get(metric) for entry in rows]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                raise ValueError("isolated metric missing in one repeat")
            row[metric] = max(values) if metric == "max_ms" else statistics.median(values)
        if "samples" in row:
            row["samples"] = sum(entry["samples"] for entry in rows)
        result["results"].append(row)
    sizes = {}
    for report in reports:
        for size in report.get("sizes", []):
            if size["id"] in sizes and sizes[size["id"]] != size:
                raise ValueError("isolated size definition changed")
            sizes[size["id"]] = size
    result["sizes"] = list(sizes.values())
    return result


def bootstrap(deltas, seed=1729, samples=10000):
    rng = random.Random(seed)
    trials = sorted(statistics.median(rng.choices(deltas, k=len(deltas))) for _ in range(samples))
    return [trials[int(0.025 * (samples - 1))], trials[int(0.975 * (samples - 1))]]


def median_interval(deltas, family_size):
    """Exact sign/order-statistic interval with Bonferroni familywise 95% coverage.

    None bounds mean the pair budget cannot resolve this family. Independent
    paired runs are required; dependence between cases does not defeat Bonferroni.
    """
    n = len(deltas)
    tail = 0.025 / family_size
    eligible = [
        k for k in range(n // 2) if sum(math.comb(n, j) for j in range(k + 1)) / 2**n <= tail
    ]
    if not eligible:
        return [None, None]
    ordered = sorted(deltas)
    k = max(eligible)
    return [ordered[k], ordered[n - k - 1]]


def compare_pairs(
    index,
    metric="median_ms",
    targets=None,
    protected_metrics=None,
    expected_pairs=10,
    expected_sampling=None,
    require_improvement=True,
    min_effect_pct=0.0,
    required_precisions=None,
    calibration=False,
    max_aa_delta_pct=5.0,
    max_baseline_spread_pct=10.0,
    family_multiplier=1,
    metric_policies=None,
    expected_clock_configuration=None,
):
    clock_record = index.get("clocks")
    if clock_record and not clock_record.get("snapshots"):
        raise ValueError("empty clock configuration evidence")
    if expected_clock_configuration is not None and (
        not clock_record or clock_record["expected_configuration"] != expected_clock_configuration
    ):
        raise ValueError("missing/different confirmed clock configuration")
    if clock_record and any(
        value["configuration"] != clock_record["expected_configuration"]
        for value in clock_record["snapshots"]
    ):
        raise ValueError("clock/governor configuration changed during paired measurements")
    if metric not in METRICS or len(index["pairs"]) < 2:
        raise ValueError("invalid objective metric or incomplete paired run")
    if len(index["pairs"]) != expected_pairs:
        raise ValueError("pair count differs from confirmed measurement budget")
    sampling = expected_sampling or index["sampling"]
    if index["sampling"] != sampling:
        raise ValueError("paired-run sampling differs from confirmed configuration")
    compatible(index["before"], index["after"])
    if protected_metrics is None:
        protected_metrics = ["median_ms", "cold_ms", "setup_ms", "override_ms"]
    if any(m not in METRICS for m in protected_metrics):
        raise ValueError("unknown protected metric")
    keys, rows, settings, platform = None, [], None, None
    for pair in index["pairs"]:
        sides = []
        for side in ("before", "after"):
            artifact = pair[side]
            if digest(artifact["path"]) != artifact["sha256"]:
                raise ValueError("raw timing report changed after measurement")
            report = read_json(artifact["path"])
            if sampling.get("process_mode") == "isolated":
                isolation = report.get("isolation", {})
                children = []
                if len({child["path"] for child in isolation.get("children", [])}) != len(
                    isolation.get("children", [])
                ):
                    raise ValueError("duplicate isolated subprocess artifact")
                for child in isolation.get("children", []):
                    if digest(child["path"]) != child["sha256"]:
                        raise ValueError("isolated raw report changed")
                    raw = read_json(child["path"])
                    report_rows(raw, index[side])
                    children.append(raw)
                if not children:
                    raise ValueError("missing isolated subprocess evidence")
                actual = aggregate_isolated(children, sampling, isolation["inventory"])
                actual["isolation"] = isolation
                if actual != report:
                    raise ValueError("isolated aggregate differs from raw reports")
            current = report_rows(report, index[side])
            for option, field in (
                ("warmup", "warmup_frames"),
                ("frames", "measured_frames"),
                ("repeats", "repeats"),
                ("scope", "scope"),
            ):
                if report["run_settings"][field] != sampling[option]:
                    raise ValueError(f"report differs from confirmed sampling: {option}")
            if (
                "sizes" in sampling
                and "cases" not in sampling
                and {r["size_id"] for r in current.values()} != set(sampling["sizes"].split(","))
            ):
                raise ValueError("report differs from confirmed size profiles")
            if "cases" in sampling and {r["scenario_id"] for r in current.values()} != set(
                sampling["cases"].split(",")
            ):
                raise ValueError("report differs from confirmed case filter")
            if settings is None:
                settings, platform = report["run_settings"], report["platform"]
                keys = current.keys()
            if report["run_settings"] != settings or report["platform"] != platform:
                raise ValueError("platform or sampling changed across paired runs")
            if current.keys() != keys:
                raise ValueError("performance case set changed")
            sides.append(current)
        for key in keys:
            for name in ("variant", "input_shape", "map_shape", "live_port", "override_context"):
                if sides[0][key].get(name) != sides[1][key].get(name):
                    raise ValueError(f"case configuration changed: {key}/{name}")
        rows.append(sides)
    results = []
    for key in sorted(keys):
        for measure in sorted(set(protected_metrics) | {metric}):
            observations = [[side[key].get(measure) for side in pair] for pair in rows]
            if all(a is None and b is None for a, b in observations):
                continue
            if any(a is None or b is None for a, b in observations):
                raise ValueError(f"missing metric: {key}/{measure}")
            direction = -1 if measure == "achieved_frames_per_second" else 1
            deltas = [direction * 100 * (a - b) / a for a, b in observations]
            interval = bootstrap(deltas)
            results.append(
                {
                    "case": list(key),
                    "metric": measure,
                    "improvement_pct": statistics.median(deltas),
                    "before_median": statistics.median(a for a, _ in observations),
                    "after_median": statistics.median(b for _, b in observations),
                    "ci95_pct": interval,
                    "verdict": (
                        "improved"
                        if interval[0] > 0
                        else "regressed" if interval[1] < 0 else "inconclusive"
                    ),
                }
            )
    if type(family_multiplier) is not int or family_multiplier < 1:
        raise ValueError("invalid family multiplier")
    family_size = len(results) * family_multiplier
    for result in results:
        key = tuple(result["case"])
        measure = result["metric"]
        observations = [[side[key].get(measure) for side in pair] for pair in rows]
        direction = -1 if measure == "achieved_frames_per_second" else 1
        deltas = [direction * 100 * (a - b) / a for a, b in observations]
        lo, hi = median_interval(deltas, family_size)
        result["familywise_ci95_pct"] = [lo, hi]
        result["verdict"] = (
            "improved"
            if lo is not None
            and lo > (metric_policies or {}).get(measure, {}).get("min_effect_pct", min_effect_pct)
            else "regressed" if hi is not None and hi < 0 else "inconclusive"
        )
        values = [value for pair in observations for value in pair]
        result["aa_max_delta_pct"] = max(abs(value) for value in deltas)
        result["baseline_spread_pct"] = (
            100 * (max(values) - min(values)) / statistics.median(values)
        )
        result["aa_p90_delta_pct"] = quantile([abs(value) for value in deltas], 0.9)
        result["baseline_iqr_pct"] = (
            100 * (quantile(values, 0.75) - quantile(values, 0.25)) / statistics.median(values)
        )
        result["stall_count"] = sum(value > 3 * statistics.median(values) for value in values)
    target_set = set(targets or [])
    present = {r["case"][1] for r in results if r["metric"] == metric}
    if target_set - present:
        raise ValueError("objective case missing or objective metric unavailable")
    objective = [
        r
        for r in results
        if r["metric"] == metric and (not target_set or r["case"][1] in target_set)
    ]
    if not objective:
        raise ValueError("no measurable objective cases")
    if required_precisions:
        for case, measure in {(r["case"][1], r["metric"]) for r in results}:
            observed = {
                r["case"][2] for r in results if r["case"][1] == case and r["metric"] == measure
            }
            if set(required_precisions) - observed:
                raise ValueError(f"performance precision coverage missing: {case}/{measure}")
    # Every confirmed target must improve; every measured case is protected.
    passed = (
        not require_improvement or all(r["verdict"] == "improved" for r in objective)
    ) and not any(
        r["verdict"] == "regressed" and r["metric"] in set(protected_metrics) | {metric}
        for r in results
    )
    insufficient = any(r["familywise_ci95_pct"][0] is None for r in results)
    passed = passed and not insufficient
    blocked = []
    if calibration:
        if index["before"]["source_id"] != index["after"]["source_id"]:
            raise ValueError("calibration must compare the same source state")
        blocked = [
            {"case": r["case"], "metric": r["metric"]}
            for r in results
            if r["aa_p90_delta_pct"]
            > (metric_policies or {})
            .get(r["metric"], {})
            .get("max_aa_p90_delta_pct", max_aa_delta_pct)
            or r["baseline_iqr_pct"]
            > (metric_policies or {})
            .get(r["metric"], {})
            .get("max_baseline_iqr_pct", max_baseline_spread_pct)
            or (
                r["familywise_ci95_pct"][0] is not None
                and (r["familywise_ci95_pct"][0] > 0 or r["familywise_ci95_pct"][1] < 0)
            )
        ]
        passed = not blocked
    return {
        "passed": passed,
        "status": "validation_blocked" if blocked else "pass" if passed else "unsuccessful",
        "blocked_cases": blocked,
        "family_size": family_size,
        "insufficient_pair_budget": insufficient,
        "minimum_resolvable_pairs": math.ceil(math.log2(40 * family_size)),
        "method": "Bonferroni exact paired-median intervals; bootstrap intervals descriptive",
        "precision_coverage": sorted({key[2] for key in keys}),
        "pairs": len(rows),
        "metric": metric,
        "results": results,
        "operators": [index["before"]["operator"]],
        "case_ids": sorted({key[1] for key in keys}),
    }
