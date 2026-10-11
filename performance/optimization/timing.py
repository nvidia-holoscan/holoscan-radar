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

METRICS = {"median_ms", "cold_ms", "setup_ms", "override_ms", "achieved_frames_per_second"}


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


def run_pairs(before, after, output, pairs=10, timeout_s=600, sampling=None, wall_time_s=None):
    if pairs < 2 or timeout_s <= 0:
        raise ValueError("need at least two pairs and a positive process timeout")
    compatible(before, after)
    deadline = (
        None
        if wall_time_s is None
        else time.monotonic() + finite(wall_time_s, "wall_time_s", positive=True)
    )
    sampling = sampling or {"scope": "overrides", "warmup": 5, "frames": 50, "repeats": 3}
    if set(sampling) - {"scope", "warmup", "frames", "repeats", "sizes"}:
        raise ValueError("unsupported sampling option")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = {"before": before, "after": after, "pairs": [], "sampling": sampling}
    write_json(output / "pairs.json", index)
    for pair in range(pairs):
        records = {}
        order = ("before", "after") if pair % 2 == 0 else ("after", "before")
        for side in order:
            build = index[side]
            verify_build(build)
            path = output / f"{pair:02d}-{side}.json"
            argv = list(build["command"])
            for name, value in sampling.items():
                argv.extend(["--" + name, str(value)])
            argv.extend(["--quiet", "--output", str(path)])
            remaining = (
                timeout_s if deadline is None else min(timeout_s, deadline - time.monotonic())
            )
            if remaining <= 0:
                raise ValueError("paired measurement wall-time limit exhausted")
            with path.with_suffix(".log").open("wb") as stream:
                subprocess.run(
                    argv,
                    cwd=build["cwd"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                    check=True,
                )
            report = read_json(path)
            report_rows(report, build)
            records[side] = {"path": str(path), "sha256": digest(path)}
        index["pairs"].append(records)
        write_json(output / "pairs.json", index)
    return index


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
):
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
            current = report_rows(report, index[side])
            for option, field in (
                ("warmup", "warmup_frames"),
                ("frames", "measured_frames"),
                ("repeats", "repeats"),
                ("scope", "scope"),
            ):
                if report["run_settings"][field] != sampling[option]:
                    raise ValueError(f"report differs from confirmed sampling: {option}")
            if "sizes" in sampling and {r["size_id"] for r in current.values()} != set(
                sampling["sizes"].split(",")
            ):
                raise ValueError("report differs from confirmed size profiles")
            if settings is None:
                settings, platform = report["run_settings"], report["platform"]
                keys = current.keys()
            if report["run_settings"] != settings or report["platform"] != platform:
                raise ValueError("platform or sampling changed across paired runs")
            if current.keys() != keys:
                raise ValueError("performance case set changed")
            sides.append(current)
        for key in keys:
            for name in ("variant", "input_shape", "map_shape", "live_port"):
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
            if lo is not None and lo > min_effect_pct
            else "regressed" if hi is not None and hi < 0 else "inconclusive"
        )
        values = [value for pair in observations for value in pair]
        result["aa_max_delta_pct"] = max(abs(value) for value in deltas)
        result["baseline_spread_pct"] = (
            100 * (max(values) - min(values)) / statistics.median(values)
        )
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
            if r["aa_max_delta_pct"] > max_aa_delta_pct
            or r["baseline_spread_pct"] > max_baseline_spread_pct
            or r["verdict"] in {"improved", "regressed"}
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
