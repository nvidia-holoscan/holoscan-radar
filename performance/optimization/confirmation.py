# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version-four randomized screens and held-out, joint A/A–B/B–A/B confirmation."""

import copy
import math
import random
import re
import statistics
import time
from pathlib import Path

from artifacts import finite, read_json, write_json
from diagnostics import clock_state
from measurement import derive_raw, device_lock, execute
from measurement import protocol as raw_protocol
from qualification import checked, reference
from timing import compatible, median_interval, quantile, report_rows, verify_build

PAIRS = (("AABB", "BBAA"), ("ABAB", "BABA"), ("ABBA", "BAAB"))
QUANTILES = {"median_ms": 0.5, "p90_ms": 0.9, "p99_ms": 0.99}


def protocol(value):
    value = copy.deepcopy(value)
    if value.get("repeats", 2) != 2:
        raise ValueError(
            "version 4 requires explicit two-runs-per-build confirmation, not legacy repeats"
        )
    seed = value.pop("random_seed", None)
    order = value.pop("pairing_order", "randomized_four_run")
    aggregation = value.pop("aggregation", "mixture_quantiles")
    if (
        type(seed) is not int
        or order != "randomized_four_run"
        or aggregation != "mixture_quantiles"
    ):
        raise ValueError("version 4 requires a random seed and randomized mixture-quantile blocks")
    value["repeats"] = 2
    result = raw_protocol(value)
    if not result.get("cases") or len(set(result["cases"].split(","))) != len(
        result["cases"].split(",")
    ):
        raise ValueError("version 4 requires distinct explicit case IDs")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in result["cases"].split(",")):
        raise ValueError("invalid confirmation case ID")
    if any(override.get("repeats", 2) != 2 for override in result["case_overrides"].values()):
        raise ValueError("case overrides cannot change confirmation run counts")
    if set(result["case_overrides"]) - set(result["cases"].split(",")):
        raise ValueError("override names an undeclared confirmation case")
    result.update(random_seed=seed, pairing_order=order, aggregation=aggregation)
    return result


def raw_sampling(sampling):
    value = {k: v for k, v in sampling.items() if k != "random_seed"}
    value.update(pairing_order="adjacent_repeat", aggregation="median")
    return raw_protocol(value)


def schedule(blocks, seed, phase):
    rng = random.Random(seed)
    result = []
    while len(result) < blocks:
        first, second = rng.choice(PAIRS if phase == "confirmation" else (("AB", "BA"),))
        pair = [first, second]
        rng.shuffle(pair)
        result.extend(pair)
    return result[:blocks]


def collect(
    before,
    after,
    output,
    blocks,
    sampling,
    phase,
    timeout_s,
    wall_time_s,
    expected_clocks=None,
    plan=None,
    screening=(),
):
    """Resume sealed units only; freeze identities/order before any measured launch."""
    if phase not in {"screen", "confirmation"} or type(blocks) is not int or blocks < 2:
        raise ValueError("declare screen/confirmation and at least two blocks")
    sampling = protocol(sampling)
    compatible(before, after)
    if before.get("device_id", "cuda-visible-device-0") != after.get(
        "device_id", "cuda-visible-device-0"
    ):
        raise ValueError("build device identities differ")
    finite(wall_time_s, "explicit measurement budget", positive=True)
    finite(timeout_s, "process timeout", positive=True)
    if phase == "confirmation" and before["source_id"] == after["source_id"]:
        raise ValueError("confirmation must compare distinct original and candidate source states")
    if phase == "confirmation" and (not plan or not screening):
        raise ValueError("confirmation requires a frozen plan and disjoint screening evidence")
    if phase == "screen" and (plan is None or screening):
        raise ValueError("screening requires a plan and cannot consume confirmation inputs")
    plan_ref = reference(plan)
    declared = checked(plan_ref)
    if declared["before_source_id"] != before["source_id"]:
        raise ValueError("plan baseline differs from confirmation comparator")
    screens = [reference(p) for p in screening]
    for ref in screens:
        screen = checked(ref)
        replay(screen, require_phase="screen")
        if screen["before"] != before or screen["after"] != after or screen["plan"] != plan_ref:
            raise ValueError("screening belongs to another candidate or comparator")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "blocks.json"
    resolved_clocks = copy.deepcopy(
        expected_clocks if expected_clocks is not None else clock_state()["configuration"]
    )
    frozen = {
        "measurement_policy_version": 4,
        "phase": phase,
        "before": before,
        "after": after,
        "sampling": sampling,
        "requested_blocks": blocks,
        "plan": plan_ref,
        "screening": screens,
        "schedule": schedule(blocks, sampling["random_seed"], phase),
        "expected_clocks": resolved_clocks,
    }
    index = read_json(path) if path.exists() else frozen | {"frozen_at": time.time(), "blocks": []}
    if any(index.get(k) != v for k, v in frozen.items()):
        raise ValueError("resume confirmation identity or schedule changed")
    if any(checked(r)["finished_at"] > index["frozen_at"] for r in screens):
        raise ValueError("confirmation must be frozen after screening finishes")
    deadline = time.monotonic() + wall_time_s
    cases = sampling["cases"].split(",")
    with device_lock(before.get("device_id", "cuda-visible-device-0")):
        for number, order in enumerate(index["schedule"]):
            units = []
            for case in cases:
                for position, side in enumerate(order):
                    build = before if side == "A" else after
                    verify_build(build)
                    remaining = min(timeout_s, deadline - time.monotonic())
                    if remaining <= 0:
                        raise TimeoutError("approved confirmation budget exhausted")
                    cfg = sampling | sampling["case_overrides"].get(case, {})
                    stem = output / f"{number:03d}-{case}-{position}-{side}"
                    report = Path(str(stem) + ".json")
                    receipt = Path(str(stem) + ".receipt.json")
                    if receipt.exists():
                        unit = read_json(receipt)
                        checked(unit["report"])
                        checked(unit["process"])
                    else:
                        if report.exists():
                            raise ValueError(
                                "unsealed confirmation output; quarantine before retry"
                            )
                        clocks = copy.deepcopy(clock_state())
                        if clocks["configuration"] != index["expected_clocks"]:
                            raise ValueError("clock configuration changed")
                        argv = build["command"] + [
                            "--scope",
                            sampling["scope"],
                            "--cases",
                            case,
                            "--warmup",
                            str(cfg["warmup"]),
                            "--frames",
                            str(cfg["frames"]),
                            "--repeats",
                            "1",
                            "--raw-frame-times",
                            "--quiet",
                            "--output",
                            str(report),
                        ]
                        process = Path(str(stem) + ".process.json")
                        write_json(path, index)
                        execute(argv, build["cwd"], Path(str(stem) + ".log"), process, remaining)
                        after_clocks = copy.deepcopy(clock_state())
                        if clocks["configuration"] != after_clocks["configuration"]:
                            raise ValueError("clock configuration changed during run")
                        data = derive_raw(read_json(report), raw_sampling(sampling))
                        report_rows(data, build)
                        unit = {
                            "case": case,
                            "position": position,
                            "side": side,
                            "report": reference(report),
                            "process": reference(process),
                            "clock_before": clocks,
                            "clock_after": after_clocks,
                        }
                        write_json(receipt, unit)
                    if (unit["case"], unit["position"], unit["side"]) != (case, position, side):
                        raise ValueError("sealed unit assignment changed")
                    units.append(unit)
            block = {"number": number, "order": order, "runs": units}
            if number < len(index["blocks"]):
                if index["blocks"][number] != block:
                    raise ValueError("sealed confirmation block changed")
            else:
                index["blocks"].append(block)
            write_json(path, index)
    index.setdefault("finished_at", time.time())
    write_json(path, index)
    replay(index, require_phase=phase)
    return index


def replay(index, require_phase="confirmation", seen=None):
    """Reconstruct raw blocks and reject reused observations, stale contexts or screens."""
    if index.get("measurement_policy_version") != 4 or index.get("phase") != require_phase:
        raise ValueError("acceptance requires fresh version-4 confirmation, not screening")
    sampling = protocol(index["sampling"])
    n = index["requested_blocks"]
    if type(n) is not int or n < 2:
        raise ValueError("confirmation requires at least two complete blocks")
    finite(index.get("frozen_at"), "confirmation freeze time", positive=True)
    finite(index.get("finished_at"), "confirmation completion time", positive=True)
    if index["finished_at"] < index["frozen_at"] or not isinstance(index["expected_clocks"], dict):
        raise ValueError("unsealed confirmation/context")
    if len(index["blocks"]) != n or index["schedule"] != schedule(
        n, sampling["random_seed"], require_phase
    ):
        raise ValueError("incomplete or changed confirmation schedule")
    compatible(index["before"], index["after"])
    for side in ("before", "after"):
        verify_build(index[side])
    plan = checked(index["plan"])
    if plan["before_source_id"] != index["before"]["source_id"]:
        raise ValueError("confirmation plan baseline mismatch")
    seen = set() if seen is None else seen
    if require_phase == "confirmation":
        if not index["screening"]:
            raise ValueError("held-out screening reference missing")
        for ref in index["screening"]:
            screen = checked(ref)
            replay(screen, require_phase="screen", seen=seen)
            if (
                any(screen[k] != index[k] for k in ("before", "after", "plan"))
                or screen["finished_at"] > index["frozen_at"]
            ):
                raise ValueError("screening/candidate identity or freeze boundary mismatch")
    result, platform = [], None
    contexts = {}
    cases = sampling["cases"].split(",")
    for number, block in enumerate(index["blocks"]):
        order = index["schedule"][number]
        if (
            block["number"] != number
            or block["order"] != order
            or len(block["runs"]) != len(cases) * len(order)
        ):
            raise ValueError("invalid confirmation block boundaries")
        groups = {}
        for unit, (case, pos) in zip(
            block["runs"], [(c, p) for c in cases for p in range(len(order))]
        ):
            if (unit["case"], unit["position"], unit["side"]) != (case, pos, order[pos]):
                raise ValueError("confirmation run assignment changed")
            ref = unit["report"]
            if ref["path"] in seen:
                raise ValueError("reused observations or screening leakage")
            seen.add(ref["path"])
            report = checked(ref)
            for child in report.get("precision_children", []):
                if child["path"] in seen:
                    raise ValueError("reused precision observations")
                seen.add(child["path"])
            process_key = "process:" + unit["process"]["path"]
            if process_key in seen:
                raise ValueError("reused process evidence")
            seen.add(process_key)
            process = checked(unit["process"])
            if (
                process.get("state") != "finished"
                or process.get("returncode") != 0
                or process.get("cleanup_unknown")
                or process["started_at"] < index["frozen_at"]
            ):
                raise ValueError("unsealed, failed or non-held-out run")
            process_identity = ("pid", process["pid"], process["started_at"])
            if process_identity in seen:
                raise ValueError("reused process identity")
            seen.add(process_identity)
            if not process["started_at"] <= process["finished_at"] <= index["finished_at"]:
                raise ValueError("process falls outside sealed confirmation boundaries")
            clocks = [unit["clock_before"]["configuration"], unit["clock_after"]["configuration"]]
            if (
                clocks[0] != clocks[1]
                or index["expected_clocks"] is not None
                and clocks[0] != index["expected_clocks"]
            ):
                raise ValueError("confirmation clock context changed")
            report = derive_raw(report, raw_sampling(sampling))
            settings = report["run_settings"]
            if (
                settings["warmup_frames"] != sampling["warmup"]
                or settings["scope"] != sampling["scope"]
            ):
                raise ValueError("confirmation timing boundaries changed")
            if platform is None:
                platform = report["platform"]
            if report["platform"] != platform:
                raise ValueError("confirmation hardware differs between runs")
            build = index["before"] if unit["side"] == "A" else index["after"]
            rows = report_rows(report, build)
            if {key[1] for key in rows} != {case}:
                raise ValueError("confirmation returned another case")
            for key, row in rows.items():
                context = {
                    k: row.get(k)
                    for k in (
                        "variant",
                        "input_shape",
                        "map_shape",
                        "live_port",
                        "override_context",
                    )
                }
                if key in contexts and contexts[key] != context:
                    raise ValueError("confirmation workload context differs")
                contexts[key] = context
                groups.setdefault(key, {"A": [], "B": []})[unit["side"]].append(row)
        if any(
            len(sides[side]) != order.count(side)
            for sides in groups.values()
            for side in ("A", "B")
        ):
            raise ValueError("missing precision/row in some confirmation runs")
        result.append(groups)
    if any(set(g) != set(result[0]) for g in result):
        raise ValueError("case/precision inventory changed between blocks")
    return result


def summary(rows, metric, early):
    base = metric.removeprefix("early_")
    if base in QUANTILES and all("raw_steady_ms" in row for row in rows):
        frames = [
            v
            for row in rows
            for v in row["raw_steady_ms"][: early if metric.startswith("early_") else None]
        ]
        return quantile(frames, QUANTILES[base])
    values = [row.get(metric) for row in rows]
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError("missing protected metric in confirmation run")
    return max(values) if metric == "max_ms" else statistics.median(values)


def compare(index, config, targets, require_improvement=True):
    groups = replay(index)
    sampling = config["sampling"]
    if index["sampling"] != sampling or index["requested_blocks"] != config["pairs"]:
        raise ValueError("confirmation differs from confirmed sampling or block count")
    if (
        config.get("clock_configuration") is not None
        and index["expected_clocks"] != config["clock_configuration"]
    ):
        raise ValueError("confirmation deployment context differs")
    metrics = sorted(set(config["protected_metrics"]) | {config["objective"]})
    results = []
    for key in sorted(groups[0]):
        precision = key[2]
        if precision not in config["timing_precisions"].get(
            index["before"]["operator"], ["float", "double"]
        ):
            raise ValueError("unexpected timing precision")
        cfg = sampling | sampling["case_overrides"].get(key[1], {})
        for metric in metrics:
            observations, controls = [], []
            for block in groups:
                a, b = block[key]["A"], block[key]["B"]
                av, bv = summary(a, metric, cfg["early_frames"]), summary(
                    b, metric, cfg["early_frames"]
                )
                if av is None and bv is None:
                    continue
                if av is None or bv is None:
                    raise ValueError("missing metric between builds")
                finite(av, metric, positive=True)
                finite(bv, metric, positive=True)
                observations.append((av, bv))
                controls.append(
                    {
                        side: 100
                        * math.expm1(
                            math.log(
                                summary(rs[1:], metric, cfg["early_frames"])
                                / summary(rs[:1], metric, cfg["early_frames"])
                            )
                        )
                        for side, rs in (("AA", a), ("BB", b))
                    }
                )
            if not observations:
                continue
            if len(observations) != len(groups):
                raise ValueError("missing metric in some blocks")
            alias = (
                metric.removeprefix("early_")
                if metric.startswith("early_") and cfg["early_frames"] == cfg["frames"]
                else None
            )
            results.append(
                {
                    "case": list(key),
                    "metric": metric,
                    "alias_of": alias,
                    "observations": observations,
                    "controls": controls,
                }
            )
    for case in sampling["cases"].split(","):
        observed = {r["metric"] for r in results if r["case"][1] == case}
        if set(metrics) - {"override_ms"} - observed:
            raise ValueError("required confirmation metrics missing")
        for metric in {row["metric"] for row in results if row["case"][1] == case}:
            required = set(
                config["timing_precisions"].get(index["before"]["operator"], ["float", "double"])
            )
            if required - {
                row["case"][2]
                for row in results
                if row["case"][1] == case and row["metric"] == metric
            }:
                raise ValueError("confirmation precision coverage missing")
    family = (
        max(1, sum(not r["alias_of"] for r in results))
        * config.get("campaign_family_size", 1)
        * (1 + len(config["consumer_policies"]))
        * (config["max_candidates"] + 1)
    )
    for row in results:
        observations = row.pop("observations")
        effects = [math.log(b / a) for a, b in observations]
        interval = median_interval(effects, family)
        lo, hi = [None if v is None else 100 * math.expm1(v) for v in interval]
        verdict = (
            "unresolved"
            if lo is None
            else (
                "regressed"
                if lo > 0
                else (
                    "improved"
                    if hi
                    < -config.get("metric_policies", {})
                    .get(row["metric"], {})
                    .get("min_effect_pct", config["min_effect_pct"])
                    else "inconclusive"
                )
            )
        )
        row.update(
            before_by_block_ms=[a for a, _ in observations],
            after_by_block_ms=[b for _, b in observations],
            block_effects_pct=[100 * math.expm1(x) for x in effects],
            before_median=statistics.median(a for a, _ in observations),
            after_median=statistics.median(b for _, b in observations),
            effect_pct=100 * math.expm1(statistics.median(effects)),
            familywise_ci95_pct=[lo, hi],
            verdict=verdict,
        )
    objective = [
        r
        for r in results
        if r["metric"] == config["objective"] and (not targets or r["case"][1] in targets)
    ]
    if not objective or set(targets) - {r["case"][1] for r in objective}:
        raise ValueError("confirmation target coverage missing")
    unresolved = any(r["verdict"] == "unresolved" for r in results)
    passed = (
        not unresolved
        and not any(r["verdict"] == "regressed" for r in results)
        and (not require_improvement or all(r["verdict"] == "improved" for r in objective))
    )
    return {
        "measurement_policy_version": 4,
        "passed": passed,
        "status": "validation_blocked" if unresolved else "pass" if passed else "unsuccessful",
        "results": results,
        "family_size": family,
        "minimum_resolvable_pairs": math.ceil(math.log2(40 * family)),
        "insufficient_pair_budget": unresolved,
        "pairs": len(groups),
        "metric": config["objective"],
        "operators": [index["before"]["operator"]],
        "case_ids": sampling["cases"].split(","),
        "orientation": "positive means slower",
        "control_policy": "AA/BB descriptive and dependent; no fixed dispersion veto",
        "method": "Equal-run mixture quantiles; exact paired-block median log-ratio intervals; campaign/candidate Bonferroni",
    }


def render_table(value):
    lines = [
        "Positive change means slower; controls share observations with A/B and have no dispersion gate.",
        "",
        "| Case | Precision | Metric | A ms | B ms | Change % | Familywise CI % | AA / BB P90 absolute change % | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in value["results"]:
        interval = row["familywise_ci95_pct"]
        aa = quantile([abs(c["AA"]) for c in row["controls"]], 0.9)
        bb = quantile([abs(c["BB"]) for c in row["controls"]], 0.9)
        lines.append(
            f'| {row["case"][1]} | {row["case"][2]} | {row["metric"]} | {row["before_median"]:.6g} | {row["after_median"]:.6g} | {row["effect_pct"]:+.3f} | {interval} | {aa:.3f} / {bb:.3f} | {row["verdict"]} |'
        )
    return lines


def budget_proposal(report, config):
    """Finite/third-order block resolution; a forecast, not a power claim."""
    if (
        report.get("schema_version") != 3
        or not report.get("results")
        or report["run_settings"].get("stopped_early")
    ):
        raise ValueError("budget requires a complete baseline preview")
    metrics = set(config["protected_metrics"]) | {config["objective"]}
    sampling = config["sampling"]
    endpoints = 0
    for row in report["results"]:
        if row.get("status") != "ok":
            raise ValueError("budget preview contains failed cases")
        cfg = sampling | sampling["case_overrides"].get(row["scenario_id"], {})
        endpoints += sum(
            row.get(m) is not None
            and not (m.startswith("early_") and cfg["frames"] == cfg["early_frames"])
            for m in metrics
        )
    if endpoints <= 0:
        raise ValueError("no measurable endpoints in preview")
    family = (
        endpoints
        * config.get("campaign_family_size", 1)
        * (1 + len(config["consumer_policies"]))
        * (config["max_candidates"] + 1)
    )
    minimum = math.ceil(math.log2(40 * family))
    blocks = max(20, minimum)
    while sum(math.comb(blocks, j) for j in range(3)) / 2**blocks > 0.025 / family:
        blocks += 1
    return {
        "family_size": family,
        "minimum_resolvable_pairs": minimum,
        "proposed_pairs": blocks,
        "confirmation_runs_per_case_precision": blocks * 4,
        "screen_runs_per_case_precision": config["screen_pairs"] * 2,
        "note": "Blocks are independent inference units; third-order bounds are not a power guarantee. Includes maximum candidate attempts and a final cumulative look; controls have no fixed agreement gate.",
    }
