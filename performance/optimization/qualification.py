# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded protocol exploration followed by immutable, disjoint A/A validation."""

import copy
import subprocess
import time
from pathlib import Path

from artifacts import digest, finite, read_json, write_json
from measurement import protocol
from timing import aggregate_isolated, budget_proposal, compare_pairs, run_pairs, verify_build


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def checked(ref):
    if not isinstance(ref, dict) or not {"path", "sha256"} <= ref.keys():
        raise ValueError("missing hash-bound evidence reference")
    if digest(ref["path"]) != ref["sha256"]:
        raise ValueError("evidence hash changed: " + ref["path"])
    return read_json(ref["path"])


def review(index, config, pairs):
    return compare_pairs(
        index,
        metric=config["objective"],
        protected_metrics=config["protected_metrics"],
        expected_pairs=pairs,
        expected_sampling=index["sampling"],
        require_improvement=False,
        required_precisions=config["timing_precisions"].get(
            config["operator"], ["float", "double"]
        ),
        calibration=True,
        min_effect_pct=config["min_effect_pct"],
        max_aa_delta_pct=config["max_aa_delta_pct"],
        max_baseline_spread_pct=config["max_baseline_spread_pct"],
        metric_policies=config["metric_policies"],
        family_multiplier=config.get("campaign_family_size", 1)
        * (config.get("campaign_family_size", 1) * (1 + len(config["consumer_policies"]))),
        expected_clock_configuration=config.get("clock_configuration"),
    )


def reaggregate(index, sampling, output):
    result = copy.deepcopy(index)
    result["sampling"] = sampling
    output.mkdir(exist_ok=True)
    for number, pair in enumerate(result["pairs"]):
        for side in ("before", "after"):
            report = checked(pair[side])
            isolation = report["isolation"]
            children = [checked(ref) for ref in isolation["children"]]
            aggregate = aggregate_isolated(children, sampling, isolation["inventory"])
            aggregate["isolation"] = isolation
            path = output / f"{number:03d}-{side}.json"
            if path.exists() and read_json(path) != aggregate:
                raise ValueError("exploratory aggregate changed")
            write_json(path, aggregate)
            pair[side] = reference(path)
    write_json(output / "pairs.json", result)
    return result


def forecast(sampling, pairs, case_count=1, precisions=2, launch_seconds=1.0, frame_ms=1.0):
    """Planning estimate, never timing evidence. Caller estimates are labeled assumptions."""
    for key, value in (("case_count", case_count), ("precisions", precisions)):
        if type(value) is not int or value < 1:
            raise ValueError("invalid forecast " + key)
    finite(launch_seconds, "launch_seconds", positive=True)
    finite(frame_ms, "frame_ms", positive=True)
    levels = [sampling] + [sampling | x for x in sampling.get("case_overrides", {}).values()]
    repeats = max(x["repeats"] for x in levels)
    frame_count = max(x["frames"] for x in levels)
    processes = 2 * pairs * repeats * case_count
    frames = processes * frame_count * precisions
    return {
        "processes": processes,
        "raw_frames": frames,
        "estimated_seconds": processes * launch_seconds + frames * frame_ms / 1000,
        "estimated_raw_bytes": frames * 24,
        "assumptions": {
            "case_count": case_count,
            "precisions": precisions,
            "launch_seconds": launch_seconds,
            "frame_ms": frame_ms,
        },
        "note": "Forecast, not evidence; excludes report/log overhead. Unknown costs may exhaust the budget.",
    }


def qualify(build, config, output, budget_seconds, exploration=None):
    budget_seconds = finite(budget_seconds, "qualification budget", positive=True)
    verify_build(build)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    exploration = {
        "frames": [50, 500, 2000],
        "repeats": [3, 7],
        "calibration_pairs": 10,
        "validation_pairs": None,
        "allow_mean": False,
    } | (exploration or {})
    if set(exploration) != {
        "frames",
        "repeats",
        "calibration_pairs",
        "validation_pairs",
        "allow_mean",
    }:
        raise ValueError("unknown exploration option")
    if type(exploration["allow_mean"]) is not bool or exploration["calibration_pairs"] is None:
        raise ValueError("invalid exploration aggregation/pair policy")
    for values, minimum in ((exploration["frames"], 50), (exploration["repeats"], 1)):
        if not values or any(type(x) is not int or x < minimum for x in values):
            raise ValueError("invalid exploration ladder")
    for key in ("calibration_pairs", "validation_pairs"):
        value = exploration[key]
        if value is not None and (type(value) is not int or value < 2):
            raise ValueError("invalid exploration pair budget")
    intent = {
        "build": build,
        "config": config,
        "exploration": exploration,
        "budget_seconds": budget_seconds,
    }
    intent_path = output / "intent.json"
    if intent_path.exists() and read_json(intent_path) != intent:
        raise ValueError("qualification intent changed; start a fresh qualification")
    if not intent_path.exists():
        write_json(intent_path, intent)
    result_path = output / "qualification.json"
    if result_path.exists():
        result = read_json(result_path)
        checked(result["validation"])
        frozen = checked(result["frozen"])
        for ref in frozen["calibration_evidence"]:
            calibration = checked(ref)
            review(calibration, config, len(calibration["pairs"]))
        replay = review(checked(result["validation_pairs"]), config, result["pairs"])
        if replay != checked(result["validation"]):
            raise ValueError("qualification replay differs")
        return result
    ledger_path = output / "budget.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"spent_seconds": 0}
    started = time.monotonic()

    def remaining():
        return budget_seconds - ledger["spent_seconds"] - (time.monotonic() - started)

    estimates = config.get("qualification_cost_estimate", {})
    estimate_args = {
        key: estimates[key]
        for key in ("case_count", "precisions", "launch_seconds", "frame_ms")
        if key in estimates
    }
    forecasts = [
        forecast(
            protocol(config["sampling"] | {"frames": f, "repeats": r}),
            exploration["calibration_pairs"],
            **estimate_args,
        )
        for f in exploration["frames"]
        for r in exploration["repeats"]
    ]
    write_json(
        output / "forecast.json",
        {
            "configurations": forecasts,
            "budget_seconds": budget_seconds,
            "estimated_exploration_seconds": sum(x["estimated_seconds"] for x in forecasts),
        },
    )
    candidates = []
    frozen_path = output / "frozen.json"
    try:
        if frozen_path.exists():
            frozen = read_json(frozen_path)
            for ref in frozen["calibration_evidence"]:
                checked(ref)
        else:
            exploration_deadline = max(0, remaining()) / 2
            exploration_start = time.monotonic()
            for frames in exploration["frames"]:
                for repeats in exploration["repeats"]:
                    cfg = protocol(config["sampling"] | {"frames": frames, "repeats": repeats})
                    path = output / f"calibration-f{frames}-r{repeats}"
                    report_path = path / "review.json"
                    if report_path.exists():
                        candidate = read_json(report_path)
                        checked(candidate["pairs"])
                        candidates.append(candidate)
                        if exploration["allow_mean"]:
                            mean_index = reaggregate(
                                checked(candidate["pairs"]),
                                candidate["sampling"] | {"aggregation": "mean"},
                                path / "mean",
                            )
                            candidates.append(
                                {
                                    "sampling": mean_index["sampling"],
                                    "review": review(
                                        mean_index, config, exploration["calibration_pairs"]
                                    ),
                                    "pairs": reference(path / "mean/pairs.json"),
                                }
                            )
                        continue
                    allowance = min(
                        remaining(), exploration_deadline - (time.monotonic() - exploration_start)
                    )
                    if allowance <= 0:
                        break
                    if (
                        forecast(cfg, exploration["calibration_pairs"], **estimate_args)[
                            "estimated_seconds"
                        ]
                        > allowance
                    ):
                        continue
                    try:
                        index = run_pairs(
                            build,
                            build,
                            path,
                            pairs=exploration["calibration_pairs"],
                            sampling=cfg,
                            wall_time_s=allowance,
                            expected_clock_configuration=config.get("clock_configuration"),
                        )
                    except (TimeoutError, subprocess.TimeoutExpired):
                        break
                    result = review(index, config, exploration["calibration_pairs"])
                    candidate = {
                        "sampling": cfg,
                        "review": result,
                        "pairs": reference(path / "pairs.json"),
                    }
                    write_json(report_path, candidate)
                    candidates.append(candidate)
                    if exploration["allow_mean"]:
                        mean_path = path / "mean"
                        mean_sampling = cfg | {"aggregation": "mean"}
                        mean_index = reaggregate(index, mean_sampling, mean_path)
                        candidates.append(
                            {
                                "sampling": mean_sampling,
                                "review": review(
                                    mean_index, config, exploration["calibration_pairs"]
                                ),
                                "pairs": reference(mean_path / "pairs.json"),
                            }
                        )
            if not candidates:
                return {
                    "passed": False,
                    "status": "budget_exhausted",
                    "scope": "no complete calibration",
                    "path": str(output),
                }

            def score(candidate):
                rows = candidate["review"]["results"]
                return (
                    len(candidate["review"]["blocked_cases"]),
                    sum(x["aa_p90_delta_pct"] for x in rows if x["metric"] == config["objective"]),
                    candidate["sampling"]["frames"] * candidate["sampling"]["repeats"],
                )

            selected = min(candidates, key=score)
            preview = checked(selected["pairs"])
            pair_report = checked(preview["pairs"][0]["before"])
            proposal = budget_proposal(
                pair_report,
                config["protected_metrics"],
                config["objective"],
                config.get("campaign_family_size", 1) * (1 + len(config["consumer_policies"])),
            )
            frozen = {
                "sampling": selected["sampling"],
                "pairs": exploration["validation_pairs"] or max(30, proposal["proposed_pairs"]),
                "calibration_evidence": [x["pairs"] for x in candidates],
                "budget_proposal": proposal,
                "selection_rule": "fewest blocked rows, then objective P90 delta, then sampling volume",
                "whole_checkout_source_id": build["source_id"],
                "validation_namespace": str(output / "validation"),
                "aggregation": selected["sampling"]["aggregation"],
                "exploratory_mean": exploration["allow_mean"],
            }
            write_json(frozen_path, frozen)
            write_json(output / "exploration.json", {"candidates": candidates})
        if remaining() <= 0:
            return {
                "passed": False,
                "status": "budget_exhausted",
                "scope": "held-out validation incomplete",
                "path": str(output),
            }
        index = run_pairs(
            build,
            build,
            output / "validation",
            pairs=frozen["pairs"],
            sampling=frozen["sampling"],
            wall_time_s=remaining(),
            expected_clock_configuration=config.get("clock_configuration"),
        )
        result = review(index, config, frozen["pairs"])
        write_json(output / "validation-review.json", result)
        cases = sorted({row["case"][1] for row in result["results"]})
        value = {
            "passed": result["passed"] and not result["insufficient_pair_budget"],
            "status": (
                "insufficient_interval_resolution"
                if result["insufficient_pair_budget"] and not result["blocked_cases"]
                else result["status"]
            ),
            "blocker_reasons": result["blocker_reasons"],
            "representative_only": True,
            "cases": cases,
            "operator": build["operator"],
            "source_id": build["source_id"],
            "build": build,
            "sampling": frozen["sampling"],
            "pairs": frozen["pairs"],
            "frozen": reference(frozen_path),
            "validation": reference(output / "validation-review.json"),
            "validation_pairs": reference(output / "validation/pairs.json"),
            "elapsed_seconds": ledger["spent_seconds"] + time.monotonic() - started,
            "note": "Representative qualification is not full-case calibration or production acceptance.",
        }
        write_json(result_path, value)
        return value
    except (TimeoutError, subprocess.TimeoutExpired):
        return {
            "passed": False,
            "status": "budget_exhausted",
            "scope": "incomplete measurement",
            "path": str(output),
        }
    finally:
        ledger["spent_seconds"] += time.monotonic() - started
        write_json(ledger_path, ledger)


def timing_table(review_result):
    labels = ("Before", "After") if review_result.get("calibration") else ("A1", "A2")
    lines = [
        f'Pairs: {review_result.get("pairs", "unknown")}; family size: {review_result.get("family_size", "unknown")}; '
        f'minimum resolvable pairs: {review_result.get("minimum_resolvable_pairs", "unknown")}.',
        "",
        f"| Case | Precision | Bucket | Metric | {labels[0]} ms | {labels[1]} ms | Paired P90 / limit | IQR / limit | Familywise interval | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    limits = {}
    for reason in review_result.get("blocker_reasons", []):
        if reason["kind"] == "dispersion":
            limits[(tuple(reason["case"]), reason["metric"], reason["statistic"])] = reason["limit"]
    for row in review_result.get("results", []):
        key = tuple(row["case"])

        def bound(field):
            value = row.get("limits", {}).get(field, limits.get((key, row["metric"], field)))
            return str(value) if value is not None else "see policy"

        lines.append(
            f'| {key[1]} | {key[2]} | {key[-1]} | {row["metric"]} | {row["before_median"]:.6g} | {row["after_median"]:.6g} | '
            f'{row["aa_p90_delta_pct"]:.3g}% / {bound("aa_p90_delta_pct")} | '
            f'{row["baseline_iqr_pct"]:.3g}% / {bound("baseline_iqr_pct")} | {row["familywise_ci95_pct"]} | {row.get("verdict", "unknown")} |'
        )
    for reason in review_result.get("blocker_reasons", []):
        lines += [
            "",
            f'- Blocker: {reason["kind"]}; {reason.get("case", "matrix")} / {reason.get("metric", "all metrics")}.',
        ]
    grouped = {}
    for regime in review_result.get("timing_regimes", []):
        key = tuple(regime["case"]), regime["side"]
        grouped.setdefault(key, []).extend(
            x for x in regime["diagnostics"] if "half_drift_pct" in x
        )
    if any(grouped.values()):
        lines += [
            "",
            "Frame-regime diagnostics are descriptive; no observations were discarded.",
            "",
            "| Case | Precision | Side | Half drift range % | Early bias range % | Raw min/max ms | Stall frames |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for (case, side), entries in sorted(grouped.items()):
            if not entries:
                continue

            def span(field):
                values = [x[field] for x in entries]
                return f"{min(values):.3g} to {max(values):.3g}"

            lines.append(
                f'| {case[1]} | {case[2]} | {labels[side]} | {span("half_drift_pct")} | {span("early_bias_pct")} | '
                f'{min(x["min_ms"] for x in entries):.6g} / {max(x["max_ms"] for x in entries):.6g} | {sum(x["stall_frames"] for x in entries)} |'
            )
    return lines
