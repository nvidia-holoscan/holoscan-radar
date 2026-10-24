# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-verified paired quantile and chronological-window exploration, never a gate."""

import argparse
import math
from pathlib import Path

from artifacts import digest, read_json, write_json
from frame_count import load_pairs, summarize


def window_runs(runs, start, frames):
    """Slice a zero-based, half-open measured window without dropping observations."""
    if (
        type(start) is not int
        or type(frames) is not int
        or start < 0
        or frames < 1
        or not runs
        or any(len(run) < start + frames for run in runs)
    ):
        raise ValueError("window exceeds measured run or has invalid bounds")
    return [run[start : start + frames] for run in runs]


def window_pairs(pairs, before, after=None):
    """Retain original pairs; temporal contrasts match all runs within each pair."""
    if after is None:
        return [
            [window_runs(side, before["start"], before["frames"]) for side in pair]
            for pair in pairs
        ]
    if before["frames"] != after["frames"]:
        raise ValueError("temporal contrasts require equal-length windows")
    return [
        [
            window_runs([run for side in pair for run in side], w["start"], w["frames"])
            for w in (before, after)
        ]
        for pair in pairs
    ]


def analyze(spec):
    windows = spec["windows"]
    datasets = spec["datasets"]
    quantiles = spec.get("quantiles", [0.5, 0.9, 0.99])
    contrasts = spec.get("contrasts", [])
    bound = spec["dataset_family_bound"]
    prior = spec.get("prior_family_slots", 0)
    if (
        not windows
        or not datasets
        or type(bound) is not int
        or bound < len(datasets)
        or type(prior) is not int
        or prior < 0
        or not quantiles
        or any(type(q) not in (float, int) or not 0 < q < 1 for q in quantiles)
        or len(set(quantiles)) != len(quantiles)
    ):
        raise ValueError("distinct interior quantiles, windows and adequate family bound required")
    by_name = {}
    for window in windows:
        name = window["id"]
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("distinct nonempty window IDs required")
        if (
            type(window["start"]) is not int
            or type(window["frames"]) is not int
            or window["start"] < 0
            or window["frames"] < 1
        ):
            raise ValueError("invalid window bounds")
        by_name[name] = window
    jobs = [(w["id"], w, None) for w in windows]
    seen = set()
    comparison_names = set(by_name)
    for contrast in contrasts:
        before, after = contrast["before"], contrast["after"]
        if before not in by_name or after not in by_name or before == after:
            raise ValueError("contrast requires two distinct declared windows")
        if (before, after) in seen:
            raise ValueError("duplicate temporal contrast")
        seen.add((before, after))
        a, b = by_name[before], by_name[after]
        if a["frames"] != b["frames"]:
            raise ValueError("temporal contrasts require equal-length windows")
        name = f"{after}/{before}"
        if name in comparison_names:
            raise ValueError(f"comparison identity collision: {name}")
        comparison_names.add(name)
        jobs.append((name, a, b))
    family = prior + bound * len(jobs) * len(quantiles) * 2
    rows, unavailable = [], []
    ids = set()
    for dataset in datasets:
        if not isinstance(dataset["id"], str) or not dataset["id"] or dataset["id"] in ids:
            raise ValueError("distinct nonempty dataset IDs required")
        ids.add(dataset["id"])
        observations, sampling, identity = load_pairs(dataset["pairs"])
        for name, before, after in jobs:
            if any(
                w["start"] + w["frames"] > sampling["frames"]
                for w in (before, after)
                if w is not None
            ):
                unavailable.append({"dataset": dataset["id"], "comparison": name})
                continue
            for precision, pairs in observations.items():
                selected = window_pairs(pairs, before, after)
                for row in summarize(selected, before["frames"], quantiles, family):
                    lo, hi = row["simultaneous_ci95_pct"]
                    row.update(
                        dataset=dataset["id"],
                        case=list(identity),
                        precision=precision,
                        comparison=name,
                        kind="temporal" if after is not None else "between_sides",
                        before_window=before,
                        after_window=after or before,
                        original_repeats_per_side=sampling["repeats"],
                        directional_evidence=(
                            "insufficient_interval_resolution"
                            if lo is None or hi is None
                            else (
                                "detected_slowdown"
                                if lo > 0
                                else "detected_speedup" if hi < 0 else "inconclusive"
                            )
                        ),
                    )
                    rows.append(row)
    return {
        "analysis_version": 1,
        "status": "exploration_only_not_qualification_or_acceptance",
        "method": "Equal-run mixture quantiles; median paired log ratios; exact sign/order-statistic simultaneous 95% intervals. Independent pairs and stable population median of contrasts assumed.",
        "orientation": "Positive means after/A2 slower; temporal contrasts pool both sides within each original pair.",
        "family_size": family,
        "minimum_resolvable_pairs": math.ceil(math.log2(40 * family)),
        "configuration": spec,
        "results": rows,
        "unavailable": unavailable,
        "limits": "No independent-frame assumption, extrapolation, exclusions, qualified steady cutoff, gate decision or population-tail guarantee. Selected policies require fresh disjoint validation.",
    }


def render_report(result):
    lines = [
        "# Paired distribution exploration",
        "",
        "Exploratory only; no qualification, acceptance or steady-state claim.",
        "Positive change means after/A2 slower. Absolute values are medians of pair quantiles.",
        f"Simultaneous 95% interval family: {result['family_size']}.",
        "",
        "| Dataset | Precision | Comparison | Quantile | Before µs | After µs | Change % | CI % | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["results"]:
        interval = ", ".join(
            "unbounded" if v is None else f"{v:+.3f}" for v in row["simultaneous_ci95_pct"]
        )
        lines.append(
            f"| {row['dataset']} | {row['precision']} | {row['comparison']} | {row['quantile']:.3g} | "
            f"{row['a1_quantile_median_us']:.3f} | {row['a2_quantile_median_us']:.3f} | "
            f"{row['median_change_pct']:+.3f} | [{interval}] | {row['directional_evidence']} |"
        )
    lines += ["", "Unmeasured comparisons: " + str(len(result["unavailable"])) + ".", ""]
    for row in result["unavailable"]:
        lines.append(f"- {row['dataset']} / {row['comparison']}: outside captured run.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    if Path(args.out).exists() or args.report and Path(args.report).exists():
        raise ValueError("refusing to overwrite analysis")
    if args.report and Path(args.report).resolve() == Path(args.out).resolve():
        raise ValueError("JSON and report destinations must differ")
    result = analyze(read_json(args.spec))
    result["spec_reference"] = {"path": str(Path(args.spec).resolve()), "sha256": digest(args.spec)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    if args.report:
        Path(args.report).write_text(render_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
