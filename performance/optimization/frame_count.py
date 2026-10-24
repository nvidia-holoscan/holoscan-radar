# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline paired distribution exploration; never an acceptance or qualification gate."""

import argparse
import math
import statistics
from pathlib import Path

from artifacts import digest, read_json, write_json
from qualification import checked
from timing import median_interval, quantile


def run_curve(runs, frames, probabilities):
    """Equal-run empirical distribution; frames are not independent replicates."""
    if not runs or frames < 1 or any(len(run) < frames for run in runs):
        raise ValueError("requested prefix exceeds measured run length")
    values = [value for run in runs for value in run[:frames]]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("frame timings must be positive and finite")
    return [quantile(values, p) for p in probabilities]


def summarize(pairs, frames, probabilities, family_size):
    """Exact sign intervals for median paired log quantile ratios, not IID frames."""
    if not pairs or any(not 0 < p < 1 for p in probabilities):
        raise ValueError("nonempty pairs and interior quantiles required")
    curves = [[run_curve(side, frames, probabilities) for side in pair] for pair in pairs]
    result = []
    for index, probability in enumerate(probabilities):
        changes = [math.log(b[index] / a[index]) for a, b in curves]
        interval = median_interval(changes, family_size)
        result.append(
            {
                "quantile": probability,
                "frames_per_run": frames,
                "pairs": len(pairs),
                "repeats_per_side": len(pairs[0][0]),
                "a1_quantile_median_us": 1000 * statistics.median(a[index] for a, _ in curves),
                "a2_quantile_median_us": 1000 * statistics.median(b[index] for _, b in curves),
                "median_change_pct": 100 * math.expm1(statistics.median(changes)),
                "simultaneous_ci95_pct": [
                    None if v is None else 100 * math.expm1(v) for v in interval
                ],
                "p90_absolute_change_pct": quantile(
                    [100 * abs(math.expm1(v)) for v in changes], 0.9
                ),
                "p90_absolute_difference_us": quantile(
                    [1000 * abs(b[index] - a[index]) for a, b in curves], 0.9
                ),
                "nominal_upper_tail_frames_per_run": frames * (1 - probability),
                "max_observed_frame_us": 1000
                * max(
                    value
                    for pair in pairs
                    for side in pair
                    for run in side
                    for value in run[:frames]
                ),
            }
        )
    return result


def load_pairs(reference):
    """Read only complete v3 isolated reports and verify descendant hashes."""
    index = checked(reference)
    sampling = index["sampling"]
    if (
        index.get("measurement_policy_version") != 3
        or sampling.get("process_mode") != "isolated"
        or len(index["pairs"]) != index["requested_pairs"]
        or not index["pairs"]
    ):
        raise ValueError("complete isolated version-3 pairs required")
    frames, repeats = sampling["frames"], sampling["repeats"]
    data = {precision: [] for precision in ("float", "double")}
    seen = set()
    case_identity = None
    for pair in index["pairs"]:
        sides = {precision: [] for precision in data}
        for name in ("before", "after"):
            aggregate = checked(pair[name])
            children = aggregate["isolation"]["children"]
            if len(children) != repeats:
                raise ValueError("one case and exactly the declared isolated repeats required")
            runs = {precision: [] for precision in data}
            for ref in children:
                path = str(Path(ref["path"]).resolve())
                if path in seen:
                    raise ValueError("reused run cannot count as an independent observation")
                seen.add(path)
                child = checked(ref)
                for precision_ref in child.get("precision_children", []):
                    checked(precision_ref)
                rows = [r for r in child["results"] if r["bucket"] == "compute"]
                if len(rows) != 2 or {r["precision"] for r in rows} != set(data):
                    raise ValueError("one successful case in both precisions required")
                for row in rows:
                    identity = (row["operator"], row["scenario_id"], row["size_id"])
                    if case_identity is not None and case_identity != identity:
                        raise ValueError("case identity changed")
                    case_identity = identity
                    values = row["raw_steady_ms"]
                    if (
                        row.get("status") != "ok"
                        or len(values) != frames
                        or row.get("raw_steady_run_offsets") != [0, frames]
                    ):
                        raise ValueError("incomplete raw run or changed repeat boundaries")
                    run_curve([values], frames, [0.5])
                    runs[row["precision"]].append(values)
            for precision in data:
                sides[precision].append(runs[precision])
        for precision in data:
            data[precision].append(sides[precision])
    return data, sampling, case_identity


def analyze(spec):
    probabilities = spec["quantiles"]
    windows = spec["frame_prefixes"]
    if (
        not windows
        or any(type(n) is not int or n < 1 for n in windows)
        or len(set(windows)) != len(windows)
        or not probabilities
        or any(not 0 < p < 1 for p in probabilities)
        or len(set(probabilities)) != len(probabilities)
    ):
        raise ValueError("distinct positive prefixes and interior quantiles required")
    datasets = spec["datasets"]
    bound = spec["dataset_family_bound"]
    if type(bound) is not int or bound < len(datasets) or not datasets:
        raise ValueError("family bound must cover all requested datasets")
    family_size = bound * len(windows) * len(probabilities) * 2
    results = []
    for dataset in datasets:
        pairs, sampling, identity = load_pairs(dataset["pairs"])
        measured_frames = sampling["frames"]
        for precision, observations in pairs.items():
            for frames in windows:
                if frames > measured_frames:
                    continue
                for row in summarize(observations, frames, probabilities, family_size):
                    row.update(
                        dataset=dataset["id"],
                        precision=precision,
                        case=list(identity),
                        measured_frames_per_run=measured_frames,
                        measured_repeats=sampling["repeats"],
                        unchanged_early_window_available=frames >= 50,
                    )
                    results.append(row)
    return {
        "analysis_version": 1,
        "method": "Equal-run empirical CDF; median paired log quantile ratios; exact sign/order-statistic Bonferroni intervals. Independent pairs and a stable population median are assumptions; no independent-frame assumption.",
        "status": "exploration_only_not_qualification_or_acceptance",
        "family_size": family_size,
        "minimum_resolvable_pairs": math.ceil(math.log2(40 * family_size)),
        "configuration": spec,
        "results": results,
        "unmeasured_frame_prefixes": [
            n for n in windows if not any(r["frames_per_run"] == n for r in results)
        ],
        "limits": "Prefixes retain chronology but are not fresh shorter runs. No extrapolated samples, outlier exclusions, power guarantee, floor, or historical reclassification. Increasing frames cannot repair inadequate independent-pair counts or nonstationarity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("refusing to overwrite analysis")
    result = analyze(read_json(args.spec))
    result["spec_reference"] = {"path": str(Path(args.spec).resolve()), "sha256": digest(args.spec)}
    write_json(args.out, result)


if __name__ == "__main__":
    main()
