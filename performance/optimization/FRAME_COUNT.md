# Offline frame-count exploration

`frame_count.py` analyzes preserved version-3 A/A pairs without launching GPU
jobs or changing qualification/acceptance gates:

```bash
python3 performance/optimization/frame_count.py --spec /absolute/spec.json --out /absolute/results.json
```

The specification contains `quantiles` (for example `[0.5, 0.9, 0.99]`),
`frame_prefixes`, a positive `dataset_family_bound` covering all datasets, and
`datasets`: objects with an `id` and a `pairs` path/SHA256 reference. The tool
rejects incomplete units, changed child hashes, repeated run artifacts,
nonpositive/nonfinite timings, inconsistent cases and changed run boundaries.
It requires one case per index, both precisions and isolated repeats.

Each side of a pair is an equal-weight empirical mixture of its run
distributions. Runs have equal prefix lengths, so its quantiles can be
computed from concatenated prefixes. This is distribution weighting, **not**
a claim that frames are independent. The independent inference units remain
complete pairs. For each requested quantile the tool reports the median paired
log ratio, transformed back to percent, and the existing exact sign/order-statistic
interval. Bonferroni covers dataset bound × requested prefixes × quantiles ×
two precisions, including unavailable prefixes conservatively. Dependence
between comparisons is permitted; independent pairs and a stable population
median are assumptions. Insufficient pair resolution produces null bounds.

A prefix of a long run is not a fresh short run. Prefixes preserve chronological
early observations; below 50 is diagnostic and cannot preserve the version-3
first-50 protection. Longer runs must retain the first-50 metrics separately.
Unavailable frame counts are reported, never extrapolated or bootstrapped into
existence. More frames cannot manufacture independent run-pair evidence.

This mixture-quantile estimand differs from median-of-repeat quantiles and can
expose stalls that that aggregation suppresses. Historical gates and artifacts
remain unchanged. The output is exploratory, not a new population-tail guarantee,
measurement qualification, acceptance decision, power calculation or proof of
stationarity. Use fresh disjoint measurements to validate any selected policy.

## Declared Windows And Temporal Contrasts

`distributions.py` promotes equal-length chronological contrasts into a
maintained offline command, sharing the same descendant-hash verification and
paired quantile estimator:

```bash
python3 performance/optimization/distributions.py --spec /absolute/windows.json --out /absolute/results.json --report /absolute/results.md
```

Example specification (paths/hashes must refer to real sealed artifacts):

```json
{
  "quantiles": [0.5, 0.9, 0.99],
  "dataset_family_bound": 1,
  "prior_family_slots": 0,
  "datasets": [{"id": "range-small", "pairs": {"path": "/absolute/pairs.json", "sha256": "actual-sha256"}}],
  "windows": [
    {"id": "early", "start": 0, "frames": 50},
    {"id": "later", "start": 50, "frames": 50},
    {"id": "complete", "start": 0, "frames": 2000}
  ],
  "contrasts": [{"before": "early", "after": "later"}]
}
```

Window bounds are zero-based, half-open positions in measured chronology.
Every window generates a between-side A/A or A/B comparison. Each declared
temporal contrast compares equal-length windows within the same runs, pooling
both sides within each original pair. Its independent count remains the number
of original pairs, not twice the repeats or the number of frames. Its direction
is later/after versus earlier/before, positive meaning slower.

The family is `prior_family_slots + dataset_family_bound * 2 * quantile_count *
(window_count + temporal_contrast_count)`. Unavailable windows remain in the
budget and appear explicitly as unmeasured. Use prior slots to conservatively
account for earlier comparisons when interpreting analyses together. No
Bonferroni correction repairs data-dependent selection or invalid pair-level
assumptions; a selected protocol still needs disjoint validation.

JSON and optional Markdown show absolute before/after timings, paired effects,
simultaneous intervals and directional evidence. Neither output is a gate
decision. Both refuse existing destinations. No independent-frame bootstrap,
outlier removal, new warmup, implicit steady cutoff or GPU execution occurs.
The precise proposed statistical policy and its distinction from historical
production metrics are documented in
[distribution analysis](../../skills/operator-perf/references/distributions.md).
