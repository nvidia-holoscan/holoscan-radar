# Frame-count and timing-distribution experiment

Offline exploration on `deustice/optimization-skills`, 2026-10-15. **No robust universal frame count has been established.** Fifty frames can describe the center usefully in some workloads; 200 frames do not consistently stabilize tails or inter-run regimes. A fresh longer-frame experiment is needed before selecting a replacement measurement policy. No GPU jobs, new acceptance gates, production changes or historical reclassification were performed.

The fixed 2/5/10 µs proposal in the earlier [baseline review](baseline-review.md) is superseded by the user's distribution-based direction. That historical report remains immutable. This experiment evaluates empirical timing distributions and independent paired uncertainty instead of introducing an absolute noise allowance.

## Data and reproducible method

Replayed 54 complete datasets: all 43 round-one scopes with complete calibration, using the latest held-out data where available (including failed validation), plus 11 complete earlier RangeProcessor/ClutterFilter protocol comparisons. Four earlier datasets contain 200 frames per run; three contain seven repeats. Both precisions are separate. Every selected pair index, aggregate, raw child and referenced precision export is hash-verified. Incomplete historical attempts are excluded by completeness, not timing results.

The maintained [offline tool](/home/deustice/Projects/holoscan-radar/performance/optimization/frame_count.py) and [usage/assumptions](/home/deustice/Projects/holoscan-radar/performance/optimization/FRAME_COUNT.md) reconstruct equal-weight empirical run distributions. Within each side of each pair, all runs contribute equally. With equal prefix lengths this is equivalent to taking quantiles of the concatenated prefixes, without treating those frames as independent experimental units. Each curve is evaluated at p50/p75/p90/p95/p99. The effect is the median across independent pairs of log(candidate/reference quantile), transformed to percentage. Positive means slower.

This differs intentionally from taking the median of three run-p99 summaries. A single repeat with stalls can influence the empirical mixture tail; it cannot be discarded by the median-of-three aggregation. This is a new exploratory estimand, not a silent revision of old results or a guarantee about the underlying population p99. Maxima remain in the output.

Simultaneous intervals use the existing exact sign/order-statistic method on complete pairs, assuming independent pairs with a stable population median. Bonferroni family size is 156 dataset slots × six requested prefixes × five quantiles × two precisions = **9,360**. The bound covers all 1,160 observed comparisons and conservatively includes unavailable prefixes. No IID-frame bootstrap, pooled-frame p-values or synthetic longer runs are used. Pair independence and stationarity remain assumptions to validate prospectively; temporal drift can undermine them.

Analyzed chronological prefixes of 20/50/100/200 frames where supported. The 20-frame prefix is diagnostic only: it cannot preserve the required first-50 protection. Five warmup frames and the original process warmup remain as recorded; no additional early frames were discarded. A 50-frame prefix of a 200-frame run is not a fresh 50-frame run. Frame-count contrasts from those same runs isolate prefix length, but can also expose within-run regime changes. Historical 50/3 versus 50/7 protocols occurred at different times and are not causal repeat-count experiments.

## What increasing frames actually did

The table uses the same 200-frame runs for each row. Values are P90 absolute paired percentage differences in the empirical-mixture quantiles. They are exploratory consistency diagnostics, not pass/fail thresholds. Ten independent A/A pairs per dataset.

| Operator / case / precision | p50: 50 → 100 → 200 frames (%) | p99: 50 → 100 → 200 frames (%) |
| --- | --- | --- |
| clutter_filter / compact / float | 9.03 → 8.35 → 9.29 | 7.37 → 7.47 → 7.80 |
| clutter_filter / compact / double | 11.86 → 12.37 → 17.74 | 12.35 → 11.92 → 12.05 |
| clutter_filter / diagnostic / float | 9.42 → 9.15 → 11.00 | 9.31 → 8.00 → 7.18 |
| clutter_filter / diagnostic / double | 40.35 → 41.02 → 41.30 | 8.11 → 6.00 → 8.50 |
| fmcw_range_processor / compact / float | 2.18 → 2.81 → 8.41 | 9.83 → 9.55 → 6.06 |
| fmcw_range_processor / compact / double | 0.63 → 0.54 → 0.49 | 50.18 → 45.32 → 55.56 |
| fmcw_range_processor / diagnostic / float | 1.25 → 1.07 → 1.21 | 58.65 → 57.58 → 37.43 |
| fmcw_range_processor / diagnostic / double | 3.46 → 3.31 → 3.27 | 24.63 → 14.68 → 10.74 |

![Paired quantile variability versus frame prefix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-15/frame-count/prefix-comparison.png)

For diagnostic RangeProcessor double, increasing the prefix from 50 to 200 reduces p99 paired variability from 24.63% to 10.74%, while median variation stays about 3.3%. Conversely, compact RangeProcessor double has a stable median near 0.5–0.6% but p99 variability around 45–56% across these prefixes. Diagnostic ClutterFilter double remains around 40–41% at the median. More frames alone therefore does not solve every workload's variability. These four workloads do not establish a universal scaling law across operators or GPUs.

## Independent pairs and tail visibility

At the declared family size, at least **19 independent pairs** are needed just to obtain finite sign-interval bounds; this is a mathematical resolution floor, not adequate-power evidence. Every ten-pair 200-frame dataset has unbounded intervals for this family. Increasing frames to 200 cannot repair that shortage. Thirty-pair held-out datasets permit finite bounds, though some remain very wide.

Examples below use the recent 50-frame/three-repeat held-out data. Intervals protect the full exploratory family, not just these rows. No scope is accepted or rejected by this new method.

| Workload / precision / quantile | A1 quantile µs | Median paired change % | Simultaneous 95% interval % |
| --- | --- | --- | --- |
| clutter_filter / scenario-stress / float / p50 | 16789.926 | -0.083 | [-0.364, +0.456] |
| clutter_filter / scenario-stress / float / p99 | 17191.598 | +0.263 | [-1.227, +2.009] |
| clutter_filter / scenario-stress / double / p50 | 32017.068 | -0.050 | [-0.441, +0.216] |
| clutter_filter / scenario-stress / double / p99 | 32786.776 | -0.213 | [-0.896, +0.678] |
| fmcw_range_processor / scenario-larger / float / p50 | 621.394 | -0.092 | [-1.254, +1.409] |
| fmcw_range_processor / scenario-larger / float / p99 | 693.651 | +0.237 | [-1.479, +25.080] |
| fmcw_range_processor / scenario-larger / double / p50 | 4009.463 | -0.010 | [-0.073, +0.190] |
| fmcw_range_processor / scenario-larger / double / p99 | 4072.608 | +0.140 | [-0.771, +1.631] |
| detection_reduction / baseline-large / float / p50 | 294.996 | +0.009 | [-1.007, +1.019] |
| detection_reduction / baseline-large / float / p99 | 341.516 | +0.593 | [-11.711, +177.647] |
| detection_reduction / baseline-large / double / p50 | 295.883 | +0.188 | [-0.633, +1.840] |
| detection_reduction / baseline-large / double / p99 | 336.512 | +2.942 | [-7.708, +27.525] |

DetectionReduction large float illustrates why changing the distribution functional matters: its median change is tightly centered near zero, while its empirical-mixture p99 interval spans roughly −12% to +178%. The original median-of-three-p99 method reported much less variation. The new result exposes retained stalls; it is not evidence of a production correctness bug or an accepted performance regression. We must decide and validate the desired tail estimand explicitly before acceptance use.

## Frame-count recommendation

Retain 50 frames for the protected early window. There is no evidence here supporting fewer than 50 for the existing policy, or making every operator use 200. For a fresh experiment, explore **50, 200, 500 and 2,000 frames** with three isolated repeats, then compare adding independent repeats/pairs where between-run variation dominates. Keep stable larger workloads as controls, not just noisy short cases. Do not assume larger tensors or longer runs automatically improve measurement quality.

A useful tail-support planning illustration is `frames × (1 − quantile)`: at p99 it is 0.5 for 50 frames, 2 for 200, 5 for 500, and 20 for 2,000. These are nominal counts, not effective independent observations or precision guarantees. The three-repeat empirical mixture has more observations, but correlated frames and variation between runs still matter. This archive contains no 500- or 2,000-frame runs for this experiment, so neither count can be declared sufficient.

Select protocols based on both within-run tail information and uncertainty between independent pairs. Check chronological stability, held-out A/A interval behavior and sensitivity to practically relevant changes. Interval width alone is a diagnostic; merely including zero is not evidence of equivalence or sufficient power. Preserve the existing candidate objective and regression policy until a replacement is concretely confirmed. Broad uncertainty remains unresolved rather than becoming a permissive regression margin.

A fresh bounded pilot should screen the declared frame counts across FFT, reduction, irregular processing and eigensolver representatives before committing to expensive full calibration. Use costs from those screens to allocate independent pairs and freeze a protocol before disjoint validation. A two-hour exploration cap is a proposed starting budget, not authorization or a promise that full validation fits. Large stress workloads at 2,000 frames can consume much of it; cap exhaustion leaves those scopes unresolved. Keep the former fixed-microsecond pilot unstarted.

## Artifacts, validation and limitations

Raw session: `build-optimization/operator-perf/assessment-2026-10-15/frame-count/`, containing the declared `spec.json`, `results.json`, figure, scripts and publication hashes. CLI refuses to overwrite a result. Host tests cover chronological prefixes, unavailable frame counts, distribution weighting, scale invariance, independent-pair resolution, tampering, reused run artifacts, incomplete units and conservative multiplicity. No changes to operators, existing measurement/acceptance gates, clocks or immutable builds. Historical data remain exploratory evidence; a new policy requires fresh qualified and confirmed sessions.

This experiment does not assess setup/cold/live-transition distributions, unmeasured cases or consumers; those existing protections remain necessary. Existing RootMusic algebraic-rank interpretation and borrowed-output retention gaps remain unresolved. No new suspected production correctness bug was found. No commits, pushes or external publication.

Validation completed: all 90 host-tool tests passed, including seven new offline-analysis tests. Pinned all-file checks and explicit checks for the new source/documentation passed. No GPU measurements were run.
