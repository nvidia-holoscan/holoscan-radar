# Chronological timing distributions

Offline assessment on Thor, 2026-10-15. Branch: `deustice/optimization-skills`. No new GPU measurements, production edits, exclusions or acceptance-policy changes.

Longer runs change the measured distribution as well as its uncertainty. These captures do not establish a universal frame count or a validated steady-state boundary. Preserve startup behavior explicitly; do not repair this by silently increasing warmup.

## Evidence

Three sealed long-run replications, 30 original A/A pairs each, three isolated repeats per side, both precisions. Compare non-overlapping 50-frame windows within the same runs. Each original pair contributes one early/later log-quantile ratio, pooling its six runs equally. Frames and repeats do not count as independent inferential replicates. Existing five-frame warmup and two-frame process warmup remain unchanged.

Exact sign/order-statistic simultaneous 95% intervals use a conservative family of 12,000, covering 1,044 window comparisons plus the earlier 9,360-slot analysis. They assume independent original pairs and a stable population median of their contrasts; they do not assume independent frames. This is post hoc exploration, not held-out qualification. Absolute columns below are medians of pair-level quantiles; percentage changes are medians of paired ratios, so they need not equal the ratio of those absolute columns.

| Operator / precision | Frames | Metric | First50 (µs) | Last50 (µs) | Paired change | Simultaneous 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| fmcw_range_processor / float | 2000 | p50 | 26.486 | 25.276 | -4.42% | [-5.23%, -3.68%] |
| fmcw_range_processor / float | 2000 | p99 | 30.087 | 30.647 | -4.33% | [-34.36%, +64.82%] |
| fmcw_range_processor / double | 2000 | p50 | 44.959 | 43.560 | -2.81% | [-3.74%, -2.12%] |
| fmcw_range_processor / double | 2000 | p99 | 56.661 | 54.225 | -3.48% | [-17.85%, +18.05%] |
| detection_reduction / float | 2000 | p50 | 294.286 | 289.649 | -1.56% | [-2.09%, -1.23%] |
| detection_reduction / float | 2000 | p99 | 341.386 | 331.999 | -2.88% | [-12.39%, -0.44%] |
| detection_reduction / double | 2000 | p50 | 294.580 | 291.559 | -1.02% | [-1.69%, -0.67%] |
| detection_reduction / double | 2000 | p99 | 343.401 | 336.080 | -2.13% | [-11.30%, +3.39%] |
| music_estimator / float | 500 | p50 | 3404.597 | 2646.856 | -20.13% | [-27.09%, -9.12%] |
| music_estimator / float | 500 | p99 | 4439.565 | 3916.271 | -10.15% | [-21.56%, -4.37%] |
| music_estimator / double | 500 | p50 | 4163.966 | 4044.655 | -2.72% | [-8.13%, +0.19%] |
| music_estimator / double | 500 | p99 | 5604.851 | 5286.543 | -5.87% | [-17.28%, +2.30%] |

## Interpretation

- RangeProcessor: median latency declines throughout much of the run in both precisions. Float also has a transient tail increase at frames 51–100: p99 +38.31%, interval [+1.03%, +81.88%], versus first50. The late median decline is therefore not evidence that all quantiles improve monotonically.
- DetectionReduction: median declines are smaller but reproducible. Adjacent-window changes still occur late in float, including frames 1901–1950; choosing a steady cutoff from these data would require further validation. Tail intervals remain broad.
- MUSIC: float median falls about 20% and p99 about 10% by the final window. Later median point estimates appear flatter, but absence of a detected adjacent-window change does not establish equivalence or stationarity. Double results differ and must remain separate.
- These observations identify chronological changes, not their cause. No GPU profiling or telemetry attribution establishes whether scheduling, caching, allocation, SDK behavior or another mechanism explains them.

## Proposed sampling rules

1. Keep 50 measured frames and three isolated repeats as the inexpensive screening protocol, retaining raw chronology, existing warmups and early-window protection. A 50-frame screen does not qualify p99 or establish a consistency floor.
2. Define the intended workload horizon before qualification: startup/short bursts versus sustained processing. Report first50 separately from the complete declared horizon. A longer horizon measures a different mixture when timing changes with frame position.
3. Use the existing 2,000-frame RangeProcessor/DetectionReduction and 500-frame MUSIC captures only to select exploratory protocols. Do not declare any of these qualified. No fixed steady cutoff, extra warmup or discarded interval is supported yet.
4. For any proposed distribution gate, compare both the early and declared complete per-frame distributions while preserving process/pair clustering. Freeze the estimand, practical regression margins, multiplicity, frame/repeat/pair counts and timing horizon before fresh disjoint validation. A distribution-equality rejection alone does not measure practical regression; failure to reject equality does not prove robustness.
5. Keep precision-specific qualification and existing setup/cold/live protections. Increasing frames cannot replace enough independent pairs. Confidence width and +/-1% remain diagnostics, not acceptance gates. Do not adopt a universal absolute-microsecond allowance.

The next measurement proposal should therefore qualify a declared workload horizon and a distribution-based comparison, rather than search for one universal robust frame count. It needs a new explicit timing budget; this offline assessment requests no GPU time. The previously completed two-hour allocation is closed.

## Coverage and validation

Only three existing workloads have replicated long-run evidence here. Larger synthetic FFT and reduction controls have screens but no equivalent long-run replication; no whole-campaign qualification follows. No new suspected correctness bugs were found. Production optimization and the full baseline remain pending.

Raw inputs are descendant-hash verified by the maintained frame-count loader. Null and known +10% synthetic controls verify contrast direction and exact interval behavior. The full host-tool suite and pinned checks are recorded in the publication receipt.

Artifacts: `build-optimization/operator-perf/assessment-2026-10-15/timing-regimes/` (`spec.json`, `analyze.py`, `results.json`, `publish.py`, `publication.json`). Historical reports and builds remain unchanged.
