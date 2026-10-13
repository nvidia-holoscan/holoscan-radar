# Sampling pilot: reduction versus FFT

**Assessment only.** ClutterFilter large (mean subtraction) and FMCWRangeProcessor small (range FFT), each in float and double. The four workloads do not represent either operator’s full static/live/optional matrix. No production optimization or acceptance-policy change was made.

Branch: `deustice/optimizations`. Session: [/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1). The pilot ran 3,360 serial fresh processes over 46.3 measurement minutes, preserving 4,788,000 measured frame times. Build source: `f6212e73a02e2f3ace2d7bff0a37add03f8859e506f709f8854f8bca01e36f13`. Image and SDK/runtime identities match the preserved baseline; each precision has its own executable.

## Experimental design

- Five warmup frames per measured graph; the existing discarded two-frame process-warmup graph remains unchanged. Cold/setup are retained separately. Raw frame export occurs after graph execution.
- Calibration: ten pairs, 50/500/2,000 measured frames, seven isolated repeats per side. Three-repeat results use the first three of the same seven repeats; these comparisons are nested, not independent.
- The mean and median options aggregate per-process medians or percentiles. Neither option changes a per-process metric into a pooled frame percentile or a frame mean.
- Each A1/A2 repeat launches a fresh executable process; order alternates at repeat level. Stratum/frame order is shuffled with a fixed seed. A shared GPU container avoids per-repeat container launch overhead. This is a fresh assessment context, not a direct speed comparison against earlier Docker-per-repeat timings.
- One common configuration was selected on calibration data by the predeclared rule: fewest failing metric rows, then smallest sum of median-latency paired P90 deltas, then fewer frames times repeats, then median in an exact tie. Calibration has insufficient pair count for the familywise interval, so it is diagnostic only.
- Configuration and calibration hashes were frozen before the thirty-pair held-out block. The held-out results did not influence selection. The familywise interval uses twenty metric rows multiplied by the campaign family of twenty-six operators (family size 520).
- Original dispersion thresholds are shown for comparison: steady 5% paired P90 delta/10% IQR; setup 15%/25%; cold 20%/30%. These experiments do not replace the original policy or qualify an editing session.

[Predeclared plan](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1/plan.json); [build/runtime/loader manifest](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1/build-manifest.json); [frozen selection](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1/frozen-selection.json); [immutable raw run index](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1/runs.jsonl).

## Calibration comparison

| Frames | Repeats | Aggregation | Failing rows / 20 | Median failures / 4 | p90 failures / 4 | p99 failures / 4 | Cold failures / 4 | Setup failures / 4 | Selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 3 | median | 10 | 2 | 2 | 4 | 1 | 1 | No |
| 50 | 3 | mean | 11 | 2 | 2 | 4 | 2 | 1 | No |
| 50 | 7 | median | 7 | 1 | 2 | 4 | 0 | 0 | No |
| 50 | 7 | mean | 10 | 2 | 3 | 3 | 2 | 0 | No |
| 500 | 3 | median | 6 | 2 | 2 | 1 | 0 | 1 | No |
| 500 | 3 | mean | 9 | 2 | 2 | 2 | 2 | 1 | No |
| 500 | 7 | median | 4 | 1 | 2 | 1 | 0 | 0 | No |
| 500 | 7 | mean | 6 | 1 | 2 | 1 | 2 | 0 | No |
| 2000 | 3 | median | 4 | 1 | 2 | 1 | 0 | 0 | No |
| 2000 | 3 | mean | 6 | 1 | 3 | 1 | 1 | 0 | No |
| 2000 | 7 | median | 3 | 0 | 2 | 1 | 0 | 0 | Yes |
| 2000 | 7 | mean | 5 | 0 | 2 | 1 | 2 | 0 | No |

The frozen choice is **2000 frames, 7 repeats, median aggregation**. Complete calibration rows and individual paired values remain in the [calibration comparison](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1/calibration-comparison.json).

## Held-out validation

Against the unchanged numerical dispersion thresholds, **3/20 rows fail**. **1/4 median-latency rows** have a familywise interval contained within +/-1% and including zero, the proposed diagnostic precision target for a 2% objective. 0 metric intervals exclude zero. A wide interval containing zero is inconclusive; no detected direction does not establish equivalence.

A1/A2 absolute columns are medians across thirty pair-side summaries. Delta P90 is computed across thirty absolute paired percentage differences, not from the difference between the two absolute columns. IQR uses all sixty pair-side observations. Positive interval values mean A2 was faster.

| Operator | Precision | Metric | A1 ms | A2 ms | Delta P90 / limit | IQR / limit | Familywise change interval | Dispersion result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clutter_filter | float | median_ms | 0.556691 | 0.555352 | 1.35% / 5% | 0.81% / 10% | [-0.69%, 1.23%] | PASS |
| clutter_filter | float | p90_ms | 0.609423 | 0.610015 | 6.63% / 5% | 4.25% / 10% | [-6.05%, 3.69%] | FAIL |
| clutter_filter | float | p99_ms | 1.436477 | 1.438519 | 1.17% / 5% | 0.72% / 10% | [-0.69%, 1.12%] | PASS |
| clutter_filter | float | setup_ms | 3.547620 | 3.532160 | 4.90% / 15% | 1.95% / 25% | [-3.13%, 3.12%] | PASS |
| clutter_filter | float | cold_ms | 0.689372 | 0.690548 | 7.86% / 20% | 3.45% / 30% | [-4.87%, 5.21%] | PASS |
| clutter_filter | double | median_ms | 0.614908 | 0.613664 | 1.14% / 5% | 0.79% / 10% | [-0.75%, 1.08%] | PASS |
| clutter_filter | double | p90_ms | 0.674713 | 0.667357 | 7.24% / 5% | 2.93% / 10% | [-5.53%, 5.48%] | FAIL |
| clutter_filter | double | p99_ms | 1.488074 | 1.487374 | 0.99% / 5% | 0.48% / 10% | [-0.68%, 0.64%] | PASS |
| clutter_filter | double | setup_ms | 4.122415 | 4.100970 | 8.35% / 15% | 3.76% / 25% | [-4.61%, 4.29%] | PASS |
| clutter_filter | double | cold_ms | 0.751351 | 0.749982 | 6.43% / 20% | 3.70% / 30% | [-5.89%, 4.13%] | PASS |
| fmcw_range_processor | float | median_ms | 0.025921 | 0.025959 | 1.94% / 5% | 1.30% / 10% | [-1.07%, 1.30%] | PASS |
| fmcw_range_processor | float | p90_ms | 0.026958 | 0.026884 | 1.53% / 5% | 0.84% / 10% | [-0.72%, 1.27%] | PASS |
| fmcw_range_processor | float | p99_ms | 0.038853 | 0.036755 | 26.78% / 5% | 22.39% / 10% | [-25.55%, 21.11%] | FAIL |
| fmcw_range_processor | float | setup_ms | 3.973050 | 3.915230 | 3.47% / 15% | 2.72% / 25% | [-1.37%, 3.11%] | PASS |
| fmcw_range_processor | float | cold_ms | 0.258597 | 0.258347 | 2.18% / 20% | 1.49% / 30% | [-1.09%, 1.30%] | PASS |
| fmcw_range_processor | double | median_ms | 0.044077 | 0.044162 | 1.59% / 5% | 0.80% / 10% | [-0.96%, 0.41%] | PASS |
| fmcw_range_processor | double | p90_ms | 0.045556 | 0.045733 | 1.49% / 5% | 0.85% / 10% | [-1.12%, 0.84%] | PASS |
| fmcw_range_processor | double | p99_ms | 0.055816 | 0.055968 | 2.22% / 5% | 0.92% / 10% | [-1.67%, 0.52%] | PASS |
| fmcw_range_processor | double | setup_ms | 3.905210 | 3.890755 | 3.55% / 15% | 2.83% / 25% | [-2.47%, 2.66%] | PASS |
| fmcw_range_processor | double | cold_ms | 0.268779 | 0.269325 | 2.73% / 20% | 1.73% / 30% | [-1.78%, 1.10%] | PASS |

[Held-out raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1/validation-review.json). Individual process setup/cold values remain in every raw report; these are not process-startup or first-CUDA-context costs.

## Frame order and settling diagnostics

These diagnostics are descriptive and do not prove stationarity. Prefix bias compares the first fifty measured frames with the full run median in the same process. Half drift compares the last-half median with the first-half median. No frames were excluded. Lag-one correlation is shown as a diagnostic, not evidence of independent observations.

| Stage | Operator | Precision | Frames/run | Runs | P90 absolute half drift | Median prefix-50 bias | P90 absolute prefix-50 bias | Median lag-1 correlation | Frames above 3x run median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | clutter_filter | double | 50 | 140 | 2.67% | 0.00% | 0.00% | -0.091 | 1 |
| calibration | clutter_filter | double | 500 | 140 | 9.21% | -0.48% | 8.52% | -0.075 | 8 |
| calibration | clutter_filter | double | 2000 | 140 | 4.59% | 0.14% | 9.39% | -0.072 | 24 |
| calibration | clutter_filter | float | 50 | 140 | 3.28% | 0.00% | 0.00% | -0.083 | 6 |
| calibration | clutter_filter | float | 500 | 140 | 8.60% | -0.84% | 8.47% | -0.063 | 16 |
| calibration | clutter_filter | float | 2000 | 140 | 7.46% | 0.21% | 11.17% | -0.056 | 98 |
| calibration | fmcw_range_processor | double | 50 | 140 | 2.47% | 0.00% | 0.00% | 0.178 | 1 |
| calibration | fmcw_range_processor | double | 500 | 140 | 1.31% | 1.21% | 2.11% | 0.225 | 8 |
| calibration | fmcw_range_processor | double | 2000 | 140 | 1.62% | 2.43% | 3.39% | 0.239 | 40 |
| calibration | fmcw_range_processor | float | 50 | 140 | 1.91% | 0.00% | 0.00% | 0.108 | 0 |
| calibration | fmcw_range_processor | float | 500 | 140 | 1.76% | 1.35% | 2.07% | 0.152 | 3 |
| calibration | fmcw_range_processor | float | 2000 | 140 | 2.87% | 3.76% | 5.18% | 0.192 | 22 |
| validation | clutter_filter | double | 2000 | 420 | 7.09% | 1.13% | 10.29% | -0.072 | 64 |
| validation | clutter_filter | float | 2000 | 420 | 7.26% | 0.97% | 10.35% | -0.059 | 184 |
| validation | fmcw_range_processor | double | 2000 | 420 | 1.64% | 2.49% | 3.61% | 0.250 | 76 |
| validation | fmcw_range_processor | float | 2000 | 420 | 2.71% | 3.57% | 4.93% | 0.190 | 67 |

[Full per-run diagnostics](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/sampling-pilot-1/frame-diagnostics.json).

The traces below use a fixed illustrative sample: calibration pair 1, repeat 1, A1, at 2,000 frames. Every measured frame is plotted. They were not selected for unusually good or bad performance.

![Chronological frame timings](/home/deustice/Projects/holoscan-radar/docs/performance/optimizations/assets/sampling-pilot-1-frame-traces.png)

## Findings and proposed criteria

The fixed illustrative ClutterFilter float trace visibly changes timing level: its first 500-frame median is 0.670494 ms, versus 0.564517 ms for the last 500 (-15.81% change without any code change). This single example is not a rate estimate, but it shows why small dispersion of an aggregate does not establish a constant in-run timing regime. The cause remains unresolved.

The larger sampling configuration reduced failures from 10/20 in the 50-frame, three-repeat calibration condition to 3/20 in the selected condition. The separate held-out block also has 3/20 failures. Median aggregation outperformed mean aggregation under the predeclared selection rule; averaging is not the general solution in this study. Changes to repeat-level pairing and container launch context mean the difference from the original campaign cannot be attributed solely to repeat/frame counts.

All four held-out median rows meet the existing 5% paired-delta/10% IQR limits: paired P90 deltas range from 1.14% to 1.94%, and IQR from 0.79% to 1.30%. Their familywise intervals include zero, but only RangeProcessor double fits within the proposed +/-1% precision target. The largest median-interval endpoint magnitude is 1.30%. This supports better precision for these measured workloads, not a guarantee of adequate power for effects arbitrarily close to 2%.

The remaining tail variability in absolute units is:

| Operator | Precision | Tail metric | P90 absolute paired difference | P90 relative paired difference |
| --- | --- | --- | --- | --- |
| clutter_filter | float | p90_ms | 39.49 microseconds | 6.63% |
| clutter_filter | double | p90_ms | 48.96 microseconds | 7.24% |
| fmcw_range_processor | float | p99_ms | 10.52 microseconds | 26.78% |

These are observed variability summaries, not permissible regressions. A practical tail budget must come from application requirements and be confirmed before candidate results, not chosen to exceed these observations.

Long runs also expose settling: in held-out RangeProcessor runs, the first fifty measured frames have a median latency approximately 3.57% higher than the full-run median for float and 2.49% higher for double. Median lag-one frame correlation is approximately 0.19/0.25. ClutterFilter prefix bias is approximately 0.97%/1.13%, with a P90 absolute prefix difference around 10.3%. No mechanism is established. We should preserve early-frame behavior when moving to a longer sampling window, rather than treating it as interchangeable with fifty-frame bursts.

The evidence-backed next policy proposal is to retain median aggregation, use this longer sampling configuration as the starting point for steady/tail assessment, and explicitly protect an early measured-frame window alongside setup/cold. Keep the >2% familywise lower-bound objective. Choose calibration precision targets and relative/absolute tail regression budgets before candidates, based on expected useful effects and application needs; do not treat a failed p99 dispersion row as proof that the median objective is unmeasurable. Confirm and validate any revised policy with fresh measurements before editing production. Under the unchanged original rules, both sampled operators remain blocked by their tail rows.

## Limits on interpretation

- Longer runs may measure a different sustained execution regime from fifty-frame bursts. Examine prefix/half-run behavior before applying a longer-run statistic to short-burst workloads. Preserve setup, cold, early steady and tail behavior rather than hiding settling with additional warmup.
- A p99 from fifty frames depends on the slowest two ordered observations. At five hundred or two thousand frames it uses more tail observations, but a median or mean of process p99 values is still different from a pooled p99. Do not silently change the protected estimand.
- Clock bounds and governors were checked before and after every process against the original configuration. This does not establish constant in-run CPU frequency or identify the cause of variable latency. No CPU affinity, governor, GPU clock or power setting was changed.
- Pair intervals require independent paired observations. Frame correlations, between-process drift and execution-order effects remain relevant even when a dispersion row passes.
- The selection study is small and explores only one size per operator. Full-case calibration, native output/ownership validation and confirmed concrete policies remain required before production candidates.

## Preparation verification

The raw export regression test verifies chronological order, repeat boundaries, exclusion of warmup/changed/reversion frames, unchanged summary statistics and JSON serialization. All twenty-three PerfProfiles fixtures and six relevant performance smoke tests pass. Every executable recipe validates, all SDK libraries resolve from the archived runtime, and each collected frame sequence reproduces its reported median/p90/p99 within JSON rounding precision. Raw input hashes are checked during analysis. Pinned checks are recorded in the publication receipt.
