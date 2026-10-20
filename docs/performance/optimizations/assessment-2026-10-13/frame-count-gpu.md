# Fresh frame-count exploration

Bounded exploration on `deustice/optimization-skills`, 2026-10-15. User confirmed a two-hour cap. **No production optimization, acceptance-policy change, fixed microsecond allowance or full-baseline qualification claim.** This experiment measures sensitivity to frame count and independently replicates predeclared protocols.

Completed units: {'screen': 22, 'calibration': 5, 'replication': 5}. Unit dispositions: {'complete': 32, 'blocked': 1, 'budget_unresolved': 1}. Time from authorization through timing completion: 102.58 minutes; the cap includes preparation, scheduling and analysis between measurements. Immutable session: `build-optimization/operator-perf/assessment-2026-10-15/frame-count-gpu/`.

The independent comparisons show no universal robust frame count. For small RangeProcessor, 2,000 frames narrowed p99 uncertainty substantially but still left a wide float interval; its median uncertainty barely changed. DetectionReduction's median intervals narrowed to about ±0.5%, while p99 did not improve consistently. A longer window can help a particular distribution region without making the whole protocol robust. Detailed intervals below are measured A/A effects, not optimization gains.

Equal-size first50/last50 comparisons also found temporal differences: float MUSIC’s late-window p99 was about 10% lower (simultaneous interval −21.6% to −4.4%), and float DetectionReduction’s was about 2.9% lower (−12.4% to −0.4%). These conditional paired intervals suggest an early/late timing-regime difference; they do not identify its cause or prove stationarity. All early observations remain protected, and warmup was not increased. Longer runs can therefore change which timing regime dominates as well as improve quantile estimation.

See [all protected metrics with absolute values and limits](frame-count-gpu-protected.md), and [overall-CDF versus typical-run and temporal diagnostics](frame-count-gpu-distributions.md). The latter's bootstrap intervals are explicitly pointwise exploratory approximations, not campaign-qualified gates.

Owned measurement-process wall time: 66.14 minutes across 1475 jobs, including startup and serialization. This is neither GPU busy time nor agent working time. Phase records remain in the ledger.

## Protocol and provenance

Both precisions; 50/200/500/2,000 measured frames; three isolated repeats; five warmup frames; existing two-frame process warmup; chronological raw capture; adjacent alternating A/A repeats; early first-50 metrics retained. No warmup increase, outlier exclusion, CPU fallback, clock/power/affinity changes or hidden pacing. Operators are FFT RangeProcessor, reduction ClutterFilter, irregular DetectionReduction and MUSIC eigensolver. The larger FFT/reduction cases are synthetic controls, not a fixed radar deployment.

Freshly compiled all eight harness executables using preserved CMake compiler/link flags and verified unchanged production archives. All eight were byte-for-byte identical to the preserved executables. Runtime dependencies were copied into a separate archive and ldd resolution recorded. The prior conservative source closure had detected the unrelated DopplerNotch harness repair; it was not bypassed or rewritten. Fresh manifests bind current provenance. Historical native and analytic correctness evidence remains applicable to unchanged production sources, selected recipes and byte-identical executables; no new correctness equivalence policy is introduced.

Deployment: NVIDIA Thor, driver 580.00, CUDA toolkit 13.0.88, GCC 13.3.0, HSDK 5.0.0, MatX 1.0.0; immutable image `sha256:d4ccdf30d119f555320cb629704a50614dab7cd0eecd1f7513481d52c0b7ddb2`. Host preflight and every-job clock observations are archived. Installed SDK source revision remains unproved. Nsight was not captured; Python bindings, Nsight Compute and CUDA sanitizer coverage remain unavailable from the established deployment. No instrumentation was added.

A two-pair breadth-first screen precedes predeclared 50- and 2,000-frame contrasts for small RangeProcessor and large DetectionReduction. Each affordable contrast receives ten fresh calibration pairs, then an immutable protocol receipt and thirty new replication pairs. The original short/long replication scopes were declared before screens. A separately recorded prospective allocation allowed 500-frame MUSIC calibration and replication if observed job costs left sufficient time; that allocation used processing-family coverage and execution cost, not favorable timing variability. Calibration failures remain visible rather than triggering repeated retry-to-pass. Replication here is evidence for exploration, not admission under a new distribution-based acceptance policy.

Frames were scheduled in ascending groups, and complete replication units ran sequentially. Comparisons between fresh frame-count protocols can include time/regime differences; they are not randomized causal estimates. Comparing chronological prefixes of the same longer run is a complementary check, also sensitive to within-run drift. Independent replication alone does not prove stationarity or independence of consecutive pairs.

## Fresh screen: absolute timings and distribution variability

Each side is an equal-weight empirical mixture of its three runs. Reported absolute values are medians across pairs of side A1 quantiles. Variability is P90 absolute paired percentage change of that mixture quantile. Two pairs are descriptive only and cannot establish robust population timing or finite familywise intervals.

| Operator / case | Frames | Float median µs | Double median µs | Float p99 µs | Double p99 µs | Float / double p99 A/A variability % |
| --- | --- | --- | --- | --- | --- | --- |
| fmcw_range_processor / baseline-small | 50 | 26.516 | 44.549 | 42.806 | 52.281 | 30.34 / 19.58 |
| clutter_filter / baseline-medium | 50 | 864.714 | 831.651 | 1615.433 | 1677.962 | 8.76 / 4.28 |
| detection_reduction / baseline-large | 50 | 294.660 | 294.181 | 354.999 | 334.078 | 11.90 / 8.72 |
| music_estimator / baseline-small | 50 | 3625.583 | 4146.895 | 4520.942 | 5692.718 | 1.49 / 4.95 |
| fmcw_range_processor / scenario-larger | 50 | 612.306 | 4001.968 | 669.525 | 4056.811 | 1.51 / 0.52 |
| fmcw_range_processor / baseline-small | 200 | 26.065 | 44.574 | 46.932 | 51.936 | 32.96 / 15.11 |
| clutter_filter / baseline-medium | 200 | 818.827 | 729.889 | 1674.091 | 1680.287 | 2.62 / 5.84 |
| detection_reduction / baseline-large | 200 | 295.000 | 293.795 | 347.904 | 343.297 | 5.86 / 78.20 |
| music_estimator / baseline-small | 200 | 3255.014 | 4084.938 | 4110.244 | 5307.941 | 10.10 / 9.65 |
| fmcw_range_processor / scenario-larger | 200 | 610.593 | 4001.822 | 681.250 | 4034.046 | 1.92 / 1.12 |
| clutter_filter / scenario-stress | 200 | 16689.361 | 32060.698 | 17199.493 | 33472.067 | 0.45 / 3.17 |
| fmcw_range_processor / baseline-small | 500 | 26.206 | 47.569 | 39.887 | 58.234 | 7.89 / 9.18 |
| clutter_filter / baseline-medium | 500 | 771.577 | 580.527 | 1575.722 | 1472.904 | 6.50 / 0.79 |
| detection_reduction / baseline-large | 500 | 292.873 | 292.558 | 367.792 | 339.669 | 9.42 / 1.82 |
| music_estimator / baseline-small | 500 | 3020.353 | 4079.718 | 4402.408 | 5636.714 | 1.45 / 8.07 |
| fmcw_range_processor / scenario-larger | 500 | 614.961 | 4004.597 | 713.952 | 4063.031 | 6.01 / 0.31 |
| clutter_filter / scenario-stress | 500 | 16318.894 | 30422.023 | 17073.366 | 32754.851 | 1.43 / 0.52 |
| fmcw_range_processor / baseline-small | 2000 | 25.709 | 43.995 | 44.510 | 57.807 | 3.56 / 7.36 |
| clutter_filter / baseline-medium | 2000 | 549.206 | 558.129 | 1541.422 | 1452.645 | 3.72 / 7.01 |
| detection_reduction / baseline-large | 2000 | 290.487 | 292.297 | 351.700 | 353.929 | 5.15 / 3.22 |
| music_estimator / baseline-small | 2000 | 2638.899 | 3899.736 | 4056.602 | 5114.169 | 2.45 / 12.55 |
| fmcw_range_processor / scenario-larger | 2000 | 616.142 | 4006.431 | 692.542 | 4075.846 | 1.03 / 0.02 |

## Independent replication

Effect is median paired log quantile ratio, transformed to percent; positive means A2 slower than A1. These are identical implementations. Simultaneous 95% sign/order-statistic intervals use the conservative 9,360-comparison family, assuming independent pairs with a stable population median. Pairs, not individual frames, are the inference units. At this multiplicity fewer than 19 pairs cannot give finite bounds; 19 is only a resolution floor, not a power guarantee. The full-window mixture quantile differs from the previous median-of-repeat quantile estimand.

| Operator / case | Frames | Precision / quantile | A1 µs | A2 change % | Simultaneous 95% interval % | P90 absolute paired change % |
| --- | --- | --- | --- | --- | --- | --- |
| fmcw_range_processor / baseline-small | 50 | float / p50 | 26.496 | -0.151 | [-1.191, +1.043] | 1.439 |
| fmcw_range_processor / baseline-small | 50 | float / p99 | 29.191 | +0.280 | [-41.513, +64.703] | 54.318 |
| fmcw_range_processor / baseline-small | 50 | double / p50 | 44.917 | -0.021 | [-1.602, +1.460] | 1.799 |
| fmcw_range_processor / baseline-small | 50 | double / p99 | 56.084 | -2.025 | [-22.838, +13.366] | 24.495 |
| detection_reduction / baseline-large | 50 | float / p50 | 294.601 | -0.192 | [-0.755, +0.704] | 0.932 |
| detection_reduction / baseline-large | 50 | float / p99 | 338.412 | -0.499 | [-21.963, +5.350] | 11.021 |
| detection_reduction / baseline-large | 50 | double / p50 | 294.804 | -0.021 | [-1.181, +1.124] | 1.491 |
| detection_reduction / baseline-large | 50 | double / p99 | 343.270 | -0.571 | [-13.645, +10.030] | 23.435 |
| fmcw_range_processor / baseline-small | 2000 | float / p50 | 25.579 | +0.018 | [-1.188, +1.577] | 1.533 |
| fmcw_range_processor / baseline-small | 2000 | float / p99 | 44.016 | +2.213 | [-14.551, +25.302] | 25.957 |
| fmcw_range_processor / baseline-small | 2000 | double / p50 | 43.982 | +0.094 | [-1.166, +1.328] | 1.422 |
| fmcw_range_processor / baseline-small | 2000 | double / p99 | 56.014 | -0.539 | [-5.006, +2.439] | 5.110 |
| detection_reduction / baseline-large | 2000 | float / p50 | 290.728 | -0.045 | [-0.538, +0.377] | 0.548 |
| detection_reduction / baseline-large | 2000 | float / p99 | 340.887 | -0.480 | [-37.883, +6.840] | 40.262 |
| detection_reduction / baseline-large | 2000 | double / p50 | 292.329 | +0.006 | [-0.398, +0.555] | 0.555 |
| detection_reduction / baseline-large | 2000 | double / p99 | 338.927 | +0.908 | [-1.686, +12.790] | 5.295 |
| music_estimator / baseline-small | 500 | float / p50 | 2910.444 | +1.110 | [-12.793, +17.079] | 17.238 |
| music_estimator / baseline-small | 500 | float / p99 | 4239.567 | -0.170 | [-7.915, +6.911] | 8.872 |
| music_estimator / baseline-small | 500 | double / p50 | 4079.392 | +0.086 | [-5.902, +8.389] | 8.307 |
| music_estimator / baseline-small | 500 | double / p99 | 5385.905 | +0.974 | [-8.057, +9.175] | 9.238 |

![Independent replication intervals](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-15/frame-count-gpu/replication-intervals.png)

The first50 and full2,000 points share longer-run captures; their dependence is retained. This figure shows the two original short/long contrasts; the conditional MUSIC result is in the table above.

## First-50 protection versus complete long windows

These paired intervals use the same replication captures, retaining the first 50 measured frames. They are not fresh 50-frame runs and do not upgrade historical evidence. A long steady window can stabilize a tail estimate while its early window remains variable. Both remain visible.

| Operator / case | Actual frames / analyzed prefix | Precision | Median interval % | P99 interval % |
| --- | --- | --- | --- | --- |
| fmcw_range_processor / baseline-small | 2000 / 50 | float | [-1.115, +1.478] | [-37.394, +38.592] |
| fmcw_range_processor / baseline-small | 2000 / 50 | double | [-1.308, +1.459] | [-16.749, +19.859] |
| fmcw_range_processor / baseline-small | 2000 / 2000 | float | [-1.188, +1.577] | [-14.551, +25.302] |
| fmcw_range_processor / baseline-small | 2000 / 2000 | double | [-1.166, +1.328] | [-5.006, +2.439] |
| detection_reduction / baseline-large | 2000 / 50 | float | [-0.655, +1.396] | [-9.958, +15.723] |
| detection_reduction / baseline-large | 2000 / 50 | double | [-0.411, +1.193] | [-28.565, +9.562] |
| detection_reduction / baseline-large | 2000 / 2000 | float | [-0.538, +0.377] | [-37.883, +6.840] |
| detection_reduction / baseline-large | 2000 / 2000 | double | [-0.398, +0.555] | [-1.686, +12.790] |
| music_estimator / baseline-small | 500 / 50 | float | [-8.730, +6.218] | [-8.973, +5.438] |
| music_estimator / baseline-small | 500 / 50 | double | [-9.312, +8.009] | [-10.549, +9.614] |
| music_estimator / baseline-small | 500 / 500 | float | [-12.793, +17.079] | [-7.915, +6.911] |
| music_estimator / baseline-small | 500 / 500 | double | [-5.902, +8.389] | [-8.057, +9.175] |

## Existing protected metrics and unresolved units

Original median/p90/p99, early metrics, setup/cold and applicable transition policies were retained separately for diagnostic review. Limits remain steady P90-delta/IQR 5%/10%, setup 15%/25%, cold 20%/30%, override 25%/40%. A full-window distribution interval including zero does not cancel any protected failure, certify equivalence or establish power. The raw reports retain maxima, throughput, half-window drift, lag-one correlation and stall counts.

| Phase / operator / case / frames | Existing-review blocker kinds |
| --- | --- |
| calibration / fmcw_range_processor / baseline-small / 50 | {'insufficient_interval_resolution': 1, 'dispersion': 6} |
| replication / fmcw_range_processor / baseline-small / 50 | {'dispersion': 8} |
| calibration / detection_reduction / baseline-large / 50 | {'insufficient_interval_resolution': 1, 'dispersion': 2} |
| replication / detection_reduction / baseline-large / 50 | {'dispersion': 4} |
| calibration / fmcw_range_processor / baseline-small / 2000 | {'insufficient_interval_resolution': 1, 'dispersion': 5} |
| replication / fmcw_range_processor / baseline-small / 2000 | {'dispersion': 7} |
| calibration / detection_reduction / baseline-large / 2000 | {'insufficient_interval_resolution': 1, 'dispersion': 5} |
| replication / detection_reduction / baseline-large / 2000 | {'dispersion': 2} |
| calibration / music_estimator / baseline-small / 500 | {'insufficient_interval_resolution': 1, 'dispersion': 14} |
| replication / music_estimator / baseline-small / 500 | {'dispersion': 15} |

Unresolved: {'operator': 'clutter_filter', 'case': 'scenario-stress', 'frames': 50, 'phase': 'screen', 'status': 'blocked', 'error': 'measurement wall budget exhausted'}

Unresolved: {'operator': 'clutter_filter', 'case': 'scenario-stress', 'frames': 2000, 'phase': 'screen', 'status': 'budget_unresolved', 'forecast_seconds': 1515.0}

## Interpretation and limits

No universal frame count is selected automatically. Use the replication and preserved early-window intervals to assess which measured scopes gained information from a longer window. Wide intervals, directional effects, drift and inadequate independent-pair resolution remain unresolved. A narrower interval is useful evidence but is not an adequate-power test. Intermittent stalls can substantially affect p99 even at 2,000 frames; all observed stalls are retained.

The 500- and 2,000-frame screens provide previously missing observations for affordable scopes. Two-pair screens cannot qualify a workload. Only the predeclared replicated case/protocol combinations have independent thirty-pair evidence; unmeasured workload variants, live transitions and affected consumers remain outside this exploration. Existing first-50, setup and cold protection prevents a long full-window result from concealing early costs. A production optimization session still requires the complete affected matrix and consolidated confirmation.

No new suspected production correctness bug was found by this timing exploration. Existing RootMusic algebraic-rank interpretation and borrowed-output retention review gaps remain open. No candidate edits, cumulative optimization gain, commits, pushes or external publication. Qualification exhaustion does not authorize threshold relaxation or an automatic budget extension.

Raw plan, authorization, source/build/runtime receipts, clock observations, per-frame captures, calibration/replication namespaces, frozen protocol receipts, full distribution analysis and legacy reviews are in the session path above. Stop reason: declared matrix completed or remaining units unaffordable within the confirmed cap. Numerical analysis covers these synthetic workloads and this deployment only.

Validation: all 90 host-tool tests and pinned all-file checks passed, with explicit checks for the new reports and previously untracked analysis tools. Source, executable and archived runtime hashes were verified; owned measurement processes finished. No production changes or new suspected correctness bugs.
