# Operator benchmark campaign

Phase: qualification_complete. Mode: assess.

Assessment timings do not establish adequate performance for omitted or blocked cases.

Baseline/final columns show ranges of casewise compute medians in milliseconds, not pooled run statistics.

| Operator | Family | Disposition | Baseline medians ms | Final medians ms | Retained changes | Coverage gaps |
| --- | --- | --- | --- | --- | --- | --- |
| clutter_filter | reduction | blocked | not collected | not collected | 0 | validation_blocked |
| fmcw_range_processor | fft | blocked | not collected | not collected | 0 | validation_blocked |

## Phase durations

Job durations include process launch and orchestration, and do not measure agent working time.

- clutter_filter / qualification: 30.3 seconds (blocked).
- fmcw_range_processor / qualification: 31.2 seconds (blocked).
- clutter_filter / qualification: 0.3 seconds (blocked).
- fmcw_range_processor / qualification: 0.3 seconds (blocked).

## clutter_filter qualification A/A

Pairs: 2; family size: 64; minimum resolvable pairs: 12.

| Case | Precision | Bucket | Metric | A1 ms | A2 ms | Paired P90 / limit | IQR / limit | Familywise interval | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | double | compute | cold_ms | 0.786038 | 0.770655 | 2.71% / 20.0 | 2.25% / 30.0 | [None, None] | inconclusive |
| baseline-small | double | compute | early_median_ms | 0.585027 | 0.546334 | 10.5% / 5.0 | 3.62% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | early_p90_ms | 0.699439 | 0.576551 | 26.1% / 5.0 | 11.6% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | early_p99_ms | 1.50963 | 1.56968 | 11.5% / 5.0 | 5.04% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | median_ms | 0.585027 | 0.546334 | 10.5% / 5.0 | 3.62% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | p90_ms | 0.699439 | 0.576551 | 26.1% / 5.0 | 11.6% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | p99_ms | 1.50963 | 1.56968 | 11.5% / 5.0 | 5.04% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | setup | setup_ms | 3.31982 | 3.34145 | 5.34% / 15.0 | 3.66% / 25.0 | [None, None] | inconclusive |
| baseline-small | float | compute | cold_ms | 0.789298 | 0.757447 | 6.65% / 20.0 | 2.37% / 30.0 | [None, None] | inconclusive |
| baseline-small | float | compute | early_median_ms | 0.599001 | 0.595325 | 19% / 5.0 | 17.8% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | early_p90_ms | 0.666867 | 0.63344 | 13.7% / 5.0 | 7.5% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | early_p99_ms | 1.52352 | 1.61712 | 7.74% / 5.0 | 5.04% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | median_ms | 0.599001 | 0.595325 | 19% / 5.0 | 17.8% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | p90_ms | 0.666867 | 0.63344 | 13.7% / 5.0 | 7.5% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | p99_ms | 1.52352 | 1.61712 | 7.74% / 5.0 | 5.04% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | setup | setup_ms | 3.30885 | 3.39433 | 4.13% / 15.0 | 1.62% / 25.0 | [None, None] | inconclusive |

- Blocker: insufficient_interval_resolution; matrix / all metrics.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / early_median_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / early_p90_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / early_p90_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / early_p99_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / median_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / p90_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / p90_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'double', 'small', 'compute'] / p99_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / early_median_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / early_median_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / early_p90_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / early_p99_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / median_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / median_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / p90_ms.

- Blocker: dispersion; ['ClutterFilterOp', 'baseline-small', 'float', 'small', 'compute'] / p99_ms.

Frame-regime diagnostics are descriptive; no observations were discarded.

| Case | Precision | Side | Half drift range % | Early bias range % | Raw min/max ms | Stall frames |
| --- | --- | --- | --- | --- | --- | --- |
| baseline-small | double | A1 | -3.03 to 1.83 | 0 to 0 | 0.513881 / 2.46957 | 1 |
| baseline-small | double | A2 | -2.43 to 1.06 | 0 to 0 | 0.499121 / 1.64777 | 4 |
| baseline-small | float | A1 | -2.57 to 1.22 | 0 to 0 | 0.522594 / 1.70956 | 0 |
| baseline-small | float | A2 | -2.08 to -0.257 | 0 to 0 | 0.483186 / 1.72022 | 4 |

## fmcw_range_processor qualification A/A

Pairs: 2; family size: 64; minimum resolvable pairs: 12.

| Case | Precision | Bucket | Metric | A1 ms | A2 ms | Paired P90 / limit | IQR / limit | Familywise interval | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | double | compute | cold_ms | 0.270562 | 0.27056 | 1.31% / 20.0 | 1.78% / 30.0 | [None, None] | inconclusive |
| baseline-small | double | compute | early_median_ms | 0.0456807 | 0.045556 | 0.507% / 5.0 | 2% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | early_p90_ms | 0.0476098 | 0.0470081 | 1.63% / 5.0 | 3.66% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | early_p99_ms | 0.0539964 | 0.0550221 | 3.61% / 5.0 | 6.1% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | median_ms | 0.0456807 | 0.045556 | 0.507% / 5.0 | 2% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | p90_ms | 0.0476098 | 0.0470081 | 1.63% / 5.0 | 3.66% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | compute | p99_ms | 0.0539964 | 0.0550221 | 3.61% / 5.0 | 6.1% / 10.0 | [None, None] | inconclusive |
| baseline-small | double | setup | setup_ms | 4.07301 | 4.02656 | 1.23% / 15.0 | 0.656% / 25.0 | [None, None] | inconclusive |
| baseline-small | float | compute | cold_ms | 0.256885 | 0.256468 | 0.955% / 20.0 | 0.467% / 30.0 | [None, None] | inconclusive |
| baseline-small | float | compute | early_median_ms | 0.026516 | 0.0263938 | 2.09% / 5.0 | 1.51% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | early_p90_ms | 0.02741 | 0.0270869 | 3.17% / 5.0 | 1.68% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | early_p99_ms | 0.0328405 | 0.035348 | 9.45% / 5.0 | 6.32% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | median_ms | 0.026516 | 0.0263938 | 2.09% / 5.0 | 1.51% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | p90_ms | 0.02741 | 0.0270869 | 3.17% / 5.0 | 1.68% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | compute | p99_ms | 0.0328405 | 0.035348 | 9.45% / 5.0 | 6.32% / 10.0 | [None, None] | inconclusive |
| baseline-small | float | setup | setup_ms | 3.9564 | 3.94276 | 1.53% / 15.0 | 0.746% / 25.0 | [None, None] | inconclusive |

- Blocker: insufficient_interval_resolution; matrix / all metrics.

- Blocker: dispersion; ['FMCWRangeProcessorOp', 'baseline-small', 'float', 'small', 'compute'] / early_p99_ms.

- Blocker: dispersion; ['FMCWRangeProcessorOp', 'baseline-small', 'float', 'small', 'compute'] / p99_ms.

Frame-regime diagnostics are descriptive; no observations were discarded.

| Case | Precision | Side | Half drift range % | Early bias range % | Raw min/max ms | Stall frames |
| --- | --- | --- | --- | --- | --- | --- |
| baseline-small | double | A1 | -3.87 to -0.0994 | 0 to 0 | 0.042945 / 0.066389 | 0 |
| baseline-small | double | A2 | -1.33 to 0.942 | 0 to 0 | 0.04062 / 0.062296 | 0 |
| baseline-small | float | A1 | -2.74 to 0.46 | 0 to 0 | 0.025639 / 0.04087 | 0 |
| baseline-small | float | A2 | -1.26 to 0.562 | 0 to 0 | 0.025222 / 0.066037 | 0 |

## Optimization handoff

Agent applies operator-perf candidate declarations and gates. No production edits are performed by this tool.

Opportunity ranking requires attributed recurring work, removable fraction, affected cases and mechanism confidence; absolute latency alone is insufficient.

Final gains require direct paired original/current-best measurements; intermediate percentages are never multiplied.
