# ClutterFilterOp assessment

Disposition: blocked.

Coverage gaps: validation_blocked.

No production candidates attempted. No cumulative gain claimed.

Unresolved correctness findings: inspect recorded blockers; unverified scopes remain unknown.

Artifacts: `build-optimization/operator-perf/benchmark-tooling-implementation/integration-final/smoke/clutter_filter`.

## Representative A/A

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
