# FMCWRangeProcessorOp assessment

Disposition: blocked.

Coverage gaps: validation_blocked.

No production candidates attempted. No cumulative gain claimed.

Unresolved correctness findings: inspect recorded blockers; unverified scopes remain unknown.

Artifacts: `build-optimization/operator-perf/benchmark-tooling-implementation/integration-final/smoke/fmcw_range_processor`.

## Representative A/A

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
