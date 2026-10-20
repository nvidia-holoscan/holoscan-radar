# snapshot_gather: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 74.362 | 74.412 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 249.333 | 427.194 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 1778.367 | 3237.197 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | channels, doppler, range, detections | [4,64,256,64] | {"two":2} | {} |
| baseline-medium | medium | channels, doppler, range, detections | [16,256,512,1024] | {"two":2} | {} |
| baseline-large | large | channels, doppler, range, detections | [32,512,1024,4096] | {"two":2} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "doppler_edge": "wrap",
  "range_edge": "shift",
  "snapshot_window": [
    1,
    1
  ]
}
```

Output wiring: `{"drain":["is_valid"],"optional_tensor_outputs":["is_valid"],"timed":"snapshot","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 310.959 | 313.214 | 2.058 / 20.0 | 1.478 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 74.412 | 74.695 | 1.643 / 5.0 | 1.006 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 77.230 | 76.956 | 12.405 / 5.0 | 8.307 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 83.558 | 88.475 | 21.907 / 5.0 | 15.605 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 74.412 | 74.695 | 1.643 / 5.0 | 1.006 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 77.230 | 76.956 | 12.405 / 5.0 | 8.307 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 83.558 | 88.475 | 21.907 / 5.0 | 15.605 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 4154.565 | 4174.630 | 4.819 / 15.0 | 1.950 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 309.459 | 314.834 | 4.991 / 20.0 | 3.109 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 74.362 | 74.623 | 4.832 / 5.0 | 0.921 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 78.489 | 79.438 | 12.966 / 5.0 | 4.898 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 92.113 | 91.484 | 13.896 / 5.0 | 12.898 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 74.362 | 74.623 | 4.832 / 5.0 | 0.921 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 78.489 | 79.438 | 12.966 / 5.0 | 4.898 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 92.113 | 91.484 | 13.896 / 5.0 | 12.898 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 3973.095 | 3956.420 | 9.724 / 15.0 | 3.062 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 713.417 | 698.362 | 3.247 / 20.0 | 2.309 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 427.194 | 429.318 | 1.160 / 5.0 | 0.430 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 470.718 | 459.272 | 4.088 / 5.0 | 1.378 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 521.100 | 484.560 | 12.265 / 5.0 | 4.294 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 427.194 | 429.318 | 1.160 / 5.0 | 0.430 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 470.718 | 459.272 | 4.088 / 5.0 | 1.378 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 521.100 | 484.560 | 12.265 / 5.0 | 4.294 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 8033.025 | 8620.475 | 8.755 / 15.0 | 4.436 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 427.283 | 425.723 | 1.361 / 20.0 | 0.745 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 249.333 | 250.018 | 0.773 / 5.0 | 0.436 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 264.262 | 262.904 | 2.357 / 5.0 | 3.089 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 288.121 | 289.688 | 2.617 / 5.0 | 1.447 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 249.333 | 250.018 | 0.773 / 5.0 | 0.436 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 264.262 | 262.904 | 2.357 / 5.0 | 3.089 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 288.121 | 289.688 | 2.617 / 5.0 | 1.447 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 6535.345 | 6715.590 | 2.869 / 15.0 | 4.279 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 3576.825 | 3475.840 | 4.876 / 20.0 | 1.504 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 3237.197 | 3208.768 | 1.162 / 5.0 | 0.703 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 3330.803 | 3327.517 | 0.520 / 5.0 | 0.282 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 3370.373 | 3371.279 | 0.957 / 5.0 | 0.737 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 3237.197 | 3208.768 | 1.162 / 5.0 | 0.703 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 3330.803 | 3327.517 | 0.520 / 5.0 | 0.282 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 3370.373 | 3371.279 | 0.957 / 5.0 | 0.737 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 30352.150 | 29935.700 | 2.744 / 15.0 | 1.002 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 2060.715 | 2085.035 | 2.915 / 20.0 | 1.350 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 1778.367 | 1740.649 | 2.540 / 5.0 | 1.876 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 1878.066 | 1859.092 | 1.179 / 5.0 | 2.640 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 1902.317 | 1891.940 | 1.395 / 5.0 | 2.625 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 1778.367 | 1740.649 | 2.540 / 5.0 | 1.876 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 1878.066 | 1859.092 | 1.179 / 5.0 | 2.640 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 1902.317 | 1891.940 | 1.395 / 5.0 | 2.625 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 18337.600 | 18294.950 | 1.128 / 15.0 | 1.636 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 3.662 | 1.613 | 1.053 | 309.011 | 12682.150 | 1 | 3.731 | 0.538 |
| baseline-small/calibration | double | 1.214 | 1.474 | 1.394 | 155.583 | 12697.850 | 0 | 2.847 | 0.688 |
| baseline-medium/screen | float | 1.928 | 4.828 | 4.574 | 320.907 | 3905.825 | 0 | 2.247 | 0.439 |
| baseline-medium/screen | double | 4.923 | 11.984 | 5.807 | 1424.102 | 2277.925 | 1 | 1.439 | 0.403 |
| baseline-large/screen | float | 45.355 | 75.802 | 36.400 | 2734.419 | 558.294 | 0 | 2.458 | 0.493 |
| baseline-large/screen | double | 37.635 | 117.240 | 41.540 | 3435.808 | 307.943 | 0 | 1.606 | 0.237 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/snapshot_gather.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/snapshot_gather/`; native evidence: `native/snapshot_gather/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
