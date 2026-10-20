# multi_source_expansion: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 131.215 | 124.711 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 467.030 | 578.232 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 2859.282 | 3761.929 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | rows | [32] | {"index_columns":2,"selected_percent":25,"sources":2,"two":2} | {} |
| baseline-medium | medium | rows | [2048] | {"index_columns":2,"selected_percent":25,"sources":2,"two":2} | {} |
| baseline-large | large | rows | [16384] | {"index_columns":2,"selected_percent":25,"sources":2,"two":2} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{}
```

Output wiring: `{"drain":["aoa_estimates_out"],"optional_tensor_outputs":[],"timed":"detection_idx_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 362.468 | 353.764 | 6.021 / 20.0 | 3.101 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 124.711 | 125.278 | 1.360 / 5.0 | 0.612 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 126.779 | 127.597 | 2.129 / 5.0 | 1.260 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 133.771 | 137.626 | 7.129 / 5.0 | 3.116 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 124.711 | 125.278 | 1.360 / 5.0 | 0.612 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 126.779 | 127.597 | 2.129 / 5.0 | 1.260 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 133.771 | 137.626 | 7.129 / 5.0 | 3.116 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 4771.660 | 4781.225 | 4.661 / 15.0 | 3.769 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 352.992 | 346.320 | 3.097 / 20.0 | 1.032 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 131.215 | 124.179 | 10.065 / 5.0 | 3.601 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 138.822 | 128.221 | 13.430 / 5.0 | 5.480 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 165.706 | 153.412 | 11.664 / 5.0 | 10.828 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 131.215 | 124.179 | 10.065 / 5.0 | 3.601 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 138.822 | 128.221 | 13.430 / 5.0 | 5.480 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 165.706 | 153.412 | 11.664 / 5.0 | 10.828 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 4563.490 | 4521.355 | 5.176 / 15.0 | 3.604 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 789.930 | 789.087 | 2.529 / 20.0 | 0.590 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 578.232 | 578.205 | 0.668 / 5.0 | 0.126 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 582.627 | 582.690 | 1.145 / 5.0 | 0.393 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 601.065 | 605.948 | 2.117 / 5.0 | 1.143 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 578.232 | 578.205 | 0.668 / 5.0 | 0.126 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 582.627 | 582.690 | 1.145 / 5.0 | 0.393 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 601.065 | 605.948 | 2.117 / 5.0 | 1.143 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 4768.395 | 4740.015 | 5.538 / 15.0 | 2.253 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 678.360 | 678.462 | 2.279 / 20.0 | 1.504 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 467.030 | 467.308 | 4.924 / 5.0 | 0.217 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 471.209 | 471.977 | 5.316 / 5.0 | 0.646 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 492.437 | 488.844 | 3.069 / 5.0 | 1.520 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 467.030 | 467.308 | 4.924 / 5.0 | 0.217 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 471.209 | 471.977 | 5.316 / 5.0 | 0.646 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 492.437 | 488.844 | 3.069 / 5.0 | 1.520 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 4729.190 | 4726.075 | 6.635 / 15.0 | 3.723 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 3968.290 | 3969.275 | 0.075 / 20.0 | 0.050 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 3761.929 | 3745.382 | 0.754 / 5.0 | 0.243 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 4152.894 | 3753.498 | 15.632 / 5.0 | 5.470 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 4190.043 | 3775.918 | 15.656 / 5.0 | 5.980 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 3761.929 | 3745.382 | 0.754 / 5.0 | 0.243 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 4152.894 | 3753.498 | 15.632 / 5.0 | 5.470 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 4190.043 | 3775.918 | 15.656 / 5.0 | 5.980 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 5104.210 | 5324.415 | 10.143 / 15.0 | 4.205 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 3084.035 | 3084.005 | 0.832 / 20.0 | 0.762 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 2859.282 | 2859.861 | 0.026 / 5.0 | 0.023 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 2867.015 | 2865.555 | 0.051 / 5.0 | 0.038 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 2878.588 | 2878.520 | 0.151 / 5.0 | 0.403 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 2859.282 | 2859.861 | 0.026 / 5.0 | 0.023 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 2867.015 | 2865.555 | 0.051 / 5.0 | 0.038 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 2878.588 | 2878.520 | 0.151 / 5.0 | 0.403 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 5380.945 | 5140.580 | 5.501 / 15.0 | 3.423 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 14.004 | 7.195 | 0.914 | 665.251 | 7870.065 | 1 | 3.797 | 0.427 |
| baseline-small/screen | double | 1.683 | 1.564 | 5.277 | 191.537 | 7766.180 | 0 | 0.833 | 0.357 |
| baseline-medium/calibration | float | 24.068 | 2.386 | 1.131 | 1006.914 | 2119.785 | 0 | 0.455 | 0.393 |
| baseline-medium/calibration | double | 3.869 | 2.519 | 1.096 | 1016.616 | 1716.700 | 0 | 0.359 | 0.428 |
| baseline-large/screen | float | 0.757 | 4.853 | 1.145 | 2900.873 | 349.236 | 0 | 0.130 | 0.323 |
| baseline-large/screen | double | 28.479 | 5.435 | 2.862 | 4626.808 | 266.578 | 0 | 0.398 | 0.802 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/multi_source_expansion.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/multi_source_expansion/`; native evidence: `native/multi_source_expansion/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
