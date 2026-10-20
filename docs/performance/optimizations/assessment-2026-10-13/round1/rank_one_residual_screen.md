# rank_one_residual_screen: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 157.387 | 155.498 |
| baseline-medium | repeat_passed | qualified | heldout | 932.101 | 861.138 |
| baseline-large | repeat_passed | qualified | heldout | 3321.531 | 3368.189 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | items, channels, looks | [32,8,3] | {"selection_percent":25,"two":2} | {} |
| baseline-medium | medium | items, channels, looks | [1024,32,9] | {"selection_percent":25,"two":2} | {} |
| baseline-large | large | items, channels, looks | [4096,64,9] | {"selection_percent":25,"two":2} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "array_geometry": {
    "factory": "ula_geometry",
    "spacing_wavelengths": 0.5,
    "wavelength": 1.0
  },
  "residual_threshold": 0.03
}
```

Output wiring: `{"drain":["selected_idx","is_valid_out","residual_score","selection_mask"],"optional_tensor_outputs":["is_valid_out","residual_score","selection_mask"],"timed":"snapshot_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 414.135 | 416.200 | 3.199 / 20.0 | 1.750 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 155.498 | 150.595 | 4.260 / 5.0 | 3.487 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 158.480 | 158.396 | 1.139 / 5.0 | 0.589 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 167.060 | 168.888 | 5.823 / 5.0 | 3.819 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 155.498 | 150.595 | 4.260 / 5.0 | 3.487 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 158.480 | 158.396 | 1.139 / 5.0 | 0.589 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 167.060 | 168.888 | 5.823 / 5.0 | 3.819 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 4929.190 | 4946.310 | 6.070 / 15.0 | 2.081 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 416.815 | 417.103 | 3.044 / 20.0 | 1.279 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 157.387 | 157.447 | 1.136 / 5.0 | 0.327 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 159.441 | 160.192 | 2.020 / 5.0 | 1.078 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 169.839 | 170.462 | 6.659 / 5.0 | 2.712 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 157.387 | 157.447 | 1.136 / 5.0 | 0.327 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 159.441 | 160.192 | 2.020 / 5.0 | 1.078 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 169.839 | 170.462 | 6.659 / 5.0 | 2.712 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 4901.730 | 4921.000 | 6.293 / 15.0 | 2.964 / 25.0 | unresolved |
| baseline-medium/heldout | double | cold_ms | 1126.430 | 1127.550 | 0.846 / 20.0 | 0.495 / 30.0 | [-0.758, 0.474] |
| baseline-medium/heldout | double | early_median_ms | 861.138 | 861.237 | 0.143 / 5.0 | 0.081 / 10.0 | [-0.122, 0.096] |
| baseline-medium/heldout | double | early_p90_ms | 866.702 | 867.648 | 0.654 / 5.0 | 0.499 / 10.0 | [-0.392, 0.430] |
| baseline-medium/heldout | double | early_p99_ms | 887.401 | 883.472 | 1.847 / 5.0 | 0.968 / 10.0 | [-1.449, 1.381] |
| baseline-medium/heldout | double | median_ms | 861.138 | 861.237 | 0.143 / 5.0 | 0.081 / 10.0 | [-0.122, 0.096] |
| baseline-medium/heldout | double | p90_ms | 866.702 | 867.648 | 0.654 / 5.0 | 0.499 / 10.0 | [-0.392, 0.430] |
| baseline-medium/heldout | double | p99_ms | 887.401 | 883.472 | 1.847 / 5.0 | 0.968 / 10.0 | [-1.449, 1.381] |
| baseline-medium/heldout | double | setup_ms | 6644.710 | 6685.110 | 6.354 / 15.0 | 2.396 / 25.0 | [-4.623, 1.585] |
| baseline-medium/heldout | float | cold_ms | 1197.610 | 1199.345 | 1.379 / 20.0 | 0.665 / 30.0 | [-0.917, 1.311] |
| baseline-medium/heldout | float | early_median_ms | 932.101 | 932.189 | 0.116 / 5.0 | 0.070 / 10.0 | [-0.077, 0.085] |
| baseline-medium/heldout | float | early_p90_ms | 935.997 | 936.843 | 0.678 / 5.0 | 0.294 / 10.0 | [-0.499, 0.301] |
| baseline-medium/heldout | float | early_p99_ms | 952.613 | 950.828 | 1.641 / 5.0 | 0.981 / 10.0 | [-1.190, 1.536] |
| baseline-medium/heldout | float | median_ms | 932.101 | 932.189 | 0.116 / 5.0 | 0.070 / 10.0 | [-0.077, 0.085] |
| baseline-medium/heldout | float | p90_ms | 935.997 | 936.843 | 0.678 / 5.0 | 0.294 / 10.0 | [-0.499, 0.301] |
| baseline-medium/heldout | float | p99_ms | 952.613 | 950.828 | 1.641 / 5.0 | 0.981 / 10.0 | [-1.190, 1.536] |
| baseline-medium/heldout | float | setup_ms | 6278.605 | 6259.510 | 6.304 / 15.0 | 2.787 / 25.0 | [-2.790, 4.336] |
| baseline-large/heldout | double | cold_ms | 3652.730 | 3646.405 | 1.638 / 20.0 | 0.736 / 30.0 | [-0.759, 1.215] |
| baseline-large/heldout | double | early_median_ms | 3368.189 | 3368.501 | 0.265 / 5.0 | 0.153 / 10.0 | [-0.263, 0.185] |
| baseline-large/heldout | double | early_p90_ms | 3403.760 | 3403.057 | 0.318 / 5.0 | 0.156 / 10.0 | [-0.315, 0.185] |
| baseline-large/heldout | double | early_p99_ms | 3431.089 | 3428.893 | 0.745 / 5.0 | 0.390 / 10.0 | [-0.409, 0.535] |
| baseline-large/heldout | double | median_ms | 3368.189 | 3368.501 | 0.265 / 5.0 | 0.153 / 10.0 | [-0.263, 0.185] |
| baseline-large/heldout | double | p90_ms | 3403.760 | 3403.057 | 0.318 / 5.0 | 0.156 / 10.0 | [-0.315, 0.185] |
| baseline-large/heldout | double | p99_ms | 3431.089 | 3428.893 | 0.745 / 5.0 | 0.390 / 10.0 | [-0.409, 0.535] |
| baseline-large/heldout | double | setup_ms | 12234.400 | 12135.900 | 7.860 / 15.0 | 5.428 / 25.0 | [-6.016, 6.930] |
| baseline-large/heldout | float | cold_ms | 3503.255 | 3501.275 | 0.531 / 20.0 | 0.303 / 30.0 | [-0.324, 0.530] |
| baseline-large/heldout | float | early_median_ms | 3321.531 | 3321.219 | 0.178 / 5.0 | 0.080 / 10.0 | [-0.071, 0.137] |
| baseline-large/heldout | float | early_p90_ms | 3336.901 | 3335.006 | 0.317 / 5.0 | 0.225 / 10.0 | [-0.211, 0.274] |
| baseline-large/heldout | float | early_p99_ms | 3358.357 | 3357.939 | 0.619 / 5.0 | 0.351 / 10.0 | [-0.528, 0.447] |
| baseline-large/heldout | float | median_ms | 3321.531 | 3321.219 | 0.178 / 5.0 | 0.080 / 10.0 | [-0.071, 0.137] |
| baseline-large/heldout | float | p90_ms | 3336.901 | 3335.006 | 0.317 / 5.0 | 0.225 / 10.0 | [-0.211, 0.274] |
| baseline-large/heldout | float | p99_ms | 3358.357 | 3357.939 | 0.619 / 5.0 | 0.351 / 10.0 | [-0.528, 0.447] |
| baseline-large/heldout | float | setup_ms | 8924.105 | 8938.570 | 8.039 / 15.0 | 3.552 / 25.0 | [-7.488, 6.906] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 1.786 | 1.733 | 0.909 | 1079.752 | 6200.070 | 1 | 3.577 | 0.419 |
| baseline-small/calibration | double | 6.689 | 7.321 | 6.500 | 1065.954 | 6367.520 | 3 | 4.110 | 0.453 |
| baseline-medium/heldout | float | 1.079 | 2.427 | 1.026 | 1849.869 | 1068.490 | 0 | 0.908 | 0.483 |
| baseline-medium/heldout | double | 1.229 | 2.530 | 1.189 | 1764.784 | 1154.790 | 0 | 0.542 | 0.564 |
| baseline-large/heldout | float | 5.912 | 8.809 | 3.620 | 4571.988 | 300.500 | 0 | 0.390 | 0.797 |
| baseline-large/heldout | double | 8.912 | 37.582 | 7.734 | 4330.401 | 296.014 | 0 | 0.884 | 0.738 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/rank_one_residual_screen.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | qualified |
| baseline-large | qualified |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/rank_one_residual_screen/`; native evidence: `native/rank_one_residual_screen/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
