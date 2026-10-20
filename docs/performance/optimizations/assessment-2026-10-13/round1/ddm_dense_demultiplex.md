# ddm_dense_demultiplex: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 32.544 | 32.731 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 98.243 | 346.854 |
| baseline-large | repeat_passed | qualified | heldout | 1180.757 | 2608.246 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | tx, rx, doppler, range | [2,4,32,128] | {"velocity_axis":0} | {} |
| baseline-medium | medium | tx, rx, doppler, range | [4,16,128,512] | {"velocity_axis":0} | {} |
| baseline-large | large | tx, rx, doppler, range | [8,32,256,1024] | {"velocity_axis":0} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "ddm_code": {
    "factory": "linear_exact_ddm_code"
  }
}
```

Output wiring: `{"drain":["velocity_bins_out"],"optional_tensor_outputs":["velocity_bins_out"],"timed":"rd_maps_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 267.135 | 262.982 | 2.174 / 20.0 | 1.835 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 32.731 | 33.072 | 1.915 / 5.0 | 0.561 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 33.953 | 37.990 | 17.443 / 5.0 | 8.219 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 36.621 | 47.548 | 34.759 / 5.0 | 23.387 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 32.731 | 33.072 | 1.915 / 5.0 | 0.561 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 33.953 | 37.990 | 17.443 / 5.0 | 8.219 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 36.621 | 47.548 | 34.759 / 5.0 | 23.387 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 3140.705 | 2975.455 | 10.971 / 15.0 | 4.569 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 265.764 | 268.741 | 1.498 / 20.0 | 0.881 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 32.544 | 32.560 | 0.758 / 5.0 | 0.523 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 35.859 | 33.389 | 11.131 / 5.0 | 3.854 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 44.044 | 37.793 | 15.164 / 5.0 | 8.473 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 32.544 | 32.560 | 0.758 / 5.0 | 0.523 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 35.859 | 33.389 | 11.131 / 5.0 | 3.854 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 44.044 | 37.793 | 15.164 / 5.0 | 8.473 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 2978.670 | 2929.850 | 1.940 / 15.0 | 1.674 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 541.963 | 544.079 | 1.598 / 20.0 | 1.077 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 346.854 | 344.764 | 1.889 / 5.0 | 1.124 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 354.540 | 356.175 | 2.194 / 5.0 | 1.497 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 373.056 | 368.195 | 1.591 / 5.0 | 1.049 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 346.854 | 344.764 | 1.889 / 5.0 | 1.124 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 354.540 | 356.175 | 2.194 / 5.0 | 1.497 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 373.056 | 368.195 | 1.591 / 5.0 | 1.049 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 9290.030 | 9028.190 | 6.850 / 15.0 | 4.451 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 338.464 | 336.843 | 0.585 / 20.0 | 1.155 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 98.243 | 98.468 | 0.765 / 5.0 | 0.523 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 104.899 | 103.626 | 6.379 / 5.0 | 5.043 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 118.340 | 132.629 | 28.722 / 5.0 | 8.400 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 98.243 | 98.468 | 0.765 / 5.0 | 0.523 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 104.899 | 103.626 | 6.379 / 5.0 | 5.043 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 118.340 | 132.629 | 28.722 / 5.0 | 8.400 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 6460.295 | 6070.255 | 9.951 / 15.0 | 3.469 / 25.0 | unresolved |
| baseline-large/heldout | double | cold_ms | 2820.755 | 2817.905 | 0.722 / 20.0 | 0.383 / 30.0 | [-0.586, 0.472] |
| baseline-large/heldout | double | early_median_ms | 2608.246 | 2611.640 | 0.434 / 5.0 | 0.253 / 10.0 | [-0.359, 0.263] |
| baseline-large/heldout | double | early_p90_ms | 2701.867 | 2697.345 | 0.782 / 5.0 | 0.443 / 10.0 | [-0.399, 0.745] |
| baseline-large/heldout | double | early_p99_ms | 2730.406 | 2727.720 | 0.610 / 5.0 | 0.444 / 10.0 | [-0.319, 0.600] |
| baseline-large/heldout | double | median_ms | 2608.246 | 2611.640 | 0.434 / 5.0 | 0.253 / 10.0 | [-0.359, 0.263] |
| baseline-large/heldout | double | p90_ms | 2701.867 | 2697.345 | 0.782 / 5.0 | 0.443 / 10.0 | [-0.399, 0.745] |
| baseline-large/heldout | double | p99_ms | 2730.406 | 2727.720 | 0.610 / 5.0 | 0.444 / 10.0 | [-0.319, 0.600] |
| baseline-large/heldout | double | setup_ms | 68778.000 | 68946.150 | 1.265 / 15.0 | 0.664 / 25.0 | [-1.252, 0.621] |
| baseline-large/heldout | float | cold_ms | 1392.275 | 1389.375 | 2.217 / 20.0 | 0.919 / 30.0 | [-1.544, 1.396] |
| baseline-large/heldout | float | early_median_ms | 1180.757 | 1180.467 | 1.000 / 5.0 | 0.437 / 10.0 | [-0.597, 0.979] |
| baseline-large/heldout | float | early_p90_ms | 1259.344 | 1260.532 | 1.044 / 5.0 | 0.620 / 10.0 | [-0.709, 0.967] |
| baseline-large/heldout | float | early_p99_ms | 1291.554 | 1288.213 | 1.668 / 5.0 | 0.938 / 10.0 | [-1.001, 1.081] |
| baseline-large/heldout | float | median_ms | 1180.757 | 1180.467 | 1.000 / 5.0 | 0.437 / 10.0 | [-0.597, 0.979] |
| baseline-large/heldout | float | p90_ms | 1259.344 | 1260.532 | 1.044 / 5.0 | 0.620 / 10.0 | [-0.709, 0.967] |
| baseline-large/heldout | float | p99_ms | 1291.554 | 1288.213 | 1.668 / 5.0 | 0.938 / 10.0 | [-1.001, 1.081] |
| baseline-large/heldout | float | setup_ms | 37180.400 | 37153.000 | 2.179 / 15.0 | 0.879 / 25.0 | [-1.146, 1.803] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.246 | 0.821 | 2.423 | 71.389 | 27557.600 | 0 | 2.129 | 0.528 |
| baseline-small/screen | double | 0.626 | 1.176 | 0.316 | 425.704 | 27229.550 | 2 | 3.836 | 0.400 |
| baseline-medium/screen | float | 0.747 | 3.709 | 1.172 | 146.694 | 9666.675 | 0 | 3.393 | 0.870 |
| baseline-medium/screen | double | 6.601 | 3.709 | 7.261 | 398.603 | 2849.230 | 0 | 0.805 | 0.371 |
| baseline-large/heldout | float | 11.914 | 58.403 | 10.542 | 2305.051 | 831.081 | 0 | 3.966 | 0.571 |
| baseline-large/heldout | double | 11.376 | 61.351 | 10.115 | 3473.638 | 379.906 | 0 | 1.798 | 0.477 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/ddm_dense_demultiplex.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | qualified |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/ddm_dense_demultiplex/`; native evidence: `native/ddm_dense_demultiplex/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
