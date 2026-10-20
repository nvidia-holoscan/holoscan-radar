# subarray_selection: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 20.639 | 20.574 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 20.578 | 20.768 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 39.357 | 39.637 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | items, channels, snapshots | [64,4,2] | {"mask_dtype":0,"with_mask":0} | {} |
| baseline-medium | medium | items, channels, snapshots | [256,8,4] | {"mask_dtype":0,"with_mask":0} | {} |
| baseline-large | large | items, channels, snapshots | [1024,16,8] | {"mask_dtype":0,"with_mask":0} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "array_geometry": {
    "factory": "holey_ula",
    "wavelength": 1.0
  },
  "min_subarray_size": 2
}
```

Output wiring: `{"drain":["array_geometry"],"drain_codecs":{"array_geometry":"array_geometry"},"optional_tensor_outputs":["is_valid_out"],"timed":"snapshot_out","unwired":["is_valid_out"]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 251.565 | 253.713 | 1.684 / 20.0 | 1.031 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 20.574 | 20.560 | 1.645 / 5.0 | 0.895 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 21.410 | 21.390 | 2.939 / 5.0 | 2.241 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 22.612 | 22.339 | 55.558 / 5.0 | 13.084 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 20.574 | 20.560 | 1.645 / 5.0 | 0.895 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 21.410 | 21.390 | 2.939 / 5.0 | 2.241 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 22.612 | 22.339 | 55.558 / 5.0 | 13.084 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 3641.460 | 3652.805 | 4.703 / 15.0 | 1.284 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 253.154 | 252.861 | 1.424 / 20.0 | 0.789 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 20.639 | 20.646 | 1.914 / 5.0 | 0.714 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 21.264 | 21.337 | 1.737 / 5.0 | 0.900 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 22.585 | 23.397 | 27.535 / 5.0 | 21.008 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 20.639 | 20.646 | 1.914 / 5.0 | 0.714 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 21.264 | 21.337 | 1.737 / 5.0 | 0.900 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 22.585 | 23.397 | 27.535 / 5.0 | 21.008 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 3666.320 | 3652.305 | 2.529 / 15.0 | 2.646 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 255.570 | 256.250 | 2.256 / 20.0 | 1.587 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 20.768 | 20.599 | 2.481 / 5.0 | 1.441 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 21.433 | 21.267 | 2.879 / 5.0 | 1.609 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 24.102 | 21.979 | 17.076 / 5.0 | 7.464 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 20.768 | 20.599 | 2.481 / 5.0 | 1.441 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 21.433 | 21.267 | 2.879 / 5.0 | 1.609 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 24.102 | 21.979 | 17.076 / 5.0 | 7.464 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 3485.915 | 3392.220 | 5.085 / 15.0 | 1.763 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 256.107 | 257.634 | 2.498 / 20.0 | 1.515 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 20.578 | 20.900 | 2.405 / 5.0 | 1.039 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 21.463 | 21.565 | 1.032 / 5.0 | 0.409 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 22.750 | 34.241 | 79.195 / 5.0 | 30.406 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 20.578 | 20.900 | 2.405 / 5.0 | 1.039 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 21.463 | 21.565 | 1.032 / 5.0 | 0.409 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 22.750 | 34.241 | 79.195 / 5.0 | 30.406 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 3760.105 | 3723.315 | 5.457 / 15.0 | 4.353 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 276.069 | 278.097 | 1.395 / 20.0 | 0.428 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 39.637 | 39.213 | 1.379 / 5.0 | 0.879 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 40.776 | 42.277 | 6.844 / 5.0 | 4.251 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 44.172 | 46.667 | 9.236 / 5.0 | 10.868 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 39.637 | 39.213 | 1.379 / 5.0 | 0.879 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 40.776 | 42.277 | 6.844 / 5.0 | 4.251 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 44.172 | 46.667 | 9.236 / 5.0 | 10.868 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 4712.885 | 4805.525 | 8.428 / 15.0 | 12.606 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 267.042 | 268.935 | 1.341 / 20.0 | 0.722 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 39.357 | 39.257 | 0.489 / 5.0 | 0.181 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 40.834 | 40.610 | 2.949 / 5.0 | 2.264 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 52.885 | 45.232 | 27.306 / 5.0 | 12.429 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 39.357 | 39.257 | 0.489 / 5.0 | 0.181 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 40.834 | 40.610 | 2.949 / 5.0 | 2.264 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 52.885 | 45.232 | 27.306 / 5.0 | 12.429 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 3967.175 | 4014.490 | 3.871 / 15.0 | 2.069 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 0.389 | 0.551 | 0.381 | 60.130 | 42343.400 | 0 | 2.884 | 0.640 |
| baseline-small/calibration | double | 0.337 | 0.564 | 0.409 | 59.388 | 42215.500 | 0 | 5.814 | 0.677 |
| baseline-medium/screen | float | 0.490 | 0.560 | 0.585 | 65.509 | 41848.350 | 1 | 1.614 | 0.437 |
| baseline-medium/screen | double | 0.522 | 0.571 | 0.443 | 49.102 | 42274.250 | 0 | 3.863 | 0.600 |
| baseline-large/screen | float | 0.193 | 0.911 | 0.373 | 91.398 | 23389.600 | 0 | 1.968 | 0.372 |
| baseline-large/screen | double | 0.548 | 0.793 | 0.516 | 56.093 | 23433.200 | 0 | 2.851 | 0.359 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/subarray_selection.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/subarray_selection/`; native evidence: `native/subarray_selection/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
