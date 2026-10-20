# feature_assembly: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 55.706 | 55.510 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 56.097 | 55.931 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 56.220 | 56.445 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | doppler, range | [64,256] | {"detections":64,"features":2,"pack":0,"two":2} | {} |
| baseline-medium | medium | doppler, range | [256,512] | {"detections":1024,"features":2,"pack":0,"two":2} | {} |
| baseline-large | large | doppler, range | [1024,2048] | {"detections":16384,"features":2,"pack":0,"two":2} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":[],"timed":"points","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 293.046 | 294.242 | 1.753 / 20.0 | 0.980 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 55.510 | 55.607 | 0.333 / 5.0 | 0.862 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 57.299 | 57.052 | 0.889 / 5.0 | 0.485 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 59.189 | 62.434 | 6.077 / 5.0 | 4.248 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 55.510 | 55.607 | 0.333 / 5.0 | 0.862 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 57.299 | 57.052 | 0.889 / 5.0 | 0.485 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 59.189 | 62.434 | 6.077 / 5.0 | 4.248 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 3539.340 | 3464.565 | 2.568 / 15.0 | 1.602 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 294.389 | 288.903 | 3.668 / 20.0 | 1.266 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 55.706 | 58.223 | 9.219 / 5.0 | 3.156 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 56.951 | 59.892 | 9.247 / 5.0 | 2.793 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 61.612 | 67.200 | 15.786 / 5.0 | 6.846 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 55.706 | 58.223 | 9.219 / 5.0 | 3.156 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 56.951 | 59.892 | 9.247 / 5.0 | 2.793 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 61.612 | 67.200 | 15.786 / 5.0 | 6.846 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 3664.275 | 3544.275 | 3.347 / 15.0 | 2.069 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 291.634 | 290.566 | 0.925 / 20.0 | 0.725 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 55.931 | 56.188 | 0.675 / 5.0 | 0.324 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 56.993 | 57.203 | 1.217 / 5.0 | 0.733 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 60.317 | 62.068 | 7.352 / 5.0 | 2.889 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 55.931 | 56.188 | 0.675 / 5.0 | 0.324 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 56.993 | 57.203 | 1.217 / 5.0 | 0.733 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 60.317 | 62.068 | 7.352 / 5.0 | 2.889 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 3497.795 | 3502.655 | 0.435 / 15.0 | 1.322 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 300.126 | 292.921 | 3.234 / 20.0 | 1.260 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 56.097 | 55.889 | 1.346 / 5.0 | 0.639 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 57.318 | 59.421 | 7.609 / 5.0 | 2.855 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 63.244 | 68.322 | 21.398 / 5.0 | 8.716 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 56.097 | 55.889 | 1.346 / 5.0 | 0.639 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 57.318 | 59.421 | 7.609 / 5.0 | 2.855 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 63.244 | 68.322 | 21.398 / 5.0 | 8.716 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 3745.825 | 3448.065 | 9.701 / 15.0 | 7.072 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 297.620 | 300.885 | 4.046 / 20.0 | 2.525 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 56.445 | 56.544 | 0.613 / 5.0 | 0.277 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 57.532 | 57.746 | 0.853 / 5.0 | 0.369 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 59.152 | 69.857 | 29.978 / 5.0 | 10.610 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 56.445 | 56.544 | 0.613 / 5.0 | 0.277 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 57.532 | 57.746 | 0.853 / 5.0 | 0.369 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 59.152 | 69.857 | 29.978 / 5.0 | 10.610 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 3752.050 | 4194.020 | 11.817 / 15.0 | 10.903 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 298.357 | 301.333 | 2.478 / 20.0 | 0.875 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 56.220 | 56.657 | 1.041 / 5.0 | 0.613 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 57.531 | 59.223 | 4.065 / 5.0 | 2.222 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 64.184 | 71.646 | 19.515 / 5.0 | 6.525 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 56.220 | 56.657 | 1.041 / 5.0 | 0.613 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 57.531 | 59.223 | 4.065 / 5.0 | 2.222 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 64.184 | 71.646 | 19.515 / 5.0 | 6.525 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 3671.810 | 3723.435 | 4.319 / 15.0 | 2.382 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 5.117 | 1.238 | 0.596 | 88.213 | 16906.700 | 0 | 3.618 | 0.342 |
| baseline-small/screen | double | 0.184 | 1.410 | 0.649 | 270.177 | 16970.050 | 1 | 3.349 | 0.529 |
| baseline-medium/screen | float | 0.754 | 1.237 | 0.737 | 81.250 | 16764.650 | 0 | 2.680 | 0.518 |
| baseline-medium/screen | double | 0.377 | 1.111 | 0.456 | 75.806 | 16895.100 | 0 | 2.294 | 0.492 |
| baseline-large/screen | float | 0.584 | 1.241 | 0.707 | 88.435 | 16610.850 | 0 | 2.200 | 0.472 |
| baseline-large/screen | double | 0.346 | 1.051 | 0.242 | 85.417 | 16706.800 | 0 | 1.980 | 0.487 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/feature_assembly.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/feature_assembly/`; native evidence: `native/feature_assembly/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
