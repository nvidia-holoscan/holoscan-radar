# doppler_notch: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 41.144 | 41.386 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 87.498 | 85.873 |
| baseline-large | repeat_passed | qualified | heldout | 439.033 | 453.167 |
| scenario-compact | capture_blocked | native_validation_blocked | none | unknown | unknown |
| scenario-larger | capture_blocked | native_validation_blocked | none | unknown | unknown |
| scenario-stress | capture_blocked | native_validation_blocked | none | unknown | unknown |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | doppler, range | [32,128] | {"rx":1,"tx":1} | {} |
| baseline-medium | medium | doppler, range | [128,512] | {"rx":1,"tx":1} | {} |
| baseline-large | large | doppler, range | [256,2048] | {"rx":1,"tx":1} | {} |
| scenario-compact | small | doppler, range | [128,512] | {"rx":4,"tx":1} | {} |
| scenario-larger | medium | doppler, range | [256,1024] | {"rx":16,"tx":1} | {} |
| scenario-stress | large | doppler, range | [512,2048] | {"rx":64,"tx":1} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "center_velocity": 0.0,
  "cyclic_velocity_axis": true,
  "stop_half_width": 0.5,
  "transition_width": 0.0
}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":[],"timed":"rd_maps_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 261.302 | 264.092 | 1.345 / 20.0 | 0.891 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 41.386 | 41.132 | 1.693 / 5.0 | 0.900 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 42.659 | 43.735 | 6.756 / 5.0 | 3.610 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 46.325 | 48.802 | 21.930 / 5.0 | 13.099 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 41.386 | 41.132 | 1.693 / 5.0 | 0.900 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 42.659 | 43.735 | 6.756 / 5.0 | 3.610 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 46.325 | 48.802 | 21.930 / 5.0 | 13.099 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 3164.370 | 3393.265 | 8.421 / 15.0 | 6.308 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 265.884 | 260.348 | 3.011 / 20.0 | 1.506 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 41.144 | 41.354 | 0.835 / 5.0 | 0.311 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 42.396 | 42.392 | 0.842 / 5.0 | 1.161 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 46.586 | 46.216 | 12.968 / 5.0 | 11.159 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 41.144 | 41.354 | 0.835 / 5.0 | 0.311 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 42.396 | 42.392 | 0.842 / 5.0 | 1.161 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 46.586 | 46.216 | 12.968 / 5.0 | 11.159 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 3215.425 | 3163.255 | 1.698 / 15.0 | 0.925 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 311.019 | 315.344 | 2.535 / 20.0 | 0.716 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 85.873 | 86.111 | 0.423 / 5.0 | 0.149 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 86.970 | 88.556 | 2.862 / 5.0 | 1.390 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 93.016 | 92.720 | 7.452 / 5.0 | 6.317 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 85.873 | 86.111 | 0.423 / 5.0 | 0.149 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 86.970 | 88.556 | 2.862 / 5.0 | 1.390 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 93.016 | 92.720 | 7.452 / 5.0 | 6.317 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 3631.840 | 3762.640 | 5.597 / 15.0 | 2.353 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 315.115 | 313.940 | 0.961 / 20.0 | 0.482 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 87.498 | 87.667 | 0.529 / 5.0 | 0.197 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 89.200 | 88.567 | 0.904 / 5.0 | 0.590 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 95.754 | 93.174 | 4.507 / 5.0 | 1.539 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 87.498 | 87.667 | 0.529 / 5.0 | 0.197 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 89.200 | 88.567 | 0.904 / 5.0 | 0.590 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 95.754 | 93.174 | 4.507 / 5.0 | 1.539 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 3723.425 | 3873.450 | 14.913 / 15.0 | 8.512 / 25.0 | unresolved |
| baseline-large/heldout | double | cold_ms | 676.344 | 675.173 | 1.392 / 20.0 | 0.684 / 30.0 | [-1.343, 0.843] |
| baseline-large/heldout | double | early_median_ms | 453.167 | 453.131 | 0.434 / 5.0 | 0.270 / 10.0 | [-0.421, 0.316] |
| baseline-large/heldout | double | early_p90_ms | 457.371 | 457.197 | 0.846 / 5.0 | 0.445 / 10.0 | [-0.419, 0.630] |
| baseline-large/heldout | double | early_p99_ms | 468.566 | 468.798 | 2.318 / 5.0 | 1.485 / 10.0 | [-2.153, 1.287] |
| baseline-large/heldout | double | median_ms | 453.167 | 453.131 | 0.434 / 5.0 | 0.270 / 10.0 | [-0.421, 0.316] |
| baseline-large/heldout | double | p90_ms | 457.371 | 457.197 | 0.846 / 5.0 | 0.445 / 10.0 | [-0.419, 0.630] |
| baseline-large/heldout | double | p99_ms | 468.566 | 468.798 | 2.318 / 5.0 | 1.485 / 10.0 | [-2.153, 1.287] |
| baseline-large/heldout | double | setup_ms | 5373.795 | 5341.135 | 5.127 / 15.0 | 1.736 / 25.0 | [-3.589, 2.691] |
| baseline-large/heldout | float | cold_ms | 663.681 | 663.002 | 0.703 / 20.0 | 0.651 / 30.0 | [-0.616, 0.625] |
| baseline-large/heldout | float | early_median_ms | 439.033 | 439.274 | 0.305 / 5.0 | 0.142 / 10.0 | [-0.209, 0.181] |
| baseline-large/heldout | float | early_p90_ms | 441.562 | 441.363 | 1.140 / 5.0 | 0.433 / 10.0 | [-0.473, 1.028] |
| baseline-large/heldout | float | early_p99_ms | 449.674 | 449.334 | 1.480 / 5.0 | 1.147 / 10.0 | [-1.403, 1.221] |
| baseline-large/heldout | float | median_ms | 439.033 | 439.274 | 0.305 / 5.0 | 0.142 / 10.0 | [-0.209, 0.181] |
| baseline-large/heldout | float | p90_ms | 441.562 | 441.363 | 1.140 / 5.0 | 0.433 / 10.0 | [-0.473, 1.028] |
| baseline-large/heldout | float | p99_ms | 449.674 | 449.334 | 1.480 / 5.0 | 1.147 / 10.0 | [-1.403, 1.221] |
| baseline-large/heldout | float | setup_ms | 4616.315 | 4676.615 | 10.236 / 15.0 | 4.202 / 25.0 | [-9.385, 3.174] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.342 | 0.970 | 0.489 | 60.630 | 22367.650 | 0 | 2.024 | 0.541 |
| baseline-small/screen | double | 0.703 | 0.721 | 0.650 | 110.407 | 22371.600 | 0 | 2.605 | 0.760 |
| baseline-medium/screen | float | 0.462 | 1.330 | 0.665 | 113.436 | 10986.650 | 0 | 0.890 | 0.392 |
| baseline-medium/screen | double | 0.362 | 1.094 | 0.354 | 139.481 | 11156.500 | 0 | 1.118 | 0.573 |
| baseline-large/heldout | float | 1.336 | 1.773 | 0.922 | 1155.381 | 2258.050 | 0 | 0.766 | 0.519 |
| baseline-large/heldout | double | 1.969 | 3.278 | 1.634 | 545.455 | 2188.015 | 0 | 1.177 | 0.576 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/doppler_notch.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | qualified |
| scenario-compact | native_validation_blocked |
| scenario-larger | native_validation_blocked |
| scenario-stress | native_validation_blocked |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/doppler_notch/`; native evidence: `native/doppler_notch/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.

Additional-scenario finding: the adapter sets RX/TX constructor extents only for its fixed MIMO fixture. Proposed tx=1, rx=4/16/64 cases validate but fail at runtime with an unexpected leading axis. All three existing static cases and eight C++ fixture tests passed. This is a harness sizing/validation gap; the three added cases remain blocked. Evidence: `doppler-notch-harness-finding.json` and `preparation/overlay-repair/doppler_notch-scenario-*-capture-0.log` under the raw session. No production defect or performance regression is established.
