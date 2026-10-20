# mvdr_estimator: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 205.077 | 508.189 |
| baseline-medium | repeat_passed | qualified | heldout | 511.892 | 2618.591 |
| baseline-large | repeat_passed | qualified | heldout | 6023.153 | 22167.276 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | channels, detections, snapshots, az, el | [4,64,16,21,11] | {"max_sources_bound":4} | {} |
| baseline-medium | medium | channels, detections, snapshots, az, el | [8,128,32,41,21] | {"max_sources_bound":4} | {} |
| baseline-large | large | channels, detections, snapshots, az, el | [16,256,48,61,31] | {"max_sources_bound":4} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "angle_convention": {
    "preset": "radar_boresight"
  },
  "array_geometry": {
    "factory": "ula_geometry",
    "spacing_wavelengths": 0.5,
    "wavelength": 1.0
  },
  "az_coordinates": {
    "count": "az",
    "factory": "linspace",
    "start": -0.5,
    "stop": 0.5
  },
  "diagonal_loading": -1.0,
  "el_coordinates": {
    "count": "el",
    "factory": "linspace",
    "start": -0.3,
    "stop": 0.3
  },
  "max_sources": 1,
  "spatial_smoothing": []
}
```

Output wiring: `{"drain":["angle_spectrum","doa_multi","source_count","eigenvalues"],"optional_tensor_outputs":["angle_spectrum","doa_multi","source_count","eigenvalues"],"queue_depth":8,"timed":"aoa_estimates","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 5995.120 | 5825.655 | 13.725 / 20.0 | 9.806 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 508.189 | 508.103 | 1.212 / 5.0 | 0.248 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 521.376 | 522.930 | 4.562 / 5.0 | 0.729 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 543.825 | 549.530 | 6.536 / 5.0 | 1.766 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 508.189 | 508.103 | 1.212 / 5.0 | 0.248 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 521.376 | 522.930 | 4.562 / 5.0 | 0.729 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 543.825 | 549.530 | 6.536 / 5.0 | 1.766 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 6051.920 | 6017.195 | 2.956 / 15.0 | 2.027 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 4739.455 | 4883.880 | 16.559 / 20.0 | 16.163 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 205.077 | 205.319 | 7.361 / 5.0 | 0.425 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 223.685 | 221.811 | 6.203 / 5.0 | 2.028 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 253.024 | 255.777 | 6.744 / 5.0 | 2.738 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 205.077 | 205.319 | 7.361 / 5.0 | 0.425 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 223.685 | 221.811 | 6.203 / 5.0 | 2.028 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 253.024 | 255.777 | 6.744 / 5.0 | 2.738 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 5991.235 | 6069.670 | 2.752 / 15.0 | 2.266 / 25.0 | unresolved |
| baseline-medium/heldout | double | cold_ms | 9060.880 | 8739.425 | 8.104 / 20.0 | 7.192 / 30.0 | [-5.688, 8.006] |
| baseline-medium/heldout | double | early_median_ms | 2618.591 | 2618.182 | 0.111 / 5.0 | 0.073 / 10.0 | [-0.040, 0.066] |
| baseline-medium/heldout | double | early_p90_ms | 2632.047 | 2631.952 | 0.387 / 5.0 | 0.143 / 10.0 | [-0.172, 0.348] |
| baseline-medium/heldout | double | early_p99_ms | 2647.109 | 2646.499 | 0.606 / 5.0 | 0.325 / 10.0 | [-0.370, 0.519] |
| baseline-medium/heldout | double | median_ms | 2618.591 | 2618.182 | 0.111 / 5.0 | 0.073 / 10.0 | [-0.040, 0.066] |
| baseline-medium/heldout | double | p90_ms | 2632.047 | 2631.952 | 0.387 / 5.0 | 0.143 / 10.0 | [-0.172, 0.348] |
| baseline-medium/heldout | double | p99_ms | 2647.109 | 2646.499 | 0.606 / 5.0 | 0.325 / 10.0 | [-0.370, 0.519] |
| baseline-medium/heldout | double | setup_ms | 6409.560 | 6371.630 | 7.044 / 15.0 | 3.455 / 25.0 | [-5.713, 4.204] |
| baseline-medium/heldout | float | cold_ms | 5588.070 | 5813.830 | 13.790 / 20.0 | 12.879 / 30.0 | [-13.178, 10.621] |
| baseline-medium/heldout | float | early_median_ms | 511.892 | 512.288 | 0.524 / 5.0 | 0.261 / 10.0 | [-0.326, 0.364] |
| baseline-medium/heldout | float | early_p90_ms | 526.023 | 527.421 | 2.029 / 5.0 | 0.815 / 10.0 | [-2.020, 1.095] |
| baseline-medium/heldout | float | early_p99_ms | 554.506 | 554.596 | 2.941 / 5.0 | 1.041 / 10.0 | [-1.932, 1.738] |
| baseline-medium/heldout | float | median_ms | 511.892 | 512.288 | 0.524 / 5.0 | 0.261 / 10.0 | [-0.326, 0.364] |
| baseline-medium/heldout | float | p90_ms | 526.023 | 527.421 | 2.029 / 5.0 | 0.815 / 10.0 | [-2.020, 1.095] |
| baseline-medium/heldout | float | p99_ms | 554.506 | 554.596 | 2.941 / 5.0 | 1.041 / 10.0 | [-1.932, 1.738] |
| baseline-medium/heldout | float | setup_ms | 6084.450 | 6100.505 | 4.987 / 15.0 | 1.757 / 25.0 | [-4.770, 2.062] |
| baseline-large/heldout | double | cold_ms | 30998.000 | 30969.550 | 2.251 / 20.0 | 0.954 / 30.0 | [-2.170, 2.008] |
| baseline-large/heldout | double | early_median_ms | 22167.276 | 22167.212 | 0.026 / 5.0 | 0.011 / 10.0 | [-0.019, 0.023] |
| baseline-large/heldout | double | early_p90_ms | 22182.589 | 22185.019 | 0.110 / 5.0 | 0.032 / 10.0 | [-0.069, 0.060] |
| baseline-large/heldout | double | early_p99_ms | 22212.868 | 22209.407 | 0.154 / 5.0 | 0.068 / 10.0 | [-0.093, 0.095] |
| baseline-large/heldout | double | median_ms | 22167.276 | 22167.212 | 0.026 / 5.0 | 0.011 / 10.0 | [-0.019, 0.023] |
| baseline-large/heldout | double | p90_ms | 22182.589 | 22185.019 | 0.110 / 5.0 | 0.032 / 10.0 | [-0.069, 0.060] |
| baseline-large/heldout | double | p99_ms | 22212.868 | 22209.407 | 0.154 / 5.0 | 0.068 / 10.0 | [-0.093, 0.095] |
| baseline-large/heldout | double | setup_ms | 8601.840 | 8762.545 | 8.588 / 15.0 | 6.349 / 25.0 | [-8.463, 6.078] |
| baseline-large/heldout | float | cold_ms | 12414.950 | 12575.450 | 7.603 / 20.0 | 5.113 / 30.0 | [-5.176, 5.231] |
| baseline-large/heldout | float | early_median_ms | 6023.153 | 6023.682 | 0.056 / 5.0 | 0.039 / 10.0 | [-0.047, 0.045] |
| baseline-large/heldout | float | early_p90_ms | 6044.149 | 6043.820 | 0.414 / 5.0 | 0.089 / 10.0 | [-0.271, 0.170] |
| baseline-large/heldout | float | early_p99_ms | 6068.697 | 6069.759 | 0.405 / 5.0 | 0.177 / 10.0 | [-0.243, 0.361] |
| baseline-large/heldout | float | median_ms | 6023.153 | 6023.682 | 0.056 / 5.0 | 0.039 / 10.0 | [-0.047, 0.045] |
| baseline-large/heldout | float | p90_ms | 6044.149 | 6043.820 | 0.414 / 5.0 | 0.089 / 10.0 | [-0.271, 0.170] |
| baseline-large/heldout | float | p99_ms | 6068.697 | 6069.759 | 0.405 / 5.0 | 0.177 / 10.0 | [-0.243, 0.361] |
| baseline-large/heldout | float | setup_ms | 7356.960 | 7386.255 | 9.848 / 15.0 | 5.643 / 25.0 | [-9.676, 7.179] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 15.198 | 4.796 | 1.381 | 298.797 | 4573.120 | 0 | 7.129 | 0.534 |
| baseline-small/calibration | double | 6.275 | 4.495 | 1.802 | 1676.855 | 1922.965 | 1 | 10.626 | 0.544 |
| baseline-medium/heldout | float | 2.679 | 4.352 | 2.229 | 1877.660 | 1905.365 | 1 | 2.298 | 0.543 |
| baseline-medium/heldout | double | 2.919 | 10.230 | 2.569 | 3876.939 | 380.265 | 0 | 0.865 | 0.688 |
| baseline-large/heldout | float | 3.352 | 11.047 | 3.426 | 7306.992 | 165.634 | 0 | 1.069 | 0.768 |
| baseline-large/heldout | double | 5.815 | 10.966 | 3.650 | 24209.694 | 45.083 | 0 | 0.700 | 0.767 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/mvdr_estimator.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | qualified |
| baseline-large | qualified |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/mvdr_estimator/`; native evidence: `native/mvdr_estimator/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
