# music_estimator: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 3515.530 | 4358.581 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 4586.625 | 7952.561 |
| baseline-large | repeat_passed | heldout_budget_unresolved | calibration | 11839.646 | 31434.184 |

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
  "el_coordinates": {
    "count": "el",
    "factory": "linspace",
    "start": -0.3,
    "stop": 0.3
  },
  "max_sources": 4,
  "num_sources": -1,
  "source_count_pfa": 0.01,
  "spatial_smoothing": []
}
```

Output wiring: `{"drain":["angle_spectrum","doa_multi","source_count","eigenvalues"],"optional_tensor_outputs":["angle_spectrum","doa_multi","source_count","eigenvalues"],"queue_depth":8,"timed":"aoa_estimates","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 4549.205 | 4816.555 | 20.583 / 20.0 | 13.544 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 4358.581 | 4202.822 | 9.836 / 5.0 | 8.562 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 5264.221 | 5018.842 | 13.646 / 5.0 | 11.862 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 5626.584 | 5440.191 | 11.498 / 5.0 | 5.872 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 4358.581 | 4202.822 | 9.836 / 5.0 | 8.562 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 5264.221 | 5018.842 | 13.646 / 5.0 | 11.862 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 5626.584 | 5440.191 | 11.498 / 5.0 | 5.872 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 6033.760 | 6010.305 | 3.322 / 15.0 | 2.165 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 3727.315 | 3562.545 | 15.880 / 20.0 | 15.145 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 3515.530 | 3491.176 | 18.862 / 5.0 | 4.742 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 4279.496 | 4118.466 | 13.846 / 5.0 | 8.792 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 4346.293 | 4374.740 | 13.189 / 5.0 | 1.861 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 3515.530 | 3491.176 | 18.862 / 5.0 | 4.742 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 4279.496 | 4118.466 | 13.846 / 5.0 | 8.792 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 4346.293 | 4374.740 | 13.189 / 5.0 | 1.861 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 6086.775 | 6050.650 | 2.390 / 15.0 | 1.100 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 7932.660 | 7909.050 | 12.244 / 20.0 | 6.751 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 7952.561 | 7954.664 | 1.931 / 5.0 | 1.017 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 8189.343 | 8382.337 | 7.927 / 5.0 | 6.479 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 8487.105 | 8711.067 | 9.602 / 5.0 | 6.566 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 7952.561 | 7954.664 | 1.931 / 5.0 | 1.017 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 8189.343 | 8382.337 | 7.927 / 5.0 | 6.479 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 8487.105 | 8711.067 | 9.602 / 5.0 | 6.566 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 6525.560 | 6528.315 | 4.712 / 15.0 | 4.732 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 4415.370 | 4497.030 | 19.283 / 20.0 | 12.840 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 4586.625 | 4549.414 | 10.918 / 5.0 | 8.639 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 5394.090 | 5302.400 | 13.877 / 5.0 | 3.785 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 5828.581 | 5451.152 | 10.449 / 5.0 | 7.545 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 4586.625 | 4549.414 | 10.918 / 5.0 | 8.639 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 5394.090 | 5302.400 | 13.877 / 5.0 | 3.785 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 5828.581 | 5451.152 | 10.449 / 5.0 | 7.545 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 6259.385 | 6209.500 | 4.500 / 15.0 | 3.368 / 25.0 | unresolved |
| baseline-large/calibration | double | cold_ms | 30680.800 | 30586.450 | 3.856 / 20.0 | 2.634 / 30.0 | unresolved |
| baseline-large/calibration | double | early_median_ms | 31434.184 | 30970.095 | 1.815 / 5.0 | 1.752 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p90_ms | 31985.809 | 31595.747 | 1.345 / 5.0 | 1.246 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p99_ms | 32142.992 | 31714.592 | 1.500 / 5.0 | 1.301 / 10.0 | unresolved |
| baseline-large/calibration | double | median_ms | 31434.184 | 30970.095 | 1.815 / 5.0 | 1.752 / 10.0 | unresolved |
| baseline-large/calibration | double | p90_ms | 31985.809 | 31595.747 | 1.345 / 5.0 | 1.246 / 10.0 | unresolved |
| baseline-large/calibration | double | p99_ms | 32142.992 | 31714.592 | 1.500 / 5.0 | 1.301 / 10.0 | unresolved |
| baseline-large/calibration | double | setup_ms | 8915.595 | 8783.660 | 8.413 / 15.0 | 6.015 / 25.0 | unresolved |
| baseline-large/calibration | float | cold_ms | 11910.500 | 12127.550 | 6.386 / 20.0 | 4.678 / 30.0 | unresolved |
| baseline-large/calibration | float | early_median_ms | 11839.646 | 11916.740 | 2.881 / 5.0 | 2.160 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p90_ms | 12400.339 | 12386.687 | 4.166 / 5.0 | 0.783 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p99_ms | 12618.294 | 12533.039 | 4.284 / 5.0 | 2.268 / 10.0 | unresolved |
| baseline-large/calibration | float | median_ms | 11839.646 | 11916.740 | 2.881 / 5.0 | 2.160 / 10.0 | unresolved |
| baseline-large/calibration | float | p90_ms | 12400.339 | 12386.687 | 4.166 / 5.0 | 0.783 / 10.0 | unresolved |
| baseline-large/calibration | float | p99_ms | 12618.294 | 12533.039 | 4.284 / 5.0 | 2.268 / 10.0 | unresolved |
| baseline-large/calibration | float | setup_ms | 7461.200 | 7480.250 | 7.177 / 15.0 | 4.385 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 598.732 | 768.114 | 641.541 | 5272.604 | 278.315 | 0 | 20.160 | 0.702 |
| baseline-small/calibration | double | 404.688 | 767.441 | 515.207 | 6751.053 | 231.363 | 0 | 17.151 | 0.902 |
| baseline-medium/calibration | float | 501.350 | 739.120 | 637.234 | 6666.496 | 213.013 | 0 | 18.220 | 0.692 |
| baseline-medium/calibration | double | 154.005 | 675.060 | 116.028 | 14586.321 | 125.469 | 0 | 9.536 | 0.864 |
| baseline-large/calibration | float | 348.745 | 698.750 | 344.003 | 14583.042 | 83.409 | 0 | 7.964 | 0.908 |
| baseline-large/calibration | double | 570.779 | 693.886 | 671.655 | 35740.613 | 31.946 | 0 | 3.102 | 0.787 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/music_estimator.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/music_estimator/`; native evidence: `native/music_estimator/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
