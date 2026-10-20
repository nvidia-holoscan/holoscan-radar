# esprit_estimator: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 4464.263 | 4155.533 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 10819.698 | 17969.395 |
| baseline-large | repeat_passed | calibration_blocked | calibration | 7023.237 | 34016.203 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | channels, detections, snapshots | [4,64,16] | {"max_sources_bound":4} | {} |
| baseline-medium | medium | channels, detections, snapshots | [8,128,32] | {"max_sources_bound":4} | {} |
| baseline-large | large | channels, detections, snapshots | [16,256,48] | {"max_sources_bound":4} | {} |

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
  "max_sources": 4,
  "num_sources": -1,
  "source_count_pfa": 0.01,
  "spatial_smoothing": []
}
```

Output wiring: `{"drain":["doa_multi","source_count","eigenvalues"],"optional_tensor_outputs":["doa_multi","source_count","eigenvalues"],"queue_depth":8,"timed":"aoa_estimates","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 4238.620 | 4226.045 | 16.732 / 20.0 | 13.609 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 4155.533 | 4055.345 | 10.606 / 5.0 | 6.963 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 4743.604 | 4767.057 | 9.948 / 5.0 | 12.481 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 5411.644 | 5356.021 | 7.665 / 5.0 | 10.290 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 4155.533 | 4055.345 | 10.606 / 5.0 | 6.963 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 4743.604 | 4767.057 | 9.948 / 5.0 | 12.481 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 5411.644 | 5356.021 | 7.665 / 5.0 | 10.290 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 5507.340 | 5486.005 | 3.712 / 15.0 | 2.062 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 4540.345 | 4547.980 | 17.738 / 20.0 | 12.346 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 4464.263 | 4473.511 | 5.665 / 5.0 | 5.477 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 5191.323 | 5283.407 | 7.144 / 5.0 | 6.731 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 5349.322 | 5695.440 | 13.059 / 5.0 | 10.476 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 4464.263 | 4473.511 | 5.665 / 5.0 | 5.477 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 5191.323 | 5283.407 | 7.144 / 5.0 | 6.731 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 5349.322 | 5695.440 | 13.059 / 5.0 | 10.476 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 5530.980 | 5519.105 | 1.095 / 15.0 | 1.904 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 17906.450 | 17507.850 | 6.941 / 20.0 | 4.181 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 17969.395 | 17426.340 | 3.798 / 5.0 | 3.143 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 19219.424 | 18809.512 | 4.794 / 5.0 | 3.248 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 19728.629 | 19182.175 | 3.645 / 5.0 | 2.945 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 17969.395 | 17426.340 | 3.798 / 5.0 | 3.143 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 19219.424 | 18809.512 | 4.794 / 5.0 | 3.248 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 19728.629 | 19182.175 | 3.645 / 5.0 | 2.945 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 5641.950 | 5639.430 | 4.115 / 15.0 | 1.618 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 10650.400 | 10479.900 | 7.623 / 20.0 | 6.039 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 10819.698 | 10793.691 | 1.737 / 5.0 | 0.895 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 11518.696 | 11548.252 | 4.722 / 5.0 | 0.767 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 11627.649 | 11649.253 | 5.445 / 5.0 | 0.985 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 10819.698 | 10793.691 | 1.737 / 5.0 | 0.895 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 11518.696 | 11548.252 | 4.722 / 5.0 | 0.767 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 11627.649 | 11649.253 | 5.445 / 5.0 | 0.985 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 5558.175 | 5566.445 | 8.684 / 15.0 | 2.648 / 25.0 | unresolved |
| baseline-large/calibration | double | cold_ms | 33696.450 | 33774.800 | 1.928 / 20.0 | 1.853 / 30.0 | unresolved |
| baseline-large/calibration | double | early_median_ms | 34016.203 | 34112.743 | 2.039 / 5.0 | 2.202 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p90_ms | 35429.555 | 35470.996 | 1.411 / 5.0 | 1.699 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p99_ms | 35602.956 | 35710.152 | 0.772 / 5.0 | 1.115 / 10.0 | unresolved |
| baseline-large/calibration | double | median_ms | 34016.203 | 34112.743 | 2.039 / 5.0 | 2.202 / 10.0 | unresolved |
| baseline-large/calibration | double | p90_ms | 35429.555 | 35470.996 | 1.411 / 5.0 | 1.699 / 10.0 | unresolved |
| baseline-large/calibration | double | p99_ms | 35602.956 | 35710.152 | 0.772 / 5.0 | 1.115 / 10.0 | unresolved |
| baseline-large/calibration | double | setup_ms | 7591.410 | 7138.750 | 7.240 / 15.0 | 9.678 / 25.0 | unresolved |
| baseline-large/calibration | float | cold_ms | 6815.870 | 6511.725 | 12.248 / 20.0 | 8.984 / 30.0 | unresolved |
| baseline-large/calibration | float | early_median_ms | 7023.237 | 6810.778 | 13.771 / 5.0 | 11.513 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p90_ms | 7539.633 | 7478.753 | 7.446 / 5.0 | 8.769 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p99_ms | 8037.263 | 8004.588 | 6.760 / 5.0 | 6.068 / 10.0 | unresolved |
| baseline-large/calibration | float | median_ms | 7023.237 | 6810.778 | 13.771 / 5.0 | 11.513 / 10.0 | unresolved |
| baseline-large/calibration | float | p90_ms | 7539.633 | 7478.753 | 7.446 / 5.0 | 8.769 / 10.0 | unresolved |
| baseline-large/calibration | float | p99_ms | 8037.263 | 8004.588 | 6.760 / 5.0 | 6.068 / 10.0 | unresolved |
| baseline-large/calibration | float | setup_ms | 6295.970 | 6249.435 | 5.303 / 15.0 | 5.304 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 251.738 | 635.939 | 504.454 | 6825.757 | 219.244 | 0 | 15.180 | 0.684 |
| baseline-small/calibration | double | 401.977 | 843.550 | 479.879 | 6135.680 | 236.223 | 0 | 21.641 | 0.951 |
| baseline-medium/calibration | float | 191.789 | 762.711 | 218.951 | 12984.165 | 91.815 | 0 | 9.556 | 0.622 |
| baseline-medium/calibration | double | 682.593 | 833.889 | 712.902 | 21805.646 | 55.320 | 0 | 5.158 | 0.616 |
| baseline-large/calibration | float | 957.966 | 777.202 | 1054.036 | 10724.962 | 142.099 | 0 | 12.756 | 0.655 |
| baseline-large/calibration | double | 692.826 | 981.658 | 983.451 | 36147.012 | 29.038 | 0 | 2.990 | 0.635 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/esprit_estimator.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/esprit_estimator/`; native evidence: `native/esprit_estimator/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
