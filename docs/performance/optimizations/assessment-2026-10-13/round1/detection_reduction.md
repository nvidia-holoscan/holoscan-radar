# detection_reduction: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 206.100 | 205.778 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 235.623 | 236.019 |
| baseline-large | repeat_passed | validation_blocked | heldout | 294.938 | 295.554 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | points, clusters, features | [256,32,2] | {"two":2} | {} |
| baseline-medium | medium | points, clusters, features | [1024,64,2] | {"two":2} | {} |
| baseline-large | large | points, clusters, features | [4096,128,2] | {"two":2} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "feature_weights": {
    "count": "features",
    "factory": "ones"
  },
  "representative": "peak",
  "score_method": "weighted_sum"
}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":["reduced_points"],"timed":"rep_idx","timed_pair":"reduced_points","unwired":[]}`.

Native output comparison: `detection-reduction-v1 with fresh input witnesses and coupled ports`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 447.450 | 449.315 | 3.490 / 20.0 | 1.595 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 205.778 | 206.262 | 2.956 / 5.0 | 0.440 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 215.341 | 215.125 | 6.590 / 5.0 | 2.703 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 242.347 | 241.021 | 6.848 / 5.0 | 5.132 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 205.778 | 206.262 | 2.956 / 5.0 | 0.440 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 215.341 | 215.125 | 6.590 / 5.0 | 2.703 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 242.347 | 241.021 | 6.848 / 5.0 | 5.132 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 3977.845 | 3832.130 | 5.956 / 15.0 | 4.116 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 452.321 | 447.945 | 3.736 / 20.0 | 2.476 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 206.100 | 205.929 | 2.871 / 5.0 | 0.637 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 210.351 | 211.493 | 7.287 / 5.0 | 2.375 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 237.140 | 239.896 | 6.668 / 5.0 | 4.613 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 206.100 | 205.929 | 2.871 / 5.0 | 0.637 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 210.351 | 211.493 | 7.287 / 5.0 | 2.375 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 237.140 | 239.896 | 6.668 / 5.0 | 4.613 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 3835.585 | 3856.145 | 6.784 / 15.0 | 4.106 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 481.491 | 477.783 | 3.656 / 20.0 | 1.276 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 236.019 | 235.711 | 0.876 / 5.0 | 0.575 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 244.085 | 245.512 | 3.609 / 5.0 | 2.935 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 270.213 | 274.437 | 6.665 / 5.0 | 2.972 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 236.019 | 235.711 | 0.876 / 5.0 | 0.575 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 244.085 | 245.512 | 3.609 / 5.0 | 2.935 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 270.213 | 274.437 | 6.665 / 5.0 | 2.972 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 3911.205 | 3829.405 | 5.719 / 15.0 | 4.264 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 482.094 | 477.487 | 5.824 / 20.0 | 2.974 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 235.623 | 235.708 | 0.808 / 5.0 | 0.450 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 243.196 | 245.141 | 2.651 / 5.0 | 1.644 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 269.346 | 276.925 | 11.605 / 5.0 | 4.418 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 235.623 | 235.708 | 0.808 / 5.0 | 0.450 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 243.196 | 245.141 | 2.651 / 5.0 | 1.644 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 269.346 | 276.925 | 11.605 / 5.0 | 4.418 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 3835.285 | 3895.125 | 7.576 / 15.0 | 6.267 / 25.0 | unresolved |
| baseline-large/heldout | double | cold_ms | 543.020 | 545.692 | 3.727 / 20.0 | 1.708 / 30.0 | [-3.413, 2.629] |
| baseline-large/heldout | double | early_median_ms | 295.554 | 295.813 | 0.872 / 5.0 | 0.444 / 10.0 | [-0.810, 0.459] |
| baseline-large/heldout | double | early_p90_ms | 305.739 | 306.535 | 4.562 / 5.0 | 1.731 / 10.0 | [-4.138, 2.546] |
| baseline-large/heldout | double | early_p99_ms | 332.266 | 337.923 | 7.016 / 5.0 | 3.655 / 10.0 | [-6.749, 4.356] |
| baseline-large/heldout | double | median_ms | 295.554 | 295.813 | 0.872 / 5.0 | 0.444 / 10.0 | [-0.810, 0.459] |
| baseline-large/heldout | double | p90_ms | 305.739 | 306.535 | 4.562 / 5.0 | 1.731 / 10.0 | [-4.138, 2.546] |
| baseline-large/heldout | double | p99_ms | 332.266 | 337.923 | 7.016 / 5.0 | 3.655 / 10.0 | [-6.749, 4.356] |
| baseline-large/heldout | double | setup_ms | 3874.665 | 3876.800 | 4.504 / 15.0 | 4.512 / 25.0 | [-3.980, 3.114] |
| baseline-large/heldout | float | cold_ms | 543.089 | 544.590 | 4.381 / 20.0 | 2.554 / 30.0 | [-4.028, 2.749] |
| baseline-large/heldout | float | early_median_ms | 294.938 | 294.876 | 0.934 / 5.0 | 0.375 / 10.0 | [-0.850, 0.596] |
| baseline-large/heldout | float | early_p90_ms | 304.245 | 304.776 | 3.509 / 5.0 | 1.355 / 10.0 | [-2.222, 1.178] |
| baseline-large/heldout | float | early_p99_ms | 335.167 | 334.358 | 7.564 / 5.0 | 2.716 / 10.0 | [-5.137, 6.138] |
| baseline-large/heldout | float | median_ms | 294.938 | 294.876 | 0.934 / 5.0 | 0.375 / 10.0 | [-0.850, 0.596] |
| baseline-large/heldout | float | p90_ms | 304.245 | 304.776 | 3.509 / 5.0 | 1.355 / 10.0 | [-2.222, 1.178] |
| baseline-large/heldout | float | p99_ms | 335.167 | 334.358 | 7.564 / 5.0 | 2.716 / 10.0 | [-5.137, 6.138] |
| baseline-large/heldout | float | setup_ms | 3859.050 | 3844.240 | 6.230 / 15.0 | 4.609 / 25.0 | [-4.431, 4.678] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 6.583 | 3.563 | 2.300 | 735.288 | 4735.325 | 1 | 1.104 | 0.480 |
| baseline-small/calibration | double | 6.066 | 3.925 | 1.969 | 395.371 | 4709.880 | 0 | 3.947 | 0.393 |
| baseline-medium/calibration | float | 1.908 | 3.990 | 2.372 | 905.121 | 4139.795 | 2 | 1.617 | 0.550 |
| baseline-medium/calibration | double | 2.077 | 3.933 | 1.652 | 702.083 | 4130.940 | 0 | 1.566 | 0.535 |
| baseline-large/heldout | float | 2.755 | 3.518 | 2.284 | 2843.212 | 3318.600 | 13 | 1.996 | 0.561 |
| baseline-large/heldout | double | 2.572 | 3.988 | 2.346 | 4815.941 | 3308.055 | 14 | 60.758 | 0.594 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/detection_reduction.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/detection_reduction/`; native evidence: `native/detection_reduction/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
