# cfar_detector: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 101.583 | 126.919 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 117.250 | 197.742 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 142.447 | 303.577 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | doppler, range | [64,256] | {"channels":1,"tx":1} | {} |
| baseline-medium | medium | doppler, range | [128,512] | {"channels":1,"tx":1} | {} |
| baseline-large | large | doppler, range | [256,512] | {"channels":1,"tx":1} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "cfar_variant": "CA",
  "guard_cells": [
    1,
    1
  ],
  "peak_mode": "none",
  "pfa": 1e-06,
  "threshold_scale": 1.0,
  "training_cells": [
    2,
    2
  ]
}
```

Output wiring: `{"drain":["detection_mask","detection_score","noise_estimate","cfar_power"],"optional_tensor_outputs":["detection_mask","detection_score","noise_estimate","cfar_power"],"timed":"detection_idx","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 394.033 | 391.042 | 3.804 / 20.0 | 1.392 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 126.919 | 126.424 | 1.371 / 5.0 | 0.439 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 133.097 | 131.752 | 5.302 / 5.0 | 4.881 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 157.227 | 156.464 | 15.563 / 5.0 | 9.701 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 126.919 | 126.424 | 1.371 / 5.0 | 0.439 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 133.097 | 131.752 | 5.302 / 5.0 | 4.881 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 157.227 | 156.464 | 15.563 / 5.0 | 9.701 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 5690.685 | 5612.505 | 5.452 / 15.0 | 5.131 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 368.691 | 367.398 | 5.052 / 20.0 | 4.428 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 101.583 | 101.303 | 2.127 / 5.0 | 0.844 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 107.824 | 106.321 | 6.971 / 5.0 | 4.699 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 135.119 | 130.116 | 11.846 / 5.0 | 6.699 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 101.583 | 101.303 | 2.127 / 5.0 | 0.844 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 107.824 | 106.321 | 6.971 / 5.0 | 4.699 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 135.119 | 130.116 | 11.846 / 5.0 | 6.699 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 5688.940 | 5714.135 | 4.149 / 15.0 | 2.811 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 470.293 | 460.141 | 4.773 / 20.0 | 3.038 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 197.742 | 195.142 | 3.850 / 5.0 | 2.284 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 205.830 | 203.443 | 7.698 / 5.0 | 3.054 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 233.211 | 231.276 | 12.744 / 5.0 | 3.070 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 197.742 | 195.142 | 3.850 / 5.0 | 2.284 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 205.830 | 203.443 | 7.698 / 5.0 | 3.054 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 233.211 | 231.276 | 12.744 / 5.0 | 3.070 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 6066.075 | 6030.690 | 3.500 / 15.0 | 1.078 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 389.751 | 385.252 | 6.603 / 20.0 | 2.207 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 117.250 | 117.153 | 1.980 / 5.0 | 1.363 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 124.380 | 128.860 | 9.090 / 5.0 | 6.635 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 152.555 | 151.006 | 16.824 / 5.0 | 6.118 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 117.250 | 117.153 | 1.980 / 5.0 | 1.363 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 124.380 | 128.860 | 9.090 / 5.0 | 6.635 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 152.555 | 151.006 | 16.824 / 5.0 | 6.118 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 5931.205 | 5748.315 | 6.422 / 15.0 | 3.571 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 594.418 | 602.714 | 1.554 / 20.0 | 1.666 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 303.577 | 308.327 | 2.126 / 5.0 | 1.093 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 319.468 | 320.763 | 0.803 / 5.0 | 0.273 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 340.097 | 345.440 | 2.796 / 5.0 | 0.818 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 303.577 | 308.327 | 2.126 / 5.0 | 1.093 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 319.468 | 320.763 | 0.803 / 5.0 | 0.273 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 340.097 | 345.440 | 2.796 / 5.0 | 0.818 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 6720.020 | 6778.350 | 1.374 / 15.0 | 0.479 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 417.394 | 414.755 | 0.776 / 20.0 | 2.371 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 142.447 | 144.414 | 5.348 / 5.0 | 3.141 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 151.098 | 155.440 | 5.292 / 5.0 | 1.451 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 186.669 | 189.625 | 2.082 / 5.0 | 3.459 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 142.447 | 144.414 | 5.348 / 5.0 | 3.141 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 151.098 | 155.440 | 5.292 / 5.0 | 1.451 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 186.669 | 189.625 | 2.082 / 5.0 | 3.459 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 6128.600 | 6149.560 | 3.136 / 15.0 | 2.411 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 2.142 | 1.844 | 1.423 | 1533.235 | 8923.595 | 5 | 1.936 | 0.653 |
| baseline-small/calibration | double | 1.756 | 2.595 | 0.996 | 328.889 | 7292.165 | 0 | 2.860 | 0.542 |
| baseline-medium/calibration | float | 2.321 | 2.801 | 2.052 | 358.103 | 7773.145 | 2 | 4.054 | 0.553 |
| baseline-medium/calibration | double | 7.682 | 6.754 | 6.602 | 1262.495 | 4827.465 | 7 | 8.387 | 0.436 |
| baseline-large/screen | float | 7.546 | 6.469 | 7.848 | 227.454 | 6434.200 | 0 | 4.688 | 0.427 |
| baseline-large/screen | double | 6.405 | 16.795 | 7.227 | 359.538 | 3149.825 | 0 | 5.277 | 0.152 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/cfar_detector.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/cfar_detector/`; native evidence: `native/cfar_detector/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
