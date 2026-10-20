# ddm_sparse_demultiplex: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 62.805 | 67.204 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 131.466 | 251.364 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 985.769 | 1597.295 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | tx, rx, doppler, range, detections | [2,4,64,256,16] | {"two":2} | {} |
| baseline-medium | medium | tx, rx, doppler, range, detections | [4,8,256,512,256] | {"two":2} | {} |
| baseline-large | large | tx, rx, doppler, range, detections | [4,16,512,1024,1024] | {"two":2} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "ddm_code": {
    "factory": "linear_exact_ddm_code"
  },
  "overlap_policy": "keep_all",
  "rx_positions": {
    "coordinate": 0,
    "factory": "linear_positions",
    "spacing": 0.002
  },
  "snapshot_window": [
    0,
    0
  ],
  "tx_positions": {
    "coordinate": 0,
    "factory": "linear_positions",
    "spacing": 0.02
  },
  "wavelength": 0.0039
}
```

Output wiring: `{"drain":["is_valid","array_geometry"],"drain_codecs":{"array_geometry":"array_geometry"},"optional_tensor_outputs":["is_valid"],"timed":"snapshot","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 324.797 | 327.107 | 1.725 / 20.0 | 1.143 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 67.204 | 63.431 | 8.818 / 5.0 | 3.619 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 69.052 | 66.027 | 9.614 / 5.0 | 3.894 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 75.015 | 85.539 | 35.539 / 5.0 | 11.756 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 67.204 | 63.431 | 8.818 / 5.0 | 3.619 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 69.052 | 66.027 | 9.614 / 5.0 | 3.894 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 75.015 | 85.539 | 35.539 / 5.0 | 11.756 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 4720.575 | 4803.815 | 2.387 / 15.0 | 1.763 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 322.390 | 325.761 | 1.313 / 20.0 | 0.769 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 62.805 | 63.289 | 0.902 / 5.0 | 0.559 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 66.432 | 69.847 | 9.582 / 5.0 | 2.831 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 79.731 | 84.492 | 6.468 / 5.0 | 4.773 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 62.805 | 63.289 | 0.902 / 5.0 | 0.559 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 66.432 | 69.847 | 9.582 / 5.0 | 2.831 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 79.731 | 84.492 | 6.468 / 5.0 | 4.773 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 4612.455 | 4730.505 | 3.607 / 15.0 | 2.021 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 454.117 | 463.450 | 4.274 / 20.0 | 1.848 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 251.364 | 256.178 | 2.989 / 5.0 | 2.065 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 267.071 | 272.318 | 4.460 / 5.0 | 1.618 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 281.289 | 292.061 | 6.266 / 5.0 | 2.304 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 251.364 | 256.178 | 2.989 / 5.0 | 2.065 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 267.071 | 272.318 | 4.460 / 5.0 | 1.618 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 281.289 | 292.061 | 6.266 / 5.0 | 2.304 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 6642.215 | 6690.950 | 0.884 / 15.0 | 0.912 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 394.750 | 401.343 | 2.931 / 20.0 | 0.885 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 131.466 | 132.648 | 1.056 / 5.0 | 0.798 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 134.771 | 143.505 | 8.329 / 5.0 | 5.208 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 152.511 | 156.975 | 16.264 / 5.0 | 9.432 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 131.466 | 132.648 | 1.056 / 5.0 | 0.798 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 134.771 | 143.505 | 8.329 / 5.0 | 5.208 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 152.511 | 156.975 | 16.264 / 5.0 | 9.432 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 5717.680 | 5813.255 | 2.848 / 15.0 | 0.943 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 1947.760 | 1916.345 | 2.389 / 20.0 | 0.857 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 1597.295 | 1596.689 | 0.124 / 5.0 | 0.166 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 1668.982 | 1691.213 | 2.185 / 5.0 | 0.803 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 1712.884 | 1715.699 | 0.281 / 5.0 | 0.092 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 1597.295 | 1596.689 | 0.124 / 5.0 | 0.166 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 1668.982 | 1691.213 | 2.185 / 5.0 | 0.803 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 1712.884 | 1715.699 | 0.281 / 5.0 | 0.092 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 18253.550 | 18370.950 | 3.620 / 15.0 | 2.594 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 1292.970 | 1311.045 | 1.808 / 20.0 | 0.861 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 985.769 | 985.021 | 0.097 / 5.0 | 0.470 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 1027.592 | 1022.111 | 1.122 / 5.0 | 0.433 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 1053.896 | 1052.544 | 1.640 / 5.0 | 1.264 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 985.769 | 985.021 | 0.097 / 5.0 | 0.470 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 1027.592 | 1022.111 | 1.122 / 5.0 | 0.433 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 1053.896 | 1052.544 | 1.640 / 5.0 | 1.264 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 12027.950 | 12184.900 | 1.894 / 15.0 | 0.679 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.564 | 1.829 | 1.039 | 758.214 | 14749.450 | 6 | 15.227 | 0.574 |
| baseline-small/screen | double | 6.239 | 1.428 | 1.323 | 102.834 | 14765.550 | 0 | 2.201 | 0.433 |
| baseline-medium/screen | float | 1.388 | 2.419 | 1.249 | 585.427 | 7280.285 | 1 | 1.370 | 0.546 |
| baseline-medium/screen | double | 7.574 | 7.669 | 5.947 | 366.315 | 3865.445 | 0 | 2.473 | 0.464 |
| baseline-large/screen | float | 0.950 | 18.445 | 5.530 | 1082.742 | 1004.555 | 0 | 0.749 | 0.500 |
| baseline-large/screen | double | 1.976 | 44.764 | 4.120 | 1742.279 | 618.154 | 0 | 0.502 | 0.529 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/ddm_sparse_demultiplex.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/ddm_sparse_demultiplex/`; native evidence: `native/ddm_sparse_demultiplex/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
