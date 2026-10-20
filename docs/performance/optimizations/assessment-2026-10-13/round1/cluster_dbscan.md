# cluster_dbscan: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | qualified | heldout | 655.846 | 1095.194 |
| baseline-medium | repeat_passed | qualified | heldout | 2746.139 | 5690.077 |
| baseline-large | repeat_passed | calibration_blocked | calibration | 4834.400 | 10884.362 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | points, features, clusters | [256,2,8] | {} | {} |
| baseline-medium | medium | points, features, clusters | [1024,3,32] | {} | {} |
| baseline-large | large | points, features, clusters | [2048,5,64] | {} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "epsilon": [
    0.5
  ],
  "max_clusters": 0,
  "min_points": {
    "axis": "features",
    "factory": "scaled_axis",
    "scale": 2
  }
}
```

Output wiring: `{"drain":["sizes"],"optional_tensor_outputs":["sizes"],"timed":"labels","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/heldout | double | cold_ms | 1352.940 | 1352.920 | 1.972 / 20.0 | 1.085 / 30.0 | [-0.819, 1.424] |
| baseline-small/heldout | double | early_median_ms | 1095.194 | 1095.685 | 1.073 / 5.0 | 0.279 / 10.0 | [-0.741, 0.398] |
| baseline-small/heldout | double | early_p90_ms | 1104.371 | 1105.557 | 1.043 / 5.0 | 0.445 / 10.0 | [-0.620, 0.493] |
| baseline-small/heldout | double | early_p99_ms | 1135.026 | 1134.591 | 2.404 / 5.0 | 1.010 / 10.0 | [-2.296, 1.265] |
| baseline-small/heldout | double | median_ms | 1095.194 | 1095.685 | 1.073 / 5.0 | 0.279 / 10.0 | [-0.741, 0.398] |
| baseline-small/heldout | double | p90_ms | 1104.371 | 1105.557 | 1.043 / 5.0 | 0.445 / 10.0 | [-0.620, 0.493] |
| baseline-small/heldout | double | p99_ms | 1135.026 | 1134.591 | 2.404 / 5.0 | 1.010 / 10.0 | [-2.296, 1.265] |
| baseline-small/heldout | double | setup_ms | 3396.410 | 3404.960 | 6.760 / 15.0 | 4.813 / 25.0 | [-6.529, 6.289] |
| baseline-small/heldout | float | cold_ms | 915.691 | 914.881 | 2.255 / 20.0 | 1.365 / 30.0 | [-1.686, 1.792] |
| baseline-small/heldout | float | early_median_ms | 655.846 | 656.467 | 2.001 / 5.0 | 0.622 / 10.0 | [-1.917, 1.468] |
| baseline-small/heldout | float | early_p90_ms | 665.535 | 665.973 | 1.639 / 5.0 | 0.870 / 10.0 | [-0.959, 0.735] |
| baseline-small/heldout | float | early_p99_ms | 691.690 | 694.804 | 3.946 / 5.0 | 2.282 / 10.0 | [-2.614, 3.045] |
| baseline-small/heldout | float | median_ms | 655.846 | 656.467 | 2.001 / 5.0 | 0.622 / 10.0 | [-1.917, 1.468] |
| baseline-small/heldout | float | p90_ms | 665.535 | 665.973 | 1.639 / 5.0 | 0.870 / 10.0 | [-0.959, 0.735] |
| baseline-small/heldout | float | p99_ms | 691.690 | 694.804 | 3.946 / 5.0 | 2.282 / 10.0 | [-2.614, 3.045] |
| baseline-small/heldout | float | setup_ms | 3393.855 | 3398.710 | 7.863 / 15.0 | 3.388 / 25.0 | [-4.382, 3.980] |
| baseline-medium/heldout | double | cold_ms | 5957.610 | 5953.280 | 0.629 / 20.0 | 0.359 / 30.0 | [-0.455, 0.441] |
| baseline-medium/heldout | double | early_median_ms | 5690.077 | 5688.188 | 0.170 / 5.0 | 0.081 / 10.0 | [-0.072, 0.138] |
| baseline-medium/heldout | double | early_p90_ms | 5707.573 | 5708.866 | 0.411 / 5.0 | 0.111 / 10.0 | [-0.171, 0.355] |
| baseline-medium/heldout | double | early_p99_ms | 5742.234 | 5736.611 | 2.518 / 5.0 | 0.334 / 10.0 | [-0.429, 2.468] |
| baseline-medium/heldout | double | median_ms | 5690.077 | 5688.188 | 0.170 / 5.0 | 0.081 / 10.0 | [-0.072, 0.138] |
| baseline-medium/heldout | double | p90_ms | 5707.573 | 5708.866 | 0.411 / 5.0 | 0.111 / 10.0 | [-0.171, 0.355] |
| baseline-medium/heldout | double | p99_ms | 5742.234 | 5736.611 | 2.518 / 5.0 | 0.334 / 10.0 | [-0.429, 2.468] |
| baseline-medium/heldout | double | setup_ms | 3404.680 | 3412.505 | 6.634 / 15.0 | 4.790 / 25.0 | [-4.831, 4.961] |
| baseline-medium/heldout | float | cold_ms | 3002.480 | 3000.915 | 0.753 / 20.0 | 0.482 / 30.0 | [-0.475, 0.715] |
| baseline-medium/heldout | float | early_median_ms | 2746.139 | 2746.103 | 0.315 / 5.0 | 0.165 / 10.0 | [-0.224, 0.309] |
| baseline-medium/heldout | float | early_p90_ms | 2760.564 | 2761.111 | 0.786 / 5.0 | 0.110 / 10.0 | [-0.120, 0.729] |
| baseline-medium/heldout | float | early_p99_ms | 2790.667 | 2788.302 | 1.717 / 5.0 | 0.432 / 10.0 | [-0.485, 1.084] |
| baseline-medium/heldout | float | median_ms | 2746.139 | 2746.103 | 0.315 / 5.0 | 0.165 / 10.0 | [-0.224, 0.309] |
| baseline-medium/heldout | float | p90_ms | 2760.564 | 2761.111 | 0.786 / 5.0 | 0.110 / 10.0 | [-0.120, 0.729] |
| baseline-medium/heldout | float | p99_ms | 2790.667 | 2788.302 | 1.717 / 5.0 | 0.432 / 10.0 | [-0.485, 1.084] |
| baseline-medium/heldout | float | setup_ms | 3383.830 | 3407.555 | 5.573 / 15.0 | 3.782 / 25.0 | [-5.146, 4.530] |
| baseline-large/calibration | double | cold_ms | 11149.750 | 11157.600 | 0.190 / 20.0 | 0.177 / 30.0 | unresolved |
| baseline-large/calibration | double | early_median_ms | 10884.362 | 10883.832 | 0.136 / 5.0 | 0.021 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p90_ms | 10900.915 | 10902.428 | 0.269 / 5.0 | 0.076 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p99_ms | 10937.437 | 12024.653 | 12.382 / 5.0 | 9.841 / 10.0 | unresolved |
| baseline-large/calibration | double | median_ms | 10884.362 | 10883.832 | 0.136 / 5.0 | 0.021 / 10.0 | unresolved |
| baseline-large/calibration | double | p90_ms | 10900.915 | 10902.428 | 0.269 / 5.0 | 0.076 / 10.0 | unresolved |
| baseline-large/calibration | double | p99_ms | 10937.437 | 12024.653 | 12.382 / 5.0 | 9.841 / 10.0 | unresolved |
| baseline-large/calibration | double | setup_ms | 3217.250 | 3299.090 | 5.797 / 15.0 | 5.228 / 25.0 | unresolved |
| baseline-large/calibration | float | cold_ms | 5088.550 | 5091.810 | 0.697 / 20.0 | 0.262 / 30.0 | unresolved |
| baseline-large/calibration | float | early_median_ms | 4834.400 | 4833.990 | 0.156 / 5.0 | 0.107 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p90_ms | 4848.738 | 4855.151 | 0.391 / 5.0 | 0.233 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p99_ms | 6004.474 | 6077.827 | 2.186 / 5.0 | 1.846 / 10.0 | unresolved |
| baseline-large/calibration | float | median_ms | 4834.400 | 4833.990 | 0.156 / 5.0 | 0.107 / 10.0 | unresolved |
| baseline-large/calibration | float | p90_ms | 4848.738 | 4855.151 | 0.391 / 5.0 | 0.233 / 10.0 | unresolved |
| baseline-large/calibration | float | p99_ms | 6004.474 | 6077.827 | 2.186 / 5.0 | 1.846 / 10.0 | unresolved |
| baseline-large/calibration | float | setup_ms | 3541.555 | 3425.525 | 6.483 / 15.0 | 6.168 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/heldout | float | 13.123 | 9.663 | 5.084 | 2713.829 | 1511.640 | 3 | 2.804 | 0.523 |
| baseline-small/heldout | double | 11.747 | 14.725 | 8.521 | 2808.894 | 909.202 | 0 | 4.730 | 0.385 |
| baseline-medium/heldout | float | 8.675 | 14.880 | 6.561 | 4645.334 | 363.254 | 0 | 1.499 | 0.513 |
| baseline-medium/heldout | double | 9.670 | 16.643 | 7.329 | 7117.160 | 175.540 | 0 | 0.883 | 0.582 |
| baseline-large/calibration | float | 7.513 | 16.321 | 8.133 | 13391.968 | 204.503 | 0 | 12.236 | 0.254 |
| baseline-large/calibration | double | 14.813 | 16.647 | 5.597 | 13721.559 | 91.723 | 0 | 0.318 | 0.573 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/cluster_dbscan.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | qualified |
| baseline-medium | qualified |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/cluster_dbscan/`; native evidence: `native/cluster_dbscan/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
