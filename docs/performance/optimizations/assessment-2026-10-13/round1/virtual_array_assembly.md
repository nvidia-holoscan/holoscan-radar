# virtual_array_assembly: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 20.118 | 20.213 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 354.566 | 594.939 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 3876.336 | 7533.869 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | tx, rx, doppler, range | [2,4,32,128] | {} | {} |
| baseline-medium | medium | tx, rx, doppler, range | [4,16,128,512] | {} | {} |
| baseline-large | large | tx, rx, doppler, range | [8,32,256,1024] | {} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "overlap_policy": "dedup",
  "rx_positions": {
    "factory": "linear_positions",
    "spacing": 0.5
  },
  "tx_positions": {
    "factory": "linear_positions",
    "spacing": 0.5
  },
  "wavelength": 0.03
}
```

Output wiring: `{"drain":["array_geometry"],"drain_codecs":{"array_geometry":"array_geometry"},"optional_tensor_outputs":[],"timed":"rd_maps_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 261.760 | 257.834 | 2.209 / 20.0 | 1.059 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 20.213 | 20.294 | 0.884 / 5.0 | 0.360 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 20.617 | 20.710 | 0.609 / 5.0 | 0.349 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 33.084 | 23.552 | 50.198 / 5.0 | 40.756 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 20.213 | 20.294 | 0.884 / 5.0 | 0.360 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 20.617 | 20.710 | 0.609 / 5.0 | 0.349 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 33.084 | 23.552 | 50.198 / 5.0 | 40.756 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 3733.305 | 3516.130 | 6.304 / 15.0 | 3.596 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 259.491 | 257.746 | 0.878 / 20.0 | 1.088 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 20.118 | 20.166 | 0.306 / 5.0 | 0.832 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 21.000 | 20.622 | 2.455 / 5.0 | 1.153 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 46.281 | 21.456 | 53.858 / 5.0 | 72.845 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 20.118 | 20.166 | 0.306 / 5.0 | 0.832 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 21.000 | 20.622 | 2.455 / 5.0 | 1.153 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 46.281 | 21.456 | 53.858 / 5.0 | 72.845 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 3337.965 | 3383.385 | 2.287 / 15.0 | 7.845 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 812.364 | 819.540 | 3.475 / 20.0 | 2.873 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 594.939 | 607.208 | 4.720 / 5.0 | 2.558 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 634.564 | 642.659 | 2.364 / 5.0 | 2.204 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 667.581 | 672.216 | 3.744 / 5.0 | 2.171 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 594.939 | 607.208 | 4.720 / 5.0 | 2.558 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 634.564 | 642.659 | 2.364 / 5.0 | 2.204 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 667.581 | 672.216 | 3.744 / 5.0 | 2.171 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 12634.050 | 12651.300 | 6.057 / 15.0 | 3.675 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 555.478 | 554.262 | 5.231 / 20.0 | 2.876 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 354.566 | 358.068 | 7.246 / 5.0 | 3.107 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 375.532 | 377.663 | 7.993 / 5.0 | 2.525 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 411.197 | 411.117 | 3.999 / 5.0 | 2.265 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 354.566 | 358.068 | 7.246 / 5.0 | 3.107 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 375.532 | 377.663 | 7.993 / 5.0 | 2.525 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 411.197 | 411.117 | 3.999 / 5.0 | 2.265 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 8222.585 | 8226.350 | 10.916 / 15.0 | 9.731 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 7671.800 | 7718.325 | 1.816 / 20.0 | 1.019 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 7533.869 | 7577.711 | 2.885 / 5.0 | 2.099 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 7634.859 | 7766.893 | 3.803 / 5.0 | 1.495 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 7697.627 | 7862.176 | 3.765 / 5.0 | 1.134 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 7533.869 | 7577.711 | 2.885 / 5.0 | 2.099 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 7634.859 | 7766.893 | 3.803 / 5.0 | 1.495 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 7697.627 | 7862.176 | 3.765 / 5.0 | 1.134 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 117158.500 | 116602.000 | 0.586 / 15.0 | 0.297 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 4116.890 | 4120.660 | 0.915 / 20.0 | 1.773 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 3876.336 | 3887.385 | 0.426 / 5.0 | 1.496 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 4012.663 | 3993.654 | 0.738 / 5.0 | 0.309 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 4113.404 | 4059.555 | 1.675 / 5.0 | 1.085 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 3876.336 | 3887.385 | 0.426 / 5.0 | 1.496 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 4012.663 | 3993.654 | 0.738 / 5.0 | 0.309 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 4113.404 | 4059.555 | 1.675 / 5.0 | 1.085 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 61431.650 | 61966.250 | 1.023 / 15.0 | 0.455 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.061 | 0.505 | 0.248 | 149.278 | 43360.650 | 1 | 2.004 | 0.456 |
| baseline-small/screen | double | 0.178 | 0.430 | 0.129 | 55.981 | 43379.150 | 0 | 0.868 | 0.485 |
| baseline-medium/calibration | float | 26.792 | 20.162 | 13.338 | 576.048 | 2799.835 | 0 | 16.823 | 0.854 |
| baseline-medium/calibration | double | 28.048 | 25.768 | 20.727 | 772.707 | 1650.390 | 0 | 2.569 | 0.606 |
| baseline-large/screen | float | 16.373 | 145.962 | 72.203 | 4196.984 | 256.765 | 0 | 1.054 | 0.380 |
| baseline-large/screen | double | 215.338 | 148.946 | 241.804 | 8101.135 | 132.161 | 0 | 1.480 | 0.810 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/virtual_array_assembly.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/virtual_array_assembly/`; native evidence: `native/virtual_array_assembly/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
