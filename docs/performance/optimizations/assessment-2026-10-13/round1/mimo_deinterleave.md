# mimo_deinterleave: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 30.505 | 30.220 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 68.007 | 98.836 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 1181.844 | 2468.381 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | tx, rx, chirps_per_tx, samples | [2,4,8,64] | {} | {} |
| baseline-medium | medium | tx, rx, chirps_per_tx, samples | [4,16,32,256] | {} | {} |
| baseline-large | large | tx, rx, chirps_per_tx, samples | [8,32,64,512] | {} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "tx_schedule": {
    "factory": "round_robin"
  }
}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":[],"timed":"adc_cube_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 254.992 | 255.936 | 4.145 / 20.0 | 3.356 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 30.220 | 30.169 | 0.502 / 5.0 | 0.289 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 31.113 | 31.093 | 0.145 / 5.0 | 0.785 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 34.313 | 40.376 | 34.542 / 5.0 | 17.764 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 30.220 | 30.169 | 0.502 / 5.0 | 0.289 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 31.113 | 31.093 | 0.145 / 5.0 | 0.785 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 34.313 | 40.376 | 34.542 / 5.0 | 17.764 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 2883.720 | 2822.510 | 5.750 / 15.0 | 3.100 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 253.089 | 255.916 | 3.352 / 20.0 | 1.296 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 30.505 | 33.916 | 19.892 / 5.0 | 6.922 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 31.243 | 35.059 | 22.110 / 5.0 | 6.466 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 34.695 | 35.859 | 21.149 / 5.0 | 14.260 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 30.505 | 33.916 | 19.892 / 5.0 | 6.922 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 31.243 | 35.059 | 22.110 / 5.0 | 6.466 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 34.695 | 35.859 | 21.149 / 5.0 | 14.260 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 2790.675 | 2938.035 | 7.476 / 15.0 | 3.861 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 333.462 | 332.505 | 1.682 / 20.0 | 0.790 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 98.836 | 99.438 | 0.859 / 5.0 | 1.225 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 107.930 | 109.049 | 13.322 / 5.0 | 10.285 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 118.971 | 134.233 | 30.322 / 5.0 | 7.315 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 98.836 | 99.438 | 0.859 / 5.0 | 1.225 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 107.930 | 109.049 | 13.322 / 5.0 | 10.285 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 118.971 | 134.233 | 30.322 / 5.0 | 7.315 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 4874.485 | 4962.220 | 19.204 / 15.0 | 15.622 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 317.658 | 318.389 | 0.674 / 20.0 | 2.258 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 68.007 | 64.116 | 10.281 / 5.0 | 3.642 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 71.558 | 70.249 | 7.911 / 5.0 | 5.883 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 76.966 | 76.515 | 15.578 / 5.0 | 7.387 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 68.007 | 64.116 | 10.281 / 5.0 | 3.642 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 71.558 | 70.249 | 7.911 / 5.0 | 5.883 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 76.966 | 76.515 | 15.578 / 5.0 | 7.387 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 4035.780 | 4264.340 | 5.987 / 15.0 | 11.222 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 2644.485 | 2632.360 | 2.206 / 20.0 | 1.635 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 2468.381 | 2463.140 | 0.250 / 5.0 | 0.162 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 2542.290 | 2543.240 | 0.120 / 5.0 | 0.158 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 2556.701 | 2561.986 | 0.387 / 5.0 | 0.177 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 2468.381 | 2463.140 | 0.250 / 5.0 | 0.162 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 2542.290 | 2543.240 | 0.120 / 5.0 | 0.158 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 2556.701 | 2561.986 | 0.387 / 5.0 | 0.177 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 19395.300 | 19529.300 | 2.911 / 15.0 | 1.664 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 1392.770 | 1443.075 | 7.132 / 20.0 | 2.289 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 1181.844 | 1189.566 | 1.368 / 5.0 | 0.642 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 1267.322 | 1271.206 | 1.039 / 5.0 | 0.509 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 1298.248 | 1305.867 | 2.298 / 5.0 | 1.219 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 1181.844 | 1189.566 | 1.368 / 5.0 | 0.642 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 1267.322 | 1271.206 | 1.039 / 5.0 | 0.509 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 1298.248 | 1305.867 | 2.298 / 5.0 | 1.219 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 12461.150 | 11980.350 | 6.019 / 15.0 | 2.491 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 6.112 | 0.656 | 2.259 | 46.019 | 29760.700 | 0 | 1.920 | 0.408 |
| baseline-small/screen | double | 0.152 | 0.695 | 0.421 | 61.491 | 29953.350 | 0 | 1.401 | 0.317 |
| baseline-medium/screen | float | 7.397 | 2.071 | 2.877 | 89.297 | 14654.900 | 0 | 3.946 | 0.599 |
| baseline-medium/screen | double | 0.854 | 4.021 | 2.005 | 144.546 | 9523.785 | 0 | 5.104 | 0.554 |
| baseline-large/screen | float | 16.187 | 62.640 | 13.354 | 1345.611 | 827.455 | 0 | 3.298 | 0.502 |
| baseline-large/screen | double | 6.163 | 106.884 | 8.082 | 2656.493 | 405.543 | 0 | 2.158 | 0.481 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/mimo_deinterleave.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/mimo_deinterleave/`; native evidence: `native/mimo_deinterleave/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
