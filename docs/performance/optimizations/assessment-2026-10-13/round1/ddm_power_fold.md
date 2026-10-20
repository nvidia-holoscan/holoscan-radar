# ddm_power_fold: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 31.540 | 33.618 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 67.660 | 232.037 |
| baseline-large | repeat_passed | calibration_blocked | calibration | 843.293 | 1742.332 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | tx, rx, doppler, range | [2,4,64,256] | {} | {} |
| baseline-medium | medium | tx, rx, doppler, range | [4,8,256,512] | {} | {} |
| baseline-large | large | tx, rx, doppler, range | [4,16,512,1024] | {} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "ddm_code": {
    "factory": "linear_exact_ddm_code"
  }
}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":[],"timed":"power_map","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 268.260 | 263.065 | 2.187 / 20.0 | 1.302 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 33.618 | 33.488 | 1.664 / 5.0 | 1.173 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 34.579 | 34.442 | 1.327 / 5.0 | 0.798 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 38.362 | 42.338 | 27.656 / 5.0 | 11.819 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 33.618 | 33.488 | 1.664 / 5.0 | 1.173 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 34.579 | 34.442 | 1.327 / 5.0 | 0.798 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 38.362 | 42.338 | 27.656 / 5.0 | 11.819 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 3143.680 | 2939.575 | 9.870 / 15.0 | 4.354 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 256.167 | 264.209 | 4.330 / 20.0 | 1.953 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 31.540 | 31.738 | 1.558 / 5.0 | 0.923 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 32.403 | 36.336 | 21.779 / 5.0 | 6.122 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 35.035 | 40.645 | 18.706 / 5.0 | 10.150 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 31.540 | 31.738 | 1.558 / 5.0 | 0.923 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 32.403 | 36.336 | 21.779 / 5.0 | 6.122 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 35.035 | 40.645 | 18.706 / 5.0 | 10.150 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 2546.060 | 2814.230 | 12.075 / 15.0 | 9.161 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 429.050 | 422.496 | 3.291 / 20.0 | 1.091 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 232.037 | 230.303 | 0.921 / 5.0 | 0.641 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 246.336 | 241.848 | 2.912 / 5.0 | 1.059 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 274.091 | 261.337 | 8.929 / 5.0 | 3.314 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 232.037 | 230.303 | 0.921 / 5.0 | 0.641 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 246.336 | 241.848 | 2.912 / 5.0 | 1.059 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 274.091 | 261.337 | 8.929 / 5.0 | 3.314 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 4826.365 | 4561.860 | 8.889 / 15.0 | 3.264 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 311.306 | 307.592 | 1.855 / 20.0 | 0.778 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 67.660 | 68.336 | 1.315 / 5.0 | 0.679 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 69.581 | 78.599 | 14.357 / 5.0 | 11.399 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 70.779 | 93.113 | 32.707 / 5.0 | 26.488 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 67.660 | 68.336 | 1.315 / 5.0 | 0.679 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 69.581 | 78.599 | 14.357 / 5.0 | 11.399 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 70.779 | 93.113 | 32.707 / 5.0 | 26.488 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 4007.610 | 3831.440 | 7.777 / 15.0 | 2.485 / 25.0 | unresolved |
| baseline-large/calibration | double | cold_ms | 1966.845 | 1971.290 | 7.910 / 20.0 | 1.006 / 30.0 | unresolved |
| baseline-large/calibration | double | early_median_ms | 1742.332 | 1736.628 | 1.358 / 5.0 | 0.554 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p90_ms | 1853.346 | 1859.191 | 1.930 / 5.0 | 1.515 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p99_ms | 1902.571 | 1891.052 | 2.085 / 5.0 | 1.529 / 10.0 | unresolved |
| baseline-large/calibration | double | median_ms | 1742.332 | 1736.628 | 1.358 / 5.0 | 0.554 / 10.0 | unresolved |
| baseline-large/calibration | double | p90_ms | 1853.346 | 1859.191 | 1.930 / 5.0 | 1.515 / 10.0 | unresolved |
| baseline-large/calibration | double | p99_ms | 1902.571 | 1891.052 | 2.085 / 5.0 | 1.529 / 10.0 | unresolved |
| baseline-large/calibration | double | setup_ms | 12049.500 | 12001.450 | 4.809 / 15.0 | 4.848 / 25.0 | unresolved |
| baseline-large/calibration | float | cold_ms | 1103.495 | 1101.220 | 2.748 / 20.0 | 0.675 / 30.0 | unresolved |
| baseline-large/calibration | float | early_median_ms | 843.293 | 844.538 | 0.874 / 5.0 | 0.551 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p90_ms | 913.368 | 916.126 | 2.075 / 5.0 | 0.975 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p99_ms | 950.736 | 962.692 | 5.244 / 5.0 | 3.052 / 10.0 | unresolved |
| baseline-large/calibration | float | median_ms | 843.293 | 844.538 | 0.874 / 5.0 | 0.551 / 10.0 | unresolved |
| baseline-large/calibration | float | p90_ms | 913.368 | 916.126 | 2.075 / 5.0 | 0.975 / 10.0 | unresolved |
| baseline-large/calibration | float | p99_ms | 950.736 | 962.692 | 5.244 / 5.0 | 3.052 / 10.0 | unresolved |
| baseline-large/calibration | float | setup_ms | 8384.690 | 8466.100 | 10.238 / 15.0 | 3.476 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.492 | 0.735 | 0.698 | 50.704 | 28703.800 | 0 | 2.365 | 0.445 |
| baseline-small/screen | double | 0.563 | 0.973 | 0.357 | 56.157 | 27021.600 | 0 | 1.816 | 0.385 |
| baseline-medium/screen | float | 0.891 | 3.006 | 0.551 | 103.083 | 13761.800 | 0 | 4.251 | 0.699 |
| baseline-medium/screen | double | 2.140 | 2.830 | 2.470 | 311.269 | 4222.240 | 0 | 1.593 | 0.501 |
| baseline-large/calibration | float | 7.389 | 32.131 | 6.906 | 1649.517 | 1156.490 | 0 | 1.788 | 0.600 |
| baseline-large/calibration | double | 23.611 | 116.251 | 17.776 | 1963.258 | 568.832 | 0 | 3.754 | 0.480 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/ddm_power_fold.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/ddm_power_fold/`; native evidence: `native/ddm_power_fold/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
