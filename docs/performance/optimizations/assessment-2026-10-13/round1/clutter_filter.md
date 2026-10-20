# clutter_filter: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 783.572 | 742.979 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 807.684 | 716.837 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 806.628 | 833.352 |
| scenario-compact | repeat_passed | calibration_budget_unresolved | screen | 881.649 | 905.918 |
| scenario-larger | repeat_passed | calibration_blocked | calibration | 1633.698 | 2701.529 |
| scenario-stress | repeat_passed | qualified | heldout | 16790.354 | 32039.511 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | chirps, range_bins | [16,128] | {"channels":1} | {} |
| baseline-medium | medium | chirps, range_bins | [64,512] | {"channels":1} | {} |
| baseline-large | large | chirps, range_bins | [128,1024] | {"channels":1} | {} |
| scenario-compact | small | chirps, range_bins | [128,512] | {"channels":4} | {} |
| scenario-larger | medium | chirps, range_bins | [256,1024] | {"channels":16} | {} |
| scenario-stress | large | chirps, range_bins | [512,2048] | {"channels":64} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "doppler_window": {
    "type": "none"
  },
  "method": "mean_subtraction"
}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":[],"timed":"range_profiles_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 934.906 | 935.359 | 25.169 / 20.0 | 6.661 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 742.979 | 747.858 | 42.456 / 5.0 | 28.310 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 823.760 | 815.746 | 54.157 / 5.0 | 20.596 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 1567.256 | 1526.039 | 6.140 / 5.0 | 5.557 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 742.979 | 747.858 | 42.456 / 5.0 | 28.310 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 823.760 | 815.746 | 54.157 / 5.0 | 20.596 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 1567.256 | 1526.039 | 6.140 / 5.0 | 5.557 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 3325.370 | 3342.235 | 4.791 / 15.0 | 1.835 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 922.360 | 944.505 | 10.244 / 20.0 | 4.481 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 783.572 | 778.040 | 10.948 / 5.0 | 3.011 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 865.078 | 958.616 | 67.365 / 5.0 | 14.901 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 1566.212 | 1584.728 | 13.341 / 5.0 | 6.656 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 783.572 | 778.040 | 10.948 / 5.0 | 3.011 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 865.078 | 958.616 | 67.365 / 5.0 | 14.901 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 1566.212 | 1584.728 | 13.341 / 5.0 | 6.656 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 3345.425 | 3365.205 | 3.561 / 15.0 | 1.451 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 804.307 | 753.955 | 21.606 / 20.0 | 14.086 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 716.837 | 776.080 | 38.194 / 5.0 | 20.230 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 766.722 | 1077.081 | 100.243 / 5.0 | 26.293 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 1598.761 | 1569.404 | 4.776 / 5.0 | 2.473 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 716.837 | 776.080 | 38.194 / 5.0 | 20.230 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 766.722 | 1077.081 | 100.243 / 5.0 | 26.293 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 1598.761 | 1569.404 | 4.776 / 5.0 | 2.473 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 3379.965 | 3411.850 | 3.760 / 15.0 | 2.214 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 898.921 | 927.672 | 4.829 / 20.0 | 2.042 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 807.684 | 798.381 | 1.308 / 5.0 | 0.628 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 1140.813 | 1161.322 | 8.440 / 5.0 | 35.162 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 1539.238 | 1596.991 | 4.480 / 5.0 | 2.243 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 807.684 | 798.381 | 1.308 / 5.0 | 0.628 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 1140.813 | 1161.322 | 8.440 / 5.0 | 35.162 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 1539.238 | 1596.991 | 4.480 / 5.0 | 2.243 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 3260.695 | 3335.480 | 3.297 / 15.0 | 1.485 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 979.774 | 753.857 | 25.573 / 20.0 | 23.303 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 833.352 | 713.276 | 23.237 / 5.0 | 7.839 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 1075.852 | 1022.869 | 52.931 / 5.0 | 39.870 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 1650.158 | 1642.127 | 5.756 / 5.0 | 3.198 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 833.352 | 713.276 | 23.237 / 5.0 | 7.839 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 1075.852 | 1022.869 | 52.931 / 5.0 | 39.870 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 1650.158 | 1642.127 | 5.756 / 5.0 | 3.198 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 4033.485 | 3957.045 | 4.192 / 15.0 | 1.746 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 925.001 | 929.718 | 0.723 / 20.0 | 1.345 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 806.628 | 803.744 | 0.505 / 5.0 | 5.791 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 1105.533 | 894.111 | 27.579 / 5.0 | 30.075 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 1587.598 | 1565.024 | 2.213 / 5.0 | 1.762 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 806.628 | 803.744 | 0.505 / 5.0 | 5.791 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 1105.533 | 894.111 | 27.579 / 5.0 | 30.075 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 1587.598 | 1565.024 | 2.213 / 5.0 | 1.762 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 3585.410 | 3633.040 | 2.848 / 15.0 | 2.129 / 25.0 | unresolved |
| scenario-compact/screen | double | cold_ms | 993.356 | 920.491 | 20.033 / 20.0 | 10.159 / 30.0 | unresolved |
| scenario-compact/screen | double | early_median_ms | 905.918 | 864.771 | 22.032 / 5.0 | 18.928 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p90_ms | 1275.098 | 1401.737 | 47.667 / 5.0 | 13.623 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p99_ms | 1764.992 | 1751.064 | 1.753 / 5.0 | 2.818 / 10.0 | unresolved |
| scenario-compact/screen | double | median_ms | 905.918 | 864.771 | 22.032 / 5.0 | 18.928 / 10.0 | unresolved |
| scenario-compact/screen | double | p90_ms | 1275.098 | 1401.737 | 47.667 / 5.0 | 13.623 / 10.0 | unresolved |
| scenario-compact/screen | double | p99_ms | 1764.992 | 1751.064 | 1.753 / 5.0 | 2.818 / 10.0 | unresolved |
| scenario-compact/screen | double | setup_ms | 4264.335 | 4250.570 | 0.728 / 15.0 | 2.279 / 25.0 | unresolved |
| scenario-compact/screen | float | cold_ms | 1010.795 | 901.005 | 19.004 / 20.0 | 5.708 / 30.0 | unresolved |
| scenario-compact/screen | float | early_median_ms | 881.649 | 876.524 | 8.546 / 5.0 | 4.793 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p90_ms | 1230.736 | 1252.674 | 11.663 / 5.0 | 44.117 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p99_ms | 1675.452 | 1649.653 | 6.596 / 5.0 | 4.735 / 10.0 | unresolved |
| scenario-compact/screen | float | median_ms | 881.649 | 876.524 | 8.546 / 5.0 | 4.793 / 10.0 | unresolved |
| scenario-compact/screen | float | p90_ms | 1230.736 | 1252.674 | 11.663 / 5.0 | 44.117 / 10.0 | unresolved |
| scenario-compact/screen | float | p99_ms | 1675.452 | 1649.653 | 6.596 / 5.0 | 4.735 / 10.0 | unresolved |
| scenario-compact/screen | float | setup_ms | 4000.985 | 4068.355 | 3.914 / 15.0 | 1.687 / 25.0 | unresolved |
| scenario-larger/calibration | double | cold_ms | 2843.145 | 2822.435 | 20.534 / 20.0 | 4.331 / 30.0 | unresolved |
| scenario-larger/calibration | double | early_median_ms | 2701.529 | 2681.033 | 2.389 / 5.0 | 1.436 / 10.0 | unresolved |
| scenario-larger/calibration | double | early_p90_ms | 3499.581 | 3473.213 | 7.573 / 5.0 | 4.297 / 10.0 | unresolved |
| scenario-larger/calibration | double | early_p99_ms | 3701.640 | 3619.379 | 7.201 / 5.0 | 6.371 / 10.0 | unresolved |
| scenario-larger/calibration | double | median_ms | 2701.529 | 2681.033 | 2.389 / 5.0 | 1.436 / 10.0 | unresolved |
| scenario-larger/calibration | double | p90_ms | 3499.581 | 3473.213 | 7.573 / 5.0 | 4.297 / 10.0 | unresolved |
| scenario-larger/calibration | double | p99_ms | 3701.640 | 3619.379 | 7.201 / 5.0 | 6.371 / 10.0 | unresolved |
| scenario-larger/calibration | double | setup_ms | 12060.950 | 12076.350 | 5.932 / 15.0 | 1.987 / 25.0 | unresolved |
| scenario-larger/calibration | float | cold_ms | 1766.020 | 1751.195 | 32.034 / 20.0 | 2.973 / 30.0 | unresolved |
| scenario-larger/calibration | float | early_median_ms | 1633.698 | 1629.379 | 4.475 / 5.0 | 1.874 / 10.0 | unresolved |
| scenario-larger/calibration | float | early_p90_ms | 2260.657 | 2275.666 | 20.976 / 5.0 | 4.155 / 10.0 | unresolved |
| scenario-larger/calibration | float | early_p99_ms | 2512.889 | 2590.662 | 13.939 / 5.0 | 6.086 / 10.0 | unresolved |
| scenario-larger/calibration | float | median_ms | 1633.698 | 1629.379 | 4.475 / 5.0 | 1.874 / 10.0 | unresolved |
| scenario-larger/calibration | float | p90_ms | 2260.657 | 2275.666 | 20.976 / 5.0 | 4.155 / 10.0 | unresolved |
| scenario-larger/calibration | float | p99_ms | 2512.889 | 2590.662 | 13.939 / 5.0 | 6.086 / 10.0 | unresolved |
| scenario-larger/calibration | float | setup_ms | 7774.515 | 7820.950 | 2.333 / 15.0 | 8.359 / 25.0 | unresolved |
| scenario-stress/heldout | double | cold_ms | 32586.950 | 32572.800 | 2.119 / 20.0 | 1.417 / 30.0 | [-2.098, 1.434] |
| scenario-stress/heldout | double | early_median_ms | 32039.511 | 32017.271 | 0.356 / 5.0 | 0.208 / 10.0 | [-0.221, 0.278] |
| scenario-stress/heldout | double | early_p90_ms | 32457.892 | 32425.946 | 0.679 / 5.0 | 0.345 / 10.0 | [-0.262, 0.632] |
| scenario-stress/heldout | double | early_p99_ms | 32710.154 | 32606.229 | 0.761 / 5.0 | 0.564 / 10.0 | [-0.375, 0.750] |
| scenario-stress/heldout | double | median_ms | 32039.511 | 32017.271 | 0.356 / 5.0 | 0.208 / 10.0 | [-0.221, 0.278] |
| scenario-stress/heldout | double | p90_ms | 32457.892 | 32425.946 | 0.679 / 5.0 | 0.345 / 10.0 | [-0.262, 0.632] |
| scenario-stress/heldout | double | p99_ms | 32710.154 | 32606.229 | 0.761 / 5.0 | 0.564 / 10.0 | [-0.375, 0.750] |
| scenario-stress/heldout | double | setup_ms | 117644.000 | 117767.000 | 0.821 / 15.0 | 0.381 / 25.0 | [-0.702, 0.568] |
| scenario-stress/heldout | float | cold_ms | 17018.050 | 17024.600 | 2.046 / 20.0 | 0.763 / 30.0 | [-1.344, 1.377] |
| scenario-stress/heldout | float | early_median_ms | 16790.354 | 16785.764 | 0.477 / 5.0 | 0.442 / 10.0 | [-0.468, 0.388] |
| scenario-stress/heldout | float | early_p90_ms | 16926.191 | 16916.924 | 0.732 / 5.0 | 0.435 / 10.0 | [-0.493, 0.592] |
| scenario-stress/heldout | float | early_p99_ms | 17060.846 | 17079.283 | 1.638 / 5.0 | 0.942 / 10.0 | [-1.319, 0.734] |
| scenario-stress/heldout | float | median_ms | 16790.354 | 16785.764 | 0.477 / 5.0 | 0.442 / 10.0 | [-0.468, 0.388] |
| scenario-stress/heldout | float | p90_ms | 16926.191 | 16916.924 | 0.732 / 5.0 | 0.435 / 10.0 | [-0.493, 0.592] |
| scenario-stress/heldout | float | p99_ms | 17060.846 | 17079.283 | 1.638 / 5.0 | 0.942 / 10.0 | [-1.319, 0.734] |
| scenario-stress/heldout | float | setup_ms | 61549.350 | 61588.200 | 2.010 / 15.0 | 1.500 / 25.0 | [-1.631, 1.131] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 89.890 | 20.472 | 70.113 | 2587.907 | 1174.095 | 3 | 31.887 | 0.267 |
| baseline-small/calibration | double | 249.180 | 20.646 | 238.600 | 2060.054 | 1250.340 | 12 | 32.820 | 0.286 |
| baseline-medium/screen | float | 10.620 | 25.891 | 33.247 | 1763.336 | 1143.620 | 0 | 12.164 | 0.181 |
| baseline-medium/screen | double | 232.827 | 18.177 | 235.819 | 1794.576 | 1105.165 | 0 | 1.919 | 0.212 |
| baseline-large/screen | float | 4.188 | 20.371 | 54.888 | 1744.788 | 1163.525 | 0 | 2.970 | 0.154 |
| baseline-large/screen | double | 191.027 | 20.929 | 167.403 | 1750.085 | 1115.245 | 0 | 3.140 | 0.241 |
| scenario-compact/screen | float | 75.795 | 20.677 | 92.410 | 1810.585 | 1092.150 | 0 | 26.177 | 0.291 |
| scenario-compact/screen | double | 218.864 | 24.090 | 177.280 | 1935.834 | 1064.340 | 0 | 9.006 | 0.275 |
| scenario-larger/calibration | float | 73.007 | 75.231 | 69.966 | 3006.673 | 563.219 | 0 | 3.680 | 0.376 |
| scenario-larger/calibration | double | 65.214 | 126.641 | 55.371 | 4514.324 | 346.093 | 0 | 3.288 | 0.480 |
| scenario-stress/heldout | float | 79.775 | 110.045 | 99.788 | 19700.883 | 59.510 | 0 | 1.567 | 0.765 |
| scenario-stress/heldout | double | 114.233 | 503.119 | 101.525 | 33435.375 | 31.236 | 0 | 1.372 | 0.874 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/clutter_filter.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |
| scenario-compact | dispersion, insufficient_interval_resolution |
| scenario-larger | dispersion, insufficient_interval_resolution |
| scenario-stress | qualified |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/clutter_filter/`; native evidence: `native/clutter_filter/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
