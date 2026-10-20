# tdm_dense_compensation: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 36.419 | 34.537 |
| baseline-medium | repeat_passed | heldout_budget_unresolved | calibration | 582.919 | 1313.752 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 9327.592 | 21008.721 |

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
  "waveform": {
    "bandwidth": 150000000,
    "carrier_frequency": 77000000000,
    "chirp_duration": 2e-05,
    "factory": "valid_fmcw_waveform",
    "pri_seconds": 2.5e-05,
    "sample_rate": 12800000
  }
}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":[],"timed":"rd_maps_out","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 265.620 | 265.000 | 0.629 / 20.0 | 0.326 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 34.537 | 34.396 | 0.518 / 5.0 | 1.320 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 35.872 | 36.767 | 5.927 / 5.0 | 2.738 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 36.758 | 41.885 | 25.609 / 5.0 | 10.102 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 34.537 | 34.396 | 0.518 / 5.0 | 1.320 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 35.872 | 36.767 | 5.927 / 5.0 | 2.738 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 36.758 | 41.885 | 25.609 / 5.0 | 10.102 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 3078.570 | 3042.175 | 2.809 / 15.0 | 7.706 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 263.899 | 262.676 | 0.544 / 20.0 | 2.183 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 36.419 | 32.623 | 16.950 / 5.0 | 5.909 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 37.409 | 33.401 | 17.402 / 5.0 | 6.073 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 38.614 | 34.135 | 18.794 / 5.0 | 6.652 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 36.419 | 32.623 | 16.950 / 5.0 | 5.909 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 37.409 | 33.401 | 17.402 / 5.0 | 6.073 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 38.614 | 34.135 | 18.794 / 5.0 | 6.652 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 2797.695 | 2877.920 | 6.343 / 15.0 | 2.084 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 1525.255 | 1527.645 | 0.965 / 20.0 | 0.436 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 1313.752 | 1314.613 | 0.723 / 5.0 | 0.411 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 1364.793 | 1369.778 | 1.228 / 5.0 | 0.615 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 1388.325 | 1396.469 | 3.019 / 5.0 | 1.264 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 1313.752 | 1314.613 | 0.723 / 5.0 | 0.411 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 1364.793 | 1369.778 | 1.228 / 5.0 | 0.615 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 1388.325 | 1396.469 | 3.019 / 5.0 | 1.264 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 12035.150 | 11978.400 | 6.837 / 15.0 | 3.339 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 789.257 | 789.984 | 1.690 / 20.0 | 0.395 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 582.919 | 584.879 | 0.817 / 5.0 | 0.647 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 632.770 | 631.499 | 2.949 / 5.0 | 2.087 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 665.781 | 665.704 | 1.734 / 5.0 | 0.781 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 582.919 | 584.879 | 0.817 / 5.0 | 0.647 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 632.770 | 631.499 | 2.949 / 5.0 | 2.087 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 665.781 | 665.704 | 1.734 / 5.0 | 0.781 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 8127.370 | 8216.890 | 11.154 / 15.0 | 5.596 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 20993.800 | 21022.450 | 0.186 / 20.0 | 0.106 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 21008.721 | 20993.692 | 0.330 / 5.0 | 0.185 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 21130.290 | 21138.671 | 0.240 / 5.0 | 0.188 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 21196.629 | 21240.677 | 0.247 / 5.0 | 0.117 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 21008.721 | 20993.692 | 0.330 / 5.0 | 0.185 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 21130.290 | 21138.671 | 0.240 / 5.0 | 0.188 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 21196.629 | 21240.677 | 0.247 / 5.0 | 0.117 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 116571.500 | 116535.500 | 0.379 / 15.0 | 0.546 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 9581.900 | 9580.895 | 0.206 / 20.0 | 0.115 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 9327.592 | 9316.835 | 0.216 / 5.0 | 0.066 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 9458.610 | 9501.107 | 1.080 / 5.0 | 0.494 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 9521.856 | 9566.958 | 1.197 / 5.0 | 0.464 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 9327.592 | 9316.835 | 0.216 / 5.0 | 0.066 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 9458.610 | 9501.107 | 1.080 / 5.0 | 0.494 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 9521.856 | 9566.958 | 1.197 / 5.0 | 0.464 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 60660.100 | 61635.150 | 2.019 / 15.0 | 1.029 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 6.759 | 0.784 | 0.575 | 43.852 | 27906.500 | 0 | 2.438 | 0.685 |
| baseline-small/screen | double | 0.180 | 0.910 | 0.612 | 131.992 | 26649.150 | 1 | 2.691 | 0.587 |
| baseline-medium/calibration | float | 4.770 | 22.849 | 6.918 | 754.734 | 1672.620 | 0 | 2.009 | 0.537 |
| baseline-medium/calibration | double | 9.520 | 30.660 | 7.292 | 3475.595 | 752.287 | 0 | 1.433 | 0.636 |
| baseline-large/screen | float | 20.177 | 148.159 | 33.575 | 9869.655 | 107.365 | 0 | 1.673 | 0.914 |
| baseline-large/screen | double | 69.458 | 156.792 | 40.823 | 21633.071 | 47.585 | 0 | 0.433 | 0.704 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/tdm_dense_compensation.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | insufficient_interval_resolution |
| baseline-large | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/tdm_dense_compensation/`; native evidence: `native/tdm_dense_compensation/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
