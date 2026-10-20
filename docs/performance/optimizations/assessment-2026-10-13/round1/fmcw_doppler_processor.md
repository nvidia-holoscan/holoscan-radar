# fmcw_doppler_processor: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 32.144 | 41.380 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 32.551 | 63.954 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 63.679 | 148.019 |
| scenario-compact | repeat_passed | calibration_budget_unresolved | screen | 90.426 | 272.760 |
| scenario-larger | repeat_passed | calibration_budget_unresolved | screen | 1485.739 | 5046.410 |
| scenario-stress | repeat_passed | qualified | heldout | 27483.005 | 93822.676 |

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
  "center_zero_doppler": true,
  "doppler_window": {
    "type": "none"
  },
  "fft_norm": "backward",
  "nfft": 0,
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

Output wiring: `{"drain":["velocity_bins"],"optional_tensor_outputs":["velocity_bins"],"timed":"rd_maps","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 268.242 | 268.286 | 0.200 / 20.0 | 0.176 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 41.380 | 34.797 | 24.911 / 5.0 | 10.384 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 50.948 | 51.187 | 1.488 / 5.0 | 0.865 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 55.436 | 53.746 | 6.494 / 5.0 | 2.803 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 41.380 | 34.797 | 24.911 / 5.0 | 10.384 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 50.948 | 51.187 | 1.488 / 5.0 | 0.865 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 55.436 | 53.746 | 6.494 / 5.0 | 2.803 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 3985.765 | 3938.760 | 2.857 / 15.0 | 2.041 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 265.038 | 269.593 | 2.401 / 20.0 | 1.192 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 32.144 | 32.451 | 1.622 / 5.0 | 0.662 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 33.600 | 34.504 | 3.387 / 5.0 | 2.232 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 48.465 | 53.981 | 24.766 / 5.0 | 33.096 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 32.144 | 32.451 | 1.622 / 5.0 | 0.662 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 33.600 | 34.504 | 3.387 / 5.0 | 2.232 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 48.465 | 53.981 | 24.766 / 5.0 | 33.096 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 3976.670 | 3939.545 | 2.312 / 15.0 | 1.966 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 30089.300 | 29609.200 | 2.663 / 20.0 | 0.921 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 63.954 | 64.148 | 1.175 / 5.0 | 0.773 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 65.666 | 65.744 | 0.580 / 5.0 | 1.153 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 72.902 | 75.214 | 13.413 / 5.0 | 8.617 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 63.954 | 64.148 | 1.175 / 5.0 | 0.773 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 65.666 | 65.744 | 0.580 / 5.0 | 1.153 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 72.902 | 75.214 | 13.413 / 5.0 | 8.617 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 4075.200 | 4093.920 | 0.582 / 15.0 | 2.374 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 27098.650 | 26212.900 | 3.557 / 20.0 | 2.840 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 32.551 | 32.864 | 1.380 / 5.0 | 1.336 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 34.636 | 45.329 | 51.832 / 5.0 | 16.405 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 56.978 | 60.564 | 14.222 / 5.0 | 17.269 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 32.551 | 32.864 | 1.380 / 5.0 | 1.336 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 34.636 | 45.329 | 51.832 / 5.0 | 16.405 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 56.978 | 60.564 | 14.222 / 5.0 | 17.269 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 3865.115 | 3832.855 | 1.607 / 15.0 | 1.534 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 35694.300 | 35958.650 | 2.262 / 20.0 | 1.082 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 148.019 | 148.252 | 0.312 / 5.0 | 0.107 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 149.420 | 149.950 | 0.576 / 5.0 | 0.217 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 166.263 | 161.623 | 4.342 / 5.0 | 1.826 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 148.019 | 148.252 | 0.312 / 5.0 | 0.107 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 149.420 | 149.950 | 0.576 / 5.0 | 0.217 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 166.263 | 161.623 | 4.342 / 5.0 | 1.826 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 4747.335 | 5080.550 | 12.484 / 15.0 | 3.517 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 39204.250 | 39399.450 | 0.793 / 20.0 | 0.555 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 63.679 | 63.913 | 0.528 / 5.0 | 0.266 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 65.315 | 65.488 | 0.577 / 5.0 | 0.231 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 72.061 | 74.370 | 15.062 / 5.0 | 9.838 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 63.679 | 63.913 | 0.528 / 5.0 | 0.266 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 65.315 | 65.488 | 0.577 / 5.0 | 0.231 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 72.061 | 74.370 | 15.062 / 5.0 | 9.838 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 4291.400 | 4255.850 | 3.515 / 15.0 | 2.414 / 25.0 | unresolved |
| scenario-compact/screen | double | cold_ms | 515.987 | 518.620 | 0.696 / 20.0 | 0.394 / 30.0 | unresolved |
| scenario-compact/screen | double | early_median_ms | 272.760 | 273.361 | 0.850 / 5.0 | 0.452 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p90_ms | 278.107 | 281.050 | 2.163 / 5.0 | 0.752 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p99_ms | 294.680 | 294.976 | 1.626 / 5.0 | 0.951 / 10.0 | unresolved |
| scenario-compact/screen | double | median_ms | 272.760 | 273.361 | 0.850 / 5.0 | 0.452 / 10.0 | unresolved |
| scenario-compact/screen | double | p90_ms | 278.107 | 281.050 | 2.163 / 5.0 | 0.752 / 10.0 | unresolved |
| scenario-compact/screen | double | p99_ms | 294.680 | 294.976 | 1.626 / 5.0 | 0.951 / 10.0 | unresolved |
| scenario-compact/screen | double | setup_ms | 5058.090 | 5041.535 | 0.468 / 15.0 | 0.335 / 25.0 | unresolved |
| scenario-compact/screen | float | cold_ms | 335.167 | 337.611 | 1.577 / 20.0 | 0.619 / 30.0 | unresolved |
| scenario-compact/screen | float | early_median_ms | 90.426 | 90.900 | 1.044 / 5.0 | 0.359 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p90_ms | 98.833 | 99.131 | 3.217 / 5.0 | 2.420 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p99_ms | 124.634 | 104.684 | 17.004 / 5.0 | 15.394 / 10.0 | unresolved |
| scenario-compact/screen | float | median_ms | 90.426 | 90.900 | 1.044 / 5.0 | 0.359 / 10.0 | unresolved |
| scenario-compact/screen | float | p90_ms | 98.833 | 99.131 | 3.217 / 5.0 | 2.420 / 10.0 | unresolved |
| scenario-compact/screen | float | p99_ms | 124.634 | 104.684 | 17.004 / 5.0 | 15.394 / 10.0 | unresolved |
| scenario-compact/screen | float | setup_ms | 4764.130 | 4755.495 | 0.412 / 15.0 | 0.153 / 25.0 | unresolved |
| scenario-larger/screen | double | cold_ms | 54431.400 | 54483.800 | 0.647 / 20.0 | 0.337 / 30.0 | unresolved |
| scenario-larger/screen | double | early_median_ms | 5046.410 | 5082.547 | 1.040 / 5.0 | 0.653 / 10.0 | unresolved |
| scenario-larger/screen | double | early_p90_ms | 6116.817 | 5883.508 | 4.423 / 5.0 | 3.458 / 10.0 | unresolved |
| scenario-larger/screen | double | early_p99_ms | 6319.167 | 6403.846 | 2.114 / 5.0 | 2.039 / 10.0 | unresolved |
| scenario-larger/screen | double | median_ms | 5046.410 | 5082.547 | 1.040 / 5.0 | 0.653 / 10.0 | unresolved |
| scenario-larger/screen | double | p90_ms | 6116.817 | 5883.508 | 4.423 / 5.0 | 3.458 / 10.0 | unresolved |
| scenario-larger/screen | double | p99_ms | 6319.167 | 6403.846 | 2.114 / 5.0 | 2.039 / 10.0 | unresolved |
| scenario-larger/screen | double | setup_ms | 12873.150 | 13347.750 | 4.694 / 15.0 | 2.460 / 25.0 | unresolved |
| scenario-larger/screen | float | cold_ms | 42009.750 | 42747.700 | 3.563 / 20.0 | 1.233 / 30.0 | unresolved |
| scenario-larger/screen | float | early_median_ms | 1485.739 | 1504.565 | 1.511 / 5.0 | 3.542 / 10.0 | unresolved |
| scenario-larger/screen | float | early_p90_ms | 2127.143 | 2130.958 | 3.762 / 5.0 | 5.198 / 10.0 | unresolved |
| scenario-larger/screen | float | early_p99_ms | 2509.101 | 2405.812 | 8.309 / 5.0 | 3.031 / 10.0 | unresolved |
| scenario-larger/screen | float | median_ms | 1485.739 | 1504.565 | 1.511 / 5.0 | 3.542 / 10.0 | unresolved |
| scenario-larger/screen | float | p90_ms | 2127.143 | 2130.958 | 3.762 / 5.0 | 5.198 / 10.0 | unresolved |
| scenario-larger/screen | float | p99_ms | 2509.101 | 2405.812 | 8.309 / 5.0 | 3.031 / 10.0 | unresolved |
| scenario-larger/screen | float | setup_ms | 9084.995 | 8816.410 | 6.001 / 15.0 | 3.698 / 25.0 | unresolved |
| scenario-stress/heldout | double | cold_ms | 126784.000 | 126884.500 | 0.948 / 20.0 | 0.566 / 30.0 | [-0.749, 0.879] |
| scenario-stress/heldout | double | early_median_ms | 93822.676 | 93828.548 | 0.390 / 5.0 | 0.253 / 10.0 | [-0.389, 0.225] |
| scenario-stress/heldout | double | early_p90_ms | 95865.112 | 95826.471 | 0.442 / 5.0 | 0.196 / 10.0 | [-0.172, 0.340] |
| scenario-stress/heldout | double | early_p99_ms | 96335.904 | 96236.905 | 0.777 / 5.0 | 0.239 / 10.0 | [-0.224, 0.697] |
| scenario-stress/heldout | double | median_ms | 93822.676 | 93828.548 | 0.390 / 5.0 | 0.253 / 10.0 | [-0.389, 0.225] |
| scenario-stress/heldout | double | p90_ms | 95865.112 | 95826.471 | 0.442 / 5.0 | 0.196 / 10.0 | [-0.172, 0.340] |
| scenario-stress/heldout | double | p99_ms | 96335.904 | 96236.905 | 0.777 / 5.0 | 0.239 / 10.0 | [-0.224, 0.697] |
| scenario-stress/heldout | double | setup_ms | 118650.000 | 118678.500 | 1.029 / 15.0 | 0.521 / 25.0 | [-0.935, 0.585] |
| scenario-stress/heldout | float | cold_ms | 57945.600 | 58172.900 | 3.848 / 20.0 | 2.881 / 30.0 | [-2.561, 2.586] |
| scenario-stress/heldout | float | early_median_ms | 27483.005 | 27420.587 | 2.269 / 5.0 | 1.120 / 10.0 | [-1.291, 2.204] |
| scenario-stress/heldout | float | early_p90_ms | 29524.188 | 29521.050 | 1.794 / 5.0 | 1.011 / 10.0 | [-0.843, 0.789] |
| scenario-stress/heldout | float | early_p99_ms | 30079.494 | 30125.455 | 1.839 / 5.0 | 0.881 / 10.0 | [-1.561, 1.445] |
| scenario-stress/heldout | float | median_ms | 27483.005 | 27420.587 | 2.269 / 5.0 | 1.120 / 10.0 | [-1.291, 2.204] |
| scenario-stress/heldout | float | p90_ms | 29524.188 | 29521.050 | 1.794 / 5.0 | 1.011 / 10.0 | [-0.843, 0.789] |
| scenario-stress/heldout | float | p99_ms | 30079.494 | 30125.455 | 1.839 / 5.0 | 0.881 / 10.0 | [-1.561, 1.445] |
| scenario-stress/heldout | float | setup_ms | 62581.250 | 62327.050 | 1.638 / 15.0 | 0.848 / 25.0 | [-0.793, 1.553] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.516 | 0.985 | 0.660 | 66.250 | 25937.300 | 0 | 3.278 | 0.736 |
| baseline-small/screen | double | 12.012 | 16.523 | 4.731 | 81.047 | 21844.450 | 0 | 46.862 | 0.440 |
| baseline-medium/screen | float | 0.444 | 0.931 | 0.493 | 78.305 | 25374.900 | 0 | 5.968 | 0.625 |
| baseline-medium/screen | double | 0.748 | 0.968 | 0.829 | 87.740 | 14315.600 | 0 | 1.580 | 0.515 |
| baseline-large/screen | float | 0.336 | 1.638 | 0.437 | 89.565 | 14497.600 | 0 | 2.985 | 0.459 |
| baseline-large/screen | double | 0.462 | 1.059 | 0.583 | 801.667 | 6494.605 | 1 | 5.359 | 0.343 |
| scenario-compact/screen | float | 0.942 | 1.312 | 0.558 | 654.482 | 10206.100 | 3 | 1.808 | 0.622 |
| scenario-compact/screen | double | 2.306 | 3.138 | 2.782 | 610.242 | 3576.095 | 0 | 0.901 | 0.552 |
| scenario-larger/screen | float | 22.825 | 91.900 | 76.173 | 2654.141 | 612.358 | 0 | 2.437 | 0.256 |
| scenario-larger/screen | double | 52.620 | 642.929 | 68.566 | 6620.757 | 189.219 | 0 | 4.464 | 0.480 |
| scenario-stress/heldout | float | 623.861 | 2494.906 | 617.057 | 32508.344 | 36.493 | 0 | 9.413 | 0.631 |
| scenario-stress/heldout | double | 366.675 | 2349.543 | 398.609 | 99679.809 | 10.630 | 0 | 1.655 | 0.608 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/fmcw_doppler_processor.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

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

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/fmcw_doppler_processor/`; native evidence: `native/fmcw_doppler_processor/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
