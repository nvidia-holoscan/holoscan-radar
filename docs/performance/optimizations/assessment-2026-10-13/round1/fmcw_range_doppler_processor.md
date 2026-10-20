# fmcw_range_doppler_processor: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 50.016 | 63.706 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 50.139 | 98.542 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 74.062 | 275.702 |
| scenario-compact | repeat_passed | calibration_budget_unresolved | screen | 103.412 | 465.986 |
| scenario-larger | repeat_passed | calibration_blocked | calibration | 2123.501 | 8601.294 |
| scenario-stress | repeat_passed | calibration_budget_unresolved | screen | 38842.303 | 156228.816 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | chirps, samples | [16,128] | {"channels":1} | {"doppler_window":{"type":"none"},"range_window":{"type":"none"}} |
| baseline-medium | medium | chirps, samples | [64,512] | {"channels":1} | {"doppler_window":{"type":"none"},"range_window":{"type":"none"}} |
| baseline-large | large | chirps, samples | [128,1024] | {"channels":1} | {"doppler_window":{"type":"none"},"range_window":{"type":"none"}} |
| scenario-compact | small | chirps, samples | [128,512] | {"channels":4} | {"waveform":{"bandwidth":150000000,"carrier_frequency":77000000000,"chirp_duration":4e-05,"factory":"valid_fmcw_waveform","pri_seconds":4.5e-05,"sample_rate":12800000}} |
| scenario-larger | medium | chirps, samples | [256,1024] | {"channels":16} | {"waveform":{"bandwidth":150000000,"carrier_frequency":77000000000,"chirp_duration":8e-05,"factory":"valid_fmcw_waveform","pri_seconds":8.5e-05,"sample_rate":12800000}} |
| scenario-stress | large | chirps, samples | [512,2048] | {"channels":64} | {"waveform":{"bandwidth":150000000,"carrier_frequency":77000000000,"chirp_duration":0.00016,"factory":"valid_fmcw_waveform","pri_seconds":0.00016500000000000003,"sample_rate":12800000}} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "apply_dechirp": false,
  "center_zero_doppler": true,
  "clutter_filter": "none",
  "doppler_window": {
    "type": "none"
  },
  "fft_norm": "backward",
  "nfft_doppler": 0,
  "nfft_range": 0,
  "range_window": {
    "type": "none"
  },
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

Output wiring: `{"drain":["range_bins","velocity_bins"],"optional_tensor_outputs":["range_bins","velocity_bins"],"timed":"rd_maps","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 287.065 | 291.005 | 2.973 / 20.0 | 1.431 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 63.706 | 53.982 | 17.394 / 5.0 | 15.006 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 68.736 | 70.135 | 2.050 / 5.0 | 1.378 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 72.319 | 76.722 | 9.312 / 5.0 | 4.478 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 63.706 | 53.982 | 17.394 / 5.0 | 15.006 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 68.736 | 70.135 | 2.050 / 5.0 | 1.378 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 72.319 | 76.722 | 9.312 / 5.0 | 4.478 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 4810.620 | 4748.975 | 2.237 / 15.0 | 1.367 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 272.115 | 277.682 | 2.416 / 20.0 | 1.799 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 50.016 | 49.993 | 0.195 / 5.0 | 0.328 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 52.214 | 51.436 | 1.726 / 5.0 | 1.153 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 64.739 | 63.796 | 8.868 / 5.0 | 20.471 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 50.016 | 49.993 | 0.195 / 5.0 | 0.328 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 52.214 | 51.436 | 1.726 / 5.0 | 1.153 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 64.739 | 63.796 | 8.868 / 5.0 | 20.471 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 4803.450 | 4845.325 | 0.952 / 15.0 | 0.562 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 54648.150 | 55156.750 | 1.304 / 20.0 | 0.633 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 98.542 | 99.167 | 0.981 / 5.0 | 0.385 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 102.216 | 101.864 | 1.920 / 5.0 | 1.071 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 112.014 | 116.681 | 11.962 / 5.0 | 5.822 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 98.542 | 99.167 | 0.981 / 5.0 | 0.385 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 102.216 | 101.864 | 1.920 / 5.0 | 1.071 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 112.014 | 116.681 | 11.962 / 5.0 | 5.822 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 4915.410 | 4950.425 | 0.923 / 15.0 | 1.220 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 47289.100 | 48347.800 | 2.866 / 20.0 | 1.657 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 50.139 | 50.701 | 1.935 / 5.0 | 0.617 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 51.377 | 53.042 | 3.499 / 5.0 | 2.881 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 72.273 | 59.590 | 22.776 / 5.0 | 13.650 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 50.139 | 50.701 | 1.935 / 5.0 | 0.617 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 51.377 | 53.042 | 3.499 / 5.0 | 2.881 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 72.273 | 59.590 | 22.776 / 5.0 | 13.650 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 4898.170 | 4811.455 | 3.303 / 15.0 | 1.032 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 87876.500 | 87829.700 | 0.341 / 20.0 | 5.421 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 275.702 | 276.019 | 0.175 / 5.0 | 0.138 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 280.211 | 282.409 | 1.416 / 5.0 | 0.394 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 293.443 | 291.433 | 2.486 / 5.0 | 1.397 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 275.702 | 276.019 | 0.175 / 5.0 | 0.138 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 280.211 | 282.409 | 1.416 / 5.0 | 0.394 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 293.443 | 291.433 | 2.486 / 5.0 | 1.397 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 5528.670 | 5503.215 | 2.651 / 15.0 | 2.083 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 79613.700 | 80051.550 | 0.900 / 20.0 | 0.331 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 74.062 | 74.144 | 0.542 / 5.0 | 0.399 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 77.117 | 77.935 | 6.189 / 5.0 | 4.666 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 88.471 | 87.306 | 11.962 / 5.0 | 9.375 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 74.062 | 74.144 | 0.542 / 5.0 | 0.399 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 77.117 | 77.935 | 6.189 / 5.0 | 4.666 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 88.471 | 87.306 | 11.962 / 5.0 | 9.375 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 5134.315 | 5074.375 | 1.785 / 15.0 | 0.909 / 25.0 | unresolved |
| scenario-compact/screen | double | cold_ms | 714.658 | 712.038 | 0.928 / 20.0 | 0.610 / 30.0 | unresolved |
| scenario-compact/screen | double | early_median_ms | 465.986 | 467.645 | 0.483 / 5.0 | 0.277 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p90_ms | 475.933 | 473.077 | 1.140 / 5.0 | 0.361 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p99_ms | 492.623 | 497.112 | 0.983 / 5.0 | 0.837 / 10.0 | unresolved |
| scenario-compact/screen | double | median_ms | 465.986 | 467.645 | 0.483 / 5.0 | 0.277 / 10.0 | unresolved |
| scenario-compact/screen | double | p90_ms | 475.933 | 473.077 | 1.140 / 5.0 | 0.361 / 10.0 | unresolved |
| scenario-compact/screen | double | p99_ms | 492.623 | 497.112 | 0.983 / 5.0 | 0.837 / 10.0 | unresolved |
| scenario-compact/screen | double | setup_ms | 6319.815 | 5895.090 | 11.784 / 15.0 | 4.138 / 25.0 | unresolved |
| scenario-compact/screen | float | cold_ms | 352.847 | 352.653 | 0.167 / 20.0 | 0.206 / 30.0 | unresolved |
| scenario-compact/screen | float | early_median_ms | 103.412 | 103.949 | 0.768 / 5.0 | 0.365 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p90_ms | 108.383 | 111.128 | 3.994 / 5.0 | 2.946 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p99_ms | 119.640 | 129.757 | 13.917 / 5.0 | 6.826 / 10.0 | unresolved |
| scenario-compact/screen | float | median_ms | 103.412 | 103.949 | 0.768 / 5.0 | 0.365 / 10.0 | unresolved |
| scenario-compact/screen | float | p90_ms | 108.383 | 111.128 | 3.994 / 5.0 | 2.946 / 10.0 | unresolved |
| scenario-compact/screen | float | p99_ms | 119.640 | 129.757 | 13.917 / 5.0 | 6.826 / 10.0 | unresolved |
| scenario-compact/screen | float | setup_ms | 5570.830 | 5707.245 | 3.359 / 15.0 | 1.635 / 25.0 | unresolved |
| scenario-larger/calibration | double | cold_ms | 103966.000 | 104471.500 | 4.190 / 20.0 | 1.633 / 30.0 | unresolved |
| scenario-larger/calibration | double | early_median_ms | 8601.294 | 8590.351 | 1.744 / 5.0 | 1.544 / 10.0 | unresolved |
| scenario-larger/calibration | double | early_p90_ms | 9561.088 | 9443.246 | 4.651 / 5.0 | 2.084 / 10.0 | unresolved |
| scenario-larger/calibration | double | early_p99_ms | 10158.130 | 10050.559 | 4.884 / 5.0 | 3.023 / 10.0 | unresolved |
| scenario-larger/calibration | double | median_ms | 8601.294 | 8590.351 | 1.744 / 5.0 | 1.544 / 10.0 | unresolved |
| scenario-larger/calibration | double | p90_ms | 9561.088 | 9443.246 | 4.651 / 5.0 | 2.084 / 10.0 | unresolved |
| scenario-larger/calibration | double | p99_ms | 10158.130 | 10050.559 | 4.884 / 5.0 | 3.023 / 10.0 | unresolved |
| scenario-larger/calibration | double | setup_ms | 13833.300 | 13959.050 | 3.969 / 15.0 | 3.676 / 25.0 | unresolved |
| scenario-larger/calibration | float | cold_ms | 82845.050 | 82936.300 | 3.362 / 20.0 | 0.843 / 30.0 | unresolved |
| scenario-larger/calibration | float | early_median_ms | 2123.501 | 2093.625 | 5.456 / 5.0 | 2.769 / 10.0 | unresolved |
| scenario-larger/calibration | float | early_p90_ms | 2992.544 | 2952.970 | 5.034 / 5.0 | 3.117 / 10.0 | unresolved |
| scenario-larger/calibration | float | early_p99_ms | 3271.084 | 3214.177 | 8.012 / 5.0 | 3.663 / 10.0 | unresolved |
| scenario-larger/calibration | float | median_ms | 2123.501 | 2093.625 | 5.456 / 5.0 | 2.769 / 10.0 | unresolved |
| scenario-larger/calibration | float | p90_ms | 2992.544 | 2952.970 | 5.034 / 5.0 | 3.117 / 10.0 | unresolved |
| scenario-larger/calibration | float | p99_ms | 3271.084 | 3214.177 | 8.012 / 5.0 | 3.663 / 10.0 | unresolved |
| scenario-larger/calibration | float | setup_ms | 9719.710 | 9943.770 | 8.873 / 15.0 | 6.986 / 25.0 | unresolved |
| scenario-stress/screen | double | cold_ms | 205673.000 | 206183.500 | 0.465 / 20.0 | 0.141 / 30.0 | unresolved |
| scenario-stress/screen | double | early_median_ms | 156228.816 | 156192.163 | 0.080 / 5.0 | 0.053 / 10.0 | unresolved |
| scenario-stress/screen | double | early_p90_ms | 157824.051 | 157585.258 | 0.211 / 5.0 | 0.114 / 10.0 | unresolved |
| scenario-stress/screen | double | early_p99_ms | 159225.570 | 159254.767 | 0.352 / 5.0 | 0.315 / 10.0 | unresolved |
| scenario-stress/screen | double | median_ms | 156228.816 | 156192.163 | 0.080 / 5.0 | 0.053 / 10.0 | unresolved |
| scenario-stress/screen | double | p90_ms | 157824.051 | 157585.258 | 0.211 / 5.0 | 0.114 / 10.0 | unresolved |
| scenario-stress/screen | double | p99_ms | 159225.570 | 159254.767 | 0.352 / 5.0 | 0.315 / 10.0 | unresolved |
| scenario-stress/screen | double | setup_ms | 119033.500 | 118612.000 | 0.673 / 15.0 | 0.216 / 25.0 | unresolved |
| scenario-stress/screen | float | cold_ms | 107278.000 | 110193.500 | 2.869 / 20.0 | 1.891 / 30.0 | unresolved |
| scenario-stress/screen | float | early_median_ms | 38842.303 | 38572.349 | 1.414 / 5.0 | 0.520 / 10.0 | unresolved |
| scenario-stress/screen | float | early_p90_ms | 40294.612 | 40063.114 | 1.303 / 5.0 | 0.565 / 10.0 | unresolved |
| scenario-stress/screen | float | early_p99_ms | 41402.487 | 41519.898 | 0.857 / 5.0 | 0.324 / 10.0 | unresolved |
| scenario-stress/screen | float | median_ms | 38842.303 | 38572.349 | 1.414 / 5.0 | 0.520 / 10.0 | unresolved |
| scenario-stress/screen | float | p90_ms | 40294.612 | 40063.114 | 1.303 / 5.0 | 0.565 / 10.0 | unresolved |
| scenario-stress/screen | float | p99_ms | 41402.487 | 41519.898 | 0.857 / 5.0 | 0.324 / 10.0 | unresolved |
| scenario-stress/screen | float | setup_ms | 62931.800 | 62921.550 | 0.270 / 15.0 | 0.223 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.097 | 1.050 | 0.484 | 86.158 | 17837.250 | 0 | 1.446 | 0.483 |
| baseline-small/screen | double | 11.360 | 16.174 | 12.690 | 87.732 | 15480.300 | 0 | 23.575 | 0.577 |
| baseline-medium/screen | float | 0.964 | 1.117 | 0.616 | 105.055 | 17755.250 | 0 | 2.374 | 0.528 |
| baseline-medium/screen | double | 0.962 | 1.643 | 1.137 | 129.629 | 9502.870 | 0 | 1.773 | 0.605 |
| baseline-large/screen | float | 0.401 | 1.438 | 0.627 | 114.621 | 12461.000 | 0 | 2.090 | 0.554 |
| baseline-large/screen | double | 0.482 | 1.524 | 0.653 | 337.028 | 3543.830 | 0 | 0.535 | 0.556 |
| scenario-compact/screen | float | 0.793 | 1.403 | 0.566 | 140.777 | 9004.300 | 0 | 1.203 | 0.569 |
| scenario-compact/screen | double | 2.246 | 2.638 | 4.236 | 513.139 | 2108.355 | 0 | 0.264 | 0.617 |
| scenario-larger/calibration | float | 121.049 | 404.940 | 118.044 | 4351.000 | 436.733 | 0 | 13.540 | 0.365 |
| scenario-larger/calibration | double | 151.857 | 304.790 | 164.680 | 10815.773 | 113.315 | 0 | 4.207 | 0.369 |
| scenario-stress/screen | float | 551.972 | 1775.230 | 487.871 | 42787.497 | 25.934 | 0 | 2.182 | 0.384 |
| scenario-stress/screen | double | 124.770 | 1460.379 | 183.832 | 163246.660 | 6.396 | 0 | 0.310 | 0.246 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/fmcw_range_doppler_processor.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |
| scenario-compact | dispersion, insufficient_interval_resolution |
| scenario-larger | dispersion, insufficient_interval_resolution |
| scenario-stress | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/fmcw_range_doppler_processor/`; native evidence: `native/fmcw_range_doppler_processor/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
