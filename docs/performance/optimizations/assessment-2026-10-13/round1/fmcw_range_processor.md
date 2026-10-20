# fmcw_range_processor: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 26.335 | 45.116 |
| baseline-medium | repeat_passed | calibration_budget_unresolved | screen | 26.933 | 61.715 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 27.199 | 156.695 |
| scenario-compact | repeat_passed | calibration_budget_unresolved | screen | 36.255 | 225.795 |
| scenario-larger | repeat_passed | qualified | heldout | 620.859 | 4008.608 |
| scenario-stress | repeat_passed | qualified | heldout | 9592.707 | 67437.527 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | chirps, samples | [16,128] | {"channels":1} | {} |
| baseline-medium | medium | chirps, samples | [64,512] | {"channels":1} | {} |
| baseline-large | large | chirps, samples | [128,1024] | {"channels":1} | {} |
| scenario-compact | small | chirps, samples | [128,512] | {"channels":4} | {"waveform":{"bandwidth":150000000,"carrier_frequency":77000000000,"chirp_duration":4e-05,"factory":"valid_fmcw_waveform","pri_seconds":4.5e-05,"sample_rate":12800000}} |
| scenario-larger | medium | chirps, samples | [256,1024] | {"channels":16} | {"waveform":{"bandwidth":150000000,"carrier_frequency":77000000000,"chirp_duration":8e-05,"factory":"valid_fmcw_waveform","pri_seconds":8.5e-05,"sample_rate":12800000}} |
| scenario-stress | large | chirps, samples | [512,2048] | {"channels":64} | {"waveform":{"bandwidth":150000000,"carrier_frequency":77000000000,"chirp_duration":0.00016,"factory":"valid_fmcw_waveform","pri_seconds":0.00016500000000000003,"sample_rate":12800000}} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "apply_dechirp": false,
  "fft_norm": "backward",
  "nfft": 0,
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

Output wiring: `{"drain":["range_bins"],"optional_tensor_outputs":["range_bins"],"timed":"range_profiles","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 268.534 | 268.472 | 0.152 / 20.0 | 0.890 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 45.116 | 45.262 | 0.677 / 5.0 | 0.299 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 46.006 | 46.905 | 2.731 / 5.0 | 1.462 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 50.425 | 54.394 | 15.283 / 5.0 | 4.894 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 45.116 | 45.262 | 0.677 / 5.0 | 0.299 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 46.006 | 46.905 | 2.731 / 5.0 | 1.462 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 50.425 | 54.394 | 15.283 / 5.0 | 4.894 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 4035.050 | 3895.240 | 4.126 / 15.0 | 3.098 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 260.518 | 258.681 | 2.272 / 20.0 | 1.150 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 26.335 | 26.405 | 0.308 / 5.0 | 0.499 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 26.823 | 27.253 | 1.673 / 5.0 | 1.365 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 30.028 | 27.667 | 14.766 / 5.0 | 5.602 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 26.335 | 26.405 | 0.308 / 5.0 | 0.499 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 26.823 | 27.253 | 1.673 / 5.0 | 1.365 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 30.028 | 27.667 | 14.766 / 5.0 | 5.602 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 3908.120 | 3994.390 | 5.278 / 15.0 | 1.710 / 25.0 | unresolved |
| baseline-medium/screen | double | cold_ms | 27298.750 | 27500.500 | 1.179 / 20.0 | 1.463 / 30.0 | unresolved |
| baseline-medium/screen | double | early_median_ms | 61.715 | 61.470 | 0.528 / 5.0 | 0.304 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p90_ms | 64.749 | 63.382 | 4.365 / 5.0 | 1.720 / 10.0 | unresolved |
| baseline-medium/screen | double | early_p99_ms | 74.535 | 73.587 | 2.803 / 5.0 | 1.088 / 10.0 | unresolved |
| baseline-medium/screen | double | median_ms | 61.715 | 61.470 | 0.528 / 5.0 | 0.304 / 10.0 | unresolved |
| baseline-medium/screen | double | p90_ms | 64.749 | 63.382 | 4.365 / 5.0 | 1.720 / 10.0 | unresolved |
| baseline-medium/screen | double | p99_ms | 74.535 | 73.587 | 2.803 / 5.0 | 1.088 / 10.0 | unresolved |
| baseline-medium/screen | double | setup_ms | 4008.915 | 4026.960 | 2.312 / 15.0 | 1.495 / 25.0 | unresolved |
| baseline-medium/screen | float | cold_ms | 23280.250 | 23366.750 | 0.373 / 20.0 | 0.330 / 30.0 | unresolved |
| baseline-medium/screen | float | early_median_ms | 26.933 | 26.905 | 0.458 / 5.0 | 0.241 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p90_ms | 27.809 | 27.763 | 0.645 / 5.0 | 0.536 / 10.0 | unresolved |
| baseline-medium/screen | float | early_p99_ms | 48.419 | 37.432 | 26.744 / 5.0 | 19.068 / 10.0 | unresolved |
| baseline-medium/screen | float | median_ms | 26.933 | 26.905 | 0.458 / 5.0 | 0.241 / 10.0 | unresolved |
| baseline-medium/screen | float | p90_ms | 27.809 | 27.763 | 0.645 / 5.0 | 0.536 / 10.0 | unresolved |
| baseline-medium/screen | float | p99_ms | 48.419 | 37.432 | 26.744 / 5.0 | 19.068 / 10.0 | unresolved |
| baseline-medium/screen | float | setup_ms | 3835.265 | 3928.745 | 2.732 / 15.0 | 2.186 / 25.0 | unresolved |
| baseline-large/screen | double | cold_ms | 53503.900 | 53836.100 | 2.331 / 20.0 | 1.506 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 156.695 | 157.312 | 0.806 / 5.0 | 0.445 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 158.300 | 158.914 | 1.278 / 5.0 | 0.621 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 168.636 | 165.437 | 3.517 / 5.0 | 1.103 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 156.695 | 157.312 | 0.806 / 5.0 | 0.445 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 158.300 | 158.914 | 1.278 / 5.0 | 0.621 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 168.636 | 165.437 | 3.517 / 5.0 | 1.103 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 4751.960 | 4796.465 | 1.096 / 15.0 | 0.793 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 43278.250 | 42878.800 | 1.114 / 20.0 | 0.654 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 27.199 | 27.148 | 0.244 / 5.0 | 0.178 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 28.093 | 28.324 | 0.931 / 5.0 | 0.980 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 35.406 | 41.761 | 39.704 / 5.0 | 7.894 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 27.199 | 27.148 | 0.244 / 5.0 | 0.178 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 28.093 | 28.324 | 0.931 / 5.0 | 0.980 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 35.406 | 41.761 | 39.704 / 5.0 | 7.894 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 4221.920 | 4169.615 | 1.495 / 15.0 | 0.658 / 25.0 | unresolved |
| scenario-compact/screen | double | cold_ms | 470.683 | 475.734 | 1.554 / 20.0 | 0.571 / 30.0 | unresolved |
| scenario-compact/screen | double | early_median_ms | 225.795 | 225.376 | 0.190 / 5.0 | 0.107 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p90_ms | 228.926 | 227.426 | 0.930 / 5.0 | 0.484 / 10.0 | unresolved |
| scenario-compact/screen | double | early_p99_ms | 236.040 | 232.095 | 2.127 / 5.0 | 1.376 / 10.0 | unresolved |
| scenario-compact/screen | double | median_ms | 225.795 | 225.376 | 0.190 / 5.0 | 0.107 / 10.0 | unresolved |
| scenario-compact/screen | double | p90_ms | 228.926 | 227.426 | 0.930 / 5.0 | 0.484 / 10.0 | unresolved |
| scenario-compact/screen | double | p99_ms | 236.040 | 232.095 | 2.127 / 5.0 | 1.376 / 10.0 | unresolved |
| scenario-compact/screen | double | setup_ms | 5067.730 | 5184.910 | 4.393 / 15.0 | 1.327 / 25.0 | unresolved |
| scenario-compact/screen | float | cold_ms | 280.449 | 285.673 | 2.685 / 20.0 | 1.285 / 30.0 | unresolved |
| scenario-compact/screen | float | early_median_ms | 36.255 | 36.319 | 48.347 / 5.0 | 39.510 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p90_ms | 45.581 | 45.999 | 0.981 / 5.0 | 0.715 / 10.0 | unresolved |
| scenario-compact/screen | float | early_p99_ms | 51.029 | 48.747 | 9.691 / 5.0 | 5.553 / 10.0 | unresolved |
| scenario-compact/screen | float | median_ms | 36.255 | 36.319 | 48.347 / 5.0 | 39.510 / 10.0 | unresolved |
| scenario-compact/screen | float | p90_ms | 45.581 | 45.999 | 0.981 / 5.0 | 0.715 / 10.0 | unresolved |
| scenario-compact/screen | float | p99_ms | 51.029 | 48.747 | 9.691 / 5.0 | 5.553 / 10.0 | unresolved |
| scenario-compact/screen | float | setup_ms | 4641.390 | 5187.380 | 19.889 / 15.0 | 6.702 / 25.0 | unresolved |
| scenario-larger/heldout | double | cold_ms | 57207.250 | 57088.250 | 4.883 / 20.0 | 1.528 / 30.0 | [-0.710, 4.840] |
| scenario-larger/heldout | double | early_median_ms | 4008.608 | 4008.782 | 0.214 / 5.0 | 0.082 / 10.0 | [-0.211, 0.083] |
| scenario-larger/heldout | double | early_p90_ms | 4016.141 | 4020.057 | 0.975 / 5.0 | 0.504 / 10.0 | [-0.853, 0.550] |
| scenario-larger/heldout | double | early_p99_ms | 4052.793 | 4047.404 | 1.454 / 5.0 | 1.057 / 10.0 | [-0.777, 0.828] |
| scenario-larger/heldout | double | median_ms | 4008.608 | 4008.782 | 0.214 / 5.0 | 0.082 / 10.0 | [-0.211, 0.083] |
| scenario-larger/heldout | double | p90_ms | 4016.141 | 4020.057 | 0.975 / 5.0 | 0.504 / 10.0 | [-0.853, 0.550] |
| scenario-larger/heldout | double | p99_ms | 4052.793 | 4047.404 | 1.454 / 5.0 | 1.057 / 10.0 | [-0.777, 0.828] |
| scenario-larger/heldout | double | setup_ms | 13147.100 | 13311.100 | 6.552 / 15.0 | 4.218 / 25.0 | [-6.376, 3.073] |
| scenario-larger/heldout | float | cold_ms | 43656.600 | 43693.600 | 3.529 / 20.0 | 2.050 / 30.0 | [-3.225, 3.036] |
| scenario-larger/heldout | float | early_median_ms | 620.859 | 620.885 | 1.715 / 5.0 | 0.671 / 10.0 | [-1.155, 1.143] |
| scenario-larger/heldout | float | early_p90_ms | 666.689 | 667.287 | 1.750 / 5.0 | 0.910 / 10.0 | [-1.629, 1.513] |
| scenario-larger/heldout | float | early_p99_ms | 689.861 | 690.462 | 2.068 / 5.0 | 1.020 / 10.0 | [-1.691, 1.520] |
| scenario-larger/heldout | float | median_ms | 620.859 | 620.885 | 1.715 / 5.0 | 0.671 / 10.0 | [-1.155, 1.143] |
| scenario-larger/heldout | float | p90_ms | 666.689 | 667.287 | 1.750 / 5.0 | 0.910 / 10.0 | [-1.629, 1.513] |
| scenario-larger/heldout | float | p99_ms | 689.861 | 690.462 | 2.068 / 5.0 | 1.020 / 10.0 | [-1.691, 1.520] |
| scenario-larger/heldout | float | setup_ms | 9101.610 | 8797.915 | 10.111 / 15.0 | 8.488 / 25.0 | [-7.670, 9.352] |
| scenario-stress/heldout | double | cold_ms | 115343.500 | 115562.000 | 0.407 / 20.0 | 0.547 / 30.0 | [-0.356, 0.235] |
| scenario-stress/heldout | double | early_median_ms | 67437.527 | 67448.551 | 0.128 / 5.0 | 0.047 / 10.0 | [-0.115, 0.072] |
| scenario-stress/heldout | double | early_p90_ms | 67616.132 | 67659.349 | 0.224 / 5.0 | 0.173 / 10.0 | [-0.174, 0.196] |
| scenario-stress/heldout | double | early_p99_ms | 67752.545 | 67766.718 | 0.196 / 5.0 | 0.128 / 10.0 | [-0.165, 0.178] |
| scenario-stress/heldout | double | median_ms | 67437.527 | 67448.551 | 0.128 / 5.0 | 0.047 / 10.0 | [-0.115, 0.072] |
| scenario-stress/heldout | double | p90_ms | 67616.132 | 67659.349 | 0.224 / 5.0 | 0.173 / 10.0 | [-0.174, 0.196] |
| scenario-stress/heldout | double | p99_ms | 67752.545 | 67766.718 | 0.196 / 5.0 | 0.128 / 10.0 | [-0.165, 0.178] |
| scenario-stress/heldout | double | setup_ms | 118491.500 | 118683.000 | 0.868 / 15.0 | 0.643 / 25.0 | [-0.811, 0.697] |
| scenario-stress/heldout | float | cold_ms | 51848.750 | 51897.000 | 4.004 / 20.0 | 1.782 / 30.0 | [-2.703, 1.787] |
| scenario-stress/heldout | float | early_median_ms | 9592.707 | 9593.595 | 0.344 / 5.0 | 0.238 / 10.0 | [-0.269, 0.181] |
| scenario-stress/heldout | float | early_p90_ms | 9705.120 | 9709.911 | 0.493 / 5.0 | 0.293 / 10.0 | [-0.391, 0.353] |
| scenario-stress/heldout | float | early_p99_ms | 9781.843 | 9772.116 | 1.624 / 5.0 | 0.683 / 10.0 | [-0.702, 1.609] |
| scenario-stress/heldout | float | median_ms | 9592.707 | 9593.595 | 0.344 / 5.0 | 0.238 / 10.0 | [-0.269, 0.181] |
| scenario-stress/heldout | float | p90_ms | 9705.120 | 9709.911 | 0.493 / 5.0 | 0.293 / 10.0 | [-0.391, 0.353] |
| scenario-stress/heldout | float | p99_ms | 9781.843 | 9772.116 | 1.624 / 5.0 | 0.683 / 10.0 | [-0.702, 1.609] |
| scenario-stress/heldout | float | setup_ms | 62423.150 | 62311.900 | 1.443 / 15.0 | 0.928 / 25.0 | [-0.692, 1.305] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 0.081 | 0.635 | 0.218 | 64.259 | 33890.700 | 0 | 2.541 | 0.408 |
| baseline-small/screen | double | 0.304 | 1.059 | 0.466 | 76.787 | 20594.650 | 0 | 2.747 | 0.274 |
| baseline-medium/screen | float | 0.124 | 0.685 | 0.231 | 67.148 | 32815.150 | 0 | 2.175 | 0.516 |
| baseline-medium/screen | double | 0.326 | 1.175 | 0.458 | 82.111 | 15218.700 | 0 | 1.947 | 0.479 |
| baseline-large/screen | float | 0.066 | 0.716 | 0.227 | 53.251 | 32516.050 | 0 | 1.937 | 0.250 |
| baseline-large/screen | double | 1.256 | 0.963 | 1.318 | 1317.399 | 6218.975 | 1 | 0.481 | 0.226 |
| scenario-compact/screen | float | 14.482 | 16.548 | 14.527 | 61.483 | 25281.550 | 0 | 54.110 | 0.232 |
| scenario-compact/screen | double | 0.428 | 1.703 | 1.086 | 245.703 | 4367.025 | 0 | 0.601 | 0.615 |
| scenario-larger/heldout | float | 10.637 | 30.246 | 8.558 | 1348.131 | 1577.795 | 0 | 3.371 | 0.551 |
| scenario-larger/heldout | double | 8.567 | 7.094 | 5.792 | 4774.169 | 249.072 | 0 | 1.072 | 0.704 |
| scenario-stress/heldout | float | 32.954 | 139.901 | 31.516 | 10574.844 | 104.233 | 0 | 2.113 | 0.893 |
| scenario-stress/heldout | double | 85.965 | 144.942 | 49.568 | 70219.838 | 14.820 | 0 | 0.519 | 0.857 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/fmcw_range_processor.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |
| scenario-compact | dispersion, insufficient_interval_resolution |
| scenario-larger | qualified |
| scenario-stress | qualified |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/fmcw_range_processor/`; native evidence: `native/fmcw_range_processor/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
