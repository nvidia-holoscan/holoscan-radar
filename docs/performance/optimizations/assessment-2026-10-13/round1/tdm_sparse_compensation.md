# tdm_sparse_compensation: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_budget_unresolved | screen | 64.329 | 64.199 |
| baseline-medium | repeat_passed | qualified | heldout | 389.312 | 679.750 |
| baseline-large | repeat_passed | calibration_budget_unresolved | screen | 2737.027 | 5078.388 |

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
  "waveform": {
    "bandwidth": 1000000000,
    "carrier_frequency": 77000000000,
    "chirp_duration": 4e-05,
    "factory": "valid_fmcw_waveform",
    "pri_seconds": 0.001,
    "sample_rate": 20000000
  }
}
```

Output wiring: `{"drain":["is_valid","array_geometry"],"drain_codecs":{"array_geometry":"array_geometry"},"optional_tensor_outputs":["is_valid"],"timed":"snapshot","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | double | cold_ms | 337.890 | 331.034 | 3.100 / 20.0 | 1.365 / 30.0 | unresolved |
| baseline-small/screen | double | early_median_ms | 64.199 | 64.283 | 0.814 / 5.0 | 0.643 / 10.0 | unresolved |
| baseline-small/screen | double | early_p90_ms | 66.334 | 69.716 | 8.496 / 5.0 | 3.631 / 10.0 | unresolved |
| baseline-small/screen | double | early_p99_ms | 72.097 | 78.493 | 13.551 / 5.0 | 7.831 / 10.0 | unresolved |
| baseline-small/screen | double | median_ms | 64.199 | 64.283 | 0.814 / 5.0 | 0.643 / 10.0 | unresolved |
| baseline-small/screen | double | p90_ms | 66.334 | 69.716 | 8.496 / 5.0 | 3.631 / 10.0 | unresolved |
| baseline-small/screen | double | p99_ms | 72.097 | 78.493 | 13.551 / 5.0 | 7.831 / 10.0 | unresolved |
| baseline-small/screen | double | setup_ms | 5372.880 | 5305.400 | 3.312 / 15.0 | 1.700 / 25.0 | unresolved |
| baseline-small/screen | float | cold_ms | 324.597 | 325.909 | 1.384 / 20.0 | 0.765 / 30.0 | unresolved |
| baseline-small/screen | float | early_median_ms | 64.329 | 64.095 | 1.992 / 5.0 | 1.316 / 10.0 | unresolved |
| baseline-small/screen | float | early_p90_ms | 66.397 | 70.579 | 8.988 / 5.0 | 4.554 / 10.0 | unresolved |
| baseline-small/screen | float | early_p99_ms | 84.546 | 78.946 | 8.603 / 5.0 | 5.505 / 10.0 | unresolved |
| baseline-small/screen | float | median_ms | 64.329 | 64.095 | 1.992 / 5.0 | 1.316 / 10.0 | unresolved |
| baseline-small/screen | float | p90_ms | 66.397 | 70.579 | 8.988 / 5.0 | 4.554 / 10.0 | unresolved |
| baseline-small/screen | float | p99_ms | 84.546 | 78.946 | 8.603 / 5.0 | 5.505 / 10.0 | unresolved |
| baseline-small/screen | float | setup_ms | 4864.735 | 4791.260 | 2.057 / 15.0 | 1.174 / 25.0 | unresolved |
| baseline-medium/heldout | double | cold_ms | 939.264 | 937.879 | 3.665 / 20.0 | 1.146 / 30.0 | [-1.620, 1.405] |
| baseline-medium/heldout | double | early_median_ms | 679.750 | 680.734 | 0.952 / 5.0 | 0.591 / 10.0 | [-0.718, 0.906] |
| baseline-medium/heldout | double | early_p90_ms | 731.457 | 729.879 | 1.666 / 5.0 | 1.082 / 10.0 | [-0.970, 1.410] |
| baseline-medium/heldout | double | early_p99_ms | 754.922 | 752.977 | 1.985 / 5.0 | 1.460 / 10.0 | [-0.993, 1.721] |
| baseline-medium/heldout | double | median_ms | 679.750 | 680.734 | 0.952 / 5.0 | 0.591 / 10.0 | [-0.718, 0.906] |
| baseline-medium/heldout | double | p90_ms | 731.457 | 729.879 | 1.666 / 5.0 | 1.082 / 10.0 | [-0.970, 1.410] |
| baseline-medium/heldout | double | p99_ms | 754.922 | 752.977 | 1.985 / 5.0 | 1.460 / 10.0 | [-0.993, 1.721] |
| baseline-medium/heldout | double | setup_ms | 12386.350 | 12378.500 | 6.908 / 15.0 | 2.750 / 25.0 | [-3.794, 6.092] |
| baseline-medium/heldout | float | cold_ms | 685.133 | 684.865 | 1.689 / 20.0 | 0.930 / 30.0 | [-1.524, 0.988] |
| baseline-medium/heldout | float | early_median_ms | 389.312 | 388.370 | 2.174 / 5.0 | 0.758 / 10.0 | [-0.771, 1.311] |
| baseline-medium/heldout | float | early_p90_ms | 411.695 | 410.907 | 3.337 / 5.0 | 1.745 / 10.0 | [-2.608, 1.271] |
| baseline-medium/heldout | float | early_p99_ms | 448.499 | 450.194 | 2.579 / 5.0 | 1.865 / 10.0 | [-2.292, 2.208] |
| baseline-medium/heldout | float | median_ms | 389.312 | 388.370 | 2.174 / 5.0 | 0.758 / 10.0 | [-0.771, 1.311] |
| baseline-medium/heldout | float | p90_ms | 411.695 | 410.907 | 3.337 / 5.0 | 1.745 / 10.0 | [-2.608, 1.271] |
| baseline-medium/heldout | float | p99_ms | 448.499 | 450.194 | 2.579 / 5.0 | 1.865 / 10.0 | [-2.292, 2.208] |
| baseline-medium/heldout | float | setup_ms | 8482.375 | 8402.280 | 9.213 / 15.0 | 7.507 / 25.0 | [-7.747, 7.982] |
| baseline-large/screen | double | cold_ms | 5332.555 | 5273.960 | 2.113 / 20.0 | 1.042 / 30.0 | unresolved |
| baseline-large/screen | double | early_median_ms | 5078.388 | 5078.071 | 0.574 / 5.0 | 0.421 / 10.0 | unresolved |
| baseline-large/screen | double | early_p90_ms | 5211.242 | 5198.675 | 0.470 / 5.0 | 0.157 / 10.0 | unresolved |
| baseline-large/screen | double | early_p99_ms | 5259.700 | 5260.869 | 0.105 / 5.0 | 0.073 / 10.0 | unresolved |
| baseline-large/screen | double | median_ms | 5078.388 | 5078.071 | 0.574 / 5.0 | 0.421 / 10.0 | unresolved |
| baseline-large/screen | double | p90_ms | 5211.242 | 5198.675 | 0.470 / 5.0 | 0.157 / 10.0 | unresolved |
| baseline-large/screen | double | p99_ms | 5259.700 | 5260.869 | 0.105 / 5.0 | 0.073 / 10.0 | unresolved |
| baseline-large/screen | double | setup_ms | 54201.000 | 54379.300 | 1.381 / 15.0 | 1.008 / 25.0 | unresolved |
| baseline-large/screen | float | cold_ms | 3072.415 | 3079.595 | 0.574 / 20.0 | 3.428 / 30.0 | unresolved |
| baseline-large/screen | float | early_median_ms | 2737.027 | 2699.405 | 2.368 / 5.0 | 0.746 / 10.0 | unresolved |
| baseline-large/screen | float | early_p90_ms | 2832.031 | 2823.277 | 1.103 / 5.0 | 0.562 / 10.0 | unresolved |
| baseline-large/screen | float | early_p99_ms | 2886.843 | 2848.872 | 1.627 / 5.0 | 1.126 / 10.0 | unresolved |
| baseline-large/screen | float | median_ms | 2737.027 | 2699.405 | 2.368 / 5.0 | 0.746 / 10.0 | unresolved |
| baseline-large/screen | float | p90_ms | 2832.031 | 2823.277 | 1.103 / 5.0 | 0.562 / 10.0 | unresolved |
| baseline-large/screen | float | p99_ms | 2886.843 | 2848.872 | 1.627 / 5.0 | 1.126 / 10.0 | unresolved |
| baseline-large/screen | float | setup_ms | 29131.700 | 29567.200 | 2.007 / 15.0 | 0.906 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/screen | float | 1.296 | 1.617 | 1.294 | 119.417 | 14622.450 | 0 | 3.841 | 0.605 |
| baseline-small/screen | double | 0.521 | 1.473 | 0.920 | 100.519 | 14627.550 | 0 | 2.297 | 0.506 |
| baseline-medium/heldout | float | 8.581 | 10.062 | 4.580 | 529.428 | 2521.030 | 0 | 2.841 | 0.676 |
| baseline-medium/heldout | double | 6.487 | 23.206 | 5.779 | 1568.377 | 1442.460 | 0 | 2.202 | 0.578 |
| baseline-large/screen | float | 65.517 | 123.250 | 53.077 | 3395.827 | 365.683 | 0 | 7.750 | 0.735 |
| baseline-large/screen | double | 29.238 | 182.591 | 45.359 | 5749.791 | 197.020 | 0 | 2.144 | 0.620 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/tdm_sparse_compensation.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | qualified |
| baseline-large | insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/tdm_sparse_compensation/`; native evidence: `native/tdm_sparse_compensation/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
