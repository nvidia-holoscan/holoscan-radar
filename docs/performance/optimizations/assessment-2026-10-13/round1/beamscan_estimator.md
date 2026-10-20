# beamscan_estimator: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 71.674 | 169.475 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 126.032 | 1307.684 |
| baseline-large | repeat_passed | qualified | heldout | 2746.221 | 63168.935 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | channels, items, looks, az, el | [4,64,1,21,11] | {} | {} |
| baseline-medium | medium | channels, items, looks, az, el | [8,256,3,41,21] | {} | {} |
| baseline-large | large | channels, items, looks, az, el | [16,1024,9,61,31] | {} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "array_geometry": {
    "factory": "ula_geometry",
    "spacing_wavelengths": 0.5,
    "wavelength": 1.0
  },
  "array_taper": {
    "type": "none"
  },
  "az_coordinates": {
    "count": "az",
    "factory": "linspace",
    "start": -0.5,
    "stop": 0.5
  },
  "el_coordinates": {
    "count": "el",
    "factory": "linspace",
    "start": -0.3,
    "stop": 0.3
  },
  "peak_interpolation": "none"
}
```

Output wiring: `{"drain":["angle_spectrum"],"optional_tensor_outputs":["angle_spectrum"],"queue_depth":8,"timed":"aoa_estimates","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 1573.315 | 1623.555 | 38.733 / 20.0 | 5.353 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 169.475 | 169.434 | 0.638 / 5.0 | 0.328 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 171.733 | 172.201 | 3.547 / 5.0 | 0.553 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 183.818 | 184.957 | 6.377 / 5.0 | 3.312 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 169.475 | 169.434 | 0.638 / 5.0 | 0.328 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 171.733 | 172.201 | 3.547 / 5.0 | 0.553 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 183.818 | 184.957 | 6.377 / 5.0 | 3.312 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 4208.535 | 4211.940 | 2.000 / 15.0 | 0.696 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 1494.245 | 1500.905 | 29.788 / 20.0 | 5.012 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 71.674 | 71.456 | 2.428 / 5.0 | 0.543 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 73.497 | 73.890 | 5.517 / 5.0 | 4.072 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 86.953 | 83.136 | 17.065 / 5.0 | 17.309 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 71.674 | 71.456 | 2.428 / 5.0 | 0.543 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 73.497 | 73.890 | 5.517 / 5.0 | 4.072 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 86.953 | 83.136 | 17.065 / 5.0 | 17.309 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 4209.550 | 4201.560 | 3.040 / 15.0 | 1.107 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 2198.160 | 2196.015 | 6.826 / 20.0 | 1.700 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 1307.684 | 1307.465 | 0.049 / 5.0 | 0.019 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 1309.876 | 1310.048 | 0.408 / 5.0 | 0.088 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 1318.667 | 1316.641 | 0.520 / 5.0 | 0.376 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 1307.684 | 1307.465 | 0.049 / 5.0 | 0.019 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 1309.876 | 1310.048 | 0.408 / 5.0 | 0.088 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 1318.667 | 1316.641 | 0.520 / 5.0 | 0.376 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 5446.440 | 5427.560 | 11.132 / 15.0 | 3.837 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 1020.995 | 1022.780 | 9.464 / 20.0 | 2.724 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 126.032 | 126.019 | 0.492 / 5.0 | 0.204 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 127.647 | 128.133 | 1.121 / 5.0 | 0.904 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 135.855 | 135.443 | 5.920 / 5.0 | 2.397 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 126.032 | 126.019 | 0.492 / 5.0 | 0.204 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 127.647 | 128.133 | 1.121 / 5.0 | 0.904 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 135.855 | 135.443 | 5.920 / 5.0 | 2.397 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 4811.905 | 4788.490 | 6.601 / 15.0 | 1.835 / 25.0 | unresolved |
| baseline-large/heldout | double | cold_ms | 64284.100 | 64292.900 | 0.392 / 20.0 | 0.085 / 30.0 | [-0.199, 0.101] |
| baseline-large/heldout | double | early_median_ms | 63168.935 | 63167.797 | 0.015 / 5.0 | 0.008 / 10.0 | [-0.011, 0.012] |
| baseline-large/heldout | double | early_p90_ms | 63183.397 | 63183.284 | 0.038 / 5.0 | 0.018 / 10.0 | [-0.036, 0.025] |
| baseline-large/heldout | double | early_p99_ms | 63217.908 | 63216.887 | 0.089 / 5.0 | 0.049 / 10.0 | [-0.061, 0.055] |
| baseline-large/heldout | double | median_ms | 63168.935 | 63167.797 | 0.015 / 5.0 | 0.008 / 10.0 | [-0.011, 0.012] |
| baseline-large/heldout | double | p90_ms | 63183.397 | 63183.284 | 0.038 / 5.0 | 0.018 / 10.0 | [-0.036, 0.025] |
| baseline-large/heldout | double | p99_ms | 63217.908 | 63216.887 | 0.089 / 5.0 | 0.049 / 10.0 | [-0.061, 0.055] |
| baseline-large/heldout | double | setup_ms | 9612.890 | 9789.340 | 10.120 / 15.0 | 3.967 / 25.0 | [-10.089, 3.367] |
| baseline-large/heldout | float | cold_ms | 3824.055 | 3820.905 | 10.517 / 20.0 | 1.826 / 30.0 | [-4.444, 2.407] |
| baseline-large/heldout | float | early_median_ms | 2746.221 | 2746.474 | 0.071 / 5.0 | 0.025 / 10.0 | [-0.051, 0.020] |
| baseline-large/heldout | float | early_p90_ms | 2750.738 | 2750.930 | 0.213 / 5.0 | 0.064 / 10.0 | [-0.124, 0.088] |
| baseline-large/heldout | float | early_p99_ms | 2757.180 | 2761.060 | 0.515 / 5.0 | 0.242 / 10.0 | [-0.309, 0.234] |
| baseline-large/heldout | float | median_ms | 2746.221 | 2746.474 | 0.071 / 5.0 | 0.025 / 10.0 | [-0.051, 0.020] |
| baseline-large/heldout | float | p90_ms | 2750.738 | 2750.930 | 0.213 / 5.0 | 0.064 / 10.0 | [-0.124, 0.088] |
| baseline-large/heldout | float | p99_ms | 2757.180 | 2761.060 | 0.515 / 5.0 | 0.242 / 10.0 | [-0.309, 0.234] |
| baseline-large/heldout | float | setup_ms | 7691.980 | 7520.625 | 9.253 / 15.0 | 6.580 / 25.0 | [-8.253, 8.340] |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 1.754 | 1.487 | 1.035 | 134.287 | 13279.550 | 0 | 2.449 | 0.569 |
| baseline-small/calibration | double | 1.086 | 1.839 | 0.871 | 396.816 | 5766.270 | 0 | 1.141 | 0.587 |
| baseline-medium/calibration | float | 0.620 | 1.268 | 0.445 | 1055.163 | 7722.805 | 7 | 3.254 | 0.573 |
| baseline-medium/calibration | double | 0.646 | 1.662 | 0.464 | 2592.887 | 762.611 | 0 | 0.110 | 0.585 |
| baseline-large/heldout | float | 1.942 | 4.325 | 1.125 | 3363.396 | 363.636 | 0 | 0.770 | 0.710 |
| baseline-large/heldout | double | 9.349 | 7.216 | 7.119 | 64942.394 | 15.829 | 0 | 0.063 | 0.708 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/beamscan_estimator.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | qualified |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/beamscan_estimator/`; native evidence: `native/beamscan_estimator/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
