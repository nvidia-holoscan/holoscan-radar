# phase_shift_beamformer: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 55.650 | 149.892 |
| baseline-medium | repeat_passed | calibration_blocked | calibration | 58.310 | 150.308 |
| baseline-large | repeat_passed | calibration_blocked | calibration | 152.788 | 2427.543 |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | channels, az, el, doppler, range | [4,17,1,16,32] | {} | {} |
| baseline-medium | medium | channels, az, el, doppler, range | [8,31,1,32,64] | {} | {} |
| baseline-large | large | channels, az, el, doppler, range | [16,41,11,32,128] | {} | {} |

Recipe defaults for the last listed scope (original defaults retained by overlays; per-case overrides are shown above):

```json
{
  "angle_convention": {
    "preset": "radar_boresight"
  },
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
  }
}
```

Output wiring: `{"drain":[],"optional_tensor_outputs":["az_grid","el_grid"],"queue_depth":8,"timed":"rda_cube","unwired":["az_grid","el_grid"]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 1612.225 | 1610.155 | 30.811 / 20.0 | 3.129 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 149.892 | 149.660 | 0.416 / 5.0 | 0.268 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 152.280 | 152.829 | 1.983 / 5.0 | 0.932 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 163.149 | 164.054 | 5.143 / 5.0 | 1.872 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 149.892 | 149.660 | 0.416 / 5.0 | 0.268 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 152.280 | 152.829 | 1.983 / 5.0 | 0.932 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 163.149 | 164.054 | 5.143 / 5.0 | 1.872 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 3857.910 | 3871.275 | 3.256 / 15.0 | 2.953 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 1505.670 | 1512.800 | 44.512 / 20.0 | 3.491 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 55.650 | 55.530 | 0.844 / 5.0 | 0.669 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 57.196 | 57.870 | 8.889 / 5.0 | 5.424 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 63.904 | 66.738 | 16.407 / 5.0 | 14.607 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 55.650 | 55.530 | 0.844 / 5.0 | 0.669 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 57.196 | 57.870 | 8.889 / 5.0 | 5.424 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 63.904 | 66.738 | 16.407 / 5.0 | 14.607 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 3848.660 | 3859.925 | 2.400 / 15.0 | 1.198 / 25.0 | unresolved |
| baseline-medium/calibration | double | cold_ms | 1514.270 | 1517.450 | 31.582 / 20.0 | 6.591 / 30.0 | unresolved |
| baseline-medium/calibration | double | early_median_ms | 150.308 | 150.364 | 0.426 / 5.0 | 0.281 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p90_ms | 151.984 | 152.064 | 0.612 / 5.0 | 0.501 / 10.0 | unresolved |
| baseline-medium/calibration | double | early_p99_ms | 158.145 | 158.173 | 5.733 / 5.0 | 3.206 / 10.0 | unresolved |
| baseline-medium/calibration | double | median_ms | 150.308 | 150.364 | 0.426 / 5.0 | 0.281 / 10.0 | unresolved |
| baseline-medium/calibration | double | p90_ms | 151.984 | 152.064 | 0.612 / 5.0 | 0.501 / 10.0 | unresolved |
| baseline-medium/calibration | double | p99_ms | 158.145 | 158.173 | 5.733 / 5.0 | 3.206 / 10.0 | unresolved |
| baseline-medium/calibration | double | setup_ms | 4178.125 | 4186.250 | 5.828 / 15.0 | 1.258 / 25.0 | unresolved |
| baseline-medium/calibration | float | cold_ms | 1395.065 | 1486.195 | 36.221 / 20.0 | 7.318 / 30.0 | unresolved |
| baseline-medium/calibration | float | early_median_ms | 58.310 | 58.359 | 0.892 / 5.0 | 0.524 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p90_ms | 60.195 | 60.305 | 3.643 / 5.0 | 1.064 / 10.0 | unresolved |
| baseline-medium/calibration | float | early_p99_ms | 65.771 | 65.935 | 13.380 / 5.0 | 9.678 / 10.0 | unresolved |
| baseline-medium/calibration | float | median_ms | 58.310 | 58.359 | 0.892 / 5.0 | 0.524 / 10.0 | unresolved |
| baseline-medium/calibration | float | p90_ms | 60.195 | 60.305 | 3.643 / 5.0 | 1.064 / 10.0 | unresolved |
| baseline-medium/calibration | float | p99_ms | 65.771 | 65.935 | 13.380 / 5.0 | 9.678 / 10.0 | unresolved |
| baseline-medium/calibration | float | setup_ms | 3978.370 | 3993.975 | 5.571 / 15.0 | 1.186 / 25.0 | unresolved |
| baseline-large/calibration | double | cold_ms | 4005.705 | 4011.320 | 2.891 / 20.0 | 0.675 / 30.0 | unresolved |
| baseline-large/calibration | double | early_median_ms | 2427.543 | 2427.522 | 0.045 / 5.0 | 0.011 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p90_ms | 2430.450 | 2430.302 | 0.088 / 5.0 | 0.033 / 10.0 | unresolved |
| baseline-large/calibration | double | early_p99_ms | 2439.990 | 2440.062 | 0.360 / 5.0 | 0.209 / 10.0 | unresolved |
| baseline-large/calibration | double | median_ms | 2427.543 | 2427.522 | 0.045 / 5.0 | 0.011 / 10.0 | unresolved |
| baseline-large/calibration | double | p90_ms | 2430.450 | 2430.302 | 0.088 / 5.0 | 0.033 / 10.0 | unresolved |
| baseline-large/calibration | double | p99_ms | 2439.990 | 2440.062 | 0.360 / 5.0 | 0.209 / 10.0 | unresolved |
| baseline-large/calibration | double | setup_ms | 8053.805 | 7347.600 | 10.822 / 15.0 | 10.223 / 25.0 | unresolved |
| baseline-large/calibration | float | cold_ms | 1673.215 | 1686.185 | 19.794 / 20.0 | 4.677 / 30.0 | unresolved |
| baseline-large/calibration | float | early_median_ms | 152.788 | 152.488 | 0.625 / 5.0 | 0.373 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p90_ms | 156.843 | 155.451 | 2.108 / 5.0 | 2.190 / 10.0 | unresolved |
| baseline-large/calibration | float | early_p99_ms | 167.309 | 164.012 | 6.200 / 5.0 | 2.426 / 10.0 | unresolved |
| baseline-large/calibration | float | median_ms | 152.788 | 152.488 | 0.625 / 5.0 | 0.373 / 10.0 | unresolved |
| baseline-large/calibration | float | p90_ms | 156.843 | 155.451 | 2.108 / 5.0 | 2.190 / 10.0 | unresolved |
| baseline-large/calibration | float | p99_ms | 167.309 | 164.012 | 6.200 / 5.0 | 2.426 / 10.0 | unresolved |
| baseline-large/calibration | float | setup_ms | 5875.605 | 5927.280 | 11.150 / 15.0 | 1.566 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 0.472 | 1.326 | 0.595 | 93.074 | 16900.450 | 0 | 3.135 | 0.576 |
| baseline-small/calibration | double | 0.625 | 1.832 | 0.901 | 679.215 | 6511.685 | 2 | 1.962 | 0.596 |
| baseline-medium/calibration | float | 0.519 | 1.464 | 0.476 | 119.260 | 16145.100 | 0 | 3.900 | 0.564 |
| baseline-medium/calibration | double | 0.638 | 1.257 | 0.623 | 279.269 | 6505.055 | 0 | 0.966 | 0.482 |
| baseline-large/calibration | float | 0.957 | 1.751 | 0.895 | 872.159 | 6388.725 | 4 | 2.107 | 0.596 |
| baseline-large/calibration | double | 1.097 | 1.974 | 0.585 | 2568.951 | 411.285 | 0 | 0.104 | 0.412 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/phase_shift_beamformer.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | dispersion, insufficient_interval_resolution |
| baseline-large | dispersion, insufficient_interval_resolution |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/phase_shift_beamformer/`; native evidence: `native/phase_shift_beamformer/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
