# root_music_estimator: round-one assessment

Assessment only, 50 frames × 3 isolated repeats, both precisions. No production changes or performance gains claimed. Timing qualification and full behavioral coverage are separate.

| Case | Native repeat | Timing disposition | Latest phase | Float A1 median µs | Double A1 median µs |
| --- | --- | --- | --- | --- | --- |
| baseline-small | repeat_passed | calibration_blocked | calibration | 57072.317 | 85886.470 |
| baseline-medium | repeat_passed | screen_execution_blocked | none | unknown | unknown |
| baseline-large | repeat_passed | screen_execution_blocked | none | unknown | unknown |

## Workload definitions

Profile axes and fixture values below define synthetic work, not universal radar size classes. Case overrides apply over the recipe defaults. Full input declarations and constructor bounds remain hash-bound in the exact recipe.

| Case | Profile slot | Axes in recipe order | Shape | Effective fixtures | Parameter overrides |
| --- | --- | --- | --- | --- | --- |
| baseline-small | small | channels, detections, snapshots | [4,64,16] | {"max_sources_bound":4} | {} |
| baseline-medium | medium | channels, detections, snapshots | [8,128,32] | {"max_sources_bound":4} | {} |
| baseline-large | large | channels, detections, snapshots | [16,256,48] | {"max_sources_bound":4} | {} |

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
  "max_sources": 4,
  "num_sources": -1,
  "source_count_pfa": 0.01,
  "spatial_smoothing": []
}
```

Output wiring: `{"drain":["doa_multi","source_count","eigenvalues"],"optional_tensor_outputs":["doa_multi","source_count","eigenvalues"],"queue_depth":8,"timed":"aoa_estimates","unwired":[]}`.

Native output comparison: `exact for every port and both precisions`.

A1/A2 below are paired run-summary distributions: each isolated run supplies a frame-window or initialization statistic; the three repeat statistics are aggregated by median. Rows show the median across paired aggregates. Individual frames are not independent experimental replicates. A/A intervals describe zero-change comparisons, not optimization gains.

| Case/phase | Precision | Metric | A1 µs | A2 µs | P90 A/A Δ % / limit | Baseline IQR % / limit | Familywise interval % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | double | cold_ms | 86573.700 | 86237.000 | 1.680 / 20.0 | 0.716 / 30.0 | unresolved |
| baseline-small/calibration | double | early_median_ms | 85886.470 | 85847.642 | 0.859 / 5.0 | 0.581 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p90_ms | 87012.511 | 86838.594 | 0.850 / 5.0 | 0.495 / 10.0 | unresolved |
| baseline-small/calibration | double | early_p99_ms | 90340.553 | 88671.070 | 6.928 / 5.0 | 3.855 / 10.0 | unresolved |
| baseline-small/calibration | double | median_ms | 85886.470 | 85847.642 | 0.859 / 5.0 | 0.581 / 10.0 | unresolved |
| baseline-small/calibration | double | p90_ms | 87012.511 | 86838.594 | 0.850 / 5.0 | 0.495 / 10.0 | unresolved |
| baseline-small/calibration | double | p99_ms | 90340.553 | 88671.070 | 6.928 / 5.0 | 3.855 / 10.0 | unresolved |
| baseline-small/calibration | double | setup_ms | 17178.000 | 16963.200 | 6.759 / 15.0 | 4.136 / 25.0 | unresolved |
| baseline-small/calibration | float | cold_ms | 57749.600 | 58069.300 | 1.561 / 20.0 | 0.999 / 30.0 | unresolved |
| baseline-small/calibration | float | early_median_ms | 57072.317 | 57243.137 | 0.813 / 5.0 | 0.476 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p90_ms | 58072.744 | 58260.622 | 1.203 / 5.0 | 0.583 / 10.0 | unresolved |
| baseline-small/calibration | float | early_p99_ms | 60217.781 | 61471.039 | 7.324 / 5.0 | 4.278 / 10.0 | unresolved |
| baseline-small/calibration | float | median_ms | 57072.317 | 57243.137 | 0.813 / 5.0 | 0.476 / 10.0 | unresolved |
| baseline-small/calibration | float | p90_ms | 58072.744 | 58260.622 | 1.203 / 5.0 | 0.583 / 10.0 | unresolved |
| baseline-small/calibration | float | p99_ms | 60217.781 | 61471.039 | 7.324 / 5.0 | 4.278 / 10.0 | unresolved |
| baseline-small/calibration | float | setup_ms | 17472.300 | 17314.550 | 7.510 / 15.0 | 4.015 / 25.0 | unresolved |

## Absolute variability diagnostics

These retain separate isolated runs. Within-run frame IQR and between-run median IQR answer different questions; neither is a universal hardware floor. No absolute allowance is used for acceptance. Drift, correlation and stalls are descriptive regime diagnostics, not a stationarity test or an additional gate.

| Case/phase | Precision | P90 absolute paired median delta µs | Median within-run frame IQR µs | Between-run median IQR µs | Largest measured frame µs | Median achieved FPS | Stall frames | Largest absolute half-window drift % | Largest absolute lag-1 correlation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small/calibration | float | 469.022 | 795.543 | 436.921 | 80339.324 | 17.413 | 0 | 1.835 | 0.540 |
| baseline-small/calibration | double | 736.443 | 857.561 | 693.368 | 103588.644 | 11.604 | 0 | 0.989 | 0.549 |

Per-run ECDF figure: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/plots/root_music_estimator.png`. Blue and orange identify A1/A2; each line is one isolated run. Figures are descriptive; no frames are excluded or treated as independent runs.

## Coverage and stop reason

Existing contract/analytic fixtures and casewise repeated snapshots are reported in the raw preparation receipts. Owning snapshots do not prove borrowed-loan retention. Live/optional/boundary coverage outside this static round remains unmeasured; no production candidate admission is implied. Source-code attribution and optimization ranking require representative profiles and complete affected-consumer verification.

| Case | Blockers or unresolved reason |
| --- | --- |
| baseline-small | dispersion, insufficient_interval_resolution |
| baseline-medium | Measurement timeout; full command, timing and cleanup receipts remain in the raw session. |
| baseline-large | Measurement timeout; full command, timing and cleanup receipts remain in the raw session. |

Raw session: `/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14`. Operator measurements: `measurements/root_music_estimator/`; native evidence: `native/root_music_estimator/`. All retained/rejected optimization changes: none. Stop: bounded round completed, validation blocked, or next units unaffordable as recorded casewise. Baseline/final and cumulative optimization comparisons: not applicable; no production changes.
