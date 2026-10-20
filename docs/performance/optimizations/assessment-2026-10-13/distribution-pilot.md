# Workload scaling and distribution qualification pilot

Status: completed bounded qualification pilot; assessment only. Branch: `deustice/optimization-skills`.

This revision uses existing compiled operators and preserved runtime libraries. Independently hashed scenario recipes are beside byte-identical executable copies under `build-optimization/operator-perf/assessment-2026-10-13/distribution-pilot/`. Original builds, recipes and previous qualification evidence are preserved. No clock, affinity, power or production changes.

Three scopes passed fresh 30-pair held-out qualification at **50 frames / 3 isolated repeats**, both precisions: larger RangeProcessor, stress RangeProcessor, and stress ClutterFilter. All protected checks passed without threshold changes. Other scopes remain blocked or incompletely explored; this is not a full baseline or a full-pass validation of the operators.

| Operator/scenario | Float median ms | Double median ms | Float/double median P90 A/A Δ % |
| --- | --- | --- | --- |
| fmcw_range_processor/larger | 0.630455 | 4.014874 | 1.197 / 0.134 |
| fmcw_range_processor/stress | 9.700866 | 67.511508 | 0.342 / 0.139 |
| clutter_filter/stress | 16.828069 | 32.421306 | 0.716 / 0.457 |

## Scope and protocol

| Scenario | Receive channels | Chirps | Samples/range bins | Input float MiB | Input double MiB |
| --- | --- | --- | --- | --- | --- |
| Diagnostic | 1 | 16 | 128 | 0.016 | 0.031 |
| Compact | 4 | 128 | 512 | 2 | 4 |
| Larger | 16 | 256 | 1024 | 32 | 64 |
| Stress | 64 | 512 | 2048 | 512 | 1024 |

These are synthetic scenarios, not universal radar sizes. RangeProcessor uses waveform durations consistent with the scenario sample count at 12.8 MSPS; ClutterFilter sees corresponding range-bin extents. TX multiplexing and deployment deadlines are unspecified. Range bins remain wired for RangeProcessor.

Two representative processing types: FFT (RangeProcessor) and reduction (ClutterFilter), both precisions. Broad two-pair screening precedes ten-pair calibration at 50 frames/3 repeats. Declared escalation order: 200/3, then 50/7. First eligible protocol is frozen before 30 fresh held-out pairs. Five warmup frames and the existing two-frame process warmup remain. No runs below 50 frames: version-3 early-window protection remains intact. All measured frames are retained.

Existing relative calibration thresholds and statistical acceptance gates remain authoritative. Absolute variability and distribution distances are diagnostics and inputs to a future explicit policy, not automatic threshold relaxation. Held-out validation is eligible only after exploratory dispersion gates pass. Insufficient interval resolution at ten pairs is reported separately.

The remaining user qualification allocation at launch is approximately 10,009 seconds (2 h 47 min), with the rejected scenario preflight attempt deducted. Actual phase duration is recorded in the pilot budget ledger. Qualification does not authorize a full baseline or production optimization.

## Preserved raw-frame evidence

Each frame timing spans source emission to a synchronized terminal sink. It includes host/graph/device interaction and cannot be interpreted as isolated kernel execution time. Each run remains a separate distribution; pooled frames are not independent experimental units. The table shows completed ten-pair 50/3 calibration only. The preserved ClutterFilter case is `baseline-medium` (one Rx, 64 chirps, 512 range bins); the preserved RangeProcessor case is `baseline-small` (one Rx, 16 chirps, 128 samples). These are distinct from the revised scenario results below.

| Operator | Precision | Median run median µs | Within-run IQR µs (median) | Between-run median IQR µs | P90 absolute paired median difference µs |
| --- | --- | --- | --- | --- | --- |
| clutter_filter | float | 790.619 | 21.007 | 63.117 | 65.494 |
| clutter_filter | double | 773.423 | 19.363 | 221.838 | 225.644 |
| fmcw_range_processor | float | 26.558 | 0.624 | 0.449 | 0.581 |
| fmcw_range_processor | double | 45.134 | 1.093 | 0.611 | 0.474 |

ClutterFilter has substantially wider between-run shifts than typical within-run spreads. RangeProcessor exhibits sub-microsecond paired median differences for this small workload. These observations do not identify a scheduling mechanism or establish a universal noise floor. Sampling levels ran at different times; differences between levels cannot establish a causal benefit of increasing frame count.

Per-run ECDF figure: `build-optimization/operator-perf/assessment-2026-10-13/distribution-pilot/existing-ecdf.png`. Raw analysis and verified pair references remain in ignored artifacts.

## Coverage and unresolved scopes

New recipes are validated before measurement and copied executables/runtime dependencies are hash-checked. Existing contract/analytic fixtures passed before this pilot. New scenario-sized repeated native exports have not been collected; the stress case emits 1.5 GiB of tensor payload across precisions per frame before serialization. Timing qualification alone does not certify correctness or full coverage of these workloads.

MUSIC and DBSCAN, all remaining operators, live overrides and transitive consumer matrices remain unqualified under the revised scenario protocol. No full-pass validation claim is made. Absolute-plus-relative limits require a declared, freshly validated policy before production acceptance.

## Results

Only complete paired units enter these tables. Confidence intervals use the existing familywise paired-run method; individual frames are not assumed independent. Screen and calibration rows do not certify held-out qualification.

| Operator | Scenario | Phase | Precision | Frames/repeats | Pairs | A1 median µs | A2 median µs | Median P90 Δ % | p99 P90 Δ % | Dispersion failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clutter_filter | compact | calibration-f200-r3 | float | 200/3 | 10 | 866.542 | 886.456 | 10.70 | 7.60 | 8 |
| clutter_filter | compact | calibration-f200-r3 | double | 200/3 | 10 | 809.419 | 884.446 | 27.63 | 12.31 | 10 |
| clutter_filter | compact | calibration-f50-r3 | float | 50/3 | 10 | 880.971 | 873.358 | 10.10 | 11.22 | 8 |
| clutter_filter | compact | calibration-f50-r3 | double | 50/3 | 10 | 936.997 | 941.460 | 24.86 | 13.66 | 9 |
| clutter_filter | compact | copy-control-f50-r3 | float | 50/3 | 2 | 31.761 | 31.549 | 0.79 | 35.96 | 4 |
| clutter_filter | compact | copy-control-f50-r3 | double | 50/3 | 2 | 39.149 | 40.199 | 5.07 | 26.21 | 8 |
| clutter_filter | compact | screen-f50-r3 | float | 50/3 | 2 | 929.084 | 911.187 | 3.17 | 4.29 | 2 |
| clutter_filter | compact | screen-f50-r3 | double | 50/3 | 2 | 1000.907 | 963.880 | 4.09 | 5.11 | 2 |
| clutter_filter | diagnostic | calibration-f200-r3 | float | 200/3 | 10 | 754.534 | 758.997 | 10.98 | 6.91 | 8 |
| clutter_filter | diagnostic | calibration-f200-r3 | double | 200/3 | 10 | 653.286 | 656.551 | 42.83 | 9.75 | 11 |
| clutter_filter | diagnostic | calibration-f50-r3 | float | 50/3 | 10 | 770.962 | 786.471 | 7.26 | 11.60 | 8 |
| clutter_filter | diagnostic | calibration-f50-r3 | double | 50/3 | 10 | 736.821 | 786.550 | 30.67 | 9.67 | 11 |
| clutter_filter | diagnostic | calibration-f50-r7 | float | 50/7 | 10 | 778.259 | 773.275 | 9.53 | 4.21 | 4 |
| clutter_filter | diagnostic | calibration-f50-r7 | double | 50/7 | 10 | 737.990 | 743.308 | 16.16 | 3.18 | 6 |
| clutter_filter | diagnostic | copy-control-f50-r3 | float | 50/3 | 2 | 14.341 | 14.249 | 1.36 | 18.90 | 2 |
| clutter_filter | diagnostic | copy-control-f50-r3 | double | 50/3 | 2 | 14.243 | 14.386 | 2.22 | 138.97 | 8 |
| clutter_filter | diagnostic | screen-f50-r3 | float | 50/3 | 2 | 772.474 | 787.390 | 2.02 | 3.52 | 0 |
| clutter_filter | diagnostic | screen-f50-r3 | double | 50/3 | 2 | 787.593 | 668.953 | 27.32 | 8.16 | 6 |
| clutter_filter | larger | calibration-f50-r3 | float | 50/3 | 10 | 1693.505 | 1674.758 | 4.56 | 10.79 | 4 |
| clutter_filter | larger | calibration-f50-r3 | double | 50/3 | 10 | 2723.736 | 2705.650 | 2.31 | 3.91 | 2 |
| clutter_filter | larger | copy-control-f50-r3 | float | 50/3 | 2 | 597.186 | 593.950 | 1.24 | 1.90 | 0 |
| clutter_filter | larger | copy-control-f50-r3 | double | 50/3 | 2 | 1180.272 | 1177.412 | 0.35 | 1.60 | 0 |
| clutter_filter | larger | screen-f50-r3 | float | 50/3 | 2 | 1667.457 | 1654.643 | 2.29 | 11.22 | 4 |
| clutter_filter | larger | screen-f50-r3 | double | 50/3 | 2 | 2758.057 | 2701.141 | 3.47 | 5.09 | 4 |
| clutter_filter | stress | calibration-f50-r3 | float | 50/3 | 10 | 16858.225 | 16855.046 | 0.60 | 1.35 | 0 |
| clutter_filter | stress | calibration-f50-r3 | double | 50/3 | 10 | 32388.855 | 32400.876 | 0.43 | 0.71 | 0 |
| clutter_filter | stress | copy-control-f50-r3 | float | 50/3 | 2 | 9384.030 | 9371.590 | 0.17 | 0.62 | 0 |
| clutter_filter | stress | copy-control-f50-r3 | double | 50/3 | 2 | 18697.469 | 18691.474 | 0.12 | 1.23 | 0 |
| clutter_filter | stress | heldout | float | 50/3 | 30 | 16828.069 | 16837.533 | 0.72 | 2.00 | 0 |
| clutter_filter | stress | heldout | double | 50/3 | 30 | 32421.306 | 32403.481 | 0.46 | 0.75 | 0 |
| clutter_filter | stress | screen-f50-r3 | float | 50/3 | 2 | 16837.386 | 16834.142 | 0.09 | 1.37 | 0 |
| clutter_filter | stress | screen-f50-r3 | double | 50/3 | 2 | 32406.520 | 32445.256 | 0.26 | 0.18 | 0 |
| fmcw_range_processor | compact | calibration-f200-r3 | float | 200/3 | 10 | 29.541 | 29.491 | 38.53 | 7.32 | 3 |
| fmcw_range_processor | compact | calibration-f200-r3 | double | 200/3 | 10 | 227.457 | 227.054 | 0.75 | 2.56 | 1 |
| fmcw_range_processor | compact | calibration-f50-r3 | float | 50/3 | 10 | 29.845 | 29.683 | 5.32 | 13.63 | 8 |
| fmcw_range_processor | compact | calibration-f50-r3 | double | 50/3 | 10 | 227.059 | 227.581 | 0.51 | 1.87 | 0 |
| fmcw_range_processor | compact | calibration-f50-r7 | float | 50/7 | 10 | 29.761 | 29.659 | 2.63 | 10.06 | 2 |
| fmcw_range_processor | compact | calibration-f50-r7 | double | 50/7 | 10 | 227.341 | 227.892 | 0.47 | 2.04 | 0 |
| fmcw_range_processor | compact | screen-f50-r3 | float | 50/3 | 2 | 37.072 | 33.714 | 33.52 | 1.72 | 4 |
| fmcw_range_processor | compact | screen-f50-r3 | double | 50/3 | 2 | 228.653 | 227.885 | 0.63 | 8.13 | 2 |
| fmcw_range_processor | diagnostic | calibration-f200-r3 | float | 200/3 | 10 | 26.334 | 26.405 | 1.25 | 47.88 | 4 |
| fmcw_range_processor | diagnostic | calibration-f200-r3 | double | 200/3 | 10 | 44.403 | 45.063 | 3.26 | 8.76 | 4 |
| fmcw_range_processor | diagnostic | calibration-f50-r3 | float | 50/3 | 10 | 26.526 | 26.641 | 2.46 | 21.07 | 2 |
| fmcw_range_processor | diagnostic | calibration-f50-r3 | double | 50/3 | 10 | 45.099 | 45.224 | 3.40 | 11.62 | 6 |
| fmcw_range_processor | diagnostic | calibration-f50-r7 | float | 50/7 | 10 | 26.549 | 26.574 | 1.75 | 42.64 | 4 |
| fmcw_range_processor | diagnostic | calibration-f50-r7 | double | 50/7 | 10 | 45.097 | 44.929 | 0.65 | 13.44 | 2 |
| fmcw_range_processor | diagnostic | screen-f50-r3 | float | 50/3 | 2 | 26.463 | 26.651 | 1.21 | 51.47 | 4 |
| fmcw_range_processor | diagnostic | screen-f50-r3 | double | 50/3 | 2 | 44.905 | 45.037 | 1.95 | 3.23 | 0 |
| fmcw_range_processor | larger | calibration-f50-r3 | float | 50/3 | 10 | 627.606 | 626.485 | 0.90 | 3.13 | 0 |
| fmcw_range_processor | larger | calibration-f50-r3 | double | 50/3 | 10 | 4013.983 | 4012.316 | 0.09 | 1.84 | 0 |
| fmcw_range_processor | larger | heldout | float | 50/3 | 30 | 630.455 | 630.091 | 1.20 | 1.95 | 0 |
| fmcw_range_processor | larger | heldout | double | 50/3 | 30 | 4014.874 | 4014.827 | 0.13 | 1.40 | 0 |
| fmcw_range_processor | larger | screen-f50-r3 | float | 50/3 | 2 | 627.531 | 630.353 | 0.54 | 1.12 | 0 |
| fmcw_range_processor | larger | screen-f50-r3 | double | 50/3 | 2 | 4013.675 | 4010.199 | 0.13 | 0.23 | 0 |
| fmcw_range_processor | stress | calibration-f50-r3 | float | 50/3 | 10 | 9691.836 | 9693.713 | 0.45 | 0.79 | 0 |
| fmcw_range_processor | stress | calibration-f50-r3 | double | 50/3 | 10 | 67501.322 | 67505.121 | 0.15 | 0.42 | 0 |
| fmcw_range_processor | stress | heldout | float | 50/3 | 30 | 9700.866 | 9697.797 | 0.34 | 1.25 | 0 |
| fmcw_range_processor | stress | heldout | double | 50/3 | 30 | 67511.508 | 67510.260 | 0.14 | 0.21 | 0 |
| fmcw_range_processor | stress | screen-f50-r3 | float | 50/3 | 2 | 9715.601 | 9699.415 | 0.21 | 0.57 | 0 |
| fmcw_range_processor | stress | screen-f50-r3 | double | 50/3 | 2 | 67487.893 | 67512.370 | 0.04 | 0.11 | 0 |

## Absolute calibration shifts and uncertainty

P90 absolute differences below compare independently aggregated run quantiles in each A/A pair. They are observations, not acceptable regression budgets or proven platform noise floors. Signed median-shift intervals are familywise 95% bounds in µs; unresolved intervals are retained.

| Operator | Scenario | Phase | Precision | P90 absolute median Δ µs | P90 absolute p90 Δ µs | P90 absolute p99 Δ µs | Signed median shift CI µs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clutter_filter | compact | calibration-f200-r3 | float | 103.423 | 239.971 | 124.619 | [None, None] |
| clutter_filter | compact | calibration-f200-r3 | double | 202.120 | 407.618 | 194.979 | [None, None] |
| clutter_filter | compact | calibration-f50-r3 | float | 93.411 | 508.586 | 177.561 | [None, None] |
| clutter_filter | compact | calibration-f50-r3 | double | 230.100 | 521.473 | 229.263 | [None, None] |
| clutter_filter | compact | copy-control-f50-r3 | float | 0.250 | 6.003 | 11.837 | [None, None] |
| clutter_filter | compact | copy-control-f50-r3 | double | 1.961 | 5.598 | 10.844 | [None, None] |
| clutter_filter | compact | screen-f50-r3 | float | 31.054 | 199.646 | 73.529 | [None, None] |
| clutter_filter | compact | screen-f50-r3 | double | 40.272 | 34.848 | 92.990 | [None, None] |
| clutter_filter | diagnostic | calibration-f200-r3 | float | 84.208 | 450.979 | 113.568 | [None, None] |
| clutter_filter | diagnostic | calibration-f200-r3 | double | 238.997 | 410.418 | 142.965 | [None, None] |
| clutter_filter | diagnostic | calibration-f50-r3 | float | 53.755 | 467.311 | 176.479 | [None, None] |
| clutter_filter | diagnostic | calibration-f50-r3 | double | 195.700 | 344.121 | 143.123 | [None, None] |
| clutter_filter | diagnostic | calibration-f50-r7 | float | 75.467 | 122.627 | 68.231 | [None, None] |
| clutter_filter | diagnostic | calibration-f50-r7 | double | 104.391 | 101.188 | 50.596 | [None, None] |
| clutter_filter | diagnostic | copy-control-f50-r3 | float | 0.196 | 0.293 | 6.036 | [None, None] |
| clutter_filter | diagnostic | copy-control-f50-r3 | double | 0.313 | 14.442 | 21.754 | [None, None] |
| clutter_filter | diagnostic | screen-f50-r3 | float | 15.649 | 19.353 | 52.855 | [None, None] |
| clutter_filter | diagnostic | screen-f50-r3 | double | 214.155 | 251.245 | 132.713 | [None, None] |
| clutter_filter | larger | calibration-f50-r3 | float | 74.672 | 201.333 | 293.472 | [None, None] |
| clutter_filter | larger | calibration-f50-r3 | double | 64.026 | 262.820 | 144.917 | [None, None] |
| clutter_filter | larger | copy-control-f50-r3 | float | 7.463 | 7.494 | 12.835 | [None, None] |
| clutter_filter | larger | copy-control-f50-r3 | double | 4.128 | 10.201 | 20.635 | [None, None] |
| clutter_filter | larger | screen-f50-r3 | float | 39.171 | 410.161 | 301.556 | [None, None] |
| clutter_filter | larger | screen-f50-r3 | double | 97.539 | 277.407 | 201.010 | [None, None] |
| clutter_filter | stress | calibration-f50-r3 | float | 101.099 | 153.082 | 231.891 | [None, None] |
| clutter_filter | stress | calibration-f50-r3 | double | 138.878 | 136.474 | 236.222 | [None, None] |
| clutter_filter | stress | copy-control-f50-r3 | float | 16.321 | 6.034 | 58.868 | [None, None] |
| clutter_filter | stress | copy-control-f50-r3 | double | 22.505 | 138.457 | 235.212 | [None, None] |
| clutter_filter | stress | heldout | float | 120.727 | 169.422 | 348.539 | [-66.30099999999928, 95.458500000003] |
| clutter_filter | stress | heldout | double | 147.810 | 254.431 | 249.005 | [-89.9765000000059, 65.62249999999636] |
| clutter_filter | stress | screen-f50-r3 | float | 15.412 | 94.611 | 236.813 | [None, None] |
| clutter_filter | stress | screen-f50-r3 | double | 85.483 | 128.155 | 59.603 | [None, None] |
| fmcw_range_processor | compact | calibration-f200-r3 | float | 11.330 | 0.379 | 3.954 | [None, None] |
| fmcw_range_processor | compact | calibration-f200-r3 | double | 1.710 | 2.025 | 6.166 | [None, None] |
| fmcw_range_processor | compact | calibration-f50-r3 | float | 1.613 | 4.438 | 6.758 | [None, None] |
| fmcw_range_processor | compact | calibration-f50-r3 | double | 1.162 | 1.976 | 4.449 | [None, None] |
| fmcw_range_processor | compact | calibration-f50-r7 | float | 0.796 | 1.685 | 5.743 | [None, None] |
| fmcw_range_processor | compact | calibration-f50-r7 | double | 1.074 | 2.653 | 4.970 | [None, None] |
| fmcw_range_processor | compact | screen-f50-r3 | float | 14.512 | 0.552 | 0.899 | [None, None] |
| fmcw_range_processor | compact | screen-f50-r3 | double | 1.438 | 6.706 | 21.704 | [None, None] |
| fmcw_range_processor | diagnostic | calibration-f200-r3 | float | 0.329 | 0.511 | 15.187 | [None, None] |
| fmcw_range_processor | diagnostic | calibration-f200-r3 | double | 1.562 | 1.711 | 5.307 | [None, None] |
| fmcw_range_processor | diagnostic | calibration-f50-r3 | float | 0.662 | 0.752 | 7.431 | [None, None] |
| fmcw_range_processor | diagnostic | calibration-f50-r3 | double | 1.522 | 4.299 | 6.429 | [None, None] |
| fmcw_range_processor | diagnostic | calibration-f50-r7 | float | 0.463 | 0.359 | 17.208 | [None, None] |
| fmcw_range_processor | diagnostic | calibration-f50-r7 | double | 0.295 | 1.613 | 7.115 | [None, None] |
| fmcw_range_processor | diagnostic | screen-f50-r3 | float | 0.318 | 0.174 | 31.779 | [None, None] |
| fmcw_range_processor | diagnostic | screen-f50-r3 | double | 0.863 | 0.419 | 1.790 | [None, None] |
| fmcw_range_processor | larger | calibration-f50-r3 | float | 5.681 | 7.589 | 22.371 | [None, None] |
| fmcw_range_processor | larger | calibration-f50-r3 | double | 3.760 | 24.393 | 74.695 | [None, None] |
| fmcw_range_processor | larger | heldout | float | 7.514 | 13.412 | 13.652 | [-6.828999999999974, 6.385000000000085] |
| fmcw_range_processor | larger | heldout | double | 5.367 | 39.355 | 57.203 | [-4.0645000000001374, 4.040999999999961] |
| fmcw_range_processor | larger | screen-f50-r3 | float | 3.402 | 2.386 | 7.772 | [None, None] |
| fmcw_range_processor | larger | screen-f50-r3 | double | 5.232 | 5.680 | 9.141 | [None, None] |
| fmcw_range_processor | stress | calibration-f50-r3 | float | 43.476 | 49.286 | 78.908 | [None, None] |
| fmcw_range_processor | stress | calibration-f50-r3 | double | 98.586 | 166.438 | 283.147 | [None, None] |
| fmcw_range_processor | stress | heldout | float | 33.134 | 76.686 | 122.822 | [-18.008999999999276, 22.491999999997958] |
| fmcw_range_processor | stress | heldout | double | 93.548 | 170.236 | 144.793 | [-45.064999999993915, 36.38600000000736] |
| fmcw_range_processor | stress | screen-f50-r3 | float | 20.860 | 89.079 | 57.330 | [None, None] |
| fmcw_range_processor | stress | screen-f50-r3 | double | 30.032 | 118.968 | 75.859 | [None, None] |

## Calibration and held-out failures against declared limits

Every row below failed a dispersion limit. A1/A2 are aggregate absolute timings; P90 delta and IQR are separate statistics across paired runs, so near-equal A1/A2 do not imply stable pairs. Early and complete metrics remain explicitly protected. Confidence intervals are conditional on the paired-run assumptions; unresolved regimes do not establish independence.

| Operator/scenario | Phase | Precision/metric | A1 µs | A2 µs | P90 Δ % / limit | IQR % / limit | Familywise CI % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clutter_filter/compact | calibration-f200-r3 | double/cold_ms | 917.012 | 900.470 | 26.13 / 20.00 | 21.78 / 30.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | double/early_median_ms | 905.204 | 888.122 | 10.16 / 5.00 | 7.99 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | double/early_p90_ms | 1063.917 | 1426.530 | 55.29 / 5.00 | 39.07 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | double/early_p99_ms | 1694.710 | 1693.213 | 12.14 / 5.00 | 6.99 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | double/median_ms | 809.419 | 884.446 | 27.63 / 5.00 | 17.71 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | double/p90_ms | 1294.930 | 1326.546 | 35.85 / 5.00 | 27.08 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | double/p99_ms | 1676.852 | 1749.831 | 12.31 / 5.00 | 8.03 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | float/cold_ms | 982.155 | 880.048 | 21.45 / 20.00 | 22.28 / 30.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | float/early_median_ms | 879.768 | 889.981 | 9.71 / 5.00 | 8.05 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | float/early_p90_ms | 1233.054 | 1073.697 | 35.09 / 5.00 | 40.04 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | float/early_p99_ms | 1655.145 | 1737.081 | 7.88 / 5.00 | 5.76 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | float/median_ms | 866.542 | 886.456 | 10.70 / 5.00 | 6.00 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | float/p90_ms | 1365.354 | 1319.088 | 17.70 / 5.00 | 8.61 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f200-r3 | float/p99_ms | 1645.305 | 1704.852 | 7.60 / 5.00 | 4.04 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | double/cold_ms | 997.701 | 1083.925 | 30.57 / 20.00 | 16.61 / 30.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | double/early_median_ms | 936.997 | 941.460 | 24.86 / 5.00 | 7.89 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | double/early_p90_ms | 1158.981 | 1362.272 | 50.45 / 5.00 | 22.64 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | double/early_p99_ms | 1768.943 | 1725.815 | 13.66 / 5.00 | 6.79 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | double/median_ms | 936.997 | 941.460 | 24.86 / 5.00 | 7.89 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | double/p90_ms | 1158.981 | 1362.272 | 50.45 / 5.00 | 22.64 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | double/p99_ms | 1768.943 | 1725.815 | 13.66 / 5.00 | 6.79 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | float/early_median_ms | 880.971 | 873.358 | 10.10 / 5.00 | 3.90 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | float/early_p90_ms | 1337.485 | 1162.892 | 40.51 / 5.00 | 31.00 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | float/early_p99_ms | 1642.016 | 1639.900 | 11.22 / 5.00 | 5.76 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | float/median_ms | 880.971 | 873.358 | 10.10 / 5.00 | 3.90 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | float/p90_ms | 1337.485 | 1162.892 | 40.51 / 5.00 | 31.00 / 10.00 | [None, None] |
| clutter_filter/compact | calibration-f50-r3 | float/p99_ms | 1642.016 | 1639.900 | 11.22 / 5.00 | 5.76 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | double/early_median_ms | 39.149 | 40.199 | 5.07 / 5.00 | 1.50 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | double/early_p90_ms | 40.104 | 44.941 | 13.81 / 5.00 | 9.18 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | double/early_p99_ms | 41.053 | 51.574 | 26.21 / 5.00 | 21.52 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | double/median_ms | 39.149 | 40.199 | 5.07 / 5.00 | 1.50 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | double/p90_ms | 40.104 | 44.941 | 13.81 / 5.00 | 9.18 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | double/p99_ms | 41.053 | 51.574 | 26.21 / 5.00 | 21.52 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | float/early_p90_ms | 32.513 | 35.612 | 18.50 / 5.00 | 5.60 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | float/early_p99_ms | 33.901 | 40.596 | 35.96 / 5.00 | 9.94 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | float/p90_ms | 32.513 | 35.612 | 18.50 / 5.00 | 5.60 / 10.00 | [None, None] |
| clutter_filter/compact | copy-control-f50-r3 | float/p99_ms | 33.901 | 40.596 | 35.96 / 5.00 | 9.94 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | double/cold_ms | 876.896 | 910.692 | 26.15 / 20.00 | 19.78 / 30.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | double/early_median_ms | 734.865 | 688.675 | 41.30 / 5.00 | 26.68 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | double/early_p90_ms | 772.578 | 743.677 | 59.29 / 5.00 | 32.40 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | double/early_p99_ms | 1466.188 | 1514.310 | 9.23 / 5.00 | 4.89 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | double/median_ms | 653.286 | 656.551 | 42.83 / 5.00 | 30.03 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | double/p90_ms | 782.066 | 716.825 | 59.06 / 5.00 | 35.01 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | double/p99_ms | 1510.901 | 1479.929 | 9.75 / 5.00 | 7.02 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | float/early_median_ms | 758.425 | 772.319 | 9.89 / 5.00 | 7.21 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | float/early_p90_ms | 856.060 | 816.607 | 21.77 / 5.00 | 12.39 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | float/early_p99_ms | 1559.752 | 1539.437 | 6.60 / 5.00 | 5.85 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | float/median_ms | 754.534 | 758.997 | 10.98 / 5.00 | 5.27 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | float/p90_ms | 818.677 | 830.942 | 41.37 / 5.00 | 12.96 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f200-r3 | float/p99_ms | 1553.169 | 1555.949 | 6.91 / 5.00 | 3.18 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | double/cold_ms | 927.353 | 924.433 | 25.00 / 20.00 | 8.84 / 30.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | double/early_median_ms | 736.821 | 786.550 | 30.67 / 5.00 | 12.12 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | double/early_p90_ms | 842.070 | 885.638 | 35.97 / 5.00 | 11.16 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | double/early_p99_ms | 1554.471 | 1555.460 | 9.67 / 5.00 | 4.88 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | double/median_ms | 736.821 | 786.550 | 30.67 / 5.00 | 12.12 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | double/p90_ms | 842.070 | 885.638 | 35.97 / 5.00 | 11.16 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | double/p99_ms | 1554.471 | 1555.460 | 9.67 / 5.00 | 4.88 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | float/early_median_ms | 770.962 | 786.471 | 7.26 / 5.00 | 6.41 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | float/early_p90_ms | 881.695 | 952.627 | 53.29 / 5.00 | 25.45 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | float/early_p99_ms | 1582.246 | 1577.800 | 11.60 / 5.00 | 6.71 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | float/median_ms | 770.962 | 786.471 | 7.26 / 5.00 | 6.41 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | float/p90_ms | 881.695 | 952.627 | 53.29 / 5.00 | 25.45 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r3 | float/p99_ms | 1582.246 | 1577.800 | 11.60 / 5.00 | 6.71 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | double/early_median_ms | 737.990 | 743.308 | 16.16 / 5.00 | 5.73 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | double/early_p90_ms | 876.239 | 826.708 | 12.26 / 5.00 | 10.85 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | double/median_ms | 737.990 | 743.308 | 16.16 / 5.00 | 5.73 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | double/p90_ms | 876.239 | 826.708 | 12.26 / 5.00 | 10.85 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | float/early_median_ms | 778.259 | 773.275 | 9.53 / 5.00 | 3.10 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | float/early_p90_ms | 886.782 | 860.013 | 11.80 / 5.00 | 7.64 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | float/median_ms | 778.259 | 773.275 | 9.53 / 5.00 | 3.10 / 10.00 | [None, None] |
| clutter_filter/diagnostic | calibration-f50-r7 | float/p90_ms | 886.782 | 860.013 | 11.80 / 5.00 | 7.64 / 10.00 | [None, None] |
| clutter_filter/diagnostic | copy-control-f50-r3 | double/early_p90_ms | 14.761 | 22.512 | 99.83 / 5.00 | 29.47 / 10.00 | [None, None] |
| clutter_filter/diagnostic | copy-control-f50-r3 | double/early_p99_ms | 23.102 | 26.103 | 138.97 / 5.00 | 76.76 / 10.00 | [None, None] |
| clutter_filter/diagnostic | copy-control-f50-r3 | double/p90_ms | 14.761 | 22.512 | 99.83 / 5.00 | 29.47 / 10.00 | [None, None] |
| clutter_filter/diagnostic | copy-control-f50-r3 | double/p99_ms | 23.102 | 26.103 | 138.97 / 5.00 | 76.76 / 10.00 | [None, None] |
| clutter_filter/diagnostic | copy-control-f50-r3 | float/early_p99_ms | 28.711 | 25.083 | 18.90 / 5.00 | 7.19 / 10.00 | [None, None] |
| clutter_filter/diagnostic | copy-control-f50-r3 | float/p99_ms | 28.711 | 25.083 | 18.90 / 5.00 | 7.19 / 10.00 | [None, None] |
| clutter_filter/larger | calibration-f50-r3 | double/early_p90_ms | 3515.055 | 3480.129 | 7.48 / 5.00 | 3.87 / 10.00 | [None, None] |
| clutter_filter/larger | calibration-f50-r3 | double/p90_ms | 3515.055 | 3480.129 | 7.48 / 5.00 | 3.87 / 10.00 | [None, None] |
| clutter_filter/larger | calibration-f50-r3 | float/early_p90_ms | 2287.384 | 2255.149 | 8.64 / 5.00 | 3.31 / 10.00 | [None, None] |
| clutter_filter/larger | calibration-f50-r3 | float/early_p99_ms | 2610.851 | 2616.274 | 10.79 / 5.00 | 7.06 / 10.00 | [None, None] |
| clutter_filter/larger | calibration-f50-r3 | float/p90_ms | 2287.384 | 2255.149 | 8.64 / 5.00 | 3.31 / 10.00 | [None, None] |
| clutter_filter/larger | calibration-f50-r3 | float/p99_ms | 2610.851 | 2616.274 | 10.79 / 5.00 | 7.06 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f200-r3 | double/early_p99_ms | 241.405 | 242.617 | 8.28 / 5.00 | 1.90 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f200-r3 | float/early_p99_ms | 50.602 | 49.968 | 7.89 / 5.00 | 3.71 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f200-r3 | float/median_ms | 29.541 | 29.491 | 38.53 / 5.00 | 0.93 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f200-r3 | float/p99_ms | 53.488 | 52.954 | 7.32 / 5.00 | 3.19 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r3 | float/early_median_ms | 29.845 | 29.683 | 5.32 / 5.00 | 1.59 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r3 | float/early_p90_ms | 46.115 | 46.203 | 8.97 / 5.00 | 3.14 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r3 | float/early_p99_ms | 52.822 | 54.794 | 13.63 / 5.00 | 10.15 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r3 | float/median_ms | 29.845 | 29.683 | 5.32 / 5.00 | 1.59 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r3 | float/p90_ms | 46.115 | 46.203 | 8.97 / 5.00 | 3.14 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r3 | float/p99_ms | 52.822 | 54.794 | 13.63 / 5.00 | 10.15 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r7 | float/early_p99_ms | 52.441 | 52.140 | 10.06 / 5.00 | 4.52 / 10.00 | [None, None] |
| fmcw_range_processor/compact | calibration-f50-r7 | float/p99_ms | 52.441 | 52.140 | 10.06 / 5.00 | 4.52 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f200-r3 | double/early_p90_ms | 46.879 | 46.515 | 8.38 / 5.00 | 2.10 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f200-r3 | double/early_p99_ms | 53.506 | 53.511 | 24.32 / 5.00 | 12.72 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f200-r3 | double/p99_ms | 55.578 | 54.922 | 8.76 / 5.00 | 6.08 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f200-r3 | float/early_p99_ms | 28.178 | 28.695 | 20.66 / 5.00 | 13.90 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f200-r3 | float/p99_ms | 33.316 | 35.953 | 47.88 / 5.00 | 27.05 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r3 | double/early_p90_ms | 46.619 | 46.751 | 8.73 / 5.00 | 1.78 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r3 | double/early_p99_ms | 51.529 | 52.585 | 11.62 / 5.00 | 14.51 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r3 | double/p90_ms | 46.619 | 46.751 | 8.73 / 5.00 | 1.78 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r3 | double/p99_ms | 51.529 | 52.585 | 11.62 / 5.00 | 14.51 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r3 | float/early_p99_ms | 28.264 | 27.769 | 21.07 / 5.00 | 5.24 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r3 | float/p99_ms | 28.264 | 27.769 | 21.07 / 5.00 | 5.24 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r7 | double/early_p99_ms | 54.006 | 51.458 | 13.44 / 5.00 | 8.46 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r7 | double/p99_ms | 54.006 | 51.458 | 13.44 / 5.00 | 8.46 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r7 | float/early_p99_ms | 28.635 | 28.003 | 42.64 / 5.00 | 33.19 / 10.00 | [None, None] |
| fmcw_range_processor/diagnostic | calibration-f50-r7 | float/p99_ms | 28.635 | 28.003 | 42.64 / 5.00 | 33.19 / 10.00 | [None, None] |

## Duration and disposition

Pilot elapsed qualification time: 162.7 minutes. Original qualification ledger before this revision: approximately 73.2 minutes. The pilot budget includes orchestration, identity checks, inventory and measurement processes; it is not an estimate of agent working time.

The run-policy, budget ledger, immutable frozen selections, paired reports, raw precision children, owned-process receipts, runtime checks and distribution diagnostics are under the ignored pilot directory. No original evidence was upgraded or replaced.

| Operator | Scenario | Selected frames/repeats | Held-out disposition |
| --- | --- | --- | --- |
| fmcw_range_processor | diagnostic | None | calibration_blocked |
| clutter_filter | diagnostic | None | calibration_blocked |
| fmcw_range_processor | compact | None | calibration_blocked |
| clutter_filter | compact | None | calibration_blocked_sampling_incomplete |
| fmcw_range_processor | larger | [50, 3] | pass |
| clutter_filter | larger | None | calibration_blocked_sampling_incomplete |
| fmcw_range_processor | stress | [50, 3] | pass |
| clutter_filter | stress | [50, 3] | pass |

## Recommendations and stop reason

Use 50/3 for these three measured workloads. Qualification of other cases, variants, optional outputs, live transitions and consumers remains necessary. Do not infer that larger inputs make every operator stable. Keep diagnostic, compact, larger and stress scenarios as separately identified workloads; none is a universal radar size or a deployment deadline.

Retain the first-50 protection and five warmup frames. Exploratory first-20 summaries from the completed 50-frame validation runs increased the larger float FFT median P90 A/A delta from 1.197% to 2.371%. The stress prefixes remained comparatively stable, but a prefix does not validate a real 20-frame process. Actual shorter tests need a separately declared early-window policy and fresh validation. No measured observations were discarded or historical evidence relabeled.

Record an empirical absolute variability band per workload, precision and metric alongside percentages. A future absolute-plus-relative calibration policy can use `max(relative_limit × latency, declared_absolute_allowance)` with separate setup/cold/transition policies. Allowances must be frozen prospectively and validated on disjoint runs. They must not be inferred from the worst observed run or automatically become permitted candidate regressions. This pilot establishes no universal system-wide numerical floor and widens no acceptance threshold. Quantile confidence intervals and distribution distances preserve run boundaries; pooling correlated frames into an independent-sample test is inappropriate.

Matched copy probes use ClutterFilter's existing `method: none`, the same shape and ports, 50/3 and two A/A pairs. They include materialized GPU copy work and remain descriptive. The diagnostic copy medians were 14.34 µs float and 14.24 µs double; mean subtraction was approximately 0.77 ms. This directs attribution toward reduction-specific work. The shared `clutter_stage.hpp` creates a MatX mean transform whose temporary is materialized during the consuming expression's PreRun; inspect its allocation/materialization path with profiling before proposing an optimization. The fused RangeDoppler operator consumes this shared stage and would require consumer validation. No gain is inferred by subtracting copy time, and no production change was attempted.

Use startup-aware cost forecasts: measured per-process cost plus additional frame cost, with explicit margins. On small workloads, extra frames cost much less than relaunching processes. The follow-up refined an overly conservative proportional frame-cost forecast, used the same remaining user budget and retained all failed measurements. Raw storage forecasts must include JSON/log/telemetry overhead and native verification captures, not only frame arrays.

Per-run calibration ECDF figures and absolute distribution-distance diagnostics are in the ignored pilot directory. Representative native correctness fixtures and original case-size repeat captures passed before timing (211 C++ tests and 290 precision-case combinations). The new stress tensors require large verification exports; no new scenario-sized native output captures were collected. These timings alone cannot admit a full correctness-qualified baseline. No new suspected correctness bug was observed; previously documented RootMusic rank-deficient-covariance and borrowed-output coverage questions remain outside this pilot. See `docs/correctness/root-music.md`.

Nsight Systems is available but was not used during qualification; Nsight Compute is unavailable. No CPU fallback, privileged profiling setup, affinity changes, clock changes or power changes were introduced. Operator source and compiled harnesses remain unchanged. Installed HSDK version is 5.0.0; its source revision remains unknown. The immutable image and archived runtime hashes identify the measured deployment.

Stop reason: the declared primary matrix is complete; remaining sampling levels were unaffordable within the original qualification allocation. Blocked calibration remains blocked. Full baseline collection, production optimization, commits and external publication were not performed.

Charged qualification time across the original and revised attempts: **235.9 minutes of 240**; approximately **4.1 minutes** remained unused. The revised pilot used **162.7 minutes**. Its **1824 timing jobs** accounted for **119.2 minutes** of process wall time (including fixture setup and both precision executables). Other phase time includes hashes, telemetry, inventories, reviews and orchestration; it is not agent working time. Build/preparation/report/test time is separate.

Validation: 83 maintained host-tool tests passed after measurements. Pinned all-file checks are recorded in the final validation receipt.
