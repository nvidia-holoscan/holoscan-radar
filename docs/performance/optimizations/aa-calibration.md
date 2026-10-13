# A/A calibration measurements

These measurements compare the same preserved baseline executable against itself. No production change was tested. Twenty alternating pairs covered baseline-small, baseline-medium and baseline-large in both precisions for 24 operators. RootMUSIC and DetectionReduction were not sampled because native validation was blocked. Static variants, live transitions and optional wiring beyond these baseline cases still require calibration.

## Reading the results

A1 and A2 identify the before/after roles in each pair, not different implementations. Execution order alternates between pairs. Each side uses three isolated repeat processes per case and precision, with five warmup and fifty measured frames per process. Each pair-side value is the median of the three repeat-report values. The A1 and A2 columns below are medians across twenty such observations. For p90 and p99 rows, these are summaries of per-process frame percentiles, not percentiles pooled across all frames.

The stability checks use the individual paired observations, not the difference between the two overall medians:

- **Paired delta P90:** the 90th percentile across twenty absolute differences, each computed as `100 * abs(A1 - A2) / A1`.
- **Baseline IQR:** `100 * (Q75 - Q25) / median` across all forty A1/A2 observations for that row.
- A row fails calibration if either value exceeds its corresponding limit. Both must pass. The proposed limits have not been relaxed.

| Metric | Paired delta P90 limit | Baseline IQR limit |
| --- | --- | --- |
| Steady median, p90, p99 | 5% | 10% |
| Setup | 15% | 25% |
| Cold | 20% | 30% |
| Live override (not sampled here) | 25% | 40% |

All absolute times are milliseconds. PASS below means that this individual row meets the calibration limits; it does not qualify the whole operator or establish a performance improvement. All 24 operators have at least one failing row.

## Worked example: ClutterFilter large double

The median-latency row has similar overall A1/A2 medians (0.664749/0.662129 ms), yet its paired delta P90 is 14.66%, above the 5% limit. The frame-p90 row fails both checks: paired delta P90 26.04% against 5%, and IQR 15.98% against 10%. Setup passes its separate, wider limits. This illustrates why a small difference between group medians does not establish repeatability.

The following are all twenty pair-side values for this workload, before taking medians across pairs. Each time is already aggregated across three isolated repeats. Delta columns are absolute percentages relative to A1.

| Pair | Median A1 | Median A2 | Median delta | Frame-p90 A1 | Frame-p90 A2 | Frame-p90 delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.672820 | 0.666709 | 0.91% | 0.703757 | 0.794764 | 12.93% |
| 2 | 0.606478 | 0.602237 | 0.70% | 0.619112 | 0.632780 | 2.21% |
| 3 | 0.671104 | 0.671543 | 0.07% | 0.697024 | 0.711552 | 2.08% |
| 4 | 0.744146 | 0.603770 | 18.86% | 0.834205 | 0.771266 | 7.54% |
| 5 | 0.669621 | 0.606737 | 9.39% | 0.786634 | 0.640530 | 18.57% |
| 6 | 0.611889 | 0.746173 | 21.95% | 0.664278 | 0.883250 | 32.96% |
| 7 | 0.605252 | 0.604969 | 0.05% | 0.678734 | 0.628065 | 7.47% |
| 8 | 0.687237 | 0.602511 | 12.33% | 0.837017 | 0.625540 | 25.27% |
| 9 | 0.659029 | 0.752580 | 14.20% | 0.768115 | 0.782461 | 1.87% |
| 10 | 0.615331 | 0.610311 | 0.82% | 1.308700 | 0.630819 | 51.80% |
| 11 | 0.667899 | 0.660802 | 1.06% | 0.772462 | 0.684541 | 11.38% |
| 12 | 0.672844 | 0.661007 | 1.76% | 0.812868 | 0.786424 | 3.25% |
| 13 | 0.661598 | 0.652973 | 1.30% | 0.688216 | 0.719642 | 4.57% |
| 14 | 0.660247 | 0.675909 | 2.37% | 0.681921 | 0.705180 | 3.41% |
| 15 | 0.677539 | 0.663251 | 2.11% | 0.720184 | 0.894824 | 24.25% |
| 16 | 0.670278 | 0.603339 | 9.99% | 0.696481 | 0.633398 | 9.06% |
| 17 | 0.603117 | 0.673409 | 11.65% | 0.656693 | 0.706181 | 7.54% |
| 18 | 0.608247 | 0.678714 | 11.59% | 0.632506 | 0.764808 | 20.92% |
| 19 | 0.685423 | 0.680325 | 0.74% | 0.702909 | 0.715741 | 1.83% |
| 20 | 0.617139 | 0.681131 | 10.37% | 0.634761 | 0.772737 | 21.74% |

[Immutable pair index and report hashes](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/clutter_filter/aa/pairs.json).

## Operator results

### beamscan_estimator

6/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/beamscan_estimator/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/beamscan_estimator/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 64.068350 | 64.088300 | 0.39% / 20% | 0.10% / 30% | PASS |
| baseline-large | double | median_ms | 63.168100 | 63.168350 | 0.01% / 5% | 0.00% / 10% | PASS |
| baseline-large | double | p90_ms | 63.181850 | 63.180850 | 0.04% / 5% | 0.02% / 10% | PASS |
| baseline-large | double | p99_ms | 63.215650 | 63.213300 | 0.10% / 5% | 0.05% / 10% | PASS |
| baseline-large | double | setup_ms | 9.525420 | 9.721175 | 7.69% / 15% | 5.32% / 25% | PASS |
| baseline-large | float | cold_ms | 3.683540 | 3.652010 | 6.13% / 20% | 4.26% / 30% | PASS |
| baseline-large | float | median_ms | 2.746305 | 2.746280 | 0.14% / 5% | 0.03% / 10% | PASS |
| baseline-large | float | p90_ms | 2.750895 | 2.750815 | 0.22% / 5% | 0.10% / 10% | PASS |
| baseline-large | float | p99_ms | 2.759530 | 2.759265 | 0.63% / 5% | 0.25% / 10% | PASS |
| baseline-large | float | setup_ms | 7.509155 | 7.466590 | 11.27% / 15% | 6.67% / 25% | PASS |
| baseline-medium | double | cold_ms | 2.097990 | 2.045085 | 12.00% / 20% | 6.92% / 30% | PASS |
| baseline-medium | double | median_ms | 1.307950 | 1.307870 | 0.05% / 5% | 0.03% / 10% | PASS |
| baseline-medium | double | p90_ms | 1.310385 | 1.310690 | 0.22% / 5% | 0.11% / 10% | PASS |
| baseline-medium | double | p99_ms | 1.318420 | 1.319530 | 0.75% / 5% | 0.37% / 10% | PASS |
| baseline-medium | double | setup_ms | 5.585605 | 5.550760 | 10.48% / 15% | 4.63% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.869734 | 0.863692 | 11.21% / 20% | 10.14% / 30% | PASS |
| baseline-medium | float | median_ms | 0.125959 | 0.126181 | 0.49% / 5% | 0.31% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.127337 | 0.127764 | 0.93% / 5% | 0.60% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.134580 | 0.134991 | 6.10% / 5% | 2.91% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 4.816550 | 4.887760 | 11.93% / 15% | 2.94% / 25% | PASS |
| baseline-small | double | cold_ms | 1.427020 | 1.428795 | 21.20% / 20% | 12.99% / 30% | FAIL: delta |
| baseline-small | double | median_ms | 0.169947 | 0.170127 | 0.48% / 5% | 0.25% / 10% | PASS |
| baseline-small | double | p90_ms | 0.172223 | 0.172446 | 1.81% / 5% | 0.53% / 10% | PASS |
| baseline-small | double | p99_ms | 0.184032 | 0.186809 | 5.17% / 5% | 3.96% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 4.193630 | 4.205555 | 3.41% / 15% | 1.43% / 25% | PASS |
| baseline-small | float | cold_ms | 1.341355 | 1.409245 | 31.93% / 20% | 13.00% / 30% | FAIL: delta |
| baseline-small | float | median_ms | 0.070891 | 0.070833 | 1.29% / 5% | 0.71% / 10% | PASS |
| baseline-small | float | p90_ms | 0.073388 | 0.072621 | 5.65% / 5% | 3.46% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.084880 | 0.082945 | 14.79% / 5% | 10.23% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 4.234255 | 4.222070 | 3.35% / 15% | 1.87% / 25% | PASS |

### cfar_detector

8/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/cfar_detector/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/cfar_detector/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 0.609366 | 0.609986 | 7.00% / 20% | 3.68% / 30% | PASS |
| baseline-large | double | median_ms | 0.314063 | 0.314018 | 4.17% / 5% | 1.19% / 10% | PASS |
| baseline-large | double | p90_ms | 0.319924 | 0.319742 | 2.55% / 5% | 0.66% / 10% | PASS |
| baseline-large | double | p99_ms | 0.342379 | 0.343349 | 3.76% / 5% | 1.73% / 10% | PASS |
| baseline-large | double | setup_ms | 6.783545 | 6.800105 | 12.17% / 15% | 6.92% / 25% | PASS |
| baseline-large | float | cold_ms | 0.429759 | 0.436347 | 25.79% / 20% | 7.08% / 30% | FAIL: delta |
| baseline-large | float | median_ms | 0.144392 | 0.146398 | 3.82% / 5% | 3.28% / 10% | PASS |
| baseline-large | float | p90_ms | 0.153106 | 0.153250 | 4.96% / 5% | 2.16% / 10% | PASS |
| baseline-large | float | p99_ms | 0.182455 | 0.186105 | 19.41% / 5% | 6.19% / 10% | FAIL: delta |
| baseline-large | float | setup_ms | 6.197400 | 6.310990 | 7.10% / 15% | 3.69% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.469440 | 0.465644 | 3.30% / 20% | 2.23% / 30% | PASS |
| baseline-medium | double | median_ms | 0.197028 | 0.193923 | 4.29% / 5% | 3.71% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.203112 | 0.201922 | 4.08% / 5% | 2.50% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.231982 | 0.232415 | 6.80% / 5% | 4.43% / 10% | FAIL: delta |
| baseline-medium | double | setup_ms | 6.173190 | 5.952755 | 10.15% / 15% | 5.43% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.391051 | 0.390018 | 4.08% / 20% | 2.52% / 30% | PASS |
| baseline-medium | float | median_ms | 0.116727 | 0.117097 | 1.51% / 5% | 1.04% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.124303 | 0.126293 | 7.22% / 5% | 5.19% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 0.153132 | 0.152655 | 25.58% / 5% | 7.17% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 5.774465 | 5.817080 | 8.22% / 15% | 4.38% / 25% | PASS |
| baseline-small | double | cold_ms | 0.395486 | 0.391556 | 4.40% / 20% | 2.50% / 30% | PASS |
| baseline-small | double | median_ms | 0.126063 | 0.126297 | 1.07% / 5% | 0.45% / 10% | PASS |
| baseline-small | double | p90_ms | 0.130422 | 0.130539 | 2.87% / 5% | 1.30% / 10% | PASS |
| baseline-small | double | p99_ms | 0.153579 | 0.156117 | 14.69% / 5% | 6.67% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 5.626810 | 5.602525 | 5.26% / 15% | 2.69% / 25% | PASS |
| baseline-small | float | cold_ms | 0.368893 | 0.367509 | 4.50% / 20% | 4.05% / 30% | PASS |
| baseline-small | float | median_ms | 0.101442 | 0.101557 | 2.12% / 5% | 0.58% / 10% | PASS |
| baseline-small | float | p90_ms | 0.104218 | 0.105006 | 5.78% / 5% | 4.03% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.131645 | 0.130124 | 25.23% / 5% | 13.26% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 5.724230 | 5.743540 | 5.17% / 15% | 3.01% / 25% | PASS |

### cluster_dbscan

1/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/cluster_dbscan/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/cluster_dbscan/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 11.153600 | 11.145300 | 0.31% / 20% | 0.14% / 30% | PASS |
| baseline-large | double | median_ms | 10.885000 | 10.885850 | 0.05% / 5% | 0.02% / 10% | PASS |
| baseline-large | double | p90_ms | 10.901600 | 10.905750 | 0.37% / 5% | 0.09% / 10% | PASS |
| baseline-large | double | p99_ms | 10.944150 | 10.939800 | 3.50% / 5% | 0.38% / 10% | PASS |
| baseline-large | double | setup_ms | 3.291035 | 3.302380 | 6.32% / 15% | 5.27% / 25% | PASS |
| baseline-large | float | cold_ms | 5.096590 | 5.110265 | 0.81% / 20% | 0.34% / 30% | PASS |
| baseline-large | float | median_ms | 4.829940 | 4.835565 | 0.69% / 5% | 0.15% / 10% | PASS |
| baseline-large | float | p90_ms | 4.851700 | 4.856210 | 0.83% / 5% | 0.20% / 10% | PASS |
| baseline-large | float | p99_ms | 6.027135 | 6.041170 | 2.75% / 5% | 1.34% / 10% | PASS |
| baseline-large | float | setup_ms | 3.511720 | 3.550210 | 8.15% / 15% | 4.99% / 25% | PASS |
| baseline-medium | double | cold_ms | 5.963615 | 5.956460 | 0.52% / 20% | 0.23% / 30% | PASS |
| baseline-medium | double | median_ms | 5.688595 | 5.687790 | 0.13% / 5% | 0.08% / 10% | PASS |
| baseline-medium | double | p90_ms | 5.709020 | 5.711750 | 0.62% / 5% | 0.15% / 10% | PASS |
| baseline-medium | double | p99_ms | 5.741315 | 5.741400 | 0.77% / 5% | 0.30% / 10% | PASS |
| baseline-medium | double | setup_ms | 3.548410 | 3.436395 | 7.05% / 15% | 5.90% / 25% | PASS |
| baseline-medium | float | cold_ms | 3.004050 | 3.010210 | 1.01% / 20% | 0.59% / 30% | PASS |
| baseline-medium | float | median_ms | 2.744875 | 2.743915 | 0.34% / 5% | 0.22% / 10% | PASS |
| baseline-medium | float | p90_ms | 2.761255 | 2.760975 | 0.85% / 5% | 0.25% / 10% | PASS |
| baseline-medium | float | p99_ms | 2.792410 | 2.790950 | 2.33% / 5% | 0.81% / 10% | PASS |
| baseline-medium | float | setup_ms | 3.475780 | 3.529100 | 6.50% / 15% | 5.02% / 25% | PASS |
| baseline-small | double | cold_ms | 1.356540 | 1.358480 | 1.77% / 20% | 1.20% / 30% | PASS |
| baseline-small | double | median_ms | 1.096265 | 1.096120 | 0.77% / 5% | 0.22% / 10% | PASS |
| baseline-small | double | p90_ms | 1.106165 | 1.105475 | 0.77% / 5% | 0.42% / 10% | PASS |
| baseline-small | double | p99_ms | 1.135975 | 1.133870 | 5.41% / 5% | 1.09% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 3.534080 | 3.508810 | 7.61% / 15% | 5.03% / 25% | PASS |
| baseline-small | float | cold_ms | 0.927566 | 0.921758 | 2.96% / 20% | 1.91% / 30% | PASS |
| baseline-small | float | median_ms | 0.657363 | 0.656227 | 1.86% / 5% | 0.52% / 10% | PASS |
| baseline-small | float | p90_ms | 0.668392 | 0.667323 | 1.80% / 5% | 1.04% / 10% | PASS |
| baseline-small | float | p99_ms | 0.701801 | 0.697102 | 3.66% / 5% | 1.65% / 10% | PASS |
| baseline-small | float | setup_ms | 3.489770 | 3.437605 | 6.97% / 15% | 4.35% / 25% | PASS |

### clutter_filter

19/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/clutter_filter/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/clutter_filter/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 0.751557 | 0.748608 | 8.00% / 20% | 4.94% / 30% | PASS |
| baseline-large | double | median_ms | 0.664748 | 0.662129 | 14.66% / 5% | 9.70% / 10% | FAIL: delta |
| baseline-large | double | p90_ms | 0.699966 | 0.713646 | 26.04% / 5% | 15.98% / 10% | FAIL: delta, IQR |
| baseline-large | double | p99_ms | 1.650885 | 1.678795 | 12.44% / 5% | 6.72% / 10% | FAIL: delta |
| baseline-large | double | setup_ms | 4.127565 | 4.121990 | 10.14% / 15% | 6.17% / 25% | PASS |
| baseline-large | float | cold_ms | 0.693964 | 0.687507 | 9.32% / 20% | 4.25% / 30% | PASS |
| baseline-large | float | median_ms | 0.599957 | 0.595703 | 16.07% / 5% | 7.03% / 10% | FAIL: delta |
| baseline-large | float | p90_ms | 0.640516 | 0.632032 | 22.14% / 5% | 7.82% / 10% | FAIL: delta |
| baseline-large | float | p99_ms | 1.605380 | 1.628745 | 15.53% / 5% | 7.00% / 10% | FAIL: delta |
| baseline-large | float | setup_ms | 3.624680 | 3.591995 | 9.00% / 15% | 3.22% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.693960 | 0.682338 | 9.87% / 20% | 5.39% / 30% | PASS |
| baseline-medium | double | median_ms | 0.600939 | 0.595300 | 10.59% / 5% | 8.07% / 10% | FAIL: delta |
| baseline-medium | double | p90_ms | 0.640772 | 0.618046 | 17.97% / 5% | 9.78% / 10% | FAIL: delta |
| baseline-medium | double | p99_ms | 1.581770 | 1.610525 | 11.27% / 5% | 8.67% / 10% | FAIL: delta |
| baseline-medium | double | setup_ms | 3.438360 | 3.427230 | 7.18% / 15% | 2.84% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.676743 | 0.685496 | 12.25% / 20% | 6.74% / 30% | PASS |
| baseline-medium | float | median_ms | 0.561226 | 0.560940 | 15.29% / 5% | 7.29% / 10% | FAIL: delta |
| baseline-medium | float | p90_ms | 0.596170 | 0.587671 | 15.37% / 5% | 7.05% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 1.571815 | 1.577100 | 13.64% / 5% | 9.54% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 3.283380 | 3.293300 | 7.40% / 15% | 3.02% / 25% | PASS |
| baseline-small | double | cold_ms | 0.759807 | 0.752468 | 17.00% / 20% | 8.17% / 30% | PASS |
| baseline-small | double | median_ms | 0.545010 | 0.539679 | 15.31% / 5% | 2.99% / 10% | FAIL: delta |
| baseline-small | double | p90_ms | 0.577738 | 0.571308 | 16.57% / 5% | 6.87% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 1.462750 | 1.447410 | 13.47% / 5% | 6.07% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 3.247580 | 3.166525 | 7.41% / 15% | 3.99% / 25% | PASS |
| baseline-small | float | cold_ms | 0.756376 | 0.755529 | 45.70% / 20% | 6.50% / 30% | FAIL: delta |
| baseline-small | float | median_ms | 0.539816 | 0.544258 | 25.47% / 5% | 25.42% / 10% | FAIL: delta, IQR |
| baseline-small | float | p90_ms | 0.580783 | 0.585973 | 20.34% / 5% | 18.04% / 10% | FAIL: delta, IQR |
| baseline-small | float | p99_ms | 1.459945 | 1.490380 | 12.76% / 5% | 8.65% / 10% | FAIL: delta |
| baseline-small | float | setup_ms | 3.185220 | 3.191765 | 6.25% / 15% | 3.20% / 25% | PASS |

### ddm_dense_demultiplex

6/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_dense_demultiplex/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_dense_demultiplex/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 2.833290 | 2.836990 | 0.97% / 20% | 0.50% / 30% | PASS |
| baseline-large | double | median_ms | 2.634670 | 2.635255 | 0.58% / 5% | 0.37% / 10% | PASS |
| baseline-large | double | p90_ms | 2.740470 | 2.739315 | 0.83% / 5% | 0.55% / 10% | PASS |
| baseline-large | double | p99_ms | 2.765960 | 2.769615 | 0.76% / 5% | 0.28% / 10% | PASS |
| baseline-large | double | setup_ms | 68.177050 | 68.250550 | 1.20% / 15% | 0.74% / 25% | PASS |
| baseline-large | float | cold_ms | 1.388460 | 1.385730 | 4.60% / 20% | 0.64% / 30% | PASS |
| baseline-large | float | median_ms | 1.190740 | 1.185780 | 1.55% / 5% | 1.15% / 10% | PASS |
| baseline-large | float | p90_ms | 1.285940 | 1.289140 | 0.94% / 5% | 0.53% / 10% | PASS |
| baseline-large | float | p99_ms | 1.330935 | 1.326785 | 2.37% / 5% | 1.08% / 10% | PASS |
| baseline-large | float | setup_ms | 36.992250 | 37.100950 | 1.78% / 15% | 1.00% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.543075 | 0.544094 | 1.89% / 20% | 1.06% / 30% | PASS |
| baseline-medium | double | median_ms | 0.352988 | 0.353577 | 1.07% / 5% | 0.89% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.362752 | 0.362322 | 2.86% / 5% | 1.30% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.388031 | 0.386984 | 2.97% / 5% | 2.45% / 10% | PASS |
| baseline-medium | double | setup_ms | 9.190890 | 9.384085 | 10.92% / 15% | 7.38% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.334154 | 0.333361 | 1.54% / 20% | 0.85% / 30% | PASS |
| baseline-medium | float | median_ms | 0.100391 | 0.100451 | 1.77% / 5% | 1.23% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.108330 | 0.109892 | 10.45% / 5% | 9.87% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 0.129724 | 0.131332 | 25.01% / 5% | 22.01% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 6.202040 | 6.223930 | 12.43% / 15% | 4.27% / 25% | PASS |
| baseline-small | double | cold_ms | 0.263311 | 0.265615 | 2.58% / 20% | 1.68% / 30% | PASS |
| baseline-small | double | median_ms | 0.032997 | 0.033030 | 1.18% / 5% | 0.65% / 10% | PASS |
| baseline-small | double | p90_ms | 0.034237 | 0.034155 | 10.33% / 5% | 2.05% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.036609 | 0.035448 | 25.89% / 5% | 26.86% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.095160 | 3.173910 | 13.89% / 15% | 7.29% / 25% | PASS |
| baseline-small | float | cold_ms | 0.267120 | 0.267713 | 3.97% / 20% | 2.04% / 30% | PASS |
| baseline-small | float | median_ms | 0.032542 | 0.032433 | 1.66% / 5% | 0.99% / 10% | PASS |
| baseline-small | float | p90_ms | 0.033522 | 0.033425 | 9.40% / 5% | 1.92% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.036818 | 0.037402 | 32.80% / 5% | 21.61% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 3.104435 | 3.063275 | 9.51% / 15% | 5.85% / 25% | PASS |

### ddm_power_fold

6/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_power_fold/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_power_fold/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 1.962160 | 1.966575 | 9.44% / 20% | 2.36% / 30% | PASS |
| baseline-large | double | median_ms | 1.754135 | 1.761855 | 1.01% / 5% | 0.70% / 10% | PASS |
| baseline-large | double | p90_ms | 1.909290 | 1.905395 | 1.36% / 5% | 0.48% / 10% | PASS |
| baseline-large | double | p99_ms | 1.955875 | 1.947180 | 1.71% / 5% | 0.85% / 10% | PASS |
| baseline-large | double | setup_ms | 11.950400 | 11.885000 | 5.76% / 15% | 2.46% / 25% | PASS |
| baseline-large | float | cold_ms | 1.113635 | 1.111835 | 1.68% / 20% | 0.70% / 30% | PASS |
| baseline-large | float | median_ms | 0.854625 | 0.853712 | 1.21% / 5% | 0.52% / 10% | PASS |
| baseline-large | float | p90_ms | 0.964530 | 0.965169 | 2.58% / 5% | 1.84% / 10% | PASS |
| baseline-large | float | p99_ms | 0.996568 | 0.999432 | 1.93% / 5% | 0.53% / 10% | PASS |
| baseline-large | float | setup_ms | 8.429740 | 8.384550 | 11.11% / 15% | 6.26% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.421038 | 0.420629 | 2.61% / 20% | 1.69% / 30% | PASS |
| baseline-medium | double | median_ms | 0.235743 | 0.236039 | 3.40% / 5% | 1.35% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.243999 | 0.247790 | 3.63% / 5% | 2.82% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.269871 | 0.268554 | 3.62% / 5% | 3.65% / 10% | PASS |
| baseline-medium | double | setup_ms | 4.594930 | 4.659485 | 14.12% / 15% | 4.06% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.311384 | 0.311310 | 2.31% / 20% | 1.10% / 30% | PASS |
| baseline-medium | float | median_ms | 0.068296 | 0.068275 | 1.78% / 5% | 1.29% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.070548 | 0.076167 | 11.88% / 5% | 9.88% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 0.079968 | 0.086935 | 23.50% / 5% | 21.33% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 3.906140 | 3.928235 | 11.20% / 15% | 4.05% / 25% | PASS |
| baseline-small | double | cold_ms | 0.265032 | 0.263463 | 1.76% / 20% | 1.12% / 30% | PASS |
| baseline-small | double | median_ms | 0.033626 | 0.033650 | 1.45% / 5% | 0.89% / 10% | PASS |
| baseline-small | double | p90_ms | 0.034570 | 0.034700 | 14.74% / 5% | 2.76% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.038215 | 0.036572 | 32.88% / 5% | 19.69% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.021190 | 2.981210 | 8.19% / 15% | 3.53% / 25% | PASS |
| baseline-small | float | cold_ms | 0.261454 | 0.262602 | 2.87% / 20% | 1.35% / 30% | PASS |
| baseline-small | float | median_ms | 0.031560 | 0.031500 | 3.93% / 5% | 0.91% / 10% | PASS |
| baseline-small | float | p90_ms | 0.032363 | 0.032433 | 6.60% / 5% | 1.21% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.036035 | 0.035110 | 22.82% / 5% | 15.85% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 2.794855 | 2.777645 | 11.95% / 15% | 5.97% / 25% | PASS |

### ddm_sparse_demultiplex

7/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_sparse_demultiplex/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_sparse_demultiplex/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 1.948530 | 1.943000 | 2.78% / 20% | 0.98% / 30% | PASS |
| baseline-large | double | median_ms | 1.626965 | 1.626060 | 0.41% / 5% | 0.19% / 10% | PASS |
| baseline-large | double | p90_ms | 1.732850 | 1.730900 | 2.10% / 5% | 0.95% / 10% | PASS |
| baseline-large | double | p99_ms | 1.755345 | 1.755445 | 1.24% / 5% | 0.67% / 10% | PASS |
| baseline-large | double | setup_ms | 18.277050 | 18.280500 | 4.47% / 15% | 2.68% / 25% | PASS |
| baseline-large | float | cold_ms | 1.313990 | 1.311800 | 3.53% / 20% | 0.86% / 30% | PASS |
| baseline-large | float | median_ms | 0.999622 | 1.001685 | 0.61% / 5% | 0.39% / 10% | PASS |
| baseline-large | float | p90_ms | 1.039595 | 1.049865 | 1.91% / 5% | 1.36% / 10% | PASS |
| baseline-large | float | p99_ms | 1.076295 | 1.080120 | 2.06% / 5% | 1.00% / 10% | PASS |
| baseline-large | float | setup_ms | 12.226150 | 12.172150 | 7.69% / 15% | 1.63% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.454935 | 0.456510 | 2.49% / 20% | 1.53% / 30% | PASS |
| baseline-medium | double | median_ms | 0.259928 | 0.260019 | 2.24% / 5% | 0.67% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.276402 | 0.278701 | 3.27% / 5% | 1.91% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.300403 | 0.300165 | 5.35% / 5% | 2.77% / 10% | FAIL: delta |
| baseline-medium | double | setup_ms | 6.745500 | 6.694680 | 8.42% / 15% | 3.04% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.396468 | 0.396732 | 3.75% / 20% | 1.98% / 30% | PASS |
| baseline-medium | float | median_ms | 0.133368 | 0.133463 | 1.00% / 5% | 0.77% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.137224 | 0.137152 | 5.66% / 5% | 2.59% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 0.156927 | 0.156326 | 10.99% / 5% | 3.94% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 5.783515 | 5.857300 | 6.87% / 15% | 1.53% / 25% | PASS |
| baseline-small | double | cold_ms | 0.329593 | 0.327213 | 3.67% / 20% | 1.95% / 30% | PASS |
| baseline-small | double | median_ms | 0.064405 | 0.064426 | 1.81% / 5% | 0.91% / 10% | PASS |
| baseline-small | double | p90_ms | 0.066398 | 0.068751 | 12.95% / 5% | 6.95% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.081489 | 0.087068 | 17.89% / 5% | 13.49% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 4.804575 | 4.847935 | 4.67% / 15% | 3.76% / 25% | PASS |
| baseline-small | float | cold_ms | 0.324731 | 0.330046 | 5.61% / 20% | 3.32% / 30% | PASS |
| baseline-small | float | median_ms | 0.063912 | 0.063701 | 2.08% / 5% | 1.08% / 10% | PASS |
| baseline-small | float | p90_ms | 0.065995 | 0.066118 | 8.51% / 5% | 2.82% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.075004 | 0.080909 | 21.18% / 5% | 17.03% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 4.620810 | 4.714925 | 6.39% / 15% | 4.32% / 25% | PASS |

### doppler_notch

6/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/doppler_notch/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/doppler_notch/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 0.675784 | 0.676782 | 1.61% / 20% | 0.71% / 30% | PASS |
| baseline-large | double | median_ms | 0.454009 | 0.454032 | 0.35% / 5% | 0.27% / 10% | PASS |
| baseline-large | double | p90_ms | 0.458026 | 0.458087 | 0.51% / 5% | 0.38% / 10% | PASS |
| baseline-large | double | p99_ms | 0.472183 | 0.470666 | 2.34% / 5% | 1.14% / 10% | PASS |
| baseline-large | double | setup_ms | 5.417640 | 5.384360 | 6.88% / 15% | 2.79% / 25% | PASS |
| baseline-large | float | cold_ms | 0.664519 | 0.661908 | 1.32% / 20% | 0.65% / 30% | PASS |
| baseline-large | float | median_ms | 0.439395 | 0.439194 | 0.22% / 5% | 0.12% / 10% | PASS |
| baseline-large | float | p90_ms | 0.441681 | 0.441658 | 1.17% / 5% | 0.23% / 10% | PASS |
| baseline-large | float | p99_ms | 0.450014 | 0.449632 | 2.13% / 5% | 1.05% / 10% | PASS |
| baseline-large | float | setup_ms | 4.743380 | 4.697970 | 7.20% / 15% | 4.06% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.311852 | 0.313445 | 3.28% / 20% | 1.43% / 30% | PASS |
| baseline-medium | double | median_ms | 0.086356 | 0.086264 | 1.59% / 5% | 0.44% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.087612 | 0.087457 | 4.77% / 5% | 0.92% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.093412 | 0.092303 | 8.92% / 5% | 9.53% / 10% | FAIL: delta |
| baseline-medium | double | setup_ms | 3.948165 | 3.950440 | 5.98% / 15% | 4.83% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.312523 | 0.312181 | 2.55% / 20% | 0.99% / 30% | PASS |
| baseline-medium | float | median_ms | 0.087902 | 0.087896 | 0.65% / 5% | 0.33% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.089274 | 0.089158 | 1.67% / 5% | 0.60% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.094444 | 0.094206 | 10.47% / 5% | 4.83% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 3.679640 | 3.637660 | 7.21% / 15% | 4.72% / 25% | PASS |
| baseline-small | double | cold_ms | 0.266533 | 0.265218 | 2.78% / 20% | 1.73% / 30% | PASS |
| baseline-small | double | median_ms | 0.041585 | 0.041468 | 1.62% / 5% | 1.11% / 10% | PASS |
| baseline-small | double | p90_ms | 0.042480 | 0.043066 | 7.67% / 5% | 3.73% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.044070 | 0.051641 | 23.80% / 5% | 19.44% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.344785 | 3.260200 | 9.43% / 15% | 5.22% / 25% | PASS |
| baseline-small | float | cold_ms | 0.264282 | 0.264625 | 2.89% / 20% | 1.29% / 30% | PASS |
| baseline-small | float | median_ms | 0.041692 | 0.041590 | 2.60% / 5% | 1.03% / 10% | PASS |
| baseline-small | float | p90_ms | 0.043180 | 0.042842 | 9.47% / 5% | 2.65% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.048443 | 0.046105 | 24.36% / 5% | 15.02% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 3.223805 | 3.330870 | 11.05% / 15% | 5.13% / 25% | PASS |

### esprit_estimator

13/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/esprit_estimator/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/esprit_estimator/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 33.699250 | 34.029750 | 3.39% / 20% | 2.38% / 30% | PASS |
| baseline-large | double | median_ms | 34.082700 | 34.018500 | 3.44% / 5% | 1.56% / 10% | PASS |
| baseline-large | double | p90_ms | 35.307300 | 35.366000 | 1.60% / 5% | 1.27% / 10% | PASS |
| baseline-large | double | p99_ms | 35.687700 | 35.857200 | 1.83% / 5% | 1.33% / 10% | PASS |
| baseline-large | double | setup_ms | 7.494140 | 7.127505 | 9.50% / 15% | 7.05% / 25% | PASS |
| baseline-large | float | cold_ms | 6.712890 | 6.813900 | 9.81% / 20% | 8.12% / 30% | PASS |
| baseline-large | float | median_ms | 6.865080 | 6.952145 | 9.74% / 5% | 5.72% / 10% | FAIL: delta |
| baseline-large | float | p90_ms | 7.604710 | 7.636375 | 9.75% / 5% | 5.82% / 10% | FAIL: delta |
| baseline-large | float | p99_ms | 8.247525 | 8.331940 | 8.11% / 5% | 6.89% / 10% | FAIL: delta |
| baseline-large | float | setup_ms | 6.249600 | 6.241830 | 4.80% / 15% | 2.52% / 25% | PASS |
| baseline-medium | double | cold_ms | 17.933250 | 17.679400 | 5.84% / 20% | 4.39% / 30% | PASS |
| baseline-medium | double | median_ms | 17.587450 | 17.464950 | 3.81% / 5% | 3.22% / 10% | PASS |
| baseline-medium | double | p90_ms | 18.950350 | 18.907050 | 5.94% / 5% | 3.26% / 10% | FAIL: delta |
| baseline-medium | double | p99_ms | 19.374650 | 19.208950 | 3.83% / 5% | 2.53% / 10% | PASS |
| baseline-medium | double | setup_ms | 5.599625 | 5.581985 | 6.25% / 15% | 2.64% / 25% | PASS |
| baseline-medium | float | cold_ms | 10.418450 | 10.421750 | 7.28% / 20% | 5.67% / 30% | PASS |
| baseline-medium | float | median_ms | 10.594350 | 10.559900 | 6.85% / 5% | 5.35% / 10% | FAIL: delta |
| baseline-medium | float | p90_ms | 11.099850 | 11.015900 | 5.97% / 5% | 3.90% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 11.468900 | 11.402600 | 6.54% / 5% | 4.31% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 5.501355 | 5.442540 | 6.93% / 15% | 2.72% / 25% | PASS |
| baseline-small | double | cold_ms | 4.257915 | 4.210145 | 15.79% / 20% | 13.93% / 30% | PASS |
| baseline-small | double | median_ms | 3.918940 | 4.030185 | 17.25% / 5% | 13.20% / 10% | FAIL: delta, IQR |
| baseline-small | double | p90_ms | 4.665220 | 4.786790 | 14.03% / 5% | 9.63% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 5.353525 | 5.361040 | 12.22% / 5% | 5.96% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 5.444795 | 5.439370 | 5.09% / 15% | 2.29% / 25% | PASS |
| baseline-small | float | cold_ms | 4.381550 | 4.466940 | 19.96% / 20% | 10.24% / 30% | PASS |
| baseline-small | float | median_ms | 3.972650 | 4.210220 | 19.44% / 5% | 8.65% / 10% | FAIL: delta |
| baseline-small | float | p90_ms | 4.454405 | 5.043610 | 16.36% / 5% | 12.74% / 10% | FAIL: delta, IQR |
| baseline-small | float | p99_ms | 5.173775 | 5.350745 | 14.97% / 5% | 5.54% / 10% | FAIL: delta |
| baseline-small | float | setup_ms | 5.418585 | 5.532330 | 4.92% / 15% | 2.45% / 25% | PASS |

### feature_assembly

12/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/feature_assembly/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/feature_assembly/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 0.300566 | 0.300172 | 4.53% / 20% | 2.60% / 30% | PASS |
| baseline-large | double | median_ms | 0.056546 | 0.056491 | 2.25% / 5% | 1.09% / 10% | PASS |
| baseline-large | double | p90_ms | 0.058900 | 0.058783 | 14.44% / 5% | 6.92% / 10% | FAIL: delta |
| baseline-large | double | p99_ms | 0.072896 | 0.075834 | 27.61% / 5% | 21.38% / 10% | FAIL: delta, IQR |
| baseline-large | double | setup_ms | 3.897570 | 3.721550 | 9.52% / 15% | 6.04% / 25% | PASS |
| baseline-large | float | cold_ms | 0.300469 | 0.300269 | 3.88% / 20% | 1.31% / 30% | PASS |
| baseline-large | float | median_ms | 0.056678 | 0.056599 | 1.65% / 5% | 1.15% / 10% | PASS |
| baseline-large | float | p90_ms | 0.058161 | 0.059875 | 11.86% / 5% | 4.34% / 10% | FAIL: delta |
| baseline-large | float | p99_ms | 0.066228 | 0.073826 | 22.69% / 5% | 16.10% / 10% | FAIL: delta, IQR |
| baseline-large | float | setup_ms | 3.779610 | 3.783525 | 7.22% / 15% | 5.09% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.297463 | 0.295106 | 5.80% / 20% | 3.26% / 30% | PASS |
| baseline-medium | double | median_ms | 0.055785 | 0.055854 | 0.97% / 5% | 0.76% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.057631 | 0.057523 | 6.77% / 5% | 3.00% / 10% | FAIL: delta |
| baseline-medium | double | p99_ms | 0.064954 | 0.065204 | 27.39% / 5% | 21.30% / 10% | FAIL: delta, IQR |
| baseline-medium | double | setup_ms | 3.677415 | 3.694890 | 7.45% / 15% | 4.66% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.296042 | 0.293741 | 4.41% / 20% | 2.46% / 30% | PASS |
| baseline-medium | float | median_ms | 0.056002 | 0.056171 | 2.14% / 5% | 0.80% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.057507 | 0.058355 | 11.44% / 5% | 3.82% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 0.066180 | 0.074188 | 24.22% / 5% | 18.34% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 3.665545 | 3.636235 | 6.37% / 15% | 5.93% / 25% | PASS |
| baseline-small | double | cold_ms | 0.295106 | 0.296709 | 5.82% / 20% | 3.08% / 30% | PASS |
| baseline-small | double | median_ms | 0.055627 | 0.055724 | 3.17% / 5% | 0.69% / 10% | PASS |
| baseline-small | double | p90_ms | 0.057625 | 0.057785 | 17.12% / 5% | 4.22% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.064755 | 0.074295 | 28.01% / 5% | 18.76% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.638920 | 3.606640 | 6.80% / 15% | 5.39% / 25% | PASS |
| baseline-small | float | cold_ms | 0.296769 | 0.294117 | 4.56% / 20% | 2.28% / 30% | PASS |
| baseline-small | float | median_ms | 0.055600 | 0.055806 | 2.46% / 5% | 1.30% / 10% | PASS |
| baseline-small | float | p90_ms | 0.057605 | 0.057618 | 10.15% / 5% | 2.40% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.070247 | 0.066142 | 21.54% / 5% | 17.43% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 3.702635 | 3.579295 | 6.38% / 15% | 5.06% / 25% | PASS |

### fmcw_doppler_processor

12/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_doppler_processor/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_doppler_processor/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 35.204700 | 35.237650 | 3.49% / 20% | 1.10% / 30% | PASS |
| baseline-large | double | median_ms | 0.148439 | 0.148556 | 0.39% / 5% | 0.26% / 10% | PASS |
| baseline-large | double | p90_ms | 0.150477 | 0.150470 | 1.85% / 5% | 0.99% / 10% | PASS |
| baseline-large | double | p99_ms | 0.161440 | 0.162644 | 6.18% / 5% | 3.11% / 10% | FAIL: delta |
| baseline-large | double | setup_ms | 4.823795 | 4.771355 | 12.20% / 15% | 4.26% / 25% | PASS |
| baseline-large | float | cold_ms | 39.665900 | 39.671000 | 3.45% / 20% | 2.71% / 30% | PASS |
| baseline-large | float | median_ms | 0.063715 | 0.063725 | 1.00% / 5% | 0.49% / 10% | PASS |
| baseline-large | float | p90_ms | 0.065531 | 0.065802 | 10.64% / 5% | 4.45% / 10% | FAIL: delta |
| baseline-large | float | p99_ms | 0.079856 | 0.077583 | 26.53% / 5% | 17.35% / 10% | FAIL: delta, IQR |
| baseline-large | float | setup_ms | 4.233395 | 4.231365 | 9.05% / 15% | 1.78% / 25% | PASS |
| baseline-medium | double | cold_ms | 30.158700 | 30.283200 | 3.73% / 20% | 2.45% / 30% | PASS |
| baseline-medium | double | median_ms | 0.064636 | 0.064818 | 2.22% / 5% | 0.51% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.066192 | 0.066323 | 11.18% / 5% | 6.08% / 10% | FAIL: delta |
| baseline-medium | double | p99_ms | 0.078400 | 0.073299 | 14.96% / 5% | 12.96% / 10% | FAIL: delta, IQR |
| baseline-medium | double | setup_ms | 4.043290 | 4.044455 | 4.78% / 15% | 2.15% / 25% | PASS |
| baseline-medium | float | cold_ms | 26.426650 | 26.417250 | 2.05% / 20% | 1.77% / 30% | PASS |
| baseline-medium | float | median_ms | 0.033028 | 0.033454 | 2.45% / 5% | 1.49% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.034329 | 0.034731 | 33.35% / 5% | 4.82% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 0.045194 | 0.046807 | 82.78% / 5% | 59.43% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 3.927245 | 3.932880 | 2.61% / 15% | 2.20% / 25% | PASS |
| baseline-small | double | cold_ms | 0.270375 | 0.270779 | 3.57% / 20% | 1.21% / 30% | PASS |
| baseline-small | double | median_ms | 0.035180 | 0.035266 | 9.36% / 5% | 1.66% / 10% | FAIL: delta |
| baseline-small | double | p90_ms | 0.051637 | 0.051542 | 7.32% / 5% | 1.52% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.055386 | 0.061196 | 24.20% / 5% | 16.17% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.974540 | 3.973490 | 3.54% / 15% | 1.24% / 25% | PASS |
| baseline-small | float | cold_ms | 0.265075 | 0.265769 | 3.56% / 20% | 1.29% / 30% | PASS |
| baseline-small | float | median_ms | 0.033118 | 0.032931 | 1.96% / 5% | 1.20% / 10% | PASS |
| baseline-small | float | p90_ms | 0.034110 | 0.034375 | 36.24% / 5% | 3.32% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.039874 | 0.055342 | 68.36% / 5% | 54.14% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 3.980195 | 3.970620 | 3.84% / 15% | 1.97% / 25% | PASS |

### fmcw_range_doppler_processor

8/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_range_doppler_processor/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_range_doppler_processor/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 85.345200 | 85.819000 | 5.95% / 20% | 5.22% / 30% | PASS |
| baseline-large | double | median_ms | 0.275422 | 0.275411 | 0.24% / 5% | 0.11% / 10% | PASS |
| baseline-large | double | p90_ms | 0.281174 | 0.280998 | 1.84% / 5% | 1.07% / 10% | PASS |
| baseline-large | double | p99_ms | 0.294495 | 0.293652 | 2.90% / 5% | 1.18% / 10% | PASS |
| baseline-large | double | setup_ms | 5.694840 | 5.752765 | 10.88% / 15% | 3.85% / 25% | PASS |
| baseline-large | float | cold_ms | 80.079500 | 80.010500 | 3.47% / 20% | 3.31% / 30% | PASS |
| baseline-large | float | median_ms | 0.074394 | 0.074308 | 1.28% / 5% | 0.85% / 10% | PASS |
| baseline-large | float | p90_ms | 0.078398 | 0.076257 | 9.64% / 5% | 7.49% / 10% | FAIL: delta |
| baseline-large | float | p99_ms | 0.094038 | 0.091170 | 20.21% / 5% | 14.47% / 10% | FAIL: delta, IQR |
| baseline-large | float | setup_ms | 5.130480 | 5.133740 | 7.67% / 15% | 2.27% / 25% | PASS |
| baseline-medium | double | cold_ms | 55.155600 | 55.741450 | 3.89% / 20% | 3.23% / 30% | PASS |
| baseline-medium | double | median_ms | 0.099169 | 0.099246 | 0.73% / 5% | 0.30% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.101687 | 0.101173 | 4.55% / 5% | 2.52% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.112223 | 0.110556 | 8.56% / 5% | 7.75% / 10% | FAIL: delta |
| baseline-medium | double | setup_ms | 4.969250 | 4.941635 | 5.65% / 15% | 2.07% / 25% | PASS |
| baseline-medium | float | cold_ms | 47.540600 | 47.328300 | 3.26% / 20% | 3.18% / 30% | PASS |
| baseline-medium | float | median_ms | 0.051079 | 0.051033 | 2.27% / 5% | 1.10% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.052668 | 0.052481 | 3.43% / 5% | 1.78% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.062749 | 0.067456 | 29.97% / 5% | 21.40% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 4.803575 | 4.861955 | 7.47% / 15% | 2.18% / 25% | PASS |
| baseline-small | double | cold_ms | 0.290413 | 0.290617 | 3.83% / 20% | 2.49% / 30% | PASS |
| baseline-small | double | median_ms | 0.062170 | 0.054695 | 27.57% / 5% | 23.44% / 10% | FAIL: delta, IQR |
| baseline-small | double | p90_ms | 0.070208 | 0.069778 | 6.65% / 5% | 1.98% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.081717 | 0.076016 | 17.40% / 5% | 15.49% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 4.831455 | 4.860585 | 2.93% / 15% | 1.67% / 25% | PASS |
| baseline-small | float | cold_ms | 0.274886 | 0.273186 | 3.57% / 20% | 2.29% / 30% | PASS |
| baseline-small | float | median_ms | 0.050709 | 0.050817 | 2.39% / 5% | 1.23% / 10% | PASS |
| baseline-small | float | p90_ms | 0.052159 | 0.052176 | 3.13% / 5% | 1.68% / 10% | PASS |
| baseline-small | float | p99_ms | 0.060836 | 0.061395 | 19.33% / 5% | 12.98% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 4.848345 | 4.833800 | 3.43% / 15% | 1.91% / 25% | PASS |

### fmcw_range_processor

6/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_range_processor/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_range_processor/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 53.682900 | 54.152100 | 5.26% / 20% | 4.93% / 30% | PASS |
| baseline-large | double | median_ms | 0.157350 | 0.157438 | 1.02% / 5% | 0.48% / 10% | PASS |
| baseline-large | double | p90_ms | 0.158560 | 0.158759 | 2.83% / 5% | 0.86% / 10% | PASS |
| baseline-large | double | p99_ms | 0.167341 | 0.168275 | 5.16% / 5% | 2.56% / 10% | FAIL: delta |
| baseline-large | double | setup_ms | 4.832045 | 4.797025 | 8.75% / 15% | 3.37% / 25% | PASS |
| baseline-large | float | cold_ms | 43.190500 | 43.285900 | 3.78% / 20% | 3.32% / 30% | PASS |
| baseline-large | float | median_ms | 0.027618 | 0.027778 | 1.83% / 5% | 0.92% / 10% | PASS |
| baseline-large | float | p90_ms | 0.028605 | 0.028628 | 3.41% / 5% | 1.19% / 10% | PASS |
| baseline-large | float | p99_ms | 0.032760 | 0.030164 | 63.46% / 5% | 24.24% / 10% | FAIL: delta, IQR |
| baseline-large | float | setup_ms | 4.254270 | 4.283305 | 4.48% / 15% | 3.01% / 25% | PASS |
| baseline-medium | double | cold_ms | 27.256650 | 27.590850 | 4.00% / 20% | 3.24% / 30% | PASS |
| baseline-medium | double | median_ms | 0.061984 | 0.062150 | 1.17% / 5% | 0.53% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.063269 | 0.063359 | 4.14% / 5% | 2.61% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.070160 | 0.074277 | 14.61% / 5% | 12.38% / 10% | FAIL: delta, IQR |
| baseline-medium | double | setup_ms | 4.078995 | 4.040950 | 4.07% / 15% | 3.04% / 25% | PASS |
| baseline-medium | float | cold_ms | 23.258200 | 23.186650 | 2.30% / 20% | 2.42% / 30% | PASS |
| baseline-medium | float | median_ms | 0.027178 | 0.027216 | 1.45% / 5% | 0.83% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.027960 | 0.027809 | 3.55% / 5% | 1.77% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.037692 | 0.028731 | 46.97% / 5% | 48.46% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 3.940360 | 3.937250 | 4.23% / 15% | 1.86% / 25% | PASS |
| baseline-small | double | cold_ms | 0.270358 | 0.270599 | 2.20% / 20% | 1.74% / 30% | PASS |
| baseline-small | double | median_ms | 0.045537 | 0.045609 | 1.56% / 5% | 0.81% / 10% | PASS |
| baseline-small | double | p90_ms | 0.046772 | 0.047663 | 4.97% / 5% | 1.88% / 10% | PASS |
| baseline-small | double | p99_ms | 0.050767 | 0.055866 | 14.80% / 5% | 12.87% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.990395 | 4.045190 | 4.54% / 15% | 3.07% / 25% | PASS |
| baseline-small | float | cold_ms | 0.260427 | 0.259209 | 3.31% / 20% | 1.55% / 30% | PASS |
| baseline-small | float | median_ms | 0.026923 | 0.026845 | 1.85% / 5% | 0.66% / 10% | PASS |
| baseline-small | float | p90_ms | 0.027541 | 0.027581 | 2.65% / 5% | 1.07% / 10% | PASS |
| baseline-small | float | p99_ms | 0.029117 | 0.028402 | 39.26% / 5% | 15.64% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 4.006350 | 4.014230 | 4.26% / 15% | 3.35% / 25% | PASS |

### mimo_deinterleave

8/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/mimo_deinterleave/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/mimo_deinterleave/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 2.655560 | 2.722375 | 4.00% / 20% | 3.40% / 30% | PASS |
| baseline-large | double | median_ms | 2.536275 | 2.534870 | 0.91% / 5% | 0.46% / 10% | PASS |
| baseline-large | double | p90_ms | 2.603500 | 2.603760 | 0.59% / 5% | 0.36% / 10% | PASS |
| baseline-large | double | p99_ms | 2.633310 | 2.628825 | 1.42% / 5% | 0.95% / 10% | PASS |
| baseline-large | double | setup_ms | 19.422850 | 19.325800 | 2.04% / 15% | 1.41% / 25% | PASS |
| baseline-large | float | cold_ms | 1.399800 | 1.406280 | 4.93% / 20% | 1.40% / 30% | PASS |
| baseline-large | float | median_ms | 1.200685 | 1.200420 | 1.48% / 5% | 0.77% / 10% | PASS |
| baseline-large | float | p90_ms | 1.302830 | 1.304040 | 1.28% / 5% | 0.70% / 10% | PASS |
| baseline-large | float | p99_ms | 1.339715 | 1.335095 | 1.45% / 5% | 1.04% / 10% | PASS |
| baseline-large | float | setup_ms | 12.031250 | 12.066250 | 6.89% / 15% | 3.91% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.329709 | 0.330991 | 2.21% / 20% | 1.07% / 30% | PASS |
| baseline-medium | double | median_ms | 0.098466 | 0.099514 | 3.45% / 5% | 2.01% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.112800 | 0.114172 | 12.58% / 5% | 7.21% / 10% | FAIL: delta |
| baseline-medium | double | p99_ms | 0.130265 | 0.132906 | 24.93% / 5% | 5.86% / 10% | FAIL: delta |
| baseline-medium | double | setup_ms | 4.547530 | 4.660720 | 10.66% / 15% | 3.97% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.310574 | 0.312885 | 1.80% / 20% | 1.19% / 30% | PASS |
| baseline-medium | float | median_ms | 0.064421 | 0.064491 | 1.64% / 5% | 1.29% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.066135 | 0.066084 | 7.50% / 5% | 5.16% / 10% | FAIL: delta |
| baseline-medium | float | p99_ms | 0.071650 | 0.072101 | 19.59% / 5% | 14.67% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 4.064245 | 4.094490 | 14.04% / 15% | 8.29% / 25% | PASS |
| baseline-small | double | cold_ms | 0.252719 | 0.254358 | 3.78% / 20% | 1.88% / 30% | PASS |
| baseline-small | double | median_ms | 0.030139 | 0.030125 | 1.95% / 5% | 0.80% / 10% | PASS |
| baseline-small | double | p90_ms | 0.030844 | 0.030935 | 8.94% / 5% | 2.07% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.034402 | 0.032010 | 23.04% / 5% | 22.35% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 2.827205 | 2.904865 | 6.36% / 15% | 5.62% / 25% | PASS |
| baseline-small | float | cold_ms | 0.255171 | 0.253278 | 2.58% / 20% | 1.99% / 30% | PASS |
| baseline-small | float | median_ms | 0.030000 | 0.030097 | 1.27% / 5% | 1.16% / 10% | PASS |
| baseline-small | float | p90_ms | 0.030903 | 0.031008 | 8.68% / 5% | 1.21% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.032339 | 0.033122 | 31.75% / 5% | 21.07% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 2.932535 | 2.983475 | 8.58% / 15% | 5.46% / 25% | PASS |

### multi_source_expansion

2/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/multi_source_expansion/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/multi_source_expansion/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 3.975460 | 3.976725 | 0.48% / 20% | 0.25% / 30% | PASS |
| baseline-large | double | median_ms | 3.745965 | 3.746420 | 0.10% / 5% | 0.05% / 10% | PASS |
| baseline-large | double | p90_ms | 3.753195 | 3.754390 | 0.42% / 5% | 0.15% / 10% | PASS |
| baseline-large | double | p99_ms | 3.767940 | 3.769685 | 0.66% / 5% | 0.42% / 10% | PASS |
| baseline-large | double | setup_ms | 5.486560 | 5.453355 | 5.76% / 15% | 3.92% / 25% | PASS |
| baseline-large | float | cold_ms | 3.082570 | 3.083120 | 0.65% / 20% | 0.33% / 30% | PASS |
| baseline-large | float | median_ms | 2.861310 | 2.861160 | 0.14% / 5% | 0.07% / 10% | PASS |
| baseline-large | float | p90_ms | 2.869050 | 2.868160 | 0.54% / 5% | 0.21% / 10% | PASS |
| baseline-large | float | p99_ms | 2.883420 | 2.885895 | 2.87% / 5% | 0.34% / 10% | PASS |
| baseline-large | float | setup_ms | 5.267735 | 5.343735 | 11.40% / 15% | 5.41% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.791160 | 0.790330 | 1.58% / 20% | 1.01% / 30% | PASS |
| baseline-medium | double | median_ms | 0.578761 | 0.578707 | 0.30% / 5% | 0.14% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.583351 | 0.584253 | 0.85% / 5% | 0.47% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.602440 | 0.602302 | 1.82% / 5% | 1.36% / 10% | PASS |
| baseline-medium | double | setup_ms | 4.782525 | 4.810480 | 7.20% / 15% | 4.93% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.679538 | 0.681686 | 2.82% / 20% | 1.83% / 30% | PASS |
| baseline-medium | float | median_ms | 0.467911 | 0.468089 | 0.46% / 5% | 0.25% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.472768 | 0.472643 | 0.93% / 5% | 0.55% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.491919 | 0.491247 | 4.35% / 5% | 1.45% / 10% | PASS |
| baseline-medium | float | setup_ms | 4.816090 | 4.864125 | 6.86% / 15% | 3.39% / 25% | PASS |
| baseline-small | double | cold_ms | 0.359236 | 0.357287 | 5.25% / 20% | 2.54% / 30% | PASS |
| baseline-small | double | median_ms | 0.126287 | 0.126000 | 1.39% / 5% | 0.83% / 10% | PASS |
| baseline-small | double | p90_ms | 0.129111 | 0.130154 | 3.62% / 5% | 1.94% / 10% | PASS |
| baseline-small | double | p99_ms | 0.145208 | 0.148132 | 13.67% / 5% | 8.05% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 4.718260 | 4.816755 | 6.02% / 15% | 4.81% / 25% | PASS |
| baseline-small | float | cold_ms | 0.351593 | 0.352584 | 5.46% / 20% | 2.79% / 30% | PASS |
| baseline-small | float | median_ms | 0.124257 | 0.124164 | 4.50% / 5% | 0.91% / 10% | PASS |
| baseline-small | float | p90_ms | 0.127621 | 0.127230 | 4.43% / 5% | 1.16% / 10% | PASS |
| baseline-small | float | p99_ms | 0.140240 | 0.143037 | 16.45% / 5% | 8.13% / 10% | FAIL: delta |
| baseline-small | float | setup_ms | 4.704670 | 4.726945 | 7.04% / 15% | 5.35% / 25% | PASS |

### music_estimator

13/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/music_estimator/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/music_estimator/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 30.969500 | 30.932700 | 3.02% / 20% | 0.98% / 30% | PASS |
| baseline-large | double | median_ms | 31.530700 | 31.591550 | 1.83% / 5% | 1.64% / 10% | PASS |
| baseline-large | double | p90_ms | 31.865700 | 31.923400 | 0.58% / 5% | 0.54% / 10% | PASS |
| baseline-large | double | p99_ms | 31.993850 | 32.033850 | 1.61% / 5% | 0.85% / 10% | PASS |
| baseline-large | double | setup_ms | 8.874070 | 9.029890 | 9.11% / 15% | 4.92% / 25% | PASS |
| baseline-large | float | cold_ms | 11.812400 | 12.025250 | 5.63% / 20% | 4.31% / 30% | PASS |
| baseline-large | float | median_ms | 11.914150 | 11.878150 | 2.43% / 5% | 0.90% / 10% | PASS |
| baseline-large | float | p90_ms | 12.400350 | 12.399950 | 4.57% / 5% | 3.54% / 10% | PASS |
| baseline-large | float | p99_ms | 12.532900 | 12.761850 | 8.17% / 5% | 3.96% / 10% | FAIL: delta |
| baseline-large | float | setup_ms | 7.832990 | 7.507345 | 7.64% / 15% | 5.64% / 25% | PASS |
| baseline-medium | double | cold_ms | 7.936630 | 7.818675 | 7.69% / 20% | 7.92% / 30% | PASS |
| baseline-medium | double | median_ms | 7.950375 | 7.926540 | 4.92% / 5% | 1.89% / 10% | PASS |
| baseline-medium | double | p90_ms | 8.223215 | 8.273745 | 6.33% / 5% | 4.41% / 10% | FAIL: delta |
| baseline-medium | double | p99_ms | 8.437470 | 8.563805 | 8.44% / 5% | 5.22% / 10% | FAIL: delta |
| baseline-medium | double | setup_ms | 6.470855 | 6.455185 | 10.24% / 15% | 2.50% / 25% | PASS |
| baseline-medium | float | cold_ms | 4.405490 | 4.311050 | 17.35% / 20% | 11.31% / 30% | PASS |
| baseline-medium | float | median_ms | 4.093900 | 4.019685 | 13.36% / 5% | 12.22% / 10% | FAIL: delta, IQR |
| baseline-medium | float | p90_ms | 4.999815 | 4.871025 | 13.58% / 5% | 12.50% / 10% | FAIL: delta, IQR |
| baseline-medium | float | p99_ms | 5.435325 | 5.441080 | 10.14% / 5% | 5.64% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 6.313380 | 6.239255 | 7.92% / 15% | 3.74% / 25% | PASS |
| baseline-small | double | cold_ms | 4.638850 | 4.335930 | 21.36% / 20% | 15.80% / 30% | FAIL: delta |
| baseline-small | double | median_ms | 4.044890 | 4.005120 | 9.80% / 5% | 8.65% / 10% | FAIL: delta |
| baseline-small | double | p90_ms | 4.829800 | 4.830205 | 12.15% / 5% | 13.34% / 10% | FAIL: delta, IQR |
| baseline-small | double | p99_ms | 5.390020 | 5.442140 | 10.61% / 5% | 7.33% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 6.017965 | 6.036045 | 3.88% / 15% | 2.13% / 25% | PASS |
| baseline-small | float | cold_ms | 3.563200 | 3.324420 | 18.77% / 20% | 18.76% / 30% | PASS |
| baseline-small | float | median_ms | 3.167710 | 3.037625 | 20.37% / 5% | 12.80% / 10% | FAIL: delta, IQR |
| baseline-small | float | p90_ms | 3.893450 | 3.818105 | 18.26% / 5% | 11.31% / 10% | FAIL: delta, IQR |
| baseline-small | float | p99_ms | 4.196900 | 4.106480 | 12.79% / 5% | 12.22% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 6.078360 | 6.079910 | 4.33% / 15% | 2.01% / 25% | PASS |

### mvdr_estimator

3/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/mvdr_estimator/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/mvdr_estimator/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 31.349650 | 31.125850 | 2.52% / 20% | 1.91% / 30% | PASS |
| baseline-large | double | median_ms | 22.168550 | 22.168750 | 0.04% / 5% | 0.02% / 10% | PASS |
| baseline-large | double | p90_ms | 22.184150 | 22.188050 | 0.08% / 5% | 0.04% / 10% | PASS |
| baseline-large | double | p99_ms | 22.214800 | 22.213550 | 0.38% / 5% | 0.10% / 10% | PASS |
| baseline-large | double | setup_ms | 8.729150 | 8.683175 | 9.64% / 15% | 6.98% / 25% | PASS |
| baseline-large | float | cold_ms | 12.228550 | 12.374400 | 5.56% / 20% | 4.39% / 30% | PASS |
| baseline-large | float | median_ms | 6.024545 | 6.024360 | 0.13% / 5% | 0.06% / 10% | PASS |
| baseline-large | float | p90_ms | 6.046730 | 6.044340 | 0.27% / 5% | 0.14% / 10% | PASS |
| baseline-large | float | p99_ms | 6.069225 | 6.071275 | 2.74% / 5% | 0.19% / 10% | PASS |
| baseline-large | float | setup_ms | 7.244455 | 7.300665 | 12.02% / 15% | 5.92% / 25% | PASS |
| baseline-medium | double | cold_ms | 9.135895 | 9.000940 | 10.66% / 20% | 6.93% / 30% | PASS |
| baseline-medium | double | median_ms | 2.618500 | 2.619145 | 0.24% / 5% | 0.07% / 10% | PASS |
| baseline-medium | double | p90_ms | 2.632300 | 2.634225 | 0.32% / 5% | 0.12% / 10% | PASS |
| baseline-medium | double | p99_ms | 2.647980 | 2.652480 | 0.89% / 5% | 0.39% / 10% | PASS |
| baseline-medium | double | setup_ms | 6.373665 | 6.351980 | 7.53% / 15% | 4.84% / 25% | PASS |
| baseline-medium | float | cold_ms | 5.824015 | 5.746490 | 13.50% / 20% | 10.98% / 30% | PASS |
| baseline-medium | float | median_ms | 0.511106 | 0.510849 | 0.67% / 5% | 0.27% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.527581 | 0.525931 | 2.41% / 5% | 1.16% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.552914 | 0.552865 | 2.68% / 5% | 1.62% / 10% | PASS |
| baseline-medium | float | setup_ms | 6.108500 | 6.136385 | 8.06% / 15% | 4.74% / 25% | PASS |
| baseline-small | double | cold_ms | 6.047025 | 5.659620 | 14.01% / 20% | 11.70% / 30% | PASS |
| baseline-small | double | median_ms | 0.507892 | 0.507535 | 0.36% / 5% | 0.26% / 10% | PASS |
| baseline-small | double | p90_ms | 0.522233 | 0.526193 | 2.50% / 5% | 1.43% / 10% | PASS |
| baseline-small | double | p99_ms | 0.545028 | 0.545782 | 8.57% / 5% | 1.85% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 6.017290 | 6.001935 | 2.96% / 15% | 2.64% / 25% | PASS |
| baseline-small | float | cold_ms | 4.810460 | 4.906605 | 22.23% / 20% | 11.74% / 30% | FAIL: delta |
| baseline-small | float | median_ms | 0.204369 | 0.203618 | 1.28% / 5% | 0.61% / 10% | PASS |
| baseline-small | float | p90_ms | 0.222294 | 0.222321 | 2.71% / 5% | 1.23% / 10% | PASS |
| baseline-small | float | p99_ms | 0.257501 | 0.257595 | 7.64% / 5% | 2.95% / 10% | FAIL: delta |
| baseline-small | float | setup_ms | 5.982170 | 5.955170 | 5.54% / 15% | 2.81% / 25% | PASS |

### phase_shift_beamformer

5/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/phase_shift_beamformer/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/phase_shift_beamformer/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 4.008050 | 4.019445 | 15.84% / 20% | 0.87% / 30% | PASS |
| baseline-large | double | median_ms | 2.427250 | 2.427430 | 0.07% / 5% | 0.02% / 10% | PASS |
| baseline-large | double | p90_ms | 2.430945 | 2.430550 | 0.22% / 5% | 0.11% / 10% | PASS |
| baseline-large | double | p99_ms | 2.441995 | 2.438485 | 0.56% / 5% | 0.26% / 10% | PASS |
| baseline-large | double | setup_ms | 8.013625 | 7.995510 | 9.79% / 15% | 9.05% / 25% | PASS |
| baseline-large | float | cold_ms | 1.688685 | 1.683055 | 12.20% / 20% | 1.40% / 30% | PASS |
| baseline-large | float | median_ms | 0.153133 | 0.152978 | 0.65% / 5% | 0.32% / 10% | PASS |
| baseline-large | float | p90_ms | 0.155514 | 0.155920 | 1.96% / 5% | 1.19% / 10% | PASS |
| baseline-large | float | p99_ms | 0.168715 | 0.168102 | 5.54% / 5% | 2.93% / 10% | FAIL: delta |
| baseline-large | float | setup_ms | 5.896825 | 5.869940 | 9.96% / 15% | 3.95% / 25% | PASS |
| baseline-medium | double | cold_ms | 1.482535 | 1.495810 | 14.30% / 20% | 4.61% / 30% | PASS |
| baseline-medium | double | median_ms | 0.149931 | 0.149931 | 0.51% / 5% | 0.33% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.151551 | 0.151559 | 1.23% / 5% | 0.74% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.159718 | 0.158133 | 4.91% / 5% | 2.30% / 10% | PASS |
| baseline-medium | double | setup_ms | 4.190180 | 4.177405 | 13.53% / 15% | 1.87% / 25% | PASS |
| baseline-medium | float | cold_ms | 1.430355 | 1.401350 | 7.31% / 20% | 4.54% / 30% | PASS |
| baseline-medium | float | median_ms | 0.058811 | 0.058711 | 1.53% / 5% | 1.05% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.060571 | 0.060862 | 4.46% / 5% | 2.74% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.066882 | 0.070954 | 15.58% / 5% | 8.39% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 4.032150 | 4.007850 | 4.76% / 15% | 2.23% / 25% | PASS |
| baseline-small | double | cold_ms | 1.593730 | 1.517685 | 18.00% / 20% | 14.28% / 30% | PASS |
| baseline-small | double | median_ms | 0.149367 | 0.149466 | 0.63% / 5% | 0.37% / 10% | PASS |
| baseline-small | double | p90_ms | 0.151790 | 0.151459 | 2.17% / 5% | 0.81% / 10% | PASS |
| baseline-small | double | p99_ms | 0.160022 | 0.159909 | 5.73% / 5% | 4.31% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 3.869355 | 3.880450 | 3.34% / 15% | 1.12% / 25% | PASS |
| baseline-small | float | cold_ms | 1.541715 | 1.520625 | 18.18% / 20% | 11.35% / 30% | PASS |
| baseline-small | float | median_ms | 0.056023 | 0.056338 | 2.32% / 5% | 0.85% / 10% | PASS |
| baseline-small | float | p90_ms | 0.057933 | 0.058230 | 6.32% / 5% | 5.34% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.068353 | 0.065076 | 12.37% / 5% | 10.59% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 3.875775 | 3.859725 | 2.25% / 15% | 1.28% / 25% | PASS |

### rank_one_residual_screen

1/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/rank_one_residual_screen/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/rank_one_residual_screen/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 3.654780 | 3.660325 | 1.17% / 20% | 0.70% / 30% | PASS |
| baseline-large | double | median_ms | 3.373070 | 3.373900 | 0.36% / 5% | 0.19% / 10% | PASS |
| baseline-large | double | p90_ms | 3.411150 | 3.411465 | 0.29% / 5% | 0.12% / 10% | PASS |
| baseline-large | double | p99_ms | 3.438670 | 3.443585 | 0.60% / 5% | 0.35% / 10% | PASS |
| baseline-large | double | setup_ms | 12.307850 | 12.348750 | 7.35% / 15% | 4.68% / 25% | PASS |
| baseline-large | float | cold_ms | 3.503345 | 3.506835 | 0.49% / 20% | 0.20% / 30% | PASS |
| baseline-large | float | median_ms | 3.327400 | 3.327510 | 0.14% / 5% | 0.09% / 10% | PASS |
| baseline-large | float | p90_ms | 3.344565 | 3.341650 | 0.25% / 5% | 0.25% / 10% | PASS |
| baseline-large | float | p99_ms | 3.366650 | 3.363305 | 1.24% / 5% | 0.40% / 10% | PASS |
| baseline-large | float | setup_ms | 9.013575 | 8.985245 | 9.66% / 15% | 5.35% / 25% | PASS |
| baseline-medium | double | cold_ms | 1.125590 | 1.130825 | 1.70% / 20% | 0.73% / 30% | PASS |
| baseline-medium | double | median_ms | 0.861843 | 0.861809 | 0.11% / 5% | 0.10% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.870174 | 0.867163 | 0.89% / 5% | 0.66% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.884369 | 0.881275 | 1.50% / 5% | 0.82% / 10% | PASS |
| baseline-medium | double | setup_ms | 6.790705 | 6.761430 | 13.28% / 15% | 4.16% / 25% | PASS |
| baseline-medium | float | cold_ms | 1.199665 | 1.202315 | 1.69% / 20% | 0.59% / 30% | PASS |
| baseline-medium | float | median_ms | 0.932840 | 0.932527 | 0.22% / 5% | 0.10% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.937341 | 0.936822 | 1.07% / 5% | 0.37% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.951769 | 0.951841 | 1.78% / 5% | 0.95% / 10% | PASS |
| baseline-medium | float | setup_ms | 6.463770 | 6.354830 | 12.28% / 15% | 3.75% / 25% | PASS |
| baseline-small | double | cold_ms | 0.417515 | 0.417516 | 2.89% / 20% | 3.26% / 30% | PASS |
| baseline-small | double | median_ms | 0.150721 | 0.151251 | 4.89% / 5% | 3.59% / 10% | PASS |
| baseline-small | double | p90_ms | 0.158345 | 0.158672 | 2.37% / 5% | 0.70% / 10% | PASS |
| baseline-small | double | p99_ms | 0.166884 | 0.168162 | 5.88% / 5% | 3.06% / 10% | FAIL: delta |
| baseline-small | double | setup_ms | 4.995790 | 4.935470 | 4.43% / 15% | 3.81% / 25% | PASS |
| baseline-small | float | cold_ms | 0.419345 | 0.422363 | 4.50% / 20% | 2.98% / 30% | PASS |
| baseline-small | float | median_ms | 0.157626 | 0.157631 | 0.86% / 5% | 0.54% / 10% | PASS |
| baseline-small | float | p90_ms | 0.159869 | 0.159920 | 1.41% / 5% | 0.95% / 10% | PASS |
| baseline-small | float | p99_ms | 0.170983 | 0.170678 | 4.40% / 5% | 3.07% / 10% | PASS |
| baseline-small | float | setup_ms | 5.022520 | 4.975335 | 5.99% / 15% | 4.47% / 25% | PASS |

### snapshot_gather

5/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/snapshot_gather/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/snapshot_gather/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 3.698200 | 3.661115 | 3.25% / 20% | 2.36% / 30% | PASS |
| baseline-large | double | median_ms | 3.410860 | 3.418640 | 1.07% / 5% | 0.42% / 10% | PASS |
| baseline-large | double | p90_ms | 3.552325 | 3.550045 | 0.56% / 5% | 0.28% / 10% | PASS |
| baseline-large | double | p99_ms | 3.583145 | 3.582305 | 1.23% / 5% | 0.38% / 10% | PASS |
| baseline-large | double | setup_ms | 30.208250 | 30.526000 | 2.97% / 15% | 2.05% / 25% | PASS |
| baseline-large | float | cold_ms | 2.185925 | 2.188255 | 2.27% / 20% | 1.24% / 30% | PASS |
| baseline-large | float | median_ms | 1.877240 | 1.877955 | 1.18% / 5% | 0.40% / 10% | PASS |
| baseline-large | float | p90_ms | 1.983200 | 1.981825 | 0.93% / 5% | 0.38% / 10% | PASS |
| baseline-large | float | p99_ms | 2.018085 | 2.014220 | 1.79% / 5% | 0.52% / 10% | PASS |
| baseline-large | float | setup_ms | 18.373200 | 18.291850 | 3.72% / 15% | 1.37% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.701030 | 0.705613 | 1.75% / 20% | 1.32% / 30% | PASS |
| baseline-medium | double | median_ms | 0.443391 | 0.443837 | 1.35% / 5% | 0.50% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.480086 | 0.480854 | 3.25% / 5% | 2.33% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.512769 | 0.513115 | 1.73% / 5% | 1.26% / 10% | PASS |
| baseline-medium | double | setup_ms | 8.361225 | 8.707015 | 12.05% / 15% | 8.34% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.430188 | 0.429552 | 2.46% / 20% | 1.00% / 30% | PASS |
| baseline-medium | float | median_ms | 0.261576 | 0.262227 | 1.90% / 5% | 0.83% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.280590 | 0.281357 | 3.98% / 5% | 2.11% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.308205 | 0.306774 | 7.35% / 5% | 2.01% / 10% | FAIL: delta |
| baseline-medium | float | setup_ms | 6.290950 | 6.306100 | 12.50% / 15% | 4.69% / 25% | PASS |
| baseline-small | double | cold_ms | 0.314177 | 0.315210 | 3.99% / 20% | 1.54% / 30% | PASS |
| baseline-small | double | median_ms | 0.074738 | 0.074722 | 1.52% / 5% | 1.01% / 10% | PASS |
| baseline-small | double | p90_ms | 0.080089 | 0.078591 | 9.20% / 5% | 7.16% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.094600 | 0.095130 | 27.27% / 5% | 14.57% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 4.175425 | 4.184140 | 9.06% / 15% | 3.48% / 25% | PASS |
| baseline-small | float | cold_ms | 0.314256 | 0.315445 | 3.30% / 20% | 2.28% / 30% | PASS |
| baseline-small | float | median_ms | 0.074512 | 0.074577 | 2.05% / 5% | 1.10% / 10% | PASS |
| baseline-small | float | p90_ms | 0.077164 | 0.078733 | 7.88% / 5% | 5.77% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.089389 | 0.096186 | 20.22% / 5% | 12.98% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 3.975215 | 4.014250 | 6.76% / 15% | 3.82% / 25% | PASS |

### subarray_selection

9/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/subarray_selection/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/subarray_selection/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 0.277140 | 0.278187 | 2.40% / 20% | 0.99% / 30% | PASS |
| baseline-large | double | median_ms | 0.039915 | 0.040225 | 2.26% / 5% | 1.48% / 10% | PASS |
| baseline-large | double | p90_ms | 0.040826 | 0.041225 | 9.49% / 5% | 1.46% / 10% | FAIL: delta |
| baseline-large | double | p99_ms | 0.044640 | 0.047166 | 18.85% / 5% | 14.19% / 10% | FAIL: delta, IQR |
| baseline-large | double | setup_ms | 4.567440 | 4.629015 | 7.74% / 15% | 4.42% / 25% | PASS |
| baseline-large | float | cold_ms | 0.266922 | 0.267251 | 2.59% / 20% | 1.46% / 30% | PASS |
| baseline-large | float | median_ms | 0.039620 | 0.039806 | 3.61% / 5% | 1.30% / 10% | PASS |
| baseline-large | float | p90_ms | 0.040814 | 0.041122 | 12.17% / 5% | 4.20% / 10% | FAIL: delta |
| baseline-large | float | p99_ms | 0.046470 | 0.049243 | 28.16% / 5% | 18.68% / 10% | FAIL: delta, IQR |
| baseline-large | float | setup_ms | 4.051540 | 3.879990 | 6.95% / 15% | 5.27% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.257987 | 0.258798 | 3.68% / 20% | 2.09% / 30% | PASS |
| baseline-medium | double | median_ms | 0.021007 | 0.021000 | 2.35% / 5% | 1.49% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.021691 | 0.021687 | 2.52% / 5% | 2.15% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.022460 | 0.022604 | 27.97% / 5% | 19.55% / 10% | FAIL: delta, IQR |
| baseline-medium | double | setup_ms | 3.570355 | 3.552910 | 6.44% / 15% | 4.93% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.254015 | 0.255654 | 2.88% / 20% | 1.27% / 30% | PASS |
| baseline-medium | float | median_ms | 0.020866 | 0.020817 | 2.40% / 5% | 1.02% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.021560 | 0.021408 | 2.98% / 5% | 1.07% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.022273 | 0.025140 | 31.35% / 5% | 20.25% / 10% | FAIL: delta, IQR |
| baseline-medium | float | setup_ms | 3.784970 | 3.811615 | 5.88% / 15% | 4.17% / 25% | PASS |
| baseline-small | double | cold_ms | 0.255084 | 0.255764 | 3.46% / 20% | 1.69% / 30% | PASS |
| baseline-small | double | median_ms | 0.020889 | 0.020928 | 2.64% / 5% | 1.30% / 10% | PASS |
| baseline-small | double | p90_ms | 0.021521 | 0.021650 | 7.93% / 5% | 2.78% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.022505 | 0.022582 | 74.95% / 5% | 16.96% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.718300 | 3.717585 | 4.29% / 15% | 4.06% / 25% | PASS |
| baseline-small | float | cold_ms | 0.253061 | 0.252936 | 3.16% / 20% | 1.60% / 30% | PASS |
| baseline-small | float | median_ms | 0.020806 | 0.020683 | 3.27% / 5% | 1.22% / 10% | PASS |
| baseline-small | float | p90_ms | 0.021366 | 0.021264 | 3.12% / 5% | 1.64% / 10% | PASS |
| baseline-small | float | p99_ms | 0.022130 | 0.021756 | 16.49% / 5% | 2.98% / 10% | FAIL: delta |
| baseline-small | float | setup_ms | 3.714520 | 3.693100 | 3.89% / 15% | 4.54% / 25% | PASS |

### tdm_dense_compensation

4/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/tdm_dense_compensation/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/tdm_dense_compensation/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 21.329000 | 21.333600 | 0.85% / 20% | 0.58% / 30% | PASS |
| baseline-large | double | median_ms | 21.196350 | 21.191650 | 0.18% / 5% | 0.09% / 10% | PASS |
| baseline-large | double | p90_ms | 21.352150 | 21.348350 | 0.26% / 5% | 0.14% / 10% | PASS |
| baseline-large | double | p99_ms | 21.416850 | 21.433000 | 1.12% / 5% | 0.31% / 10% | PASS |
| baseline-large | double | setup_ms | 116.369500 | 116.036000 | 1.71% / 15% | 1.24% / 25% | PASS |
| baseline-large | float | cold_ms | 9.712960 | 9.719520 | 3.36% / 20% | 1.22% / 30% | PASS |
| baseline-large | float | median_ms | 9.509270 | 9.506895 | 0.54% / 5% | 0.26% / 10% | PASS |
| baseline-large | float | p90_ms | 9.736065 | 9.715985 | 1.65% / 5% | 1.00% / 10% | PASS |
| baseline-large | float | p99_ms | 9.816815 | 9.829245 | 2.71% / 5% | 1.03% / 10% | PASS |
| baseline-large | float | setup_ms | 60.851250 | 61.106750 | 1.78% / 15% | 1.06% / 25% | PASS |
| baseline-medium | double | cold_ms | 1.530415 | 1.531940 | 0.96% / 20% | 0.56% / 30% | PASS |
| baseline-medium | double | median_ms | 1.323055 | 1.324165 | 0.42% / 5% | 0.27% / 10% | PASS |
| baseline-medium | double | p90_ms | 1.385260 | 1.383210 | 1.20% / 5% | 0.70% / 10% | PASS |
| baseline-medium | double | p99_ms | 1.410615 | 1.409960 | 1.38% / 5% | 0.87% / 10% | PASS |
| baseline-medium | double | setup_ms | 12.108450 | 11.968600 | 7.39% / 15% | 3.02% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.798801 | 0.798278 | 1.86% / 20% | 0.98% / 30% | PASS |
| baseline-medium | float | median_ms | 0.591338 | 0.592993 | 1.42% / 5% | 0.81% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.647421 | 0.654653 | 3.13% / 5% | 2.46% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.684549 | 0.686219 | 2.04% / 5% | 1.37% / 10% | PASS |
| baseline-medium | float | setup_ms | 8.286465 | 8.242435 | 11.18% / 15% | 9.04% / 25% | PASS |
| baseline-small | double | cold_ms | 0.266504 | 0.266773 | 3.34% / 20% | 1.95% / 30% | PASS |
| baseline-small | double | median_ms | 0.034919 | 0.034905 | 1.90% / 5% | 0.85% / 10% | PASS |
| baseline-small | double | p90_ms | 0.036052 | 0.036018 | 16.29% / 5% | 1.01% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.038849 | 0.037471 | 33.40% / 5% | 22.93% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.165695 | 3.163760 | 11.27% / 15% | 5.35% / 25% | PASS |
| baseline-small | float | cold_ms | 0.262964 | 0.262999 | 2.50% / 20% | 1.21% / 30% | PASS |
| baseline-small | float | median_ms | 0.032525 | 0.032442 | 3.69% / 5% | 0.70% / 10% | PASS |
| baseline-small | float | p90_ms | 0.033401 | 0.033319 | 9.01% / 5% | 1.84% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.036167 | 0.036245 | 27.76% / 5% | 16.54% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 2.899325 | 2.850025 | 9.26% / 15% | 6.20% / 25% | PASS |

### tdm_sparse_compensation

4/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/tdm_sparse_compensation/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/tdm_sparse_compensation/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 5.529895 | 5.503645 | 3.82% / 20% | 2.67% / 30% | PASS |
| baseline-large | double | median_ms | 5.197415 | 5.203855 | 0.80% / 5% | 0.53% / 10% | PASS |
| baseline-large | double | p90_ms | 5.329110 | 5.314735 | 0.78% / 5% | 0.44% / 10% | PASS |
| baseline-large | double | p99_ms | 5.401310 | 5.390570 | 1.01% / 5% | 0.46% / 10% | PASS |
| baseline-large | double | setup_ms | 53.646150 | 54.124350 | 2.94% / 15% | 1.75% / 25% | PASS |
| baseline-large | float | cold_ms | 3.066450 | 3.079730 | 3.39% / 20% | 1.42% / 30% | PASS |
| baseline-large | float | median_ms | 2.750830 | 2.777910 | 2.06% / 5% | 1.43% / 10% | PASS |
| baseline-large | float | p90_ms | 2.891355 | 2.891810 | 0.61% / 5% | 0.31% / 10% | PASS |
| baseline-large | float | p99_ms | 2.919610 | 2.922820 | 2.05% / 5% | 0.92% / 10% | PASS |
| baseline-large | float | setup_ms | 29.883700 | 30.163400 | 3.41% / 15% | 1.70% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.945107 | 0.947950 | 3.12% / 20% | 1.36% / 30% | PASS |
| baseline-medium | double | median_ms | 0.690830 | 0.691685 | 1.45% / 5% | 0.51% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.745661 | 0.748243 | 1.50% / 5% | 0.88% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.771814 | 0.774406 | 2.49% / 5% | 1.45% / 10% | PASS |
| baseline-medium | double | setup_ms | 12.392800 | 12.399700 | 6.44% / 15% | 1.41% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.693366 | 0.691324 | 1.73% / 20% | 1.28% / 30% | PASS |
| baseline-medium | float | median_ms | 0.394802 | 0.394640 | 1.28% / 5% | 0.63% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.421416 | 0.424064 | 2.81% / 5% | 1.87% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.461076 | 0.457505 | 2.32% / 5% | 1.70% / 10% | PASS |
| baseline-medium | float | setup_ms | 9.018185 | 8.416000 | 8.75% / 15% | 7.59% / 25% | PASS |
| baseline-small | double | cold_ms | 0.337788 | 0.339111 | 4.00% / 20% | 2.13% / 30% | PASS |
| baseline-small | double | median_ms | 0.064340 | 0.064271 | 3.03% / 5% | 1.33% / 10% | PASS |
| baseline-small | double | p90_ms | 0.065939 | 0.066170 | 10.52% / 5% | 1.88% / 10% | FAIL: delta |
| baseline-small | double | p99_ms | 0.072750 | 0.075013 | 15.34% / 5% | 15.58% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 5.355160 | 5.339200 | 10.88% / 15% | 3.72% / 25% | PASS |
| baseline-small | float | cold_ms | 0.328978 | 0.326898 | 3.00% / 20% | 2.46% / 30% | PASS |
| baseline-small | float | median_ms | 0.064375 | 0.064021 | 1.98% / 5% | 1.05% / 10% | PASS |
| baseline-small | float | p90_ms | 0.066159 | 0.066467 | 6.23% / 5% | 2.06% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.073589 | 0.080198 | 21.31% / 5% | 16.45% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 4.870860 | 4.887155 | 8.70% / 15% | 2.24% / 25% | PASS |

### virtual_array_assembly

3/30 rows fail calibration. [Raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/virtual_array_assembly/review.json); [all twenty pair reports](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/virtual_array_assembly/aa/pairs.json).

| Case | Precision | Metric | A1 median ms | A2 median ms | Paired delta P90 / limit | Baseline IQR / limit | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-large | double | cold_ms | 8.641180 | 8.617470 | 1.28% / 20% | 0.79% / 30% | PASS |
| baseline-large | double | median_ms | 8.381790 | 8.384150 | 1.14% / 5% | 0.58% / 10% | PASS |
| baseline-large | double | p90_ms | 8.489780 | 8.478930 | 1.06% / 5% | 0.65% / 10% | PASS |
| baseline-large | double | p99_ms | 8.535400 | 8.511325 | 1.82% / 5% | 0.73% / 10% | PASS |
| baseline-large | double | setup_ms | 117.220000 | 117.669000 | 0.99% / 15% | 0.74% / 25% | PASS |
| baseline-large | float | cold_ms | 4.520915 | 4.516195 | 4.00% / 20% | 2.11% / 30% | PASS |
| baseline-large | float | median_ms | 4.333710 | 4.333920 | 1.56% / 5% | 1.27% / 10% | PASS |
| baseline-large | float | p90_ms | 4.453320 | 4.447695 | 2.34% / 5% | 1.98% / 10% | PASS |
| baseline-large | float | p99_ms | 4.504280 | 4.513505 | 2.84% / 5% | 1.71% / 10% | PASS |
| baseline-large | float | setup_ms | 61.782800 | 61.793800 | 1.55% / 15% | 1.35% / 25% | PASS |
| baseline-medium | double | cold_ms | 0.859923 | 0.860344 | 2.93% / 20% | 1.35% / 30% | PASS |
| baseline-medium | double | median_ms | 0.647716 | 0.651075 | 2.61% / 5% | 1.63% / 10% | PASS |
| baseline-medium | double | p90_ms | 0.683578 | 0.686718 | 2.03% / 5% | 1.86% / 10% | PASS |
| baseline-medium | double | p99_ms | 0.711455 | 0.714372 | 2.30% / 5% | 1.39% / 10% | PASS |
| baseline-medium | double | setup_ms | 12.650800 | 12.626500 | 5.61% / 15% | 2.22% / 25% | PASS |
| baseline-medium | float | cold_ms | 0.572080 | 0.570126 | 3.70% / 20% | 1.59% / 30% | PASS |
| baseline-medium | float | median_ms | 0.385159 | 0.385370 | 3.37% / 5% | 1.89% / 10% | PASS |
| baseline-medium | float | p90_ms | 0.401556 | 0.403821 | 4.69% / 5% | 2.22% / 10% | PASS |
| baseline-medium | float | p99_ms | 0.433945 | 0.434678 | 3.13% / 5% | 2.19% / 10% | PASS |
| baseline-medium | float | setup_ms | 8.599610 | 8.900715 | 10.15% / 15% | 8.13% / 25% | PASS |
| baseline-small | double | cold_ms | 0.262028 | 0.260495 | 3.41% / 20% | 1.32% / 30% | PASS |
| baseline-small | double | median_ms | 0.020512 | 0.020435 | 1.82% / 5% | 0.81% / 10% | PASS |
| baseline-small | double | p90_ms | 0.021089 | 0.021002 | 4.50% / 5% | 1.61% / 10% | PASS |
| baseline-small | double | p99_ms | 0.023909 | 0.022628 | 37.69% / 5% | 20.34% / 10% | FAIL: delta, IQR |
| baseline-small | double | setup_ms | 3.661265 | 3.638010 | 8.05% / 15% | 3.04% / 25% | PASS |
| baseline-small | float | cold_ms | 0.261103 | 0.258912 | 2.46% / 20% | 1.27% / 30% | PASS |
| baseline-small | float | median_ms | 0.020727 | 0.020701 | 1.84% / 5% | 2.00% / 10% | PASS |
| baseline-small | float | p90_ms | 0.021404 | 0.021396 | 9.59% / 5% | 2.47% / 10% | FAIL: delta |
| baseline-small | float | p99_ms | 0.022561 | 0.022678 | 89.26% / 5% | 28.98% / 10% | FAIL: delta, IQR |
| baseline-small | float | setup_ms | 3.467000 | 3.461425 | 8.76% / 15% | 5.31% / 25% | PASS |
