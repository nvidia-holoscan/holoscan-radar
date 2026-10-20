# Frame-count exploration: distribution estimands

Companion to [fresh frame-count exploration](frame-count-gpu.md). These are diagnostic analyses of the same authorized raw captures; no extra GPU measurements or acceptance-policy changes. The fixed microsecond proposal remains superseded.

## Overall early-frame distribution versus a typical paired-run tail

The main report estimates a median paired quantile effect: construct each side's three-run empirical mixture, compare its quantile, then infer the median effect across independent A/A pairs. That answers a question about the typical paired experiment. It is not the same quantity as the p99 of the overall frame-time distribution across all runs.

The alternative here constructs one equal-run empirical mixture across all 90 runs per side, always using the first 50 measured frames. Thus each side contributes 4,500 observed frames while the independent evidence remains 30 pairs. We resample entire paired clusters, retaining their three runs, chronological frames, paired A1/A2 sides and common pair indices for both precisions. Frames are never independently resampled. All completed replication scopes are included; no timing-dependent exclusions.

The overall-mixture intervals use 4,096 paired-cluster bootstrap draws with fixed seed 1729 and pointwise 95% percentile bounds. The comparison column uses pointwise 95% sign intervals for the different typical-pair effect. Neither column is campaign-familywise. They must not be compared directly with the much more conservative 9,360-comparison intervals in the main report as though the confidence level were identical. Bootstrap coverage is approximate and has not been qualified, especially for sparse or multimodal tails and only 30 independent clusters. Pair independence and a stable timing population remain assumptions; resampling pairs does not remove serial dependence between pairs.

This is an exploratory comparison of estimands, not a new gate, equivalence test, power calculation or demonstration that a smaller frame count is sufficient. A narrower interval does not make an estimand the right performance requirement. These observations informed method exploration and cannot serve as held-out validation of a newly selected gate.

| Operator / case / actual run length | Precision / quantile | Overall first50 A1 µs | Overall A2 change % | Pointwise bootstrap 95% interval % | Typical-pair pointwise sign 95% interval % |
| --- | --- | --- | --- | --- | --- |
| fmcw_range_processor / baseline-small / 50 | float / p50 | 26.518 | -0.140 | [-0.418, +0.141] | [-0.610, +0.386] |
| fmcw_range_processor / baseline-small / 50 | float / p99 | 38.540 | -8.124 | [-30.657, +24.263] | [-26.581, +22.571] |
| fmcw_range_processor / baseline-small / 50 | double / p50 | 44.935 | +0.083 | [-0.246, +0.453] | [-0.226, +0.341] |
| fmcw_range_processor / baseline-small / 50 | double / p99 | 57.418 | -0.840 | [-3.952, +2.767] | [-6.835, +2.886] |
| detection_reduction / baseline-large / 50 | float / p50 | 294.602 | -0.071 | [-0.226, +0.094] | [-0.333, -0.001] |
| detection_reduction / baseline-large / 50 | float / p99 | 352.737 | -1.949 | [-5.880, +0.614] | [-3.241, +0.881] |
| detection_reduction / baseline-large / 50 | double / p50 | 294.857 | -0.162 | [-0.429, +0.103] | [-0.484, +0.459] |
| detection_reduction / baseline-large / 50 | double / p99 | 359.919 | -1.527 | [-8.445, +3.450] | [-5.793, +3.901] |
| fmcw_range_processor / baseline-small / 2000 | float / p50 | 26.454 | +0.178 | [-0.102, +0.522] | [-0.261, +0.477] |
| fmcw_range_processor / baseline-small / 2000 | float / p99 | 38.664 | -21.451 | [-34.445, +14.638] | [-11.073, +1.053] |
| fmcw_range_processor / baseline-small / 2000 | double / p50 | 44.881 | +0.163 | [-0.167, +0.435] | [-0.167, +0.715] |
| fmcw_range_processor / baseline-small / 2000 | double / p99 | 57.677 | +0.049 | [-3.667, +4.726] | [-2.732, +5.609] |
| detection_reduction / baseline-large / 2000 | float / p50 | 294.204 | +0.082 | [-0.093, +0.275] | [-0.076, +0.260] |
| detection_reduction / baseline-large / 2000 | float / p99 | 351.125 | +0.242 | [-7.850, +5.485] | [-0.475, +3.268] |
| detection_reduction / baseline-large / 2000 | double / p50 | 294.335 | +0.227 | [+0.034, +0.431] | [-0.124, +0.478] |
| detection_reduction / baseline-large / 2000 | double / p99 | 347.172 | +1.751 | [-1.623, +3.930] | [-0.466, +2.663] |
| music_estimator / baseline-small / 500 | float / p50 | 3442.131 | -0.136 | [-3.179, +3.040] | [-1.456, +0.472] |
| music_estimator / baseline-small / 500 | float / p99 | 4489.240 | -0.304 | [-1.466, +0.271] | [-3.528, +1.975] |
| music_estimator / baseline-small / 500 | double / p50 | 4177.931 | -0.448 | [-1.347, +0.504] | [-0.869, +0.924] |
| music_estimator / baseline-small / 500 | double / p99 | 5795.728 | -0.559 | [-2.849, +3.856] | [-2.991, +3.517] |

The first50 sample count is identical for fresh 50-frame runs and for prefixes of longer runs. More steady frames do not add observations to that protected early window. Any differences between those cohorts can also reflect launch/time/regime differences. The bootstrap prototype passed synthetic paired-dependence and scale-invariance checks; that is a code check, not validation of deployment coverage or campaign error control. Raw global-mixture p50/p75/p90/p95/p99 results and the declared analysis intent are archived in `ensemble-cdf.json` and `ensemble-cdf-intent.json`.

## Equal-size temporal comparison

To examine timing-regime changes without comparing unequal quantile sample sizes, the second diagnostic compares first50 against last50 from each same longer run. For each original A/A pair, both labels' six runs contribute equally to the early distribution and equally to the late distribution. Inference still has 30 independent pairs; it does not manufacture six independent replicates per pair. Positive change means the last50 is slower. Same-size windows avoid attributing finite-sample quantile changes from unequal window sizes solely to temporal drift.

These are simultaneous sign intervals with the primary conservative 9,360-comparison bound. The extra temporal contrasts fit within the bound's unused dataset slots; the actual explored comparison count remains smaller. They are diagnostic, not a stationarity proof, kernel attribution or permission to discard early observations. All full windows and original first50 protections remain retained.

| Operator / case / actual frames | Precision / quantile | Last50 versus first50 change % | Simultaneous 95% interval % |
| --- | --- | --- | --- |
| fmcw_range_processor / baseline-small / 2000 | float / p50 | -4.421 | [-5.234, -3.675] |
| fmcw_range_processor / baseline-small / 2000 | float / p99 | -4.331 | [-34.355, +64.818] |
| fmcw_range_processor / baseline-small / 2000 | double / p50 | -2.812 | [-3.739, -2.123] |
| fmcw_range_processor / baseline-small / 2000 | double / p99 | -3.481 | [-17.853, +18.046] |
| detection_reduction / baseline-large / 2000 | float / p50 | -1.556 | [-2.085, -1.226] |
| detection_reduction / baseline-large / 2000 | float / p99 | -2.879 | [-12.391, -0.445] |
| detection_reduction / baseline-large / 2000 | double / p50 | -1.016 | [-1.694, -0.674] |
| detection_reduction / baseline-large / 2000 | double / p99 | -2.128 | [-11.300, +3.393] |
| music_estimator / baseline-small / 500 | float / p50 | -20.126 | [-27.093, -9.116] |
| music_estimator / baseline-small / 500 | float / p99 | -10.151 | [-21.557, -4.372] |
| music_estimator / baseline-small / 500 | double / p50 | -2.725 | [-8.131, +0.195] |
| music_estimator / baseline-small / 500 | double / p99 | -5.871 | [-17.276, +2.303] |

## Consequence for choosing frame counts

Choose what needs protection before choosing a sample count: a typical run's tail, the overall equal-run frame-time distribution, startup/early behavior, or rare stalls/maxima. These are related but different questions. The distribution should retain within-run timing dependence and between-run variability rather than treating all frames as independent. Statistical uncertainty should remain distinct from permissible performance regression.

The new data support further offline method evaluation without another GPU sweep. They do not establish a universal robust count, qualify the bootstrap for campaign acceptance, waive early-window protection or authorize production optimization. A replacement policy must declare its estimand, simultaneous error control and regression protection, then receive fresh disjoint validation. Existing correctness/equivalence rules and historical artifacts remain unchanged.
