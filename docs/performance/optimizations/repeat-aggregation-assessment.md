# Repeat aggregation and calibration assessment

Exploratory analysis of the preserved baseline A/A measurements. No new GPU timing runs, production changes or changes to acceptance rules were made. All source report hashes and reconstruction checks are retained in the [analysis artifact](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/repeat-aggregation-analysis.json). This comparison replaces only the median across three isolated repeat processes with their arithmetic mean. Individual run medians/percentiles and the median-based inference across twenty paired observations remain unchanged. An average of run medians is not an average of per-frame runtimes.

## Mean versus median of three repeats

| Metric | Failing rows with median | Failing rows with mean |
| --- | --- | --- |
| median_ms | 15 | 34 |
| p90_ms | 57 | 64 |
| p99_ms | 89 | 111 |
| cold_ms | 6 | 16 |
| setup_ms | 0 | 0 |
| Total | 167/720 | 225/720 |

All 24 sampled operators still have at least one failing row. Eight previously failing rows pass with the mean; 66 previously passing rows fail. No directional familywise interval excludes zero under either aggregation. That absence of detected bias does not establish equivalence: some intervals are much wider than the 2% objective.

The mean is not intrinsically inferior or superior. Its behavior depends on the distribution; extremes affect the mean more strongly, while a median of three can switch between values when their ordering changes. [NIST measures of location](https://www.itl.nist.gov/div898/handbook/eda/section3/eda351.htm) explains the distribution-dependent tradeoff. These empirical results do not justify changing aggregation solely to remove blockers.

## ClutterFilter large double example

| Metric | Aggregation | A1 group median ms | A2 group median ms | Paired delta P90 | Limit | IQR | Limit | Familywise interval of paired change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cold_ms | Median of 3 | 0.751557 | 0.748608 | 8.00% | 20% | 4.94% | 30% | [-9.59%, 7.61%] |
| cold_ms | Mean of 3 | 0.753146 | 0.751906 | 37.89% | 20% | 30.98% | 30% | [-38.87%, 25.96%] |
| median_ms | Median of 3 | 0.664748 | 0.662129 | 14.66% | 5% | 9.70% | 10% | [-14.20%, 12.33%] |
| median_ms | Mean of 3 | 0.671414 | 0.663314 | 14.06% | 5% | 8.74% | 10% | [-10.29%, 14.00%] |
| p90_ms | Median of 3 | 0.699966 | 0.713646 | 26.04% | 5% | 15.98% | 10% | [-24.25%, 25.27%] |
| p90_ms | Mean of 3 | 0.742293 | 0.723278 | 41.59% | 5% | 33.99% | 10% | [-46.18%, 38.64%] |
| p99_ms | Median of 3 | 1.650885 | 1.678795 | 12.44% | 5% | 6.72% | 10% | [-12.07%, 10.14%] |
| p99_ms | Mean of 3 | 1.638697 | 1.633845 | 11.67% | 5% | 7.90% | 10% | [-11.64%, 10.97%] |
| setup_ms | Median of 3 | 4.127565 | 4.121990 | 10.14% | 15% | 6.17% | 25% | [-8.41%, 9.96%] |
| setup_ms | Mean of 3 | 4.295640 | 4.227517 | 6.61% | 15% | 8.88% | 25% | [-7.23%, 5.40%] |

Positive paired change denotes lower A2 latency. Group medians summarize the twenty pair sides; the mean option changes only how the three process repeats are aggregated within each side. The confidence intervals retain the original exact paired-median method and 780-row family (30 rows multiplied by 26 operators). They are not confidence intervals for a population mean.

For pair 6, the individual process medians already differ:

| Role | Three process medians, ms | Median of 3, ms | Mean of 3, ms |
| --- | --- | --- | --- |
| before | 0.654242, 0.598622, 0.611889 | 0.611889 | 0.621584 |
| after | 0.664612, 0.746173, 0.759006 | 0.746173 | 0.723264 |

This pair’s difference shrinks from 21.95% with medians to approximately 16.36% with means, but substantial variation remains. It is present in the underlying process measurements, not created entirely by the final median operation. The cause is unresolved; these observations alone do not identify scheduling, allocator state, clocks or a specific execution mode.

## What the existing calibration does and does not show

- The 5% paired-delta and 10% IQR checks measure dispersion. They do not directly measure the uncertainty of the estimated speedup or the probability of detecting a 2% improvement.
- Several steady median rows are already precise despite their operator being blocked by a tail row. RankOneResidualScreen large float has a familywise paired-change interval of approximately [-0.11%, 0.14%], while ClutterFilter large double spans [-14.20%, 12.33%]. Those workloads should not receive the same diagnosis.
- A process has only fifty measured frames. Linear interpolation places p99 between sorted observations 49 and 50, effectively relying on the slowest two. Averaging three such estimates does not produce the p99 of 150 pooled frames.
- If frames were independent and identically distributed, fifty observations would have a probability of `0.99^50`, approximately 60.5%, of containing no observation above the population p99. Actual frame dependence remains unmeasured.
- Per-frame timing sequences were not persisted in the original JSON reports. We can compare run summaries, but cannot reconstruct pooled quantiles, within-run drift or frame autocorrelation from them. [NIST autocorrelation guidance](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm) explains why dependence matters.

## Proposed next assessment before changing policy

This proposed pilot has now been completed for ClutterFilter large and FMCWRangeProcessor small, both precisions. See the [sampling pilot report](/home/deustice/Projects/holoscan-radar/docs/performance/optimizations/sampling-pilot-1.md) for the calibration comparison, frozen selection, held-out results and timing-regime limitations.

Run a small, fresh pilot spanning a clearly unstable median (ClutterFilter large double), a stable median with a tail blocker (RankOneResidualScreen large float), and a small latency workload (FMCWRangeProcessor small float). Keep existing deployment clocks and power. Preserve raw per-frame timing sequences with process/repeat identifiers and retain setup/cold measurements separately. Record the execution order and timing boundaries.

Compare 50 versus 500 versus 2,000 measured frames, and 3 versus 7 process repeats, using a predeclared, bounded pilot budget. Those counts are experimental sampling levels, not claims that any level is adequate. Alternate A/A at the individual case/repeat level to reduce the time between matched measurements. Run longer frame counts only for steady/tail diagnostics; initialization still needs independent processes. Do not treat all frames as independent statistical replicates or change the warmup/state semantics.

Use disjoint calibration and validation blocks when choosing the estimator and budget. Assess mean and median on calibration evidence; freeze the choice before validation or candidate timing. Retain every repeat, including stalls. Check order effects, drift and confidence width, not merely which estimator passes more rows.

A candidate policy to discuss is to qualify measurement precision relative to the intended improvement: for a 2% steady objective, an A/A familywise interval contained within +/-1% and including zero is a concrete conservative pilot target. This is a proposal, not a new gate. Tail protection needs a separately defined frame-level estimand, sufficient observed tail samples, and a confirmed practical regression margin in relative and/or absolute time before assessing candidates. A wide interval containing zero is inconclusive, not a pass.

Keep the final >2% familywise speed gate and exact-output policy. Any change to aggregation, sampling or calibration/protection policies requires confirmation and fresh baseline/validation measurements; this exploratory reanalysis cannot retrospectively qualify an optimization session.

## Operator summary

| Operator | Rows | Median failures | Mean failures | Newly passing with mean | Newly failing with mean |
| --- | --- | --- | --- | --- | --- |
| beamscan_estimator | 30 | 6 | 8 | 0 | 2 |
| cfar_detector | 30 | 8 | 11 | 1 | 4 |
| cluster_dbscan | 30 | 1 | 4 | 0 | 3 |
| clutter_filter | 30 | 19 | 24 | 0 | 5 |
| ddm_dense_demultiplex | 30 | 6 | 10 | 0 | 4 |
| ddm_power_fold | 30 | 6 | 12 | 0 | 6 |
| ddm_sparse_demultiplex | 30 | 7 | 9 | 1 | 3 |
| doppler_notch | 30 | 6 | 9 | 0 | 3 |
| esprit_estimator | 30 | 13 | 12 | 1 | 0 |
| feature_assembly | 30 | 12 | 16 | 0 | 4 |
| fmcw_doppler_processor | 30 | 12 | 11 | 1 | 0 |
| fmcw_range_doppler_processor | 30 | 8 | 8 | 0 | 0 |
| fmcw_range_processor | 30 | 6 | 9 | 0 | 3 |
| mimo_deinterleave | 30 | 8 | 11 | 0 | 3 |
| multi_source_expansion | 30 | 2 | 7 | 0 | 5 |
| music_estimator | 30 | 13 | 11 | 2 | 0 |
| mvdr_estimator | 30 | 3 | 3 | 1 | 1 |
| phase_shift_beamformer | 30 | 5 | 8 | 0 | 3 |
| rank_one_residual_screen | 30 | 1 | 3 | 1 | 3 |
| snapshot_gather | 30 | 5 | 10 | 0 | 5 |
| subarray_selection | 30 | 9 | 13 | 0 | 4 |
| tdm_dense_compensation | 30 | 4 | 7 | 0 | 3 |
| tdm_sparse_compensation | 30 | 4 | 4 | 0 | 0 |
| virtual_array_assembly | 30 | 3 | 5 | 0 | 2 |
