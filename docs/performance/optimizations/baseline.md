# Baseline assessment: 26 existing performance harnesses

Assessment milestone, 2026-10-12–13 (America/New_York). All 26 existing harnesses were assessed. No production optimization has been performed.

**Acceptance is blocked:** 24 operators have complete exact native repeat matrices, but every sampled operator fails at least one proposed A/A stability limit. RootMUSIC has runtime failures and DetectionReduction has non-exact repeats. These results do not establish that all operators have adequate performance.

Branch: `deustice/optimizations`. Original timing source fingerprint: `3da676743e511eceeb090d9eb02fee3db0a427b2d706ccc18d2821a1d106bebe`. Original build: [original-qualified](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/original-qualified).

Image: `sha256:d4ccdf30d119f555320cb629704a50614dab7cd0eecd1f7513481d52c0b7ddb2`. NVIDIA Thor; current 120W deployment power mode; NVIDIA driver 580.00; CUDA 13.0, nvcc 13.0.88, GCC 13.3.0, installed HSDK 5.0.0, MatX package 1.0.0. SDK and MatX source revisions remain unverified; image, runtime dependency, compiler and MatX header identities are preserved.

SAR/I/O operators without existing recipes are excluded. All measurements use current deployment settings, unpaced synthetic processing, fresh processes per case/precision/repeat, five graph warmup frames, fifty measured frames, and three repeats. The existing discarded two-frame small process-warmup graph precedes each measured graph. Cold means the measured graph’s first frame; process startup and first CUDA context use are outside these metrics. Source publication and terminal GPU synchronization are included; achieved FPS is isolated harness throughput.

Raw child reports are immutable. Absolute values below are medians of repeat report metrics; maxima retain the worst repeat. p90/p99 describe each 50-frame report and are aggregated across repeats rather than pooled frame-percentile inference. A/A stall diagnostics count scalar pair observations, not individual GPU frames.

## Baseline overview

All latency columns are milliseconds for the default baseline case. Native and A/A results have separate coverage.

| Operator | Cases | Float S/M/L | Double S/M/L | Native complete/exact | Baseline-size A/A |
| --- | --- | --- | --- | --- | --- |
| beamscan_estimator | 40 | 0.0707825/0.125723/2.74626 | 0.169019/1.30779/63.1746 | True/True | blocked subset |
| esprit_estimator | 40 | 4.42796/10.0677/6.84885 | 4.3916/18.0051/34.0573 | True/True | blocked subset |
| music_estimator | 52 | 3.32917/4.45038/11.7703 | 4.05071/8.01337/30.8746 | True/True | blocked subset |
| mvdr_estimator | 46 | 0.202969/0.513521/6.03042 | 0.507789/2.61754/22.1652 | True/True | blocked subset |
| phase_shift_beamformer | 36 | 0.055676/0.0586205/0.152404 | 0.149385/0.149968/2.42755 | True/True | blocked subset |
| rank_one_residual_screen | 17 | 0.158107/0.932639/3.31905 | 0.15545/0.86035/3.36446 | True/True | blocked subset |
| root_music_estimator | 40 (partial timing) | 58.1016/191.801/766.342 | 85.4754/416.442/3038.48 | False/False | blocked/not run |
| snapshot_gather | 16 | 0.0748425/0.25169/1.76546 | 0.0758435/0.428681/3.22361 | True/True | blocked subset |
| subarray_selection | 42 | 0.0204675/0.020593/0.0387965 | 0.0206255/0.020616/0.039505 | True/True | blocked subset |
| cluster_dbscan | 29 | 0.65738/2.74951/4.8358 | 1.09201/5.68844/10.8886 | True/True | blocked subset |
| cfar_detector | 64 | 0.101265/0.115274/0.149157 | 0.127001/0.190481/0.313441 | True/True | blocked subset |
| detection_reduction | 30 | 0.204171/0.235351/0.291119 | 0.202661/0.234573/0.293086 | True/False | blocked/not run |
| feature_assembly | 15 | 0.0550595/0.0552685/0.0566015 | 0.0556015/0.0556895/0.0567085 | True/True | blocked subset |
| multi_source_expansion | 8 | 0.123634/0.467581/2.86457 | 0.125341/0.5779/3.74497 | True/True | blocked subset |
| clutter_filter | 21 | 0.663382/0.524447/0.551424 | 0.518424/0.549994/0.660651 | True/True | blocked subset |
| ddm_dense_demultiplex | 11 | 0.032412/0.098694/1.18145 | 0.0328605/0.350096/2.60487 | True/True | blocked subset |
| ddm_power_fold | 3 | 0.0313975/0.067949/0.835461 | 0.0336025/0.231485/1.73335 | True/True | blocked subset |
| ddm_sparse_demultiplex | 22 | 0.0629445/0.132217/0.975952 | 0.0632455/0.250393/1.59238 | True/True | blocked subset |
| doppler_notch | 26 | 0.041546/0.0881105/0.438791 | 0.04181/0.0858425/0.453026 | True/True | blocked subset |
| fmcw_doppler_processor | 43 | 0.0329125/0.0329535/0.0637405 | 0.035472/0.0643565/0.147759 | True/True | blocked subset |
| fmcw_range_doppler_processor | 66 | 0.049861/0.0508705/0.0827865 | 0.0664305/0.0993335/0.275501 | True/True | blocked subset |
| fmcw_range_processor | 43 | 0.0269815/0.0270975/0.027523 | 0.0449585/0.061472/0.156301 | True/True | blocked subset |
| mimo_deinterleave | 3 | 0.0300235/0.0635925/1.18389 | 0.030069/0.0977965/2.46224 | True/True | blocked subset |
| tdm_dense_compensation | 12 | 0.0326715/0.581324/9.29193 | 0.0352685/1.31175/20.9399 | True/True | blocked subset |
| tdm_sparse_compensation | 23 | 0.064361/0.385468/2.69929 | 0.063787/0.672491/5.06896 | True/True | blocked subset |
| virtual_array_assembly | 12 | 0.0203935/0.353991/3.64782 | 0.0205465/0.587801/7.1379 | True/True | blocked subset |

## Correctness and unresolved findings

The final scoped-trust non-performance rerun executed 1,097 tests: 1,090 passed and seven unchanged visualization/application tests failed (`IMPLICIT_DEVICE_ZERO_REJECTED`). Native performance verification passed; the earlier eight-failure rerun retained the Git-ownership failure as separate environment evidence. The earlier 1,095-test run initially failed ten; three stale recipe-inventory expectations were corrected. The unchanged application/visualization failures are recorded separately from operator performance findings. All 72 performance smoke tests passed on the final preparation build. Pinned all-file pre-commit checks passed with the compatible Node 22 wrapper and the same hook pins.

The complete RootMUSIC native matrix reproduces `cusolverDnXgeev` status 7 in 31 float cases across both output-capture repeats (62 failures). All double captures complete. The unchanged production sources are bound by the separate verification build receipt. The default model order is auto (-1). Timing and native initialization/data contexts differ; the failing frame and cause remain unresolved. This blocks affected correctness validation and is not a confirmed SDK defect. No performance change repairs it.

[RootMUSIC full native matrix failure evidence](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/root-native-full-matrix-finding.json)

The warmed float spatial-smoothing-medium and spatial-smoothing-large timing cases, including explicitly primed cached variants, reproduced status 7 in all three repeats. Skipped double children are assessed separately; failed float cases remain blocked. [Warmed timing failures](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/root-timing-failure-finding-v2.json).

[Existing test failures](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/baseline-findings.json)

The timing reports embed an unknown Git commit and a misleading default clean field because container Git rejects checkout ownership. The independently recorded actual commit, full source fingerprint, image and artifact hashes bind this baseline. Raw report fields were not rewritten. Later A/A invocations use process-scoped Git trust.

[Git provenance gap](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/report-git-provenance-gap.json)

## Calibration review

[Focused sampling pilot](/home/deustice/Projects/holoscan-radar/docs/performance/optimizations/sampling-pilot-1.md) adds fresh reduction/FFT measurements with raw frame timing and a separately frozen thirty-pair validation block. Original baseline evidence and acceptance policies are unchanged.

[Mean-versus-median investigation](/home/deustice/Projects/holoscan-radar/docs/performance/optimizations/repeat-aggregation-assessment.md) compares both aggregations using the preserved individual process reports and proposes further sampling experiments. Existing policies and blockers remain unchanged.

[Detailed A/A measurements](/home/deustice/Projects/holoscan-radar/docs/performance/optimizations/aa-calibration.md) show all 720 rows, both absolute timing groups, measured variability against each limit, and the twenty individual pairs for ClutterFilter large double.

Twenty alternating A/A pairs assessed the three baseline sizes in both precisions for each of 24 operators. All 24 failed at least one proposed limit; no assessment has insufficient interval resolution. Across 720 metric/case rows, 167 are blocked: 89 p99, 57 p90, 15 median, and 6 cold. An ordered-distribution audit retains aggregated side observations, quartiles, ranges and first/second-half medians. Clutter median half-drift spans 0.44–4.46%; MUSIC includes a median-row half-drift of -7.02%. These descriptive diagnostics neither identify the noise mechanism nor establish stationarity or exclude multiple modes. [Ordered distribution audit](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/aa-ordered-distribution-audit-v2.json). The limits remain unchanged.

| Operator | Blocked metric/case rows | Failing metrics | Disposition |
| --- | --- | --- | --- |
| beamscan_estimator | 6/30 | cold_ms: 2, p90_ms: 1, p99_ms: 3 | blocked |
| esprit_estimator | 13/30 | median_ms: 4, p90_ms: 5, p99_ms: 4 | blocked |
| music_estimator | 13/30 | cold_ms: 1, median_ms: 3, p90_ms: 4, p99_ms: 5 | blocked |
| mvdr_estimator | 3/30 | cold_ms: 1, p99_ms: 2 | blocked |
| phase_shift_beamformer | 5/30 | p90_ms: 1, p99_ms: 4 | blocked |
| rank_one_residual_screen | 1/30 | p99_ms: 1 | blocked |
| root_music_estimator | not sampled | native validation incomplete or non-exact | blocked |
| snapshot_gather | 5/30 | p90_ms: 2, p99_ms: 3 | blocked |
| subarray_selection | 9/30 | p90_ms: 3, p99_ms: 6 | blocked |
| cluster_dbscan | 1/30 | p99_ms: 1 | blocked |
| cfar_detector | 8/30 | cold_ms: 1, p90_ms: 2, p99_ms: 5 | blocked |
| detection_reduction | not sampled | native validation incomplete or non-exact | blocked |
| feature_assembly | 12/30 | p90_ms: 6, p99_ms: 6 | blocked |
| multi_source_expansion | 2/30 | p99_ms: 2 | blocked |
| clutter_filter | 19/30 | cold_ms: 1, median_ms: 6, p90_ms: 6, p99_ms: 6 | blocked |
| ddm_dense_demultiplex | 6/30 | p90_ms: 3, p99_ms: 3 | blocked |
| ddm_power_fold | 6/30 | p90_ms: 3, p99_ms: 3 | blocked |
| ddm_sparse_demultiplex | 7/30 | p90_ms: 3, p99_ms: 4 | blocked |
| doppler_notch | 6/30 | p90_ms: 2, p99_ms: 4 | blocked |
| fmcw_doppler_processor | 12/30 | median_ms: 1, p90_ms: 5, p99_ms: 6 | blocked |
| fmcw_range_doppler_processor | 8/30 | median_ms: 1, p90_ms: 2, p99_ms: 5 | blocked |
| fmcw_range_processor | 6/30 | p99_ms: 6 | blocked |
| mimo_deinterleave | 8/30 | p90_ms: 4, p99_ms: 4 | blocked |
| tdm_dense_compensation | 4/30 | p90_ms: 2, p99_ms: 2 | blocked |
| tdm_sparse_compensation | 4/30 | p90_ms: 2, p99_ms: 2 | blocked |
| virtual_array_assembly | 3/30 | p90_ms: 1, p99_ms: 2 | blocked |

The 29-pair proposal resolves the complete-matrix family more conservatively than the 20-pair starting budget; it is not evidence that 2% effects have adequate power or that unstable calibration will pass. Do not approve a candidate using these blocked subsets. Full static/live/optional calibration remains mandatory.

## Consolidated session proposal

Proposal: **29 alternating A/B pairs**, selected from complete matrices with a 26-operator family multiplier. This is a confidence-resolution budget, not a power guarantee. The 20-pair A/A assessment covers only three baseline sizes; complete static/live/optional matrices need fresh calibration before any production candidate. Failed native validation prevents that operator’s pilot; none of these subsets authorize dropping cases.

Every output port and precision is proposed for exact comparison of values, indices, masks, descriptor content, shapes, ordering, emission coordinates, errors and temporal state. Ownership remains a protected requirement; copied snapshots alone do not validate retained native lifetime, as detailed below. Objective: every predeclared target must exceed 2% median latency improvement under the familywise gate. Protect median, p90, p99, setup, cold and applicable live-transition latency across complete measured matrices and affected consumers. Retain max/range/stall diagnostics.

Calibration P90-delta/IQR limits: steady 5%/10%; setup 15%/25%; cold 20%/30%; override 25%/40%. No clock or power changes. Limits: twenty attempted candidates, stop at three consecutive unsuccessful candidates or an exhausted actionable queue; blocked attempts count toward the attempt limit without increasing that streak.

Each config permits its operator directory and declares policies for the other 25 allowable consumers to account conservatively for the campaign family. Before editing, each candidate must audit actual transitive consumers and declare targets, expected gain, files and mechanism. Shared edits outside the listed scope require new confirmation. Native failure, repeat mismatch, missing coverage or unstable calibration blocks affected acceptance.

Configuration hashes and full case IDs: [Versioned unconfirmed configuration index](/home/deustice/Projects/holoscan-radar/docs/performance/optimizations/session-configs/index.json). Production edits require the consolidated user confirmation requested by the campaign plan and operator-perf skill.

The existing harness executes the Params-fallback frame but excludes it from latency statistics. Native captures verify fallback correctness; a separate fallback-latency metric is unavailable. The proposal protects reported changed-frame/revisit metrics and makes no independent fallback-latency claim. [Fallback timing coverage](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/live-fallback-timing-coverage.json).

## Attribution review

The research priority below ranks avoidable recurring work and mechanism confidence. Every entry remains blocked under current calibration; this ordering does not authorize production edits or claim a speedup. Estimates refer to profiled compute ranges, which include profiling overhead and differ from the end-to-end unprofiled latency. Independent medians are not additive.

| Priority | Operator | Evidence and removable-cost estimate | Affected cases/consumers | Confidence |
| --- | --- | --- | --- | --- |
| 1 | ClutterFilterOp | Mean transform allocates each tick. `cudaMallocAsync` appears in all 50 steady ranges of all six profiles and occupies approximately 81.5–83.9% of profiled compute median. Reusing private mean scratch could remove much of that cost; no unprofiled gain is validated. | Mean-subtraction/window cases, both precisions; shared header consumer FMCWRangeDopplerProcessorOp. | High recurrence/source confidence; gain and exact reduction/lifetime preservation untested. |
| 2 | VirtualArrayAssemblyOp | Host virtual plan rebuilt before the existing device-table cache check. Small-float outside-API time is 9.76/16.05 µs (61% of profiled compute); this is an upper envelope for all non-API work, not a measured planner cost. | All recurring layouts; wavelength match/fallback and optional wiring protected. | High source recurrence confidence; removable fraction unknown within the envelope. |
| 3 | DDMSparseDemultiplexOp | Host virtual plan rebuilt before device-table reuse. Small-float outside-API time is 18.51/55.11 µs (34% envelope); plan construction has not been isolated from other host work/waits. | Repeated layouts; windows, wavelength, band/fold rejection, mask/geometry outputs. | High recurrence confidence; lower quantitative confidence. |
| 4 | TDMSparseCompensationOp | Same host-plan rebuild pattern. Small-float outside-API time is 19.49/56.72 µs (34% envelope). Existing error synchronization enforces rejection behavior. | Repeated layouts; schedule/window/wavelength overrides and geometry/mask outputs. | High recurrence confidence; lower quantitative confidence. |
| 5 | MUSIC / ESPRIT | Repeated allocation/free and synchronization appear in all 50 steady ranges. Existing cache-clearing code protects scratch-pointer lifetimes and global MatX tracking; removing those safeguards alone is unsafe. | Eigendecomposition/smoothing paths, both precisions and shape/live transitions. | High recurring-cost confidence; no qualified safe replacement or removable-fraction estimate. |
| 6 | FeatureAssemblyOp | Rebuilds a small feature-id vector although upload is already cached. Small-float outside-API envelope is 6.06/36.88 µs; only a fraction could be vector construction. A gain above 2% is uncertain. | Feature selection, score/AoA inputs, ordering and optional wiring. | High recurrence confidence; weak gain evidence. |

For the remaining operators, profiling and source review did not establish a comparably strong bounded change. This is assessment-only evidence, not proof that no optimization exists. RootMUSIC's absolute runtime does not make it the first optimization target: solver failures block affected validation, and its serial general-eigensolver path already has deferred-work notes. DetectionReduction's baseline output differences block exact validation.

## beamscan_estimator

Editable scope: `operators/angle_estimation/beamscan_estimator_op`. Output ports (both precisions exact): `angle_spectrum`, `aoa_estimates`.

Application consumer inventory: `cluster_dbscan`, `detection_reduction`, `feature_assembly`, `multi_source_expansion`, `mvdr_estimator`, `rank_one_residual_screen`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/beamscan_estimator.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 6; insufficient pair budget False; worst P90 paired delta 31.93%, worst baseline IQR 13%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/beamscan_estimator/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0707825 | 0.0760181 | 0.0916563 | 0.12124 | 4.2667 | 1.53062 | — | 13298.4 |
| baseline-small | double | 0.169019 | 0.170931 | 0.182328 | 0.189278 | 4.30649 | 1.53299 | — | 5790.39 |
| baseline-medium | float | 0.125723 | 0.127375 | 0.135763 | 0.153778 | 5.03268 | 0.886964 | — | 7750.86 |
| baseline-medium | double | 1.30779 | 1.30972 | 1.31973 | 1.33096 | 5.62099 | 2.0472 | — | 762.655 |
| baseline-large | float | 2.74626 | 2.75202 | 2.76465 | 2.84981 | 7.33469 | 3.76338 | — | 363.42 |
| baseline-large | double | 63.1746 | 63.1975 | 63.2303 | 63.263 | 10.364 | 64.3335 | — | 15.8267 |
| tapered-medium | float | 0.125959 | 0.127462 | 0.129495 | 0.162991 | 4.86846 | 0.997011 | — | 7724.16 |
| tapered-medium | double | 1.30749 | 1.31142 | 1.31687 | 1.32399 | 5.6027 | 2.19087 | — | 762.658 |
| tapered-large | float | 2.74728 | 2.75209 | 2.76085 | 2.81744 | 8.03938 | 3.87521 | — | 363.506 |
| tapered-large | double | 63.1816 | 63.2122 | 63.23 | 63.2559 | 9.4875 | 64.3194 | — | 15.825 |
| geometry-small | float | 0.0727035 | 0.074667 | 0.0839031 | 0.098991 | 4.49369 | 1.31035 | 0.913288 | 13069.4 |
| geometry-small | double | 0.170899 | 0.172637 | 0.186631 | 0.195528 | 4.40651 | 1.72624 | 1.06021 | 5725.71 |
| geometry-medium | float | 0.127204 | 0.128834 | 0.135142 | 0.142547 | 5.1557 | 0.939335 | 1.05993 | 7642.58 |
| geometry-medium | double | 1.30892 | 1.31137 | 1.32197 | 1.33189 | 5.98595 | 2.11818 | 2.31669 | 761.649 |
| geometry-large | float | 2.74837 | 2.75144 | 2.75906 | 2.79734 | 7.65087 | 3.86998 | 3.6162 | 363.432 |
| geometry-large | double | 63.1707 | 63.1802 | 63.2287 | 63.2996 | 10.0864 | 64.4542 | 64.2297 | 15.8282 |
| az-small | float | 0.0731295 | 0.0745966 | 0.0761105 | 0.087704 | 4.28051 | 2.02554 | 0.769132 | 13109.6 |
| az-small | double | 0.17101 | 0.1736 | 0.185647 | 0.258046 | 4.40946 | 1.4076 | 1.01019 | 5706.53 |
| az-medium | float | 0.128019 | 0.130001 | 0.139223 | 0.14725 | 4.90802 | 0.846095 | 1.02007 | 7643.31 |
| az-medium | double | 1.30889 | 1.3108 | 1.31286 | 1.39223 | 6.00556 | 2.48567 | 2.3097 | 762.296 |
| az-large | float | 2.74736 | 2.75141 | 2.7569 | 2.7634 | 7.56476 | 3.80957 | 3.72065 | 363.504 |
| az-large | double | 63.1721 | 63.1889 | 63.2314 | 63.2564 | 9.90305 | 64.4405 | 64.1482 | 15.8277 |
| el-small | float | 0.0729495 | 0.0754449 | 0.0820682 | 0.093816 | 4.53109 | 1.43696 | 0.996715 | 13106.7 |
| el-small | double | 0.170944 | 0.172628 | 0.183038 | 0.197946 | 4.38366 | 1.65775 | 1.06383 | 5739.06 |
| el-medium | float | 0.12757 | 0.128798 | 0.13024 | 0.139065 | 5.27683 | 0.861623 | 0.924234 | 7658.81 |
| el-medium | double | 1.30887 | 1.31206 | 1.32068 | 1.36974 | 5.48582 | 2.02178 | 2.15465 | 761.752 |
| el-large | float | 2.7495 | 2.75315 | 2.76408 | 3.36481 | 8.35651 | 3.76536 | 4.14298 | 363.249 |
| el-large | double | 63.1794 | 63.2088 | 63.2837 | 63.6397 | 10.4563 | 64.3545 | 64.1863 | 15.8213 |
| taper-small | float | 0.072167 | 0.0745109 | 0.0821013 | 0.092611 | 4.39725 | 1.35438 | 0.962586 | 13227.9 |
| taper-small | double | 0.170899 | 0.17302 | 0.186826 | 0.259602 | 4.41111 | 1.40873 | 1.16845 | 5722.19 |
| taper-medium | float | 0.12719 | 0.129185 | 0.138247 | 0.142954 | 5.02264 | 0.854973 | 0.987279 | 7649.1 |
| taper-medium | double | 1.30938 | 1.31199 | 1.32355 | 1.36158 | 5.65225 | 2.12482 | 2.2965 | 761.309 |
| taper-large | float | 2.74812 | 2.75335 | 2.76078 | 2.76701 | 8.10893 | 3.7409 | 3.66792 | 363.365 |
| taper-large | double | 63.1696 | 63.184 | 63.241 | 63.2748 | 10.0968 | 64.4155 | 64.2579 | 15.8283 |
| peak-small | float | 0.0712095 | 0.0734655 | 0.0895562 | 0.097722 | 4.31694 | 1.3331 | 0.080186 | 13412.4 |
| peak-small | double | 0.170741 | 0.172939 | 0.175395 | 0.213473 | 4.32292 | 1.66722 | 0.178981 | 5744.54 |
| peak-medium | float | 0.125977 | 0.127715 | 0.138098 | 0.141805 | 4.86596 | 1.44272 | 0.136881 | 7745.49 |
| peak-medium | double | 1.30827 | 1.31107 | 1.31981 | 1.32358 | 5.80628 | 2.04816 | 1.32185 | 762.29 |
| peak-large | float | 2.7449 | 2.74897 | 2.75498 | 2.76228 | 7.1309 | 4.29165 | 2.75497 | 363.854 |
| peak-large | double | 63.1674 | 63.1742 | 63.22 | 64.1846 | 9.78019 | 64.3329 | 63.1934 | 15.8293 |
| tapered-peak-medium | float | 0.12663 | 0.127805 | 0.133658 | 0.160398 | 4.98994 | 1.53288 | 0.138268 | 7692.63 |
| tapered-peak-medium | double | 1.30788 | 1.31071 | 1.32383 | 1.33317 | 5.46002 | 2.04145 | 1.32355 | 762.097 |
| geometry-small-cached | float | 0.0727785 | 0.0796672 | 0.0893235 | 0.098742 | 4.29908 | 1.37428 | 0.110954 | 12964.3 |
| geometry-small-cached | double | 0.170617 | 0.174369 | 0.182311 | 0.201564 | 4.45544 | 1.70329 | 0.209676 | 5731.96 |
| geometry-medium-cached | float | 0.127107 | 0.128399 | 0.134527 | 0.137463 | 5.15624 | 0.898584 | 0.171324 | 7658.4 |
| geometry-medium-cached | double | 1.30951 | 1.31244 | 1.31719 | 1.32241 | 5.67537 | 2.13156 | 1.35305 | 761.53 |
| geometry-large-cached | float | 2.7506 | 2.75588 | 2.77396 | 2.7918 | 7.85981 | 3.88581 | 2.79091 | 363.059 |
| geometry-large-cached | double | 63.1734 | 63.2 | 63.3018 | 63.9915 | 9.90554 | 64.2029 | 63.217 | 15.8247 |
| az-small-cached | float | 0.07294 | 0.0773585 | 0.0906889 | 0.100898 | 4.41471 | 1.23649 | 0.08625 | 12947.5 |
| az-small-cached | double | 0.170972 | 0.172647 | 0.18811 | 0.193871 | 4.43644 | 1.40625 | 0.177111 | 5726.53 |
| az-medium-cached | float | 0.127199 | 0.128794 | 0.135392 | 0.141351 | 5.15093 | 0.963585 | 0.147075 | 7654.19 |
| az-medium-cached | double | 1.30875 | 1.31142 | 1.3187 | 1.3341 | 6.54056 | 2.10704 | 1.32143 | 761.92 |
| az-large-cached | float | 2.7506 | 2.75635 | 2.76381 | 2.76592 | 9.43725 | 3.81018 | 2.76628 | 363.096 |
| az-large-cached | double | 63.1702 | 63.1847 | 63.2245 | 63.2449 | 10.3277 | 64.1914 | 63.1912 | 15.8278 |
| el-small-cached | float | 0.0724725 | 0.074611 | 0.091448 | 0.095084 | 4.29779 | 1.20621 | 0.085667 | 13070.3 |
| el-small-cached | double | 0.171232 | 0.172944 | 0.184478 | 0.19249 | 4.41077 | 1.65295 | 0.178742 | 5710.34 |
| el-medium-cached | float | 0.127117 | 0.130498 | 0.136966 | 0.140667 | 5.27419 | 0.900659 | 0.147731 | 7643.18 |
| el-medium-cached | double | 1.30964 | 1.31248 | 1.32428 | 1.33009 | 6.30864 | 2.10807 | 1.32397 | 761.362 |
| el-large-cached | float | 2.74871 | 2.75521 | 2.76007 | 2.7794 | 8.49226 | 3.79044 | 2.76228 | 363.384 |
| el-large-cached | double | 63.1723 | 63.1834 | 63.2111 | 63.2266 | 10.222 | 64.5186 | 63.1904 | 15.8275 |
| taper-small-cached | float | 0.0728655 | 0.0780593 | 0.0924991 | 0.1095 | 4.48218 | 1.50435 | 0.113519 | 12969.6 |
| taper-small-cached | double | 0.170996 | 0.174343 | 0.180549 | 0.190056 | 4.41862 | 1.67115 | 0.209352 | 5718.02 |
| taper-medium-cached | float | 0.12788 | 0.128893 | 0.137928 | 0.144167 | 5.0788 | 1.08537 | 0.176325 | 7640.95 |
| taper-medium-cached | double | 1.30891 | 1.31069 | 1.31576 | 1.31664 | 6.11477 | 2.24207 | 1.34686 | 761.876 |
| taper-large-cached | float | 2.74973 | 2.75613 | 2.76325 | 2.7709 | 7.92134 | 3.85062 | 2.79782 | 363.228 |
| taper-large-cached | double | 63.1773 | 63.1978 | 63.2234 | 63.947 | 9.92247 | 64.3366 | 63.2147 | 15.8265 |
| peak-small-cached | float | 0.0709635 | 0.0755691 | 0.0964912 | 0.103214 | 4.44875 | 1.35121 | 0.079631 | 13166.1 |
| peak-small-cached | double | 0.169612 | 0.17124 | 0.179946 | 0.196983 | 4.2777 | 1.83485 | 0.178 | 5770.95 |
| peak-medium-cached | float | 0.126074 | 0.127783 | 0.133584 | 0.140926 | 4.85653 | 0.865909 | 0.136546 | 7717.29 |
| peak-medium-cached | double | 1.30857 | 1.31015 | 1.31567 | 1.32256 | 6.0717 | 2.10879 | 1.3244 | 762.412 |
| peak-large-cached | float | 2.74635 | 2.75058 | 2.75617 | 2.78342 | 7.29767 | 3.81176 | 2.75504 | 363.697 |
| peak-large-cached | double | 63.1694 | 63.1925 | 63.2307 | 63.2786 | 10.1706 | 64.3666 | 63.1914 | 15.8279 |
| tapered-peak-medium-cached | float | 0.126657 | 0.127899 | 0.14087 | 0.148166 | 4.79616 | 0.898779 | 0.135185 | 7691.74 |
| tapered-peak-medium-cached | double | 1.30736 | 1.30964 | 1.31836 | 1.32117 | 5.64683 | 2.1233 | 1.32323 | 762.731 |
| optional-minimal-small | float | 0.068648 | 0.0717944 | 0.0755993 | 0.094297 | 3.98805 | 1.33956 | — | 13711.4 |
| optional-minimal-small | double | 0.167593 | 0.170595 | 0.179918 | 0.18926 | 3.89012 | 1.4328 | — | 5831.27 |
| optional-minimal-medium | float | 0.124213 | 0.125904 | 0.135378 | 0.141194 | 4.35349 | 0.868734 | — | 7823.49 |
| optional-minimal-medium | double | 1.3059 | 1.31004 | 1.31712 | 1.33799 | 5.33903 | 2.04137 | — | 763.361 |
| optional-minimal-large | float | 2.74493 | 2.75151 | 2.75732 | 2.76489 | 6.40308 | 3.63018 | — | 363.647 |
| optional-minimal-large | double | 63.1667 | 63.1842 | 63.2211 | 63.2343 | 8.14165 | 64.7858 | — | 15.8289 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/beamscan_estimator/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/beamscan_estimator/repeat-review.json).

## esprit_estimator

Editable scope: `operators/angle_estimation/esprit_estimator_op`. Output ports (both precisions exact): `aoa_estimates`, `doa_multi`, `eigenvalues`, `source_count`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/esprit_estimator.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 13; insufficient pair budget False; worst P90 paired delta 19.96%, worst baseline IQR 13.93%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/esprit_estimator/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 4.42796 | 5.44815 | 5.89398 | 6.65204 | 5.47814 | 4.45671 | — | 208.572 |
| baseline-small | double | 4.3916 | 5.33546 | 5.79079 | 6.6928 | 5.5069 | 4.1754 | — | 217.998 |
| baseline-medium | float | 10.0677 | 10.7627 | 10.8618 | 11.5528 | 5.40142 | 10.8815 | — | 96.7847 |
| baseline-medium | double | 18.0051 | 19.3642 | 19.6864 | 19.9039 | 5.61554 | 17.7003 | — | 54.4512 |
| baseline-large | float | 6.84885 | 7.4425 | 8.25312 | 8.45603 | 6.26061 | 6.48649 | — | 145.241 |
| baseline-large | double | 34.0573 | 35.532 | 35.8254 | 36.2956 | 7.07583 | 34.2631 | — | 29.0095 |
| parameter-medium | float | 4.22299 | 5.04501 | 5.3243 | 5.77344 | 5.57762 | 4.8636 | — | 225.058 |
| parameter-medium | double | 6.23553 | 7.03427 | 7.46853 | 8.31611 | 5.76263 | 6.97263 | — | 155.237 |
| parameter-large | float | 6.35751 | 6.95558 | 7.4711 | 7.56142 | 6.56628 | 6.7147 | — | 157.324 |
| parameter-large | double | 13.13 | 14.1242 | 14.2514 | 14.7947 | 7.62919 | 13.0041 | — | 75.7789 |
| array-geometry-small | float | 4.11388 | 4.91153 | 5.32603 | 6.0818 | 5.60725 | 4.48903 | 4.25791 | 232.587 |
| array-geometry-small | double | 3.97502 | 4.64215 | 5.53422 | 5.89344 | 5.66325 | 3.99405 | 4.32254 | 244.725 |
| array-geometry-medium | float | 10.572 | 11.3159 | 11.467 | 11.7845 | 5.77727 | 11.2119 | 10.8815 | 94.8703 |
| array-geometry-medium | double | 17.621 | 18.6473 | 19.3139 | 19.6632 | 5.66757 | 17.3744 | 18.7339 | 56.0671 |
| array-geometry-large | float | 6.61123 | 7.23913 | 7.61759 | 8.37616 | 6.32441 | 6.12475 | 6.69044 | 151.903 |
| array-geometry-large | double | 34.1891 | 35.144 | 35.6729 | 37.0735 | 7.27933 | 34.1661 | 35.0323 | 29.1534 |
| spatial-smoothing-small | float | 3.91783 | 4.49524 | 4.89737 | 5.57105 | 5.75782 | 4.24816 | 3.25311 | 249.894 |
| spatial-smoothing-small | double | 3.64004 | 4.65363 | 5.36983 | 5.42957 | 5.66091 | 3.87496 | 4.65321 | 251.329 |
| spatial-smoothing-medium | float | 10.4377 | 11.0358 | 11.2567 | 11.5222 | 5.72723 | 10.218 | 4.40325 | 94.4488 |
| spatial-smoothing-medium | double | 17.5863 | 18.9952 | 19.186 | 19.5619 | 5.8844 | 18.1123 | 25.6616 | 55.9208 |
| spatial-smoothing-large | float | 6.80106 | 7.27837 | 8.05483 | 8.38264 | 6.56363 | 6.54987 | 6.35244 | 146.028 |
| spatial-smoothing-large | double | 33.7472 | 35.0392 | 35.4778 | 35.7069 | 7.68235 | 33.3859 | 38.4345 | 29.3697 |
| num-sources-small | float | 3.78706 | 4.54459 | 5.21904 | 5.98952 | 5.52263 | 4.77374 | 3.44727 | 248.825 |
| num-sources-small | double | 4.29227 | 5.01555 | 5.10204 | 5.55842 | 5.54946 | 4.16856 | 3.69705 | 234.976 |
| num-sources-medium | float | 10.5635 | 11.1077 | 11.609 | 11.8819 | 5.63522 | 10.5635 | 4.75876 | 93.2661 |
| num-sources-medium | double | 17.5453 | 18.5835 | 19.1924 | 19.7034 | 5.46568 | 17.5138 | 6.63583 | 56.0984 |
| num-sources-large | float | 6.60443 | 7.31944 | 7.98489 | 8.25916 | 6.29852 | 6.38108 | 5.46866 | 150.822 |
| num-sources-large | double | 33.5946 | 34.9601 | 35.284 | 35.5937 | 7.59152 | 33.8268 | 13.1741 | 29.4508 |
| source-count-pfa-small | float | 3.84775 | 4.42457 | 4.81838 | 5.29301 | 5.57047 | 4.0142 | 4.22116 | 250.337 |
| source-count-pfa-small | double | 3.61253 | 4.62854 | 5.45286 | 5.6235 | 5.51435 | 4.9145 | 3.6292 | 250.189 |
| source-count-pfa-medium | float | 10.5564 | 10.974 | 11.3871 | 12.216 | 5.59116 | 10.0727 | 10.9494 | 95.8236 |
| source-count-pfa-medium | double | 17.8289 | 19.3797 | 19.6008 | 19.8649 | 5.68967 | 18.0684 | 18.3623 | 54.786 |
| source-count-pfa-large | float | 6.03644 | 7.56026 | 7.67532 | 8.59007 | 6.4999 | 6.50058 | 5.99441 | 155.076 |
| source-count-pfa-large | double | 33.7411 | 35.0662 | 35.2963 | 35.6787 | 7.63321 | 33.8685 | 33.5513 | 29.3576 |
| max-sources-small | float | 3.67414 | 4.44197 | 4.52458 | 5.20907 | 5.5927 | 4.83711 | 3.64724 | 250.558 |
| max-sources-small | double | 3.81762 | 4.6341 | 5.02739 | 5.9409 | 5.57015 | 4.22716 | 4.17701 | 248.081 |
| max-sources-medium | float | 10.652 | 10.9781 | 11.0743 | 11.1039 | 5.60772 | 10.5633 | 4.61905 | 95.7095 |
| max-sources-medium | double | 17.9723 | 19.491 | 19.7207 | 19.8994 | 5.62647 | 17.4559 | 8.52411 | 54.3382 |
| max-sources-large | float | 7.06444 | 7.72109 | 8.31358 | 8.52212 | 6.40301 | 6.59769 | 6.87564 | 140.09 |
| max-sources-large | double | 34.2708 | 35.4808 | 35.6636 | 36.0687 | 7.66414 | 33.3132 | 16.5571 | 28.8919 |
| transition-medium | float | 3.52208 | 4.33544 | 5.13985 | 5.55696 | 5.60569 | 4.4381 | 4.08887 | 260.516 |
| transition-medium | double | 6.06973 | 7.15475 | 7.50763 | 8.10395 | 5.64196 | 6.01228 | 6.66301 | 155.975 |
| array-geometry-small-cached | float | 4.1805 | 5.16423 | 5.76453 | 7.57753 | 5.60543 | 4.56601 | 3.78853 | 244.023 |
| array-geometry-small-cached | double | 4.25634 | 5.11477 | 5.31588 | 5.58336 | 5.70327 | 5.01482 | 4.22369 | 226.679 |
| array-geometry-medium-cached | float | 10.4458 | 10.9757 | 11.2799 | 12.0701 | 5.67157 | 10.9744 | 10.5135 | 95.0829 |
| array-geometry-medium-cached | double | 17.3938 | 18.2302 | 18.9772 | 19.3085 | 5.78648 | 18.8599 | 17.3807 | 56.4522 |
| array-geometry-large-cached | float | 6.30325 | 7.03715 | 7.48457 | 7.91662 | 6.4938 | 7.04656 | 6.45504 | 152.845 |
| array-geometry-large-cached | double | 33.6522 | 35.0537 | 35.1712 | 36.0738 | 7.5076 | 33.6573 | 34.3986 | 29.415 |
| spatial-smoothing-small-cached | float | 4.12514 | 4.92686 | 5.06881 | 5.70594 | 5.53443 | 3.87064 | 3.10786 | 226.567 |
| spatial-smoothing-small-cached | double | 4.35515 | 5.0694 | 5.28653 | 5.41967 | 5.64927 | 4.3876 | 4.17195 | 220.335 |
| spatial-smoothing-medium-cached | float | 10.5946 | 11.3928 | 11.6574 | 11.9734 | 5.94574 | 4.30511 | 4.14438 | 93.5593 |
| spatial-smoothing-medium-cached | double | 17.4625 | 18.8704 | 19.2261 | 19.4371 | 6.69311 | 25.0542 | 25.2744 | 56.0894 |
| spatial-smoothing-large-cached | float | 7.07013 | 7.58065 | 8.02231 | 8.42192 | 6.36391 | 7.35936 | 6.2861 | 141.976 |
| spatial-smoothing-large-cached | double | 33.6581 | 34.9999 | 35.5547 | 36.3168 | 7.58002 | 38.6626 | 37.8762 | 29.3505 |
| num-sources-small-cached | float | 4.06006 | 4.91761 | 5.5883 | 6.30834 | 5.68501 | 4.01601 | 3.15963 | 233.609 |
| num-sources-small-cached | double | 4.12665 | 5.11737 | 5.67038 | 6.03435 | 5.60493 | 5.18458 | 3.69 | 225.639 |
| num-sources-medium-cached | float | 10.1045 | 10.9363 | 11.1608 | 11.5428 | 5.79245 | 4.21463 | 3.90719 | 96.6943 |
| num-sources-medium-cached | double | 17.4432 | 18.5092 | 19.2072 | 19.8493 | 5.86275 | 6.01593 | 6.74117 | 56.2677 |
| num-sources-large-cached | float | 6.73559 | 7.65036 | 8.02645 | 9.1513 | 6.43373 | 5.71074 | 5.96061 | 140.222 |
| num-sources-large-cached | double | 34.1275 | 35.1289 | 35.779 | 35.9983 | 7.26805 | 13.6853 | 12.7136 | 29.1131 |
| source-count-pfa-small-cached | float | 4.30231 | 5.34127 | 5.68146 | 5.90805 | 5.67346 | 4.51301 | 4.1037 | 226.816 |
| source-count-pfa-small-cached | double | 3.99426 | 4.45586 | 5.14167 | 5.95945 | 5.85318 | 4.15752 | 4.04332 | 244.969 |
| source-count-pfa-medium-cached | float | 11.063 | 11.565 | 11.6683 | 11.7966 | 5.71209 | 11.1988 | 10.4366 | 90.8394 |
| source-count-pfa-medium-cached | double | 18.2162 | 19.5448 | 19.8629 | 22.1422 | 5.91183 | 19.1078 | 17.8957 | 53.737 |
| source-count-pfa-large-cached | float | 6.58905 | 7.32834 | 7.89649 | 8.35132 | 6.38079 | 6.87075 | 7.01061 | 151.445 |
| source-count-pfa-large-cached | double | 33.412 | 34.6259 | 35.2009 | 35.5263 | 7.19028 | 34.0186 | 33.6068 | 29.6374 |
| max-sources-small-cached | float | 4.07284 | 4.89671 | 5.50524 | 5.70505 | 5.59132 | 4.48891 | 4.56246 | 233.023 |
| max-sources-small-cached | double | 3.66414 | 4.65444 | 5.38212 | 5.70398 | 5.74929 | 3.89689 | 3.62705 | 249.636 |
| max-sources-medium-cached | float | 10.726 | 11.4151 | 11.6055 | 11.6776 | 5.55932 | 4.86979 | 4.89697 | 94.1883 |
| max-sources-medium-cached | double | 17.8986 | 18.8582 | 19.6175 | 19.7151 | 5.68587 | 8.67902 | 8.51576 | 55.6487 |
| max-sources-large-cached | float | 6.68731 | 7.7026 | 8.23487 | 10.123 | 6.353 | 6.45386 | 6.10203 | 141.609 |
| max-sources-large-cached | double | 34.3942 | 35.584 | 35.8271 | 36.3522 | 7.11968 | 16.2689 | 16.2995 | 28.8758 |
| transition-medium-cached | float | 4.44797 | 5.00386 | 5.54194 | 5.63957 | 5.57361 | 4.059 | 4.11844 | 221.141 |
| transition-medium-cached | double | 7.21763 | 7.49595 | 7.86078 | 8.40965 | 5.69355 | 7.02784 | 7.48433 | 140.991 |
| optional-minimal-small | float | 4.28006 | 5.08626 | 5.3437 | 5.89634 | 4.59622 | 4.25818 | — | 221.073 |
| optional-minimal-small | double | 4.32022 | 5.093 | 5.62192 | 5.94314 | 4.63372 | 4.31308 | — | 222.563 |
| optional-minimal-medium | float | 10.6485 | 11.1264 | 11.4168 | 11.8108 | 4.44494 | 10.6611 | — | 94.6073 |
| optional-minimal-medium | double | 17.4383 | 18.2787 | 18.9868 | 19.4241 | 4.66449 | 17.2286 | — | 56.5132 |
| optional-minimal-large | float | 6.36493 | 6.9856 | 8.10721 | 8.64587 | 5.28043 | 6.34842 | — | 152.202 |
| optional-minimal-large | double | 34.2113 | 35.5992 | 35.8024 | 36.0863 | 6.27028 | 33.9639 | — | 28.9128 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/esprit_estimator/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/esprit_estimator/repeat-review.json).

## music_estimator

Editable scope: `operators/angle_estimation/music_estimator_op`. Output ports (both precisions exact): `angle_spectrum`, `aoa_estimates`, `doa_multi`, `eigenvalues`, `source_count`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/music_estimator.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 13; insufficient pair budget False; worst P90 paired delta 21.36%, worst baseline IQR 18.76%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/music_estimator/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 3.32917 | 4.06271 | 4.47136 | 4.57728 | 6.04504 | 3.25669 | — | 289.949 |
| baseline-small | double | 4.05071 | 4.5155 | 5.32366 | 5.64031 | 5.97737 | 4.39198 | — | 244.848 |
| baseline-medium | float | 4.45038 | 5.26307 | 5.85344 | 6.32703 | 6.50103 | 4.10015 | — | 210.865 |
| baseline-medium | double | 8.01337 | 8.79649 | 9.32314 | 9.49585 | 6.43996 | 7.31719 | — | 122.38 |
| baseline-large | float | 11.7703 | 11.94 | 12.0149 | 12.5083 | 8.11016 | 11.2551 | — | 85.9531 |
| baseline-large | double | 30.8746 | 31.6393 | 31.7108 | 32.1932 | 9.16165 | 30.6494 | — | 32.3184 |
| parameter-medium | float | 4.09204 | 4.67306 | 5.08832 | 5.41955 | 6.09617 | 4.65161 | — | 240.989 |
| parameter-medium | double | 7.87778 | 8.12067 | 8.53361 | 8.79787 | 6.15256 | 7.34177 | — | 128.314 |
| parameter-large | float | 11.7453 | 11.9014 | 12.1637 | 13.3691 | 7.29858 | 11.3739 | — | 86.4653 |
| parameter-large | double | 30.7528 | 31.4821 | 31.5583 | 32.0087 | 8.60321 | 30.7875 | — | 32.5174 |
| array-geometry-small | float | 2.96575 | 3.73151 | 3.85575 | 4.72795 | 6.39425 | 3.34283 | 3.44337 | 307.483 |
| array-geometry-small | double | 4.23174 | 5.21873 | 5.78141 | 5.99835 | 6.19724 | 4.21317 | 4.07845 | 219.977 |
| array-geometry-medium | float | 3.75571 | 4.62396 | 5.35393 | 5.86476 | 6.41212 | 4.20394 | 4.53861 | 248.14 |
| array-geometry-medium | double | 7.98146 | 8.21231 | 8.97994 | 9.25871 | 6.77256 | 7.85871 | 8.17087 | 125.019 |
| array-geometry-large | float | 11.7246 | 11.8806 | 12.4827 | 13.1031 | 7.55231 | 12.2397 | 11.4018 | 86.6184 |
| array-geometry-large | double | 31.42 | 31.9968 | 32.0918 | 32.837 | 8.75558 | 30.8373 | 31.9272 | 31.7974 |
| az-coordinates-small | float | 2.69494 | 3.5117 | 3.55538 | 3.80123 | 6.44699 | 3.84723 | 3.09807 | 337.175 |
| az-coordinates-small | double | 3.97922 | 4.58987 | 5.17799 | 5.56316 | 6.32417 | 5.02979 | 5.06881 | 243 |
| az-coordinates-medium | float | 4.10928 | 5.11207 | 5.36525 | 5.74162 | 6.26635 | 4.07951 | 4.0548 | 233.893 |
| az-coordinates-medium | double | 7.95941 | 8.59622 | 8.79912 | 9.40846 | 6.76804 | 7.73305 | 8.37374 | 124.088 |
| az-coordinates-large | float | 11.9405 | 12.369 | 12.479 | 12.6016 | 7.67859 | 11.5238 | 12.7433 | 83.2841 |
| az-coordinates-large | double | 31.1508 | 31.5717 | 31.9761 | 32.3181 | 9.69707 | 30.8822 | 31.7761 | 32.2027 |
| el-coordinates-small | float | 3.33999 | 4.20371 | 4.46434 | 4.73628 | 6.08976 | 4.16049 | 3.60032 | 275.728 |
| el-coordinates-small | double | 4.54648 | 5.47116 | 5.71043 | 6.19219 | 6.35223 | 5.32285 | 4.50242 | 206.6 |
| el-coordinates-medium | float | 4.25547 | 5.09669 | 5.79433 | 5.90187 | 6.53339 | 4.14615 | 4.24615 | 232.399 |
| el-coordinates-medium | double | 7.8415 | 7.97383 | 8.38206 | 8.84333 | 6.76302 | 8.03288 | 8.06535 | 130.056 |
| el-coordinates-large | float | 12.0024 | 12.5566 | 13.2061 | 13.3341 | 8.09501 | 12.3321 | 12.6304 | 82.3731 |
| el-coordinates-large | double | 31.6891 | 31.9995 | 32.0578 | 32.4264 | 8.88251 | 31.2619 | 31.4934 | 31.6743 |
| spatial-smoothing-small | float | 3.45576 | 4.39187 | 4.5452 | 4.75696 | 6.22056 | 3.67747 | 3.8271 | 274.216 |
| spatial-smoothing-small | double | 4.57079 | 5.40206 | 5.98094 | 6.06891 | 6.20333 | 5.01651 | 4.42859 | 218.444 |
| spatial-smoothing-medium | float | 4.32502 | 5.1918 | 5.72525 | 5.89465 | 6.69321 | 4.17451 | 4.9014 | 228.556 |
| spatial-smoothing-medium | double | 7.92928 | 8.14752 | 8.27162 | 8.57348 | 6.60953 | 8.01627 | 7.92339 | 127.251 |
| spatial-smoothing-large | float | 11.76 | 11.9371 | 12.0437 | 13.1134 | 8.5132 | 11.3493 | 7.6392 | 86.2283 |
| spatial-smoothing-large | double | 30.856 | 31.5668 | 31.6716 | 32.1801 | 9.4409 | 30.7731 | 29.2741 | 32.3469 |
| num-sources-small | float | 3.58514 | 4.38965 | 4.54144 | 4.60242 | 6.24566 | 4.61751 | 2.95025 | 275.987 |
| num-sources-small | double | 4.10122 | 4.86845 | 5.43004 | 5.75227 | 6.13226 | 4.34026 | 4.14659 | 237.879 |
| num-sources-medium | float | 4.17202 | 4.89547 | 5.56681 | 5.9099 | 6.76334 | 4.99731 | 3.77979 | 236.086 |
| num-sources-medium | double | 7.76225 | 8.48084 | 8.52804 | 8.89327 | 6.55751 | 7.9152 | 7.95906 | 126.186 |
| num-sources-large | float | 11.6602 | 11.9373 | 12.3145 | 12.5172 | 8.05014 | 12.2628 | 11.8781 | 86.8802 |
| num-sources-large | double | 30.8299 | 31.5676 | 31.9628 | 32.4418 | 8.8996 | 30.753 | 31.5911 | 32.3658 |
| source-count-pfa-small | float | 2.68381 | 3.50303 | 3.65221 | 4.17128 | 5.99294 | 2.98506 | 3.01943 | 336.862 |
| source-count-pfa-small | double | 4.10025 | 4.8822 | 5.48174 | 6.03294 | 6.13119 | 4.27191 | 3.88681 | 231.386 |
| source-count-pfa-medium | float | 4.24051 | 5.06084 | 5.76773 | 6.23011 | 6.23751 | 4.54922 | 5.00074 | 219.823 |
| source-count-pfa-medium | double | 7.88309 | 8.48161 | 8.91692 | 9.51125 | 6.57402 | 8.33337 | 7.76049 | 124.87 |
| source-count-pfa-large | float | 11.6703 | 12.3081 | 12.4947 | 12.851 | 8.1918 | 11.3721 | 10.8989 | 86.4488 |
| source-count-pfa-large | double | 30.7835 | 31.4988 | 31.6367 | 31.684 | 8.88785 | 29.825 | 30.6524 | 32.4605 |
| max-sources-small | float | 3.1271 | 3.94596 | 3.99996 | 4.52295 | 6.17131 | 3.89576 | 3.15875 | 291.781 |
| max-sources-small | double | 4.23372 | 5.27064 | 6.05796 | 6.27347 | 6.15738 | 4.44422 | 5.19851 | 214.566 |
| max-sources-medium | float | 3.82291 | 4.56505 | 5.23209 | 5.46937 | 6.23729 | 4.07978 | 4.46201 | 246.066 |
| max-sources-medium | double | 7.97983 | 8.04274 | 8.38573 | 9.53373 | 6.49584 | 7.98716 | 7.87608 | 124.939 |
| max-sources-large | float | 12.1345 | 12.5437 | 12.8097 | 13.3024 | 7.41974 | 11.7636 | 12.3237 | 82.8487 |
| max-sources-large | double | 31.5126 | 32.0033 | 32.3383 | 32.5272 | 9.33685 | 30.9265 | 31.14 | 31.6913 |
| transition-medium | float | 4.57365 | 5.33808 | 5.57199 | 6.2692 | 6.60816 | 4.89048 | 5.05315 | 215.033 |
| transition-medium | double | 7.8449 | 8.10082 | 8.50309 | 8.80251 | 6.83655 | 7.76808 | 7.95691 | 130.271 |
| array-geometry-small-cached | float | 3.09788 | 3.64629 | 4.27196 | 4.50474 | 6.15502 | 3.3911 | 3.4921 | 319.832 |
| array-geometry-small-cached | double | 3.65193 | 4.63502 | 5.25716 | 5.50782 | 6.22637 | 5.14766 | 4.3924 | 249.462 |
| array-geometry-medium-cached | float | 4.48039 | 5.35969 | 5.91815 | 6.30105 | 6.51634 | 4.46013 | 4.26213 | 210.173 |
| array-geometry-medium-cached | double | 7.91462 | 8.5031 | 8.90555 | 9.35951 | 7.58957 | 7.90339 | 7.98269 | 125.354 |
| array-geometry-large-cached | float | 11.9593 | 12.4174 | 13.1225 | 14.148 | 7.66674 | 11.5033 | 11.7844 | 83.3435 |
| array-geometry-large-cached | double | 31.4781 | 32.0172 | 32.2268 | 32.3889 | 9.44793 | 31.6774 | 31.4568 | 31.7718 |
| az-coordinates-small-cached | float | 3.32939 | 4.09992 | 4.2054 | 4.56383 | 6.37371 | 3.59009 | 3.32695 | 285.135 |
| az-coordinates-small-cached | double | 4.31524 | 5.34713 | 6.06016 | 6.19942 | 6.26674 | 4.70719 | 4.26815 | 211.333 |
| az-coordinates-medium-cached | float | 4.22325 | 5.06551 | 5.98207 | 6.70254 | 6.50608 | 4.98958 | 3.82253 | 235.615 |
| az-coordinates-medium-cached | double | 7.84927 | 8.615 | 8.9384 | 9.29827 | 6.6193 | 8.18727 | 8.34365 | 122.469 |
| az-coordinates-large-cached | float | 11.7403 | 11.9561 | 12.0325 | 12.0749 | 8.11379 | 11.2676 | 11.9212 | 86.3453 |
| az-coordinates-large-cached | double | 30.8201 | 31.5314 | 31.7456 | 32.1345 | 9.32269 | 30.5478 | 30.7125 | 32.3872 |
| el-coordinates-small-cached | float | 2.66773 | 3.48464 | 3.79376 | 4.42709 | 6.3426 | 3.79611 | 2.69544 | 338.361 |
| el-coordinates-small-cached | double | 4.18382 | 5.02699 | 5.36192 | 5.55134 | 6.22055 | 4.1949 | 3.85385 | 246.628 |
| el-coordinates-medium-cached | float | 4.31063 | 5.16699 | 5.49646 | 5.99374 | 6.6679 | 4.41886 | 3.78682 | 227.768 |
| el-coordinates-medium-cached | double | 7.98438 | 8.61475 | 9.00782 | 9.43513 | 6.63348 | 8.39966 | 7.89126 | 123.8 |
| el-coordinates-large-cached | float | 12.0163 | 12.3514 | 12.3749 | 12.4857 | 7.67971 | 12.1071 | 11.7026 | 83.4728 |
| el-coordinates-large-cached | double | 31.7021 | 31.983 | 32.0512 | 32.0866 | 9.22501 | 30.8005 | 31.4608 | 31.5961 |
| spatial-smoothing-small-cached | float | 3.09059 | 3.92976 | 4.39389 | 4.66137 | 6.31231 | 4.54836 | 3.42458 | 301.637 |
| spatial-smoothing-small-cached | double | 3.76518 | 4.71715 | 5.18845 | 5.38831 | 6.33647 | 4.95277 | 3.80732 | 244.658 |
| spatial-smoothing-medium-cached | float | 3.90256 | 4.62494 | 5.40453 | 5.97721 | 6.51996 | 5.39328 | 3.61365 | 243.003 |
| spatial-smoothing-medium-cached | double | 7.9661 | 8.03986 | 8.38406 | 8.6036 | 6.65645 | 7.90622 | 7.59258 | 126.333 |
| spatial-smoothing-large-cached | float | 11.8415 | 12.0582 | 12.9834 | 13.6465 | 8.19948 | 8.15409 | 7.63674 | 85.0132 |
| spatial-smoothing-large-cached | double | 31.2741 | 31.518 | 32.0582 | 33.1065 | 8.57251 | 28.3642 | 28.5683 | 32.0222 |
| num-sources-small-cached | float | 2.89343 | 3.61451 | 3.76861 | 4.55527 | 6.18573 | 3.37868 | 2.72119 | 317.145 |
| num-sources-small-cached | double | 4.1287 | 4.88842 | 5.32197 | 6.0734 | 6.16702 | 4.93532 | 4.13707 | 229.397 |
| num-sources-medium-cached | float | 4.3591 | 5.14508 | 5.26043 | 5.35953 | 6.45238 | 4.42318 | 4.10009 | 222.007 |
| num-sources-medium-cached | double | 7.82915 | 8.45561 | 8.53859 | 8.57306 | 6.58544 | 8.25312 | 7.6037 | 124.949 |
| num-sources-large-cached | float | 11.9286 | 12.4594 | 12.9521 | 13.2568 | 8.28496 | 11.6964 | 11.7645 | 83.0279 |
| num-sources-large-cached | double | 30.8481 | 31.493 | 32.0418 | 34.6311 | 9.32197 | 30.4945 | 30.814 | 32.3794 |
| source-count-pfa-small-cached | float | 3.3801 | 4.10724 | 4.29331 | 4.42343 | 6.13481 | 3.61186 | 3.50504 | 293.65 |
| source-count-pfa-small-cached | double | 4.03524 | 4.78597 | 5.54877 | 5.92284 | 6.17833 | 4.27039 | 3.86333 | 239.107 |
| source-count-pfa-medium-cached | float | 3.97194 | 4.53318 | 5.20615 | 5.97904 | 6.54784 | 4.99903 | 4.18284 | 245.265 |
| source-count-pfa-medium-cached | double | 7.96229 | 8.14331 | 8.48535 | 9.07029 | 7.43975 | 7.93326 | 7.85624 | 126.2 |
| source-count-pfa-large-cached | float | 11.8448 | 12.3508 | 12.8738 | 13.8278 | 7.91056 | 12.09 | 11.7568 | 84.0956 |
| source-count-pfa-large-cached | double | 31.2237 | 32.067 | 32.1806 | 33.0662 | 9.22386 | 30.4357 | 31.4266 | 32.1332 |
| max-sources-small-cached | float | 3.4673 | 4.3123 | 4.46202 | 4.55399 | 6.25945 | 3.66678 | 4.09113 | 262.61 |
| max-sources-small-cached | double | 4.12406 | 4.89219 | 5.48211 | 6.35076 | 6.09504 | 5.05355 | 4.45613 | 237.42 |
| max-sources-medium-cached | float | 4.32623 | 5.12121 | 5.73624 | 6.27506 | 6.45949 | 4.29386 | 3.81557 | 225.399 |
| max-sources-medium-cached | double | 7.90361 | 8.15345 | 8.25122 | 8.79106 | 6.58258 | 7.25337 | 8.04176 | 125.573 |
| max-sources-large-cached | float | 11.7889 | 12.275 | 12.321 | 12.3919 | 7.86786 | 12.2534 | 11.8756 | 85.8295 |
| max-sources-large-cached | double | 30.9541 | 31.6158 | 31.706 | 32.1759 | 9.04929 | 30.9819 | 30.5407 | 32.2328 |
| transition-medium-cached | float | 4.02092 | 4.68235 | 4.98024 | 5.32493 | 6.31702 | 4.99938 | 4.25471 | 240.009 |
| transition-medium-cached | double | 7.98564 | 8.45637 | 9.11615 | 9.32697 | 6.60962 | 8.41428 | 8.32191 | 122.884 |
| optional-minimal-small | float | 3.27706 | 4.08402 | 4.33324 | 4.60221 | 4.56546 | 3.56114 | — | 304.256 |
| optional-minimal-small | double | 3.61766 | 4.67273 | 5.0531 | 5.432 | 4.51982 | 4.89541 | — | 250.895 |
| optional-minimal-medium | float | 4.19489 | 4.94003 | 5.51543 | 5.86786 | 4.83367 | 4.49402 | — | 229.493 |
| optional-minimal-medium | double | 7.83083 | 8.024 | 8.15364 | 8.41921 | 4.80715 | 7.2261 | — | 130.556 |
| optional-minimal-large | float | 11.61 | 12.1188 | 12.3905 | 12.6186 | 5.73627 | 11.3762 | — | 86.7545 |
| optional-minimal-large | double | 31.0272 | 31.6088 | 31.7327 | 32.0104 | 6.75129 | 29.979 | — | 32.2551 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/music_estimator/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/music_estimator/repeat-review.json).

## mvdr_estimator

Editable scope: `operators/angle_estimation/mvdr_estimator_op`. Output ports (both precisions exact): `angle_spectrum`, `aoa_estimates`, `doa_multi`, `eigenvalues`, `source_count`.

Application consumer inventory: `feature_assembly`, `multi_source_expansion`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/mvdr_estimator.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 3; insufficient pair budget False; worst P90 paired delta 22.23%, worst baseline IQR 11.74%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/mvdr_estimator/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.202969 | 0.228165 | 0.259963 | 0.282213 | 5.97415 | 4.73286 | — | 4610.65 |
| baseline-small | double | 0.507789 | 0.525947 | 0.564036 | 0.793107 | 6.03609 | 6.18419 | — | 1914.81 |
| baseline-medium | float | 0.513521 | 0.540977 | 0.576479 | 0.886339 | 6.19331 | 5.40121 | — | 1864.04 |
| baseline-medium | double | 2.61754 | 2.62998 | 2.64523 | 2.67423 | 6.69412 | 8.47766 | — | 380.555 |
| baseline-large | float | 6.03042 | 6.06449 | 6.144 | 6.5634 | 8.28934 | 12.9379 | — | 165.207 |
| baseline-large | double | 22.1652 | 22.1814 | 22.2165 | 22.2271 | 9.55969 | 31.051 | — | 45.0878 |
| parameter-medium | float | 0.542892 | 0.559048 | 0.579423 | 0.596373 | 6.03403 | 5.89001 | — | 1801.91 |
| parameter-medium | double | 2.67107 | 2.68255 | 2.70031 | 2.70579 | 6.33528 | 9.36463 | — | 372.815 |
| parameter-large | float | 6.11434 | 6.12982 | 6.1568 | 7.11687 | 7.51946 | 12.9894 | — | 163.196 |
| parameter-large | double | 22.3526 | 22.3687 | 22.3948 | 22.4037 | 8.58971 | 31.2291 | — | 44.7103 |
| array-geometry-small | float | 0.206431 | 0.218109 | 0.256077 | 0.274853 | 5.93256 | 5.32179 | 0.424141 | 4554.27 |
| array-geometry-small | double | 0.509258 | 0.541691 | 0.556844 | 0.572892 | 6.04214 | 5.32831 | 0.692198 | 1913.82 |
| array-geometry-medium | float | 0.512721 | 0.53284 | 0.560161 | 0.578697 | 6.37223 | 5.94392 | 0.764959 | 1901.51 |
| array-geometry-medium | double | 2.62039 | 2.63561 | 2.65308 | 3.23122 | 6.54441 | 8.54324 | 2.83544 | 379.951 |
| array-geometry-large | float | 6.02623 | 6.04768 | 6.06371 | 6.16054 | 8.13556 | 12.436 | 6.35153 | 165.541 |
| array-geometry-large | double | 22.1833 | 22.209 | 22.2338 | 22.2903 | 9.11229 | 31.3816 | 22.4997 | 45.0412 |
| az-coordinates-small | float | 0.206765 | 0.229431 | 0.25925 | 0.264159 | 6.19803 | 4.91461 | 0.348807 | 4542.7 |
| az-coordinates-small | double | 0.508942 | 0.527132 | 0.550756 | 0.560225 | 6.36403 | 6.00256 | 0.632179 | 1918.58 |
| az-coordinates-medium | float | 0.513462 | 0.526561 | 0.554977 | 1.34832 | 6.1402 | 5.87863 | 0.673466 | 1899.31 |
| az-coordinates-medium | double | 2.62017 | 2.63583 | 2.64817 | 2.65195 | 6.86916 | 9.21522 | 2.79796 | 380.014 |
| az-coordinates-large | float | 6.0251 | 6.04961 | 6.07409 | 6.07624 | 7.93097 | 12.0313 | 6.31663 | 165.528 |
| az-coordinates-large | double | 22.1743 | 22.1946 | 22.2186 | 23.394 | 8.56607 | 30.9558 | 22.4594 | 45.0688 |
| el-coordinates-small | float | 0.205922 | 0.221677 | 0.265171 | 0.289308 | 6.11541 | 4.62582 | 0.35764 | 4578.22 |
| el-coordinates-small | double | 0.50929 | 0.521362 | 0.535194 | 0.565744 | 6.10942 | 5.47022 | 0.630457 | 1926.59 |
| el-coordinates-medium | float | 0.513785 | 0.532657 | 0.561137 | 0.587679 | 6.41704 | 5.64213 | 0.687411 | 1898.15 |
| el-coordinates-medium | double | 2.62153 | 2.63922 | 2.6513 | 3.27512 | 6.85843 | 9.25696 | 2.80412 | 379.75 |
| el-coordinates-large | float | 6.02337 | 6.04926 | 6.08044 | 6.09637 | 7.54952 | 12.7024 | 6.25026 | 165.566 |
| el-coordinates-large | double | 22.1702 | 22.1834 | 22.1978 | 22.2235 | 9.47954 | 30.7122 | 22.4684 | 45.0822 |
| spatial-smoothing-small | float | 0.206273 | 0.220481 | 0.255802 | 0.266223 | 6.24311 | 4.80272 | 3.60331 | 4569.44 |
| spatial-smoothing-small | double | 0.507304 | 0.526617 | 0.537365 | 0.550938 | 6.33112 | 5.89612 | 4.92748 | 1929.29 |
| spatial-smoothing-medium | float | 0.534091 | 0.546741 | 0.574302 | 1.66588 | 6.15508 | 5.91041 | 5.07897 | 1829.75 |
| spatial-smoothing-medium | double | 2.61905 | 2.63111 | 2.64582 | 2.65259 | 6.89529 | 8.89951 | 8.50179 | 380.238 |
| spatial-smoothing-large | float | 6.02237 | 6.04317 | 6.07138 | 6.90887 | 7.4065 | 11.9571 | 7.7471 | 165.591 |
| spatial-smoothing-large | double | 22.1673 | 22.1931 | 22.2103 | 22.2619 | 9.27106 | 31.0748 | 28.7491 | 45.0727 |
| diagonal-loading-small | float | 0.205571 | 0.229418 | 0.256326 | 0.264093 | 6.0202 | 4.7051 | 0.236002 | 4558.34 |
| diagonal-loading-small | double | 0.532808 | 0.550358 | 0.567467 | 0.656161 | 6.00776 | 6.12321 | 0.544197 | 1839.15 |
| diagonal-loading-medium | float | 0.511295 | 0.529605 | 0.560302 | 0.592318 | 6.86866 | 5.38634 | 0.532975 | 1911.8 |
| diagonal-loading-medium | double | 2.62161 | 2.63506 | 2.64007 | 3.47154 | 6.713 | 8.61183 | 2.64536 | 379.979 |
| diagonal-loading-large | float | 6.02428 | 6.07027 | 6.28353 | 7.09621 | 7.5771 | 12.7297 | 6.04425 | 164.64 |
| diagonal-loading-large | double | 22.1849 | 22.2121 | 22.2356 | 22.261 | 8.44774 | 31.1413 | 22.1953 | 45.0323 |
| max-sources-small | float | 0.206385 | 0.232777 | 0.245558 | 0.274667 | 6.2461 | 4.75016 | 0.251621 | 4550.09 |
| max-sources-small | double | 0.509668 | 0.520041 | 0.542869 | 0.554012 | 6.11026 | 5.72513 | 0.534252 | 1923.02 |
| max-sources-medium | float | 0.511613 | 0.533284 | 0.553321 | 0.588614 | 6.51059 | 6.11821 | 0.565763 | 1904.39 |
| max-sources-medium | double | 2.61906 | 2.63388 | 2.64574 | 2.65577 | 7.31267 | 8.72185 | 2.68 | 380.315 |
| max-sources-large | float | 6.02305 | 6.05034 | 6.07171 | 6.09394 | 7.67554 | 12.0216 | 6.11131 | 165.622 |
| max-sources-large | double | 22.1707 | 22.1868 | 22.2223 | 22.2451 | 8.59176 | 31.3417 | 22.3292 | 45.0716 |
| transition-medium | float | 0.546295 | 0.558893 | 0.587275 | 0.599401 | 6.85187 | 5.52097 | 0.77342 | 1789.72 |
| transition-medium | double | 2.67125 | 2.68377 | 2.697 | 2.70973 | 7.0615 | 9.30665 | 2.91287 | 372.741 |
| array-geometry-small-cached | float | 0.206279 | 0.215433 | 0.251678 | 0.274705 | 6.13967 | 5.52245 | 0.23851 | 4582.95 |
| array-geometry-small-cached | double | 0.508207 | 0.520826 | 0.550472 | 0.581031 | 6.28436 | 5.81715 | 0.541642 | 1924.43 |
| array-geometry-medium-cached | float | 0.512415 | 0.526866 | 0.551202 | 0.555187 | 6.42131 | 5.59177 | 0.552623 | 1904.16 |
| array-geometry-medium-cached | double | 2.6202 | 2.63244 | 2.64782 | 2.66415 | 6.46572 | 9.39667 | 2.65328 | 380.097 |
| array-geometry-large-cached | float | 6.02842 | 6.05302 | 6.08223 | 6.43044 | 7.71012 | 12.3792 | 6.06464 | 165.489 |
| array-geometry-large-cached | double | 22.1707 | 22.1854 | 22.2033 | 22.2299 | 9.03531 | 31.131 | 22.2144 | 45.0772 |
| az-coordinates-small-cached | float | 0.206066 | 0.223314 | 0.269697 | 0.276788 | 6.23609 | 5.08155 | 0.218455 | 4540.73 |
| az-coordinates-small-cached | double | 0.509252 | 0.532712 | 0.55782 | 0.591318 | 6.29644 | 5.89794 | 0.518151 | 1917.64 |
| az-coordinates-medium-cached | float | 0.514247 | 0.524158 | 0.5525 | 0.558252 | 7.11315 | 5.55658 | 0.526326 | 1899.98 |
| az-coordinates-medium-cached | double | 2.62088 | 2.63707 | 2.65455 | 2.66366 | 6.82632 | 9.47218 | 2.63379 | 379.843 |
| az-coordinates-large-cached | float | 6.02436 | 6.04644 | 6.07171 | 6.91488 | 7.79576 | 12.131 | 6.04746 | 165.531 |
| az-coordinates-large-cached | double | 22.1685 | 22.18 | 22.21 | 22.2299 | 8.44073 | 31.2081 | 22.186 | 45.0877 |
| el-coordinates-small-cached | float | 0.207672 | 0.234386 | 0.271929 | 0.288307 | 6.40041 | 5.09746 | 0.218677 | 4492.64 |
| el-coordinates-small-cached | double | 0.508461 | 0.523219 | 0.541861 | 0.548095 | 6.12127 | 6.01435 | 0.516882 | 1925.89 |
| el-coordinates-medium-cached | float | 0.512896 | 0.531488 | 0.567664 | 0.584633 | 6.11629 | 5.4491 | 0.520187 | 1898.31 |
| el-coordinates-medium-cached | double | 2.61926 | 2.63408 | 2.65135 | 3.36062 | 6.42235 | 9.08828 | 2.62125 | 380.326 |
| el-coordinates-large-cached | float | 6.02586 | 6.04941 | 6.07356 | 6.13481 | 7.24657 | 12.7584 | 6.03052 | 165.579 |
| el-coordinates-large-cached | double | 22.1729 | 22.2122 | 22.2312 | 22.9665 | 9.03969 | 30.4162 | 22.1989 | 45.0365 |
| spatial-smoothing-small-cached | float | 0.205723 | 0.220528 | 0.260132 | 0.272529 | 6.02043 | 5.56712 | 0.231251 | 4580.81 |
| spatial-smoothing-small-cached | double | 0.508243 | 0.520537 | 0.539776 | 0.557372 | 6.13517 | 6.22905 | 0.407065 | 1925.12 |
| spatial-smoothing-medium-cached | float | 0.511302 | 0.532449 | 0.551018 | 0.552799 | 6.33878 | 6.58063 | 0.406975 | 1907.11 |
| spatial-smoothing-medium-cached | double | 2.62027 | 2.63442 | 2.67246 | 3.38014 | 6.56206 | 9.20842 | 2.31128 | 379.819 |
| spatial-smoothing-large-cached | float | 6.02285 | 6.04397 | 6.07338 | 6.08765 | 7.86132 | 8.97065 | 2.41485 | 165.586 |
| spatial-smoothing-large-cached | double | 22.1681 | 22.182 | 22.228 | 22.2451 | 9.14545 | 29.8572 | 21.2844 | 45.0779 |
| diagonal-loading-small-cached | float | 0.204778 | 0.214554 | 0.245557 | 0.25826 | 5.96345 | 5.05453 | 0.214798 | 4602.93 |
| diagonal-loading-small-cached | double | 0.507822 | 0.524159 | 0.544161 | 0.556789 | 6.14718 | 6.25879 | 0.517965 | 1928.6 |
| diagonal-loading-medium-cached | float | 0.51178 | 0.529619 | 0.544326 | 0.577503 | 6.38723 | 6.20348 | 0.522419 | 1908.34 |
| diagonal-loading-medium-cached | double | 2.61624 | 2.62819 | 2.63663 | 2.67022 | 6.61442 | 9.16934 | 2.61912 | 380.699 |
| diagonal-loading-large-cached | float | 6.02911 | 6.06825 | 6.08458 | 6.60572 | 7.52097 | 12.8284 | 6.03679 | 165.411 |
| diagonal-loading-large-cached | double | 22.1669 | 22.1818 | 22.2164 | 22.2512 | 8.74826 | 31.4471 | 22.1523 | 45.0864 |
| max-sources-small-cached | float | 0.206589 | 0.212282 | 0.249823 | 0.295751 | 6.07759 | 5.45451 | 0.223306 | 4579.97 |
| max-sources-small-cached | double | 0.508789 | 0.523062 | 0.547737 | 0.55342 | 6.05279 | 6.24587 | 0.523909 | 1923.85 |
| max-sources-medium-cached | float | 0.512363 | 0.533556 | 0.55347 | 0.791753 | 6.15437 | 5.583 | 0.543559 | 1900.07 |
| max-sources-medium-cached | double | 2.6192 | 2.63026 | 2.64512 | 2.6592 | 6.44934 | 8.45754 | 2.65879 | 380.304 |
| max-sources-large-cached | float | 6.02389 | 6.04531 | 6.06532 | 6.07053 | 7.57113 | 12.3121 | 6.09058 | 165.61 |
| max-sources-large-cached | double | 22.1682 | 22.1853 | 22.207 | 22.2645 | 9.03787 | 31.0869 | 22.3043 | 45.0763 |
| transition-medium-cached | float | 0.546225 | 0.562854 | 0.592005 | 0.601651 | 6.2291 | 6.21953 | 0.58865 | 1786.52 |
| transition-medium-cached | double | 2.67147 | 2.68424 | 2.69661 | 2.7079 | 6.38945 | 9.42509 | 2.70699 | 372.84 |
| optional-minimal-small | float | 0.09625 | 0.0973739 | 0.101606 | 0.448234 | 4.7547 | 5.07014 | — | 10073.4 |
| optional-minimal-small | double | 0.199686 | 0.205182 | 0.222188 | 0.226492 | 4.7403 | 5.8832 | — | 4898.17 |
| optional-minimal-medium | float | 0.303862 | 0.308579 | 0.318809 | 0.329233 | 4.60283 | 5.64451 | — | 3268.9 |
| optional-minimal-medium | double | 1.42591 | 1.43025 | 1.4391 | 1.4447 | 4.85219 | 8.98016 | — | 699.22 |
| optional-minimal-large | float | 5.46709 | 5.479 | 5.49786 | 5.52424 | 5.42801 | 12.2395 | — | 182.679 |
| optional-minimal-large | double | 18.3603 | 18.3689 | 18.4122 | 18.4544 | 6.36104 | 30.9849 | — | 54.4479 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/mvdr_estimator/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/mvdr_estimator/repeat-review.json).

## phase_shift_beamformer

Editable scope: `operators/angle_estimation/phase_shift_beamformer_op`. Output ports (both precisions exact): `az_grid`, `el_grid`, `rda_cube`.

Application consumer inventory: `cfar_detector`, `feature_assembly`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/phase_shift_beamformer.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 5; insufficient pair budget False; worst P90 paired delta 18.18%, worst baseline IQR 14.28%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/phase_shift_beamformer/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.055676 | 0.0573333 | 0.0591302 | 0.061277 | 3.82802 | 2.19781 | — | 17100.3 |
| baseline-small | double | 0.149385 | 0.151871 | 0.163045 | 0.17112 | 3.79055 | 1.58895 | — | 6518.88 |
| baseline-medium | float | 0.0586205 | 0.0607273 | 0.0746482 | 0.083602 | 4.01923 | 1.39269 | — | 15994.8 |
| baseline-medium | double | 0.149968 | 0.152084 | 0.154805 | 0.193066 | 4.32296 | 1.89185 | — | 6523.72 |
| baseline-large | float | 0.152404 | 0.154515 | 0.16533 | 0.183334 | 6.19327 | 1.69061 | — | 6417.79 |
| baseline-large | double | 2.42755 | 2.42959 | 2.43571 | 2.43774 | 8.55233 | 4.07543 | — | 411.33 |
| tapered-medium | float | 0.0585655 | 0.0621787 | 0.0731759 | 0.089213 | 4.00679 | 1.53869 | — | 15876 |
| tapered-medium | double | 0.150644 | 0.151797 | 0.157339 | 0.162084 | 4.21086 | 2.0023 | — | 6510.78 |
| geometry-small | float | 0.05737 | 0.0594472 | 0.0741679 | 0.084538 | 4.11762 | 1.32443 | 0.969078 | 16407.3 |
| geometry-small | double | 0.151482 | 0.153994 | 0.166513 | 0.276038 | 4.11315 | 1.40225 | 0.993023 | 6434.95 |
| geometry-medium | float | 0.061546 | 0.0648172 | 0.0737083 | 0.10137 | 4.2261 | 1.83882 | 0.891272 | 15371.4 |
| geometry-medium | double | 0.152038 | 0.155009 | 0.162046 | 0.164455 | 4.45718 | 1.5032 | 0.986374 | 6429.98 |
| geometry-large | float | 0.154824 | 0.157887 | 0.176172 | 0.209408 | 6.15498 | 2.40265 | 1.13264 | 6283.71 |
| geometry-large | double | 2.43048 | 2.43622 | 2.45639 | 2.68173 | 8.07888 | 4.01961 | 3.41199 | 410.718 |
| az-small | float | 0.057435 | 0.0588953 | 0.0667097 | 0.075352 | 4.13276 | 1.52642 | 0.920577 | 16425.7 |
| az-small | double | 0.152154 | 0.1558 | 0.164267 | 0.171214 | 4.12528 | 1.72104 | 0.992921 | 6439.85 |
| az-medium | float | 0.060158 | 0.0623081 | 0.0653226 | 0.093084 | 4.17495 | 1.37096 | 0.892346 | 15681 |
| az-medium | double | 0.151747 | 0.154244 | 0.162662 | 0.164945 | 4.6798 | 1.47117 | 0.981476 | 6433.61 |
| az-large | float | 0.154413 | 0.156345 | 0.160548 | 0.178806 | 6.74631 | 2.21461 | 1.07093 | 6332.12 |
| az-large | double | 2.43027 | 2.4337 | 2.44637 | 2.45766 | 8.39014 | 4.29444 | 3.35459 | 410.76 |
| el-small | float | 0.0578945 | 0.0636881 | 0.0783996 | 0.108779 | 4.20214 | 1.43693 | 0.887625 | 16003.6 |
| el-small | double | 0.151612 | 0.157236 | 0.166956 | 0.173649 | 4.1247 | 1.62287 | 0.97079 | 6423.69 |
| el-medium | float | 0.0603985 | 0.0619952 | 0.0670569 | 0.07501 | 4.38851 | 1.52401 | 0.935689 | 15656.2 |
| el-medium | double | 0.152714 | 0.154442 | 0.158343 | 0.160075 | 4.48428 | 1.50889 | 0.96931 | 6411.44 |
| el-large | float | 0.155052 | 0.157243 | 0.163673 | 0.175203 | 6.15919 | 1.57607 | 1.03373 | 6307.41 |
| el-large | double | 2.42985 | 2.43524 | 2.44069 | 2.44344 | 7.64379 | 3.86785 | 3.59491 | 410.869 |
| taper-small | float | 0.057703 | 0.0655953 | 0.0776769 | 0.085574 | 4.28043 | 1.55391 | 1.0139 | 16066.1 |
| taper-small | double | 0.151774 | 0.156487 | 0.169255 | 0.176871 | 4.0557 | 1.38804 | 1.03774 | 6414.32 |
| taper-medium | float | 0.0599455 | 0.0617511 | 0.0672379 | 0.085 | 4.3587 | 1.53821 | 0.969819 | 15592 |
| taper-medium | double | 0.151459 | 0.15397 | 0.161108 | 0.171047 | 4.34949 | 1.48633 | 1.09727 | 6460.48 |
| taper-large | float | 0.154079 | 0.157735 | 0.164665 | 0.182306 | 6.15591 | 1.59187 | 1.15741 | 6310.76 |
| taper-large | double | 2.43025 | 2.43216 | 2.45 | 2.45895 | 7.79955 | 4.05915 | 3.40956 | 410.878 |
| tapered-geometry-medium | float | 0.06076 | 0.0635139 | 0.0658963 | 0.089269 | 4.50132 | 1.44182 | 0.996893 | 15614.5 |
| tapered-geometry-medium | double | 0.152635 | 0.154513 | 0.162953 | 0.169196 | 4.49457 | 1.51701 | 1.05143 | 6381.05 |
| geometry-small-cached | float | 0.057968 | 0.0652483 | 0.0768669 | 0.083305 | 4.1926 | 1.61582 | 0.098047 | 16055.6 |
| geometry-small-cached | double | 0.151648 | 0.154582 | 0.170242 | 0.17377 | 4.0503 | 1.47651 | 0.191777 | 6420.78 |
| geometry-medium-cached | float | 0.0611665 | 0.0624443 | 0.071239 | 0.076778 | 4.50318 | 1.45714 | 0.100806 | 15507.8 |
| geometry-medium-cached | double | 0.152362 | 0.154161 | 0.159937 | 0.17126 | 4.48949 | 1.64404 | 0.19264 | 6426.37 |
| geometry-large-cached | float | 0.154205 | 0.157704 | 0.167723 | 0.17239 | 6.29713 | 1.73586 | 0.196482 | 6325.64 |
| geometry-large-cached | double | 2.42973 | 2.43265 | 2.44101 | 2.44466 | 8.53422 | 4.10891 | 2.47092 | 410.917 |
| az-small-cached | float | 0.057273 | 0.0594339 | 0.0679341 | 0.074769 | 4.29891 | 1.57007 | 0.070834 | 16418.5 |
| az-small-cached | double | 0.152728 | 0.154375 | 0.165724 | 0.191843 | 4.1673 | 1.65315 | 0.165454 | 6421.51 |
| az-medium-cached | float | 0.0602785 | 0.0650185 | 0.0831784 | 0.087453 | 4.33299 | 1.38409 | 0.073416 | 15532 |
| az-medium-cached | double | 0.151288 | 0.152215 | 0.157435 | 0.161093 | 5.06516 | 1.5074 | 0.162102 | 6473.89 |
| az-large-cached | float | 0.154542 | 0.15844 | 0.170322 | 0.197593 | 6.20384 | 1.62182 | 0.170094 | 6308.09 |
| az-large-cached | double | 2.43841 | 2.44491 | 2.46949 | 2.89431 | 8.47299 | 4.07112 | 2.44519 | 409.444 |
| el-small-cached | float | 0.0569255 | 0.0635267 | 0.0820876 | 0.121121 | 4.07512 | 1.44739 | 0.070231 | 16175.5 |
| el-small-cached | double | 0.152195 | 0.160666 | 0.167124 | 0.177176 | 4.1443 | 1.40366 | 0.16262 | 6372.96 |
| el-medium-cached | float | 0.0597965 | 0.0663156 | 0.0733948 | 0.077426 | 4.24508 | 2.15801 | 0.072888 | 15635.2 |
| el-medium-cached | double | 0.152649 | 0.154051 | 0.164787 | 0.557697 | 4.44066 | 2.24492 | 0.164741 | 6426.56 |
| el-large-cached | float | 0.15496 | 0.157301 | 0.166887 | 0.226769 | 6.10274 | 1.59814 | 0.173798 | 6305.61 |
| el-large-cached | double | 2.43 | 2.43329 | 2.44336 | 2.45111 | 7.9284 | 4.10624 | 2.44467 | 410.782 |
| taper-small-cached | float | 0.057352 | 0.0594592 | 0.0717458 | 0.080713 | 4.27231 | 1.61895 | 0.096833 | 16374.7 |
| taper-small-cached | double | 0.152158 | 0.153976 | 0.161857 | 0.172751 | 4.16725 | 2.23396 | 0.191638 | 6441.19 |
| taper-medium-cached | float | 0.0604725 | 0.0621066 | 0.0671387 | 0.072861 | 4.89445 | 1.47255 | 0.10001 | 15686.7 |
| taper-medium-cached | double | 0.152497 | 0.15683 | 0.166399 | 0.174575 | 4.52703 | 1.56919 | 0.189491 | 6394.3 |
| taper-large-cached | float | 0.154788 | 0.158635 | 0.169132 | 0.170835 | 6.9969 | 1.67485 | 0.199926 | 6298.1 |
| taper-large-cached | double | 2.43049 | 2.4353 | 2.44389 | 2.46156 | 7.68318 | 4.05788 | 2.47858 | 410.79 |
| tapered-geometry-medium-cached | float | 0.0603425 | 0.0622825 | 0.0698792 | 0.075269 | 4.48552 | 1.46868 | 0.101621 | 15674 |
| tapered-geometry-medium-cached | double | 0.153437 | 0.155583 | 0.165376 | 0.170612 | 4.60408 | 2.07839 | 0.18938 | 6386.39 |
| optional-all-small | float | 0.050454 | 0.0550169 | 0.0878959 | 0.09875 | 4.56266 | 1.33875 | — | 17755.9 |
| optional-all-small | double | 0.155625 | 0.164721 | 0.166294 | 0.253251 | 4.57094 | 2.20974 | — | 6138.36 |
| optional-all-medium | float | 0.055301 | 0.0561998 | 0.0649497 | 0.087945 | 4.53563 | 1.38407 | — | 17017.7 |
| optional-all-medium | double | 0.156417 | 0.160395 | 0.165662 | 0.168565 | 4.92593 | 1.55716 | — | 6260.99 |
| optional-all-large | float | 0.158145 | 0.164708 | 0.177728 | 0.202483 | 6.55837 | 1.59282 | — | 6137.19 |
| optional-all-large | double | 2.43554 | 2.43874 | 2.44662 | 2.45648 | 8.7119 | 4.03279 | — | 409.94 |
| optional-minimal-small | float | 0.0553795 | 0.0600891 | 0.073291 | 0.108029 | 3.76931 | 1.32855 | — | 16763.7 |
| optional-minimal-small | double | 0.149135 | 0.151987 | 0.158921 | 0.161815 | 3.88332 | 1.63729 | — | 6550.64 |
| optional-minimal-medium | float | 0.0584775 | 0.0596439 | 0.0620151 | 0.068241 | 4.06898 | 1.41925 | — | 16383 |
| optional-minimal-medium | double | 0.149367 | 0.150858 | 0.15833 | 0.164094 | 4.20309 | 1.47067 | — | 6541.63 |
| optional-minimal-large | float | 0.151945 | 0.155233 | 0.174709 | 0.185612 | 5.73156 | 1.69398 | — | 6390.34 |
| optional-minimal-large | double | 2.42712 | 2.42966 | 2.4455 | 2.45378 | 8.3756 | 4.04646 | — | 411.372 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/phase_shift_beamformer/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/phase_shift_beamformer/repeat-review.json).

## rank_one_residual_screen

Editable scope: `operators/angle_estimation/rank_one_residual_screen_op`. Output ports (both precisions exact): `is_valid_out`, `residual_score`, `selected_idx`, `selection_mask`, `snapshot_out`.

Application consumer inventory: `feature_assembly`, `multi_source_expansion`, `mvdr_estimator`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/rank_one_residual_screen.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 1; insufficient pair budget False; worst P90 paired delta 13.28%, worst baseline IQR 5.352%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/rank_one_residual_screen/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.158107 | 0.159537 | 0.171298 | 0.188092 | 4.88265 | 0.416464 | — | 6173.54 |
| baseline-small | double | 0.15545 | 0.158057 | 0.169949 | 0.177279 | 4.94318 | 0.417121 | — | 6360.74 |
| baseline-medium | float | 0.932639 | 0.935318 | 0.946524 | 0.961938 | 6.40799 | 1.20303 | — | 1068.36 |
| baseline-medium | double | 0.86035 | 0.87101 | 0.882789 | 0.895633 | 6.59744 | 1.12672 | — | 1155.61 |
| baseline-large | float | 3.31905 | 3.3337 | 3.34627 | 3.36067 | 9.8134 | 3.50179 | — | 300.739 |
| baseline-large | double | 3.36446 | 3.40246 | 3.42567 | 3.43703 | 12.5023 | 3.65215 | — | 296.042 |
| no-selection-medium | float | 0.929781 | 0.932549 | 0.947995 | 0.96016 | 6.4452 | 1.19598 | — | 1071.27 |
| no-selection-medium | double | 0.858572 | 0.868238 | 0.893217 | 0.900938 | 7.61007 | 1.13794 | — | 1157.84 |
| most-selected-medium | float | 0.932642 | 0.936245 | 0.956995 | 0.982106 | 6.26502 | 1.21004 | — | 1067.96 |
| most-selected-medium | double | 0.86097 | 0.875094 | 0.910138 | 0.930838 | 6.61345 | 1.12872 | — | 1151.77 |
| geometry-small | float | 0.158153 | 0.159959 | 0.169055 | 0.175566 | 5.34096 | 0.428983 | 0.250955 | 6181.29 |
| geometry-small | double | 0.157158 | 0.160012 | 0.173441 | 0.186909 | 5.41365 | 0.433186 | 0.232769 | 6265.3 |
| geometry-medium | float | 0.932934 | 0.934862 | 0.947643 | 0.970104 | 6.52888 | 1.2023 | 1.04073 | 1068.12 |
| geometry-medium | double | 0.861207 | 0.86985 | 0.879676 | 0.891604 | 7.60539 | 1.13071 | 0.952874 | 1155.7 |
| geometry-large | float | 3.32258 | 3.34549 | 3.35844 | 3.37377 | 9.16899 | 3.532 | 3.43126 | 300.365 |
| geometry-large | double | 3.35753 | 3.39763 | 3.42908 | 4.06303 | 13.6524 | 3.63982 | 3.46237 | 296.526 |
| all-selected-medium | float | 0.950419 | 0.953793 | 0.961074 | 0.965679 | 6.45914 | 1.19184 | — | 1050.44 |
| all-selected-medium | double | 0.879498 | 0.895098 | 0.911134 | 0.938577 | 6.57074 | 1.13311 | — | 1127.93 |
| geometry-most-selected-large | float | 3.34842 | 3.36613 | 3.39054 | 3.42779 | 9.36071 | 3.52776 | 3.45934 | 297.991 |
| geometry-most-selected-large | double | 3.51753 | 3.55878 | 3.57932 | 3.58674 | 12.0358 | 3.80551 | 3.61836 | 283.473 |
| geometry-small-cached | float | 0.157608 | 0.16399 | 0.167714 | 0.200057 | 4.9839 | 0.458882 | 0.20439 | 6139.51 |
| geometry-small-cached | double | 0.155556 | 0.158813 | 0.166154 | 0.168241 | 5.17133 | 0.468428 | 0.206056 | 6349.89 |
| geometry-medium-cached | float | 0.932929 | 0.935643 | 0.946029 | 0.983217 | 6.85237 | 1.26245 | 0.988253 | 1066.92 |
| geometry-medium-cached | double | 0.861767 | 0.874005 | 0.973111 | 1.73931 | 7.7228 | 1.17809 | 0.936725 | 1148.1 |
| geometry-large-cached | float | 3.31861 | 3.33106 | 3.34905 | 3.37436 | 10.0751 | 3.57259 | 3.40715 | 300.801 |
| geometry-large-cached | double | 3.3638 | 3.39771 | 3.4294 | 3.44105 | 12.8294 | 3.71436 | 3.43994 | 296.568 |
| geometry-most-selected-large-cached | float | 3.35003 | 3.36581 | 3.39034 | 3.85704 | 9.41868 | 3.57036 | 3.41246 | 297.585 |
| geometry-most-selected-large-cached | double | 3.51409 | 3.5496 | 3.58066 | 3.58471 | 12.7486 | 3.87566 | 3.57865 | 283.708 |
| optional-minimal-small | float | 0.148635 | 0.150444 | 0.1632 | 0.170196 | 4.37826 | 0.410112 | — | 6559.5 |
| optional-minimal-small | double | 0.146477 | 0.149362 | 0.165276 | 0.187325 | 4.01766 | 0.405475 | — | 6748.04 |
| optional-minimal-medium | float | 0.923415 | 0.926848 | 0.950376 | 0.971809 | 5.68704 | 1.2038 | — | 1079.01 |
| optional-minimal-medium | double | 0.851387 | 0.856566 | 0.878145 | 0.910549 | 5.79562 | 1.12631 | — | 1167.83 |
| optional-minimal-large | float | 3.3342 | 3.34355 | 3.3688 | 3.38156 | 8.08211 | 3.50081 | — | 299.485 |
| optional-minimal-large | double | 3.35941 | 3.39697 | 3.43078 | 3.44299 | 11.6626 | 3.63905 | — | 296.49 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/rank_one_residual_screen/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/rank_one_residual_screen/repeat-review.json).

## root_music_estimator

Editable scope: `operators/angle_estimation/root_music_estimator_op`. Output ports (both precisions exact): `aoa_estimates`, `doa_multi`, `eigenvalues`, `source_count`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/root_music_estimator.json)

Timing matrix complete: **False**. Primary timing failures: 12. Native captures complete: **False**. Exact repeat: **False**. Capture errors: 62. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: Complete-matrix native correctness or exact-repeat validation failed; A/A pilot not run. Timings and failure evidence remain visible..

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 58.1016 | 59.0466 | 59.4818 | 63.9944 | 18.0305 | 58.2208 | — | 17.1981 |
| baseline-small | double | 85.4754 | 86.6113 | 87.9238 | 89.9597 | 18.4563 | 85.9035 | — | 11.6685 |
| baseline-medium | float | 191.801 | 193.953 | 199.775 | 205.641 | 17.5583 | 195.75 | — | 5.19074 |
| baseline-medium | double | 416.442 | 423.461 | 435.062 | 457.438 | 18.0322 | 418.489 | — | 2.38412 |
| baseline-large | float | 766.342 | 770.339 | 772.114 | 824.005 | 17.0743 | 780.317 | — | 1.30193 |
| baseline-large | double | 3038.48 | 3049.24 | 3068.97 | 3096.14 | 19.5668 | 3050.56 | — | 0.328598 |
| parameter-medium | float | 191.869 | 193.598 | 198.891 | 207.17 | 17.9457 | 194.404 | — | 5.2043 |
| parameter-medium | double | 549.098 | 562.739 | 569.451 | 572.458 | 18.1607 | 548.701 | — | 1.8067 |
| parameter-large | float | 842.161 | 846.763 | 854.128 | 911.856 | 18.2948 | 855.09 | — | 1.18648 |
| parameter-large | double | 3547.48 | 3564.31 | 3599.65 | 3606.1 | 20.5719 | 3553.37 | — | 0.281731 |
| array-geometry-small | float | 57.9672 | 58.7986 | 63.1157 | 68.1895 | 18.6454 | 58.3991 | 58.8619 | 17.1234 |
| array-geometry-small | double | 85.6355 | 86.4459 | 87.2192 | 103.563 | 17.6474 | 85.6831 | 85.6993 | 11.664 |
| array-geometry-medium | float | 191.944 | 193.313 | 200.409 | 207.516 | 19.0112 | 192.85 | 191.73 | 5.20532 |
| array-geometry-medium | double | 415.895 | 430.785 | 432.729 | 436.141 | 18.0484 | 416.537 | 415.975 | 2.38867 |
| array-geometry-large | float | 763.059 | 765.416 | 768.098 | 771.078 | 18.61 | 775.913 | 762.582 | 1.30968 |
| array-geometry-large | double | 3034.4 | 3047.53 | 3083.64 | 3092.6 | 20.3981 | 3055.11 | 3036.57 | 0.329057 |
| spatial-smoothing-small | float | 57.9071 | 59.0725 | 61.7072 | 64.8652 | 18.4076 | 58.1782 | 46.5068 | 17.1851 |
| spatial-smoothing-small | double | 86.2432 | 87.3806 | 93.6903 | 97.0737 | 19.0233 | 86.0075 | 41.8162 | 11.5265 |
| spatial-smoothing-medium | double | 413.509 | 416.446 | 422.908 | 429.955 | 17.4281 | 415.629 | 472.908 | 2.41304 |
| spatial-smoothing-large | double | 3038.32 | 3046.05 | 3080.56 | 3096.59 | 19.6796 | 3047.17 | 2117.77 | 0.328897 |
| num-sources-small | float | 57.4986 | 59.0612 | 64.0317 | 68.16 | 18.5587 | 58.2498 | 52.2633 | 17.2764 |
| num-sources-small | double | 86.2074 | 86.966 | 94.5956 | 96.5861 | 16.961 | 88.3865 | 119.02 | 11.5422 |
| num-sources-medium | float | 192.149 | 194.162 | 200.032 | 208.853 | 18.1698 | 192.717 | 191.773 | 5.17566 |
| num-sources-medium | double | 417.4 | 420.69 | 424.259 | 435.239 | 18.4262 | 417.966 | 548.261 | 2.39436 |
| num-sources-large | float | 764.284 | 767.091 | 772.201 | 828.471 | 18.6119 | 786.478 | 843.66 | 1.30724 |
| num-sources-large | double | 3030.8 | 3051.3 | 3067.55 | 3087.04 | 20.2956 | 3040.49 | 3537.3 | 0.32963 |
| source-count-pfa-small | float | 57.5612 | 58.4717 | 60.2054 | 64.6484 | 18.5586 | 58.1355 | 57.8011 | 17.272 |
| source-count-pfa-small | double | 86.0227 | 87.0206 | 89.7216 | 91.7018 | 17.3385 | 86.2785 | 85.7348 | 11.588 |
| source-count-pfa-medium | float | 190.761 | 192.515 | 195.263 | 197.155 | 16.5678 | 194.538 | 191.261 | 5.23254 |
| source-count-pfa-medium | double | 416.009 | 418.966 | 427.241 | 430.676 | 17.0462 | 419.548 | 424.602 | 2.39766 |
| source-count-pfa-large | float | 764.396 | 766.75 | 772.295 | 772.776 | 18.2019 | 781.15 | 766.354 | 1.30787 |
| source-count-pfa-large | double | 3037.79 | 3044.19 | 3059.52 | 3092 | 19.4945 | 3050.21 | 3039.81 | 0.328955 |
| max-sources-small | float | 57.2275 | 58.3483 | 60.5889 | 65.6551 | 18.0538 | 58.3721 | 57.7312 | 17.4415 |
| max-sources-small | double | 86.0234 | 86.9886 | 87.6922 | 91.1374 | 17.8872 | 86.3692 | 85.9069 | 11.6009 |
| max-sources-medium | float | 189.564 | 190.831 | 197.829 | 212.834 | 17.1059 | 191.778 | 177.314 | 5.25978 |
| max-sources-medium | double | 416.003 | 417.628 | 430.59 | 434.079 | 17.587 | 418.532 | 612.507 | 2.39851 |
| max-sources-large | float | 766.748 | 770.373 | 781.621 | 828.598 | 18.7181 | 777.857 | 802.977 | 1.3028 |
| max-sources-large | double | 3039.02 | 3047.56 | 3051.68 | 3080.1 | 20.6842 | 3049.66 | 2552.37 | 0.328699 |
| transition-medium | float | 190.901 | 191.969 | 194.848 | 198.095 | 18.3272 | 192.23 | 192.202 | 5.23133 |
| transition-medium | double | 546.551 | 553.797 | 567.683 | 572.047 | 18.5279 | 548.352 | 546.895 | 1.82139 |
| array-geometry-small-cached | float | 57.9041 | 58.9685 | 63.8807 | 67.7577 | 16.823 | 57.608 | 57.7711 | 17.1729 |
| array-geometry-small-cached | double | 86.5481 | 87.7009 | 91.8673 | 98.3792 | 18.1492 | 86.5365 | 86.0786 | 11.5084 |
| array-geometry-medium-cached | float | 191.751 | 193.549 | 196.155 | 202.079 | 18.3115 | 192.346 | 191.636 | 5.20013 |
| array-geometry-medium-cached | double | 416.058 | 421.605 | 430.207 | 441.674 | 17.7216 | 417.168 | 429.901 | 2.40083 |
| array-geometry-large-cached | float | 765.566 | 769.851 | 773.022 | 818.551 | 19.2182 | 778.852 | 765.303 | 1.30454 |
| array-geometry-large-cached | double | 3039.07 | 3045.56 | 3053.01 | 3086.47 | 18.9718 | 3047.62 | 3036.88 | 0.328895 |
| spatial-smoothing-small-cached | float | 57.3985 | 58.5687 | 64.9243 | 74.998 | 16.9514 | 47.2434 | 46.1541 | 17.2791 |
| spatial-smoothing-small-cached | double | 85.7892 | 86.6359 | 87.3873 | 88.429 | 17.963 | 42.3717 | 41.5293 | 11.6456 |
| spatial-smoothing-medium-cached | double | 415.981 | 416.997 | 430.18 | 433.736 | 17.4142 | 474.802 | 470.955 | 2.4026 |
| spatial-smoothing-large-cached | double | 3039.62 | 3051.53 | 3087.12 | 3092.52 | 18.9346 | 2126.38 | 2118.77 | 0.32868 |
| num-sources-small-cached | float | 57.6678 | 58.7123 | 60.4056 | 65.8139 | 19.2265 | 54.5173 | 52.3039 | 17.2764 |
| num-sources-small-cached | double | 85.5085 | 87.0412 | 90.3275 | 98.5442 | 17.7116 | 118.795 | 118.825 | 11.6439 |
| num-sources-medium-cached | float | 191.271 | 193.173 | 196.608 | 200.024 | 17.038 | 193.828 | 191.356 | 5.21344 |
| num-sources-medium-cached | double | 415.221 | 418.512 | 423.958 | 433.594 | 17.4052 | 547.895 | 547.306 | 2.39963 |
| num-sources-large-cached | float | 763.468 | 766.728 | 769.796 | 771.715 | 18.2567 | 853.491 | 840.553 | 1.30899 |
| num-sources-large-cached | double | 3040.61 | 3055.68 | 3074.42 | 3094.67 | 18.3792 | 3554.85 | 3551 | 0.328462 |
| source-count-pfa-small-cached | float | 57.6092 | 58.7096 | 59.0945 | 62.0626 | 16.9071 | 57.9279 | 57.3307 | 17.2992 |
| source-count-pfa-small-cached | double | 86.1208 | 87.4948 | 91.9355 | 98.8535 | 17.1927 | 86.7421 | 85.3675 | 11.5392 |
| source-count-pfa-medium-cached | float | 190.915 | 192.123 | 195.941 | 198.628 | 17.2706 | 191.96 | 189.784 | 5.22872 |
| source-count-pfa-medium-cached | double | 416.022 | 418.906 | 423.599 | 428.757 | 17.2693 | 416.356 | 415.855 | 2.3996 |
| source-count-pfa-large-cached | float | 762.145 | 765.831 | 785.661 | 821.07 | 18.6501 | 771.935 | 762.398 | 1.30681 |
| source-count-pfa-large-cached | double | 3040.17 | 3063.67 | 3093.41 | 3096.54 | 20.1835 | 3047.35 | 3042.14 | 0.328252 |
| max-sources-small-cached | float | 57.3512 | 58.6928 | 59.4631 | 69.5427 | 19.0883 | 57.6103 | 57.5389 | 17.2985 |
| max-sources-small-cached | double | 86.2566 | 87.279 | 95.3643 | 102.214 | 17.6681 | 86.1581 | 86.4185 | 11.5535 |
| max-sources-medium-cached | float | 191.693 | 193.549 | 196.867 | 201.451 | 19.3523 | 180.889 | 178.232 | 5.21182 |
| max-sources-medium-cached | double | 415.408 | 417.451 | 420.823 | 440.327 | 18.5828 | 614.271 | 616.574 | 2.40528 |
| max-sources-large-cached | float | 764.377 | 768.683 | 774.278 | 824.293 | 19.1541 | 816.247 | 801.783 | 1.30725 |
| max-sources-large-cached | double | 3032.81 | 3046.34 | 3073.25 | 3082.85 | 19.7585 | 2553.18 | 2548.51 | 0.329325 |
| transition-medium-cached | float | 191.986 | 192.726 | 202.201 | 210.925 | 19.2642 | 191.961 | 192.193 | 5.20201 |
| transition-medium-cached | double | 547.575 | 557.536 | 568.474 | 571.256 | 19.5413 | 548.497 | 548.691 | 1.82129 |
| optional-minimal-small | float | 57.5708 | 58.7164 | 59.7526 | 60.0512 | 16.9168 | 58.016 | — | 17.2842 |
| optional-minimal-small | double | 85.8036 | 86.75 | 87.3391 | 91.178 | 16.3824 | 86.1211 | — | 11.6214 |
| optional-minimal-medium | float | 191.352 | 193.635 | 198.69 | 203.501 | 17.1469 | 192.843 | — | 5.21614 |
| optional-minimal-medium | double | 415.747 | 419.967 | 422.778 | 429.829 | 17.6068 | 418.444 | — | 2.40432 |
| optional-minimal-large | float | 766.501 | 773.885 | 805.506 | 826.053 | 17.9272 | 779.513 | — | 1.30049 |
| optional-minimal-large | double | 3038.21 | 3056.24 | 3086.68 | 3095.26 | 19.3794 | 3044.17 | — | 0.328748 |
| spatial-smoothing-medium | float | unavailable | — | — | — | — | — | — | — |
| spatial-smoothing-large | float | unavailable | — | — | — | — | — | — | — |
| spatial-smoothing-medium-cached | float | unavailable | — | — | — | — | — | — | — |
| spatial-smoothing-large-cached | float | unavailable | — | — | — | — | — | — | — |

[Descriptive partial timing matrix; not gate evidence](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/timing-recovery-2/root_music_estimator/partial-preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/root_music_estimator/repeat-review.json).

## snapshot_gather

Editable scope: `operators/angle_estimation/snapshot_gather_op`. Output ports (both precisions exact): `is_valid`, `snapshot`.

Application consumer inventory: `beamscan_estimator`, `cluster_dbscan`, `detection_reduction`, `feature_assembly`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/snapshot_gather.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 5; insufficient pair budget False; worst P90 paired delta 27.27%, worst baseline IQR 14.57%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/snapshot_gather/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0748425 | 0.0769739 | 0.0860902 | 0.090991 | 4.00176 | 0.314769 | — | 12760.9 |
| baseline-small | double | 0.0758435 | 0.0771085 | 0.0806629 | 0.111148 | 4.10629 | 0.316862 | — | 12673.2 |
| baseline-medium | float | 0.25169 | 0.263376 | 0.295097 | 0.313676 | 6.21321 | 0.435214 | — | 3877.63 |
| baseline-medium | double | 0.428681 | 0.469338 | 0.495465 | 0.529047 | 8.74123 | 0.690917 | — | 2284.42 |
| baseline-large | float | 1.76546 | 1.86967 | 1.92413 | 1.95267 | 18.6963 | 2.07178 | — | 554.138 |
| baseline-large | double | 3.22361 | 3.33976 | 3.37057 | 4.01752 | 31.1106 | 3.51353 | — | 308.514 |
| point-medium | float | 0.209736 | 0.237438 | 0.249546 | 0.257592 | 5.94485 | 0.382537 | — | 4577 |
| point-medium | double | 0.352116 | 0.381735 | 0.414397 | 0.421297 | 7.98651 | 0.611565 | — | 2769.03 |
| wide-medium | float | 0.332784 | 0.345486 | 0.37582 | 0.378853 | 6.33001 | 0.51952 | — | 2938.69 |
| wide-medium | double | 0.583468 | 0.615838 | 0.64343 | 0.651242 | 8.2924 | 0.84302 | — | 1685.96 |
| window-small | float | 0.0558105 | 0.0583247 | 0.0670887 | 0.094861 | 4.2917 | 0.300853 | 0.086815 | 16494.2 |
| window-small | double | 0.0568895 | 0.0645003 | 0.067051 | 0.090223 | 4.27546 | 0.308575 | 0.090046 | 16234.8 |
| window-medium | float | 0.206843 | 0.21891 | 0.245894 | 0.249389 | 6.57285 | 0.388501 | 0.257722 | 4724.82 |
| window-medium | double | 0.358606 | 0.39905 | 0.425573 | 0.466742 | 8.25919 | 0.616622 | 0.452361 | 2694.54 |
| window-large | float | 1.30953 | 1.36975 | 1.40291 | 1.4238 | 18.8093 | 1.5925 | 1.84676 | 754.419 |
| window-large | double | 2.47774 | 2.56311 | 2.61492 | 2.70323 | 31.509 | 2.84547 | 3.2763 | 402.458 |
| wide-window-medium | float | 0.254371 | 0.277387 | 0.307843 | 1.08001 | 6.56875 | 0.431344 | 0.353899 | 3833.28 |
| wide-window-medium | double | 0.425806 | 0.452443 | 0.485604 | 0.489186 | 9.00397 | 0.693538 | 0.614844 | 2302.59 |
| window-small-cached | float | 0.0560875 | 0.0668102 | 0.0822386 | 0.09138 | 4.24692 | 0.343704 | 0.082361 | 16247.5 |
| window-small-cached | double | 0.056259 | 0.0573513 | 0.0642321 | 0.092713 | 4.43652 | 0.335769 | 0.082435 | 16825.5 |
| window-medium-cached | float | 0.207557 | 0.214352 | 0.242702 | 0.254714 | 6.96785 | 0.439621 | 0.250658 | 4700.23 |
| window-medium-cached | double | 0.352533 | 0.391458 | 0.420837 | 0.428399 | 8.42428 | 0.702936 | 0.428742 | 2746.25 |
| window-large-cached | float | 1.29997 | 1.37276 | 1.40912 | 1.42428 | 18.1908 | 2.1739 | 1.75315 | 755.991 |
| window-large-cached | double | 2.48796 | 2.57629 | 2.58914 | 2.60873 | 31.8128 | 3.65802 | 3.34347 | 402.655 |
| wide-window-medium-cached | float | 0.252894 | 0.272473 | 0.290797 | 0.315594 | 6.61437 | 0.52352 | 0.336511 | 3876.77 |
| wide-window-medium-cached | double | 0.428907 | 0.464655 | 0.490503 | 0.517168 | 8.6312 | 0.872308 | 0.596298 | 2268.99 |
| optional-minimal-small | float | 0.0731115 | 0.0746909 | 0.0855125 | 0.115194 | 3.96893 | 0.327695 | — | 13047.1 |
| optional-minimal-small | double | 0.072718 | 0.0746103 | 0.076128 | 0.080399 | 3.78288 | 0.31014 | — | 13110.1 |
| optional-minimal-medium | float | 0.249649 | 0.267584 | 0.279295 | 0.293907 | 5.93876 | 0.43002 | — | 3911.69 |
| optional-minimal-medium | double | 0.428652 | 0.451507 | 0.494827 | 0.548658 | 8.71529 | 0.699872 | — | 2271.97 |
| optional-minimal-large | float | 1.76025 | 1.87074 | 1.89468 | 1.91584 | 17.7944 | 2.09436 | — | 557.216 |
| optional-minimal-large | double | 3.2313 | 3.34403 | 3.38994 | 4.43568 | 31.3741 | 3.49756 | — | 307.195 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/snapshot_gather/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/snapshot_gather/repeat-review.json).

## subarray_selection

Editable scope: `operators/angle_estimation/subarray_selection_op`. Output ports (both precisions exact): `array_geometry`, `is_valid_out`, `snapshot_out`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/subarray_selection.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 9; insufficient pair budget False; worst P90 paired delta 74.95%, worst baseline IQR 20.25%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/subarray_selection/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0204675 | 0.0209505 | 0.0220136 | 0.044343 | 3.676 | 0.253463 | — | 42524.4 |
| baseline-small | double | 0.0206255 | 0.0210775 | 0.0221676 | 0.029731 | 3.86441 | 0.252444 | — | 42741.8 |
| baseline-medium | float | 0.020593 | 0.0210289 | 0.021504 | 0.021722 | 3.72664 | 0.254325 | — | 42830.4 |
| baseline-medium | double | 0.020616 | 0.021096 | 0.0214176 | 0.02163 | 3.58409 | 0.260222 | — | 42589.8 |
| baseline-large | float | 0.0387965 | 0.0401145 | 0.0434853 | 0.054611 | 4.03543 | 0.267723 | — | 23592.7 |
| baseline-large | double | 0.039505 | 0.0401896 | 0.046587 | 0.055518 | 4.96869 | 0.283047 | — | 23481.5 |
| minimum-medium | float | 0.020523 | 0.0210236 | 0.0217696 | 0.033426 | 3.66192 | 0.252732 | — | 42292.6 |
| minimum-medium | double | 0.020556 | 0.021097 | 0.0325049 | 0.041935 | 3.37739 | 0.259398 | — | 41714.2 |
| array_geometry-small | float | 0.022014 | 0.0229225 | 0.0231667 | 0.043426 | 3.68703 | 0.254982 | 0.045731 | 39994.2 |
| array_geometry-small | double | 0.0220595 | 0.0224829 | 0.0232154 | 0.023482 | 3.83712 | 0.252648 | 0.045333 | 40109.1 |
| array_geometry-medium | float | 0.0221295 | 0.0227727 | 0.0231601 | 0.037417 | 3.88029 | 0.253473 | 0.047324 | 40109.7 |
| array_geometry-medium | double | 0.02188 | 0.0227618 | 0.0231395 | 0.050548 | 3.74621 | 0.257028 | 0.046408 | 40408 |
| array_geometry-large | float | 0.0404535 | 0.0412978 | 0.0424008 | 0.05938 | 4.13814 | 0.26888 | 0.064102 | 23006.4 |
| array_geometry-large | double | 0.04081 | 0.0425465 | 0.0489856 | 0.054185 | 4.46594 | 0.275103 | 0.064964 | 22532.5 |
| subarray-small | float | 0.0219395 | 0.022714 | 0.0243597 | 0.050148 | 3.83202 | 0.25524 | 0.046463 | 40299.7 |
| subarray-small | double | 0.0220835 | 0.0226672 | 0.0232093 | 0.034176 | 3.63751 | 0.248778 | 0.044991 | 39852 |
| subarray-medium | float | 0.021963 | 0.0226376 | 0.0231359 | 0.055935 | 3.83295 | 0.257889 | 0.050417 | 40213 |
| subarray-medium | double | 0.02212 | 0.0226781 | 0.0266224 | 0.047037 | 3.64216 | 0.262353 | 0.046139 | 39984.5 |
| subarray-large | float | 0.040361 | 0.0413423 | 0.042467 | 0.048398 | 4.03788 | 0.265121 | 0.051055 | 23027.8 |
| subarray-large | double | 0.0403705 | 0.0411779 | 0.0424334 | 0.043055 | 4.69229 | 0.276259 | 0.06 | 23100.4 |
| min-geometry-medium | float | 0.0220885 | 0.0227049 | 0.0237147 | 0.024019 | 3.79905 | 0.251047 | 0.046129 | 39971.8 |
| min-geometry-medium | double | 0.021801 | 0.0224278 | 0.0227456 | 0.023102 | 3.59453 | 0.252898 | 0.044796 | 40461.1 |
| array_geometry-small-cached | float | 0.0218565 | 0.022427 | 0.0300675 | 0.056342 | 3.92177 | 0.277333 | 0.041019 | 39979.1 |
| array_geometry-small-cached | double | 0.0221575 | 0.0228115 | 0.0272439 | 0.033111 | 3.86432 | 0.268713 | 0.039491 | 39874.6 |
| array_geometry-medium-cached | float | 0.022116 | 0.0229355 | 0.028126 | 0.047973 | 3.70834 | 0.265612 | 0.038991 | 39657.6 |
| array_geometry-medium-cached | double | 0.0221295 | 0.0225858 | 0.0232652 | 0.024 | 3.53952 | 0.272666 | 0.039092 | 39921.8 |
| array_geometry-large-cached | float | 0.0405415 | 0.0418303 | 0.0433962 | 0.048834 | 3.98917 | 0.280565 | 0.059815 | 22889.3 |
| array_geometry-large-cached | double | 0.040931 | 0.0418029 | 0.045586 | 0.054556 | 5.04478 | 0.297139 | 0.060787 | 22716.3 |
| subarray-small-cached | float | 0.0221715 | 0.0227033 | 0.0230837 | 0.049491 | 4.05311 | 0.263121 | 0.037602 | 39998.7 |
| subarray-small-cached | double | 0.022273 | 0.022809 | 0.023446 | 0.025723 | 3.86966 | 0.261416 | 0.0395 | 39731.2 |
| subarray-medium-cached | float | 0.0220555 | 0.0230014 | 0.023475 | 0.052343 | 3.9184 | 0.268177 | 0.038769 | 40152.6 |
| subarray-medium-cached | double | 0.0222645 | 0.0227618 | 0.0241595 | 0.042379 | 3.66637 | 0.266815 | 0.039945 | 39982.4 |
| subarray-large-cached | float | 0.0402175 | 0.0410182 | 0.0418177 | 0.042194 | 4.12865 | 0.276501 | 0.061028 | 23186.6 |
| subarray-large-cached | double | 0.0408935 | 0.0430938 | 0.0533226 | 0.056668 | 4.63048 | 0.286372 | 0.060344 | 22265.9 |
| min-geometry-medium-cached | float | 0.0219305 | 0.024234 | 0.0483826 | 0.060862 | 3.94158 | 0.26875 | 0.044639 | 38426.6 |
| min-geometry-medium-cached | double | 0.0220735 | 0.0225957 | 0.0229048 | 0.023602 | 3.71011 | 0.268492 | 0.039408 | 40180.3 |
| optional-all-small | float | 0.0209585 | 0.021787 | 0.0445006 | 0.220927 | 4.12265 | 0.253704 | — | 41573.3 |
| optional-all-small | double | 0.020741 | 0.0215296 | 0.0425643 | 0.046407 | 4.17 | 0.252473 | — | 40938.1 |
| optional-all-medium | float | 0.0204075 | 0.0210849 | 0.0213964 | 0.022019 | 4.05276 | 0.254334 | — | 42968.4 |
| optional-all-medium | double | 0.0206945 | 0.021084 | 0.0218738 | 0.029166 | 3.92929 | 0.257011 | — | 42450 |
| optional-all-large | float | 0.0391995 | 0.0401563 | 0.0411993 | 0.041963 | 4.25436 | 0.266343 | — | 23639 |
| optional-all-large | double | 0.039116 | 0.0422646 | 0.0476499 | 0.05438 | 4.85073 | 0.273621 | — | 23325.3 |
| mask-bool-small | float | 0.0376805 | 0.0386598 | 0.0391876 | 0.048139 | 4.23209 | 0.26589 | — | 24547.3 |
| mask-bool-small | double | 0.0378375 | 0.0386209 | 0.0416204 | 0.048518 | 4.15894 | 0.269121 | — | 24369 |
| mask-bool-medium | float | 0.037468 | 0.038808 | 0.0565496 | 0.061232 | 4.0182 | 0.262074 | — | 24225.4 |
| mask-bool-medium | double | 0.0380745 | 0.038862 | 0.0397242 | 0.043435 | 4.22152 | 0.267648 | — | 24321.4 |
| mask-bool-large | float | 0.03763 | 0.0388074 | 0.0460673 | 0.063668 | 4.57758 | 0.272121 | — | 24256.6 |
| mask-bool-large | double | 0.0386115 | 0.0397937 | 0.0404782 | 0.063676 | 4.96887 | 0.276065 | — | 24050.2 |
| mask-bool-geometry-small | float | 0.0378425 | 0.0386996 | 0.0395057 | 0.049046 | 4.37301 | 0.261148 | 0.060194 | 24593 |
| mask-bool-geometry-small | double | 0.037912 | 0.0388425 | 0.0400337 | 0.048047 | 4.15851 | 0.26163 | 0.063195 | 24356.3 |
| mask-bool-geometry-medium | float | 0.0381115 | 0.0389505 | 0.0398773 | 0.052417 | 4.08404 | 0.261112 | 0.061455 | 24273.3 |
| mask-bool-geometry-medium | double | 0.0387915 | 0.0394856 | 0.0449157 | 0.051519 | 4.06508 | 0.261259 | 0.061768 | 24041.9 |
| mask-bool-geometry-large | float | 0.038227 | 0.0391006 | 0.0460359 | 0.058806 | 4.651 | 0.270242 | 0.062583 | 24075.7 |
| mask-bool-geometry-large | double | 0.039037 | 0.0398657 | 0.0407311 | 0.069761 | 4.91299 | 0.273009 | 0.063963 | 23765.1 |
| mask-bool-subarray-small | float | 0.0377645 | 0.0386501 | 0.0400578 | 0.048908 | 4.48855 | 0.263138 | 0.060121 | 24532.5 |
| mask-bool-subarray-small | double | 0.0387175 | 0.0396879 | 0.0399372 | 0.060676 | 4.21754 | 0.26139 | 0.060981 | 23937.6 |
| mask-bool-subarray-medium | float | 0.038236 | 0.0393585 | 0.0399859 | 0.054631 | 4.23762 | 0.260501 | 0.065157 | 24256.1 |
| mask-bool-subarray-medium | double | 0.0382175 | 0.0389583 | 0.046249 | 0.054444 | 4.25692 | 0.260917 | 0.060167 | 24171.6 |
| mask-bool-subarray-large | float | 0.038435 | 0.0399603 | 0.0581935 | 0.075305 | 5.0988 | 0.278407 | 0.064769 | 23519.5 |
| mask-bool-subarray-large | double | 0.0386945 | 0.0392774 | 0.0447791 | 0.049389 | 5.10775 | 0.271852 | 0.062277 | 23998.9 |
| mask-uint8-small | float | 0.0377685 | 0.0385661 | 0.0525151 | 0.225278 | 4.07434 | 0.260565 | — | 24311.5 |
| mask-uint8-small | double | 0.038125 | 0.0387888 | 0.0393565 | 0.039805 | 4.08201 | 0.266129 | — | 24431.9 |
| mask-uint8-medium | float | 0.0375045 | 0.0382221 | 0.0386157 | 0.067917 | 4.19663 | 0.264148 | — | 24746 |
| mask-uint8-medium | double | 0.0378845 | 0.0387157 | 0.0392336 | 0.041213 | 4.0938 | 0.260074 | — | 24508.5 |
| mask-uint8-large | float | 0.037764 | 0.0384351 | 0.0435584 | 0.057065 | 4.43469 | 0.27487 | — | 24521.4 |
| mask-uint8-large | double | 0.039005 | 0.0394857 | 0.0463086 | 0.053723 | 5.01828 | 0.274695 | — | 23752.3 |
| mask-uint8-geometry-small | float | 0.038075 | 0.0390084 | 0.0445987 | 0.049657 | 4.13966 | 0.260344 | 0.061 | 24298.6 |
| mask-uint8-geometry-small | double | 0.0384035 | 0.0390985 | 0.04059 | 0.049787 | 4.17669 | 0.257872 | 0.060046 | 24094.6 |
| mask-uint8-geometry-medium | float | 0.0385135 | 0.0393078 | 0.0457998 | 0.050741 | 4.15567 | 0.266898 | 0.065981 | 23985.5 |
| mask-uint8-geometry-medium | double | 0.038255 | 0.0391644 | 0.0396135 | 0.040176 | 4.12802 | 0.261167 | 0.060158 | 24200.4 |
| mask-uint8-geometry-large | float | 0.037842 | 0.0385245 | 0.0389458 | 0.050103 | 4.57585 | 0.268723 | 0.059712 | 24621.8 |
| mask-uint8-geometry-large | double | 0.038806 | 0.0394815 | 0.045365 | 0.051815 | 4.94284 | 0.269889 | 0.062759 | 23959.3 |
| mask-uint8-subarray-small | float | 0.038194 | 0.0391815 | 0.0444439 | 0.051945 | 4.16758 | 0.266231 | 0.059529 | 24248.4 |
| mask-uint8-subarray-small | double | 0.0380975 | 0.039466 | 0.0475888 | 0.063046 | 4.19276 | 0.26713 | 0.059315 | 24163.6 |
| mask-uint8-subarray-medium | float | 0.0382785 | 0.0391969 | 0.0445435 | 0.052481 | 4.3867 | 0.264408 | 0.063899 | 24311.2 |
| mask-uint8-subarray-medium | double | 0.0387595 | 0.0395407 | 0.0398201 | 0.050241 | 4.35584 | 0.259917 | 0.0605 | 24019.6 |
| mask-uint8-subarray-large | float | 0.0382825 | 0.0393921 | 0.0442224 | 0.074278 | 4.80127 | 0.273694 | 0.075824 | 24142.9 |
| mask-uint8-subarray-large | double | 0.0389955 | 0.039817 | 0.041477 | 0.064991 | 5.0523 | 0.271659 | 0.06214 | 23846.7 |
| optional-minimal-small | float | 0.020454 | 0.0210957 | 0.0218286 | 0.030324 | 3.88508 | 0.255196 | — | 42627.2 |
| optional-minimal-small | double | 0.020602 | 0.0214278 | 0.0225052 | 0.056111 | 3.66571 | 0.251962 | — | 41815.4 |
| optional-minimal-medium | float | 0.0205925 | 0.0215012 | 0.022061 | 0.043092 | 3.84178 | 0.25762 | — | 42537.8 |
| optional-minimal-medium | double | 0.0209905 | 0.0226374 | 0.0421253 | 0.044908 | 3.45102 | 0.263361 | — | 39781 |
| optional-minimal-large | float | 0.0392125 | 0.040383 | 0.0419702 | 0.055176 | 4.62464 | 0.265788 | — | 23723 |
| optional-minimal-large | double | 0.0393475 | 0.0404281 | 0.0417625 | 0.042686 | 4.76466 | 0.276964 | — | 23611.3 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/subarray_selection/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-5/subarray_selection/repeat-review.json).

## cluster_dbscan

Editable scope: `operators/clustering/cluster_dbscan_op`. Output ports (both precisions exact): `labels`, `sizes`.

Application consumer inventory: `detection_reduction`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/cluster_dbscan.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 1; insufficient pair budget False; worst P90 paired delta 8.15%, worst baseline IQR 5.904%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/cluster_dbscan/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.65738 | 0.663025 | 0.698294 | 0.723862 | 3.38959 | 0.91239 | — | 1507.2 |
| baseline-small | double | 1.09201 | 1.10061 | 1.13283 | 1.17729 | 3.55151 | 1.34926 | — | 911.64 |
| baseline-medium | float | 2.74951 | 2.7631 | 2.79079 | 2.80203 | 3.3969 | 3.0057 | — | 362.791 |
| baseline-medium | double | 5.68844 | 5.71491 | 5.74418 | 5.76319 | 3.40016 | 5.94475 | — | 175.491 |
| baseline-large | float | 4.8358 | 4.84915 | 6.07067 | 7.20223 | 3.47614 | 5.08388 | — | 204.513 |
| baseline-large | double | 10.8886 | 10.9096 | 12.0556 | 13.1493 | 3.41908 | 11.1621 | — | 91.381 |
| wider-epsilon-medium | float | 2.74839 | 2.76125 | 2.79217 | 2.81986 | 3.56614 | 3.01632 | — | 363.016 |
| wider-epsilon-medium | double | 5.69022 | 5.70323 | 5.73018 | 5.99979 | 3.40642 | 5.95104 | — | 175.594 |
| dense-core-medium | float | 2.75078 | 2.7692 | 2.79658 | 2.84007 | 3.49682 | 3.02558 | — | 362.527 |
| dense-core-medium | double | 5.69337 | 5.70612 | 5.83526 | 6.40793 | 3.40273 | 5.95197 | — | 175.059 |
| cluster-cap-medium | float | 2.75218 | 2.7668 | 2.7979 | 2.82718 | 3.66201 | 3.09275 | — | 362.626 |
| cluster-cap-medium | double | 5.69581 | 5.71213 | 5.74371 | 6.26109 | 3.41066 | 6.01635 | — | 175.381 |
| epsilon-small | float | 0.660011 | 0.669359 | 0.6966 | 0.714697 | 3.65019 | 0.912418 | 0.694927 | 1502.08 |
| epsilon-small | double | 1.09774 | 1.10714 | 1.14824 | 1.78873 | 3.84204 | 1.35672 | 1.14967 | 904.075 |
| epsilon-medium | float | 2.7557 | 2.77092 | 2.78557 | 3.20437 | 3.96491 | 3.0074 | 2.78768 | 361.966 |
| epsilon-medium | double | 5.69432 | 5.71111 | 5.74276 | 5.75732 | 3.83022 | 5.9566 | 5.71201 | 175.426 |
| epsilon-large | float | 4.83775 | 4.88227 | 6.11082 | 7.13311 | 3.92542 | 5.09946 | 4.89344 | 204.013 |
| epsilon-large | double | 10.8873 | 10.9137 | 12.0835 | 13.36 | 3.4439 | 11.1512 | 10.9153 | 91.3978 |
| min-points-small | float | 0.659408 | 0.676253 | 1.83887 | 2.95103 | 3.64407 | 0.916168 | 0.68052 | 1403.01 |
| min-points-small | double | 1.09866 | 1.10797 | 2.38423 | 3.95866 | 3.57691 | 1.36369 | 1.15006 | 839.38 |
| min-points-medium | float | 2.75057 | 2.75894 | 2.7894 | 2.80464 | 3.48571 | 3.01034 | 2.77023 | 362.621 |
| min-points-medium | double | 5.69055 | 5.70473 | 5.73298 | 5.96217 | 3.51856 | 5.94197 | 5.69315 | 175.481 |
| min-points-large | float | 4.84061 | 4.86389 | 6.12826 | 7.31319 | 3.59238 | 5.13456 | 4.84693 | 204.126 |
| min-points-large | double | 10.8838 | 10.8986 | 12.1596 | 13.497 | 3.51084 | 11.1634 | 10.8955 | 91.4353 |
| max-clusters-small | float | 0.659746 | 0.675284 | 2.1778 | 3.60998 | 3.56869 | 0.928039 | 0.719214 | 1385.13 |
| max-clusters-small | double | 1.09789 | 1.10188 | 1.13176 | 1.1561 | 3.55384 | 1.36311 | 1.15125 | 906.888 |
| max-clusters-medium | float | 2.75082 | 2.76242 | 2.78882 | 2.79566 | 3.66845 | 2.99599 | 2.80513 | 362.726 |
| max-clusters-medium | double | 5.68865 | 5.70312 | 5.73471 | 5.75134 | 3.6909 | 5.95518 | 5.76317 | 175.617 |
| max-clusters-large | float | 4.83996 | 4.85571 | 6.16022 | 7.39494 | 3.59479 | 5.10452 | 4.91664 | 204.155 |
| max-clusters-large | double | 10.8854 | 10.9172 | 12.0438 | 13.1159 | 3.35537 | 11.145 | 10.9469 | 91.4121 |
| wide-core-medium | float | 2.74672 | 2.75523 | 2.77983 | 2.82445 | 3.5657 | 3.0337 | — | 363.822 |
| wide-core-medium | double | 5.6891 | 5.70632 | 5.74865 | 5.78707 | 3.38715 | 5.9425 | — | 175.583 |
| tight-cap-large | float | 4.8392 | 4.85536 | 4.89744 | 4.97698 | 3.64381 | 5.16323 | — | 206.342 |
| tight-cap-large | double | 10.8895 | 10.9051 | 12.0494 | 13.1272 | 3.42866 | 11.1801 | — | 91.3991 |
| epsilon-small-cached | float | 0.658913 | 0.664165 | 0.697522 | 0.709576 | 3.85593 | 0.925187 | 0.723604 | 1508.82 |
| epsilon-small-cached | double | 1.09964 | 1.10847 | 1.12748 | 1.2046 | 3.8554 | 1.3673 | 1.13047 | 905.771 |
| epsilon-medium-cached | float | 2.75349 | 2.76942 | 2.82157 | 2.85132 | 3.63153 | 3.01893 | 2.78263 | 362.113 |
| epsilon-medium-cached | double | 5.69611 | 5.71493 | 5.74744 | 5.77762 | 3.65984 | 5.99851 | 5.72691 | 175.344 |
| epsilon-large-cached | float | 4.84065 | 4.86335 | 6.08455 | 7.61786 | 3.84391 | 5.11739 | 4.88066 | 204.413 |
| epsilon-large-cached | double | 10.8888 | 10.924 | 12.1465 | 13.4149 | 3.74749 | 11.1931 | 10.9218 | 91.3182 |
| min-points-small-cached | float | 0.656942 | 0.675424 | 1.86248 | 2.95552 | 3.54327 | 0.9364 | 0.650288 | 1405.07 |
| min-points-small-cached | double | 1.09728 | 1.10726 | 2.37448 | 3.72529 | 3.6855 | 1.37052 | 1.10236 | 868.819 |
| min-points-medium-cached | float | 2.7474 | 2.76019 | 2.79966 | 2.83757 | 3.50792 | 3.00727 | 2.76094 | 363.394 |
| min-points-medium-cached | double | 5.68827 | 5.7099 | 5.72969 | 7.10908 | 3.65218 | 5.95833 | 5.69166 | 175.658 |
| min-points-large-cached | float | 4.83269 | 4.85412 | 6.04192 | 10.0093 | 3.52235 | 5.11441 | 4.84226 | 204.619 |
| min-points-large-cached | double | 10.8888 | 10.909 | 12.1536 | 13.3356 | 3.31047 | 11.1781 | 10.8886 | 91.3814 |
| max-clusters-small-cached | float | 0.659307 | 0.669264 | 1.86158 | 3.53199 | 3.54605 | 1.00039 | 0.667686 | 1406.3 |
| max-clusters-small-cached | double | 1.09802 | 1.11035 | 2.25014 | 3.61186 | 3.73097 | 1.40866 | 1.11099 | 869.954 |
| max-clusters-medium-cached | float | 2.74866 | 2.76371 | 3.01163 | 3.76206 | 3.82539 | 3.0471 | 2.76586 | 361.512 |
| max-clusters-medium-cached | double | 5.68788 | 5.70507 | 5.73448 | 5.79535 | 3.54119 | 6.00782 | 5.7011 | 175.59 |
| max-clusters-large-cached | float | 4.8393 | 4.84957 | 6.04563 | 7.2243 | 3.5696 | 5.14493 | 4.84064 | 204.529 |
| max-clusters-large-cached | double | 10.8889 | 10.9197 | 12.0815 | 13.6475 | 3.51921 | 11.2073 | 10.9014 | 91.3739 |
| optional-minimal-small | float | 0.652562 | 0.672809 | 0.702149 | 0.832817 | 3.24485 | 0.946233 | — | 1513.64 |
| optional-minimal-small | double | 1.0902 | 1.10804 | 1.12633 | 1.13745 | 3.10851 | 1.38071 | — | 911.376 |
| optional-minimal-medium | float | 2.74467 | 2.75994 | 2.78707 | 3.70406 | 3.11992 | 3.01604 | — | 363.551 |
| optional-minimal-medium | double | 5.68692 | 5.70425 | 5.7764 | 6.18405 | 3.34537 | 5.96496 | — | 175.402 |
| optional-minimal-large | float | 4.83486 | 4.84621 | 4.88462 | 5.22734 | 3.07925 | 5.09408 | — | 206.614 |
| optional-minimal-large | double | 10.8821 | 10.8956 | 10.9186 | 10.948 | 3.04332 | 11.1941 | — | 91.8376 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/cluster_dbscan/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/cluster_dbscan/repeat-review.json).

## cfar_detector

Editable scope: `operators/detection/cfar_detector_op`. Output ports (both precisions exact): `cfar_power`, `detection_idx`, `detection_mask`, `detection_score`, `noise_estimate`.

Application consumer inventory: `beamscan_estimator`, `cluster_dbscan`, `detection_reduction`, `feature_assembly`, `multi_source_expansion`, `mvdr_estimator`, `rank_one_residual_screen`, `snapshot_gather`, `tdm_sparse_compensation`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/cfar_detector.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 8; insufficient pair budget False; worst P90 paired delta 25.79%, worst baseline IQR 13.26%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/cfar_detector/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.101265 | 0.105153 | 0.124477 | 0.142881 | 5.83524 | 0.359473 | — | 8951.79 |
| baseline-small | double | 0.127001 | 0.139544 | 0.150583 | 0.156639 | 5.38607 | 0.390056 | — | 7262.61 |
| baseline-medium | float | 0.115274 | 0.124999 | 0.138533 | 0.148 | 5.77467 | 0.389455 | — | 7935.56 |
| baseline-medium | double | 0.190481 | 0.200358 | 0.238259 | 0.276287 | 6.0522 | 0.469046 | — | 4908.36 |
| baseline-large | float | 0.149157 | 0.161969 | 0.198824 | 0.205759 | 6.26787 | 0.419453 | — | 6275.82 |
| baseline-large | double | 0.313441 | 0.318879 | 0.343208 | 0.350092 | 6.70742 | 0.599955 | — | 3148.49 |
| backend-go-medium | float | 0.138143 | 0.15198 | 0.171702 | 0.198704 | 5.77888 | 342.327 | — | 6694.39 |
| backend-go-medium | double | 0.323283 | 0.33094 | 0.360916 | 0.367649 | 6.2035 | 340.72 | — | 2992.22 |
| backend-so-medium | float | 0.138 | 0.140386 | 0.16637 | 0.186611 | 5.72484 | 318.058 | — | 6795.48 |
| backend-so-medium | double | 0.324518 | 0.327824 | 0.359495 | 0.532825 | 6.04478 | 318.391 | — | 2978.93 |
| backend-os-medium | float | 0.148852 | 0.151373 | 0.160848 | 0.177361 | 6.20905 | 1.98221 | — | 6386.1 |
| backend-os-medium | double | 0.210404 | 0.217994 | 0.252718 | 0.287703 | 6.18572 | 2.61912 | — | 4536.61 |
| footprint-mid-medium | float | 0.179472 | 0.192263 | 0.215565 | 0.222705 | 6.39567 | 0.466805 | — | 5210.54 |
| footprint-mid-medium | double | 0.359635 | 0.366734 | 0.40078 | 0.418093 | 6.67576 | 0.654269 | — | 2692.69 |
| peak-medium | float | 0.127931 | 0.135361 | 0.165372 | 0.24902 | 5.8202 | 0.399473 | — | 7155.72 |
| peak-medium | double | 0.201338 | 0.220402 | 0.235506 | 0.24676 | 6.17866 | 0.521658 | — | 4674.63 |
| os_rank-small | float | 0.113787 | 0.117301 | 0.136515 | 0.152426 | 6.27009 | 1.95223 | 0.223066 | 8003.74 |
| os_rank-small | double | 0.137264 | 0.143478 | 0.16837 | 0.178685 | 6.0587 | 2.6737 | 0.241936 | 6888.15 |
| os_rank-medium | float | 0.149663 | 0.15618 | 0.183617 | 0.194779 | 6.21546 | 2.72012 | 0.251982 | 6262.27 |
| os_rank-medium | double | 0.211338 | 0.231408 | 0.262361 | 0.280759 | 6.32154 | 2.5949 | 0.330407 | 4461.68 |
| os_rank-large | float | 0.189445 | 0.199124 | 0.225438 | 0.239159 | 6.88861 | 1.98795 | 0.301074 | 4966.61 |
| os_rank-large | double | 0.313696 | 0.332446 | 0.366326 | 0.388278 | 7.0764 | 2.76456 | 0.430705 | 3064.09 |
| guard_cells-small | float | 0.101181 | 0.10544 | 0.133267 | 0.147085 | 5.73607 | 0.361538 | 0.203129 | 8964.01 |
| guard_cells-small | double | 0.126898 | 0.130733 | 0.162687 | 0.832454 | 5.54844 | 0.380742 | 0.225389 | 7288.81 |
| guard_cells-medium | float | 0.115487 | 0.121993 | 0.140335 | 0.159037 | 6.10483 | 0.370565 | 0.21525 | 7957.88 |
| guard_cells-medium | double | 0.199343 | 0.211816 | 0.231161 | 0.235814 | 6.10829 | 0.465668 | 0.27802 | 4761.99 |
| guard_cells-large | float | 0.141477 | 0.160659 | 0.202625 | 0.217353 | 6.41818 | 0.451103 | 0.238482 | 6392.02 |
| guard_cells-large | double | 0.312858 | 0.321511 | 0.346477 | 0.716954 | 7.16537 | 0.611224 | 0.352472 | 3115.83 |
| training_cells-small | float | 0.101365 | 0.103166 | 0.11006 | 0.178472 | 6.2377 | 0.379066 | 0.220871 | 9064.59 |
| training_cells-small | double | 0.128926 | 0.140562 | 0.170358 | 0.174992 | 5.64836 | 0.398436 | 0.249694 | 7064.11 |
| training_cells-medium | float | 0.117496 | 0.121327 | 0.149915 | 0.173212 | 6.11564 | 0.390048 | 0.244018 | 7819.37 |
| training_cells-medium | double | 0.199792 | 0.207975 | 0.225035 | 0.241769 | 6.22694 | 0.482102 | 0.359214 | 4806.94 |
| training_cells-large | float | 0.141371 | 0.160067 | 0.181956 | 0.193907 | 6.45119 | 0.447548 | 0.290435 | 6473.72 |
| training_cells-large | double | 0.314 | 0.320014 | 0.349345 | 0.391112 | 7.61742 | 0.611556 | 0.481898 | 3126.02 |
| pfa-small | float | 0.102296 | 0.105838 | 0.108527 | 0.144167 | 5.85778 | 0.369621 | 0.216612 | 8958.89 |
| pfa-small | double | 0.12501 | 0.128024 | 0.155979 | 0.163825 | 5.77069 | 0.393399 | 0.231518 | 7377.5 |
| pfa-medium | float | 0.115505 | 0.119527 | 0.158527 | 0.191556 | 5.85808 | 0.403251 | 0.235398 | 7881.46 |
| pfa-medium | double | 0.193385 | 0.200706 | 0.219674 | 0.234361 | 7.20938 | 0.478047 | 0.296981 | 4882.02 |
| pfa-large | float | 0.141351 | 0.151839 | 0.180272 | 0.186111 | 6.23864 | 0.426464 | 0.250129 | 6550.48 |
| pfa-large | double | 0.313528 | 0.322083 | 0.339291 | 0.35937 | 6.76347 | 0.586751 | 0.398037 | 3126.93 |
| alpha_lut-small | float | 0.102482 | 0.104254 | 0.109127 | 0.142222 | 5.94209 | 0.374751 | 0.134371 | 8979.48 |
| alpha_lut-small | double | 0.127487 | 0.139968 | 0.170023 | 0.195574 | 5.77876 | 0.39727 | 0.166157 | 7236.16 |
| alpha_lut-medium | float | 0.119046 | 0.132097 | 0.161918 | 0.172741 | 6.25141 | 0.402427 | 0.156983 | 7617.56 |
| alpha_lut-medium | double | 0.20013 | 0.208965 | 0.238579 | 0.243315 | 6.33758 | 0.461815 | 0.231741 | 4772.91 |
| alpha_lut-large | float | 0.144398 | 0.153522 | 0.19057 | 0.205926 | 6.2212 | 0.410241 | 0.186065 | 6409.56 |
| alpha_lut-large | double | 0.31021 | 0.320057 | 0.341881 | 0.363648 | 6.89402 | 0.641723 | 0.336102 | 3129.02 |
| threshold_scale-small | float | 0.101667 | 0.11208 | 0.116122 | 0.13812 | 5.97841 | 0.370648 | 0.120213 | 8785.47 |
| threshold_scale-small | double | 0.126158 | 0.131171 | 0.154547 | 0.184675 | 5.63214 | 0.388852 | 0.146852 | 7336.45 |
| threshold_scale-medium | float | 0.11812 | 0.12665 | 0.156005 | 0.250057 | 5.98781 | 0.388121 | 0.136195 | 7652.46 |
| threshold_scale-medium | double | 0.19462 | 0.201931 | 0.22428 | 0.237316 | 6.35112 | 0.45715 | 0.211 | 4864.98 |
| threshold_scale-large | float | 0.143501 | 0.153595 | 0.183866 | 0.196501 | 6.42088 | 0.451658 | 0.160955 | 6384.56 |
| threshold_scale-large | double | 0.31181 | 0.325219 | 0.358864 | 0.370251 | 7.68849 | 0.649528 | 0.327713 | 3097.39 |
| nci_axes-small | float | 0.136468 | 0.149501 | 0.175457 | 0.204047 | 6.18541 | 0.421622 | 0.314009 | 6751.78 |
| nci_axes-small | double | 0.22601 | 0.23443 | 0.272153 | 0.283056 | 6.60495 | 0.512955 | 0.353963 | 4200.38 |
| nci_axes-medium | float | 0.229741 | 0.236831 | 0.270928 | 0.288176 | 7.19047 | 0.532677 | 0.582195 | 4161.6 |
| nci_axes-medium | double | 0.544259 | 0.556636 | 0.57447 | 0.590233 | 8.6569 | 1.011 | 1.11323 | 1797.42 |
| nci_axes-large | float | 0.361673 | 0.376107 | 0.402157 | 0.433057 | 8.593 | 0.813112 | 1.02264 | 2686.29 |
| nci_axes-large | double | 1.00364 | 1.01495 | 1.04317 | 1.05372 | 9.90537 | 1.55774 | 1.39517 | 985.971 |
| max_detections-small | float | 0.101621 | 0.103917 | 0.109501 | 0.163935 | 6.17633 | 0.368472 | 0.111667 | 9044.08 |
| max_detections-small | double | 0.12656 | 0.136255 | 0.163499 | 0.177122 | 5.84495 | 0.403233 | 0.135528 | 7284.48 |
| max_detections-medium | float | 0.117029 | 0.121538 | 0.157605 | 0.83102 | 6.10358 | 0.398574 | 0.133472 | 7883.88 |
| max_detections-medium | double | 0.191648 | 0.200436 | 0.223928 | 0.236546 | 6.31739 | 0.471973 | 0.200315 | 4926.56 |
| max_detections-large | float | 0.141422 | 0.150771 | 0.183243 | 0.203463 | 6.87912 | 0.405306 | 0.161935 | 6491.24 |
| max_detections-large | double | 0.314172 | 0.319965 | 0.342635 | 0.353936 | 6.82683 | 0.622732 | 0.328889 | 3126.32 |
| footprint-pfa-medium | float | 0.18695 | 0.192455 | 0.219988 | 0.226991 | 6.53745 | 0.447186 | 0.282759 | 5146.81 |
| footprint-pfa-medium | double | 0.361023 | 0.370635 | 0.387534 | 0.411945 | 6.97197 | 0.642224 | 0.475704 | 2692.65 |
| peak-all-rank4-small | float | 0.238644 | 0.249014 | 0.282634 | 1.3614 | 6.6693 | 0.527954 | — | 3995.16 |
| peak-all-rank4-small | double | 0.435269 | 0.454389 | 0.473975 | 0.486103 | 6.75396 | 0.795492 | — | 2242.04 |
| peak-all-rank4-medium | float | 0.494148 | 0.517228 | 0.532891 | 0.535353 | 8.92522 | 0.993891 | — | 1977.09 |
| peak-all-rank4-medium | double | 1.1423 | 1.16848 | 1.17754 | 1.38184 | 9.75685 | 1.75846 | — | 866.252 |
| peak-all-rank4-large | float | 0.85459 | 0.889247 | 0.915706 | 0.957159 | 10.0939 | 1.52493 | — | 1151.85 |
| peak-all-rank4-large | double | 2.23511 | 2.25168 | 2.31124 | 2.5365 | 11.9827 | 2.97448 | — | 444.874 |
| os_rank-small-cached | float | 0.113418 | 0.116783 | 0.141265 | 0.148816 | 5.99064 | 2.03643 | 0.143843 | 8049.06 |
| os_rank-small-cached | double | 0.139093 | 0.146908 | 0.181233 | 0.192677 | 5.95579 | 2.62754 | 0.175954 | 6726.81 |
| os_rank-medium-cached | float | 0.150782 | 0.152973 | 0.170702 | 0.196241 | 6.27322 | 3.06639 | 0.190296 | 6292.31 |
| os_rank-medium-cached | double | 0.211552 | 0.221203 | 0.255284 | 0.260816 | 6.59887 | 2.90222 | 0.239972 | 4515.12 |
| os_rank-large-cached | float | 0.1903 | 0.203706 | 0.233795 | 0.243453 | 6.47695 | 2.28254 | 0.244444 | 4939.38 |
| os_rank-large-cached | double | 0.318242 | 0.340384 | 0.358261 | 0.38601 | 7.21031 | 3.56229 | 0.339056 | 3037.68 |
| guard_cells-small-cached | float | 0.100996 | 0.110922 | 0.128171 | 0.140305 | 5.77454 | 0.411705 | 0.135807 | 8960.97 |
| guard_cells-small-cached | double | 0.134514 | 0.141356 | 0.172803 | 0.205454 | 5.78406 | 0.392361 | 0.145074 | 6912.73 |
| guard_cells-medium-cached | float | 0.117662 | 0.128885 | 0.150341 | 0.155741 | 5.88173 | 0.455057 | 0.142723 | 7732.18 |
| guard_cells-medium-cached | double | 0.196833 | 0.200349 | 0.225639 | 0.262111 | 6.0342 | 0.557964 | 0.194186 | 4900.76 |
| guard_cells-large-cached | float | 0.14013 | 0.149772 | 0.185915 | 0.205778 | 6.45333 | 0.568213 | 0.162381 | 6580.92 |
| guard_cells-large-cached | double | 0.308532 | 0.317367 | 0.345185 | 0.361269 | 6.84534 | 0.701094 | 0.271575 | 3154.16 |
| training_cells-small-cached | float | 0.101889 | 0.113195 | 0.137151 | 0.149981 | 6.07197 | 0.602686 | 0.148852 | 8851.98 |
| training_cells-small-cached | double | 0.125658 | 0.130244 | 0.152654 | 1.12471 | 5.95021 | 0.607853 | 0.166278 | 7308.44 |
| training_cells-medium-cached | float | 0.116982 | 0.129752 | 0.145827 | 0.156028 | 6.26956 | 0.72002 | 0.168066 | 7754.7 |
| training_cells-medium-cached | double | 0.191398 | 0.201865 | 0.230867 | 0.236723 | 6.18169 | 0.834659 | 0.272094 | 4900.51 |
| training_cells-large-cached | float | 0.141927 | 0.158884 | 0.181226 | 0.190546 | 6.34667 | 0.777519 | 0.220537 | 6478.59 |
| training_cells-large-cached | double | 0.314194 | 0.317962 | 0.341543 | 0.360649 | 7.04573 | 1.09727 | 0.424306 | 3133.89 |
| pfa-small-cached | float | 0.10113 | 0.116342 | 0.13081 | 0.133705 | 5.89696 | 0.474621 | 0.13574 | 8878.05 |
| pfa-small-cached | double | 0.126334 | 0.130765 | 0.15801 | 0.171584 | 5.64404 | 0.45164 | 0.157825 | 7297.02 |
| pfa-medium-cached | float | 0.115088 | 0.116687 | 0.123727 | 0.158898 | 6.17035 | 0.572908 | 0.144351 | 8071.3 |
| pfa-medium-cached | double | 0.198816 | 0.204095 | 0.231014 | 0.248611 | 6.3411 | 0.642436 | 0.224953 | 4797.2 |
| pfa-large-cached | float | 0.142768 | 0.153656 | 0.182697 | 0.207483 | 6.4773 | 0.622695 | 0.1735 | 6425.23 |
| pfa-large-cached | double | 0.315153 | 0.322899 | 0.347302 | 0.362131 | 7.20735 | 0.822408 | 0.328094 | 3127.7 |
| alpha_lut-small-cached | float | 0.102093 | 0.104247 | 0.122146 | 0.150445 | 6.02237 | 0.394177 | 0.140686 | 8960.64 |
| alpha_lut-small-cached | double | 0.128199 | 0.134301 | 0.161528 | 0.168148 | 6.01885 | 0.42314 | 0.169639 | 7188.72 |
| alpha_lut-medium-cached | float | 0.117677 | 0.124513 | 0.164894 | 0.180435 | 6.22088 | 0.405788 | 0.153796 | 7764.97 |
| alpha_lut-medium-cached | double | 0.192319 | 0.202236 | 0.228 | 0.251853 | 6.41892 | 0.47564 | 0.226935 | 4907.13 |
| alpha_lut-large-cached | float | 0.142033 | 0.157044 | 0.186578 | 0.211963 | 6.44355 | 0.460853 | 0.215436 | 6370.22 |
| alpha_lut-large-cached | double | 0.313798 | 0.318166 | 0.341381 | 0.366047 | 7.86765 | 0.654372 | 0.338315 | 3129.51 |
| threshold_scale-small-cached | float | 0.101293 | 0.103122 | 0.107549 | 0.147148 | 5.86914 | 0.378666 | 0.111232 | 9086.93 |
| threshold_scale-small-cached | double | 0.126441 | 0.12878 | 0.14254 | 0.166472 | 5.80524 | 0.406844 | 0.13224 | 7378.69 |
| threshold_scale-medium-cached | float | 0.116102 | 0.117682 | 0.119931 | 0.1645 | 6.19456 | 0.39202 | 0.122629 | 8020.33 |
| threshold_scale-medium-cached | double | 0.199703 | 0.204364 | 0.232781 | 0.234316 | 6.38822 | 0.481732 | 0.200945 | 4823.97 |
| threshold_scale-large-cached | float | 0.140755 | 0.154104 | 0.181332 | 0.1991 | 6.34127 | 0.410016 | 0.155341 | 6496.46 |
| threshold_scale-large-cached | double | 0.305179 | 0.319207 | 0.346023 | 0.362507 | 6.64747 | 0.642986 | 0.323306 | 3162.87 |
| nci_axes-small-cached | float | 0.13562 | 0.143855 | 0.175101 | 0.19823 | 6.11941 | 0.574302 | 0.168767 | 6802.35 |
| nci_axes-small-cached | double | 0.227221 | 0.243141 | 0.264508 | 0.269146 | 6.40277 | 0.588181 | 0.201527 | 4174.8 |
| nci_axes-medium-cached | float | 0.245036 | 0.252237 | 0.274415 | 0.337655 | 7.38792 | 0.753838 | 0.594544 | 3937.48 |
| nci_axes-medium-cached | double | 0.544006 | 0.555031 | 0.577328 | 0.596006 | 8.28334 | 0.926726 | 0.831717 | 1801.21 |
| nci_axes-large-cached | float | 0.362596 | 0.378278 | 0.409544 | 0.430423 | 8.63851 | 1.75556 | 0.694413 | 2674.87 |
| nci_axes-large-cached | double | 1.00228 | 1.01677 | 1.05826 | 1.47042 | 9.90886 | 1.1269 | 1.05045 | 976.037 |
| max_detections-small-cached | float | 0.102684 | 0.104254 | 0.111484 | 0.196035 | 6.13832 | 0.381758 | 0.10887 | 8949.72 |
| max_detections-small-cached | double | 0.125879 | 0.138632 | 0.160233 | 0.169869 | 5.71154 | 0.387997 | 0.130972 | 7285.98 |
| max_detections-medium-cached | float | 0.117157 | 0.119833 | 0.128359 | 0.156842 | 6.406 | 0.397868 | 0.123093 | 7898.6 |
| max_detections-medium-cached | double | 0.189924 | 0.197531 | 0.219352 | 0.243601 | 6.18991 | 0.481664 | 0.196925 | 4981.44 |
| max_detections-large-cached | float | 0.141133 | 0.15133 | 0.187489 | 0.211619 | 6.35102 | 0.453952 | 0.159323 | 6491.21 |
| max_detections-large-cached | double | 0.313239 | 0.319016 | 0.34329 | 0.349868 | 6.81226 | 0.634562 | 0.32048 | 3140.74 |
| footprint-pfa-medium-cached | float | 0.181671 | 0.192542 | 0.22676 | 0.24886 | 6.37745 | 1.21484 | 0.212998 | 5146.7 |
| footprint-pfa-medium-cached | double | 0.361452 | 0.374096 | 0.436445 | 0.647977 | 6.95889 | 1.36907 | 0.396377 | 2630.4 |
| optional-minimal-small | float | 0.089857 | 0.100657 | 0.111766 | 0.127675 | 4.74238 | 0.37272 | — | 10459.1 |
| optional-minimal-small | double | 0.114365 | 0.117805 | 0.133354 | 0.14836 | 4.39854 | 0.393609 | — | 8396.27 |
| optional-minimal-medium | float | 0.103634 | 0.10621 | 0.112844 | 0.136065 | 5.43182 | 0.40098 | — | 9324.34 |
| optional-minimal-medium | double | 0.186897 | 0.191256 | 0.205514 | 0.218786 | 4.90207 | 0.481581 | — | 5274.23 |
| optional-minimal-large | float | 0.128319 | 0.140743 | 0.164347 | 0.181406 | 4.8467 | 0.438609 | — | 7461.46 |
| optional-minimal-large | double | 0.284087 | 0.288394 | 0.310742 | 0.324609 | 5.27481 | 0.622737 | — | 3504.49 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/cfar_detector/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/cfar_detector/repeat-review.json).

## detection_reduction

Editable scope: `operators/detection/detection_reduction_op`. Output ports (both precisions exact): `reduced_points`, `rep_idx`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/detection_reduction.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **False**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: Complete-matrix native correctness or exact-repeat validation failed; A/A pilot not run. Timings and failure evidence remain visible..

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.204171 | 0.208255 | 0.236711 | 0.248258 | 3.87706 | 0.45034 | — | 4781.24 |
| baseline-small | double | 0.202661 | 0.213423 | 0.22592 | 0.235851 | 3.85127 | 0.45998 | — | 4803.96 |
| baseline-medium | float | 0.235351 | 0.240268 | 0.275221 | 0.299527 | 4.23708 | 0.481535 | — | 4144.23 |
| baseline-medium | double | 0.234573 | 0.248815 | 0.309913 | 0.981308 | 4.02648 | 0.476665 | — | 4099.57 |
| baseline-large | float | 0.291119 | 0.298709 | 0.325477 | 0.333111 | 4.03562 | 0.546877 | — | 3356.77 |
| baseline-large | double | 0.293086 | 0.303909 | 0.327394 | 0.34897 | 3.9892 | 0.539914 | — | 3338.2 |
| first-medium | float | 0.191309 | 0.193974 | 0.218438 | 0.339118 | 4.0158 | 0.483646 | — | 5095.23 |
| first-medium | double | 0.193189 | 0.196719 | 0.248021 | 0.261869 | 3.99376 | 0.481848 | — | 4989.41 |
| medoid-medium | float | 0.251757 | 0.26564 | 0.286177 | 0.321017 | 3.96916 | 0.586349 | — | 3875.76 |
| medoid-medium | double | 0.254146 | 0.26472 | 0.288794 | 0.35598 | 3.77771 | 0.585534 | — | 3836.75 |
| weighted_mean-medium | float | 0.283013 | 0.295001 | 0.333206 | 1.16489 | 3.99407 | 1.1589 | — | 3436.61 |
| weighted_mean-medium | double | 0.285443 | 0.293083 | 0.328929 | 0.338248 | 4.00426 | 1.1349 | — | 3434.38 |
| l2-large | float | 0.296184 | 0.329131 | 0.371968 | 0.954837 | 3.78147 | 0.5621 | — | 3145.64 |
| l2-large | double | 0.295687 | 0.301487 | 0.333108 | 0.348878 | 3.85157 | 0.55145 | — | 3319.75 |
| representative-small | float | 0.205624 | 0.211262 | 0.248495 | 0.290535 | 4.15527 | 0.476414 | 0.341628 | 4735.34 |
| representative-small | double | 0.206568 | 0.214096 | 0.24286 | 0.25973 | 3.85856 | 0.449044 | 0.301378 | 4707.99 |
| representative-medium | float | 0.235415 | 0.245894 | 0.265043 | 0.278424 | 3.83619 | 0.470312 | 0.34372 | 4137.91 |
| representative-medium | double | 0.23467 | 0.248834 | 0.278961 | 0.288073 | 4.16534 | 0.474645 | 0.348499 | 4134.83 |
| representative-large | float | 0.295287 | 0.301034 | 0.328895 | 0.372128 | 3.97021 | 0.533812 | 0.435433 | 3327.78 |
| representative-large | double | 0.294471 | 0.308423 | 0.341963 | 0.398822 | 3.89519 | 0.551858 | 0.434443 | 3309.92 |
| score_method-small | float | 0.20468 | 0.212228 | 0.241858 | 0.262434 | 4.08783 | 0.475506 | 0.213851 | 4743.55 |
| score_method-small | double | 0.206004 | 0.215825 | 0.237424 | 1.72121 | 3.99504 | 0.459062 | 0.213054 | 4696.84 |
| score_method-medium | float | 0.234628 | 0.241704 | 0.280475 | 0.286424 | 4.09228 | 0.487545 | 0.25924 | 4154.73 |
| score_method-medium | double | 0.236513 | 0.247712 | 0.276795 | 0.316258 | 4.17176 | 0.522228 | 0.244101 | 4118.9 |
| score_method-large | float | 0.295994 | 0.30305 | 0.34204 | 1.17852 | 4.1238 | 0.557914 | 0.302064 | 3294.95 |
| score_method-large | double | 0.294998 | 0.307103 | 0.332762 | 0.346775 | 4.12894 | 0.560654 | 0.30522 | 3303.04 |
| feature_weights-small | float | 0.207105 | 0.212236 | 0.2485 | 0.286536 | 4.08264 | 0.472692 | 0.22036 | 4656.96 |
| feature_weights-small | double | 0.204878 | 0.211508 | 0.233203 | 0.248638 | 4.27544 | 0.453775 | 0.210165 | 4760.33 |
| feature_weights-medium | float | 0.235911 | 0.245438 | 0.277724 | 0.32122 | 4.0361 | 0.49035 | 0.276017 | 4143.12 |
| feature_weights-medium | double | 0.234985 | 0.256394 | 0.281464 | 0.295018 | 4.51814 | 0.493804 | 0.24198 | 4111.21 |
| feature_weights-large | float | 0.296739 | 0.307134 | 0.324139 | 0.371285 | 4.25987 | 0.59184 | 0.336887 | 3303.58 |
| feature_weights-large | double | 0.294234 | 0.308169 | 0.332733 | 0.429332 | 4.10171 | 0.5547 | 0.302036 | 3320.3 |
| weighted-weights-medium | float | 0.284864 | 0.298231 | 0.332202 | 0.341323 | 4.31851 | 1.13878 | 0.291322 | 3434.47 |
| weighted-weights-medium | double | 0.285226 | 0.296512 | 0.325935 | 0.33386 | 4.11973 | 1.14074 | 0.288664 | 3421.89 |
| representative-small-cached | float | 0.204304 | 0.210286 | 0.230242 | 0.276953 | 3.91426 | 0.552026 | 0.244101 | 4760.77 |
| representative-small-cached | double | 0.204434 | 0.220808 | 0.23685 | 0.256564 | 3.92519 | 0.549831 | 0.236536 | 4734.21 |
| representative-medium-cached | float | 0.233406 | 0.241791 | 0.263216 | 0.31848 | 4.13131 | 0.633905 | 0.269897 | 4149.43 |
| representative-medium-cached | double | 0.234215 | 0.244971 | 0.271589 | 0.292444 | 3.97636 | 0.602812 | 0.268842 | 4139.54 |
| representative-large-cached | float | 0.292948 | 0.300235 | 0.329396 | 0.345553 | 3.88758 | 0.676108 | 0.324896 | 3343.19 |
| representative-large-cached | double | 0.294471 | 0.304019 | 0.334895 | 0.527035 | 4.18218 | 0.686774 | 0.328971 | 3322.19 |
| score_method-small-cached | float | 0.2061 | 0.210378 | 0.229766 | 0.256369 | 4.20877 | 0.46334 | 0.210435 | 4755.2 |
| score_method-small-cached | double | 0.243966 | 0.252982 | 0.263199 | 0.284545 | 3.93703 | 0.512192 | 0.248916 | 4005.03 |
| score_method-medium-cached | float | 0.235276 | 0.240295 | 0.26264 | 0.264175 | 3.94253 | 0.486165 | 0.237934 | 4153.18 |
| score_method-medium-cached | double | 0.235017 | 0.241149 | 0.279467 | 0.309573 | 4.10151 | 0.488794 | 0.244119 | 4146.73 |
| score_method-large-cached | float | 0.29348 | 0.303906 | 0.342968 | 0.350137 | 4.1163 | 0.559516 | 0.300017 | 3323.53 |
| score_method-large-cached | double | 0.295322 | 0.301745 | 0.330226 | 0.338406 | 3.8999 | 0.56711 | 0.303333 | 3315.47 |
| feature_weights-small-cached | float | 0.203235 | 0.215688 | 0.240986 | 0.262156 | 4.07555 | 0.466341 | 0.211258 | 4750.04 |
| feature_weights-small-cached | double | 0.206194 | 0.219289 | 0.247404 | 0.296591 | 4.23316 | 0.468359 | 0.212083 | 4683.83 |
| feature_weights-medium-cached | float | 0.234411 | 0.248949 | 0.262064 | 0.273138 | 4.26364 | 0.496434 | 0.240897 | 4158.52 |
| feature_weights-medium-cached | double | 0.234359 | 0.240246 | 0.270766 | 0.276675 | 4.53484 | 0.503016 | 0.238648 | 4159.17 |
| feature_weights-large-cached | float | 0.292767 | 0.304378 | 0.326729 | 0.373277 | 4.05321 | 0.556294 | 0.303138 | 3346.96 |
| feature_weights-large-cached | double | 0.296643 | 0.309956 | 0.336141 | 0.35247 | 4.23494 | 0.564626 | 0.300619 | 3297.49 |
| weighted-weights-medium-cached | float | 0.284883 | 0.297251 | 0.32371 | 0.336442 | 4.2253 | 1.12877 | 0.287915 | 3426.39 |
| weighted-weights-medium-cached | double | 0.284914 | 0.295154 | 0.320543 | 0.349359 | 3.977 | 1.11239 | 0.290906 | 3433.45 |
| optional-minimal-small | float | 0.184443 | 0.190787 | 0.207719 | 0.234999 | 3.80526 | 0.459424 | — | 5257.62 |
| optional-minimal-small | double | 0.185111 | 0.191013 | 0.22362 | 0.241832 | 3.79907 | 0.4586 | — | 5247.15 |
| optional-minimal-medium | float | 0.199174 | 0.21014 | 0.234306 | 0.285378 | 3.67856 | 0.538665 | — | 4877.57 |
| optional-minimal-medium | double | 0.199216 | 0.204719 | 0.228313 | 0.242249 | 3.67582 | 0.489609 | — | 4887.69 |
| optional-minimal-large | float | 0.232327 | 0.243121 | 0.275693 | 0.282257 | 3.84628 | 0.553516 | — | 4208.85 |
| optional-minimal-large | double | 0.231077 | 0.241544 | 0.279803 | 0.292314 | 3.54851 | 0.548563 | — | 4214.07 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/detection_reduction/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/detection_reduction/repeat-review.json).

## feature_assembly

Editable scope: `operators/detection/feature_assembly_op`. Output ports (both precisions exact): `points`.

Application consumer inventory: `cluster_dbscan`, `detection_reduction`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/feature_assembly.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 12; insufficient pair budget False; worst P90 paired delta 28.01%, worst baseline IQR 21.38%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/feature_assembly/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0550595 | 0.0565275 | 0.074396 | 0.082445 | 3.8151 | 0.298934 | — | 16964 |
| baseline-small | double | 0.0556015 | 0.0567888 | 0.0640516 | 0.08762 | 3.53843 | 0.288137 | — | 16836.5 |
| baseline-medium | float | 0.0552685 | 0.0573343 | 0.0700691 | 0.075518 | 3.75089 | 0.296201 | — | 16904.1 |
| baseline-medium | double | 0.0556895 | 0.0569071 | 0.0628918 | 0.082832 | 3.72624 | 0.297323 | — | 16945.6 |
| baseline-large | float | 0.0566015 | 0.0626777 | 0.0749477 | 0.086731 | 3.87256 | 0.30386 | — | 16354.2 |
| baseline-large | double | 0.0567085 | 0.0595121 | 0.0726686 | 0.079426 | 3.78054 | 0.292907 | — | 16378.5 |
| sparse-medium | float | 0.0551385 | 0.0562555 | 0.0615763 | 0.15211 | 3.72178 | 0.290842 | — | 17087.6 |
| sparse-medium | double | 0.0554625 | 0.0571337 | 0.0620311 | 0.08474 | 3.6689 | 0.292694 | — | 16914.7 |
| dense-medium | float | 0.0553655 | 0.0580939 | 0.0739986 | 0.099269 | 3.53505 | 0.299573 | — | 16870 |
| dense-medium | double | 0.055713 | 0.0573622 | 0.0621451 | 0.08612 | 3.73098 | 0.301331 | — | 16844 |
| empty-medium | float | 0.024782 | 0.0253448 | 0.0260721 | 0.03637 | 3.84711 | 0.266555 | — | 35838.3 |
| empty-medium | double | 0.024949 | 0.0254189 | 0.0303536 | 0.04538 | 3.53503 | 0.267989 | — | 35937.8 |
| dense-large | float | 0.066671 | 0.0745796 | 0.0883432 | 0.099924 | 4.14841 | 0.304407 | — | 13881.4 |
| dense-large | double | 0.0666475 | 0.0678691 | 0.0709438 | 0.07362 | 4.42313 | 0.303693 | — | 14392.5 |
| empty-small | float | 0.0249675 | 0.0256841 | 0.0259261 | 0.026398 | 3.51938 | 0.260814 | — | 35795.8 |
| empty-small | double | 0.0251845 | 0.0255849 | 0.0257612 | 0.026862 | 3.68803 | 0.268156 | — | 35519.3 |
| dense-small | float | 0.0553935 | 0.0565189 | 0.064649 | 0.070334 | 3.71631 | 0.299192 | — | 17042.8 |
| dense-small | double | 0.0558055 | 0.0566797 | 0.0617029 | 0.070889 | 3.72591 | 0.29749 | — | 17018.1 |
| score-pack-small | float | 0.064398 | 0.0663215 | 0.0830993 | 0.206537 | 3.81488 | 0.294129 | — | 14681.7 |
| score-pack-small | double | 0.0650045 | 0.0700444 | 0.110327 | 0.180351 | 3.79777 | 0.296183 | — | 13691.6 |
| score-pack-medium | float | 0.064236 | 0.066574 | 0.0720047 | 0.091102 | 3.95796 | 0.302972 | — | 14538.4 |
| score-pack-medium | double | 0.0646295 | 0.0670203 | 0.0879226 | 0.104277 | 4.3566 | 0.30998 | — | 14421.1 |
| score-pack-large | float | 0.090319 | 0.092779 | 0.121679 | 0.209786 | 5.43689 | 0.324509 | — | 10385.2 |
| score-pack-large | double | 0.198495 | 0.212045 | 0.224258 | 0.242129 | 6.2226 | 0.371581 | — | 4926.12 |
| score-aoa-pack-small | float | 0.072027 | 0.0737382 | 0.0970765 | 0.102165 | 4.00195 | 0.299054 | — | 13111.8 |
| score-aoa-pack-small | double | 0.0725555 | 0.0747114 | 0.0961671 | 0.107869 | 4.18286 | 0.304712 | — | 12884.1 |
| score-aoa-pack-medium | float | 0.0720135 | 0.0736773 | 0.0791705 | 0.095767 | 4.19678 | 0.311786 | — | 13121.8 |
| score-aoa-pack-medium | double | 0.072764 | 0.0753714 | 0.0955009 | 0.152639 | 4.60419 | 0.299434 | — | 12934.3 |
| score-aoa-pack-large | float | 0.0936995 | 0.106315 | 0.11746 | 0.147925 | 5.63393 | 0.333711 | — | 10152 |
| score-aoa-pack-large | double | 0.197833 | 0.221797 | 0.236844 | 0.654016 | 7.339 | 0.379045 | — | 4856.18 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/feature_assembly/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/feature_assembly/repeat-review.json).

## multi_source_expansion

Editable scope: `operators/detection/multi_source_expansion_op`. Output ports (both precisions exact): `aoa_estimates_out`, `detection_idx_out`.

Application consumer inventory: `feature_assembly`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/multi_source_expansion.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 2; insufficient pair budget False; worst P90 paired delta 16.45%, worst baseline IQR 8.132%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/multi_source_expansion/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.123634 | 0.125825 | 0.134337 | 0.153268 | 5.15414 | 0.361526 | — | 8001.35 |
| baseline-small | double | 0.125341 | 0.126545 | 0.128564 | 0.18385 | 4.52898 | 0.350933 | — | 7782.35 |
| baseline-medium | float | 0.467581 | 0.469711 | 0.489352 | 0.510053 | 4.74202 | 0.686154 | — | 2120.79 |
| baseline-medium | double | 0.5779 | 0.581441 | 0.604821 | 0.616174 | 4.82498 | 0.791006 | — | 1716.74 |
| baseline-large | float | 2.86457 | 2.88061 | 2.89442 | 2.9061 | 5.40422 | 3.10145 | — | 348.298 |
| baseline-large | double | 3.74497 | 3.75191 | 3.76492 | 3.7742 | 5.88564 | 3.96999 | — | 266.739 |
| sparse-medium | float | 0.281726 | 0.284198 | 0.304009 | 0.310248 | 4.71074 | 0.510415 | — | 3493.54 |
| sparse-medium | double | 0.33061 | 0.333283 | 0.35587 | 0.388157 | 5.02627 | 0.551692 | — | 2981.8 |
| dense-medium | float | 0.77534 | 0.784993 | 0.803651 | 0.807117 | 5.00136 | 0.994227 | — | 1281.83 |
| dense-medium | double | 0.998524 | 1.01833 | 1.02843 | 1.05484 | 4.93575 | 1.22456 | — | 995.542 |
| four-sources-medium | float | 0.614771 | 0.620702 | 0.634888 | 0.649775 | 4.74033 | 0.829923 | — | 1615.15 |
| four-sources-medium | double | 0.824831 | 0.838523 | 0.850478 | 0.944505 | 4.86853 | 1.04064 | — | 1203.8 |
| wide-indices-medium | float | 0.467761 | 0.471212 | 0.492707 | 0.716858 | 4.94139 | 0.680432 | — | 2117.33 |
| wide-indices-medium | double | 0.578284 | 0.583446 | 0.593244 | 0.607859 | 4.47362 | 0.775932 | — | 1716.28 |
| full-selected-large | float | 15.0076 | 15.0123 | 15.0343 | 15.0631 | 6.00788 | 15.2352 | — | 66.6176 |
| full-selected-large | double | 21.5321 | 21.5691 | 21.5866 | 22.2854 | 6.4455 | 21.7584 | — | 46.415 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/multi_source_expansion/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/multi_source_expansion/repeat-review.json).

## clutter_filter

Editable scope: `operators/fmcw/clutter_filter_op`. Output ports (both precisions exact): `range_profiles_out`.

Application consumer inventory: `beamscan_estimator`, `cfar_detector`, `cluster_dbscan`, `detection_reduction`, `feature_assembly`, `fmcw_doppler_processor`, `multi_source_expansion`, `mvdr_estimator`, `phase_shift_beamformer`, `rank_one_residual_screen`, `snapshot_gather`, `tdm_sparse_compensation`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/clutter_filter.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 19; insufficient pair budget False; worst P90 paired delta 45.7%, worst baseline IQR 25.42%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/clutter_filter/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.663382 | 0.711064 | 1.48711 | 1.69896 | 3.16258 | 0.969839 | — | 1357.7 |
| baseline-small | double | 0.518424 | 0.544098 | 1.40914 | 1.4394 | 3.23385 | 0.767766 | — | 1725.52 |
| baseline-medium | float | 0.524447 | 0.549252 | 1.5204 | 1.57845 | 3.1 | 0.608748 | — | 1720.97 |
| baseline-medium | double | 0.549994 | 0.568931 | 1.47315 | 1.68263 | 3.42397 | 0.682303 | — | 1620.56 |
| baseline-large | float | 0.551424 | 0.572077 | 1.62436 | 1.66784 | 3.64638 | 0.686017 | — | 1631.69 |
| baseline-large | double | 0.660651 | 0.693043 | 1.64252 | 1.73682 | 3.97435 | 0.740914 | — | 1358.83 |
| single-delay-medium | float | 0.0338845 | 0.0358826 | 0.0413143 | 0.047065 | 3.34247 | 0.283712 | — | 27760.2 |
| single-delay-medium | double | 0.0336065 | 0.0341936 | 0.0360662 | 0.037666 | 3.44491 | 0.275777 | — | 27125.2 |
| double-delay-medium | float | 0.034556 | 0.0356724 | 0.0410657 | 0.051463 | 3.34269 | 0.279128 | — | 26409.6 |
| double-delay-medium | double | 0.036046 | 0.0370287 | 0.0378661 | 0.052611 | 3.46656 | 0.286527 | — | 25445.1 |
| passthrough-medium | float | 0.0315 | 0.0353024 | 0.0435036 | 0.07236 | 3.22794 | 0.263564 | — | 28706.4 |
| passthrough-medium | double | 0.0315975 | 0.0322114 | 0.032609 | 0.061194 | 3.30906 | 0.26899 | — | 29043.5 |
| method-small | float | 0.548285 | 0.587187 | 1.46048 | 1.53988 | 3.24338 | 0.76147 | 0.163453 | 1603.44 |
| method-small | double | 0.677679 | 0.697463 | 1.54314 | 1.66388 | 3.29625 | 0.791026 | 0.164083 | 1330.9 |
| method-medium | float | 0.573697 | 0.599675 | 1.65241 | 2.02047 | 3.31191 | 0.677034 | 0.164138 | 1550.75 |
| method-medium | double | 0.607998 | 0.643828 | 1.59054 | 1.67176 | 3.59308 | 0.714377 | 0.146008 | 1502.2 |
| method-large | float | 0.615017 | 0.685199 | 1.69072 | 2.02333 | 3.77901 | 0.702617 | 0.171879 | 1396.07 |
| method-large | double | 0.687984 | 0.807054 | 1.71386 | 1.82502 | 4.25644 | 0.791349 | 0.192657 | 1288.01 |
| window-small | float | 0.567438 | 0.664889 | 1.462 | 1.53462 | 3.37837 | 0.766414 | 0.87396 | 1572.53 |
| window-small | double | 0.556007 | 0.603253 | 1.46134 | 1.67776 | 3.41688 | 0.769571 | 0.840989 | 1613.52 |
| window-medium | float | 0.576183 | 0.618869 | 1.66567 | 1.7287 | 3.50664 | 0.691609 | 1.00042 | 1511.12 |
| window-medium | double | 0.609568 | 0.649415 | 1.56731 | 1.70771 | 3.60608 | 0.7121 | 0.92433 | 1461.09 |
| window-large | float | 0.703372 | 0.74165 | 1.57982 | 1.7553 | 3.81491 | 0.688386 | 1.21758 | 1270.53 |
| window-large | double | 0.674478 | 0.719407 | 1.55188 | 1.66254 | 4.29493 | 0.797693 | 1.25863 | 1355.31 |
| four-channels-medium | float | 0.575272 | 0.61792 | 1.6648 | 1.74662 | 3.45673 | 0.690359 | — | 1497.65 |
| four-channels-medium | double | 0.693359 | 0.778161 | 1.68222 | 1.82363 | 3.97926 | 0.804413 | — | 1262.9 |
| hann-medium | float | 0.647795 | 0.697223 | 1.5784 | 1.73646 | 3.18792 | 0.981043 | — | 1364.21 |
| hann-medium | double | 0.611206 | 0.652994 | 1.50577 | 1.61787 | 3.33151 | 0.916691 | — | 1452.04 |
| sixteen-channels-large | float | 1.20428 | 1.65937 | 2.06473 | 2.15187 | 5.64893 | 1.43802 | — | 761.352 |
| sixteen-channels-large | double | 2.03133 | 2.17979 | 2.96749 | 3.17199 | 8.36602 | 2.17761 | — | 473.465 |
| method-small-cached | float | 0.484842 | 0.519648 | 1.41705 | 1.50673 | 3.15813 | 0.278703 | 0.056815 | 1765.36 |
| method-small-cached | double | 0.516147 | 0.548938 | 1.32707 | 1.38728 | 3.16554 | 0.27098 | 0.061426 | 1751.59 |
| method-medium-cached | float | 0.554331 | 0.580951 | 1.4911 | 1.68441 | 3.47928 | 0.291565 | 0.066028 | 1645.32 |
| method-medium-cached | double | 0.551044 | 0.570149 | 1.60413 | 1.65524 | 3.41511 | 0.283777 | 0.06162 | 1577.14 |
| method-large-cached | float | 0.601276 | 0.668843 | 1.60013 | 1.71277 | 3.53117 | 0.28912 | 0.072722 | 1469.58 |
| method-large-cached | double | 0.663905 | 0.723827 | 1.67259 | 1.74633 | 4.47683 | 0.299342 | 0.085241 | 1339.13 |
| window-small-cached | float | 0.544892 | 0.56673 | 1.58495 | 1.67923 | 3.56632 | 1.0102 | 1.39557 | 1658.24 |
| window-small-cached | double | 0.561017 | 0.598506 | 1.60272 | 1.6574 | 3.55368 | 1.02314 | 0.694609 | 1542.97 |
| window-medium-cached | float | 0.557952 | 0.575956 | 1.56174 | 1.64962 | 3.39366 | 1.08741 | 0.794006 | 1636.57 |
| window-medium-cached | double | 0.620405 | 0.66003 | 1.64671 | 1.8018 | 3.74588 | 1.09718 | 0.767739 | 1424.85 |
| window-large-cached | float | 0.613507 | 0.647042 | 1.6546 | 1.71568 | 3.9335 | 1.25463 | 0.925349 | 1433.05 |
| window-large-cached | double | 0.684072 | 0.798618 | 1.7456 | 1.79881 | 4.34214 | 1.12047 | 0.833998 | 1266.26 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/clutter_filter/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/clutter_filter/repeat-review.json).

## ddm_dense_demultiplex

Editable scope: `operators/fmcw/ddm_dense_demultiplex_op`. Output ports (both precisions exact): `rd_maps_out`, `velocity_bins_out`.

Application consumer inventory: `beamscan_estimator`, `cfar_detector`, `cluster_dbscan`, `detection_reduction`, `doppler_notch`, `feature_assembly`, `snapshot_gather`, `virtual_array_assembly`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/ddm_dense_demultiplex.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 6; insufficient pair budget False; worst P90 paired delta 32.8%, worst baseline IQR 26.86%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_dense_demultiplex/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.032412 | 0.0334514 | 0.0444277 | 0.051547 | 3.0711 | 0.265878 | — | 27654.8 |
| baseline-small | double | 0.0328605 | 0.035642 | 0.0485295 | 0.059129 | 3.11799 | 0.262907 | — | 27052.1 |
| baseline-medium | float | 0.098694 | 0.10058 | 0.130945 | 0.135898 | 6.41473 | 0.333156 | — | 9739.79 |
| baseline-medium | double | 0.350096 | 0.359806 | 0.376386 | 0.39049 | 9.53318 | 0.548461 | — | 2821.87 |
| baseline-large | float | 1.18145 | 1.24491 | 1.2711 | 1.53669 | 38.7901 | 1.38134 | — | 833.611 |
| baseline-large | double | 2.60487 | 2.6778 | 2.70921 | 2.72207 | 71.7328 | 2.80708 | — | 381.962 |
| nonlinear-medium | float | 0.911183 | 0.952616 | 0.967398 | 0.974256 | 6.99936 | 34.6989 | — | 1081.84 |
| nonlinear-medium | double | 4.86307 | 4.90074 | 4.93528 | 4.96333 | 8.90386 | 40.9688 | — | 205.168 |
| linear-axis-medium | float | 0.144569 | 0.15633 | 0.176077 | 0.181184 | 6.57914 | 0.330267 | — | 6579.34 |
| linear-axis-medium | double | 0.362596 | 0.372768 | 0.390219 | 0.40699 | 9.70456 | 0.535212 | — | 2707.89 |
| nonlinear-axis-medium | float | 0.925423 | 0.958079 | 0.98066 | 1.024 | 7.22458 | 34.2447 | — | 1065.94 |
| nonlinear-axis-medium | double | 4.86286 | 4.89294 | 4.90558 | 4.94148 | 10.112 | 40.4306 | — | 205.187 |
| nonlinear-small | float | 0.0831485 | 0.0849261 | 0.103231 | 0.126436 | 3.06761 | 27.813 | — | 11478.7 |
| nonlinear-small | double | 0.116101 | 0.124832 | 0.141918 | 0.149064 | 3.04143 | 29.8582 | — | 8275.21 |
| nonlinear-large | float | 21.1726 | 21.2538 | 21.3303 | 22.6296 | 38.6312 | 65.8155 | — | 47.2078 |
| nonlinear-large | double | 88.7111 | 88.8022 | 88.8586 | 89.1078 | 71.9501 | 125.193 | — | 11.2708 |
| optional-minimal-small | float | 0.0325095 | 0.0331085 | 0.0574403 | 0.48548 | 3.12425 | 0.270295 | — | 27832.1 |
| optional-minimal-small | double | 0.0405135 | 0.0453546 | 0.0554938 | 0.069315 | 3.1886 | 0.268758 | — | 22333.7 |
| optional-minimal-medium | float | 0.0983795 | 0.101783 | 0.102812 | 0.10511 | 6.20553 | 0.335999 | — | 9875.32 |
| optional-minimal-medium | double | 0.351133 | 0.363183 | 0.383228 | 0.399823 | 8.94193 | 0.540646 | — | 2803.2 |
| optional-minimal-large | float | 1.18685 | 1.24746 | 1.28975 | 1.31633 | 38.2004 | 1.46214 | — | 831.471 |
| optional-minimal-large | double | 2.5999 | 2.67675 | 2.72505 | 2.74236 | 70.437 | 2.81165 | — | 381.963 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/ddm_dense_demultiplex/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/ddm_dense_demultiplex/repeat-review.json).

## ddm_power_fold

Editable scope: `operators/fmcw/ddm_power_fold_op`. Output ports (both precisions exact): `power_map`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/ddm_power_fold.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 6; insufficient pair budget False; worst P90 paired delta 32.88%, worst baseline IQR 21.33%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_power_fold/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0313975 | 0.0325967 | 0.0381544 | 0.051601 | 2.66461 | 0.260981 | — | 28527.3 |
| baseline-small | double | 0.0336025 | 0.0411197 | 0.0504701 | 0.055435 | 2.96235 | 0.265434 | — | 25884.5 |
| baseline-medium | float | 0.067949 | 0.06999 | 0.071983 | 0.100602 | 3.86436 | 0.310267 | — | 14067.4 |
| baseline-medium | double | 0.231485 | 0.237868 | 0.256697 | 0.271342 | 4.43365 | 0.417202 | — | 4238.77 |
| baseline-large | float | 0.835461 | 0.901793 | 0.934227 | 0.949728 | 8.78707 | 1.09858 | — | 1166.55 |
| baseline-large | double | 1.73335 | 1.81889 | 1.86031 | 1.88769 | 12.3816 | 1.95436 | — | 573.498 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/ddm_power_fold/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/ddm_power_fold/repeat-review.json).

## ddm_sparse_demultiplex

Editable scope: `operators/fmcw/ddm_sparse_demultiplex_op`. Output ports (both precisions exact): `array_geometry`, `is_valid`, `snapshot`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/ddm_sparse_demultiplex.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 7; insufficient pair budget False; worst P90 paired delta 21.18%, worst baseline IQR 17.03%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/ddm_sparse_demultiplex/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0629445 | 0.0653168 | 0.0661018 | 0.094074 | 4.89806 | 0.33798 | — | 14939.2 |
| baseline-small | double | 0.0632455 | 0.0668261 | 0.0847825 | 0.093732 | 5.03587 | 0.334351 | — | 14503.7 |
| baseline-medium | float | 0.132217 | 0.136542 | 0.155643 | 0.173073 | 6.03279 | 0.407148 | — | 7289.29 |
| baseline-medium | double | 0.250393 | 0.259045 | 0.293136 | 0.307583 | 6.63107 | 0.469314 | — | 3919.86 |
| baseline-large | float | 0.975952 | 1.01429 | 1.04023 | 1.2983 | 12.7433 | 1.33164 | — | 1011.19 |
| baseline-large | double | 1.59238 | 1.68262 | 1.69294 | 1.72677 | 18.6826 | 1.93451 | — | 620.67 |
| neighborhood-medium | float | 0.13042 | 0.133152 | 0.152388 | 0.157269 | 5.75603 | 0.388721 | — | 7406.23 |
| neighborhood-medium | double | 0.249467 | 0.266658 | 0.284772 | 0.295471 | 6.98307 | 0.456935 | — | 3899.04 |
| dedup-medium | float | 0.132347 | 0.135832 | 0.157488 | 0.166889 | 5.63231 | 0.391304 | — | 7304.93 |
| dedup-medium | double | 0.243457 | 0.260166 | 0.291205 | 0.304379 | 6.65941 | 0.450424 | — | 3983.93 |
| wavelength-small | float | 0.062782 | 0.0644359 | 0.0701926 | 0.094426 | 4.92664 | 0.327915 | 0.08012 | 15073.4 |
| wavelength-small | double | 0.063324 | 0.0646825 | 0.0681693 | 0.083528 | 5.00785 | 0.328804 | 0.069934 | 15009.9 |
| wavelength-medium | float | 0.132176 | 0.136355 | 0.148849 | 0.163194 | 5.86804 | 0.395165 | 0.135417 | 7348.39 |
| wavelength-medium | double | 0.253976 | 0.272652 | 0.301708 | 0.312602 | 6.84879 | 0.46098 | 0.258158 | 3834.66 |
| wavelength-large | float | 0.979382 | 1.01116 | 1.03419 | 1.05097 | 12.5112 | 1.2949 | 0.984173 | 1012.1 |
| wavelength-large | double | 1.58865 | 1.67961 | 1.70685 | 1.76311 | 19.1694 | 1.90577 | 1.58842 | 619.011 |
| window-small | float | 0.063574 | 0.0660128 | 0.0840929 | 0.095898 | 4.80053 | 0.32211 | 0.075517 | 14725.8 |
| window-small | double | 0.0639675 | 0.0730404 | 0.0833367 | 0.094861 | 5.25158 | 0.336721 | 0.075676 | 14533.3 |
| window-medium | float | 0.131523 | 0.134053 | 0.156262 | 0.164129 | 6.05178 | 0.405471 | 0.141778 | 7408.3 |
| window-medium | double | 0.249638 | 0.272261 | 0.290313 | 0.300463 | 7.31106 | 0.457972 | 0.262212 | 3862.46 |
| window-large | float | 0.97879 | 1.00396 | 1.04381 | 1.05665 | 13.0042 | 1.29167 | 1.01212 | 1012.34 |
| window-large | double | 1.59534 | 1.68593 | 1.73083 | 1.75651 | 19.5186 | 1.93295 | 1.6798 | 616.071 |
| average-window-medium | float | 0.135337 | 0.139218 | 0.152286 | 0.163584 | 6.31008 | 0.414193 | 0.145935 | 7184.31 |
| average-window-medium | double | 0.345315 | 0.36044 | 0.376032 | 0.384222 | 7.21946 | 0.547101 | 0.352157 | 2858.52 |
| wavelength-small-cached | float | 0.0636435 | 0.0650229 | 0.0785182 | 0.105129 | 4.87396 | 0.324556 | 0.068093 | 14811.3 |
| wavelength-small-cached | double | 0.0642825 | 0.0736955 | 0.0887464 | 0.133296 | 4.79366 | 0.335323 | 0.069277 | 14470.3 |
| wavelength-medium-cached | float | 0.132838 | 0.138388 | 0.162253 | 0.16723 | 5.93648 | 0.396656 | 0.134704 | 7250.52 |
| wavelength-medium-cached | double | 0.25269 | 0.268124 | 0.292693 | 0.318129 | 7.00414 | 0.461119 | 0.266203 | 3842.79 |
| wavelength-large-cached | float | 0.975688 | 1.02322 | 1.04762 | 1.05748 | 11.8389 | 1.28196 | 0.980266 | 1013.58 |
| wavelength-large-cached | double | 1.58717 | 1.66997 | 1.6942 | 1.70832 | 19.0551 | 1.91371 | 1.62965 | 621.219 |
| window-small-cached | float | 0.0626485 | 0.0650923 | 0.0695241 | 0.099239 | 4.92735 | 0.328286 | 0.067638 | 14938.3 |
| window-small-cached | double | 0.0649395 | 0.0685936 | 0.0889148 | 0.09837 | 4.99577 | 0.338945 | 0.069823 | 14299.6 |
| window-medium-cached | float | 0.132389 | 0.13565 | 0.155898 | 0.16725 | 5.95641 | 0.39636 | 0.133574 | 7308.74 |
| window-medium-cached | double | 0.255528 | 0.269543 | 0.291117 | 0.306202 | 7.17445 | 0.462666 | 0.255888 | 3807.86 |
| window-large-cached | float | 0.987545 | 1.01463 | 1.02848 | 1.05033 | 12.4428 | 1.33428 | 1.01978 | 1004.16 |
| window-large-cached | double | 1.59059 | 1.67622 | 1.70847 | 1.71735 | 19.1437 | 1.93701 | 1.59306 | 620.86 |
| average-window-medium-cached | float | 0.133972 | 0.139421 | 0.156874 | 0.162287 | 6.22601 | 0.420184 | 0.135592 | 7186.78 |
| average-window-medium-cached | double | 0.342425 | 0.346711 | 0.37288 | 0.380239 | 7.43649 | 0.564805 | 0.343026 | 2882.44 |
| optional-minimal-small | float | 0.060528 | 0.0646807 | 0.0844077 | 0.089389 | 4.57303 | 0.32275 | — | 15236.4 |
| optional-minimal-small | double | 0.0607585 | 0.0621365 | 0.0717211 | 0.227406 | 4.76892 | 0.323686 | — | 15578.1 |
| optional-minimal-medium | float | 0.129703 | 0.131621 | 0.150245 | 0.166658 | 6.20402 | 0.399592 | — | 7462.8 |
| optional-minimal-medium | double | 0.253064 | 0.266074 | 0.280886 | 0.293693 | 6.61801 | 0.44762 | — | 3863.53 |
| optional-minimal-large | float | 0.972507 | 0.996461 | 1.03551 | 1.45591 | 12.1672 | 1.29906 | — | 1019.03 |
| optional-minimal-large | double | 1.58554 | 1.67245 | 1.70071 | 1.71967 | 18.6949 | 1.90497 | — | 621.712 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/ddm_sparse_demultiplex/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/ddm_sparse_demultiplex/repeat-review.json).

## doppler_notch

Editable scope: `operators/fmcw/doppler_notch_op`. Output ports (both precisions exact): `rd_maps_out`.

Application consumer inventory: `beamscan_estimator`, `cfar_detector`, `cluster_dbscan`, `detection_reduction`, `feature_assembly`, `snapshot_gather`, `virtual_array_assembly`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/doppler_notch.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 6; insufficient pair budget False; worst P90 paired delta 24.36%, worst baseline IQR 19.44%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/doppler_notch/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.041546 | 0.0426029 | 0.0463089 | 0.068491 | 3.38485 | 0.265019 | — | 22333.7 |
| baseline-small | double | 0.04181 | 0.0452585 | 0.0550395 | 0.058592 | 3.35077 | 0.271574 | — | 22128.3 |
| baseline-medium | float | 0.0881105 | 0.089167 | 0.100023 | 0.112704 | 3.43983 | 0.310841 | — | 10939.6 |
| baseline-medium | double | 0.0858425 | 0.0868855 | 0.0883751 | 0.100639 | 3.71972 | 0.313397 | — | 11218.3 |
| baseline-large | float | 0.438791 | 0.440227 | 0.442694 | 0.449786 | 5.29587 | 0.667202 | — | 2265.37 |
| baseline-large | double | 0.453026 | 0.456553 | 0.469288 | 0.478879 | 5.31379 | 0.673425 | — | 2186 |
| cosine-medium | float | 0.0873845 | 0.0882863 | 0.0935632 | 0.109158 | 3.85977 | 0.31137 | — | 11051.2 |
| cosine-medium | double | 0.0859355 | 0.0869006 | 0.0931674 | 0.109083 | 3.94998 | 0.310786 | — | 11235.8 |
| mimo-medium | float | 0.856688 | 0.86189 | 0.868879 | 0.889701 | 6.12776 | 1.07705 | — | 1161.86 |
| mimo-medium | double | 0.937896 | 0.946159 | 0.961916 | 0.988601 | 7.51184 | 1.09863 | — | 1061.38 |
| noncyclic-medium | float | 0.0779305 | 0.0791219 | 0.0820884 | 0.090582 | 3.58082 | 0.305351 | — | 12316.3 |
| noncyclic-medium | double | 0.0770785 | 0.0781494 | 0.0888926 | 0.100259 | 3.90338 | 0.299841 | — | 12407.4 |
| center-small | float | 0.0423055 | 0.043083 | 0.0525935 | 0.068278 | 3.48493 | 0.268869 | 0.059861 | 21934.1 |
| center-small | double | 0.0417775 | 0.0427366 | 0.0521572 | 0.098213 | 3.52631 | 0.267611 | 0.05775 | 21786.1 |
| center-medium | float | 0.0874205 | 0.09002 | 0.0963633 | 0.098787 | 3.61984 | 0.31024 | 0.10312 | 10963.8 |
| center-medium | double | 0.086236 | 0.0881073 | 0.0974998 | 0.107185 | 3.86413 | 0.315639 | 0.100899 | 11131.2 |
| center-large | float | 0.438819 | 0.441244 | 0.450643 | 0.454239 | 4.74956 | 0.659842 | 0.453342 | 2259.89 |
| center-large | double | 0.452295 | 0.456208 | 0.459677 | 0.473047 | 5.62033 | 0.674944 | 0.456602 | 2191.43 |
| stop-small | float | 0.0411665 | 0.0418469 | 0.0424943 | 0.058482 | 3.51047 | 0.265676 | 0.058732 | 22568.5 |
| stop-small | double | 0.0417405 | 0.0430946 | 0.0461237 | 0.061223 | 3.37704 | 0.260797 | 0.057231 | 21992.8 |
| stop-medium | float | 0.088111 | 0.0897394 | 0.0980677 | 0.106694 | 3.68934 | 0.31536 | 0.103148 | 10898.6 |
| stop-medium | double | 0.086426 | 0.0882131 | 0.0994854 | 0.110648 | 4.11097 | 0.310684 | 0.101121 | 11084.3 |
| stop-large | float | 0.439651 | 0.441243 | 0.451511 | 0.454638 | 4.72097 | 0.662119 | 0.453759 | 2257.4 |
| stop-large | double | 0.452346 | 0.455931 | 0.463591 | 0.481185 | 5.54465 | 0.685388 | 0.456656 | 2192.88 |
| transition-small | float | 0.048666 | 0.0519899 | 0.0529233 | 0.06388 | 3.27654 | 0.273943 | 0.067009 | 19348.3 |
| transition-small | double | 0.0415415 | 0.0422765 | 0.0460024 | 0.058953 | 3.48326 | 0.264851 | 0.059639 | 22399 |
| transition-medium | float | 0.0875505 | 0.0890111 | 0.0916756 | 0.097508 | 4.09476 | 0.31212 | 0.103777 | 11004.9 |
| transition-medium | double | 0.08619 | 0.08792 | 0.0989031 | 0.101675 | 3.84898 | 0.307815 | 0.100537 | 11095.8 |
| transition-large | float | 0.43974 | 0.442269 | 0.456643 | 0.466554 | 4.95268 | 0.665564 | 0.456722 | 2254.75 |
| transition-large | double | 0.452842 | 0.45657 | 0.469843 | 0.481804 | 5.5839 | 0.677378 | 0.459665 | 2189.24 |
| cosine-mimo-medium | float | 0.868035 | 0.872808 | 0.883333 | 0.886481 | 5.46892 | 1.08846 | — | 1146.05 |
| cosine-mimo-medium | double | 0.949606 | 0.956202 | 0.99627 | 1.05036 | 7.28642 | 1.10991 | — | 1046.67 |
| noncyclic-mimo-large | float | 5.87056 | 5.91014 | 5.93205 | 6.20013 | 15.072 | 6.1218 | — | 170.129 |
| noncyclic-mimo-large | double | 6.23502 | 6.31106 | 6.33566 | 6.33892 | 26.307 | 6.50012 | — | 159.959 |
| center-small-cached | float | 0.0413145 | 0.0419843 | 0.0474133 | 0.073231 | 3.44743 | 0.272314 | 0.045148 | 22501.4 |
| center-small-cached | double | 0.0417545 | 0.0435347 | 0.0463963 | 0.055574 | 3.27926 | 0.268639 | 0.046491 | 22091.5 |
| center-medium-cached | float | 0.088153 | 0.0892114 | 0.0923567 | 0.097898 | 3.62999 | 0.316852 | 0.091787 | 10989.2 |
| center-medium-cached | double | 0.086079 | 0.0870787 | 0.0877065 | 0.092341 | 3.91593 | 0.316046 | 0.089315 | 11206 |
| center-large-cached | float | 0.43923 | 0.441732 | 0.45234 | 0.456204 | 4.67958 | 0.66573 | 0.442332 | 2257.98 |
| center-large-cached | double | 0.452466 | 0.456562 | 0.466079 | 0.478073 | 5.47557 | 0.684999 | 0.456527 | 2191.19 |
| stop-small-cached | float | 0.0414625 | 0.0426906 | 0.0471842 | 0.051435 | 3.50208 | 0.273954 | 0.045667 | 22277.6 |
| stop-small-cached | double | 0.041528 | 0.0435491 | 0.0464351 | 0.055665 | 3.52112 | 0.271843 | 0.046417 | 22122.1 |
| stop-medium-cached | float | 0.087287 | 0.0893867 | 0.0987426 | 0.110713 | 4.01646 | 0.316999 | 0.09476 | 11010.3 |
| stop-medium-cached | double | 0.0860275 | 0.0876419 | 0.0959317 | 0.105074 | 4.34676 | 0.318119 | 0.090065 | 11154.9 |
| stop-large-cached | float | 0.439031 | 0.44103 | 0.449491 | 0.451443 | 4.94361 | 0.672674 | 0.442166 | 2258.62 |
| stop-large-cached | double | 0.453629 | 0.456253 | 0.467545 | 0.479906 | 6.30825 | 0.674444 | 0.454721 | 2184.93 |
| transition-small-cached | float | 0.0418935 | 0.043087 | 0.0443524 | 0.0555 | 3.34324 | 0.265295 | 0.046397 | 22076.3 |
| transition-small-cached | double | 0.0419175 | 0.04355 | 0.0469545 | 0.057018 | 3.33144 | 0.267361 | 0.047046 | 21925.8 |
| transition-medium-cached | float | 0.0876575 | 0.0886129 | 0.0892372 | 0.095889 | 3.98939 | 0.31663 | 0.090814 | 11014.7 |
| transition-medium-cached | double | 0.086519 | 0.0876675 | 0.0960886 | 0.101889 | 3.9174 | 0.322388 | 0.096416 | 11084.6 |
| transition-large-cached | float | 0.439245 | 0.442071 | 0.450466 | 0.458064 | 5.10248 | 0.676221 | 0.444749 | 2257.55 |
| transition-large-cached | double | 0.452287 | 0.456855 | 0.473405 | 0.480387 | 6.39914 | 0.683805 | 0.461175 | 2190.08 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/doppler_notch/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/doppler_notch/repeat-review.json).

## fmcw_doppler_processor

Editable scope: `operators/fmcw/fmcw_doppler_processor_op`. Output ports (both precisions exact): `rd_maps`, `velocity_bins`.

Application consumer inventory: `beamscan_estimator`, `cfar_detector`, `cluster_dbscan`, `ddm_dense_demultiplex`, `detection_reduction`, `doppler_notch`, `feature_assembly`, `multi_source_expansion`, `mvdr_estimator`, `phase_shift_beamformer`, `rank_one_residual_screen`, `snapshot_gather`, `tdm_sparse_compensation`, `virtual_array_assembly`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/fmcw_doppler_processor.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 12; insufficient pair budget False; worst P90 paired delta 82.78%, worst baseline IQR 59.43%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_doppler_processor/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0329125 | 0.0337113 | 0.0342139 | 0.055574 | 3.97322 | 0.26488 | — | 26513.3 |
| baseline-small | double | 0.035472 | 0.0508333 | 0.0532857 | 0.056528 | 3.98844 | 0.268564 | — | 21873.6 |
| baseline-medium | float | 0.0329535 | 0.0337147 | 0.0340713 | 0.050435 | 3.85839 | 26.3175 | — | 26458.2 |
| baseline-medium | double | 0.0643565 | 0.0739216 | 0.0790183 | 1.04136 | 4.00068 | 30.2763 | — | 14212.6 |
| baseline-large | float | 0.0637405 | 0.0651531 | 0.0674618 | 0.107343 | 4.20546 | 39.4738 | — | 14525.8 |
| baseline-large | double | 0.147759 | 0.151464 | 0.165507 | 0.172193 | 4.89934 | 34.8131 | — | 6515.87 |
| hann-medium | float | 0.040972 | 0.0425569 | 0.0475735 | 0.08112 | 3.98796 | 18.2401 | — | 21683.8 |
| hann-medium | double | 0.0694255 | 0.0710499 | 0.0990409 | 0.1105 | 3.9121 | 21.4196 | — | 13198 |
| four-channels-medium | float | 0.0703 | 0.0792492 | 0.0940222 | 0.110148 | 4.19757 | 26.3187 | — | 12885.8 |
| four-channels-medium | double | 0.142699 | 0.144478 | 0.14547 | 0.146676 | 5.29842 | 29.6189 | — | 6787.44 |
| padded-medium | float | 0.044287 | 0.0681363 | 0.0712858 | 0.088787 | 4.01675 | 23.3144 | — | 17883 |
| padded-medium | double | 0.100875 | 0.104324 | 0.128314 | 0.13725 | 4.93638 | 26.7624 | — | 9366.3 |
| window-small | float | 0.033736 | 0.0362452 | 0.0569352 | 0.068454 | 4.00196 | 0.262944 | 13.523 | 25016.5 |
| window-small | double | 0.0364075 | 0.0523018 | 0.0542598 | 0.071908 | 4.09866 | 0.265823 | 14.16 | 21106.3 |
| window-medium | float | 0.0346385 | 0.0365987 | 0.0453944 | 0.078176 | 4.77697 | 26.0853 | 15.0059 | 25106.3 |
| window-medium | double | 0.064815 | 0.0706872 | 0.0860186 | 0.095259 | 4.4077 | 30.6107 | 18.9717 | 14024.3 |
| window-large | float | 0.064097 | 0.0658638 | 0.0676585 | 0.073203 | 4.54432 | 40.5487 | 20.5561 | 14405.6 |
| window-large | double | 0.157648 | 0.160619 | 0.169135 | 0.173954 | 4.93418 | 35.0328 | 23.2076 | 6116.3 |
| shift-small | float | 0.0334575 | 0.0348585 | 0.0376765 | 0.060019 | 4.12727 | 0.27263 | 0.091723 | 25812.9 |
| shift-small | double | 0.0347315 | 0.0512336 | 0.0596833 | 0.072444 | 3.99266 | 0.271778 | 0.106221 | 22180.8 |
| shift-medium | float | 0.032894 | 0.0339905 | 0.0359336 | 0.075462 | 4.12972 | 26.8921 | 0.092398 | 25914.5 |
| shift-medium | double | 0.0641295 | 0.067781 | 0.0813034 | 0.11423 | 4.16038 | 30.0184 | 0.104861 | 14203.4 |
| shift-large | float | 0.063875 | 0.0711926 | 0.0807269 | 0.085065 | 4.39236 | 39.2839 | 0.108518 | 14248.6 |
| shift-large | double | 0.148342 | 0.149482 | 0.164087 | 0.167009 | 5.04415 | 36.6833 | 0.235769 | 6490.14 |
| waveform-small | float | 0.034227 | 0.0358217 | 0.0379994 | 0.05163 | 4.28914 | 0.264463 | 0.094824 | 25553.3 |
| waveform-small | double | 0.036088 | 0.0525329 | 0.0553517 | 0.073473 | 4.20492 | 0.277861 | 0.091083 | 21705.3 |
| waveform-medium | float | 0.034338 | 0.035318 | 0.0442777 | 0.072297 | 4.10086 | 26.9467 | 0.098278 | 25315.9 |
| waveform-medium | double | 0.0648565 | 0.066434 | 0.0774291 | 0.095444 | 4.18309 | 30.3447 | 0.125213 | 14176.6 |
| waveform-large | float | 0.064398 | 0.0655958 | 0.0767143 | 0.083148 | 4.45848 | 39.2912 | 0.141203 | 14399.7 |
| waveform-large | double | 0.149185 | 0.1506 | 0.159918 | 0.162954 | 4.9236 | 34.8028 | 0.205722 | 6468.81 |
| norm-small | float | 0.0336385 | 0.0344914 | 0.0393765 | 0.081898 | 4.01726 | 0.27298 | 0.11712 | 25819.7 |
| norm-small | double | 0.035467 | 0.0519894 | 0.055186 | 0.064037 | 4.07508 | 0.273667 | 0.112537 | 22514.2 |
| norm-medium | float | 0.033296 | 0.0341319 | 0.034737 | 0.051981 | 4.0422 | 26.1241 | 0.116547 | 26196.6 |
| norm-medium | double | 0.0641475 | 0.071102 | 0.0786774 | 0.089778 | 4.05339 | 30.0581 | 0.11949 | 14132.6 |
| norm-large | float | 0.0638195 | 0.0663926 | 0.0672358 | 0.082129 | 4.2164 | 39.243 | 0.125954 | 14299.2 |
| norm-large | double | 0.148514 | 0.150369 | 0.164388 | 0.169554 | 4.78782 | 36.2231 | 0.218056 | 6475.41 |
| nfft-small | float | 0.0328475 | 0.0343057 | 0.0584008 | 0.066056 | 4.07956 | 0.261824 | 17.5383 | 26015.7 |
| nfft-small | double | 0.036157 | 0.0517464 | 0.054718 | 0.067037 | 4.23977 | 0.272897 | 20.5633 | 22215.6 |
| nfft-medium | float | 0.0331575 | 0.0345056 | 0.043682 | 0.053463 | 4.21851 | 26.9527 | 20.4533 | 26091.7 |
| nfft-medium | double | 0.064379 | 0.0669205 | 0.0815362 | 0.089518 | 4.35324 | 29.701 | 23.6719 | 14230.2 |
| nfft-large | float | 0.0641575 | 0.0688234 | 0.0767913 | 0.079815 | 4.39654 | 39.8614 | 29.256 | 14230.8 |
| nfft-large | double | 0.148357 | 0.151482 | 0.163709 | 0.19513 | 5.03535 | 34.9208 | 23.6639 | 6490.34 |
| ortho-hann-medium | float | 0.04488 | 0.0464309 | 0.0526506 | 0.08287 | 3.88261 | 17.8283 | — | 20033.7 |
| ortho-hann-medium | double | 0.074791 | 0.0848382 | 0.0889979 | 0.104223 | 3.95825 | 21.5074 | — | 12336.6 |
| hamming-window-medium | float | 0.047366 | 0.0685684 | 0.090362 | 0.102092 | 4.17434 | 17.7999 | 0.190454 | 17652.6 |
| hamming-window-medium | double | 0.0771385 | 0.082335 | 0.101408 | 0.114296 | 4.1427 | 22.002 | 0.225528 | 11970 |
| nfft-cached-medium | float | 0.033421 | 0.0347926 | 0.0362817 | 0.059481 | 4.01091 | 23.1999 | 0.071259 | 25945.7 |
| nfft-cached-medium | double | 0.064227 | 0.066241 | 0.078545 | 0.086435 | 4.08388 | 26.5473 | 0.12487 | 14274.4 |
| window-small-cached | float | 0.033954 | 0.0353187 | 0.0361684 | 0.045158 | 4.10327 | 13.4282 | 0.091759 | 25797.3 |
| window-small-cached | double | 0.03531 | 0.0523668 | 0.0567854 | 0.074741 | 4.25165 | 14.6649 | 0.09861 | 22836.1 |
| window-medium-cached | float | 0.035476 | 0.0372512 | 0.0385999 | 0.056222 | 4.18377 | 18.0978 | 0.104017 | 24408 |
| window-medium-cached | double | 0.064963 | 0.0661131 | 0.0716483 | 0.07913 | 4.39225 | 21.8477 | 0.122972 | 14263.6 |
| window-large-cached | float | 0.064819 | 0.0703555 | 0.0808556 | 0.099056 | 4.45999 | 22.9202 | 0.109611 | 14083.4 |
| window-large-cached | double | 0.149764 | 0.15175 | 0.160924 | 0.166305 | 5.11656 | 26.5215 | 0.201871 | 6437.02 |
| shift-small-cached | float | 0.032801 | 0.033991 | 0.0463793 | 0.066389 | 4.10087 | 0.28262 | 0.039796 | 26234.5 |
| shift-small-cached | double | 0.0355975 | 0.0586374 | 0.0700908 | 0.081861 | 4.11991 | 0.294759 | 0.044259 | 19949.8 |
| shift-medium-cached | float | 0.0337275 | 0.0348014 | 0.0426268 | 0.069574 | 3.97973 | 26.3797 | 0.046176 | 25627.4 |
| shift-medium-cached | double | 0.064037 | 0.0658821 | 0.0707027 | 0.08013 | 4.1924 | 30.7027 | 0.073491 | 14319.7 |
| shift-large-cached | float | 0.0636065 | 0.0650844 | 0.0701067 | 0.077972 | 4.28988 | 40.1319 | 0.074054 | 14576.7 |
| shift-large-cached | double | 0.148417 | 0.150352 | 0.160684 | 0.16187 | 5.45017 | 36.2222 | 0.159815 | 6489.55 |
| waveform-small-cached | float | 0.0343145 | 0.0359744 | 0.0418241 | 0.075102 | 4.21443 | 0.31675 | 0.05225 | 25003 |
| waveform-small-cached | double | 0.037829 | 0.0537226 | 0.0633123 | 0.068268 | 4.20918 | 0.312056 | 0.073907 | 20640 |
| waveform-medium-cached | float | 0.034315 | 0.0355014 | 0.0362861 | 0.062473 | 4.1854 | 26.6975 | 0.064796 | 25318.5 |
| waveform-medium-cached | double | 0.064893 | 0.0662179 | 0.0732246 | 0.084695 | 4.11756 | 30.3385 | 0.096741 | 14175.3 |
| waveform-large-cached | float | 0.0654215 | 0.0715222 | 0.0858698 | 0.090666 | 4.29279 | 39.4934 | 0.096805 | 13884.2 |
| waveform-large-cached | double | 0.149634 | 0.151083 | 0.164265 | 0.176009 | 5.14352 | 34.7636 | 0.18 | 6460.28 |
| norm-small-cached | float | 0.03382 | 0.0347558 | 0.0353722 | 0.068694 | 3.9783 | 0.286175 | 0.047611 | 25817.7 |
| norm-small-cached | double | 0.0339215 | 0.0504477 | 0.0520086 | 0.066787 | 4.09085 | 0.287648 | 0.044565 | 23135.7 |
| norm-medium-cached | float | 0.0325885 | 0.0332927 | 0.0364542 | 0.04962 | 4.06777 | 26.6685 | 0.048139 | 26741.9 |
| norm-medium-cached | double | 0.0649305 | 0.0693161 | 0.0758659 | 0.0915 | 3.99857 | 29.8017 | 0.075194 | 14128.6 |
| norm-large-cached | float | 0.0638425 | 0.0688692 | 0.0849467 | 0.088083 | 4.14955 | 40.3658 | 0.07813 | 14108.4 |
| norm-large-cached | double | 0.148574 | 0.149753 | 0.162188 | 0.170824 | 4.6346 | 36.0451 | 0.163342 | 6474.72 |
| nfft-small-cached | float | 0.0335785 | 0.0432328 | 0.0605539 | 0.081426 | 4.18679 | 17.3465 | 0.068481 | 24370.6 |
| nfft-small-cached | double | 0.034644 | 0.0510803 | 0.0557939 | 0.088852 | 4.1585 | 20.5544 | 0.087667 | 22329.5 |
| nfft-medium-cached | float | 0.033324 | 0.0356948 | 0.0503722 | 0.077212 | 4.14073 | 23.3462 | 0.069575 | 25755.5 |
| nfft-medium-cached | double | 0.0644995 | 0.0667187 | 0.0746937 | 0.09063 | 4.20925 | 26.6158 | 0.124407 | 14293.7 |
| nfft-large-cached | float | 0.063926 | 0.0742928 | 0.089607 | 0.18562 | 4.50975 | 32.1581 | 0.101778 | 13952.1 |
| nfft-large-cached | double | 0.148815 | 0.149731 | 0.150371 | 0.163046 | 4.98484 | 26.9557 | 0.270194 | 6516.1 |
| hamming-window-medium-cached | float | 0.0473185 | 0.0495331 | 0.0666915 | 0.091426 | 4.18335 | 18.144 | 0.07962 | 18821.4 |
| hamming-window-medium-cached | double | 0.075028 | 0.0776039 | 0.0825791 | 0.100065 | 4.26112 | 21.4081 | 0.105407 | 12429.8 |
| optional-minimal-small | float | 0.039801 | 0.0411933 | 0.0444372 | 0.050991 | 3.72412 | 0.263972 | — | 26302.2 |
| optional-minimal-small | double | 0.0425005 | 0.0438798 | 0.0554498 | 0.065056 | 3.73023 | 0.267556 | — | 21781.1 |
| optional-minimal-medium | float | 0.042431 | 0.0487566 | 0.0601605 | 0.06738 | 3.72293 | 26.8397 | — | 21240.5 |
| optional-minimal-medium | double | 0.0584725 | 0.059893 | 0.0722005 | 0.081121 | 3.66438 | 29.8475 | — | 16144.5 |
| optional-minimal-large | float | 0.0594345 | 0.0605164 | 0.0623795 | 0.072296 | 3.96299 | 40.6892 | — | 15982.5 |
| optional-minimal-large | double | 0.141865 | 0.143255 | 0.15133 | 0.165686 | 4.31317 | 34.9851 | — | 6892.05 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/fmcw_doppler_processor/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/fmcw_doppler_processor/repeat-review.json).

## fmcw_range_doppler_processor

Editable scope: `operators/fmcw/fmcw_range_doppler_processor_op`. Output ports (both precisions exact): `range_bins`, `rd_maps`, `velocity_bins`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/fmcw_range_doppler_processor.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 8; insufficient pair budget False; worst P90 paired delta 29.97%, worst baseline IQR 23.44%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_range_doppler_processor/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.049861 | 0.0511536 | 0.0623767 | 0.071306 | 4.83958 | 0.272898 | — | 17981.8 |
| baseline-small | double | 0.0664305 | 0.0734828 | 0.0867392 | 0.090278 | 4.841 | 0.297139 | — | 15039.1 |
| baseline-medium | float | 0.0508705 | 0.0524008 | 0.0594279 | 0.091685 | 4.875 | 47.0187 | — | 17774 |
| baseline-medium | double | 0.0993335 | 0.100268 | 0.101508 | 0.109527 | 4.96325 | 56.2506 | — | 9558.21 |
| baseline-large | float | 0.0827865 | 0.0924099 | 0.096304 | 0.106056 | 5.08277 | 79.8105 | — | 11168.4 |
| baseline-large | double | 0.275501 | 0.278574 | 0.29961 | 0.304351 | 5.5946 | 86.0137 | — | 3540.23 |
| mean-medium | float | 0.0743515 | 0.0900834 | 0.097244 | 0.113213 | 4.95011 | 39.4713 | — | 11898.2 |
| mean-medium | double | 0.15437 | 0.156455 | 0.175369 | 0.186093 | 4.85156 | 46.5655 | — | 6211.28 |
| single-delay-medium | float | 0.0575885 | 0.0594222 | 0.0864771 | 0.107602 | 4.82406 | 50.6063 | — | 15570.7 |
| single-delay-medium | double | 0.122287 | 0.130237 | 0.139756 | 0.24626 | 4.83298 | 58.6677 | — | 7732.31 |
| hann-medium | float | 0.0653615 | 0.0662963 | 0.0790767 | 0.093426 | 4.77608 | 40.2886 | — | 14114.7 |
| hann-medium | double | 0.111912 | 0.118585 | 0.120345 | 0.149602 | 4.98622 | 47.5643 | — | 8381.71 |
| four-channels-medium | float | 0.080685 | 0.0913252 | 0.108261 | 0.1115 | 5.17559 | 47.7791 | — | 11243.7 |
| four-channels-medium | double | 0.244388 | 0.246199 | 0.257698 | 0.891555 | 5.70846 | 54.5293 | — | 3994.69 |
| range-window-small | float | 0.052763 | 0.0540229 | 0.0557093 | 0.066686 | 4.99951 | 0.275778 | 0.256648 | 17247.9 |
| range-window-small | double | 0.0574355 | 0.0750199 | 0.0853185 | 0.104547 | 4.98335 | 0.293444 | 0.254481 | 14917.8 |
| range-window-medium | float | 0.053181 | 0.0568535 | 0.082675 | 0.118435 | 5.08604 | 47.9369 | 0.286436 | 16685.9 |
| range-window-medium | double | 0.100217 | 0.103721 | 0.117331 | 0.132639 | 5.05566 | 54.834 | 0.319721 | 9353.6 |
| range-window-large | float | 0.0758135 | 0.077272 | 0.0840169 | 0.08987 | 5.30796 | 79.85 | 0.315241 | 12311.6 |
| range-window-large | double | 0.276907 | 0.282903 | 0.298603 | 0.302981 | 6.31625 | 85.8053 | 0.525843 | 3521.76 |
| doppler-window-small | float | 0.0520505 | 0.0531066 | 0.0567621 | 0.072074 | 5.02832 | 0.272611 | 13.3893 | 17457.8 |
| doppler-window-small | double | 0.0560135 | 0.0714269 | 0.0741109 | 0.082361 | 5.1369 | 0.287806 | 14.4224 | 15195.1 |
| doppler-window-medium | float | 0.0528235 | 0.0544984 | 0.0623477 | 0.081528 | 5.20856 | 48.121 | 15.2301 | 17076.8 |
| doppler-window-medium | double | 0.101519 | 0.102683 | 0.10767 | 0.193509 | 5.19898 | 54.4939 | 19.2547 | 9333.83 |
| doppler-window-large | float | 0.076565 | 0.077482 | 0.0787412 | 0.101139 | 5.49969 | 79.9374 | 0.756064 | 12214.4 |
| doppler-window-large | double | 0.277324 | 0.282794 | 0.296847 | 1.03016 | 6.70144 | 89.6339 | 0.981547 | 3518.28 |
| dechirp-small | float | 0.0505835 | 0.0519614 | 0.0608134 | 0.077556 | 4.95615 | 0.277185 | 0.201351 | 17808.4 |
| dechirp-small | double | 0.0538845 | 0.0696909 | 0.0827726 | 0.093481 | 4.94027 | 0.289796 | 0.203176 | 15228 |
| dechirp-medium | float | 0.0513885 | 0.0530017 | 0.0645612 | 0.083213 | 4.92944 | 48.1208 | 0.222249 | 17418 |
| dechirp-medium | double | 0.099583 | 0.101773 | 0.107594 | 0.120222 | 5.03317 | 55.1121 | 0.280814 | 9502.08 |
| dechirp-large | float | 0.0748475 | 0.0757718 | 0.0801573 | 0.083333 | 5.18879 | 79.4069 | 0.263693 | 12468.8 |
| dechirp-large | double | 0.276074 | 0.284019 | 0.299622 | 0.306954 | 5.9152 | 85.6793 | 0.498536 | 3530.37 |
| clutter-small | float | 0.0501385 | 0.0548595 | 0.0769592 | 0.082213 | 4.98348 | 0.271352 | 13.4455 | 17238.9 |
| clutter-small | double | 0.0532215 | 0.0701727 | 0.0784374 | 0.086111 | 4.91477 | 0.286388 | 14.8415 | 15349.6 |
| clutter-medium | float | 0.051093 | 0.0520771 | 0.0800579 | 0.088047 | 4.85568 | 48.6386 | 16.0271 | 17565.4 |
| clutter-medium | double | 0.100399 | 0.103419 | 0.112062 | 0.123797 | 4.98711 | 54.873 | 18.5914 | 9386.56 |
| clutter-large | float | 0.074472 | 0.0800487 | 0.0937393 | 0.096843 | 5.18096 | 82.7507 | 0.885082 | 12330.4 |
| clutter-large | double | 0.275199 | 0.278769 | 0.294505 | 0.302 | 5.89346 | 85.8662 | 0.924278 | 3551.51 |
| waveform-small | float | 0.0517685 | 0.0540687 | 0.0785843 | 0.085491 | 5.0596 | 0.273435 | 0.1195 | 17000.8 |
| waveform-small | double | 0.054704 | 0.0708602 | 0.084297 | 0.093676 | 5.10179 | 0.289973 | 0.111759 | 15083.3 |
| waveform-medium | float | 0.0522175 | 0.0530113 | 0.0544237 | 0.069648 | 5.15548 | 47.2449 | 0.125833 | 17468.6 |
| waveform-medium | double | 0.101014 | 0.1031 | 0.11448 | 0.126574 | 5.01367 | 54.242 | 0.158991 | 9366.28 |
| waveform-large | float | 0.0755235 | 0.0766799 | 0.0878759 | 0.108333 | 5.30878 | 83.507 | 0.146342 | 12379.7 |
| waveform-large | double | 0.276865 | 0.282356 | 0.300736 | 0.309954 | 6.02877 | 85.6648 | 0.33886 | 3516.81 |
| norm-small | float | 0.050412 | 0.0513286 | 0.0523636 | 0.089027 | 4.87112 | 0.281046 | 0.149314 | 18003.8 |
| norm-small | double | 0.0666995 | 0.0696293 | 0.086299 | 0.096306 | 4.8487 | 0.287796 | 0.162167 | 15008.4 |
| norm-medium | float | 0.050981 | 0.0524737 | 0.0766439 | 0.082592 | 5.06926 | 47.1375 | 0.15988 | 17428.9 |
| norm-medium | double | 0.0997495 | 0.103362 | 0.121713 | 0.133806 | 5.052 | 56.5683 | 0.189796 | 9399.79 |
| norm-large | float | 0.074699 | 0.0791409 | 0.0903963 | 0.103407 | 5.2294 | 80.0252 | 0.190926 | 12296.6 |
| norm-large | double | 0.276204 | 0.278742 | 0.29413 | 0.298232 | 5.78773 | 90.4164 | 0.361731 | 3540.44 |
| shift-small | float | 0.050889 | 0.0527079 | 0.0601442 | 0.083601 | 4.97026 | 0.2755 | 0.140611 | 17775.5 |
| shift-small | double | 0.0525275 | 0.0693575 | 0.0724811 | 0.093084 | 4.7861 | 0.28399 | 0.14863 | 16270.7 |
| shift-medium | float | 0.0510605 | 0.0519353 | 0.0645071 | 0.081546 | 4.99798 | 48.8217 | 0.150953 | 17764.1 |
| shift-medium | double | 0.099833 | 0.107498 | 0.117192 | 0.126861 | 5.08119 | 56.6768 | 0.152963 | 9417.19 |
| shift-large | float | 0.074648 | 0.0759876 | 0.0810201 | 0.101398 | 5.23435 | 79.4162 | 0.157415 | 12469.3 |
| shift-large | double | 0.275643 | 0.281078 | 0.292605 | 0.296963 | 5.79235 | 85.6947 | 0.376342 | 3539.44 |
| nfft-range-small | float | 0.050958 | 0.0522436 | 0.0567788 | 0.078222 | 4.88209 | 0.275944 | 33.5478 | 17700.8 |
| nfft-range-small | double | 0.0671985 | 0.0733303 | 0.0834633 | 0.097991 | 4.93075 | 0.287694 | 27.3351 | 14471.5 |
| nfft-range-medium | float | 0.051051 | 0.0523749 | 0.0596434 | 0.06824 | 5.01809 | 47.8536 | 41.1235 | 17832.3 |
| nfft-range-medium | double | 0.0989725 | 0.100391 | 0.101355 | 0.127833 | 4.96793 | 54.2927 | 51.3838 | 9578.68 |
| nfft-range-large | float | 0.0747085 | 0.0795353 | 0.0915607 | 0.098194 | 5.34418 | 82.8976 | 44.9458 | 12384.6 |
| nfft-range-large | double | 0.275791 | 0.278991 | 0.300943 | 0.311777 | 5.82301 | 86.8123 | 53.3735 | 3538.5 |
| nfft-doppler-small | float | 0.050732 | 0.0516765 | 0.0647363 | 0.07701 | 4.89477 | 0.274139 | 17.7092 | 17727.1 |
| nfft-doppler-small | double | 0.0535085 | 0.0702556 | 0.084471 | 0.092926 | 4.95944 | 0.288499 | 21.0656 | 15405.9 |
| nfft-doppler-medium | float | 0.0512635 | 0.0524434 | 0.0611999 | 0.088926 | 5.03365 | 47.4685 | 0.686093 | 17614.5 |
| nfft-doppler-medium | double | 0.100176 | 0.103387 | 0.111898 | 0.121638 | 5.24154 | 55.0018 | 0.763815 | 9410.13 |
| nfft-doppler-large | float | 0.075384 | 0.0837179 | 0.0969092 | 0.106917 | 5.4377 | 79.703 | 29.3627 | 12214.3 |
| nfft-doppler-large | double | 0.275588 | 0.277973 | 0.290653 | 0.299657 | 5.80376 | 85.1639 | 23.477 | 3548.97 |
| dechirp-hann-medium | float | 0.198417 | 0.201231 | 0.229651 | 0.235342 | 4.85339 | 39.0799 | — | 4857.56 |
| dechirp-hann-medium | double | 0.187963 | 0.202642 | 0.214001 | 0.222658 | 4.91932 | 46.2791 | — | 5092.28 |
| delay-hamming-large | float | 0.13006 | 0.132305 | 0.157961 | 0.172 | 5.12561 | 60.5333 | — | 7292.42 |
| delay-hamming-large | double | 0.547967 | 0.559369 | 0.565493 | 0.591556 | 5.79939 | 71.4374 | — | 1801.7 |
| range-window-small-cached | float | 0.0520235 | 0.053164 | 0.0604204 | 0.067176 | 4.98911 | 0.543944 | 0.104065 | 17363 |
| range-window-small-cached | double | 0.0543005 | 0.0710049 | 0.0770296 | 0.082333 | 5.11078 | 0.508667 | 0.101935 | 15523 |
| range-window-medium-cached | float | 0.0529255 | 0.0556576 | 0.0791487 | 0.092065 | 5.06012 | 49.1352 | 0.113361 | 16752.5 |
| range-window-medium-cached | double | 0.10106 | 0.102233 | 0.104183 | 0.114685 | 5.17003 | 57.3893 | 0.155851 | 9418.25 |
| range-window-large-cached | float | 0.07644 | 0.0792792 | 0.0937282 | 0.109046 | 5.47316 | 79.3579 | 0.141287 | 12073.9 |
| range-window-large-cached | double | 0.277551 | 0.28488 | 0.302313 | 0.469129 | 6.87307 | 90.001 | 0.349963 | 3506.32 |
| doppler-window-small-cached | float | 0.051685 | 0.0531295 | 0.0611078 | 0.069213 | 5.11199 | 13.8608 | 0.110416 | 17553.8 |
| doppler-window-small-cached | double | 0.055227 | 0.072105 | 0.076297 | 0.096528 | 5.04043 | 14.593 | 0.111741 | 15119.7 |
| doppler-window-medium-cached | float | 0.0526615 | 0.0539017 | 0.0617053 | 0.108093 | 4.99047 | 39.7718 | 0.122436 | 17035.4 |
| doppler-window-medium-cached | double | 0.101142 | 0.102753 | 0.118566 | 0.126436 | 4.97195 | 46.0963 | 0.142676 | 9324.21 |
| doppler-window-large-cached | float | 0.076356 | 0.0780967 | 0.0818216 | 0.094852 | 5.42504 | 43.4826 | 0.125621 | 12257 |
| doppler-window-large-cached | double | 0.277398 | 0.279429 | 0.288202 | 0.300046 | 5.83782 | 54.8159 | 0.314352 | 3524.92 |
| dechirp-small-cached | float | 0.05106 | 0.0521753 | 0.0567876 | 0.080991 | 4.95901 | 0.454519 | 0.071056 | 17692.7 |
| dechirp-small-cached | double | 0.055935 | 0.07559 | 0.0895226 | 0.112824 | 4.79237 | 0.438973 | 0.072074 | 15110.1 |
| dechirp-medium-cached | float | 0.0511855 | 0.0531046 | 0.0811778 | 0.088167 | 4.90705 | 47.6057 | 0.086037 | 17299.4 |
| dechirp-medium-cached | double | 0.099361 | 0.100681 | 0.103004 | 0.116286 | 5.02991 | 54.7231 | 0.134112 | 9535.02 |
| dechirp-large-cached | float | 0.0754675 | 0.0765749 | 0.0799754 | 0.090371 | 5.67895 | 80.0167 | 0.108815 | 12364.2 |
| dechirp-large-cached | double | 0.27601 | 0.277392 | 0.284048 | 0.2875 | 5.71008 | 89.5875 | 0.322445 | 3548.74 |
| clutter-small-cached | float | 0.0501015 | 0.0513217 | 0.080586 | 0.130186 | 5.03205 | 13.8308 | 0.117656 | 17545.3 |
| clutter-small-cached | double | 0.059463 | 0.0693411 | 0.0742549 | 0.088786 | 4.96255 | 14.4055 | 0.116945 | 14910.4 |
| clutter-medium-cached | float | 0.0516395 | 0.0542394 | 0.081114 | 0.084527 | 4.86788 | 39.8713 | 0.122325 | 17250.9 |
| clutter-medium-cached | double | 0.099662 | 0.103033 | 0.118038 | 0.12413 | 5.14153 | 46.7376 | 0.166167 | 9458.14 |
| clutter-large-cached | float | 0.074593 | 0.0780736 | 0.0878044 | 0.098074 | 5.25221 | 44.1266 | 0.152917 | 12359.6 |
| clutter-large-cached | double | 0.283403 | 0.287995 | 0.302783 | 0.311444 | 5.58209 | 55.3752 | 0.380658 | 3447.6 |
| waveform-small-cached | float | 0.051172 | 0.0521506 | 0.0538604 | 0.060435 | 4.99233 | 0.326657 | 0.069009 | 17652.4 |
| waveform-small-cached | double | 0.0547135 | 0.0740565 | 0.0895729 | 0.097555 | 4.95563 | 0.329973 | 0.07875 | 15022.4 |
| waveform-medium-cached | float | 0.0528285 | 0.0565536 | 0.0817883 | 0.088269 | 5.01517 | 47.7537 | 0.086472 | 16535.1 |
| waveform-medium-cached | double | 0.100393 | 0.10363 | 0.108147 | 0.126862 | 5.1665 | 55.3944 | 0.131805 | 9390.17 |
| waveform-large-cached | float | 0.0755605 | 0.077102 | 0.0820804 | 0.086936 | 6.03447 | 79.8782 | 0.109528 | 12336.4 |
| waveform-large-cached | double | 0.276412 | 0.281954 | 0.294019 | 0.303713 | 5.73841 | 85.1855 | 0.313351 | 3524.34 |
| norm-small-cached | float | 0.0505645 | 0.0516654 | 0.0614451 | 0.081991 | 4.94904 | 0.33975 | 0.072806 | 17813.7 |
| norm-small-cached | double | 0.0648745 | 0.0693897 | 0.0731813 | 0.078907 | 4.9407 | 0.311148 | 0.071259 | 15096.8 |
| norm-medium-cached | float | 0.051755 | 0.0527067 | 0.0554352 | 0.066667 | 4.89997 | 48.8485 | 0.071944 | 17488.3 |
| norm-medium-cached | double | 0.100301 | 0.104489 | 0.109195 | 0.120722 | 5.03442 | 54.5314 | 0.125204 | 9364.19 |
| norm-large-cached | float | 0.0743425 | 0.0757564 | 0.0957147 | 0.106899 | 5.29683 | 83.208 | 0.097574 | 12426.9 |
| norm-large-cached | double | 0.275454 | 0.283802 | 0.296204 | 0.298722 | 5.83372 | 85.3161 | 0.303574 | 3539.69 |
| shift-small-cached | float | 0.050375 | 0.0514459 | 0.053298 | 0.063148 | 4.92607 | 0.310129 | 0.059342 | 18045.7 |
| shift-small-cached | double | 0.0541715 | 0.0708139 | 0.0773397 | 0.089601 | 4.95046 | 0.341435 | 0.068656 | 15353.1 |
| shift-medium-cached | float | 0.051283 | 0.0569138 | 0.0764137 | 0.091028 | 4.93704 | 48.0366 | 0.064343 | 16887.3 |
| shift-medium-cached | double | 0.0991385 | 0.101678 | 0.109671 | 0.115083 | 5.0293 | 54.8536 | 0.108491 | 9496.59 |
| shift-large-cached | float | 0.07494 | 0.0763867 | 0.0779794 | 0.100917 | 5.5082 | 79.7229 | 0.08625 | 12478.7 |
| shift-large-cached | double | 0.275875 | 0.282321 | 0.297027 | 0.305093 | 5.66247 | 89.9967 | 0.2885 | 3536.57 |
| nfft-range-small-cached | float | 0.051037 | 0.0529556 | 0.0816151 | 0.088889 | 5.02564 | 33.7819 | 0.090046 | 17291.5 |
| nfft-range-small-cached | double | 0.0532775 | 0.0734008 | 0.0859026 | 0.089046 | 4.94796 | 28.1982 | 0.090482 | 15444.2 |
| nfft-range-medium-cached | float | 0.0513565 | 0.0522835 | 0.0538252 | 0.054047 | 5.71584 | 68.1157 | 0.089148 | 17718.8 |
| nfft-range-medium-cached | double | 0.0995135 | 0.10369 | 0.122152 | 0.134352 | 5.12254 | 80.597 | 0.203628 | 9406.45 |
| nfft-range-large-cached | float | 0.0748145 | 0.076204 | 0.0781902 | 0.101935 | 5.4239 | 81.6894 | 0.13462 | 12440.7 |
| nfft-range-large-cached | double | 0.27607 | 0.279467 | 0.295301 | 0.305519 | 5.71425 | 87.6287 | 0.540695 | 3530.51 |
| nfft-doppler-small-cached | float | 0.0513335 | 0.0527406 | 0.0552347 | 0.084454 | 4.94909 | 17.7275 | 0.08362 | 17698.2 |
| nfft-doppler-small-cached | double | 0.053398 | 0.0697708 | 0.0739913 | 0.083945 | 4.96875 | 20.9396 | 0.088463 | 15205.3 |
| nfft-doppler-medium-cached | float | 0.050648 | 0.053965 | 0.0842339 | 0.085842 | 4.95148 | 23.968 | 0.083824 | 17220.1 |
| nfft-doppler-medium-cached | double | 0.0998745 | 0.103192 | 0.113379 | 0.134176 | 5.11832 | 28.9922 | 0.14363 | 9444.52 |
| nfft-doppler-large-cached | float | 0.0744395 | 0.0798817 | 0.0954586 | 0.099917 | 5.33995 | 72.4314 | 0.110945 | 12360.2 |
| nfft-doppler-large-cached | double | 0.275338 | 0.277888 | 0.297003 | 0.307389 | 5.68053 | 77.7416 | 0.403491 | 3547.64 |
| optional-minimal-small | float | 0.0305045 | 0.0475377 | 0.052101 | 0.054555 | 4.25712 | 0.276463 | — | 25181.7 |
| optional-minimal-small | double | 0.056186 | 0.057287 | 0.0593031 | 0.084898 | 4.23309 | 0.288204 | — | 16810.4 |
| optional-minimal-medium | float | 0.0471855 | 0.0488784 | 0.0564147 | 0.075592 | 4.22008 | 47.2132 | — | 19469.2 |
| optional-minimal-medium | double | 0.0910415 | 0.0927648 | 0.106854 | 0.108935 | 4.46373 | 54.1992 | — | 10484.4 |
| optional-minimal-large | float | 0.066153 | 0.0692276 | 0.0828738 | 0.105565 | 4.57978 | 79.8964 | — | 14141.8 |
| optional-minimal-large | double | 0.265736 | 0.267789 | 0.279044 | 0.287481 | 5.07173 | 86.1511 | — | 3704.58 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/fmcw_range_doppler_processor/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/fmcw_range_doppler_processor/repeat-review.json).

## fmcw_range_processor

Editable scope: `operators/fmcw/fmcw_range_processor_op`. Output ports (both precisions exact): `range_bins`, `range_profiles`.

Application consumer inventory: `beamscan_estimator`, `cfar_detector`, `cluster_dbscan`, `clutter_filter`, `ddm_dense_demultiplex`, `detection_reduction`, `doppler_notch`, `feature_assembly`, `fmcw_doppler_processor`, `multi_source_expansion`, `mvdr_estimator`, `phase_shift_beamformer`, `rank_one_residual_screen`, `snapshot_gather`, `tdm_sparse_compensation`, `virtual_array_assembly`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/fmcw_range_processor.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 6; insufficient pair budget False; worst P90 paired delta 63.46%, worst baseline IQR 48.46%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/fmcw_range_processor/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0269815 | 0.027621 | 0.032712 | 0.037222 | 3.9207 | 0.256546 | — | 33364.9 |
| baseline-small | double | 0.0449585 | 0.0462811 | 0.0478986 | 0.059787 | 4.07261 | 0.275388 | — | 20593.8 |
| baseline-medium | float | 0.0270975 | 0.0279486 | 0.0363934 | 0.068037 | 3.91906 | 23.136 | — | 32844.3 |
| baseline-medium | double | 0.061472 | 0.0633049 | 0.0713161 | 0.074139 | 4.1474 | 28.377 | — | 15302.3 |
| baseline-large | float | 0.027523 | 0.0315063 | 0.0582911 | 0.061222 | 4.22446 | 43.1185 | — | 30059.1 |
| baseline-large | double | 0.156301 | 0.158109 | 0.168669 | 0.178806 | 4.84219 | 54.4798 | — | 6248.61 |
| hann-medium | float | 0.0359215 | 0.0368386 | 0.0377268 | 0.053055 | 3.74963 | 24.1928 | — | 25591.6 |
| hann-medium | double | 0.072694 | 0.0780283 | 0.088718 | 0.092509 | 4.0905 | 28.5993 | — | 12953.9 |
| four-channels-medium | float | 0.0274025 | 0.0281517 | 0.0288346 | 0.043926 | 4.26216 | 23.7742 | — | 32955.8 |
| four-channels-medium | double | 0.127533 | 0.129178 | 0.136384 | 0.141593 | 4.75117 | 27.1329 | — | 7617.23 |
| padded-medium | float | 0.0384165 | 0.0392488 | 0.0407742 | 0.042121 | 3.81982 | 44.7046 | — | 24054.3 |
| padded-medium | double | 0.112574 | 0.113431 | 0.120066 | 0.129879 | 4.15614 | 54.2384 | — | 8621.2 |
| window-small | float | 0.028102 | 0.0289121 | 0.0344033 | 0.044871 | 4.34667 | 0.267473 | 0.21525 | 32004.5 |
| window-small | double | 0.0475695 | 0.0499846 | 0.0547324 | 0.067584 | 4.24871 | 0.267454 | 0.226259 | 19652.4 |
| window-medium | float | 0.0281435 | 0.0290823 | 0.0294928 | 0.079907 | 4.15486 | 23.2356 | 0.22786 | 31801.2 |
| window-medium | double | 0.0628795 | 0.0673021 | 0.0816971 | 0.105593 | 4.18562 | 27.2401 | 0.282657 | 14753.4 |
| window-large | float | 0.0285005 | 0.0296053 | 0.0548937 | 0.075972 | 4.635 | 44.5035 | 0.267435 | 30164.3 |
| window-large | double | 0.158449 | 0.159793 | 0.17365 | 0.179425 | 4.91554 | 53.8648 | 0.404546 | 6176.04 |
| dechirp-small | float | 0.026968 | 0.0278611 | 0.0344139 | 0.041769 | 4.15006 | 0.26675 | 0.175269 | 33041.2 |
| dechirp-small | double | 0.046051 | 0.0473737 | 0.0492564 | 0.053676 | 4.12118 | 0.265852 | 0.191426 | 20381.6 |
| dechirp-medium | float | 0.027079 | 0.0290849 | 0.0470598 | 0.054074 | 4.06341 | 23.6218 | 0.206944 | 31931.1 |
| dechirp-medium | double | 0.0619255 | 0.0646028 | 0.0734452 | 0.084805 | 4.19452 | 28.078 | 0.240731 | 15184.7 |
| dechirp-large | float | 0.027616 | 0.0285187 | 0.0484556 | 0.07412 | 4.28228 | 43.2789 | 0.213741 | 31733.9 |
| dechirp-large | double | 0.157732 | 0.159535 | 0.173574 | 0.206092 | 5.05989 | 53.6011 | 0.369009 | 6187.04 |
| waveform-small | float | 0.0286575 | 0.0299808 | 0.0490918 | 0.052602 | 4.14286 | 0.257388 | 0.081916 | 30862.7 |
| waveform-small | double | 0.046884 | 0.0477658 | 0.0484935 | 0.062463 | 4.32471 | 0.269083 | 0.108482 | 19972.1 |
| waveform-medium | float | 0.028366 | 0.029013 | 0.0294689 | 0.046463 | 4.26291 | 23.8129 | 0.084287 | 31649.2 |
| waveform-medium | double | 0.062653 | 0.0640749 | 0.0745645 | 0.11563 | 4.30395 | 27.7548 | 0.133842 | 14959.7 |
| waveform-large | float | 0.0284305 | 0.0304279 | 0.0536171 | 0.05599 | 4.46989 | 43.3084 | 0.084083 | 30297.3 |
| waveform-large | double | 0.157547 | 0.160534 | 0.167682 | 0.617269 | 4.95643 | 54.414 | 0.223028 | 6201.95 |
| norm-small | float | 0.027116 | 0.0278712 | 0.0283296 | 0.058342 | 4.11433 | 0.256713 | 0.107259 | 33133.6 |
| norm-small | double | 0.0455045 | 0.0474857 | 0.0546551 | 0.056509 | 4.07681 | 0.266463 | 0.119435 | 20283.3 |
| norm-medium | float | 0.0273885 | 0.028086 | 0.0285612 | 0.029222 | 4.03817 | 23.4428 | 0.106342 | 32854.5 |
| norm-medium | double | 0.061583 | 0.06592 | 0.0709152 | 0.080463 | 4.22817 | 27.4781 | 0.122889 | 15257.4 |
| norm-large | float | 0.027505 | 0.0280798 | 0.0288215 | 0.054241 | 4.30902 | 42.8953 | 0.116204 | 32780.2 |
| norm-large | double | 0.157565 | 0.1588 | 0.172208 | 0.186639 | 4.83575 | 53.7336 | 0.220602 | 6205.95 |
| nfft-small | float | 0.027 | 0.0276645 | 0.0284131 | 0.050518 | 4.2229 | 0.260343 | 32.2514 | 33236.4 |
| nfft-small | double | 0.045713 | 0.0465418 | 0.0485564 | 0.049333 | 4.16333 | 0.26774 | 27.8483 | 20516.2 |
| nfft-medium | float | 0.0270045 | 0.0275986 | 0.0284856 | 0.056704 | 4.03576 | 23.701 | 42.513 | 33381.3 |
| nfft-medium | double | 0.06213 | 0.0626994 | 0.065248 | 0.079223 | 4.28252 | 27.7267 | 51.4958 | 15313.2 |
| nfft-large | float | 0.0275735 | 0.0282304 | 0.0292346 | 0.029602 | 4.43874 | 44.6929 | 44.4784 | 32781.8 |
| nfft-large | double | 0.157328 | 0.159668 | 0.166861 | 0.173445 | 4.98803 | 53.5887 | 53.9127 | 6210.23 |
| dechirp-hann-medium | float | 0.037259 | 0.0382888 | 0.0468524 | 0.055796 | 3.8315 | 24.0119 | — | 24556.8 |
| dechirp-hann-medium | double | 0.0746065 | 0.0808097 | 0.0917473 | 0.117565 | 4.04659 | 28.7989 | — | 12657.8 |
| hamming-window-medium | float | 0.042125 | 0.043266 | 0.0663017 | 0.096685 | 3.99968 | 23.8154 | 0.189342 | 21639.5 |
| hamming-window-medium | double | 0.08031 | 0.0881157 | 0.104702 | 0.119694 | 4.32937 | 28.4535 | 0.231917 | 11589.5 |
| nfft-cached-medium | float | 0.0272915 | 0.028076 | 0.0290116 | 0.029833 | 4.29469 | 43.2074 | 0.078611 | 32965.2 |
| nfft-cached-medium | double | 0.062477 | 0.0635833 | 0.0652516 | 0.069806 | 4.25466 | 53.6776 | 0.147473 | 15271.2 |
| window-small-cached | float | 0.0285745 | 0.0290245 | 0.0369347 | 0.057916 | 4.10896 | 0.465463 | 0.075129 | 31387.2 |
| window-small-cached | double | 0.0472125 | 0.0485736 | 0.0554303 | 0.057843 | 4.25507 | 0.463065 | 0.087389 | 19577.2 |
| window-medium-cached | float | 0.028597 | 0.0298156 | 0.0550712 | 0.072398 | 4.17622 | 24.5494 | 0.095861 | 30637.8 |
| window-medium-cached | double | 0.0631205 | 0.0654535 | 0.0744998 | 0.080148 | 4.28011 | 27.5807 | 0.127352 | 14924.4 |
| window-large-cached | float | 0.0289305 | 0.0293737 | 0.0384528 | 0.056287 | 4.5718 | 44.9282 | 0.098278 | 31101.1 |
| window-large-cached | double | 0.157676 | 0.159275 | 0.165224 | 0.172602 | 4.9823 | 53.7516 | 0.233676 | 6191.18 |
| dechirp-small-cached | float | 0.027171 | 0.028177 | 0.0344613 | 0.042777 | 4.13233 | 0.407657 | 0.065269 | 32566.6 |
| dechirp-small-cached | double | 0.046287 | 0.0476519 | 0.0553198 | 0.064658 | 4.20078 | 0.406166 | 0.065306 | 20076.9 |
| dechirp-medium-cached | float | 0.0275 | 0.029562 | 0.0558916 | 0.066 | 4.21291 | 24.1195 | 0.074694 | 30764.7 |
| dechirp-medium-cached | double | 0.0616715 | 0.0631716 | 0.0671028 | 0.076065 | 4.29367 | 28.3478 | 0.104806 | 15275.9 |
| dechirp-large-cached | float | 0.0280225 | 0.0291399 | 0.0302316 | 0.079473 | 4.42341 | 43.4733 | 0.070713 | 31705.5 |
| dechirp-large-cached | double | 0.157158 | 0.158541 | 0.168367 | 0.17412 | 4.81021 | 53.739 | 0.209805 | 6218.66 |
| waveform-small-cached | float | 0.0277735 | 0.0283134 | 0.0285478 | 0.029407 | 4.28996 | 0.299028 | 0.044148 | 32586.2 |
| waveform-small-cached | double | 0.0468795 | 0.048226 | 0.0568395 | 0.062731 | 4.2672 | 0.324898 | 0.074953 | 19912.3 |
| waveform-medium-cached | float | 0.02819 | 0.0289223 | 0.0514341 | 0.096037 | 4.0721 | 23.3866 | 0.056296 | 30874.8 |
| waveform-medium-cached | double | 0.0628465 | 0.0673319 | 0.0784748 | 0.092425 | 4.36228 | 28.4397 | 0.09325 | 14894.3 |
| waveform-large-cached | float | 0.0286715 | 0.0293809 | 0.030522 | 0.046675 | 4.56981 | 44.974 | 0.058185 | 31417 |
| waveform-large-cached | double | 0.157473 | 0.158548 | 0.168978 | 0.177713 | 5.14868 | 53.6364 | 0.187444 | 6215.2 |
| norm-small-cached | float | 0.026852 | 0.0279008 | 0.0420764 | 0.053047 | 3.96749 | 0.273435 | 0.038815 | 32512.7 |
| norm-small-cached | double | 0.0457915 | 0.0469716 | 0.0577226 | 0.063482 | 4.07106 | 0.286565 | 0.056648 | 20351.8 |
| norm-medium-cached | float | 0.027324 | 0.0279993 | 0.0298654 | 0.05976 | 4.04781 | 23.3329 | 0.041092 | 32896.9 |
| norm-medium-cached | double | 0.0619865 | 0.0632263 | 0.0726074 | 0.082676 | 4.14156 | 28.2485 | 0.074861 | 15192.8 |
| norm-large-cached | float | 0.027347 | 0.0283376 | 0.0432185 | 0.054389 | 4.21663 | 43.6653 | 0.041991 | 31977.6 |
| norm-large-cached | double | 0.157361 | 0.15944 | 0.17113 | 0.177166 | 4.81583 | 56.7025 | 0.174398 | 6192.43 |
| nfft-small-cached | float | 0.027148 | 0.0282842 | 0.0298178 | 0.040138 | 4.14069 | 33.2919 | 0.0785 | 32397.7 |
| nfft-small-cached | double | 0.0464865 | 0.0474924 | 0.0505989 | 0.060722 | 4.01172 | 26.8327 | 0.076435 | 20127.9 |
| nfft-medium-cached | float | 0.0271615 | 0.0278215 | 0.0287121 | 0.029537 | 4.1604 | 43.1094 | 0.080759 | 33041.5 |
| nfft-medium-cached | double | 0.062005 | 0.0635002 | 0.0775387 | 0.231556 | 4.04856 | 56.5389 | 0.15149 | 15168.6 |
| nfft-large-cached | float | 0.0279675 | 0.0289181 | 0.0372183 | 0.059917 | 4.50261 | 47.0685 | 0.07838 | 31580.3 |
| nfft-large-cached | double | 0.156959 | 0.159914 | 0.166807 | 0.17163 | 5.00487 | 55.8373 | 0.339694 | 6224.64 |
| hamming-window-medium-cached | float | 0.041463 | 0.0462025 | 0.0810319 | 0.090194 | 4.18402 | 24.0907 | 0.076963 | 21033.5 |
| hamming-window-medium-cached | double | 0.0801015 | 0.0919496 | 0.106173 | 0.110611 | 4.32783 | 27.8066 | 0.111768 | 11639.4 |
| optional-minimal-small | float | 0.033194 | 0.0339709 | 0.0350596 | 0.04613 | 3.60545 | 0.261833 | — | 27790.6 |
| optional-minimal-small | double | 0.0400365 | 0.0436547 | 0.0493341 | 0.05913 | 3.73346 | 0.277306 | — | 22932.8 |
| optional-minimal-medium | float | 0.0329495 | 0.0346393 | 0.0490868 | 0.058583 | 3.55325 | 23.1297 | — | 26811.2 |
| optional-minimal-medium | double | 0.0574075 | 0.058343 | 0.0600673 | 0.068287 | 3.89212 | 27.2739 | — | 16422.6 |
| optional-minimal-large | float | 0.0329445 | 0.034086 | 0.0392632 | 0.056972 | 4.05235 | 42.9168 | — | 27370.1 |
| optional-minimal-large | double | 0.151208 | 0.152636 | 0.161872 | 0.167407 | 4.49937 | 53.2718 | — | 6462.08 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/fmcw_range_processor/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/fmcw_range_processor/repeat-review.json).

## mimo_deinterleave

Editable scope: `operators/fmcw/mimo_deinterleave_op`. Output ports (both precisions exact): `adc_cube_out`.

Application consumer inventory: `beamscan_estimator`, `cfar_detector`, `cluster_dbscan`, `clutter_filter`, `detection_reduction`, `feature_assembly`, `fmcw_doppler_processor`, `fmcw_range_processor`, `multi_source_expansion`, `mvdr_estimator`, `rank_one_residual_screen`, `tdm_sparse_compensation`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/mimo_deinterleave.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 8; insufficient pair budget False; worst P90 paired delta 31.75%, worst baseline IQR 22.35%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/mimo_deinterleave/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0300235 | 0.0309685 | 0.0313666 | 0.042602 | 2.94775 | 0.250361 | — | 29912.8 |
| baseline-small | double | 0.030069 | 0.0307879 | 0.0315251 | 0.046102 | 2.77658 | 0.252362 | — | 30222.2 |
| baseline-medium | float | 0.0635925 | 0.0648903 | 0.0836902 | 0.101167 | 3.98105 | 0.314046 | — | 14941.6 |
| baseline-medium | double | 0.0977965 | 0.115038 | 0.128695 | 0.140954 | 4.58449 | 0.331333 | — | 9508.96 |
| baseline-large | float | 1.18389 | 1.2648 | 1.29628 | 1.72977 | 12.3112 | 1.40106 | — | 821.966 |
| baseline-large | double | 2.46224 | 2.53513 | 2.5601 | 2.57007 | 20.199 | 2.65678 | — | 404.329 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/mimo_deinterleave/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/mimo_deinterleave/repeat-review.json).

## tdm_dense_compensation

Editable scope: `operators/fmcw/tdm_dense_compensation_op`. Output ports (both precisions exact): `rd_maps_out`.

Application consumer inventory: no downstream public operator found in application graphs. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/tdm_dense_compensation.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 4; insufficient pair budget False; worst P90 paired delta 33.4%, worst baseline IQR 22.93%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/tdm_dense_compensation/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0326715 | 0.0333087 | 0.0338864 | 0.056815 | 2.86164 | 0.265297 | — | 27995 |
| baseline-small | double | 0.0352685 | 0.0368168 | 0.0375451 | 0.05038 | 3.01696 | 0.264907 | — | 25793.9 |
| baseline-medium | float | 0.581324 | 0.645707 | 0.659205 | 0.666463 | 8.18646 | 0.782833 | — | 1676.09 |
| baseline-medium | double | 1.31175 | 1.35202 | 1.37449 | 1.40468 | 11.4647 | 1.5276 | — | 752.74 |
| baseline-large | float | 9.29193 | 9.49492 | 9.576 | 9.67579 | 64.4326 | 9.5148 | — | 107.791 |
| baseline-large | double | 20.9399 | 21.0232 | 21.1032 | 21.1955 | 122.798 | 21.0636 | — | 47.7953 |
| longer-pri-medium | float | 0.58075 | 0.627303 | 0.656408 | 0.658131 | 7.63943 | 0.790426 | — | 1677.86 |
| longer-pri-medium | double | 1.30492 | 1.35355 | 1.38377 | 1.40751 | 12.3979 | 1.50433 | — | 758.374 |
| waveform-small | float | 0.0340415 | 0.0376416 | 0.0466721 | 0.047889 | 3.04148 | 0.263213 | 0.099528 | 26467.3 |
| waveform-small | double | 0.0365465 | 0.0376311 | 0.0426663 | 0.052213 | 3.27617 | 0.265796 | 0.118398 | 24979.8 |
| waveform-medium | float | 0.57574 | 0.62641 | 0.657502 | 0.673047 | 8.67254 | 0.784361 | 0.677528 | 1697.79 |
| waveform-medium | double | 1.31248 | 1.36417 | 1.38207 | 1.39146 | 12.5543 | 1.52962 | 1.376 | 753.458 |
| waveform-large | float | 9.27297 | 9.45728 | 9.57135 | 9.63983 | 63.3201 | 9.38413 | 9.42122 | 107.822 |
| waveform-large | double | 20.887 | 21.0334 | 21.0975 | 21.1434 | 121.404 | 21.0653 | 20.9869 | 47.8832 |
| alternate-waveform-medium | float | 0.584982 | 0.631216 | 0.662106 | 0.702908 | 7.93255 | 0.785769 | 0.664528 | 1678.05 |
| alternate-waveform-medium | double | 1.31272 | 1.35982 | 1.39032 | 1.41713 | 12.6226 | 2.22562 | 1.36014 | 753.462 |
| waveform-small-cached | float | 0.0342545 | 0.0379515 | 0.0484582 | 0.05424 | 3.24558 | 0.309092 | 0.053149 | 26022.7 |
| waveform-small-cached | double | 0.0363105 | 0.0377347 | 0.043606 | 0.054371 | 3.43921 | 0.309834 | 0.054361 | 25211.9 |
| waveform-medium-cached | float | 0.579152 | 0.626268 | 0.660129 | 0.66726 | 8.62598 | 0.832788 | 0.628611 | 1679.69 |
| waveform-medium-cached | double | 1.31512 | 1.35422 | 1.37092 | 1.38549 | 12.4552 | 1.5943 | 1.34194 | 753.433 |
| waveform-large-cached | float | 9.28715 | 9.37254 | 9.47059 | 9.56703 | 63.8029 | 9.5704 | 9.32481 | 107.738 |
| waveform-large-cached | double | 20.9378 | 21.0259 | 21.0503 | 21.1112 | 122.226 | 21.1449 | 21.0063 | 47.7691 |
| alternate-waveform-medium-cached | float | 0.582296 | 0.623409 | 0.668437 | 0.694203 | 8.69183 | 0.842879 | 0.591176 | 1678.64 |
| alternate-waveform-medium-cached | double | 1.32374 | 1.36025 | 1.37625 | 1.41038 | 12.7643 | 1.57858 | 1.33413 | 749.903 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/tdm_dense_compensation/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/tdm_dense_compensation/repeat-review.json).

## tdm_sparse_compensation

Editable scope: `operators/fmcw/tdm_sparse_compensation_op`. Output ports (both precisions exact): `array_geometry`, `is_valid`, `snapshot`.

Application consumer inventory: `beamscan_estimator`, `cluster_dbscan`, `detection_reduction`, `feature_assembly`, `multi_source_expansion`, `mvdr_estimator`, `rank_one_residual_screen`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/tdm_sparse_compensation.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 4; insufficient pair budget False; worst P90 paired delta 21.31%, worst baseline IQR 16.45%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/tdm_sparse_compensation/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.064361 | 0.065797 | 0.0715941 | 0.084731 | 4.89355 | 0.324148 | — | 14765.9 |
| baseline-small | double | 0.063787 | 0.0681058 | 0.0859552 | 0.092149 | 5.69741 | 0.338676 | — | 14560.6 |
| baseline-medium | float | 0.385468 | 0.400651 | 0.438452 | 0.447945 | 8.24406 | 0.676759 | — | 2546.73 |
| baseline-medium | double | 0.672491 | 0.716971 | 0.737184 | 0.755685 | 12.5824 | 0.931268 | — | 1462.47 |
| baseline-large | float | 2.69929 | 2.81394 | 2.84831 | 2.86379 | 31.2084 | 3.02385 | — | 366.687 |
| baseline-large | double | 5.06896 | 5.17951 | 5.22497 | 5.24364 | 55.5851 | 5.48657 | — | 198.002 |
| neighborhood-medium | float | 0.384329 | 0.410227 | 0.441143 | 0.454029 | 8.52696 | 0.68264 | — | 2545.8 |
| neighborhood-medium | double | 0.669524 | 0.725242 | 0.74618 | 0.751436 | 12.9945 | 0.930112 | — | 1461.39 |
| dedup-medium | float | 0.387107 | 0.404627 | 0.449084 | 0.467361 | 9.15018 | 0.68476 | — | 2530.56 |
| dedup-medium | double | 0.676824 | 0.722274 | 0.739924 | 0.767861 | 12.7604 | 0.940648 | — | 1453.2 |
| longer-pri-medium | float | 0.38706 | 0.413667 | 0.448019 | 0.470185 | 8.4054 | 0.690055 | — | 2534.28 |
| longer-pri-medium | double | 0.67732 | 0.72883 | 0.750452 | 0.767333 | 12.1173 | 0.929695 | — | 1449.51 |
| waveform-small | float | 0.064972 | 0.0679059 | 0.0745674 | 0.097685 | 5.56316 | 0.320083 | 0.131778 | 14324.3 |
| waveform-small | double | 0.071754 | 0.0731504 | 0.0765416 | 0.090713 | 5.89382 | 0.332666 | 0.134056 | 13269.8 |
| waveform-medium | float | 0.387916 | 0.412684 | 0.448602 | 0.462083 | 9.41229 | 0.692917 | 0.426102 | 2523.96 |
| waveform-medium | double | 0.673857 | 0.721738 | 0.742943 | 0.770649 | 12.1174 | 0.952666 | 0.713612 | 1457.78 |
| waveform-large | float | 2.68581 | 2.80348 | 2.83261 | 2.85428 | 30.9148 | 3.0248 | 2.81397 | 368.442 |
| waveform-large | double | 5.04698 | 5.15446 | 5.2049 | 5.23958 | 55.9306 | 5.36208 | 5.00524 | 198.332 |
| window-small | float | 0.0655695 | 0.0678988 | 0.0810946 | 0.116787 | 4.97271 | 0.324398 | 0.078112 | 14296.3 |
| window-small | double | 0.063917 | 0.0666117 | 0.0755456 | 0.105954 | 5.45119 | 0.338093 | 0.088371 | 14648.9 |
| window-medium | float | 0.385379 | 0.409795 | 0.451677 | 0.535315 | 8.737 | 0.693093 | 0.412611 | 2542.34 |
| window-medium | double | 0.680944 | 0.73675 | 0.763669 | 0.775842 | 13.48 | 0.940555 | 0.686796 | 1437.62 |
| window-large | float | 2.6742 | 2.81132 | 2.83432 | 2.84129 | 31.7459 | 3.10408 | 2.67768 | 367.534 |
| window-large | double | 5.0456 | 5.15486 | 5.20675 | 5.23207 | 56.8112 | 5.23956 | 5.15447 | 198.564 |
| average-window-medium | float | 0.389542 | 0.40877 | 0.447026 | 0.459686 | 9.04543 | 0.698981 | 0.398027 | 2520.34 |
| average-window-medium | double | 0.764023 | 0.793891 | 0.826727 | 0.841084 | 13.5538 | 1.02854 | 0.774602 | 1289.64 |
| waveform-small-cached | float | 0.065657 | 0.0751345 | 0.0875449 | 0.21699 | 5.22671 | 0.335324 | 0.075972 | 14204.7 |
| waveform-small-cached | double | 0.064477 | 0.0691272 | 0.0785788 | 0.100537 | 5.65063 | 0.355926 | 0.076129 | 14533.4 |
| waveform-medium-cached | float | 0.388982 | 0.412229 | 0.452563 | 0.460389 | 8.58977 | 0.700074 | 0.393722 | 2527.74 |
| waveform-medium-cached | double | 0.677394 | 0.715812 | 0.741097 | 0.763194 | 12.9465 | 0.963398 | 0.682834 | 1447.86 |
| waveform-large-cached | float | 2.67625 | 2.80099 | 2.82958 | 2.9242 | 30.0287 | 3.03828 | 2.81957 | 367.753 |
| waveform-large-cached | double | 5.05231 | 5.15064 | 5.22484 | 5.25126 | 54.9623 | 5.27313 | 4.96043 | 198.137 |
| window-small-cached | float | 0.0656205 | 0.0676836 | 0.0911648 | 0.097334 | 5.03912 | 0.329777 | 0.070917 | 14443.7 |
| window-small-cached | double | 0.064375 | 0.06617 | 0.0919055 | 0.131352 | 5.2845 | 0.349527 | 0.075343 | 14381.6 |
| window-medium-cached | float | 0.38875 | 0.406773 | 0.449846 | 0.461666 | 9.05256 | 0.696436 | 0.386204 | 2514.2 |
| window-medium-cached | double | 0.678218 | 0.727672 | 0.735945 | 0.777445 | 12.77 | 0.945946 | 0.675622 | 1449.87 |
| window-large-cached | float | 2.69791 | 2.81407 | 2.83284 | 2.85092 | 31.6009 | 3.04888 | 2.76909 | 366.683 |
| window-large-cached | double | 5.0423 | 5.18789 | 5.23314 | 5.26031 | 56.784 | 5.44777 | 5.08521 | 197.965 |
| average-window-medium-cached | float | 0.392107 | 0.426563 | 0.448841 | 0.456297 | 9.72258 | 0.705158 | 0.393417 | 2486.54 |
| average-window-medium-cached | double | 0.760732 | 0.803706 | 0.828914 | 0.861037 | 12.4565 | 1.05579 | 0.770361 | 1295.1 |
| optional-minimal-small | float | 0.061213 | 0.0660915 | 0.0790194 | 0.096223 | 4.58983 | 0.320658 | — | 15184.9 |
| optional-minimal-small | double | 0.060528 | 0.0622556 | 0.0819162 | 0.093537 | 5.27308 | 0.332557 | — | 15376.9 |
| optional-minimal-medium | float | 0.385 | 0.415052 | 0.447306 | 0.457842 | 8.16137 | 0.685324 | — | 2537.32 |
| optional-minimal-medium | double | 0.670037 | 0.715672 | 0.735617 | 0.750972 | 12.2376 | 0.925778 | — | 1465.88 |
| optional-minimal-large | float | 2.66653 | 2.80473 | 2.83714 | 2.9006 | 30.2558 | 3.00133 | — | 368.789 |
| optional-minimal-large | double | 5.03051 | 5.16507 | 5.20243 | 5.24672 | 55.3469 | 5.44779 | — | 197.896 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/tdm_sparse_compensation/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/tdm_sparse_compensation/repeat-review.json).

## virtual_array_assembly

Editable scope: `operators/fmcw/virtual_array_assembly_op`. Output ports (both precisions exact): `array_geometry`, `rd_maps_out`.

Application consumer inventory: `beamscan_estimator`, `cfar_detector`, `cluster_dbscan`, `detection_reduction`, `feature_assembly`, `snapshot_gather`. This inventory supplements, rather than replaces, candidate-specific tests/workflow/shared-helper audits.

[Concrete unconfirmed config](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/review-configs-3/virtual_array_assembly.json)

Timing matrix complete: **True**. Primary timing failures: 0. Native captures complete: **True**. Exact repeat: **True**. Capture errors: 0. Profile captures: 6; profile errors: 0.

A/A baseline-profile assessment: validation_blocked; blocked metric/case rows 3; insufficient pair budget False; worst P90 paired delta 89.26%, worst baseline IQR 28.98%. [A/A raw review](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/calibration-pilot-3/virtual_array_assembly/review.json).

| Case | Precision | Median | p90 | p99 | Max | Setup | Cold | Live | FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-small | float | 0.0203935 | 0.0209089 | 0.0212678 | 0.056139 | 3.17936 | 0.255018 | — | 42973.7 |
| baseline-small | double | 0.0205465 | 0.0208865 | 0.022096 | 0.050713 | 3.69003 | 0.259176 | — | 42803.8 |
| baseline-medium | float | 0.353991 | 0.372578 | 0.411939 | 0.421685 | 8.95442 | 0.550843 | — | 2792.78 |
| baseline-medium | double | 0.587801 | 0.627388 | 0.646578 | 0.661399 | 13.1095 | 0.794779 | — | 1676.05 |
| baseline-large | float | 3.64782 | 3.77275 | 3.82636 | 3.91523 | 63.8745 | 3.97302 | — | 274.207 |
| baseline-large | double | 7.1379 | 7.2701 | 7.32122 | 7.85553 | 122.496 | 7.3187 | — | 140.042 |
| average-medium | float | 0.584973 | 0.605658 | 0.64301 | 0.659787 | 8.54278 | 0.917852 | — | 1699.05 |
| average-medium | double | 1.7725 | 1.79808 | 1.8478 | 1.86144 | 12.9698 | 2.11305 | — | 561.672 |
| keep-all-medium | float | 0.465371 | 0.495791 | 0.521973 | 0.542362 | 8.8186 | 0.68339 | — | 2107.95 |
| keep-all-medium | double | 1.00563 | 1.05644 | 1.08053 | 1.09489 | 13.0089 | 1.22656 | — | 980.46 |
| wavelength-small | float | 0.020417 | 0.0210336 | 0.0216434 | 0.021953 | 3.69552 | 0.263658 | 0.028731 | 43088.5 |
| wavelength-small | double | 0.020602 | 0.0211679 | 0.0242463 | 0.030842 | 3.79196 | 0.260593 | 0.02926 | 42560.7 |
| wavelength-medium | float | 0.343866 | 0.36667 | 0.41303 | 0.425852 | 8.50556 | 0.545972 | 0.36025 | 2843.65 |
| wavelength-medium | double | 0.587686 | 0.625142 | 0.65703 | 0.696268 | 13.0022 | 0.808287 | 0.570139 | 1683.64 |
| wavelength-large | float | 3.57724 | 3.71704 | 3.7495 | 3.77271 | 63.9268 | 3.93659 | 3.6394 | 278.182 |
| wavelength-large | double | 7.22951 | 7.38849 | 7.4678 | 7.58422 | 124.062 | 7.40928 | 7.15916 | 138.376 |
| average-large | float | 6.30747 | 6.40879 | 6.4383 | 6.49206 | 63.9267 | 6.53779 | — | 158.794 |
| average-large | double | 17.7545 | 17.8764 | 17.9133 | 18.0342 | 122.565 | 18.1816 | — | 56.3124 |
| wavelength-small-cached | float | 0.020384 | 0.0207329 | 0.021479 | 0.048351 | 3.67738 | 0.262981 | 0.024861 | 43123.9 |
| wavelength-small-cached | double | 0.0203935 | 0.0209179 | 0.0236804 | 0.043991 | 4.32229 | 0.263833 | 0.024463 | 42957.6 |
| wavelength-medium-cached | float | 0.346047 | 0.362213 | 0.404891 | 0.416547 | 9.14707 | 0.557259 | 0.347814 | 2825.17 |
| wavelength-medium-cached | double | 0.594135 | 0.620827 | 0.660322 | 0.67637 | 12.9957 | 0.828158 | 0.604861 | 1663.1 |
| wavelength-large-cached | float | 3.61327 | 3.74387 | 3.83053 | 3.86815 | 65.7456 | 3.83873 | 3.6292 | 276.554 |
| wavelength-large-cached | double | 7.18361 | 7.32527 | 7.37717 | 7.54079 | 121.836 | 7.43005 | 7.06153 | 139.501 |

[Complete timing matrix](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/assessment-2/virtual_array_assembly/preview.json); [Exact-repeat details](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/captures-4/virtual_array_assembly/repeat-review.json).

## DetectionReduction exact-repeat finding

DetectionReduction completes every native capture, but repeated output bytes differ in 13 case/precision combinations (22 port comparisons), affecting `rep_idx` and `reduced_points` in representative/medoid and weighted-mean cases. The repository already documents floating-point centroid near-tie variation. These cases remain blocked under the agreed exact policy; no tolerances or field exclusions were added. This is a baseline nondeterminism finding, not automatically a newly established contract bug. [Exact-repeat evidence](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/detection-reduction-exact-repeat-finding.json); [Decoded
differences](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/detection-reduction-repeat-difference-details.json).

## Retained output ownership coverage

The SDK exporter captures independent copies of three varying frames. This verifies values and publication metadata, but does not prove retained native output lifetime across later ticks. The installed SDK Retain mode supports only finite single-output sources and rejects these input-driven topologies. Exact ownership remains protected: candidates affecting allocation, pool reuse or emission require qualified lifetime tests, and missing coverage blocks acceptance. [Ownership coverage gap](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/retained-output-ownership-coverage.json).

Timing and output verification use separate preserved builds. The main verification source is `db6a97c445469d00af0b879fe3b9c542d3f920eaae6b51e22fb58d83388a70ec`; the SubarraySelection supplement source is `5e88c0bb55f41554533c26fda94997771c1c92ed62f061300df7c900a91cb8c9`. The corrected verification fixture waits for every expected secondary output; all production sources match the original checkout commit. Neither build source fingerprint is rewritten. [Two-build production equivalence](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/two-build-production-equivalence.json).

## Coverage and next gate

The report covers measured curated workloads on this Thor/image only. It does not certify unmeasured workload variants, pipeline throughput, process-first-use costs, unavailable Nsight Compute counters, or unstable/incomplete cases. CPU sampling/context-switch tracing were disabled without privilege changes. All unavailable/failed profile exports remain in [Profile index](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/profiles-qualified/profile-index.json).

Preparation fixed measurement-only problems (precision coverage, missing live revisits/optional wiring, masks, CFAR rank-four suppression, FeatureAssembly score/AoA, velocity fixture generation and YAML selection aliases). These are not production speed gains. Immutable original artifacts remain separate from future current-best/candidate builds. No commits, pushes, worktrees or external issues were created.

Evidence is sealed and the baseline milestone is published. The 26 versioned configurations remain unconfirmed. Under the current limits, no production candidate is eligible: calibration or correctness must be qualified first. The next review must confirm concrete policies and budgets before production edits; no limit, workload exclusion, initialization context or editable scope may be changed silently. Subsequent sessions need fresh provenance after these versioned reports are written.

## Preparation verification and evidence seal

The final scoped-trust non-performance rerun executed 1,097 tests: 1,090 passed and seven unchanged visualization/application tests failed (`IMPLICIT_DEVICE_ZERO_REJECTED`). Native performance verification passed; the earlier eight-failure rerun retained the Git-ownership failure as separate environment evidence. All 72 performance smoke tests passed on the final preparation build. Pinned all-file pre-commit checks passed with the compatible Node 22 wrapper and the same hook pins.

[Measurement seal](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/baseline-measurement-seal-3.json); [preparation checks](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/final-preparation-checks.json); [scoped profile summary](/home/deustice/Projects/holoscan-radar/build-optimization/operator-perf/campaign/scoped-profile-summary.json). The unavailable SDK/MatX source revisions, absent Nsight Compute counters, process-first-use and fallback-latency gaps, and retained-output lifetime gap remain explicit limitations. No commits, pushes or external issues were created.
