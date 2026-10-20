# Baseline scope and consistency review

Preparation on `deustice/optimization-skills`, using sealed round-one evidence. No new calibration, held-out timing, acceptance-policy change or production optimization has started. This review supersedes no historical result. Machine-readable proposal: [baseline-review.json](baseline-review.json); its budget is deliberately unconfirmed.

## Recommendation

Run a focused two-hour policy-validation pilot before committing to another bulk sweep. Test a deployment-specific additive consistency allowance of **2 µs for median, 5 µs for p90, and 10 µs for p99**, including their first-50 counterparts. These are hypotheses to validate, not measured hardware resolution or already approved limits. Keep the existing 5% relative component, IQR limits, setup/cold/live limits, output equivalence, candidate confidence gates and regression protection. Do not use this allowance to permit a slower candidate.

The proposed calibration statistic is `P90_pairs(|A1 − A2| / (0.05 × A1 + allowance)) <= 1`. Compute it from each pair before taking P90. Adding a constant to the final percentage, or subtracting a nominal noise value from measured latency, gives different results. Absolute sensitivity below uses this exact formula and reproduces the original P90 percentage decision at zero allowance.

No evidence here establishes a universal microsecond floor. The proposed numbers are deliberately bounded test hypotheses: the current record includes small absolute shifts and much larger processing-dependent variation. Applying a floor chosen from these observations requires a new policy identity, prospective selection and fresh disjoint validation. The old held-out observations have now informed exploration and cannot validate the new policy. If the proposed pilot remains unstable, report those scopes unresolved; do not increase allowances until something passes.

## Evidence behind the recommendation

There are 43 complete ten-pair calibrations and 15 complete thirty-pair held-out scopes. Each pair compares the medians of three isolated process repeats; each repeat's median/p90/p99 comes from 50 chronological measured frames. Early and full metrics coincide at 50 frames, remain explicitly recorded, and remain in the conservative family calculation. Float and double remain separate.

| Workload / phase / precision / metric | A1 µs | P90 absolute A/A difference µs | P90 relative difference % | Existing limit % |
| --- | --- | --- | --- | --- |
| beamscan_estimator / baseline-small / calibration / float / median_ms | 71.674 | 1.754 | 2.428 | 5 |
| beamscan_estimator / baseline-small / calibration / float / p99_ms | 86.953 | 15.599 | 17.065 | 5 |
| detection_reduction / baseline-large / heldout / float / median_ms | 294.938 | 2.755 | 0.934 | 5 |
| detection_reduction / baseline-large / heldout / float / p99_ms | 335.167 | 25.477 | 7.564 | 5 |
| detection_reduction / baseline-large / heldout / double / p99_ms | 332.266 | 23.120 | 7.016 | 5 |
| music_estimator / baseline-small / calibration / float / median_ms | 3515.530 | 598.732 | 18.862 | 5 |
| music_estimator / baseline-small / calibration / double / median_ms | 4358.581 | 404.688 | 9.836 | 5 |
| esprit_estimator / baseline-small / calibration / double / median_ms | 4155.533 | 401.977 | 10.606 | 5 |

Absolute differences are P90 of the actual per-pair absolute differences, not a percentage multiplied by the table's median. MUSIC/ESPRIT median instability is hundreds of microseconds; the proposed allowance will not resolve it. DetectionReduction has stable medians and unstable tails; its failed held-out result remains failed. Beamscan also has cold-time failures that a steady allowance cannot resolve.

Across the calibration data's 258 full-window steady metric/precision rows, 77 fail the relative A/A delta limit and 14 fail IQR. A hypothetical uniform 10 µs allowance would reduce delta-only failures to 44, but would not clear IQR, cold/setup, directional bias, confidence resolution or full-case admission. Across the selectively eligible held-out set's 90 rows, two delta failures become zero under that hypothetical allowance. This sensitivity is not an unbiased pass-rate estimate or permission to relabel any scope. Early duplicates are omitted only from this descriptive count, never from gates.

## Per-frame distributions

The offline analysis reconstructs every complete repeat from hash-verified children, retaining all 50 frames. It compares paired repeat ECDFs and the mean absolute distance between sorted frame times (one-dimensional Wasserstein distance), then reports median repeat and pair summaries. These describe distribution location, spread and tails without assuming frames are independent. Existing intervals and acceptance inference continue to use isolated pairs.

For DetectionReduction large float, median paired-repeat distribution distance is 2.938 µs; for MUSIC small float it is 222.295 µs. Maximum ECDF differences alone are misleading: stable ClutterFilter stress float has a median paired-repeat ECDF distance of 0.40, despite passing held-out gates. A small shift of a narrow distribution can move much of its ECDF. Do not add a pooled-frame KS p-value or ECDF-distance gate: temporal dependence and narrow distributions would make it answer a different question.

Keep raw distributions, within-run IQR, between-run variation, frame maxima, stalls, chronological drift and pairwise quantile differences in reports. With 50 frames, p99 depends heavily on the largest observations; additional pairs do not turn each repeat's p99 into a well-sampled population tail. Changing frame count changes the estimand and costs; any later sampling experiment must be frozen and validated separately. Do not discard stalls or increase warmup to make tails pass.

## Concrete pilot scope

Both precisions, 50 measured frames, three isolated repeats, five warmup frames, existing two-frame process warmup, adjacent alternating A/A pairing, median aggregation, first 50 and full median/p90/p99 plus setup/cold. Ten fresh calibration pairs, then freeze selection and collect disjoint held-out pairs with a 30-pair floor and recalculated family budget. Retain the prior conservative 156 scope-family multiplier for this pilot; it bounds the unchanged 93 static scopes. Expanded full matrices need their own budget calculation.

| Operator | Case | Purpose | Forecast for 10 + 30 pairs, minutes |
| --- | --- | --- | --- |
| fmcw_range_processor | baseline-small | Short FFT | 11.1 |
| beamscan_estimator | baseline-small | Short beamforming and cold cost | 10.4 |
| detection_reduction | baseline-large | Stable center / variable tail | 9.5 |
| music_estimator | baseline-small | Variable eigensolver | 12.0 |
| esprit_estimator | baseline-small | Second eigensolver check | 12.1 |
| clutter_filter | scenario-stress | Qualified large reduction control | 43.9 |

Total observed-cost projection is 99 minutes; a two-hour qualification cap gives roughly 21 minutes for checks and overhead. Reject failed calibrations before spending held-out time; all six may not qualify. Stop at the cap with unresolved scopes visible. No frame/repeat escalation, repeated retry-to-pass, extra warmup or scope expansion within this pilot. No production files are editable. All affected consumer acceptance is deferred; static pilot evidence cannot authorize a shared-source optimization.

The additive statistic is a proposed tool extension, not available in the current gate implementation. Before this pilot, implement it with host tests only after policy confirmation, bind it into measurement identity, freeze limits and prospective selection, and verify zero allowance reproduces existing behavior. Existing absolute sensitivity calculations are offline diagnostics only.

## Baseline collection scope and forecasts

Keep the original 93 static scopes as the first assessment layer. Fourteen qualify under the current policy, 26 failed calibration, one failed held-out validation, 45 lack full calibration, two await held-out validation, two RootMusic screens exceeded cost limits, and three added DopplerNotch scopes require new harness evidence. The harness repair and one new permanent SIMO recipe case are separate from these historical counts.

| Work package | Measured-cost projection | Planning allowance / limitation |
| --- | --- | --- |
| Two already-calibrated scopes: MUSIC large and TDMDenseCompensation medium | 23.3 minutes | Fresh context/reuse checks required |
| Calibration of 45 unfinished scopes | 2.70 hours | Ten pairs each, using observed screen costs |
| Conditional held-out for those 45 plus the two ready scopes | 8.50 hours | Thirty pairs each; assumes every new calibration passes |
| Combined unfinished measured static queue | 11.21 hours | Recommend 14–16-hour cap; fewer eligible cases cost less |
| Fresh calibration + held-out for all 88 costed static scopes | 21.93 hours | Roughly 27–30 hours with overhead; excludes five unknown-cost scopes |
| Full prior 760-case canonical matrix | About 165 hours | Crude static-cost extrapolation only, not a schedulable forecast |

The 165-hour illustration sums 40 pairs per canonical case times each operator's observed static median job cost. It excludes native correctness, profiling, new scenarios and all extra live/calibration work; static costs may badly underestimate expensive live variants and RootMusic larger cases. The canonical matrix is now 761 cases after adding SIMO; including 15 scenario cases gives 776. The prior round's 775-case accounting remains unchanged. Before authorizing the full matrix, inventory each case, resolve sampling and metrics, recalculate multiplicity, and forecast setup/live costs separately. Missing CFAR rank-four suppression and FeatureAssembly score/AoA recipes remain coverage work, not implicitly covered.

Round-one timing artifacts total about 0.206 GiB for 1,056 complete A/A pairs plus partial units. At the same report format, a fresh 40-pair run of the 88 costed scopes is roughly 1 GiB of timing records; reserve several GiB for logs/receipts. Native output snapshots are separate and can dominate storage by tens of GiB for large complex cubes. No full-matrix native-storage estimate is available yet.

## Per-operator remaining static queue

Costs below cover missing calibration and conditional held-out only, under unchanged 50/3. Blank/zero costs do not mean qualified; already-failed scopes need diagnosis, not blind reruns. Full variant/live matrices remain protected prerequisites for future optimization.

| Operator | Static scopes | Qualified | Missing calibration | Waiting held-out | Calibration minutes | Conditional held-out minutes |
| --- | --- | --- | --- | --- | --- | --- |
| beamscan_estimator | 3 | 1 | 0 | 0 | 0.0 | 0.0 |
| cfar_detector | 3 | 0 | 1 | 0 | 2.7 | 8.1 |
| cluster_dbscan | 3 | 2 | 0 | 0 | 0.0 | 0.0 |
| clutter_filter | 6 | 1 | 3 | 0 | 8.4 | 25.2 |
| ddm_dense_demultiplex | 3 | 1 | 2 | 0 | 5.4 | 16.1 |
| ddm_power_fold | 3 | 0 | 2 | 0 | 5.4 | 16.1 |
| ddm_sparse_demultiplex | 3 | 0 | 3 | 0 | 8.9 | 26.8 |
| detection_reduction | 3 | 0 | 0 | 0 | 0.0 | 0.0 |
| doppler_notch | 6 | 1 | 2 | 0 | 5.3 | 15.9 |
| esprit_estimator | 3 | 0 | 0 | 0 | 0.0 | 0.0 |
| feature_assembly | 3 | 0 | 3 | 0 | 7.9 | 23.7 |
| fmcw_doppler_processor | 6 | 1 | 5 | 0 | 15.1 | 45.4 |
| fmcw_range_doppler_processor | 6 | 0 | 5 | 0 | 31.8 | 95.5 |
| fmcw_range_processor | 6 | 2 | 4 | 0 | 11.3 | 34.0 |
| mimo_deinterleave | 3 | 0 | 3 | 0 | 8.8 | 26.4 |
| multi_source_expansion | 3 | 0 | 2 | 0 | 5.7 | 17.0 |
| music_estimator | 3 | 0 | 0 | 1 | 0.0 | 14.9 |
| mvdr_estimator | 3 | 2 | 0 | 0 | 0.0 | 0.0 |
| phase_shift_beamformer | 3 | 0 | 0 | 0 | 0.0 | 0.0 |
| rank_one_residual_screen | 3 | 2 | 0 | 0 | 0.0 | 0.0 |
| root_music_estimator | 3 | 0 | 0 | 0 | 0.0 | 0.0 |
| snapshot_gather | 3 | 0 | 2 | 0 | 7.2 | 21.6 |
| subarray_selection | 3 | 0 | 2 | 0 | 5.3 | 15.9 |
| tdm_dense_compensation | 3 | 0 | 2 | 1 | 12.8 | 46.8 |
| tdm_sparse_compensation | 3 | 1 | 2 | 0 | 8.5 | 25.6 |
| virtual_array_assembly | 3 | 0 | 2 | 0 | 11.7 | 35.2 |

## Harness repair and evidence reuse

DopplerNotch's adapter omitted `max_rx` when `tx=1, rx>1`, despite generating a named RX axis. The repair supplies that existing constructor bound and labels those rows SIMO. It preserves the existing 4×4 MIMO fixture restriction and production error behavior. A permanent `simo-medium` Limited case prevents the recipe from silently losing this layout. No production operator or API changed.

Separate harness executables were compiled using the original CMake-generated compiler/link flags and unchanged production archives. Commands, original artifact hash checks, source-closure differences and runtime resolution are archived in `scope-review-2026-10-14/`. They are functional builds, not qualified timing artifacts.

Validation passed: both-precision recipe checks; two exact native captures of eight canonical/SIMO/MIMO/live cases and the three added scenarios (22 case/precision combinations, 132 observed frames); and all eight existing DopplerNotch C++ tests. Correctness results are recorded in `correctness-completion.json`; historical incomplete outputs are untouched. These checks do not qualify timing or close the borrowed-output retention review gap.

New DopplerNotch measurement identity and calibration are required before timing reuse or baseline admission. Changing one recipe does not erase other operators' old results; reuse them only through verified closure/runtime/context receipts.

Host-tool verification: all 83 tests passed after the GPU checks released their device lock. An earlier concurrent host-test attempt hit that lock in seven tests; its failed log is preserved. Pinned checks cover the changed harness and reports.

## Decision and next actions

Confirm the proposed additive policy for a bounded pilot and its explicit two-hour qualification budget. The skill requires a positive user-specified budget before timing; no new cap is assumed here. Keep current acceptance rules and historical dispositions. If the policy pilot validates useful scopes, freeze their exact identities and collect missing static evidence within a separately confirmed cap. If it does not, retain the blocked scopes and report measurement limits.

No full-baseline rerun is needed merely because this report exists. Changed policies require fresh qualification of affected scopes; unchanged scopes can retain historical reporting and may be reused only after receipt checks. No current static result clears the full case/consumer matrix for production optimization. RootMusic algebraic-rank interpretation and borrowed-output retention remain open review gaps. No new suspected production correctness defect was found by this offline review; the observed DopplerNotch issue is in the harness.

Raw analysis, all 928 metric rows, pair distribution summaries, input hashes, scripts and validation receipts: `build-optimization/operator-perf/assessment-2026-10-13/scope-review-2026-10-14/`. Original immutable session: `build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/`. No commits, pushes or external publication.
