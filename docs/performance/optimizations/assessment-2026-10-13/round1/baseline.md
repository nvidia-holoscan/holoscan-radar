# Round-one operator assessment

Bounded assessment on `deustice/optimization-skills`; no production optimization. User confirmed eight hours qualification and up to four hours preparation/correctness. The prior four-hour pilot budget and evidence remain separate.

This bounded round covered **26 operators / 93 static workloads**, in both precisions. **90 workloads passed native repeat checks; 88 completed two-pair screening; 43 completed calibration; 15 received held-out validation and 14 passed**, spanning ten operators. These results qualify measurement scopes, not operator-wide performance or the full variant/live matrix.

| Outcome | Workloads |
| --- | --- |
| Held-out qualified | 14 |
| Calibration variability blocked | 26 |
| Held-out variability blocked | 1 |
| Calibration budget unresolved | 45 |
| Held-out budget unresolved | 2 |
| RootMusic screening cost/time limits | 2 |
| Added DopplerNotch harness coverage blocked | 3 |

| Operator | Requested scopes | Native repeat passed | Held-out qualified | Timing dispositions |
| --- | --- | --- | --- | --- |
| [beamscan_estimator](beamscan_estimator.md) | 3 | 3 | 1 | calibration_blocked: 2; qualified: 1 |
| [cfar_detector](cfar_detector.md) | 3 | 3 | 0 | calibration_blocked: 2; calibration_budget_unresolved: 1 |
| [cluster_dbscan](cluster_dbscan.md) | 3 | 3 | 2 | calibration_blocked: 1; qualified: 2 |
| [clutter_filter](clutter_filter.md) | 6 | 6 | 1 | calibration_blocked: 2; calibration_budget_unresolved: 3; qualified: 1 |
| [ddm_dense_demultiplex](ddm_dense_demultiplex.md) | 3 | 3 | 1 | calibration_budget_unresolved: 2; qualified: 1 |
| [ddm_power_fold](ddm_power_fold.md) | 3 | 3 | 0 | calibration_blocked: 1; calibration_budget_unresolved: 2 |
| [ddm_sparse_demultiplex](ddm_sparse_demultiplex.md) | 3 | 3 | 0 | calibration_budget_unresolved: 3 |
| [detection_reduction](detection_reduction.md) | 3 | 3 | 0 | calibration_blocked: 2; validation_blocked: 1 |
| [doppler_notch](doppler_notch.md) | 6 | 3 | 1 | calibration_budget_unresolved: 2; native_validation_blocked: 3; qualified: 1 |
| [esprit_estimator](esprit_estimator.md) | 3 | 3 | 0 | calibration_blocked: 3 |
| [feature_assembly](feature_assembly.md) | 3 | 3 | 0 | calibration_budget_unresolved: 3 |
| [fmcw_doppler_processor](fmcw_doppler_processor.md) | 6 | 6 | 1 | calibration_budget_unresolved: 5; qualified: 1 |
| [fmcw_range_doppler_processor](fmcw_range_doppler_processor.md) | 6 | 6 | 0 | calibration_blocked: 1; calibration_budget_unresolved: 5 |
| [fmcw_range_processor](fmcw_range_processor.md) | 6 | 6 | 2 | calibration_budget_unresolved: 4; qualified: 2 |
| [mimo_deinterleave](mimo_deinterleave.md) | 3 | 3 | 0 | calibration_budget_unresolved: 3 |
| [multi_source_expansion](multi_source_expansion.md) | 3 | 3 | 0 | calibration_blocked: 1; calibration_budget_unresolved: 2 |
| [music_estimator](music_estimator.md) | 3 | 3 | 0 | calibration_blocked: 2; heldout_budget_unresolved: 1 |
| [mvdr_estimator](mvdr_estimator.md) | 3 | 3 | 2 | calibration_blocked: 1; qualified: 2 |
| [phase_shift_beamformer](phase_shift_beamformer.md) | 3 | 3 | 0 | calibration_blocked: 3 |
| [rank_one_residual_screen](rank_one_residual_screen.md) | 3 | 3 | 2 | calibration_blocked: 1; qualified: 2 |
| [root_music_estimator](root_music_estimator.md) | 3 | 3 | 0 | calibration_blocked: 1; screen_execution_blocked: 2 |
| [snapshot_gather](snapshot_gather.md) | 3 | 3 | 0 | calibration_blocked: 1; calibration_budget_unresolved: 2 |
| [subarray_selection](subarray_selection.md) | 3 | 3 | 0 | calibration_blocked: 1; calibration_budget_unresolved: 2 |
| [tdm_dense_compensation](tdm_dense_compensation.md) | 3 | 3 | 0 | calibration_budget_unresolved: 2; heldout_budget_unresolved: 1 |
| [tdm_sparse_compensation](tdm_sparse_compensation.md) | 3 | 3 | 1 | calibration_budget_unresolved: 2; qualified: 1 |
| [virtual_array_assembly](virtual_array_assembly.md) | 3 | 3 | 0 | calibration_blocked: 1; calibration_budget_unresolved: 2 |

## Qualified measurement scopes

A1 median latency in milliseconds, summarized across the 30 held-out pairs. P99 differences are P90 absolute paired percentage differences, float / double, against the unchanged 5% limit. No optimization gain is implied. Operator reports show all protected metrics, confidence intervals, absolute variability, workload definitions and unresolved cases.

| Operator / case | Float median ms | Double median ms | Float / double p99 A/A difference % |
| --- | --- | --- | --- |
| [beamscan_estimator / baseline-large](beamscan_estimator.md) | 2.746221 | 63.168935 | 0.515 / 0.089 |
| [cluster_dbscan / baseline-medium](cluster_dbscan.md) | 2.746138 | 5.690077 | 1.717 / 2.518 |
| [cluster_dbscan / baseline-small](cluster_dbscan.md) | 0.655846 | 1.095194 | 3.946 / 2.404 |
| [clutter_filter / scenario-stress](clutter_filter.md) | 16.790354 | 32.039511 | 1.638 / 0.761 |
| [ddm_dense_demultiplex / baseline-large](ddm_dense_demultiplex.md) | 1.180757 | 2.608246 | 1.668 / 0.610 |
| [doppler_notch / baseline-large](doppler_notch.md) | 0.439033 | 0.453167 | 1.480 / 2.318 |
| [fmcw_doppler_processor / scenario-stress](fmcw_doppler_processor.md) | 27.483005 | 93.822676 | 1.839 / 0.777 |
| [fmcw_range_processor / scenario-larger](fmcw_range_processor.md) | 0.620859 | 4.008608 | 2.068 / 1.454 |
| [fmcw_range_processor / scenario-stress](fmcw_range_processor.md) | 9.592707 | 67.437527 | 1.624 / 0.196 |
| [mvdr_estimator / baseline-large](mvdr_estimator.md) | 6.023153 | 22.167276 | 0.405 / 0.154 |
| [mvdr_estimator / baseline-medium](mvdr_estimator.md) | 0.511892 | 2.618591 | 2.941 / 0.606 |
| [rank_one_residual_screen / baseline-large](rank_one_residual_screen.md) | 3.321531 | 3.368189 | 0.619 / 0.745 |
| [rank_one_residual_screen / baseline-medium](rank_one_residual_screen.md) | 0.932100 | 0.861138 | 1.641 / 1.847 |
| [tdm_sparse_compensation / baseline-medium](tdm_sparse_compensation.md) | 0.389312 | 0.679750 | 2.579 / 1.985 |

## Protocol and limits

50 measured frames, three isolated repeats, five warmup frames, two process-warmup frames, median aggregation, chronological raw frames and alternating adjacent A/A repeat pairs. Two-pair screens are descriptive; ten-pair calibration precedes frozen selection and disjoint held-out validation with a 30-pair floor and matrix-derived proposal. No threshold changes or numeric absolute allowances. All early/full median/p90/p99, setup, cold and applicable override metrics remain protected.

Per-case checks conservatively allocate multiplicity to 26 operators × six case slots = 156 scope families; measured metric rows and both precisions multiply that bound. This covers the 93-case static round without treating each case as an independent statistical family. It does not qualify the remaining variant/live matrix or any future optimization session.

Steady P90-delta/IQR limits: 5%/10%; setup: 15%/25%; cold: 20%/30%; live transitions: 25%/40%. Confidence width and ±1% precision are diagnostics. Unsupported or missing observations remain unknown. Stable static timing does not certify unmeasured live or ownership behavior.

Scheduling: after the breadth-first screen, remaining scopes were ranked prospectively by screen blocker count, normalized limit excess and measured cost, rotating six processing families. Twelve already completed calibrations were retained. Eligible calibrations receive fresh held-out validation promptly. The scheduling amendment and source/evidence receipts are in `queue-amendment/`; the pause counts against the original eight-hour wall-clock budget. No sampling or acceptance gate changed.

## Actual durations

Initial preparation/native verification: 54.034 minutes. Functional smoke checks, host tests and publication/check closeout bring the first sealed preparation/correctness total to 57.261 minutes against 240 minutes; subsequent final-publication verification is recorded separately. Qualification elapsed: 478.269 minutes against 480 minutes. Measurement-process wall time and storage are recorded separately in the publication receipt; neither measures agent working time.

## Scope and unresolved findings

The 78 existing static cases remain independently identified. Fifteen larger synthetic cube scenarios use independently hashed overlays with compact/larger/stress labels. Overlay profile slots small/medium/large map to compact/larger/stress; they do not change the canonical recipe profiles. Inputs are not a fixed deployment or an end-to-end radar scenario. The remaining 682 existing variant/live/optional cases and six harnessless operators are outside this round.

No production candidate changes were attempted or retained; cumulative improvement is not applicable. RootMusic algebraic-rank interpretation and borrowed-output retention remain open review gaps. Missing CUDA sanitizer, Python bindings, Nsight Compute and proof of installed SDK source revision remain limitations. Nsight Systems is available but was not captured in this qualification round; no kernel attribution or optimization claim is made. Native failures below are validation findings; exact-repeat failures alone do not establish an incorrect algorithm.

DopplerNotch has a new harness coverage finding: its receive-only multichannel sizing is not encoded by the existing adapter, although YAML validation permits those fixture values. Three added scenarios are blocked; canonical cases passed. See its operator report and `doppler-notch-harness-finding.json`. No production fix is included.

| Native failure | Evidence |
| --- | --- |
| doppler_notch/scenario-compact: capture_blocked | `native/doppler_notch/scenario-compact/`, preparation logs |
| doppler_notch/scenario-larger: capture_blocked | `native/doppler_notch/scenario-larger/`, preparation logs |
| doppler_notch/scenario-stress: capture_blocked | `native/doppler_notch/scenario-stress/`, preparation logs |

Stop: actionable 50/3 queue completed or the next units were unaffordable within the confirmed caps. No automatic escalation or threshold relaxation. This is a partial assessment milestone, not a full-pass validation or a complete optimization baseline. The next round must be forecast using these measured costs and remaining coverage.

Raw session: `build-optimization/operator-perf/assessment-2026-10-13/round1-2026-10-14/`. See [scope proposal](../baseline-proposal.md) and [earlier distribution pilot](../distribution-pilot.md).
