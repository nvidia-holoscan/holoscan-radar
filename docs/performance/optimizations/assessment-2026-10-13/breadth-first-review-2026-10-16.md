# Breadth-first operator opportunity review

2026-10-16 · `deustice/optimization-skills` · assessment only. No new GPU measurements or production edits.

## Purpose and campaign correction

The campaign has two outcomes: useful, trustworthy tools for finding and validating performance improvements; and a measured assessment of avoidable work in all 26 harness-backed public operators. Measurement-method development supports those outcomes. It is not a prerequisite research project that must qualify every timing regime before any operator can be investigated.

The two-operator four-hour proposal is superseded as the next campaign-wide prerequisite. Preserve it as historical planning evidence, without executing it. The later suggestion to merge selection and validation is also not an adopted protocol change: any fresh policy or production session must still satisfy the confirmed skill requirements. No new timing budget is active.

## What is already available

The original breadth-first round covers 26 operators and 93 static workloads: 90 native repeats passed, 88 screens completed and 14 scopes across 10 operators qualified under their original configurations. These scopes are starting evidence, not current-source admission receipts or proof that whole operators are optimized. Reuse only after closure, runtime and context checks. Missing calibration does not prevent read-only source review or attribution, but still blocks affected production acceptance.

This review inventories all 26 current recipes and implementation entry points, then examines selected allocation/cache mechanisms. It is an initial opportunity inventory, not a completed source audit, new baseline, profile or validated performance disposition. Detailed investigation remains open for every operator.

## First concrete attribution priority

**ClusterDBSCAN** is the strongest structural lead inspected here. For each non-empty tick, `compute()` calls `prepare_dbscan_workspace(num_points)`. That function creates four scratch device tensors; subsequent compaction also creates workspace tensors. The convergence loop copies a changed flag to the host and synchronizes the stream. Qualified historical small/medium scopes provide a possible starting point.

Source: [scratch creation](../../../../operators/clustering/cluster_dbscan_op/cluster_dbscan_op.cu), [compute call](../../../../operators/clustering/cluster_dbscan_op/cluster_dbscan_op.cu), [convergence check](../../../../operators/clustering/cluster_dbscan_op/cluster_dbscan_op.cu). Relevant source lines are 241 (workspace), 566 (tick call) and 292 (convergence); hashes are in the ignored inventory.

The first bounded hypothesis is reusable scratch storage with unchanged kernels and formulas. Verify reset behavior, dynamic shapes, stream lifetimes and retained-output ownership before proposing an edit. The convergence synchronization may serve a real algorithm dependency; do not remove it casually or bundle it with scratch reuse. **No removable latency fraction or gain is measured yet.** Profile first, and deprioritize this lead if the allocations are not material.

Two useful negative checks prevent false leads: Beamscan allocates its beamformed scratch in the constructor, and MVDR retains covariance/inverse workspace; its `cudaDeviceSynchronize()` helper is documented teardown. Searching allocation/synchronization names alone would misrank these. ClutterFilter mean-expression allocation remains an attribution question, not a demonstrated per-frame allocation defect.

## All-operator inventory

Qualified counts below are historical workload scopes, not counts of fully validated operators. The final column states an inspected observation or a question for investigation; it never claims a retained improvement. Full original dispositions remain in the linked reports.

| Operator | Static workloads | Historical qualified scopes | Source observation / next investigation |
| --- | --- | --- | --- |
| [beamscan_estimator](round1/beamscan_estimator.md) | 3 | 1 | Scratch beamformed tensor is allocated in the constructor; inspect matrix/scan kernels rather than assume per-tick allocation. |
| [cfar_detector](round1/cfar_detector.md) | 3 | 0 | Attribute scan, suppression and output compaction separately; retain count, ordering and rank-four suppression coverage. |
| [cluster_dbscan](round1/cluster_dbscan.md) | 3 | 2 | Verified per-tick scratch creation and host convergence/count transfers; first attribution priority. |
| [clutter_filter](round1/clutter_filter.md) | 6 | 1 | MatX mean-expression temporary is a lead; allocation frequency and removable cost remain unprofiled. |
| [ddm_dense_demultiplex](round1/ddm_dense_demultiplex.md) | 3 | 1 | Linear/nonlinear tables are cached; inspect gather/correlation work and optional velocity-axis cost. |
| [ddm_power_fold](round1/ddm_power_fold.md) | 3 | 0 | Demux table is cached; inspect reduction memory traffic and repeated host table-key work. |
| [ddm_sparse_demultiplex](round1/ddm_sparse_demultiplex.md) | 3 | 0 | Inspect sparse gather/correlation traffic, scratch lifetime and optional-output work. |
| [detection_reduction](round1/detection_reduction.md) | 3 | 0 | Inspect grouping/compaction and scratch reuse; retain authorized coupled semantic validation and tie rules. |
| [doppler_notch](round1/doppler_notch.md) | 6 | 1 | Inspect copy/mask traffic and identity configurations while preserving output ownership; sizing harness repair is separate. |
| [esprit_estimator](round1/esprit_estimator.md) | 3 | 0 | Attribute covariance, solver and source-count stages; mathematical solver cost alone is not avoidable work. |
| [feature_assembly](round1/feature_assembly.md) | 3 | 0 | Inspect multi-input gathers and score/AoA wiring; rank only after coverage and timing evidence. |
| [fmcw_doppler_processor](round1/fmcw_doppler_processor.md) | 6 | 1 | Inspect window/FFT/shift/output traffic in existing qualified stress scope; preserve exact arithmetic. |
| [fmcw_range_doppler_processor](round1/fmcw_range_doppler_processor.md) | 6 | 0 | Compare internal materialization and stage traffic; fused-operator name does not prove optimal fusion. |
| [fmcw_range_processor](round1/fmcw_range_processor.md) | 6 | 2 | Inspect staging, optional axis and FFT plan reuse; keep tiny and larger regimes separate. |
| [mimo_deinterleave](round1/mimo_deinterleave.md) | 3 | 0 | Inspect gather/coalescing and copies; preserve named TX/RX layout and ownership. |
| [multi_source_expansion](round1/multi_source_expansion.md) | 3 | 0 | Inspect repeated sparse gathers and dynamic-count work; preserve emission cardinality and ordering. |
| [music_estimator](round1/music_estimator.md) | 3 | 0 | Attribute covariance/eigensolver/scan costs; early/late drift remains a measurement limitation. |
| [mvdr_estimator](round1/mvdr_estimator.md) | 3 | 2 | Covariance/inverse scratch already retained in state; teardown synchronization is not a hot-path defect. |
| [phase_shift_beamformer](round1/phase_shift_beamformer.md) | 3 | 0 | Inspect steering cache and projection traffic; no allocation or fusion benefit established. |
| [rank_one_residual_screen](round1/rank_one_residual_screen.md) | 3 | 2 | Use qualified medium/large scopes to attribute scoring and optional compaction separately. |
| [root_music_estimator](round1/root_music_estimator.md) | 3 | 0 | Separate companion solver cost from sparse-output preparation; correctness review gaps and timing caps remain visible. |
| [snapshot_gather](round1/snapshot_gather.md) | 3 | 0 | Inspect gather launches, valid-mask generation and repeated snapshot reads. |
| [subarray_selection](round1/subarray_selection.md) | 3 | 0 | Inspect selection-index preparation and gather traffic; retain descriptor and live behavior. |
| [tdm_dense_compensation](round1/tdm_dense_compensation.md) | 3 | 0 | Inspect correction application and repeated host schedule work; cached tables are not automatic wins. |
| [tdm_sparse_compensation](round1/tdm_sparse_compensation.md) | 3 | 1 | Use qualified medium scope to attribute snapshot gathering/correction and descriptor preparation. |
| [virtual_array_assembly](round1/virtual_array_assembly.md) | 3 | 0 | Index tables are cached; inspect gather and descriptor preparation while retaining causal publication. |

## Work sequence and completion criteria

1. **Complete discovery across all 26.** Reuse reports and verified captures; inspect implementations and shared helpers. Record a plausible mechanism, a reason to deprioritize it, or a coverage blocker for each operator. Do not label an operator optimal merely because no lead was found in this first pass.
2. **Attribute a rotating shortlist.** Start with the strongest recurring-work evidence, including ClusterDBSCAN. Use clean timing for absolute latency/throughput and profiles for allocation, copy, launch and synchronization costs. Heavy runtime alone does not determine priority. Continue independent scopes while others remain blocked.
3. **Use qualification where it changes a decision.** Prefer existing qualified workloads after verified reuse; collect fresh evidence where necessary. Do not require the two pilot operators to qualify the other 24, rebaseline all existing cases automatically, or repeat known failures unchanged. Full affected case/precision/consumer calibration remains necessary before a candidate can be accepted.
4. **Confirm concrete production sessions.** Present one consolidated configuration review for the proposed operator sessions: outputs, targets, protected cases/metrics, equivalence, editable files, consumers, sampling and limits. Existing directional protection and greater-than-2% objective remain unchanged. No production edits are authorized by this assessment document.
5. **Exercise the tools on real decisions.** A useful tool chain must attribute a credible mechanism, bind original/candidate artifacts, reject correctness or protected regressions, assess targets with the familywise gate, restore rejected candidates, and produce a readable disposition. Synthetic tests and A/A checks are evidence of mechanics; no accepted production optimization has demonstrated this entire loop in this campaign yet.
6. **Return to the remaining queue.** After each operator disposition, reassess opportunity and continue until every requested operator has an explicit improved, assessed-with-no-validated-improvement, or blocked outcome. Final direct original/current-best measurements substantiate retained gains; unmeasured or unstable cases remain limitations.

## Decision and next action

**Next:** finish source/consumer review of the ClusterDBSCAN scratch-reuse lead and the rest of the operator inventory; prepare a small, evidence-bound attribution/verification job list rather than another generic qualification sweep. That list must identify the information each GPU job will resolve, its measured cost forecast, and the complete existing/new evidence required.

Read-only source review and offline replay can proceed now. Fresh GPU timing/profiling needs a positive user-specified budget before execution. Production sessions need the concrete review above. No numerical relaxation, protocol migration, four-hour allocation, commit or push is implied by the goals discussion.

Keep [overview.md](overview.md) as the stable entry point. Report both dimensions separately: tool-chain validation completed versus operator coverage and directly validated gains. No new suspected correctness bugs were identified in this source triage. Existing RootMusic review gaps and timing/coverage blockers remain in prior reports.

Artifacts: `build-optimization/operator-perf/assessment-2026-10-16/breadth-first-review/inventory.json`. The inventory binds the 26 source/recipe hashes and the original baseline report.
