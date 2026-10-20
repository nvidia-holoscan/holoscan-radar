# Operator performance campaign — overview

Last updated: 2026-10-20. Branch: `deustice/optimization-skills`.

**Current phase: version-4 integrated confirmation implemented and host-tested.** The full baseline and
production optimization are pending. No production performance changes or
validated optimization gains have been retained in this campaign.

| At a glance | Current state |
| --- | --- |
| Requested scope | 26 public operators with existing performance harnesses; six operators without harnesses excluded by the user. |
| Original round-one coverage | 93 static workloads: 90 native correctness passes and 88 completed timing screens. See the baseline report for overlaps, blocked cases and missing coverage. |
| Original qualification | 14 qualified workloads across 10 operators under their original policies; this does not qualify the full matrices or the proposed distribution estimator. |
| Latest sampling exploration | 50/200/500/2,000-frame screens; five protocols independently replicated in both precisions. Three long-run replications support chronological analysis. |
| Current limitation | Timing changes with frame position; no universal robust frame count or validated steady cutoff. Larger controls lack comparable long-run replication. |
| Budget | Approved four-hour expansion session stopped after 15.6 minutes of preparation/collection on a proven calibration blocker; no additional GPU jobs scheduled. |
| Retained performance gains | None. Correctness fixes and harness/tooling changes are tracked separately. |
| Latest tooling verification | 119 host tests pass, including 14 new confirmation tests; pinned checks accompany the implementation; maintained replay matches all 1,044 historical contrast estimates. |
| Current priority | MultiSourceExpansion serial planning: about 99% of measured kernel work in the fully selected large case. DetectionReduction scratch churn second, ClusterDBSCAN third. All 24 diagnostic jobs completed; no candidate gains claimed. |

## Current decision and next action

The user confirmed **directional protection**: a statistically demonstrated
slowdown blocks acceptance; an inconclusive protected result is allowed.
Missing/unbounded intervals still block validation. Inconclusive does not mean
equivalent or prove a bounded regression.

The version-4 distribution comparison uses p50/p90/p99 of equal-weight repeat-run
frame distributions within each original pair. Estimate the median paired log
quantile ratio, with simultaneous intervals across the declared matrix. Keep
first50 and complete horizons separate. See the
[precise definition](../../../../skills/operator-perf/references/distributions.md).
Version 4 implements this estimator for integrated confirmation. Legacy
version-2/version-3 gates remain available for historical replay; their A/A
admission limits do not govern new version-4 work. No universal noise floor has
been adopted.

**Two session goals:** (1) demonstrate useful tools for attributing and validating
performance changes; (2) assess all 26 harness-backed operators for avoidable
work. Tool development supports concrete operator decisions. No production
performance gain has been retained, and no claim that all low-hanging fruit is
absent is justified yet.

**Next action:** start the next assessment cycle with the
[implemented version-4 workflow](integrated-confirmation-2026-10-20.md), discovery
of all 26 harness-backed operators and an explicit caller measurement budget.
The [command guide](../../../../performance/optimization/CONFIRMATION.md) supports
correctness-only admission, direct A/B screening and fresh integrated A/A–B/B
confirmation. No standalone A/A qualification is required. Complete real native
coverage before a production candidate; host tests do not establish GPU gains.
Historical timing stops remain preserved under their original policies.

## Milestones and evidence

| Milestone | Outcome | Detail |
| --- | --- | --- |
| Baseline round one | Partial correctness and timing coverage; original qualified and blocked scopes remain explicit. | [Baseline and operator reports](round1/baseline.md) |
| Scope/consistency review | DopplerNotch harness sizing repaired; fixed-microsecond allowance proposal superseded. | [Review](baseline-review.md) |
| Historical frame-count replay | 54 datasets and 1,160 comparisons; no universal robust frame count. | [Offline replay](frame-count-experiment.md) |
| Fresh sampling exploration | About 106 minutes within the two-hour cap; larger frame counts help some estimates but not uniformly. | [GPU experiment](frame-count-gpu.md), [protected metrics](frame-count-gpu-protected.md), [distribution diagnostics](frame-count-gpu-distributions.md) |
| Chronological analysis | 1,044 contrasts; RangeProcessor and DetectionReduction medians drift, float MUSIC changes substantially. No steady cutoff validated. | [Absolute timings and intervals](timing-regimes.md) |
| Skill/tool consolidation | Explicit distribution definition, maintained window-analysis command and running overview. Replay matches all 1,044 historical contrasts; no acceptance-rule replacement. | [Tool usage](../../../../performance/optimization/FRAME_COUNT.md), [statistical policy](../../../../skills/operator-perf/references/distributions.md) |
| Qualification proposal, 2026-10-16 | Historical two-case proposal, superseded as the next campaign prerequisite. No budget authorized or GPU jobs launched. | [Configuration, costs and stop conditions](qualification-proposal-2026-10-16.md) |
| Breadth-first review, 2026-10-16 | All 26 source/recipe entry points inventoried against prior coverage; ClusterDBSCAN recurring scratch creation identified as an unprofiled lead. Two session goals recorded separately. | [Inventory and work sequence](breadth-first-review-2026-10-16.md) |
| Attribution plan, 2026-10-19 | Implementation-header review finds recurring DetectionReduction allocations and serial MultiSourceExpansion planning; consumer map and 24 diagnostic jobs prepared. GPU tracing untested. | [Ranked leads, jobs and stop conditions](attribution-plan-2026-10-19.md) |
| Attribution completed, 2026-10-19 | All 12 clean runs / 12 captures succeeded. Fully selected large expansion planning dominates GPU work; scratch overhead quantified in two other operators. Unconfirmed eight-case expansion handoff prepared. | [Absolute timings, profiles and next session](attribution-results-2026-10-19.md) |
| Expansion admission preparation, 2026-10-19 | Both-precision fixture compiles with 28 tests; consumer-specific protocols collect/replay correctly; 104 host tests pass. GPU correctness, complete A/A and session confirmation remain pending. | [Session scope, forecast and admission gaps](multi-source-admission-2026-10-19.md) |
| Expansion session disposition, 2026-10-19 | Correctness passed; six complete A/A matrices prove small float p99 cannot meet the fixed 30-pair 5% dispersion limit (optimistic lower bound 5.560%). Stopped before production edits. | [Pairwise timings, rejection proof and gaps](multi-source-expansion-session-2026-10-19.md) |
| Measurement approach review, 2026-10-19 | Fixed dispersion veto identified as distinct from statistical A/B inference. Proposed workload-specific uncertainty, explicit distribution estimands, randomized paired blocks, tail-resolution checks and offline validation. Existing gates unchanged. | [Review and implementation sequence](../../../../skills/operator-perf/references/measurement-redesign.md) |
| Integrated confirmation, 2026-10-20 | Skills and version-4 tools implement correctness-only admission, A/B screening, fresh joint controls, raw replay and acceptance. 119 host tests pass; no new GPU run or production gain. | [Implementation and next cycle](integrated-confirmation-2026-10-20.md) |

## Boundaries and findings

- The deployment is NVIDIA Thor with the recorded HSDK 5 image and existing
  clock/power configuration. Conclusions are limited to measured workloads.
- Existing relative recipe sizes remain useful overhead/scaling cases. With no
  fixed deployment, larger workloads are labeled synthetic scenarios, not
  universal radar sizes. No extra deployment-matching requirement is imposed.
- Exact correctness remains required except the explicitly authorized
  DetectionReduction semantic comparison. RootMusic correctness remediation
  and DetectionReduction validation are separate from performance gains;
  [RootMusic](../../../correctness/root-music.md) retains documented review gaps.
- No new suspected correctness bugs were found in the latest admission session.
  Timing drift is an unresolved measurement finding, not an attributed SDK bug.
- The latest approved GPU admission session is sealed. Subsequent version-4
  tool/skill changes used host tests only. No production performance gain or
  external publication is claimed; historical evidence is preserved unchanged.

Artifacts: `build-optimization/operator-perf/assessment-2026-10-13/` and
`build-optimization/operator-perf/assessment-2026-10-15/`. Raw ledgers, manifests,
profiles and receipts remain there; this overview links the readable reports.
Update this page between sealed sessions at every milestone and operator
disposition. Preserve historical reports rather than silently reclassifying them.
