# Finite-horizon distribution qualification proposal

Prepared 2026-10-16 on `deustice/optimization-skills`. **Not confirmed or
executed.** This proposal uses existing captures; no GPU measurements started.
Recommended total cap: **four hours**, including preparation and publication.

## Decision this experiment should enable

Determine whether two larger existing FFT/reduction workloads can reproduce
their early and complete finite-horizon timing distributions while satisfying
all existing correctness and calibration protections. A successful result
supports baseline admission for those measured cases under current gates and
provides held-out evidence for the proposed quantile estimator. It does not
adopt a new production gate, qualify other operators or authorize optimization.

The previous two-hour allocation is closed. The statistical protection rule is
already confirmed: demonstrated slowdown blocks; ordinary inconclusive
protected results are allowed. Configuration and a new positive timing budget
still need confirmation before execution.

## Why this scope

Retain all four processing families in the campaign. Do not rerun protocols
that already fail unchanged calibration merely to collect more frames.

| Existing case | Evidence | Consequence for this proposal |
| --- | --- | --- |
| RangeProcessor `baseline-small`, 2,000 frames | 30-pair replication: first50 p99 P90 A/A delta 31.03% float / 18.51% double against 5%. | Remains blocked; no repeat scheduled. |
| DetectionReduction `baseline-large`, 2,000 frames | 30-pair replication: first50 p99 P90 A/A delta 7.00% float / 5.70% double against 5%. | Remains blocked; no repeat scheduled. |
| MUSIC `baseline-small`, 500 frames | 30-pair replication: first50 p99 P90 A/A delta 16.47% float / 11.70% double against 5%. | Remains blocked; no repeat scheduled. |
| ClutterFilter `baseline-medium`, 2,000 frames | Only two screen pairs; full-window float p99 P90 delta 5.53% against 5%, with other dispersion failures. | Unresolved screen with adverse evidence; not prioritized for repetition. |
| RangeProcessor `scenario-larger`, 500 frames | Two screen pairs; first50 p99 P90 delta 3.16% float / 0.19% double. No measured dispersion limit exceeded in this screen. | Fresh calibration proposed; no qualification inferred. |
| ClutterFilter `scenario-stress`, 200 frames | Two screen pairs; first50 p99 P90 delta 0.94% float / 0.54% double. No measured dispersion limit exceeded in this screen. | Fresh calibration proposed; no qualification inferred. |

These selection observations are exploratory and selection-dependent. Only
fresh disjoint measurements can validate the frozen protocol. Larger data is
not presumed to eliminate variability. The failed 50-frame Clutter stress job
was an execution-cap problem, not a correctness failure; it is not reused.

## Concrete configuration

| Scope | Shape and role | Measured horizon | Calibration / held-out | Expected job time per stage |
| --- | --- | --- | --- | --- |
| RangeProcessor `scenario-larger` | 16 receive channels × 256 chirps × 1,024 complex ADC samples; FFT workload | 500 measured frames | 30 / 30 original pairs, both precisions | 16.45 minutes |
| ClutterFilter `scenario-stress` | 64 receive channels × 512 chirps × 2,048 complex range bins; mean subtraction | 200 measured frames | 30 / 30 original pairs, both precisions | 55.38 minutes |

These are synthetic workloads, with no claimed radar deployment or real-time
deadline. They are unpaced processing horizons, not a steady-state assertion.
No new recipes or production changes are proposed. Retain archived scenario
recipes and existing wiring: RangeProcessor times `range_profiles` and drains
`range_bins`; ClutterFilter times `range_profiles_out`. No live transitions are
introduced. Existing static configurations and generators remain unchanged.

For every original pair, run three isolated repeat processes per side, with
separate float/double executables. Alternate side order between original pairs.
Keep five warmup frames and the existing two-frame process warmup. Capture all
raw frames and original boundaries. Serialize measurements on the existing GPU;
do not change clocks, power, affinity, source cadence or profiling permissions.

Calibration order: RangeProcessor, then ClutterFilter. Held-out order:
ClutterFilter, then RangeProcessor, only where calibration passes. Use separate
artifact namespaces and processes for both stages. Freeze this configuration
and its identity before either stage; no selection based on favorable results.

## Statistic, protections and family size

For each original pair, side and declared window, form the equal-weight mixture
of the three repeat-run empirical frame distributions. Compare mixture p50,
p90 and p99 using median paired log quantile ratios and exact simultaneous
sign/order-statistic intervals. Positive means A2 slower. The independent unit
is the original pair; frames and repeat processes do not manufacture more
independent pairs. See the [statistical definition](../../../../skills/operator-perf/references/distributions.md).

Primary windows are **first50** and **complete declared horizon**. Setup and
cold costs stay separate. All legacy full/early median/p90/p99 metrics and
their existing gates remain authoritative alongside the exploratory estimator.
Retain maxima, stalls, cadence, pair chronology and latency distributions.

Three same-run temporal diagnostics per precision compare equal 50-frame
windows: first versus second, first versus last, and penultimate versus last.
Pool the six runs within each original pair for temporal contrasts, retaining
30 independent original units. No temporal comparison removes early frames or
defines a steady cutoff. Within-run change is reported, not automatically
treated as failure of a deliberately finite-horizon population. Unresolved
changes between pairs/stages can still undermine the inference assumptions.

Predeclare a conservative **4,784-comparison family**:

`26 operator slots × 2 workload slots × 2 precisions × 23 metrics × 2 stages`.

The 23 metrics comprise six proposed quantiles (three across two windows),
eight legacy metrics (full/early median/p90/p99 plus setup/cold), and nine
temporal quantiles (three across three contrasts). Duplicate estimands are
counted conservatively. Maxima and stall counts are descriptive, without
additional inferential decisions. No consumer or live-mode qualification is
claimed. A later full matrix or production session needs its own complete
operator/consumer family calculation.

The minimum pair count for finite exact bounds is 18 for this family. Thirty
pairs permit fourth-order bounds rather than only extrema. Neither establishes
adequate power. +/-1% interval width remains diagnostic. Historical exploration
selects the protocol; it is not pooled into prospective validation evidence.

## Qualification decisions and stops

1. Before timing, verify immutable image, runtime resolution, source/recipe
   hashes, sampling and deployment context. Preserve old manifests. Produce a
   fresh provenance/reuse receipt or rebuild if needed; never silently rebind
   changed sources. Use the existing Thor configuration. Installed SDK source
   revision remains unknown unless independently established.
2. Verify native correctness and repeat-output evidence in both precisions for
   these exact case dimensions and wiring. Reuse evidence only with reviewed
   case coverage and source/runtime identity. Missing coverage, unsupported
   exports or unaffordable capture/rebuild work block the affected case.
3. Complete 30 calibration pairs per admitted workload. Current legacy gates
   must pass: steady P90-delta/IQR 5%/10%, setup 15%/25%, cold 20%/30%, plus
   finite familywise intervals and no directional A/A bias. Preserve all
   additional correctness, configuration and measurement-context checks.
4. Proposed mixture-quantile A/A intervals must be finite and include zero at
   both windows for both precisions; exclusion in either direction indicates
   bias for this A/A test. Report this supplementary result separately from
   authoritative legacy admission. Do not invent new dispersion thresholds for
   the mixture estimator or waive failed legacy limits.
5. A calibration failure skips that workload's held-out stage and leaves it
   blocked. Continue independent admitted workloads. Complete another 30 fresh
   pairs for each surviving workload; apply the same frozen criteria. Do not
   stop a unit early because an interim result looks favorable.
6. Unbounded intervals, artifact/source mismatch, correctness failure, context
   change or unresolved pair-level timing regimes block validation. Preserve
   complete units and partial artifacts with their correct dispositions.
7. Do not start a unit unless its capped forecast plus a 25-minute publication
   reserve fits the remaining total budget. Unit caps are 21 minutes for each
   RangeProcessor stage and 70 minutes for each ClutterFilter stage. Stop at
   the total cap, clean up owned processes, and publish partial coverage.
8. No in-run horizon/frame/repeat changes, extra warmup, outlier exclusion,
   threshold relaxation, repeated rejected protocols, automatic extensions or
   production edits. Policy changes require another declared session.

For future A/B sessions, a protected interval wholly above zero demonstrates a
slowdown; an interval including zero is allowed for protection but proves no
equivalence. An objective needs its entire interval below -2%. This proposal
contains no candidate builds or A/B optimization claims.

## Runtime, storage and implementation readiness

If both cases reach held-out validation, observed two-pair costs project:

| Component | Forecast |
| --- | --- |
| Calibration jobs | 71.83 minutes |
| Conditional held-out jobs | 71.83 minutes |
| Job-time contingency, 25% | 35.92 minutes |
| Preparation, analysis and publication allowance | 45 minutes |
| Total forecast, all stages admitted | 224.58 minutes, about 3 hours 45 minutes |
| Recommended total cap | 240 minutes, four hours |

Costs include measurement-process launch/orchestration and both precisions;
they measure neither GPU busy time nor agent working time. Extrapolation from
two-pair screens is uncertain. If all calibration scopes fail, held-out work
is skipped and the allocation ends earlier without searching for replacements.

Timing artifacts project about **52 MB**, or **104 MB with 2× headroom**.
This excludes new immutable binaries/runtime archives and native tensor/input
exports, whose required storage and capture cost must be checked during
preparation. The 64-channel double tensor alone is about 1 GiB; do not confuse
small timing JSON with total campaign storage. Missing capacity or evidence
does not authorize dropping correctness coverage.

The maintained timing collector and `distributions.py` provide raw capture and
offline comparison; the supplementary estimator is not wired into production
acceptance. Before execution, materialize the frozen per-stage manifests and
analysis specifications, verify no child processes overlap historical runs,
and record confirmation/budget. This document is a reviewable policy proposal,
not a ready-to-run legacy campaign configuration or a claim of qualification.

Evidence and machine-readable proposal:
`build-optimization/operator-perf/assessment-2026-10-16/qualification-proposal/`.
`proposal.json` binds original pairs/reviews, costs and sampling. Historical
artifacts remain unchanged. No new suspected correctness bugs were found.
