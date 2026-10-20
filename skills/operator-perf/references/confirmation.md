# Paired screening and integrated confirmation

Approved skill workflow, 2026-10-20. Implemented as explicit measurement-policy
version 4; see the [command guide](../../../performance/optimization/CONFIRMATION.md).
Version-2/version-3 reporting/replay remain historical. Host tests exercise
orchestration and acceptance; they are not GPU qualification or authorization
for a new allocation.

## Purpose and sequence

Use **correctness and inexpensive screening → bounded candidate → paired A/B
screen → freeze promising candidate → fresh integrated confirmation → decision**.
No standalone full-matrix A/A or B/B sweep is an admission prerequisite. A/A
and B/B are matched negative controls collected alongside the final A/B evidence.
Do not resurrect the old percentage-consistency barrier under another name.

Preserve exact output equivalence, contracts, ownership, precision, errors,
state/temporal behavior, immutable builds, source/runtime checks and verified
restoration. Prepare native fixtures and baseline correctness before candidate
edits. Obtain confirmation of editable scope, targets, protected cases/consumers,
objective, statistical design and limits. Caller budget bounds all measurement
phases; a closed allocation does not authorize another run.

## Screening and freeze

Use profiles/source attribution to rank avoidable work across the requested
operators. Screening estimates opportunity; it does not accept a candidate or
prove the absence of low-hanging fruit. Use bounded paired A/B runs for a
correct candidate. Existing within-build repeats may supply exploratory A/A or
B/B diagnostics without additional runs, but are dependent when observations
are shared.

Before confirmation, freeze candidate source/build/recipe/runtime identities,
all targets and protected endpoints, sampling, order schedule, statistical
method, multiplicity and stopping rules. Confirmation uses fresh processes and
data disjoint from screening and protocol selection. Do not pool exploratory
screens into the acceptance test or change targets after seeing confirmation.

## Confirmation block

For each confirmed case and precision, one block contains:

- Two independently launched runs of A, the preserved current-best comparator.
- Two independently launched runs of B, the frozen candidate.
- A predeclared randomized, position-balanced order across blocks, drawn from
  `AABB`, `ABAB`, `ABBA`, `BAAB`, `BABA`, `BBAA`. Record the seed and complete
  schedule before execution; do not always use one pattern.

Each run is a fresh case/precision process with identical workload and timing
boundaries. Keep five warmup frames and existing process-warmup semantics by
default. Frame counts, block counts and horizons are explicit; do not silently
translate legacy `repeats: 3` into this two-runs-per-build design. Collect setup,
cold, applicable live transitions, chronological measured frames, early/full
windows, maxima, stalls and context. Forecast four run processes per block per
case/precision, including setup/cold cost, storage and remaining budget.

Within each block and declared horizon, form equal-weight empirical mixtures
of the two A runs and the two B runs. Compare their p50/p90/p99 using the
[distribution estimand](distributions.md). Setup, cold and live transitions use
separately declared run-level summaries; do not mix them into steady frames.
The complete four-run block is the inferential unit. Independence between
blocks and the required contrast assumptions need evidence; isolation alone is
insufficient. Do not count frames, shared repeats or four cross-build pairings
as independent replicates.

A/A compares the two A run distributions; B/B compares the two B run
distributions, with chronological order recorded. The A/B estimate and both
controls share observations. Preserve that joint structure in diagnostics,
resampling and uncertainty calculations; do not combine their p-values as if
independent or subtract a control difference as an invented noise correction.

## Control interpretation and acceptance

Controls diagnose order/time effects, changing deployment conditions, carryover
or unusually variable candidate behavior. Report absolute and relative values,
uncertainty, chronology and context. High variation alone does not invalidate a
run, establish a physical noise floor or identify its cause. A non-significant
control result is not proof of equality, stationarity or adequate power.

There is **no fixed percentage or microsecond agreement gate** for A/A or B/B.
Do not turn observed control dispersion into a permissible regression margin.
Candidate variability matters through the declared A/B distribution endpoints
and any separately confirmed variability objective. It is not an automatic
failure because B/B is noisier than A/A.

Investigate suspicious controls and corroborate causal explanations. Established
provenance failures, invalid timing or unresolved confounding can leave
acceptance blocked. Retain all suspect data and report why evidence is
uninterpretable. Do not rerun or discard an unfavorable block until controls
look favorable. Further looks require predeclared stopping/error rules and
remaining authorization.

Use valid simultaneous intervals with positive effects meaning slower:

| Evidence | Decision |
| --- | --- |
| Every predeclared target has upper effect bound below −2%, or its explicitly confirmed objective | Target improvement demonstrated. |
| A protected endpoint has lower effect bound above zero | Demonstrated slowdown; reject. |
| A finite protected interval spans zero | Inconclusive; permitted under the selected directional rule, with uncertainty reported. No equivalence or bounded-harm claim. |
| Required estimate/interval is unsupported, missing or unbounded | Unresolved; do not silently count as protected success. |

Only retain a candidate passing correctness/equivalence, coverage and the full
confirmation decision. Account for all operators, consumers, unique endpoints,
candidate attempts and any interim looks in the declared error budget. Derive
block count from resolution and sensitivity, not a universal repeatability
percentage. Duplicate early/full endpoints when horizons coincide share one
statistical endpoint while retaining both report labels. A screening win alone
never authorizes acceptance.

After a keep, refresh attribution and reassess opportunities. Do not launch a
new mandatory current-best A/A campaign; future confirmation blocks measure
current-best repeatability alongside the next candidate. Final cumulative gains
require direct preserved-original versus final measurements, using the same
integrated-control design and confirmed matrix. Never multiply intermediate
percentages.

## Tool support and limits

Version 4 implements block scheduling, child/process identities, fresh
confirmation namespaces, joint control diagnostics, the mixture-quantile
estimator, conservative multiplicity and replay. `inspect-confirmation` produces
statistics; an evidence-backed agent control review records whether the data
are interpretable. `compare-timing` replays statistics and review, and cannot
accept screening alone or a favorable control review over a failed speed gate.
This is not an extra user approval. See the command guide for schemas and flags.

Tests include noisy controls, genuine protected tail regression, inconclusive
protection, scale invariance, missing precision, stale references, screening
leakage, resume, correctness-only admission and complete session acceptance.
Offline synthetic controls validate tools, not an operator gain. A deployment
exercise needs an explicit caller budget. Never bypass native coverage, replay
or source/runtime checks, and preserve historical artifacts unchanged.
