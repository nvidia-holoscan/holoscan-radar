# Proportionate optimization acceptance

Use this engineering workflow by default for profiling and routine operator
optimization. It replaces the universal full-matrix timing recommendation;
the version-4 [formal confirmation protocol](confirmation.md) remains opt-in.
Correctness requirements are unchanged. Performance evidence and claims are
scoped to the experiment actually run.

## Separate the decisions

| Decision | Evidence needed |
| --- | --- |
| Is there an actionable bottleneck? | Source review, a few clean timings and targeted attribution. No acceptance experiment or general qualification study. |
| Does the edit preserve behavior? | Relevant existing native/analytic tests, both precisions, exact or already approved output comparison, and candidate-specific boundary/temporal checks. |
| Is the edit useful enough to keep? | Small predeclared representative timing set, paired screening and one fresh bounded A/B check. Engineering judgment with visible variability and limitations. |
| Is a formal statistical claim required? | A separately scoped and budgeted confirmatory design. Use version 4 only when its guarantees justify its cost. |

Do not promote every recipe variant, downstream operator and diagnostic into
a repeated statistical acceptance endpoint. A full correctness inventory is
not a performance sampling matrix. Large validation cost should trigger a
design review before a larger allocation is proposed.

## Correctness follows the changed surface

Review the diff and identify the behavior it could affect. Run relevant existing
tests and native output comparisons on original and candidate builds, with
identical fixtures and both precisions. Retain exact discrete/descriptor
checks and the established numerical policies. Add missing tests that could
detect a mistake in this change: for a tiled gather, partial tiles, tiny planes,
channel boundaries and exact selection/averaging are especially relevant.

Review outputs, errors, live behavior, ownership and consumers. Reuse valid
source/build-bound evidence for unchanged behavior, with an explicit dependency
rationale. A launch-layout edit does not automatically require rebuilding the
entire operator's ownership/error/live test infrastructure. New allocation,
aliasing, publication, lifetime or synchronization behavior does require fresh
coverage of those risks. Existing failures or missing evidence that undermine
the changed behavior still block acceptance; do not mark untested behavior as
passing or widen tolerances.

Trace shared-code dependencies broadly for functional validation. Time another
operator or pipeline when the edit could plausibly change its execution cost,
such as shared kernels, tensor layout/shape, synchronization, scheduling,
ownership or resource contention. An unchanged downstream consumer of identical
data does not automatically require its entire recipe suite to be timed.
Use a relevant composition smoke test where useful and record omitted timing
as a claim limit, not failed correctness or implicit measured protection.

## Small, bounded performance experiment

Before candidate measurements, record the mechanism, editable files, target,
representative cases/precisions, comparator, timing boundaries, sampling, order,
practical decision criteria and total run/time ceiling. Reuse existing campaign
authorization where applicable; a changed acceptance policy needs a fresh
recorded design before new candidate evidence. Never retrofit a failed trial.

Select cases by execution path and risk, rather than a Cartesian product:

- The intended expensive workload and important distinct changed paths.
- A small overhead case and a representative intermediate/boundary case.
- Both precisions when their execution costs can differ; keep any omitted
  timing precision explicit, independently of mandatory correctness coverage.
- Live/setup/cold or integration timing when affected or application-critical.

For a simple local change, roughly 3–6 cases often cover these roles; that is
a starting point, not a minimum, maximum or permission to omit a distinct risk.
Start with existing recipe dimensions, five warmups and 50 frames unless the
decision requires a different horizon. Record raw frames, setup/cold/live
observations and timing context. With 50 frames, p99 is an extreme-sample
diagnostic, not a well-established population tail. Steady p50 is the usual
primary objective; promote tails or startup to acceptance objectives only for
a concrete requirement or affected path.

Screen with a few paired A/B runs, then freeze a promising candidate and run
one fresh bounded paired check, disjoint from selection data. About 3–5 pairs
per phase is a useful cheap starting point, not a statistical power guarantee.
Balance/randomize AB and BA order; use the same workload and timing boundaries.
Keep separate processes where initialization/carryover matters. Longer runs,
four-run controls or extra diagnostic cases should answer an observed question,
not be automatic prerequisites. Forecast wall time including verification and
launches before running; do not spend hours resolving a marginal microsecond
gain by default. Enforce the shared ceiling with per-process timeouts and
owned-process/container cleanup, preserving a cleanup reserve. A harness flag
that checks time only after a case is not a hard orchestration deadline; do not
launch another job unless it can fit the remaining phase budget.

For engineering acceptance, show absolute before/after latency, paired effects,
run-to-run range and chronological order for each selected case. Keep when
correctness passes, the fresh evidence supports the predeclared useful gain,
and there is no unresolved material regression in the selected risk cases.
The usual >2% objective must also be put in absolute latency/application context;
it does not itself establish usefulness or statistical significance. Declare
any practical regression allowance in advance; it changes the claim from
zero-slowdown protection and must be visible. Do not invent one after a loss.

Reject wrong behavior or a confirmed unacceptable tradeoff. A gain comparable
to variability, conflicting runs, insufficient coverage or unresolved drift
ends as inconclusive/validation-blocked at the predeclared ceiling; restore the
candidate and move on. Do not discard genuine unfavorable results or continue
until a pass. Preserve infrastructure failures separately and explain any
bounded replacement run. Unexpected regressions may justify a focused follow-up
within the allocation; they do not mandate a full-library rerun.

This is an engineering keep decision, not a familywise statistical guarantee,
proof of zero regressions, population-tail bound or validation of unmeasured
variants. Do not call small-sample ranges confidence intervals. Formal tests,
interim looks and significance claims need an appropriate declared design;
use [formal confirmation](confirmation.md) when those claims are requested.

## Example: local VirtualArrayAssembly tiling

A reasonable starting timing set is `baseline-small`, `baseline-large`,
`average-large` and `keep-all-medium`, both precisions. These cover overhead,
large deduplication, averaging and a distinct channel-count/layout scenario.
Add a case only for an uncovered changed path or requirement. With three
screen pairs and three fresh-check pairs, this is 96 precision-child runs
(`4 cases × 2 precisions × 2 builds × 6 pairs`), not tens of thousands of
consumer wrappers. Forecast their real cost plus native work and cleanup;
this count alone is not a wall-time or sensitivity guarantee.

Run relevant existing VAA tests and exact native comparisons, adding typed
partial-tile/marker checks. Review unchanged descriptor/lifetime behavior and
reuse applicable evidence; retain relevant geometry-to-Beamscan functional
coverage. Do not time every Beamscan/CFAR/SnapshotGather recipe solely because
they consume VAA outputs. If the edit changes synchronization or resource
behavior that could affect the pipeline, add a targeted composition timing
case. Claims cover only the measured cases. This example does not authorize
runs, replace an active frozen design or waive a real correctness gap.

## Avoid duplicate experiments and provenance work

If the candidate's comparator is the preserved original and its fresh check
covers the final retained bytes, that evidence is already original-versus-final.
Do not repeat it solely to label it final. After multiple keeps, measure a
direct original/final comparison only for the cumulative results being claimed;
otherwise report cumulative performance as unmeasured. Never multiply gains.
Refresh attribution only when it will inform the next decision.

Verify executable/runtime/source/recipe identities and runtime resolution at
phase boundaries, preserve immutable artifacts, and verify again at closure.
A collector may use a phase-scoped verification receipt with cheap per-launch
identity checks if it detects mutations, invalidates affected data and fully
reverifies on resume or change. An unchecked path/mtime-only cache is inadequate.
Do not repeatedly hash gigabytes before each tiny run as a process requirement.
Legacy helpers currently do so; the recommendation alone does not implement a
cache or authorize bypassing their integrity checks.

## Execution and records

The maintained `radar_perf.py` automated acceptance/replay state machine still
implements version 4; it does not implement this engineering decision. There
is no implied version-5 schema or new CLI flag. For the default workflow, use
the existing clean harness and source-bound native tests, preserve explicit
argv/settings, child exit status, raw reports, identities and checks in an
ignored experiment directory, and make a reviewed engineering decision record.
Existing collectors may be used for bounded screening/collection at their
actual cost, but never pass screening JSON off as a version-4 speed gate.

Record correctness, contract/coverage review, equivalence, performance evidence,
decision criteria, disposition, restoration and limitations. Publish every
attempt in the [candidate register](candidate-records.md), labeling acceptance
basis `engineering` or `formal-v4`. Do not invoke version-4 `candidate`/`finish`
to manufacture a passing automated session for engineering evidence. Preserve
all historical formal sessions unchanged. Tools and reports must distinguish
their stronger statistical guarantees from this default scoped judgment.
