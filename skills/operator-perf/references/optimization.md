# Optimization

## Establish The Session

Use [proportionate acceptance](acceptance.md) unless the request needs formal
statistical confirmation. Record a concrete engineering experiment: scope,
comparator, per-output policies, correctness checks, representative timing cases,
criteria and budget. Existing authorization covers routine choices within the
confirmed scope; changed acceptance rules are prospective and explicit.

The maintained session JSON/`init`/five-gate acceptance state machine implements
[formal version 4](confirmation.md), not the default engineering review. Use
that optional path only when selected, and never fabricate its passing gates.
Engineering records must clearly identify their acceptance basis and raw evidence.

Respect the caller's single-checkout or worktree choice and preserve the intended
starting state in the layout described in SKILL.md. If the user
has uncommitted relevant work, preserve and reproduce it there before taking
the baseline. Do not stash, reset, or discard their work. Keep original and
current-best builds separately, including their runtime libraries; merely
copying an executable is not adequate isolation for shared-library changes.

Run baseline correctness and analytic-reference tests. Capture outputs at least
twice using native `holoscan::testing` fixtures. Compare repeated captures to
detect nondeterminism; stop affected candidates and consult the user if policy
cannot express valid baseline behavior. Snapshot copies never enter timing.

Audit verification coverage against the candidate's affected surface: both
precisions; all outputs/descriptors; optional wiring; empty/boundary inputs;
multi-frame ownership; live match/miss and shape changes; cached/unseen
transitions; invalid-input behavior. Bundled fixtures cover ClutterFilter and
expanded split/fused FMCW range-stage subsets. Reuse valid unchanged evidence
with a dependency rationale; extend native
fixtures and analytic tests for gaps relevant to the candidate, rather than
requiring a complete new whole-operator suite for every local edit.
Do not build a second radar-owned single-operator Graph harness.

Extend fixtures and record the STATUS claim **before** freezing the intended
starting state, creating manifests and initializing the session. Both baseline
and candidate use identical fixture/recipe code. A preparatory commit is not
required or authorized by this workflow. STATUS edits also change repository
fingerprints: keep subsequent coordination notes in ignored session artifacts
until final validation, then record the final reviewed source state separately.

## Candidate Loop

1. Select one attributed bottleneck against current best. Review dependencies of
   edited code and relevant application paths. Separate the functional coverage
   set from the smaller performance-risk set. Register the proposed mechanism,
   target cases, useful gain and possible costs before editing.
2. Checkpoint explicitly listed files and preserve immutable baseline builds.
   Apply the smallest behavior-preserving change. Both sides use identical
   fixture/recipe code and timing boundaries.
3. Run relevant existing tests, analytic checks and native output comparisons,
   both precisions. Add missing checks for the changed path. Explain reuse of
   unchanged ownership/live/error coverage and any remaining limits. A failure
   undermining the changed behavior blocks the candidate.
4. Screen with a few paired A/B runs on the predeclared representative set.
   Freeze a promising candidate and run a fresh bounded paired check. Record
   every run, variability and unfavorable result. Time affected downstream paths
   when there is a plausible performance dependency, rather than automatically
   repeating every consumer recipe. Follow [acceptance](acceptance.md) for
   engineering criteria and the distinct optional formal protocol.
5. Record contract/coverage review, tests, equivalence and performance evidence,
   then keep, reject or mark validation-blocked. Explain absolute benefit and
   tradeoffs. Restore rejected/blocked edits with a verified checkpoint receipt.
   Publish the [candidate explanation and register entry](candidate-records.md)
   between sealed experiments. Never count a source lead as an attempted edit.
6. Reassess the queue. Refresh attribution if it can guide the next change;
   avoid a mandatory profiler or A/A sweep after every keep. Stop at the
   confirmed attempt, unsuccessful, time or target limit.

Choose attempt limits appropriate to the task. One attempted candidate is one
turn; preparatory work and messages are not turns. Missing validation counts
separately from rejection but remains visible. Neither a time limit nor weak
sensitivity authorizes extra runs or a retroactive change in criteria.

After a candidate, report its measured cases, acceptance basis, absolute and
relative results, uncertainty/variability, regressions and omitted scope.
A fresh first-candidate check against the preserved original already supplies
original/final evidence when retained bytes match; do not duplicate it. After
multiple keeps, use direct original/final measurements for cumulative claims or
label them unmeasured. Do not multiply intermediate gains.

For temporary profiling, follow [workspace safety](workspace.md). Restore the
current best and verify rebuild receipts before rerunning gates. An applicable
existing session claim can be reused; do not edit STATUS every candidate or
exclude locked documentation from provenance. Keep ongoing notes in artifacts.

Record the baseline test failing set. No-new-failures is supporting evidence,
not a blanket correctness waiver: a pre-existing failure that undermines the
affected contract, output or temporal checks blocks that candidate. Explain
unrelated SDK/environment failures and repeat suspected flakes on both builds;
do not silently treat a rerun as proof the original failure was irrelevant.

## Bugs And Completion

Persist a suspected bug immediately (use `bug` for an active formal session,
or the engineering experiment's findings record), with baseline reproduction,
evidence, expected/observed behavior, and the cases whose validation it blocks.
Alert the user immediately through the agent's normal conversation interface.
Continue only candidates independent of those
cases; keep failing cases visible. Do not alter baseline snapshots, relax rules,
or repair a discovered bug as part of performance work.

Distinguish suspected operator correctness bugs from measurement limitations.
Do not automatically drop affected cases. For measurement-only limitations,
the user may confirm a new, explicitly reduced session scope before fresh
measurements, retaining the omitted cases as visible coverage gaps and making
only scoped performance claims. Correctness/contract/equivalence coverage for
an affected candidate is never optional. Do not retroactively accept a failed
candidate by switching objectives, thresholds or exclusions.

At completion or interruption, seal the engineering decision record with its
concrete stop reason, or use version-4 `finish`/`report` for an actual formal
session. Do not feed engineering evidence into formal replay. Include remaining
findings, coverage gaps, attempt decisions and directly measured original/final
results where available. An all-rejected session is a valid outcome. Remove
instrumentation, verify the retained build, and run pinned lint before handoff.
