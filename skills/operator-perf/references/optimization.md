# Optimization

## Establish The Session

Prepare a session JSON using the interface in [tools](tools.md). Propose rules
for every output port separately for float and double; integer/index/mask and
descriptor content are exact. Floating rules are exact or confirmed explicit
absolute/relative tolerances with evidence from analytic tests. Confirm the
configuration and limits before edits. Record confirmation through `init`.

Default to a dedicated worktree with the intended starting state. If the user
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
expanded split/fused FMCW range-stage subsets. Extend native fixtures and
analytic tests for uncovered behavior before accepting a dependent candidate.
Do not build a second radar-owned single-operator Graph harness.

Extend fixtures and record the STATUS claim **before** freezing the intended
starting state, creating manifests and initializing the session. Both baseline
and candidate use identical fixture/recipe code. A preparatory commit is not
required or authorized by this workflow. STATUS edits also change repository
fingerprints: keep subsequent coordination notes in ignored session artifacts
until final validation, then record the final reviewed source state separately.

## Candidate Loop

1. Select one actionable finding against the current best. Identify transitive
   consumers of every edited shared helper, kernel, or header, including tests,
   applications, SAR, and other operators. Add their validation to the matrix.
   Before editing, register `plan`: mechanism, baseline source, target cases
   chosen from the confirmed allowable set, and expected percentage gain. Bind
   its hash into every gate context; do not select targets after seeing results.
2. Checkpoint explicitly listed files with `checkpoint`. Add new paths before
   creating them. Apply the smallest candidate preserving behavior and contract.
3. Build and run relevant correctness tests, native output comparisons, and
   temporal/error checks. Save contract-review and coverage evidence. Missing
   output/timestamp coverage is a failed gate, not an exception.
4. Run `measure` using distinct source/build manifests and the same immutable
   recipe and sampling settings. It alternates A/B order and writes raw reports.
   Supply current-best A/A raw pairs to `compare-timing`. Descriptive paired
   medians/bootstrap intervals accompany exact Bonferroni familywise intervals
   used for decisions. Confirm practical gain thresholds before measurements.
   Confirm the target cases and protect all measured cases by default. Reject
   regressions in setup/cold/live timing where these are protected objectives.
5. Record the candidate with evidence for contract, coverage, tests,
   equivalence, and speed gates. Keep only if all pass. For failure, seal the
   candidate file state and `restore` the checkpoint; it refuses unexpected
   intervening edits and preserves previously accepted changes.
6. After a keep, refresh attribution and compare structural evidence with the
   prior run; stale exports cannot establish convergence. Update current best.
   Stop on the confirmed attempt/dry/time/target limit or an empty queue.

Single-fix mode attempts one candidate. Iterative mode defaults to 20 candidate
turns and three consecutive failures; no-win/inconclusive trials count.
Each attempted candidate is one turn, irrespective of chat/tool messages.
Blocked validation is reported separately and does not increment the failure
streak, but an attempted blocked candidate still consumes the 20-turn budget.
Do not spend extra trials outside the confirmed budget. Session checkpoints do
not authorize commits. Session state records stop decisions but the agent owns
the optimization loop and code changes.

## Bugs And Completion

Use `bug` to persist a suspected bug immediately, with baseline reproduction,
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

At completion or interruption, run `finish` with a concrete stop reason, then
`report`. Include remaining findings and coverage gaps, every unresolved bug,
kept/rejected changes, and final versus original-baseline timing. An all-rejected
session is a valid outcome. Remove scratch instrumentation, rebuild, rerun the
final relevant gates, and run pinned lint before handing off.
