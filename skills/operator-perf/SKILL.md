---
name: operator-perf
description: Assess and optimize holoscan-radar operator performance using the HSDK 5 harness, with confirmed per-output equivalence policies and gated single-fix or bounded iterative optimization. Use for profiling, bottleneck analysis, or operator tuning in this repository.
---

# Operator Performance

Use this repository's maintained performance recipes and locked contracts.
PR #20's `perf-analyze`, `perf-optimize`, and `perf-autotune` motivated this
workflow; their HSDK 4 commands and retired agent harness are not dependencies.

## Agent Compatibility

These instructions and Python/C++ tools are model- and agent-independent.
No provider API, model choice, agent runtime, or special messaging channel is
required. Any coding agent with repository read/edit and shell access can
follow this workflow. Native skill discovery depends on the agent; otherwise,
read this file directly and follow its relative reference links. Repository
coordination and user confirmation remain required regardless of agent.

## Choose The Mode

- **Assess:** measure, profile, and recommend. No permanent production edits.
- **Fix:** apply and validate one candidate requested by the user.
- **Optimize:** repeat gated candidates within confirmed limits when the user
  requests automated or iterative optimization.

Infer mode from the request; clarify ambiguous authorization to edit. Read
[assessment](references/assessment.md) for measurement and attribution. For
either editing mode also read [optimization](references/optimization.md).
Read [tool interface](references/tools.md) when preparing configuration,
verification, or reports. Resolve repository paths from the checkout, not from
the skill's installation path.

## Workspace Layout

Respect the caller's workspace choice. Both layouts are supported:

- **Single checkout:** use a dedicated branch and separate ignored
  `build-optimization/operator-perf/<operator>/<session>/{original,best,candidate,artifacts}/`
  directories. Archive executables and resolved runtime libraries before another
  build. Checkpoint current-best files; restore only through verified receipts.
- **Worktrees:** use `.worktrees/operator-perf/<operator>/session-<N>/` with
  original, best, candidate and artifact directories. Create no worktrees when
  the caller requests the existing checkout.

Build identities, runtime resolution, source fingerprints and checkpoint rules
apply to both layouts. Never overwrite an immutable build. Tell the user the
branch and session path at milestones. Absolute, hash-bound artifact paths must
remain fixed during a session. Read [workspace safety](references/workspace.md)
for restoration and scratch rebuilds.

## Measurement Design And Confirmation

Use the [proportionate acceptance workflow](references/acceptance.md) by default.
Separate broad relevant correctness checks from a small representative performance
experiment. Profile to answer a concrete question; do not require a general
qualification study before trying a bounded change.

Require a positive caller-specified measurement budget before timing. Predeclare
targets, risk cases, precisions, sampling, useful gain, regression criteria and
stopping limits. Screen with a few paired A/B runs, then use one fresh bounded
check of a promising frozen candidate. Record absolute latency, variability,
chronology and limits. Small overhead cases and distinct changed paths matter
more than timing every recipe combination. Correctness remains both-precision
and exact or covered by an already approved per-output policy.

Time downstream operators when the diff plausibly affects their performance;
functional dependency does not itself require repeated downstream timing.
Reuse valid unchanged correctness evidence and fill candidate-specific gaps.
Stop an ambiguous gain at the declared ceiling rather than escalating repeats.
No universal four-run blocks, familywise endpoint inventory, complete-consumer
timing matrix or duplicate fresh-final experiment is required for the default
engineering keep decision. It makes scoped empirical claims, not formal
statistical guarantees or proof of no regressions.

Use [formal confirmation](references/confirmation.md) and
[distribution inference](references/distributions.md) when explicitly needed
for the requested assurance or statistical claim. The automated tools implement
that optional version-4 workflow; the default engineering decision is recorded
and reviewed separately as explained in [tool compatibility](references/tools.md).
Do not label manual or screening evidence a passing version-4 gate. Historical
policies/evidence stay unchanged. A revised active policy needs a fresh design,
not retroactive reinterpretation of a previous result.

For deployment-wide inventory, qualification and reporting use
[operator-benchmark](../operator-benchmark/SKILL.md). It delegates candidate
edits and acceptance to this skill.

## Invariants

Identify the actual execution path: public operator, application adapter, or
benchmark-only code. Trace the requested target's call site to the changed
function. Attribute gains to that path; an isolated candidate kernel is not an
integrated operator improvement.

Read the applicable FMCW/SAR `OPERATOR_API.md`, relevant tests and workflows,
`TEMPORAL_CONTRACTS.md`, `OPERATOR_DEVELOPMENT.md`, `NEW_OPERATOR_CHECKLIST.md`,
`STATUS.md`, and `SDK_DEVELOPMENT.md` before substantial work. Inspect the
operator's implementation and recipe; HSDK 5 source is authoritative where
public documentation is insufficient. Record implementation claims in STATUS.

Preserve public ports, Params and defaults, constructor sizing, precision,
algorithms, errors, descriptors, output ownership and ordering, optional-output
behavior, and state/temporal semantics. Numerical tolerance authorizes rounding
differences only; it does not authorize changed math, discrete decisions,
precision, or interface behavior. Never widen tests or comparison tolerances
to make a candidate pass.

DetectionReduction has an explicitly authorized, opt-in
[input-derived semantic policy](../../docs/performance/optimizations/detection-reduction-validation.md).
It requires fresh native input witnesses, coupled port checks, and existing
computed-tie fixtures. It does not authorize changed formulas or consumer
outputs. Retain exact comparison elsewhere; never migrate historical evidence.

Before editing, confirm the concrete scope, per-output comparison rules,
representative timing design, acceptance basis and limits. Existing campaign
confirmation covers routine candidates within that scope; do not ask again
without a material change. Assessment does not require editing confirmation.
Use bounded candidate/unsuccessful-attempt limits appropriate to the request;
do not multiply the timing family by a hypothetical 20 future attempts.

A turn means one attempted candidate, not one chat message or tool call.
Prepare relevant baseline correctness before edits. Preserve contracts and
run candidate-specific correctness/analytic/native checks; a whole-operator
coverage expansion is not automatically a prerequisite for a local change.
New behavior/lifetime/synchronization risks or failures undermining the changed
surface still need resolution. Exact equivalence is independent of timing scope.
Confirm deployment context; clock changes require separate authorization.

Do not change exclusions, numerical policies, objectives or practical regression
criteria after seeing an unfavorable result. Preserve failed/inconclusive data
and stop at the declared limit. A revised design is prospective, source-bound
and explicit about any weaker guarantee. Formal-v4 sessions retain their frozen
rules until closed; never migrate their evidence in place.

Use [workspace safety](references/workspace.md) for scratch profiling and moved
archives. Never restore instrumentation with Git checkout/reset: checkpoint the
current best, verify its fingerprint after restore, and rebuild every touched
target before final gates. A moved archive needs an audited path map, not edits
to its hash-bound JSON.

Suspected bugs are exceptional findings, not optimization opportunities. Alert
the user promptly, reproduce on the original baseline where possible, and
record expected/observed behavior and affected cases. Do not fix them, weaken
validation, or silently exclude cases. Continue independent optimization when
its validation remains sound; block affected candidates otherwise. Revert
candidate-introduced defects. Highlight every unresolved suspected bug in the
final response and report, including on interruption; explicitly state when
none were found. Draft tracker-ready evidence; file external issues only when
authorized.

## Deliver

For every attempted candidate, follow [candidate records](references/candidate-records.md):
record a linked register and a concrete before/after explanation, gate-based
decision rationale and measured impact, including rejected/blocked attempts.
Keep these reports and evidence in ignored session directories by default;
recording and linking evidence does not imply committing it.

When a commit is authorized, include the retained implementation, its required
integration, and focused regression tests. Keep temporary probes, experiment
runners, copied benchmarks, raw measurements and supporting reports local unless
the user requests them as repository deliverables. Maintained project benchmarks
or fixtures may belong in the change when needed for its requested behavior or
regression coverage. Do not add an unused kernel to stand in for integration.

Keep raw artifacts, provenance, candidate decisions, coverage gaps, and the stop
reason. Remove temporary instrumentation and verify the retained state. Present
the diff, cumulative results versus the original baseline, and report locations.
Do not commit, push, publish, or install the skill outside the repository unless
authorized. A lack of GPU/SDK/profiler access must be reported, not worked around
with CPU compute or guessed measurements.
