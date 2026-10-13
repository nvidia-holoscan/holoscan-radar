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

## Qualify Before Baseline Collection

Use [measurement qualification](references/qualification.md) before expensive
baselines. Require a positive caller-specified qualification time budget; do
not silently choose a budget. Separate exploratory calibration, frozen protocol
selection, disjoint held-out validation, full-case calibration and production
acceptance. Representative results qualify only measured workloads.

Median aggregation remains the default. Keep five warmup frames and the
harness's existing process-warmup semantics; capture chronological raw frames
and boundaries. Version 3 protects median/p90/p99 for the first 50 measured
frames and the complete window, plus setup, cold and applicable transitions.
Confidence width and +/-1% precision are descriptive diagnostics, never new
acceptance gates or evidence of adequate power. Report unresolved regimes and
blocked cases without extending budgets or relaxing thresholds automatically.

For deployment-wide inventory, qualification and reporting use
[operator-benchmark](../operator-benchmark/SKILL.md). It delegates candidate
edits and acceptance to this skill.

## Invariants

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

Before editing, propose and obtain confirmation of the concrete session:
operator, comparison rules for **each port and precision**, cases, objective,
protected cases, editable scope, and limits. Defaults: 20 candidate turns, three
consecutive unsuccessful candidates, all three sizes, steady latency, 20
alternating A/B pairs with isolated case/repeat processes. Before confirmation,
use the selected matrix to propose an adequate budget; 20 is not a power
guarantee. Wall-time and target limits are optional. All defaults
are configurable. Existing confirmation remains valid until scope or policy
changes. Assessment does not need an editing-session confirmation.

A turn means one attempted candidate, not one chat message or tool call.
Qualify the protocol first, then calibrate every full-matrix baseline/metric with repeated A/A runs before editing. Timing
precision coverage is separate from output equivalence. A noisy calibration
blocks affected validation; it does not justify accepting differences. Follow
the reference's familywise gates and report insufficient measurement power.
Confirm metric-specific calibration policies and the intended clock/power
environment. Clock changes need separate authorization; never silently lock
clocks or substitute a controlled-clock benchmark for deployment behavior.

On a failed gate, do not retroactively change exclusions, objectives, numerical
rules, initialization semantics or thresholds to accept that candidate. A new
confirmed policy requires a fresh session and baseline. A significant first-use
regression remains relevant even when its immediate code path is unchanged.

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

Keep raw artifacts, provenance, candidate decisions, coverage gaps, and the stop
reason. Remove temporary instrumentation and verify the retained state. Present
the diff, cumulative results versus the original baseline, and report locations.
Do not commit, push, publish, or install the skill outside the repository unless
authorized. A lack of GPU/SDK/profiler access must be reported, not worked around
with CPU compute or guessed measurements.
