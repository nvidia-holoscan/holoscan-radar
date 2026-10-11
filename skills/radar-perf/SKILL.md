---
name: radar-perf
description: Assess and optimize holoscan-radar operator performance using the HSDK 5 harness, with confirmed per-output equivalence policies and gated single-fix or bounded iterative optimization. Use for profiling, bottleneck analysis, or operator tuning in this repository.
---

# Radar Performance

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

Before editing, propose and obtain confirmation of the concrete session:
operator, comparison rules for **each port and precision**, cases, objective,
protected cases, editable scope, and limits. Defaults: five candidates, two
consecutive unsuccessful candidates, all three sizes, steady latency, ten
alternating A/B pairs. Wall-time and target limits are optional. All defaults
are configurable. Existing confirmation remains valid until scope or policy
changes. Assessment does not need an editing-session confirmation.

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
