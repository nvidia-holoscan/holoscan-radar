# Integrated candidate confirmation — implemented

2026-10-20 · `deustice/optimization-skills`. Skills and maintained host tools now
implement explicit measurement-policy version 4. No production operator was
changed or measured in this task. Historical version-2/version-3 evidence and
decisions remain unchanged.

## Workflow

**Discover and verify correctness → inexpensive opportunity screening → bounded
candidate → paired A/B screen → freeze → fresh integrated confirmation → decision.**

Each confirmation block has two current-best A runs and two frozen-candidate B
runs in a recorded randomized balanced order. A/A and B/B are matched controls
from that same experiment. There is no standalone full-matrix A/A admission,
mandatory current-best A/A refresh or fixed percentage/microsecond agreement
threshold. Controls diagnose interpretation and uncertainty; a non-significant
control does not prove reliability, and observed noise is not a regression margin.

The complete block is the inference unit. Equal-run frame mixtures supply
p50/p90/p99 and early/full-window summaries. Setup/cold/live costs remain
separately protected. Exact paired-block log-ratio intervals use conservative
campaign/consumer/candidate multiplicity, with duplicate identical endpoints
counted once. Positive effect means slower. All declared targets must meet the
practical gain objective; a demonstrated protected slowdown rejects. Finite
inconclusive protected intervals are permitted and reported; unsupported or
unbounded inference remains unresolved.

## Tools and safeguards

- Version-4 campaign baseline preparation verifies native correctness/coverage
  and collects inexpensive original-build screens. Admission replays native
  snapshots and hashed fixture evidence; no A/A calibration artifact is required.
- `measure --phase screen` and `measure --phase confirmation` freeze build,
  recipe/runtime, plan, sampling, order and screening identities. Production
  measurements bind a confirmed session and its remaining time budget.
- Confirmation rejects screening reuse, repeated observations, stale identities,
  changed timing/workload context, missing precision and unsealed outputs.
  Per-device locks and owned process cleanup remain active; verified completed
  units resume without relaunching them.
- `inspect-confirmation` reconstructs statistics and joint controls. An
  evidence-backed agent review records interpretability and unresolved
  confounding. This is not another user approval or a way to override failed
  speed/correctness gates. Acceptance replays the raw evidence and review.
- Reports show absolute timings, signed effects, simultaneous intervals,
  chronological block effects and dependent A/A–B/B diagnostics. Cumulative
  gains require direct preserved-original versus retained comparisons.

Executable examples and schemas are in the
[version-4 command guide](../../../../performance/optimization/CONFIRMATION.md).
The [skill protocol](../../../../skills/operator-perf/references/confirmation.md)
and [measurement rationale](../../../../skills/operator-perf/references/measurement-redesign.md)
record the learnings and approved interpretation.

## Validation and limits

All **119 host-tool tests passed**, including 14 new confirmation tests. These
exercise actual synthetic subprocess collection through complete session
acceptance/reporting; highly variable controls without an automatic veto;
rejection of a true protected-tail slowdown despite median improvement;
inconclusive protection; unit scaling; correct consumer scope; holdout and
replay checks; interruption/resume; and legacy compatibility. Simulated null
experiments include within-run autocorrelation, process shocks, rare stalls
and small/long latency scales, with a Monte Carlo familywise-coverage check.

Synthetic tests validate tools under declared models; they do not prove the
pair-independence/contrast assumptions for a deployment, measure actual operator
speedups or replace GPU correctness fixtures. A real deployment exercise still
needs a positive explicit budget. The prior MultiSourceExpansion ownership and
native boundary coverage gaps remain open; the historical 5% policy stop is
preserved and is not retroactively accepted under version 4.

## Next cycle

Start a fresh chat with `skills/operator-benchmark/SKILL.md`, discovery of all
26 harness-backed operators, the existing deployment context, and an explicit
budget for the intended measurement phases. Preserve existing checkout edits.
Use version 4 and begin breadth-first opportunity assessment; do not restart the
legacy A/A qualification campaign. Complete affected native verification before
candidate edits, then confirm the concrete scope and collect fresh integrated
validation for a promising candidate. Retained optimization gains remain **none**.

Artifacts and validation logs:
`build-optimization/operator-perf/assessment-2026-10-20/integrated-confirmation/`.
No commits to unrelated operator/harness edits, worktrees, pushes, external
issues, clock/power changes or skill installation are part of this change.
