---
name: operator-benchmark
description: Qualify measurement protocols and assess or coordinate bounded optimization campaigns for sets of holoscan-radar operators on HSDK 5 CUDA deployments. Uses operator-perf for candidate edits and acceptance.
---

# Operator Benchmark Campaign

Use **discover → relevant correctness → representative paired screen → fresh bounded check → report**.
This skill coordinates deployment-wide evidence; delegate operator assessment,
attribution and candidate sessions to [operator-perf](../operator-perf/SKILL.md).
Its correctness/equivalence rules and selected acceptance basis remain
authoritative. Use [proportionate acceptance](../operator-perf/references/acceptance.md)
by default; formal version-4 confirmation is an optional statistical tier.
Scope is this repository's HSDK 5 CUDA deployments.

Read [configuration](references/configuration.md) before preparing a campaign
and [reporting](references/reporting.md) before milestone publication. Use
`python3 performance/optimization/radar_perf.py campaign --help` for maintained
commands. Resolve repository paths from the checkout, not an installed skill.

## Fresh campaign defaults

A sufficient request is: **“Use operator-benchmark to start a fresh optimization
campaign for all operators with performance harnesses.”** The user need not
repeat the workflow, statistical policy, reporting or safety rules below.

Infer **optimize** from an explicit optimization request; assessment-only remains
the default for benchmarking/assessment requests without editing intent. An
optimization request authorizes preparation of the optimization plan, while the
concrete consolidated review is still required before production edits.

- With no narrower operator scope, inventory all public operators and assess
  those with existing harnesses. List harnessless operators as coverage gaps;
  do not create recipes or expand the campaign implicitly. Discover the count
  from this checkout rather than hard-coding a historical total.
- Begin breadth-first: review implementations and verified historical evidence,
  identify correctness gaps, forecast costs, then collect affordable baseline
  screens across the inventory before deep optimization. Rank avoidable work
  and process one operator session at a time; reassess the queue after each.
- Start discovery, evidence review and forecasting immediately. Ask only for
  missing deployment facts that affect decisions and a positive measurement
  budget before timing. Propose bounded phases with assumptions and actionable
  alternatives. Do not spend the allocation on the first operators without
  exposing the coverage tradeoff. Exhaustion leaves explicit partial coverage.
- Use current deployment settings, existing recipe dimensions and both
  precisions for correctness. Select timing cases by distinct execution path,
  intended use and risk. Start with five warmups/50 frames and a few paired
  runs; choose additional horizons/controls only for a concrete question.
  Broad correctness coverage does not mandate timing every variant or consumer.
  Formal version-4 collection is available when its guarantees are requested.
- A fresh campaign gets new session identities and current-state reports.
  Reuse older builds/evidence only through verified receipts; historical budgets,
  confirmations and failures are not automatically active campaign authority.
  Respect explicit continuing authorization when the user is resuming instead.
- Use the existing checkout and dedicated campaign branch unless the caller or
  repository guidance selects another layout. Preserve edits; do not create
  worktrees merely to start discovery. Keep immutable builds/raw evidence in
  ignored paths and versioned overview, baseline, disposition and final reports.

For authorized skill commits, use `deustice/optimization-skills`, then rebase
the campaign branch on top before continuing. Follow the shared
[branch and workspace rules](../operator-perf/references/workspace.md#skills-and-campaign-branches)
to preserve sealed evidence and uncommitted work.

Ask for one consolidated review once concrete operator configurations and
baseline coverage are available. After confirmation, run autonomous bounded
operator sessions within that scope. No separate user approval is needed for
each candidate, control interpretation or routine disposition. Do not infer
commit/push/publication authorization from a new optimization request.

## Campaign execution

- Inventory requested public operators, existing recipes, cases, precisions,
  verification and consumers. Missing harnesses are blocked coverage; creating
  recipes requires explicit scope. Review transitive consumers with operator-perf.
- For library-wide opportunity reviews, begin breadth-first using existing
  evidence and implementation review. Representative measurement pilots do not
  narrow the requested campaign. Blocked timing prevents affected acceptance,
  not independent discovery or attribution. Report tool-chain validation and
  operator optimization outcomes separately; further tool work should resolve
  a concrete assessment or candidate decision.
- Require a positive **user-specified measurement budget** before timing.
  Choose useful existing workloads across processing families, subject to cost;
  honor explicit overrides. Do not make a general qualification study or a
  standalone full-matrix A/A sweep a prerequisite to candidate testing.
- Predeclare the candidate, representative timing set, useful gain/regression
  criteria, order and stopping ceiling. Screen a correct change, then run one
  fresh bounded paired check before an engineering keep. Record uncertainty
  without claiming familywise guarantees or performance of unmeasured cases.
  Use formal [confirmation](../operator-perf/references/confirmation.md) only
  for a deliberately selected and affordable statistical claim.
- Run relevant native correctness/equivalence before acceptance. Reuse valid
  unchanged coverage and add candidate-specific gaps. Trace transitive consumers
  for functional impact; time consumers only for plausible performance effects,
  rather than automatically multiplying all their recipe variants.
- Infer assessment or optimization intent using the defaults above. Before
  production edits, present one
  consolidated review of concrete operator configurations: outputs, targets,
  correctness scope, representative timing, editable files, relevant consumers,
  acceptance basis and limits.
  Record confirmation. Subsequent operator-perf sessions proceed autonomously
  within those configurations; expanded scope or changed policy needs confirmation.
- Rank opportunities by attributed avoidable recurring work, removable fraction,
  affected workloads and mechanism confidence. Reassess after each disposition.
  Tool commands coordinate sessions; the agent applies and validates edits.
- Publish baseline, per-operator disposition and final reports between sealed
  sessions. Link branch/artifact paths in chat at every milestone. Final gains
  require direct original/current-best paired measurements. Reuse the fresh
  first-candidate check when it already compares original and final retained
  bytes; do not collect an identical second confirmation.
- Track every attempted candidate in a linked `candidates.md`, with a document
  explaining the actual change, acceptance/rejection/blocker and measured
  impact. Follow operator-perf's [candidate records](../operator-perf/references/candidate-records.md);
  include unsuccessful attempts and reconcile counts with sealed sessions.
- Maintain a linked, user-readable `overview.md` throughout the campaign,
  following [reporting](references/reporting.md). At each milestone state the
  decision enabled by the evidence and the next action with its dependencies.
  For unstable timing, use operator-perf's
  [distribution analysis](../operator-perf/references/distributions.md) before
  expanding frame counts. Do not let exploratory protocols become gates.
  Distinguish invalid evidence, ordinary variability, limited sensitivity and
  demonstrated regressions. Controls help investigate confounding; a failure to
  reject a control null does not prove measurement reliability.

For engineering experiments, keep reviewed source-bound decision records and
raw harness/native artifacts as described in [acceptance](../operator-perf/references/acceptance.md).
The maintained campaign/session state machine and [command guide](../../performance/optimization/CONFIRMATION.md)
still implement formal version 4. Do not invent a new schema, pass engineering
screens as formal gates or imply automated support that does not exist. Frozen
historical sessions retain their policy. A new timing design needs an explicit
positive budget and prospective scope; no additional allocation is implied.

Respect the caller's worktree or single-checkout choice. Keep immutable builds,
archived runtime dependencies, checkpoints and active notes in ignored session
paths. Preserve user edits and historical evidence. Version-2/version-3 evidence supports
historical reporting/replay; new experiments explicitly record their acceptance
basis and fresh evidence. Automated formal acceptance requires supported policy
versions. No implicit migration, CPU fallback,
threshold relaxation, clock changes, skill installation or external publication.
