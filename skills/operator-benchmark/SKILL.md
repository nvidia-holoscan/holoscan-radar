---
name: operator-benchmark
description: Qualify measurement protocols and assess or coordinate bounded optimization campaigns for sets of holoscan-radar operators on HSDK 5 CUDA deployments. Uses operator-perf for candidate edits and acceptance.
---

# Operator Benchmark Campaign

Use **discover → correctness/screen → freeze design → paired A/B → integrated confirmation → report**.
This skill coordinates deployment-wide evidence; delegate operator assessment,
attribution and candidate sessions to [operator-perf](../operator-perf/SKILL.md).
Its correctness, equivalence and declared statistical acceptance rules remain
authoritative. Scope is this repository's HSDK 5 CUDA deployments.

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
- Use version 4, the current deployment settings, both supported precisions,
  applicable existing workload variants and protected early/full/setup/cold/live
  metrics. Start from five warmups and 50 measured frames; resolve affordable
  case coverage, screening and confirmation counts from costs and inference
  resolution in the consolidated design. No universal frame or block count
  guarantees adequate sensitivity. With no fixed deployment, label workload
  scenarios synthetic and retain recipe dimensions unless a change is scoped.
- A fresh campaign gets new session identities and current-state reports.
  Reuse older builds/evidence only through verified receipts; historical budgets,
  confirmations and failures are not automatically active campaign authority.
  Respect explicit continuing authorization when the user is resuming instead.
- Use the existing checkout and dedicated campaign branch unless the caller or
  repository guidance selects another layout. Preserve edits; do not create
  worktrees merely to start discovery. Keep immutable builds/raw evidence in
  ignored paths and versioned overview, baseline, disposition and final reports.

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
- Freeze the design, sampling, identities and budgets before confirmation.
  Screen correct bounded candidates with paired A/B. Confirm promising frozen
  candidates on fresh randomized balanced four-run blocks: two current-best A
  runs and two candidate B runs. Follow operator-perf's
  [integrated confirmation protocol](../operator-perf/references/confirmation.md).
  A/A and B/B are dependent matched controls, with no fixed agreement limits.
- Collect native correctness and repeat-output evidence before candidate edits.
  Protect the full confirmed operator/consumer matrix during confirmation.
  Screening or representative results do not validate unmeasured cases. Keep
  unresolved scopes visible while independent discovery proceeds.
- Infer assessment or optimization intent using the defaults above. Before
  production edits, present one
  consolidated review of concrete operator configurations: outputs, targets,
  eligibility, protected metrics, editable scope, consumers, sampling and limits.
  Record confirmation. Subsequent operator-perf sessions proceed autonomously
  within those configurations; expanded scope or changed policy needs confirmation.
- Rank opportunities by attributed avoidable recurring work, removable fraction,
  affected workloads and mechanism confidence. Reassess after each disposition.
  Tool commands coordinate sessions; the agent applies and validates edits.
- Publish baseline, per-operator disposition and final reports between sealed
  sessions. Link branch/artifact paths in chat at every milestone. Final gains
  require direct original/current-best paired measurements.
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

Use explicit version-4 configurations and the
[command guide](../../performance/optimization/CONFIRMATION.md). Campaign design
checks do not launch A/A; baseline preparation verifies native behavior and
collects inexpensive original-build screens. Session admission requires native
coverage, not timing repeatability. Production acceptance uses integrated
confirmation and evidence-backed agent control review. Host validation does not
qualify a GPU deployment; no new measurement budget is implied.

Respect the caller's worktree or single-checkout choice. Keep immutable builds,
archived runtime dependencies, checkpoints and active notes in ignored session
paths. Preserve user edits and historical evidence. Version-2/version-3 evidence supports
historical reporting/replay; the new workflow requires implemented policy-version
support and fresh confirmed evidence. No implicit migration, CPU fallback,
threshold relaxation, clock changes, skill installation or external publication.
