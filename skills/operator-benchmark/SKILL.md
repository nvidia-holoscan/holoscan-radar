---
name: operator-benchmark
description: Qualify measurement protocols and assess or coordinate bounded optimization campaigns for sets of holoscan-radar operators on HSDK 5 CUDA deployments. Uses operator-perf for candidate edits and acceptance.
---

# Operator Benchmark Campaign

Use **discover → qualify → freeze protocol → baseline → assess/optimize → report**.
This skill coordinates deployment-wide evidence; delegate operator assessment,
attribution and candidate sessions to [operator-perf](../operator-perf/SKILL.md).
Its correctness, equivalence, calibration and familywise acceptance gates remain
authoritative. Scope is this repository's HSDK 5 CUDA deployments.

Read [configuration](references/configuration.md) before preparing a campaign
and [reporting](references/reporting.md) before milestone publication. Use
`python3 performance/optimization/radar_perf.py campaign --help` for maintained
commands. Resolve repository paths from the checkout, not an installed skill.

- Inventory requested public operators, existing recipes, cases, precisions,
  verification and consumers. Missing harnesses are blocked coverage; creating
  recipes requires explicit scope. Review transitive consumers with operator-perf.
- Require a positive **user-specified qualification time budget** before timing.
  Choose existing reduction, FFT, eigensolver and irregular representatives,
  subject to coverage and cost; honor explicit case overrides. Report forecasts
  and evaluate affordable sampling levels. Exhaustion leaves unresolved scopes.
- Select from calibration, freeze identity and budgets, then validate using
  disjoint runs. Representative qualification covers measured cases only.
  Calibrate complete operator/consumer matrices before production acceptance.
- Collect native correctness and repeat-output evidence with clean baselines.
  Keep blocked cases visible while proceeding with independent qualified scopes.
- Default to **assessment-only**. Before production edits, present one
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

Respect the caller's worktree or single-checkout choice. Keep immutable builds,
archived runtime dependencies, checkpoints and active notes in ignored session
paths. Preserve user edits and historical evidence. Version-2 evidence supports
historical reporting/replay; fresh version-3 qualification and confirmation are
required for new production sessions. No implicit migration, CPU fallback,
threshold relaxation, clock changes, skill installation or external publication.
