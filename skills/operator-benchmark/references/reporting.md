# Milestone Reports

Keep versioned summaries in `docs/performance/optimizations/<campaign>/`:
`baseline.md`, one report per operator and `final.md`. Raw notes, process logs,
profiles, checkpoints, budgets and manifests remain in ignored artifacts.
`campaign report --out ...` writes the campaign summary and adjacent operator
reports, checks evidence hashes and records publication provenance separately.
Run it only between sealed candidate sessions; do not mutate active identities.

## Running Overview

Create `overview.md` in the campaign report directory at discovery and keep the
same entry point through qualification, baselines, optimization and completion.
Update it between sealed sessions at each milestone, operator disposition,
material blocker or policy decision. It is an agent-maintained summary, not an
extra measurement receipt; the campaign report command does not generate it.
Use fresh publication provenance afterward and preserve immutable evidence.

Keep the first screen readable without opening raw artifacts:

- Current phase, last update, branch/session path, requested operator scope.
- Completed/remaining coverage, explicitly separating operators, workloads,
  precisions and qualified cases; do not mix denominators or combine campaigns.
- Retained production changes and directly validated gains, or explicitly none.
- Active blockers, policy status and budget status; distinguish historical
  limits and superseded proposals from current decisions.
- One concrete next action, the decision it should enable, dependencies and
  whether new budget or confirmation is needed. Do not request it preemptively.
- A compact milestone history linking detailed baseline, operator and final
  reports. Preserve history while replacing the current-state summary.

Show elapsed phase/job time only where recorded; label estimates and unknowns.
Do not make users reconstruct current status from a chronological chat or many
reports. At every milestone say whether evidence supports baseline collection,
a bounded experiment, a policy decision or an unresolved disposition.

## Detailed Evidence

Report the measured deployment/image/SDK, source and measurement identities,
branch/session paths, recipe matrix, precision, verification coverage and gaps.
Show absolute latency/throughput and setup/cold/live costs, early/full windows,
maxima/stalls, A/B effect intervals and dependent A/A–B/B control diagnostics
in absolute and relative units. Show screening versus fresh confirmation,
randomized order, independent block counts, the control interpretation review
and remaining uncertainty. There is no fixed new-policy dispersion limit; label
legacy limits only when replaying historical results. Confidence width and
+/-1% remain diagnostic.
Unknown telemetry remains unknown. Distinguish correctness, bias, insufficient
resolution, ordinary variability, unresolved confounding/regimes and budget
exhaustion. A non-significant control is not proof of reliability.

Include forecasts with assumptions, raw storage and actual phase durations.
Measurement-job durations include launch/orchestration; they do not measure
agent working time. Never infer an agent-versus-harness time percentage from
those durations alone.

After each operator session include retained/rejected candidates, mechanism,
casewise absolute/percentage results and intervals, protected regressions,
direct cumulative measurements, affected consumers, unresolved findings and
stop reason. A blocked attempt consumes the attempt budget, but does not add to
the unsuccessful streak. Reassess the actionable opportunity queue using
avoidable work, removable fraction, affected cases and attribution confidence.
No actionable queue is an assessment disposition, not a reason to invent edits.

The final table covers every requested operator as improved, assessed with no
validated improvement, or blocked. Show baseline/final timing and retained
changes where evidence exists. List unresolved suspected bugs explicitly, or
state none were found within verified scopes. Missing or unstable cases prevent
a claim that the full campaign validated those workloads.
