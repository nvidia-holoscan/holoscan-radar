# Milestone Reports

Keep versioned summaries in `docs/performance/optimizations/<campaign>/`:
`baseline.md`, one report per operator and `final.md`. Raw notes, process logs,
profiles, checkpoints, budgets and manifests remain in ignored artifacts.
`campaign report --out ...` writes the campaign summary and adjacent operator
reports, checks evidence hashes and records publication provenance separately.
Run it only between sealed candidate sessions; do not mutate active identities.

Report the measured deployment/image/SDK, source and measurement identities,
branch/session paths, recipe matrix, precision, verification coverage and gaps.
Show absolute latency/throughput and setup/cold/live costs, early/full windows,
maxima/stalls, P90 paired differences and IQR against concrete limits, confidence
intervals and calibration disposition. Confidence width and +/-1% are diagnostic.
Unknown telemetry remains unknown. Distinguish correctness, bias, insufficient
resolution, dispersion, unresolved regimes and budget exhaustion.

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
