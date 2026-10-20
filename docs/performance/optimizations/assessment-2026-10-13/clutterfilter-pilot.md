# ClutterFilter workload qualification

Assessment only; no retained/rejected production changes or optimization gains. Branch: `deustice/optimization-skills`. Original compiled operators and runtime archive preserved.

| Scenario | Completed frame/repeat levels | Final disposition | Stop reason |
| --- | --- | --- | --- | --- |
| diagnostic | 50/3, 200/3, 50/7 | calibration_blocked | All declared affordable levels blocked |
| compact | 50/3, 200/3 | calibration_blocked_sampling_incomplete | Observed dispersion at measured levels; further declared levels unaffordable within remaining budget |
| larger | 50/3 | calibration_blocked_sampling_incomplete | Observed dispersion at measured levels; further declared levels unaffordable within remaining budget |
| stress | 50/3 | pass | Frozen protocol passed fresh held-out validation |

See [the complete pilot report](distribution-pilot.md) for casewise absolute timings, percentages, familywise intervals, thresholds, protected failures, distribution diagnostics and cost accounting. All timings cover source-to-synchronized-sink execution; no cumulative optimization gain is claimed.

Cases with selected protocols passed 30 fresh held-out pairs in both precisions at 50/3. Unselected cases retain their observed dispersion blockers. No universal noise allowance was introduced. Full variant/live/consumer matrices and new scenario-sized native correctness exports remain unmeasured. Existing C++ fixture evidence does not close those coverage gaps.

Artifact directory: `build-optimization/operator-perf/assessment-2026-10-13/distribution-pilot/`. No new suspected correctness bugs were observed in this pilot; existing RootMusic coverage questions are documented separately.
