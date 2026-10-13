# Deployment benchmark tooling implementation

Implemented on `deustice/optimizations` in the existing checkout. No production
operators were optimized and no full baseline was rerun. Historical reports,
immutable builds and existing user edits remain preserved.

The updated [operator-perf skill](../../../../skills/operator-perf/SKILL.md)
qualifies measurement before expensive baselines and supports both worktrees
and a single checkout. The new
[operator-benchmark skill](../../../../skills/operator-benchmark/SKILL.md)
coordinates discovery, bounded qualification, frozen selection, held-out
validation, full-case baselines, confirmed sessions, dispositions and reports.

Version 3 keeps median aggregation and five warmup frames by default, captures
chronological raw frames and isolated repeat boundaries, alternates adjacent
A/A or A/B units, and protects first-50 median/p90/p99 alongside complete-window,
setup, cold and live metrics. Configurable per-case sampling is sealed before
execution. Optional exploratory means require frozen policy and fresh validation.
Existing statistical and output-equivalence gates remain in force; confidence
width and +/-1% precision are diagnostics. Version-2 reporting and gate replay
remain available; new production work requires fresh version-3 admission.

Maintained tools record cost/storage forecasts, cumulative qualification budgets,
source/build/runtime identities, owned-process cleanup, per-device locks,
hash-verified resume and separate report-publication provenance. Admissions
require native correctness and complete calibration, including declared
consumers. Changed sources or configurations invalidate reuse. Reports show
absolute A/A values, concrete limits, intervals, blockers, early/half-window drift,
stalls and partial coverage. The agent still performs production edits through
operator-perf; campaign commands perform no operator optimization.

## Validation

- 67 host tests pass, covering sampling, raw reconstruction, early windows,
  aggregation, family size, frozen/disjoint validation, explicit budgets,
  inventory, cleanup, resume, stale/shared-source rejection, native mismatches,
  consumer admission, reporting and version-2 compatibility.
- Fresh C++ tooling fixtures pass: 23 profile tests and 9 recipe tests.
- Relevant production fixtures pass: 22 ClutterFilter and 18 RangeProcessor tests.
- Five relevant C++ performance smokes pass. Both skills pass their validator.
- Pinned repository checks use the existing compatible Node 22 wrapper with
  unchanged hook versions. Check logs and final receipts are retained below.

The final bounded GPU exercise used the immutable HSDK 5 image
`sha256:d4ccdf30d119f555320cb629704a50614dab7cd0eecd1f7513481d52c0b7ddb2`
on Thor, with archived SDK runtime libraries and verified loader resolution.
Production, compiled C++ and recipe hashes remain unchanged. No clock, power,
affinity or profiling privilege changes were made.

| Operator | Precision runs | Raw measured frames | Final measurement phase |
| --- | --- | --- | --- |
| ClutterFilter | 48, float and double | 2,400 | 30.2 seconds |
| RangeProcessor | 48, float and double | 2,400 | 31.2 seconds |

Each operator exercised two calibration pairs, frozen selection and two disjoint
held-out pairs, with 50 frames and three isolated repeats. Resume replayed sealed
hashes without new measurement jobs. Final measurement phases took approximately
61 seconds; all measurement attempts, including preserved failed attempts and an
earlier successful smoke, totaled approximately 128 seconds. Relevant GPU/C++
fixtures took another approximately 25 seconds. This remained within the explicit
five-minute integration limit. Job time includes launch and orchestration and
does not measure agent working time.

Both operator scopes remain **validation blocked**: two pairs cannot resolve the
familywise intervals, and dispersion failures are retained. This exercises
orchestration and integrity; it does not qualify this deployment or replace the
existing baseline. See the [smoke report](smoke.md),
[ClutterFilter report](clutter_filter.md) and
[RangeProcessor report](fmcw_range_processor.md) for absolute A/A results and
thresholds. Missing profiling observations remain unknown; this exercise did
not collect new Nsight profiles.

## Findings and remaining work

Integration found and corrected YAML-recipe reference handling and container
Git provenance. The latter uses command-local trust for this checkout, without
changing Git configuration files. Failed artifacts were preserved, not upgraded
into evidence. No new production correctness findings arose in the verified
ClutterFilter/RangeProcessor scopes. Earlier RootMusic and DetectionReduction
findings, and the sampling pilot's unresolved timing regimes, remain unresolved;
see the [original baseline](../baseline.md) and
[sampling pilot](../sampling-pilot-1.md).

A deployment campaign still needs its own explicit qualification budget,
qualified complete matrices and consolidated confirmation before production
sessions. This implementation makes no full-pass runtime-performance claim.

Artifacts: `build-optimization/operator-perf/benchmark-tooling-implementation/`.
The final campaign lives in `integration-final/smoke/`; build/runtime provenance,
raw reports, frozen selections, process/budget ledgers, integration checks and
fixture/hook logs remain in that ignored directory. No commits, pushes, external
issues or skill installation were performed.

## Commit tracking

After this assessment, the user authorized modular commits on
`deustice/optimization-skills`. The harness, campaign tools, skills and reports
are tracked separately. Measurements retain their original branch and provenance;
no production optimization has begun and no changes were pushed.
