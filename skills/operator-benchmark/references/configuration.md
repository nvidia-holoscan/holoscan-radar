# Campaign configuration

Use explicit **measurement-policy version 4** and operator-perf's
[command guide](../../../performance/optimization/CONFIRMATION.md). The workflow is
**discover → correctness/baseline screen → bounded A/B candidate screen → freeze
→ fresh integrated confirmation → disposition/report**. Standalone A/A or B/B
campaigns are not prerequisites. Version-2/version-3 workflows remain available
for historical replay; do not implicitly migrate their sessions or evidence.

## Discover and prepare

```bash
python3 performance/optimization/radar_perf.py campaign discover --config /absolute/campaign-config.json --out /absolute/campaign
python3 performance/optimization/radar_perf.py campaign qualify --campaign /absolute/campaign
python3 performance/optimization/radar_perf.py campaign baseline --campaign /absolute/campaign
python3 performance/optimization/radar_perf.py campaign status --campaign /absolute/campaign
python3 performance/optimization/radar_perf.py campaign resume --campaign /absolute/campaign --phase baseline
python3 performance/optimization/radar_perf.py campaign report --campaign /absolute/campaign --out /absolute/reports/baseline.md
```

For version 4, `qualify` checks design/build identities without launching A/A
measurements. It is not deployment qualification. Baseline runs the provided
native correctness fixtures, compares two output captures and collects one clean
original-build screen per case. Sampling and coverage claims remain scoped to
the actual measured matrix; representative results never qualify omitted cases.

```json
{
  "root": "/absolute/checkout",
  "mode": "assess",
  "operators": ["clutter_filter", "fmcw_range_processor"],
  "configs": {
    "clutter_filter": "/absolute/clutter-config-v4.json",
    "fmcw_range_processor": "/absolute/range-config-v4.json"
  },
  "builds": {
    "clutter_filter": "/absolute/clutter-build.json",
    "fmcw_range_processor": "/absolute/range-build.json"
  },
  "baseline_budget_seconds": 300,
  "verification": {}
}
```

The budget and operators are examples, not authorization or universal defaults.
Require a positive caller-specified budget before new timing. The baseline
allocation is shared across all operators and persists across resume. Leaf
session budgets bound candidate measurement phases. Forecast correctness,
builds, screening and four-run confirmation separately; exhausted allocations
leave explicit unresolved scopes without automatic extensions.

Each operator config includes version 4, both-precision output policies,
verification cases, timing precision, targets, protected metrics, scope and
limits. Use explicit sampling case IDs, two confirmation runs per build and a
random seed as shown in the command guide. Confirm a consumer-specific protocol
when case IDs differ. `pairs` means independent confirmation blocks, not frames,
repeat medians or four independent cross-build pairings. `screen_pairs` bounds
exploratory A/B blocks separately. Derive confirmation resolution with the
version-4 `budget` command; no single count establishes power.

Record workload dimensions, variants, compute role and horizon. Relative
small/medium/large labels are not universal deployment sizes. Keep useful
small overhead cases without requiring them to satisfy a fixed percentage of
repeatability. No fixed deployment means scenarios are explicitly synthetic;
do not silently enlarge recipes or require deployment matching.

## Provenance and native coverage

Build manifests bind operator/class, compiled/source identity, recipe, immutable
SDK image, settings, cwd/argv, device and executable/runtime artifact hashes.
A reviewed `measurement_files` closure permits verified reuse after unrelated
report edits; otherwise the whole checkout identity must match. Source hashes
alone do not prove an older executable was compiled from those sources. Keep
build logs, runtime resolution and reuse receipts. Changed shared consumers
invalidate affected evidence closures.

`verification[operator]` supplies `correctness_command` and a hash-bound coverage
receipt. It binds source ID, complete native case inventory, `passed: true`,
reviewed `ownership`, `boundaries`, `analytic_reference`, `errors`, `live_fallback`
flags and nonempty hashed `evidence` from real fixtures. A non-applicable behavior
needs an explicit reviewed rationale, not an invented test. The tool executes
the command, captures both precisions twice and checks exact policies and case
coverage. Missing evidence blocks correctness admission; output tolerance is
never relaxed to fix timing uncertainty.

## Optimization handoff

Default to assessment. For an authorized optimization campaign, use
`mode: optimize` and present all concrete operator configurations together:
outputs, targets, protected matrix, consumers, editable scope, sampling,
statistical error budget, control review and stopping limits. Record existing
user confirmation; do not ask again while its scope remains valid.

```bash
python3 performance/optimization/radar_perf.py campaign prepare-sessions --campaign /absolute/campaign
python3 performance/optimization/radar_perf.py campaign prepare-sessions --campaign /absolute/campaign --confirmed
```

Version-4 `benchmark_admission` validates native correctness/coverage and affected
consumer native records; no A/A calibration artifact is needed. Use the generated
session configuration for subsequent commands so its admission reference and
confirmed hash remain intact. These commands do not edit production operators.

For each bounded candidate, operator-perf declares the plan, checkpoints sources,
checks exact behavior, screens A/B, freezes the candidate, then collects fresh
randomized balanced two-A/two-B confirmation. A/A and B/B controls share those
runs with A/B. The agent records an evidence-backed interpretation review;
this is not another user approval, an agreement-percentage test or permission
to override a demonstrated slowdown. Repeated control checks do not authorize
rerunning until a pass. Use validated error/stopping rules and remaining budget.

After an accepted change, refresh attribution and reassess the queue. There is
no mandatory current-best A/A refresh. Final gains require direct original/final
measurements with integrated controls; never multiply intermediate percentages.

```bash
python3 performance/optimization/radar_perf.py campaign disposition --campaign /absolute/campaign --operator clutter_filter --session /absolute/campaign/sessions/clutter_filter-session.json
python3 performance/optimization/radar_perf.py campaign reuse-build --build /absolute/preserved-build.json --out /absolute/new-binding
```

Publish only between sealed sessions and capture fresh provenance afterward.
Reuse verifies closure/runtime/context and preserves original artifacts. It does
not reinterpret old calibration, authorize new GPU runs or silently upgrade a
historical policy. Host-tool tests validate orchestration; real deployment
qualification and performance conclusions require actual GPU evidence.
