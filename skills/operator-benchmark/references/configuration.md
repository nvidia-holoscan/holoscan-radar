# Campaign Configuration

All commands use host Python 3 and argv arrays. Start in an ignored directory:

```bash
python3 performance/optimization/radar_perf.py campaign discover --config /absolute/config.json --out /absolute/campaign
python3 performance/optimization/radar_perf.py campaign qualify --campaign /absolute/campaign
python3 performance/optimization/radar_perf.py campaign baseline --campaign /absolute/campaign
python3 performance/optimization/radar_perf.py campaign status --campaign /absolute/campaign
python3 performance/optimization/radar_perf.py campaign resume --campaign /absolute/campaign --phase qualify
python3 performance/optimization/radar_perf.py campaign report --campaign /absolute/campaign --out /absolute/reports/baseline.md
```

The qualification command performs exploration, immutable selection and held-out
validation. Baseline uses a validated family recommendation as a sampling
proposal and performs full-case correctness/calibration; it never assumes a
representative pilot qualifies other cases. `resume --phase baseline` replays
completed hash-bound evidence. Changing measurement identity requires a fresh
campaign; do not patch old manifests or budgets.

## Campaign JSON

```json
{
  "root": "/absolute/checkout",
  "mode": "assess",
  "operators": ["clutter_filter", "fmcw_range_processor"],
  "representatives": ["clutter_filter", "fmcw_range_processor"],
  "qualification_budget_seconds": 300,
  "configs": {
    "clutter_filter": "/absolute/clutter-config.json",
    "fmcw_range_processor": "/absolute/range-config.json"
  },
  "builds": {
    "clutter_filter": "/absolute/clutter-build.json",
    "fmcw_range_processor": "/absolute/range-build.json"
  },
  "exploration": {
    "frames": [50, 500, 2000],
    "repeats": [3, 7],
    "calibration_pairs": 10,
    "validation_pairs": null,
    "allow_mean": false
  },
  "verification": {},
  "baseline_budget_seconds": null
}
```

The sample budget is an example, not a default or authorization for measurement.
Require the caller to specify a positive budget. It is divided among selected
representatives and persists across resumes. `validation_pairs: null` uses the
matrix-derived proposal with a 30-pair floor; confidence resolution is checked,
not presumed. An explicit smaller smoke budget can exercise orchestration while
remaining blocked for qualification.

`representative_cases` optionally maps operator names to comma-separated case IDs.
Default pilots use `baseline-small`; representative selection is separate from
the full baseline matrix in each operator configuration.

Operator configs follow [operator-perf tools](../../operator-perf/references/tools.md)
with explicit `measurement_policy_version: 3`. Their `sampling.cases` selects the complete baseline matrix;
use campaign `representative_cases` to narrow qualification independently. The campaign carries the number of harness-backed requested operators
into familywise budget calculations. Review affected consumers explicitly.
`qualification_cost_estimate` may specify `case_count`, `precisions`,
`launch_seconds` and `frame_ms`; these are forecasts, never evidence. Default
forecast assumptions are labeled and refined by observed job durations.

A build manifest binds operator/class, source ID, git commit, recipe hash,
immutable SDK image, build settings, absolute cwd, argv command, device ID and
hashes of executable/runtime artifacts. Supply `measurement_files` as a reviewed
path/hash closure containing production sources, shared consumers, harnesses,
recipes and measurement tools to permit reuse after unrelated report edits.
Without that closure, whole-checkout source identity must match. Freeze the
closure with build provenance; an arbitrary current-source hash list cannot
prove an older executable was built from those sources. Record the build log,
resolved runtime libraries and a source-verification/reuse receipt separately.

`verification[operator]` supplies `correctness_command` and `coverage` (an
absolute path/SHA256 reference). Coverage receipts bind the build source ID,
`passed: true`, the complete confirmed native `cases`, and reviewed true flags
for `ownership`, `boundaries`, `analytic_reference`, `errors`, `live_fallback`.
These flags are claims backed by fixture logs, not substitutes for tests.
The tool runs the supplied fixtures, captures two native exports and requires
exact comparison under the declared per-port policies. Missing evidence blocks
baseline admission. No correctness tolerance is silently weakened.

## Optimization Handoff

Set `mode: optimize` only with authorization. After qualified full baselines:

```bash
python3 performance/optimization/radar_perf.py campaign prepare-sessions --campaign /absolute/campaign
```

Review every generated `sessions/*-config.json` together, including targets,
output policies, full protected matrix, precision, consumers, scope and limits.
After recording user confirmation, use `prepare-sessions --confirmed` (or its
`optimize --confirmed` alias) to initialize operator-perf sessions. This command
performs no production edits. The agent follows operator-perf for each bounded
candidate and refreshes current-best calibration after retained changes.

Seal each operator session with operator-perf `finish`, then register it:

```bash
python3 performance/optimization/radar_perf.py campaign disposition --campaign /absolute/campaign --operator clutter_filter --session /absolute/session.json
```

An improved disposition requires direct original/current-best cumulative timing.
Publish reports between sealed sessions. Shared-source edits invalidate affected
build/evidence closures; rebuild and refresh required matrices before proceeding.

After publishing reports, create a fresh build-provenance binding only after
verified closure/runtime checks:

```bash
python3 performance/optimization/radar_perf.py campaign reuse-build --build /absolute/preserved-build.json --out /absolute/new-binding
```

The new receipt distinguishes compiled-source identity from current whole-checkout
provenance, preserves the original manifest and rejects changed source/runtime
hashes. It does not upgrade old measurements or authorize reusing calibration
under changed policies or contexts. Rebind evidence only through a reviewed
receipt and replay; otherwise collect fresh calibration. Affected consumer
correctness and full-matrix calibration must also pass before session admission.
