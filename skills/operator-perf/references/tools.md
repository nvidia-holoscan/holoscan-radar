# Tool Interface

Run host helpers with Python 3 (standard library only). They do not depend on
the disabled production Python bindings:

```bash
python3 performance/optimization/radar_perf.py --help
python3 -m unittest discover -s performance/optimization/tests -v
```

Keep session files in an ignored directory, for example
`build-optimization/session/`. Use absolute paths for artifacts. JSON commands
use argument arrays, never shell strings. A container command must mount the
artifact directory at the same absolute path so `--output` reaches the host.

## Workflow And Tool Compatibility

The default [engineering acceptance](acceptance.md) uses reviewed decision
records and raw source-bound harness/native evidence. The automated
`radar_perf.py` acceptance state machine currently implements **formal version
4**, not that engineering judgment. There is no new version-5 option. Do not
pass exploratory screens as formal speed gates or imply automated engineering
acceptance support. A manual engineering record must preserve identities,
commands/settings, exit status, raw reports, correctness and decision evidence.

When formal statistical guarantees are needed, select version 4 explicitly and
follow the [command guide](../../../performance/optimization/CONFIRMATION.md).
Its randomized four-run blocks, control review, replay and declared statistical
rules apply to that selected experiment. Host tests do not qualify a GPU.
Keep historical version-2/3 evidence unchanged; legacy examples below are not
default recommendations. Generic provenance, native checks and safe restoration
remain applicable across acceptance bases.

## Legacy Configuration And Identity

Propose this configuration, adjusting policies/cases to the actual operator;
the example is not permission to optimize or select exact numerical rules:

```json
{
  "operator": "clutter_filter",
  "mode": "optimize",
  "editable_scope": ["operators/fmcw/clutter_filter_op", "operators/common"],
  "policies": {
    "range_profiles_out": {
      "float": {"mode": "exact"},
      "double": {"mode": "exact"}
    }
  },
  "max_candidates": 20,
  "max_unsuccessful": 3,
  "wall_time_s": null,
  "target_ms": null,
  "measurement_policy_version": 3,
  "pairs": 20,
  "objective": "median_ms",
  "target_cases": ["baseline-small", "baseline-medium", "baseline-large"],
  "timing_precisions": {},
  "min_effect_pct": 2.0,
  "max_aa_delta_pct": 5.0,
  "max_baseline_spread_pct": 10.0,
  "metric_policies": {
    "cold_ms": {"max_aa_p90_delta_pct": 20.0, "max_baseline_iqr_pct": 30.0},
    "setup_ms": {"max_aa_p90_delta_pct": 15.0, "max_baseline_iqr_pct": 25.0},
    "override_ms": {"max_aa_p90_delta_pct": 25.0, "max_baseline_iqr_pct": 40.0}
  },
  "clock_configuration": null,
  "verification_cases": {
    "clutter_filter": ["small-baseline", "small-live-match-miss", "medium-baseline", "medium-live-match-miss", "large-baseline", "large-live-match-miss"]
  },
  "protected_metrics": ["median_ms", "cold_ms", "setup_ms", "override_ms"],
  "sampling": {
    "scope": "overrides",
    "sizes": "small,medium,large",
    "warmup": 5,
    "frames": 50,
    "repeats": 3,
    "process_mode": "isolated",
    "raw_frame_times": true,
    "early_frames": 50,
    "pairing_order": "adjacent_repeat",
    "aggregation": "median",
    "case_overrides": {}
  }
}
```

Tolerance policies use `mode: tolerance`, nonnegative `atol`, `rtol`, and an
evidence-backed `rationale`. Descriptor and discrete content remain exact.
All output ports, including unwired optional ones, must appear in policies and
snapshot exports. Do not reuse one operator's tolerances for another.

DetectionReduction may opt into `mode: detection-reduction-v1` with a rationale
for both ports and both precisions. Its [semantic policy](../../../docs/performance/optimizations/detection-reduction-validation.md)
requires fresh input witnesses and coupled checks; bounds cannot be adjusted
in configuration. Consumer policies remain separate.

```bash
python3 performance/optimization/radar_perf.py init \
  --config /absolute/config.json --confirmed --out /absolute/session.json
python3 performance/optimization/radar_perf.py fingerprint --root /absolute/checkout
```

`--confirmed` records a confirmation already received from the user; it does
not obtain or imply it. The session stores and checks the configuration hash.

Build manifests for A and B each require `operator` (stem), `operator_class`
(the report's operator name), `source_id` (fingerprint at build),
`git_commit` (matching report provenance), `recipe_sha256`, `sdk_image` (immutable
identity), `build_settings` (nonempty object), `artifacts` (absolute file paths
to SHA256 for executable and runtime libraries), and `command` (argv prefix
invoking the matching perf executable), plus `cwd` (existing absolute source
checkout directory). Docker wrappers also require an explicit container
`--workdir`; host cwd alone does not establish in-container git provenance.
Include CUDA/compiler/MatX identity in
build settings. Preserve separate complete builds, and validate manifest hashes
after building, before capturing the baseline. `measure` checks hashes before
each launch and writes reports and logs without overwriting prior runs.

```bash
python3 performance/optimization/radar_perf.py measure \
  --before /absolute/before-build.json --after /absolute/after-build.json \
  --config /absolute/config.json --session /absolute/session.json --out /absolute/ab-run
```

The default objective is lower `median_ms`; `achieved_frames_per_second` is
higher-is-better. Session target cases define the allowable candidate targets;
each predeclared candidate subset must all improve. Empty session targets allow
any measured objective case. All measured cases are checked
for regressions in the protected metrics. The agent enforces optional absolute
target and wall-time limits before launching further work; `candidate` also
checks attempt, dry, and elapsed budgets.
Passing `--session` also bounds paired measurements by the remaining confirmed
session time; without it, the configured wall-time bounds that invocation.
Additional shared-code consumers must show no regression; they need not improve.
The selected operator still must improve on its confirmed target cases.

`timing_precisions` maps operator stems to required timing precisions, defaulting
to both float and double. An explicitly confirmed `["float"]` allows a recipe
that times float only, but does not claim double-performance protection. Missing
required precision for any available case/metric is invalid evidence.

### Legacy Timing And Calibration

For version-2/version-3 replay only, the old workflow ran the same baseline
manifest on both sides before edits:

```bash
python3 performance/optimization/radar_perf.py calibrate \
  --pairs /absolute/aa-run/pairs.json --config /absolute/config.json \
  --out /absolute/calibration-review.json
```

Speed gates consume the **raw A/A pairs**, not this review summary. Calibration
is recomputed at acceptance and must match the current-best source, SDK/image,
recipe, hardware, sampling and complete case/metric inventory. Hashes bind all
raw reports. Directional A/A findings or exceeded delta/spread thresholds block
validation; do not call the suspected cause a confirmed SDK defect.

Decisions use distribution-free paired-median order-statistic intervals with
Bonferroni adjustment over every measured metric/case and the number of
confirmed operators. This gives conservative familywise 95% coverage under
independent, stationary paired runs; correlated cases do not require independence.
Reproducible 10,000-resample bootstrap 95% intervals remain descriptive.
An objective's adjusted lower bound must exceed the confirmed `min_effect_pct`;
any protected adjusted upper bound below zero rejects the candidate. No minimum
gain threshold excuses statistically supported regressions.

Twenty pairs are the initial proposal, not a promise of adequate power. If the
family cannot resolve an interval, its bounds are `null` and the verdict is
inconclusive. `minimum_resolvable_pairs = ceil(log2(40 * family_size))` is only
the minimum to obtain finite bounds, not a power calculation. Propose a larger
confirmed budget before further measurements; never cherry-pick repeat batches
until one passes. Calibration spread/delta thresholds and practical gain must
be selected before candidate results; no automatic tolerance widening occurs.

Before confirming the session, obtain a baseline preview report and run:

```bash
python3 performance/optimization/radar_perf.py budget \
  --report /absolute/baseline-preview.json --config /absolute/proposed-config.json \
  --out /absolute/budget-proposal.json
```

Run this for each affected operator's complete matrix and propose the largest
budget. It counts actual available metrics and the confirmed operator family,
and recommends at least third-order confidence bounds. It does not estimate
power for an arbitrary expected effect; inspect A/A evidence and expected gains
before confirmation. `measurement_policy_version: 2` distinguishes these
semantics. Old sessions cannot accept new candidates under changed rules.

`metric_policies` sets `max_aa_p90_delta_pct`, `max_baseline_iqr_pct` and optional
`min_effect_pct` separately for each metric. The legacy-named global fields
`max_aa_delta_pct`/`max_baseline_spread_pct` supply **P90/IQR** defaults only in
legacy versions 2/3; they no longer gate on extrema. Cold/setup/override proposed defaults
are examples to confirm, not permission to increase thresholds after results.
Extremes and stall counts remain diagnostics, and all observations enter the
familywise comparison. Minimum gain never waives a supported regression.

`sampling.process_mode: isolated` discovers matching case lists on both builds,
then launches one fresh process per case/repeat with `--cases ID --repeats 1`.
It retains every child report and recomputes aggregates during acceptance.
Scalar metrics are medians of repeat report metrics; `max_ms` is the worst
repeat maximum, not a median. This is not pooled per-frame percentile inference.
`cases: "id-a,id-b"` optionally selects an explicit confirmed subset. Bulk mode
is available only as a confirmed measurement context; run position probes first.

Clocks are read outside timed runs and recorded in pair indexes. An optional
`clock_configuration` object binds the preflight's sysfs governor/bound map.
Changing that map during a run, or between calibration and A/B, invalidates
evidence. Empty telemetry is unknown; use platform-specific read-only evidence
to confirm unsupported controls. No helper modifies clocks or power settings.

`insufficient_pair_budget` flags unresolved intervals even on a stable A/A run;
propose a larger budget before editing. A speed gate with any unresolved
protected interval cannot pass, including a consumer-only gate.

## Native Verification

Build `radar_verify_operator` with performance and testing enabled in the pinned HSDK 5
container. It uses native fixtures and the recipes' three profile shapes, both
precisions, multiple frames, a live match/miss, and range optional-output modes:

```bash
./build/performance/verification/radar_verify_operator clutter_filter \
  performance/operators/fmcw/clutter_filter/recipe.yaml /absolute/snapshots.json
./build/performance/verification/radar_verify_operator fmcw_range_processor \
  performance/operators/fmcw/fmcw_range_processor/recipe.yaml /absolute/range.json
./build/performance/verification/radar_verify_operator fmcw_range_doppler_processor \
  performance/operators/fmcw/fmcw_range_doppler_processor/recipe.yaml /absolute/fused.json
```

The exporter records actual output sample sequence, capture time, and clock
from each native snapshot, plus stable structured SDK diagnostics. Never
substitute input timestamps for observed output timestamps. Clock IDs are
plan-local: these fixtures use the SDK's sole default `fixture_clock`; new
fixtures must establish matching clock-catalog semantics before comparing IDs.
Review retained runtime logs separately for operator warnings and errors.
Other live ports, cached transitions, boundary/empty cases,
descriptor outputs, and unsupported operators require additional native fixture
coverage according to the candidate's affected surface.

Run the GPU-required qualification separately from timing:

```bash
ctest --test-dir build/holoscan-radar -R '^radar_perf_native_verification$' \
  --output-on-failure
```

It repeats the reviewed inventory across both precisions and three sizes, checks observed
coordinates and live fallback, covers wired/suppressed/unwired range outputs,
and rejects injected value and temporal mismatches. Each invocation preserves
fresh artifacts in `performance/verification/qualification/run-*` under the
build tree. Qualification is not a candidate acceptance artifact; use the
source-bound `capture` workflow below for optimization sessions.

`performance/optimization/fixture_inventory.json` lists case IDs by operator;
qualification expands both precisions and compares exact identities, not magic
case counts. Extend it alongside native fixtures, or pass `--inventory PATH`
to `qualify_native.py`. The current inventory has ClutterFilter baseline/live
mean-subtraction and split/fused range-stage baseline, Hann window, padding,
dechirp, real-IF, four-channel, live nfft/window/waveform match-then-miss, and
wired/unwired/suppressed optional-axis cases. There are 12 ClutterFilter cases
and 48 cases per range-stage operator across both precisions. Cached/unseen
configuration revisits, boundary/error cases, other live ports, and broader
Doppler behavior still require additional coverage. Prior session artifacts
are not evidence for the current checkout. Build all three operator targets
to enable the bundled native verifier.

Snapshot schema 1 contains `byte_order: little`, `fixture_id`, and `cases`.
It also identifies `operator` and `source_id`. Use `capture --build PATH --out
PATH` with a hashed verification build manifest to run the native executable
and stamp its build source identity. Its `command` argv ends with the operator
and recipe path; the helper appends the output path. Capturing raw values
directly does not establish source identity. Reusing original snapshots as
candidate snapshots cannot pass a candidate's source check.
Each case has `id`, `precision`, `termination`, `diagnostics`, `coverage_gaps`,
and `outputs`. Each output has `wired` and ordered `emissions`. Tensor emissions
have `kind: tensor`, `sequence`, actual `capture_time_ns` and `clock_id`,
`memory_kind`, `shape`, `dtype`, and `data_base64` of logical tensor bytes.
Descriptor emissions have the same emission metadata, `kind: descriptor`, and
canonical field `value` instead of tensor fields. Padding and addresses are not
descriptor values. Only supported dtype formats are accepted; add explicit
handling rather than coercing unknown types. Repeat baseline captures first.

## Candidate Evidence And Rollback

Every gate artifact has `kind` (contract, coverage, tests, equivalence, speed),
boolean `passed`, and an identical `context`:

```json
{
  "candidate_id": "candidate-1",
  "confirmed_config_sha256": "the hash stored in session.json",
  "before_source_id": "current-best source fingerprint",
  "after_source_id": "candidate source fingerprint",
  "affected_operators": ["clutter_filter"],
  "candidate_plan_sha256": "hash of the predeclared plan artifact"
}
```

Supply that object as `--context` to comparison commands. Review/test gates
also require `evidence`: include contract diff review, complete coverage
inventory, actual test commands/status/logs, and every affected consumer.
Comparisons include input paths/hashes and are recomputed during acceptance.
Failed/unrun gates can be recorded with `passed: false` and an explanation.

Register a plan before editing with `plan --session PATH --record PATH --root
CHECKOUT`. Its JSON contains `id`, `before_source_id`, `mechanism`, nonempty
`target_cases`, and positive `expected_gain_pct`. The helper checks the checkout
fingerprint, current best, confirmed target set, and stores the artifact hash.
Include `candidate_plan_sha256` in every gate context. Declarations are immutable;
changed mechanisms or targets require a new candidate declaration before edits.

```bash
python3 performance/optimization/radar_perf.py compare-outputs \
  --before /absolute/baseline.json --after /absolute/candidate.json \
  --policies /absolute/policies.json --context /absolute/context.json \
  --out /absolute/equivalence.json
python3 performance/optimization/radar_perf.py compare-timing \
  --pairs /absolute/ab-run/pairs.json --config /absolute/config.json \
  --calibration /absolute/aa-run/pairs.json --plan /absolute/candidate-plan.json \
  --context /absolute/context.json --out /absolute/speed.json
```

Comparison exit codes: 0 passes, 1 valid evidence fails the gate, 2 invalid or
missing evidence. Keep failed artifacts and do not reinterpret exit 2 as success.

Candidate records have `id`, `summary`, `files`, `cases`, `affected_operators`,
`before_source_id`, `after_source_id`, and `gates`, mapping each gate name to its
JSON path. Review/test gates must also list `operators` matching every affected
consumer. `cases` declares affected behavioral case IDs, not objective targets
or permission to omit measured cases; acceptance unions it with actual gate
inventories when checking bug conflicts. Shared comparisons use `aggregate --kind equivalence|speed --context
PATH --checks PATH... --out PATH` to combine independently verified consumer
results. Acceptance rechecks the children and requires every affected operator.
Confirm each extra operator's `consumer_policies`, `consumer_targets`, and
`verification_cases` in the session config before changing shared code.
By default, consumers use the common confirmed sampling settings. A fresh
version-3 session may declare `consumer_sampling`, keyed by an operator already
in `consumer_policies`. Each entry resolves its own complete protocol, with
explicit unique comma-separated `cases`; it does not inherit primary case IDs.
For example, expansion cases differ from FeatureAssembly's score/AoA cases:

```json
{
  "consumer_sampling": {
    "feature_assembly": {
      "cases": "score-aoa-pack-small,score-aoa-pack-medium,score-aoa-pack-large",
      "warmup": 5,
      "frames": 50,
      "repeats": 3
    }
  }
}
```

Pass the same confirmed config to `measure` and comparison commands. Collection
and replay select the protocol using the build's operator identity; confirmation
hashes bind these selections. An absent map preserves historical behavior.
Version-2 sessions cannot opt into this field; start and confirm a fresh
version-3 session instead of modifying historical evidence. Acceptance checks
bug conflicts against actual measured/verified cases, not only declared cases.

```bash
python3 performance/optimization/radar_perf.py checkpoint --root /absolute/worktree \
  --files operators/fmcw/clutter_filter_op/clutter_stage.hpp --out /absolute/checkpoint.json
python3 performance/optimization/radar_perf.py candidate \
  --session /absolute/session.json --record /absolute/candidate-record.json
python3 performance/optimization/radar_perf.py seal --root /absolute/worktree \
  --files operators/fmcw/clutter_filter_op/clutter_stage.hpp --out /absolute/candidate-state.json
python3 performance/optimization/radar_perf.py restore \
  --checkpoint /absolute/checkpoint.json --candidate-state /absolute/candidate-state.json
```

List every candidate file, including not-yet-created paths. Checkpoints include
previously accepted edits and permissions. Restore verifies the sealed state
before changing files, refuses intervening edits and symlinks, and never uses
git reset/checkout. Store artifacts outside those files. It does not automatically
undo edits outside the listed set; audit the full diff before/after each candidate.

## Profiling, Bugs, And Reports

`profile --capture PATH --tables CSV... --started-at UNIX_SECONDS --source-id ID
--out PATH` records fresh nonempty Nsight artifacts and identifies CUDA API rows
for inspection. It does not prove that those rows recur in steady frames.

```bash
python3 performance/optimization/diagnostics.py preflight --out /absolute/environment.json
python3 performance/optimization/diagnostics.py phases \
  --sqlite /absolute/capture.sqlite --started-at UNIX_SECONDS \
  --source-id SOURCE --case CASE_ID --repeat 0 --warmup 5 \
  --timed-output PORT --out /absolute/phases.json
```

Phase exports must be fresh and scoped to one operator/case. Select `--global-tid`
if needed; unsupported schemas or missing NVTX/runtime evidence fail explicitly.
Output is attribution only, never acceptance timing. Temporary `op.*` marks
are optional and must be removed before acceptance runs.

Use the explicit harness session labels, not thread/YAML-order guesses. The
`after_timed_emit_median_us` field is CPU-side attribution, not evidence that
the tail is outside the sink's CUDA dependency chain. `capture` owns `<out>.log`;
do not redirect its own stdout/stderr to that same path or append to old captures.

See [workspace safety](workspace.md) for scratch restore receipts and relocated
archive audits. Gate template/manifest generation and a one-command candidate
runner remain future conveniences; hand-written evidence still requires review.

`bug --session PATH --record PATH` appends a bug with `title`, `expected`,
`observed`, `baseline_reproduction`, `affected_cases`, `blocks_validation`, and
`evidence`. Report it to the user immediately. Keep findings unresolved in this
session; resolving them is separate work. Independent candidates may continue.

```bash
python3 performance/optimization/radar_perf.py finish --session /absolute/session.json \
  --reason 'candidate limit' --coverage-gaps 'unsupported boundary capture' \
  --findings 'remaining launch overhead' --cumulative-timing /absolute/final-vs-original.json
python3 performance/optimization/radar_perf.py report --session /absolute/session.json \
  --out /absolute/report.md
```

Final-versus-original evidence must use a separate clean paired measurement.
For legacy version-2/3 sessions, also supply
`finish --calibration /absolute/original-aa/pairs.json`. Version-4 sessions use
`finish --control-review /absolute/control-review.json` instead.
Do not multiply intermediate percentages. Reports preserve suspected bugs first
and explicitly identify absent cumulative evidence.

## Version 3 And Campaigns

Legacy version-3 production acceptance requires explicit version 3 and a hash-bound full-case
`benchmark_admission` prepared by the campaign tool. Version 2 remains readable
for reporting and historical gate replay; its evidence is never upgraded in
place. New formal sessions use version 4; default engineering experiments follow
the separate acceptance reference above.

Run `radar_perf.py campaign --help` for discovery, qualification, baseline,
session preparation, disposition, status, resume and reporting commands. Read
[qualification](qualification.md) and the campaign skill's
[configuration reference](../../operator-benchmark/references/configuration.md)
for budget and admission requirements. `measurement.py`, `qualification.py` and
`campaign.py` coordinate evidence; they do not edit production operators.

Per-case overrides specify frame/repeat counts, are resolved before execution
and are sealed in `run_settings.sampling_by_case`. Raw one-repeat children
record chronological frames and repeat offsets. Pair order alternates adjacent
A1/A2 executions per repeat and pair; aggregates retain every child reference.
`paired_precision.py OPERATOR BUILD_DIR ...` runs separate float/double
executables using archived runtime libraries, verifies matching inventories and
retains both child reports. Include the wrapper and runtime hashes in manifests.
