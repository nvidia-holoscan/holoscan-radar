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

## Configuration And Identity

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
  "max_candidates": 5,
  "max_unsuccessful": 2,
  "wall_time_s": null,
  "target_ms": null,
  "pairs": 10,
  "objective": "median_ms",
  "target_cases": ["baseline-small", "baseline-medium", "baseline-large"],
  "verification_cases": {
    "clutter_filter": ["small-baseline", "small-live-match-miss", "medium-baseline", "medium-live-match-miss", "large-baseline", "large-live-match-miss"]
  },
  "protected_metrics": ["median_ms", "cold_ms", "setup_ms", "override_ms"],
  "sampling": {
    "scope": "overrides",
    "sizes": "small,medium,large",
    "warmup": 5,
    "frames": 50,
    "repeats": 3
  }
}
```

Tolerance policies use `mode: tolerance`, nonnegative `atol`, `rtol`, and an
evidence-backed `rationale`. Descriptor and discrete content remain exact.
All output ports, including unwired optional ones, must appear in policies and
snapshot exports. Do not reuse one operator's tolerances for another.

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
invoking the matching perf executable). Include CUDA/compiler/MatX identity in
build settings. Preserve separate complete builds, and validate manifest hashes
after building, before capturing the baseline. `measure` checks hashes before
each launch and writes reports and logs without overwriting prior runs.

```bash
python3 performance/optimization/radar_perf.py measure \
  --before /absolute/before-build.json --after /absolute/after-build.json \
  --config /absolute/config.json --session /absolute/session.json --out /absolute/ab-run
```

The default objective is lower `median_ms`; `achieved_frames_per_second` is
higher-is-better. Confirmed target cases must all improve. An empty target list
protects and targets every case with that metric. All measured cases are checked
for regressions in the protected metrics. The agent enforces optional absolute
target and wall-time limits before launching further work; `candidate` also
checks attempt, dry, and elapsed budgets.
Passing `--session` also bounds paired measurements by the remaining confirmed
session time; without it, the configured wall-time bounds that invocation.
Additional shared-code consumers must show no regression; they need not improve.
The selected operator still must improve on its confirmed target cases.

## Native Verification

Build `radar_verify_operator` with performance and testing enabled in the pinned HSDK 5
container. It uses native fixtures and the recipes' three profile shapes, both
precisions, multiple frames, a live match/miss, and range optional-output modes:

```bash
./build/performance/verification/radar_verify_operator clutter_filter \
  performance/operators/fmcw/clutter_filter/recipe.yaml /absolute/snapshots.json
./build/performance/verification/radar_verify_operator fmcw_range_processor \
  performance/operators/fmcw/fmcw_range_processor/recipe.yaml /absolute/range.json
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

It repeats all 28 cases across both precisions and three sizes, checks observed
coordinates and live fallback, covers wired/suppressed/unwired range outputs,
and rejects injected value and temporal mismatches. Each invocation preserves
fresh artifacts in `performance/verification/qualification/run-*` under the
build tree. Qualification is not a candidate acceptance artifact; use the
source-bound `capture` workflow below for optimization sessions.

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
  "affected_operators": ["clutter_filter"]
}
```

Supply that object as `--context` to comparison commands. Review/test gates
also require `evidence`: include contract diff review, complete coverage
inventory, actual test commands/status/logs, and every affected consumer.
Comparisons include input paths/hashes and are recomputed during acceptance.
Failed/unrun gates can be recorded with `passed: false` and an explanation.

```bash
python3 performance/optimization/radar_perf.py compare-outputs \
  --before /absolute/baseline.json --after /absolute/candidate.json \
  --policies /absolute/policies.json --context /absolute/context.json \
  --out /absolute/equivalence.json
python3 performance/optimization/radar_perf.py compare-timing \
  --pairs /absolute/ab-run/pairs.json --config /absolute/config.json \
  --context /absolute/context.json --out /absolute/speed.json
```

Comparison exit codes: 0 passes, 1 valid evidence fails the gate, 2 invalid or
missing evidence. Keep failed artifacts and do not reinterpret exit 2 as success.

Candidate records have `id`, `summary`, `files`, `cases`, `affected_operators`,
`before_source_id`, `after_source_id`, and `gates`, mapping each gate name to its
JSON path. Review/test gates must also list `operators` matching every affected
consumer. Shared comparisons use `aggregate --kind equivalence|speed --context
PATH --checks PATH... --out PATH` to combine independently verified consumer
results. Acceptance rechecks the children and requires every affected operator.
Confirm each extra operator's `consumer_policies`, `consumer_targets`, and
`verification_cases` in the session config before changing shared code.
Each operator uses the common confirmed sampling settings. Acceptance checks
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

Final-versus-original evidence must use a separate clean paired measurement;
do not multiply intermediate percentages. Reports preserve suspected bugs first
and explicitly identify absent cumulative evidence.
