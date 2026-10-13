# Operator performance characterization

The C++ tools measure one HSDK 5 operator in a source → operator → terminal-sink
Graph. Results are diagnostic measurements, not timing pass/fail thresholds.
The terminal sink synchronizes the GPU. Run the executable commands in this
guide **inside a development container**, after configuring a performance build
as shown in the [developer guide](../DEVELOPER.md#performance-characterization).
A process warmup precedes the matrix;
each scenario then runs three independent Graph sessions by default, with five
warmup frames and fifty steady frames per session. The report separates setup,
first-frame, steady latency, changed-frame latency, achieved frames/s, and
observed source cadence. Source publication and Graph scheduling remain part
of these end-to-end numbers.

The HSDK 5 unit-test harness is useful for staging valid inputs and Exact
sidebands, but it captures bounded output snapshots with extra copies and does
not expose paired per-frame timestamps. The performance runner keeps the Graph
wiring and records source and synchronized sink times directly.

## Layout

Each operator's YAML recipe and optional typed C++ adapter live together under
`performance/operators/<group>/<operator>/`. The group matches `operators/` and
`tests/cpp/`; the shared C++ runner lives under `performance/harness/cpp/utils/`.
`performance/CMakeLists.txt` is the explicit registration point. Executables
still appear at `build/performance/cpp/perf_<operator>`. The C++ YAML suites are
the maintained operator performance assessments.

## C++ recipe boundary

The common driver loads and validates **every** case before scope selection. It
owns process warmup, independent Graph runs, sampling, and schema-v3 reports.
The registered recipe runner stages one, two, three, or six data tensors,
constructs the operator through the pinned HSDK 5 authoring registry, resolves
named ports, wires the source and timed/drained sinks, and records paired
source/synchronized-sink timestamps. The recipe declares deliberately unwired
outputs, too. HSDK port records check direction, payload identity, and authored
Tensor rank, dtype, and memory constraints during Graph construction.

Schema version 2 adds named axes and one fixed recipe to the existing explicit
case list. The recipe names input ports, shapes, dtype, device placement,
generators, canonical Params, constructor extents or a named sizing policy,
outputs, and live ports with physical names and typed payload codecs. The
shared runner stages and connects only the live port named by a selected case;
a scaled-axis codec declares its source axis in the recipe. A bounded
`derived_axes` rule supports products and rounded percentages for related
fixtures. Recipes remain single-operator declarations rather than a general
Graph wiring language. Case IDs, scope tags, order, and `{profile}` expansion
retain their earlier behavior. All 26 in-tree C++ suites use schema version 2;
the loader still accepts version 1 for compatibility.

### Adding a C++ operator

1. Create `performance/operators/<group>/<operator>/recipe.yaml` with three
   valid sizes, a version-2 recipe, and curated cases. Declare every documented live port and
   cover it alone at each size in Overrides. Run `--validate` and `--list-cases`;
   both validate unselected cases without authoring a Graph.
2. For direct `MaxTensorExtents`, built-in tensor generators, canonical scalar
   or sequence Params, and registered scalar or host-vector live payloads, register the group/name and
   operator type in `performance/CMakeLists.txt` with `radar_add_yaml_perf`.
   The generated entry point uses `YamlStandardTraits`; no operator-owned
   C++ source is needed. `test_perf_yaml_standard` exercises this path with a
   real two-input operator and a one-frame scalar live case.
3. If the locked contract needs a descriptor builder, related-input fixture,
   cross-field check, derived bounds, payload codec, throughput unit, or
   report postprocessing, put `adapter.cpp` beside the recipe. That file owns only those semantics;
   source publication, Graph assembly, timing, and report buckets stay shared.
   Add its `group/name` to `HOLOSCAN_RADAR_PERF_OPERATORS`; CMake registers
   the executable and Minimal GPU smoke.

The original five pilot tools keep typed adapters beside their recipes for their
contract-specific parts:
DDM power fold's DDM descriptor and divisibility, SnapshotGather's related
indices and window, Beamscan's geometry/grid descriptors, FMCW range's FFT
bounds and cached/unseen transitions, and CFAR's variant-dependent limits and
calibration summary. Their executable names and CLI flags remain unchanged.
The current HSDK registry cannot register local enum live payloads as external
topics; the Beamscan and FMCW adapters use a typed flow connector for them.

### Adapter size and new-operator walkthrough

The original five pilot adapters fell from 1,543 to 1,114 lines in the first
stage. Migrating the other 21 reduced their combined adapters from 5,839 to
2,278 lines (3,561 fewer, about 61%). The shared registry, recipe, fixture,
and angle helpers are additional code. The remaining custom code constructs
contract-valid Params and data, validates cross-field rules, computes derived
bounds, and supplies operator-specific throughput units. Shared code owns case identity and value merging, staged input publication,
Graph assembly, on-demand live dispatch, cached-transition warmup, paired
timestamps, and rate conversion. The typed adapters read named YAML axes and
validate constraints instead of duplicating case IDs or size ladders.
SnapshotGather enum Params still need typed conversion: the pinned HSDK 5
parameter materializer rejects a YAML string for `doppler_edge`. FeatureAssembly
needs a typed constructor hook for its fixed feature list; MultiSourceExpansion
keeps a related six-input fixture builder. A standard operator with registered
components follows the YAML-only path regardless of these exceptional adapters.

To assess a new standard operator: use its locked
contract to choose three profiles; declare one fixed recipe with direct bounds
and built-in generators; author explicit cases; register its HSDK operator type
through `radar_add_yaml_perf`; run validation, Minimal, and relevant Overrides
smokes. Add an `adapter.cpp` only when a declared component cannot encode
a valid contract value or sizing rule.

## Precision and native recipe snapshots

Each of the 26 typed adapters builds two executables: `perf_<operator>` for
float and `perf_<operator>_double` for double. Precision remains a compile-time
operator instantiation. Both executables consume the same canonical recipe;
the double build selects paired float64/complex128 input representations and
reports `precision: double`. Discrete tensors and precision-agnostic live
configuration retain their declared types.

Registered recipes also accept `--verify-output <path>`. This runs the selected
cases through `holoscan::testing::NativeOperatorTest`, outside the timing Graph,
and exports owning output snapshots with observed sequence, capture time,
clock, shape, dtype, and bytes. Descriptor content is exported separately from
transport storage. One-frame live cases include a match and following miss;
cached cases additionally prime the changed configuration. Deliberately unwired
outputs remain named in the export. This recipe coverage complements the
analytic, boundary/error, optional-wiring, and ownership tests; it does not
replace them. Snapshot limits come from authored output bounds.

```bash
./build/performance/cpp/perf_clutter_filter_double --scope overrides \
  --output clutter-double.json
./build/performance/cpp/perf_clutter_filter --scope overrides \
  --verify-output clutter-native-float.json
```

Native staging copies imported generator buffers into SDK-owned Tensor storage
with a producer-readiness contract. Those copies never enter timing runs.
Floating data and supplied validity masks vary across three frames to expose
stale output reuse; indices and labels retain valid fixture values. Capture both
precision executables and retain their individual exports when assembling an
optimization session's complete verification evidence. Campaign captures use a
fresh process per case and precision, independently of timing runs.

## Raw frame timings

Pass `--raw-frame-times` to include chronological `raw_steady_ms` and
`raw_steady_run_offsets` in each compute row. The offsets delimit individual
Graph repeats in the concatenated steady samples. Warmup, changed and reversion
frames are excluded using the same rules as the existing summary statistics.
Export and copying happen after Graph execution; ordinary reports omit these
fields. Use `--repeats 1` in a fresh process for each isolated repeat. Increasing
`--repeats` inside one executable instead pools multiple Graph sessions within
that process. Raw frames retain their order and are not independent process
replicates.

## YAML profiles and scopes

All C++ tools read their colocated `recipe.yaml` automatically from the
executable's source checkout, even when that checkout is mounted at a
different path from its build. No YAML path flag is needed. Each suite names
small, medium, and large shapes and explicitly lists parameter and one-frame
`[live]` cases. Tags select cases without generating a Cartesian product.
`minimal` includes default cases at every size; `limited` adds important
parameterizations; `overrides` includes Limited and every live port alone at
every size; `moderate` expands Limited with selected live cases; `thorough`
includes all cases. An override source emits its sideband once, on the changed
frame, so the following frame takes Params through Exact + SafeDefault.
Thorough suites include primed revisits of live configurations: frame zero
publishes the changed configuration during warmup, subsequent misses take Params,
and the measured override revisits the changed value. Unseen cases make the same
change without priming. Cached cases require at least one warmup frame; the label
describes prior configuration use, without asserting an implementation cache hit.

Cases may set `output_mode` to `all` or `minimal`. `all` connects configured
unwired Tensor outputs; `minimal` disconnects only the recipe's explicit
`optional_tensor_outputs`, retaining required outputs and descriptor drains.
An optional paired timed output then becomes unwired. Default `configured`
wiring comes from the base recipe. These variants keep public port names and
output policies explicit in native exports.

Check every YAML case and typed adapter conversion before running a Graph:

```bash
./build/performance/cpp/perf_clutter_filter --validate
./build/performance/cpp/perf_clutter_filter --list-cases --scope overrides
```

`--list-cases` prints selected case IDs, sizes, and live ports in YAML order.
Both commands validate unselected cases too. They do not create a Graph or
query a CUDA device; the executable still needs its linked HSDK/CUDA libraries.
Errors name the suite file and, for adapter conversion failures, the case ID
and YAML line.

Example in the pinned HSDK 5 GPU container:

```bash
./build/performance/cpp/perf_clutter_filter --scope overrides \
  --output clutter-overrides.json
```

`--warmup`, `--frames`, and `--repeats` override the sampling budget.
`--sizes small,medium` filters migrated suites. JSON schema version 3 adds
scope, steady sample count, changed live port, achieved frames/s, and source
cadence; old schema-version-2 readers must be updated.

All 26 C++ operator tools use these YAML suites. The legacy Python operator
performance tools were removed because their separate scenario matrices could
not run against the pinned HSDK 5 image, which does not build the Python
bindings. The production Python package and its tests are separate from this
harness. A future Python Graph measurement tool should consume the same recipes
when bindings become available.

The C++ recipes include CFAR full-axis peak suppression on rank-four inputs
and FeatureAssembly score/AoA feature packs at all three sizes. Feature packs
stage only the inputs required by their selected feature list. DopplerNotch and
DDM's optional velocity-axis fixtures use a uniformly spaced center-zero
physical axis rather than a power-map pattern. SubarraySelection also covers
supplied bool and uint8 masks, with mask publication before the snapshot trigger,
and optional mask-output wiring.

## Automated optimization

The repository [operator-perf skill](../skills/operator-perf/SKILL.md) provides
assessment, single-candidate fixes, and bounded iterative optimization.
It preserves locked contracts and behavior, confirms numerical policies per
output and precision, compares clean alternating A/B measurements, and reports
suspected bugs without silently fixing them. Shared changes require validation
of all affected consumers. Timing remains diagnostic in CTest; optimization
acceptance is a separate confirmed session decision.

The default loop is at most **20 attempted candidate turns**, stopping after
**three consecutive unsuccessful candidates** (including inconclusive timing).
Calibration-blocked validation is reported separately. Before edits, calibrate
each current-best build against itself, declare candidate targets/mechanism,
and confirm practical gain and timing precision coverage. Default protection
requires both precisions; explicitly scoped float-only timing cannot establish
double-performance protection. Acceptance uses conservative familywise paired
median intervals; descriptive bootstrap intervals alone do not decide a gate.
Twenty proposed pairs may lack power for a large matrix; use the baseline
matrix's `budget` proposal and confirm adequate sampling
before rerunning, rather than accepting a favorable repeat. See the tool
interface for the updated plan, calibration, manifest `cwd`, and diagnostics
commands. Existing sessions need fresh confirmation/evidence for the new gates.

Version-2 measurement policies use isolated case/repeat processes by default,
robust P90/IQR calibration with metric-specific thresholds, and recorded clock
configuration. Extreme stalls remain visible; optional p90/p99/max protection
is explicit. Shared `--cases` and `--reverse-order` flags enable baseline
position probes, while NVTX session labels support explicit case/repeat phase
attribution. `workspace.py` protects current-best scratch restores and audits
relocated archives without rewriting hash-bound records. These changes require
fresh baselines; they do not retroactively accept previously rejected work.

The host-side tools under `performance/optimization/` use Python's standard
library and do not require the disabled radar Python bindings. See the
[tool interface](../skills/operator-perf/references/tools.md) for session,
comparison, checkpoint, profiler, and report commands. Run their checks with:

```bash
python3 -m unittest discover -s performance/optimization/tests -v
```

With performance and testing enabled, `radar_verify_operator` is a separate
HSDK 5 executable using native testing
fixtures and bounded snapshots; it never adds capture copies to the timed
runner. Native fixtures cover ClutterFilter, FMCWRangeProcessor and the fused
FMCWRangeDopplerProcessor shared-stage consumer,
both precisions, recipe profile sizes, live match/miss, and range optional
outputs. Expanded range-stage cases exercise Hann windows, padding, dechirp,
real-IF, four named channels, and live window/waveform match-then-miss. This is
range-stage coverage, not exhaustive Doppler, boundary/error or cache-revisit
coverage. Snapshots export observed output sequence, capture time, clock,
representation, bytes, and stable SDK diagnostics; runtime logs are retained
for separate warning/error review. GPU qualification against SDK `main-5x`
revision `5fed2c24acf5b0bc1e538deeaae311efd8d256e1` passed 108 cases with exact
repeat equivalence and six deliberate value/temporal mismatch rejections.
Additional affected behavior and operators require fixture coverage before
optimization can be accepted. No CPU operator fallback is provided.

```bash
ctest --test-dir build/holoscan-radar -R '^radar_perf_native_verification$' \
  --output-on-failure
```

The GPU-required test preserves fresh snapshots, logs, and its report under
`build/holoscan-radar/performance/verification/qualification/run-*`. This
qualifies the fixture implementation, not a performance optimization candidate.
Use the source-bound build manifests and `capture` command for session evidence.

The skill and helpers are model- and coding-agent-independent: they use
Markdown instructions, standard Python, Git, and the repository's native GPU
toolchain, without provider APIs or an agent-specific runtime. Agents with
skill discovery can link or copy `skills/operator-perf` into their supported skill
location. For agents without discovery, start with this instruction:

```text
Read skills/operator-perf/SKILL.md and follow its linked references for this task.
```

Run from this checkout, or give the agent the absolute path to `SKILL.md`.
Tool paths resolve from the active repository, not the installed skill folder.
No provider-specific metadata is required. Installation outside this repository
is not performed automatically.

## Deployment-qualified campaigns

Use [operator-benchmark](../skills/operator-benchmark/SKILL.md) to inventory a
set of operators, qualify affordable measurement protocols, freeze selection,
validate on held-out runs, then collect full-case baselines and reports. It uses
[operator-perf](../skills/operator-perf/SKILL.md) for assessments and confirmed
candidate sessions. Campaigns default to assessment-only; missing harnesses and
unstable or uncovered cases remain visible as blocked scopes.

```bash
python3 performance/optimization/radar_perf.py campaign --help
python3 -m unittest discover -s performance/optimization/tests -v
```

Version 3 captures chronological raw frames with isolated repeat boundaries,
alternates adjacent A/A or A/B processes and protects first-50-frame
median/p90/p99 alongside the complete window and initialization/live costs.
Qualification requires an explicit positive caller budget. Sampling exploration
starts at 50/500/2,000 frames and 3/7 repeats, subject to cost and budget; these
are exploration levels, not universal benchmark defaults. Median aggregation
remains the default. Optional exploratory means require frozen policy and fresh
validation. Confidence width and +/-1% precision are diagnostics, not gates.

The maintained `measurement.py`, `qualification.py` and `campaign.py` tools
record forecasts, budget ledgers, frozen selections, disjoint validation,
per-case sampling, clocks, owned subprocesses, device locks and hashed resume
receipts. Reports include absolute A/A milliseconds, variability/limits,
confidence intervals and partial coverage. Version-2 artifacts remain usable
for historical reports/replay; new production sessions require fresh version-3
qualification, full-case admission and confirmed configurations. Both existing
checkout and worktree layouts preserve immutable builds and archived runtimes.
