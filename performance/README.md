# Operator performance characterization

The C++ tools measure one HSDK 5 operator in a source → operator → terminal-sink
Graph. Results are diagnostic measurements, not timing pass/fail thresholds.
The terminal sink synchronizes the GPU. A process warmup precedes the matrix;
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
still appear at `build/performance/cpp/perf_<operator>`; Python tools remain in
`performance/python/` until their runtime migration.

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
a valid contract value or sizing rule. Python runtime parity still awaits a
buildable HSDK 5 binding.

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
The `fmcw_range_processor` and `fmcw_doppler_processor` Thorough suites include
cached `nfft` transitions: frame zero primes the new FFT plan during warmup,
then the measured change reuses it. Their unseen cases make the same change
without priming. Cached cases require at least one warmup frame.

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

All 26 C++ operator tools use these YAML suites. The Python operator tools still
use their existing scenario matrices; runtime parity awaits a buildable HSDK 5
Python binding in the pinned development image.
