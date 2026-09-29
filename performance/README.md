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
shared pilot runner stages and connects only the live port named by a selected
case; a scaled-axis codec declares its source axis in the recipe. It is a single-operator declaration,
not a general Graph wiring language. Case IDs, scope tags, order, and
`{profile}` expansion retain their version-1 behavior. The other 21 drivers
continue to load version-1 suites.

### Adding a C++ operator

1. Create `performance/profiles/<operator>.yaml` with three valid sizes, a
   version-2 recipe, and curated cases. Declare every documented live port and
   cover it alone at each size in Overrides. Run `--validate` and `--list-cases`;
   both validate unselected cases without authoring a Graph.
2. For direct `MaxTensorExtents`, built-in tensor generators, canonical scalar
   or sequence Params, and registered scalar or host-vector live payloads, add the name and
   operator type to `performance/cpp/CMakeLists.txt` with `radar_add_yaml_perf`.
   The generated entry point uses `YamlStandardTraits`; no operator-owned
   C++ source is needed. `test_perf_yaml_standard` exercises this path with a
   real two-input operator and a one-frame scalar live case.
3. If the locked contract needs a descriptor builder, related-input fixture,
   cross-field check, derived bounds, payload codec, throughput unit, or
   report postprocessing, put the typed adapter in `performance/cpp/perf_<operator>.cpp`.
   That file owns only those semantics;
   source publication, Graph assembly, timing, and report buckets stay shared.
   Add the operator to `HOLOSCAN_RADAR_PERF_TARGETS` for its Minimal GPU smoke.

The five pilot tools keep typed adapters in their `perf_*.cpp` files for their
contract-specific parts:
DDM power fold's DDM descriptor and divisibility, SnapshotGather's related
indices and window, Beamscan's geometry/grid descriptors, FMCW range's FFT
bounds and cached/unseen transitions, and CFAR's variant-dependent limits and
calibration summary. Their executable names and CLI flags remain unchanged.
The current HSDK registry cannot register local enum live payloads as external
topics; the Beamscan and FMCW adapters use a typed flow connector for them.

### Pilot size and new-operator walkthrough

The five original adapter files totaled 1,543 lines. After consolidating the
pilot mechanics, the five `perf_*.cpp` adapters total 1,114 lines
(429 fewer, about 28%). The shared pilot helper is 274 lines; registry,
recipe, and YAML validation infrastructure is additional. Shared code now owns
case identity and value merging, typed input staging, baseline and live Graph
calls, cached-transition warmup checks, common payload builders, and rate
conversion. A further pass reads named YAML axes directly and validates
window, waveform, geometry, and live values by their constraints instead of
repeating the exact curated case values in C++. The remaining per-operator
C++ constructs contract-valid Params and derived bounds, validates relationships
among fields, supplies the local enum and semantic live payloads, and provides
operator-specific report fields. Beamscan now declares its direct pool bounds
in YAML. SnapshotGather enum Params still need typed conversion: the pinned
HSDK 5 parameter materializer rejects a YAML string for `doppler_edge`. These five operators have unusual semantic needs, so the new
YAML-only path is the better guide for a standard operator than their adapter
sizes alone.

To assess a new standard operator: use its locked
contract to choose three profiles; declare one fixed recipe with direct bounds
and built-in generators; author explicit cases; register its HSDK operator type
through `radar_add_yaml_perf`; run validation, Minimal, and relevant Overrides
smokes. Add a `perf_*.cpp` adapter only when a declared component cannot encode
a valid contract value or sizing rule. Python runtime parity still awaits a
buildable HSDK 5 binding.

## YAML profiles and scopes

Migrated tools read `performance/profiles/<operator>.yaml` automatically
from the executable's source checkout, even when that checkout is mounted at a
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
