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

## C++ adapter boundary

The common driver validates and converts every YAML case before scope selection.
It owns process warmup, independent Graph runs, sampling, and report rows. The
typed recipe runner stages one, two, three, or six data tensors, creates the
source, binds device outputs, assembles one partition, and captures paired
source/synchronized-sink timestamps. `PerfRecipeOutputs` connects the timed
output or aligned timed pair, drains the other outputs, and declares placement
for optional unwired Tensor outputs. The same typed data-port source serves
baseline and selected-live cases. An adapter builds valid tensors and Params, connects the operator's typed data ports, and names
its outputs. Live adapters declare one selected value and its typed destination
port; the shared source emits it on one frame and the runner checks emission
and Params reversion. The adapter retains contract-specific value conversion
and throughput units.

### Adding a C++ operator

1. Define `performance/profiles/<operator>.yaml` with three sizes and curated
   scope cases. Name every documented live port in its manifest. Use an explicit
   case family such as `id: "baseline-{profile}", profiles: [small, medium, large]`
   when the parameters and scope tags are identical across sizes. Expansion
   follows that list in order; it never combines independent parameter axes.
2. Add `performance/cpp/perf_<operator>.cpp`. Its typed adapter defines a
   `Scenario`, `parse_case`, `size_def`, `run_graph`, and `base_row`. In
   `run_graph`, call `run_data_recipe` for defaults or declare `live_option`
   and `selected_live_case` entries for overrides, then call
   `run_selected_live_recipe`. Use `PerfRecipeOutputs` for timed and drained
   outputs. Add `decorate_row` only for throughput units or notes, and
   `validate_run_settings` only for special constraints such as a cached
   transition needing a warmup frame. The shared driver supplies CLI, run
   planning, sampling, and report rows.
3. Add the operator name once to `HOLOSCAN_RADAR_PERF_TARGETS` in
   `performance/cpp/CMakeLists.txt`. That list registers its target and Minimal
   GPU smoke. Add a focused Overrides smoke when its live payload has a new
   representation or timing behavior.

`perf_ddm_power_fold.cpp` is a small no-live example; an angle estimator shows
selected descriptor, host-vector, and scalar live cases. Keep valid input
generation, typed HSDK data connections, `MaxTensorExtents`, and
operator-specific scale calculations in the adapter. The shared payload codecs
emit geometry, windows, double/int64 host vectors, smoothing, waveform, and
scalar sidebands. The unit-test harness remains suitable for correctness
checks; its bounded output snapshots add copies and lack the paired timestamps
used here.

### Onboarding trial: DDM power fold

All 26 current C++ operators already have performance drivers. To check the
new-operator path, the smallest existing driver, `perf_ddm_power_fold.cpp`, was
used as a fresh-adapter walkthrough. The required work is:

1. Choose three valid input shapes and curated cases in the YAML suite. The
   shared loader supplies size selection and case IDs; `--validate` checks
   every case before any Graph work.
2. Decode one typed scenario and reject contract-invalid parameters. For this
   operator the rank-4 shape and Doppler divisibility by TX count are specific
   to its locked API.
3. Stage a valid input tensor, create `DDMPowerFoldOp<float>` with its
   `MaxTensorExtents` and descriptor, connect the named data port, and declare
   the timed output. The recipe runner creates the source and sink.
4. Describe input/output scale in `base_row`, then register one CMake target
   name. `--list-cases` checks ordering; the generated Minimal smoke checks
   Graph execution and a complete report without a speed threshold.

The no-live adapter remains mostly descriptor and shape construction,
operator creation, named data wiring, and scale calculation. A live adapter
adds one typed case per YAML live port; the shared selected-live source owns
payload pools and one-frame emission. `PerfRecipeOutputs` owns sink creation,
output placement, and partition membership. No adapter implements a source
`compute()` loop or a sampling loop.

## YAML profiles and scopes

Migrated tools read `performance/profiles/<operator>.yaml`. Each suite names
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
