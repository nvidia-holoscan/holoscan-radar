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

The common driver parses every YAML case and asks each operator adapter to
validate and convert one case before applying scope or size filters. YAML
override metadata selects the live run plan; the adapter must name the same
live port. The runner checks that one changed payload was emitted at the
planned frame in every independent run. It owns
process warmup, run planning, independent Graph sessions, and the setup and
compute report rows. An adapter supplies valid input tensors and descriptors,
operator parameters and port wiring, plus any operator-specific throughput
units. Shared typed sources publish one to three inputs and a one-frame live
sideband; MultiSourceExpansion uses the shared six-input source required by its
contract. The source and sink capture paired per-frame timestamps.

### Adding a C++ operator

1. Define `performance/profiles/<operator>.yaml` with three sizes and curated
   scope cases. Name every documented live port in its manifest. Use an explicit
   case family such as `id: "baseline-{profile}", profiles: [small, medium, large]`
   when the parameters and scope tags are identical across sizes. Expansion
   follows that list in order; it never combines independent parameter axes.
2. Add `performance/cpp/perf_<operator>.cpp`. Its typed adapter defines a
   `Scenario`, `parse_case`, `size_def`, `run_graph`, and `base_row`. Add
   `decorate_row` only for throughput units or notes, and `validate_run_settings`
   only for constraints such as a cached transition needing a warmup frame.
   The shared driver supplies CLI, process warmup, run planning, all independent
   runs, scenario/size identity, and the setup/compute report rows.
3. Add the operator name once to `HOLOSCAN_RADAR_PERF_TARGETS` in
   `performance/cpp/CMakeLists.txt`. That list registers its target and Minimal
   GPU smoke. Add a focused Overrides smoke when its live payload has a new
   representation or timing behavior.

`perf_ddm_power_fold.cpp` is a small no-live example; angle estimators show
multi-port descriptor payloads. Keep input generation, typed HSDK ports,
`MaxTensorExtents`, and operator-specific scale calculations in the adapter.
The shared `perf_payloads.hpp` helpers set up typed host/device source ports
and emit angle geometry, windows, coordinate vectors, smoothing, and scalar
sidebands. The unit-test harness remains suitable for correctness checks; its
bounded output snapshots add copies and lack the paired timestamps used here.

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
