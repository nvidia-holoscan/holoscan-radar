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
