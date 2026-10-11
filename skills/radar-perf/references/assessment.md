# Assessment

## Preflight And Matrix

Use the host Holoscan CLI with the pinned HSDK 5 image, as documented in
`SDK_DEVELOPMENT.md`. Record SDK/image identity, device/driver, build settings,
source fingerprints, recipes, and sampling configuration. Discover tools and
actual CLI options instead of copying historical `./holohub`, NVTX build flags,
or Python bindings commands. Do not provision tools or elevate profiler
privileges beyond existing authorization.

Build the selected performance target, then run `--validate` and `--list-cases`.
Use small/medium/large profiles and curated static/live cases from its YAML
suite; avoid generating a Cartesian product. Default assessment scope is
`overrides`; select `thorough` where cached/unseen transitions or affected
variants require it. Identify gaps such as absent SAR recipes explicitly.

Run clean, unprofiled timing using `--output`, preserving schema-v3 JSON.
Source scheduling and the terminal GPU synchronization are part of these
end-to-end measurements. Distinguish setup, cold, steady, changed-frame,
source cadence, and achieved throughput. Never add overlapping CUDA API and
kernel durations as if they were disjoint parts of the tick.

## Attribution

Capture Nsight Systems in a fresh artifact directory. Export its CUDA API,
GPU kernel, and memory summaries to CSV with the installed tool's documented
options. `radar_perf.py profile` validates capture freshness and nonempty
tables and records their hashes; inspect raw evidence before choosing a fix.
If the operator has usable NVTX ranges, attribute phases; otherwise use scoped
temporary instrumentation only when needed, then remove it and rebuild.

Prioritize recurring alloc/free, avoidable copies/synchronizations, redundant
launches, and avoidable host work. Cold initialization is not a steady-path
finding. A live rebuild already cached on revisiting a configuration is usually
advisory, not a candidate. Contract-required synchronization or output sizing
must stay intact.

Use scoped Nsight Compute for a kernel-dominated decision when available.
Profile a few confirmed steady launches of one representative case; verify
nonempty collected kernels and distinguish bandwidth, compute, occupancy, and
stall evidence. Counter access failure is a reported limitation. Profilers
explain performance; they do not supply acceptance timing.

Rank findings with case IDs, measurement evidence, mechanism, expected benefit,
and whether actionable. A structural count reduction corroborates small timing
deltas but does not override an inconclusive speed gate.
