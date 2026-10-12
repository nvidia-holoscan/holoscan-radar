# Assessment

## Preflight And Matrix

Use the host Holoscan CLI with the pinned HSDK 5 image, as documented in
`SDK_DEVELOPMENT.md`. Record SDK/image identity, device/driver, build settings,
source fingerprints, recipes, and sampling configuration. Discover tools and
actual CLI options instead of copying historical `./holohub`, NVTX build flags,
or Python bindings commands. Do not provision tools or elevate profiler
privileges beyond existing authorization.

Run `diagnostics.py preflight` on the host and inside the selected image. Missing
Holoscan CLI can be handled with an explicit Docker argv wrapper: mount source,
build and artifact paths consistently, and set `docker --workdir` to each
manifest's source checkout. Set its host `cwd` too. Never reuse a CMake tree
configured at another mount path. Record immutable Docker image ID and actual
SDK source revision; a historical qualification revision is not necessarily
the current build pin. If no SDK label exists, record the evidence and gap.
Check installed `nsys --help` before selecting sampling options; ranges vary
by platform. Report unavailable `ncu`, counter permission failures, and blocked
CPU sampling; do not change `perf_event_paranoid`. Run pinned hooks with a
compatible Node runtime (Node 22 or newer for Unicode-set regex support), not
an unpinned substitute linter. Resolve a symlinked skill to its repository path
when file discovery does not follow the installation link.

Preflight records devfreq/CPU governors, frequency bounds and current-frequency
snapshots, power mode queries, GPU clocks/power, compiler/CUDA versions, L4T and
optional `--docker-image` inspection. Permission/tool failures are limitations,
not permission to elevate. Never infer an SDK revision from an image tag.
Confirm controlled versus deployment clock behavior with the user. If stable
controlled clocks are required, ask for separate authorization and recapture
the configuration afterward. `clock_configuration` can bind the confirmed
sysfs configuration; paired runs record frequency observations and reject
governor/bound changes. Missing hardware telemetry means unknown, not locked.

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

Before candidate edits, use the same baseline manifest on both sides of
`measure`, then `calibrate` its raw pairs. Check every objective/protected
metric, case and precision for directional bias, drift, and bimodality. Inspect
the distributions as well as the automated delta/spread limits. Propose
practical effect thresholds using this evidence and obtain confirmation before
`init`; never derive a more permissive threshold after a rejection. Defaults
for steady metrics are 5% P90 absolute A/A paired delta and 10% IQR/median
baseline dispersion, both configurable. Cold/setup/override have separate
proposed defaults and may need evidence-backed policies and more samples.
Directional familywise A/A intervals still block bias even when the expected
gain threshold is larger. Maxima, full ranges and 3x-median stall counts remain
visible; no samples are dropped. Protect `p90_ms`, `p99_ms` or `max_ms` explicitly
when tail behavior is part of the objective. Robust diagnostics do not prove
stationarity; inspect distributions, drift and mode changes too.
A failed calibration is **validation blocked**, not proof of an SDK bug.
Rerun calibration on the current best after an accepted candidate. Keep the
original calibration for final-versus-original measurements.

Record timing precision coverage independently of float/double verification.
The current recipes may time float only. Default timing protection requires
both precisions; either extend the recipe/typed adapter or confirm an explicit
float-only scope and report that double performance is unprotected.

Use `--cases` to select exact expanded IDs and `--reverse-order` for a position
probe before trusting bulk-process measurements. Compare same-work cases at
early/late positions and vary Graph-session counts on the unchanged baseline.
Default optimization measurements launch a fresh process for **each case and
repeat**, retaining every raw child report; process startup is outside timing.
This prevents earlier cases contaminating later initialization or steady work,
but intentionally changes the measurement context: rebaseline both builds.

Keep setup, first Graph tick, first new configuration and explicitly cached
configuration revisits distinct. `override_ms` in a fresh process still includes
first-use work triggered by that changed frame; recipe cached cases prime it
explicitly. Do not warm all paths or rename first-use costs away.
The schema-v3 `override_context` records `none`, `cached_configuration`, or
`first_changed_configuration_after_baseline` from validated recipe priming.
It is compared across baseline/candidate and calibration. This labels context,
not a claim to decompose every global CUDA initialization cost.
Confirm which contexts are protected before edits and add missing cached cases when needed.
Unchanged immediate CUDA calls do not rule out a real process-state regression.

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

Use operator-local NVTX compute/receive/emit ranges as corroborating attribution,
not as an automatic replacement for the user's end-to-end objective. SQLite
`StringIds` may store `op:compute` without the rendered `holoscan:` domain.
`diagnostics.py phases` clips same-thread CUDA runtime calls to scoped NVTX
ranges and reports their union, outside-API time, and temporary `op.*` phase
marks. Use single-case captures and explicit thread selection when ambiguous;
never infer recipe order by gaps or silently zip scenarios to range groups.
The harness now emits `radar-perf:case:<id>:repeat=<n>` session ranges, distinct
from process warmup. Use `phases --case ID --repeat N` for explicit containment
selection in multi-session captures. `--timed-output PORT` reports CPU time
after its emit mark when present. Map that to the sink and CUDA dependencies;
work after emit can still affect terminal synchronization and critical latency.
Outside-API time includes waits, not just CPU work. Inspect asynchronous
allocation/free, host setup and launch gaps separately when predicting gains.
Reduced compute with worse end-to-end timing can be a real regression; a
candidate-correlated slow mode is not automatically noise or a scheduler bug.

Rank findings with case IDs, measurement evidence, mechanism, expected benefit,
and whether actionable. A structural count reduction corroborates small timing
deltas but does not override an inconclusive speed gate.
