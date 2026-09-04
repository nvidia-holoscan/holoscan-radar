# HSDK 5 Port

[EA HSDK5 ONLY] Playbook and tracker for the Holoscan SDK 5 migration of
this library. Remove this file (and every `[EA HSDK5 ONLY]` fence in the
tree) before promoting the port to `main` for the public HSDK 5 release.

This document covers **how** we port (phases, per-operator recipe, gates,
verification) and **what is still open** (EA bugs, durable API deltas,
deferred slices). It is **not** [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md):
those rows need a radar-contract team decision. Operator-local algorithm /
contract notes stay in each op’s `OPEN_ITEMS.md`; put a one-line pointer
here when the root cause is HSDK 5 / EA.

| Resource | Role |
| --- | --- |
| [`HSDK5_BRANCH_SUMMARY.md`](HSDK5_BRANCH_SUMMARY.md) | Executive summary: branch plan, issues + workarounds, EA blockers |
| [HSDK 5 preview docs](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/introduction/overview.md) | SDK API source of truth (prefer `.md`; index via `llms.txt`) — do not infer from HSDK 4.x |
| [`OPERATOR_API.md`](OPERATOR_API.md) | Locked operator contracts (ports, shapes, dtypes, params) |
| [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md) | Done gate once an op is on HSDK 5 (adapt 4.x checklist wording to Graph / `Operator<Params>`) |
| [`operators/common/README.md`](../../operators/common/README.md) | Pilot patterns: Tensor I/O, `[live]` overrides, Graph harness |
| [`STATUS.md`](STATUS.md) | Who owns what; claim / land coordination |
| [`MAX_TENSOR_EXTENTS.md`](MAX_TENSOR_EXTENTS.md) | **C++ landed:** module `MaxTensorExtents` ctor-capture for Tensor pool sizing + drop opaque leading batch; apps/Python phase 4; pinned [`MAX_TENSOR_EXTENTS_2026-08-15.md`](MAX_TENSOR_EXTENTS_2026-08-15.md) |
| [`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md) | Settled mixed `[live]` payload policy: schema ownership/evolution, ChannelPod tuples, bounded host-Tensor vectors, bridges/tests, and Python limitation |
| This file | Migration process + open HSDK 5 / EA friction |

---

## Goal and scope

**Goal.** Build, test, and (eventually) package the radar module against
Holoscan 5 — operators as `Operator<Params>` with typed Tensor ports,
exercised on the Graph runtime — without keeping a 4.x shim.

The EA branch covers, in dependency order:

1. Foundation (container / CMake / MatX / Holoviz feature probes).
2. Shared `radar_common/sdk/` helpers and the Graph C++ test harness.
3. Per-operator C++ ports + unit tests (allow-listed).
4. Graph perf drivers (separate allow-list) once an op’s C++ surface is
   stable.
5. Python bindings, Holoviz demos, and example apps once EA exposes a
   usable Python operator surface / Holoviz (or we accept a C++-only path).
6. Close issues deferred on the EA SDK and/or settled implementation work
   (live-override Graph coverage, durable port/descriptor API deltas, and the
   descriptor-output Tensor-pack representation).
7. Cleanup before `main` (drop allow-lists, `[EA HSDK5 ONLY]` fences, and
   this file).

**Out of scope here:** changing radar algorithm contracts for non-SDK
reasons (that is `OPERATOR_API.md` + team review), Pulse-Doppler / SAR /
CDM, and inventing new shared descriptor types without a contract change.

**Branch / env.** Work lands on `dpc/hs5.0-fixes` against the HSDK 5 EA
image (see [`.devcontainer/README.md`](../../.devcontainer/README.md)).
`find_package(holoscan)` must resolve 5.x; there is no 4.x compatibility
layer on this branch.

---

## Porting phases

Work proceeds in layers so unported code does not break the tree:

| Phase | What lands | Gate |
| --- | --- | --- |
| **0 — Foundation** | Dockerfile / yaml-cpp / MatX `find_package`, C++20, Holoviz probe, Python forced off | Tree configures on EA |
| **1 — Shared glue** | `radar_common/sdk/` Tensor I/O + `[live]` override helpers; `sdk_harness.hpp` Graph runners | ClutterFilter pilot compiles |
| **2 — Operator wave** | Rewrite ops to `Operator<Params>` + typed ports; Graph unit tests; allow-list each name | `ctest -L unit` green for ported ops (intentional skips OK) |
| **3 — Perf wave** | Rewrite `perf_*.cpp` to Graph; add names to `HOLOSCAN_RADAR_HSDK5_PORTED_PERF` | `ctest -L performance` smokes pass |
| **4 — Apps / Python** | Demo apps + bindings when Holoviz / Python EA surface exists | App smoke / pytest |
| **5 — Close deferred** | Re-probe EA gaps; settle durable HSDK 5 API deltas with the team; restore Graph live-override / live descriptor ports / typed geometry outs where ratified; unskip deferred suites | Every **Open — SDK / EA** item closed or moved to durable docs; no intentional EA `GTEST_SKIP` left unexplained |
| **6 — Cleanup** | Drop allow-lists, `[EA HSDK5 ONLY]` fences, and this file before `main` | See **Cleanup gate** |

Phases 2–3 are per-operator and may interleave (an op can be checklist-near
complete in C++ while its Graph perf driver is still deferred). Phase 4 may
start once enough ops are stable for an app chain; it does **not** clear
phase-5 EA / team debt. Phase 5 is the dedicated close-out for everything
parked under **Open — SDK / EA** (and the deferred Graph suites that depend
on those answers) before cleanup is allowed to strip the fences.

### Phase 5 — what “close deferred” means

Inventory is the numbered sections under **Open — SDK / EA** (and any
new EA/team items added there). Typical work:

| Class | Close by |
| --- | --- |
| **EA limitation** (e.g. §1 live-override Graph units / enum lives; remaining §8 enum/vector perf) | Re-probe on newer EA / GA; restore `contract()` override policy + unskip Graph tests; document the working harness recipe — or reclassify as durable if the SDK will not support it |
| **Durable HSDK 5** (e.g. §2 unique in/out names, §9 vector/descriptor lives, §10 `ArrayGeometry` Tensor pack) | Apply the locked contract in `OPERATOR_API.md`; for §9 this means generated schemas for shared owning descriptors, ChannelPod fixed tuples, and bounded rank-1 pageable-host Tensors for variable scalar vectors |
| **Our deferred work blocked on the above** (optional-mask Graph wiring, Graph live-override unskips) | Land against the settled contract after the matching EA prerequisite is closed; do not invent a different physical payload locally |

Phase 5 is **not** for new algorithm contracts (those stay
[`TEAM_BLOCKED.md`](TEAM_BLOCKED.md) / per-op `OPEN_ITEMS.md`). It **is**
for every interim that exists only because of EA friction or an unsettled
HSDK 5 API shape.

---

## How to port an operator

Copy the ClutterFilter / SnapshotGather patterns; do not reintroduce
HSDK 4.x `InputContext` / `TensorMap` / `HOLOSCAN_OPERATOR_FORWARD_ARGS`.

1. **Read the contract** in `OPERATOR_API.md` and any op `OPEN_ITEMS.md`.
   Note unique in/out names (§2), the settled mixed live-payload mapping
   (§9), and whether a descriptor output must ride a Tensor pack (§10).
2. **Rewrite the op** to `Operator<Params>` + `HOLOSCAN_PARAM_SCHEMA`;
   typed `Input<Tensor>` / `Output<Tensor>` (or POD live Inputs); receive
   via `receive_tensor`, emit via `emit_device_tensor` /
   `bind_tensor_output_device` on the producer side; stream from
   `context.cuda_stream()`; POD lives via
   `add_live_override_to_contract` + `resolve_live_override`
   (`Exact{}` + `SafeDefault{}`, unconditional under every trigger shape —
   wiring is the app's choice, and no `expect_*` parameter exists, §11).
3. **Allow-list** the CMake target name in
   `HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS`
   ([`cmake/HoloscanRadarOptionalFeatures.cmake`](../../cmake/HoloscanRadarOptionalFeatures.cmake))
   and register its `test_*` binary in `tests/cpp/CMakeLists.txt`.
4. **Graph tests** using `tests/cpp/utils/sdk_harness.hpp` (pick the
   runner that matches port arity). Mid-stream Exact override cases follow
   `test_fmcw_range_processor.cpp` (`nfft`); elsewhere they are still
   skipped under §1 pending the generalized harness, with Params
   configurability coverage in their place. Prefer stage / kernel tests when
   optional Inputs cannot be wired yet (§11).
5. **Checklist** against [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md)
   (interpret “TensorMap / Application harness” as typed ports + Graph).
6. **Perf (optional same PR or follow-up):** Graph-rewrite `perf_<name>.cpp`,
   append `<name>` to `HOLOSCAN_RADAR_HSDK5_PORTED_PERF` in
   `performance/cpp/CMakeLists.txt`, run the smoke.
7. **Record** progress in the table below, durable deltas under **Open**
   or **Resolved**, and a `STATUS.md` line when claiming or landing.

**C++ operator first-pass:** complete (all 19 operators allow-listed). CFAR
GO/SO/OS/alpha_lut Graph unit suites + statistical MC (`ctest -L statistical`)
landed. Graph perf (phase 3) cold/steady smokes landed for all drivers; POD
`override_ms` restored for applicable consumers (§8). Enum `ClutterFilterOp.method`
`override_ms` landed; remaining enum lives and §9 lives remain deferred
(§1/§9). Remaining: apps/Python (phase 4), and EA gaps in §§1/9/10
(phase 5) — see progress table.

---

## Build and verify

```bash
cmake -S . -B build \
  -DHOLOSCAN_RADAR_BUILD_TESTING=ON \
  -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON
cmake --build build -j"$(nproc)"

# Ported C++ units (pytest is unregistered while Python is forced off)
ctest --test-dir build -L unit --output-on-failure

# Graph-ported perf smokes (all drivers on HOLOSCAN_RADAR_HSDK5_PORTED_PERF)
ctest --test-dir build -L performance --output-on-failure
```

Pytest is unregistered while Python is forced off, so bare `ctest` is fine
for C++ unit/perf labels. Unported `perf_*` binaries must not enter the
build — that is what `HOLOSCAN_RADAR_HSDK5_PORTED_PERF` enforces.

---

## Tracking open issues

| Action | Do this |
| --- | --- |
| Hit a new HSDK 5 / EA snag | Add a numbered section below with class, repro, interim, and status |
| Land a workaround | Mark the code `[EA HSDK5 ONLY]`, note it under the matching item |
| Confirm EA vs final API | Update **Class** (`EA limitation` → `durable HSDK 5` or close as fixed) |
| Finish an item | Move a one-line summary into **Resolved** (or delete if noise); drop fences |
| Port the next operator | Follow **How to port an operator**; expand the allow-lists |

### Item classes

| Class | Meaning |
| --- | --- |
| **EA limitation** | Works in design / docs, fails or is rough on the current EA build; expect to re-test on newer EA / GA |
| **Durable HSDK 5** | Intentional SDK 5 behavior we must keep adapting to (not a temporary bug) |
| **Our deferred work** | In-scope for this repo; blocked only by sequencing / allow-list, not by an open SDK question |

---

## Port progress

Allow-list: `HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS` in
[`cmake/HoloscanRadarOptionalFeatures.cmake`](../../cmake/HoloscanRadarOptionalFeatures.cmake).

| Area | State |
| --- | --- |
| Foundation (Dockerfile yaml-cpp/ssh, `find_package(matx)`, Holoviz gate, C++20) | landed on `dpc/hs5.0-fixes` |
| Shared Tensor I/O + Exact `[live]` helpers (`radar_common/sdk/`) | landed (ClutterFilter pilot; Exact since 2026-08-21) |
| `ClutterFilterOp` F32/F64 + Graph C++ tests | landed; Exact `method` (unconditional entry); Graph perf cold/steady landed; enum `method` live-override units landed now that §12 is fixed; Graph perf `method` `override_ms` landed (§8) |
| `SnapshotGatherOp` F32/F64 + Graph C++ tests (OnAligned 2-in/2-out) | landed; tick-local config/plan and constructor-created workspace; Exact `snapshot_window` on an unconditional entry with mid-stream/revert Graph coverage; optional/always `is_valid`; Graph perf `snapshot_window` `override_ms` restored (§8) |
| `PhaseShiftBeamformerOp` F32/F64 + Graph C++ tests | landed; `[live]` geometry/grid/taper on wire-schema and bounded host-Tensor ports (§9); no POD live Inputs, so Graph live tests still skipped; Graph perf cold/steady landed (live deferred §1/§8/§9) |
| `BeamscanEstimatorOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; Exact `peak_interpolation` (unconditional entry) with its enum live-override unit landed now that §12 is fixed; optional `is_valid` Graph wiring deferred (§11); geometry/grid/taper on wire-schema and bounded host-Tensor ports (§9); Graph perf cold/steady landed (live deferred §1/§8/§9) |
| `SubarraySelectionOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; tick-local config/plan, constructor-created selection cache + bounded index table; required `snapshot_out` before independent latched geometry/mask outputs; outputs `snapshot_out` / `is_valid_out` (§2); `array_geometry` emitted as packed float64 Tensor (§10); geometry wire-schema and bounded host-Tensor `subarray` lives (§9); optional `is_valid` Graph wiring deferred (§11); Graph perf cold/steady landed (live deferred §9) |
| `ClusterDBSCANOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; tick-local config/plan; Exact `epsilon`/`min_points`/`max_clusters` with Graph coverage; bounded host-Tensor `epsilon`, `max_features` checked on arrival (§9); required `labels` plus optional/latched `sizes`; Graph perf uses production Exact `min_points` (§8) |
| `MVDREstimatorOp` F32/F64 + Graph C++ tests (1-in/5-out) | landed; always emits optional `angle_spectrum`/`doa_multi`/`source_count`/`eigenvalues`; Exact `diagonal_loading`/`max_sources` (unconditional entries) (Graph live deferred §1); geometry/grid/`spatial_smoothing` on wire-schema, bounded host-Tensor, and ChannelPod ports (§9); optional `is_valid` Graph deferred (§11); `stop()` MatX retire preserved; Graph perf `diagonal_loading` `override_ms` restored (§8) |
| `MUSICEstimatorOp` F32/F64 + Graph C++ tests (1-in/5-out) | landed; always emits optional `angle_spectrum`/`doa_multi`/`source_count`/`eigenvalues`; Exact `num_sources`/`source_count_pfa`/`max_sources` (unconditional entries) (Graph live deferred §1); geometry/grid/`spatial_smoothing` on wire-schema, bounded host-Tensor, and ChannelPod ports (§9); optional `is_valid` Graph deferred (§11); `stop()` MatX retire preserved; Graph perf `max_sources` `override_ms` restored (§8) |
| `RootMUSICEstimatorOp` F32/F64 + Graph C++ tests (1-in/4-out) | landed; always emits optional `doa_multi`/`source_count`/`eigenvalues` (no `angle_spectrum`); Exact `num_sources`/`source_count_pfa`/`max_sources` (unconditional entries) (Graph live deferred §1); geometry/`spatial_smoothing` on wire-schema, bounded host-Tensor, and ChannelPod ports (§9); optional `is_valid` Graph deferred (§11); `stop()` MatX/cuSOLVER teardown preserved; Graph perf `max_sources` `override_ms` restored (§8) |
| `ESPRITEstimatorOp` F32/F64 + Graph C++ tests (1-in/4-out) | landed; always emits optional `doa_multi`/`source_count`/`eigenvalues` (no `angle_spectrum`); Exact `num_sources`/`source_count_pfa`/`max_sources` (unconditional entries) (Graph live deferred §1); geometry/`spatial_smoothing` on wire-schema, bounded host-Tensor, and ChannelPod ports, `subarray_spec` static (§9); optional `is_valid` Graph deferred (§11); `stop()` MatX retire preserved; Graph perf `max_sources` `override_ms` restored (§8) |
| `FeatureAssemblyOp` F32/F64 + Graph C++ tests (Params-derived aligned gathers) | landed; static `features` selects required zero-window aligned sources; unselected Tensor gathers use Exact + SafeDefault (§11); no `[live]` params; C++ apps use the library op; Graph perf cold/steady landed |
| `DetectionReductionOp` F32/F64 + Graph C++ tests (2/3-in/1-out) | landed; static `representative` Params selects the compiled aligned presence set (`first`: labels + idx, `peak`/`medoid`: labels + idx + points), with unselected `points` as Exact + SafeDefault; Exact `representative`/`score_method`/`feature_weights` are unconditional non-member entries; all `expect_*` retired (§11); enum live units landed (§12 fixed); `feature_weights` is a bounded host Tensor (§9); Graph perf cold/steady landed |
| `CFARDetectorOp` F32/F64 + Graph C++ tests (1-in/5-out; CA + GO/SO/OS + alpha_lut) | landed; all 5 outputs optional/latched; unconditional Exact lives with scalar, bounded host-Tensor, and ChannelPod payloads; Graph tests cover every live family and unwired outputs; statistical MC suite on label `statistical`; Graph perf measures `threshold_scale` `override_ms` (§8) |
| `MIMODeinterleaveOp` F32/F64 + Graph C++ tests | landed; output `adc_cube_out` (§2); Graph live-override/`TXSchedule` N/A (Params-only schedule); Graph perf cold/steady landed |
| `MIMOCoherentCompensationOp` F32/F64 + Graph C++ tests | landed; output `rd_maps_out` (§2); `[live]` `waveform` on a generated wire-schema port (§9); Graph perf cold/steady landed (live deferred §1/§8/§9) |
| `VirtualArrayAssemblyOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; output `rd_maps_out` (§2); `array_geometry` packed float64 Tensor (§10); Exact `wavelength` (unconditional entry) (Graph live deferred §1); Graph perf `wavelength` `override_ms` restored (§8); lg/xl vs 4.3 gap accepted as Graph publish-path artifact (§3) |
| `FMCWRangeProcessorOp` F32/F64 + Graph C++ tests (1-in/1–2-out) | landed; optional `range_bins` only with waveform; Exact lives (unconditional entries); `range_window`/`waveform` restored through generated schema payloads (§9); Graph perf cold/steady landed (live timing deferred §1/§8/§9) |
| `FMCWDopplerProcessorOp` F32/F64 + Graph C++ tests (1-in/1–2-out) | landed; optional `velocity_bins` only with waveform; Exact lives (unconditional entries); `doppler_window`/`waveform` restored through generated schema payloads (§9); Graph perf cold/steady landed (live timing deferred §1/§8/§9) |
| `FMCWRangeDopplerProcessorOp` F32/F64 + Graph C++ tests (1-in/1–3-out) | landed; optional axis outs with waveform; Exact lives incl. `clutter_filter` (unconditional entries); windows/`waveform` restored through generated schema payloads (§9); UlpParity vs split chain preserved; Graph perf cold/steady landed (live timing deferred §1/§8/§9) |
| Python bindings / Holoviz demos / apps | deferred (phase 4); X11 RD + Graph demos siso/simo/dense/mimo/aoa landed (§6) |

---

## Open — SDK / EA

### 1. Graph mid-stream `[live]` override harness

**Class:** EA limitation (not a permanent ban on live overrides).

**Contract.** `OPERATOR_API.md` requires Exact `[live]` parameters. Their
`contract()` entries are unconditional and an app enables an override by wiring
the port, under `OnAligned` as much as under `OnEach` (§11). FMCW
windows/waveform use generated schema payloads; the other descriptor and
vector lives ride the same wire-schema / bounded host-Tensor mapping (§9). An
override applies to a single tick; there is no operator-side latch to test.

**What the reproducer established.** The earlier failures were invalid
authoring shapes, not evidence that scalar sidebands are unsupported:

- independent clock roots need an explicit exact partition cover
  (`PARTITION_MANUAL_AUTHORING_REQUIRED` otherwise);
- a wired input must appear in the consumer's temporal contract, and every
  declared co-emitted output must publish on each replay
  (`RUNTIME_OPERATOR_REPLAY_MISMATCH` otherwise);
- enum outputs still fail runtime binding on this EA, in any wiring — tracked
  on its own as §12.

The supported perf shape is one source that emits the device Tensor and a
fixed-width scalar/POD sideband every tick, switching the sideband value at the
measurement frame. The source, operator, and drains are placed in one explicit
partition. When the consumer already declares the sideband with production
`Exact{}` (`add_live_override_to_contract`), wire that port directly — restacking
`PerfLiveContractOp` / `LatestWithin` is `TEMPORAL_CONTRACT_INVALID`.
`ClutterFilterOp.method` is the enum worked example. Historical POD drivers
still wrap with `LatestWithin` + `SafeDefault`; prefer Exact when the source
already re-publishes every tick.

**Narrowed 2026-08-20 (scalar-only probe).** A minimal scalar probe
separated the causes, and the second clock is *not* one of them:

- **A wired input port omitted from `contract()` fails at runtime** —
  `RUNTIME_OPERATOR_REPLAY_MISMATCH` on the second activation
  (`RUNTIME_EXECUTION_LANE_PLAN_INVALID` in the two-clock topology).
- **A declared port with no producer fails at compile** —
  `TEMPORAL_INPUT_UNCONNECTED`, `on_missing SafeDefault{}`
  notwithstanding.
- **Declared *and* wired works**, using the
  `radar_common/sdk/override_param.hpp` mapping (`LatestWithin` +
  `SafeDefault`) unchanged — including with an independent OnClock
  control-plane source publishing on a subset of activations.
- **A second independent OnClock source only needs manual partitions** —
  `graph.partition("data_plane").add(...)` /
  `graph.partition("control_plane").add(...)`. Once any partition is
  authored, every operator must be covered exactly once.

So the two failure modes are complementary, and the interim workaround
below sits on the wrong side of the trade: omitting the port keeps an
*unwired* graph compiling, but an application that actually connects the
override gets a run that dies on the second frame.

**Encoding found 2026-08-20 (declaration-replay probe; the pattern it
settled on is now `tmp-app/optional_live_port.cpp`).** A port an
application *may* connect
does have a single-build encoding, and an ordinary parameter is enough to
select it — provided the split between the two declaration sites is
respected:

- **`setup()` and `contract()` are replayed, and both may read
  `params()`.** They run on the authoring prototype, again after each
  accepted `set_params`, again during `compile()`, and again per run —
  three to four times in a trivial graph. Treat them as pure declaration
  code (the SDK says the same); the parameter value they see is the
  resolved one.
- **The port set is fixed; the contract is not.** `setup()` must register
  the same ports for every parameter value: a `set_params` that changes
  which ports it registers throws `GRAPH_PARAMETER_PORT_SHAPE_CHANGED`
  immediately. A `set_params` that only changes `contract()` is accepted
  and re-replayed — both a selector bound (`LatestWithin::max_age`) and
  the presence or absence of an entire `input()` entry.
- **So a per-port `expect_*` parameter worked** — this is what the module
  used before §11 landed, and it is recorded because the mechanism is still
  the general answer for a genuinely parameter-dependent contract entry:
  register the `[live]` port unconditionally in `setup()` and emit its
  contract entry only when the parameter is set. It is no longer used for
  `[live]` overrides anywhere; every entry is unconditional and the gate is
  gone module-wide (§11).
- **Wired-undeclared is still a failure mode.** A port the application
  wires but the contract omits kills the second activation with
  `RUNTIME_OPERATOR_REPLAY_MISMATCH`. The reverse — declared and unwired —
  is now legal for a non-trigger `Exact{}` + `SafeDefault{}` entry, which
  is what retired the gate (§11).
- **The selector bounds parameterize too, and they bite.** A parameter may
  set `LatestWithin::max_age`, and the window really gates selection: with
  a 1 ms window against a 100 ms control plane, most activations report
  `is_missing()`. So the window is
  the knob that converts silent staleness into visible absence — at the
  price of the operator needing a defined answer for "control plane went
  quiet". A practically-infinite `years{100}` never expires, but no window
  value makes `LatestWithin` safe against a control
  plane that publishes *faster* than the operator ticks: unselected survivors
  are retained rather than evicted, and exhausting the queue fails the
  producer's `emit` with `kBackpressured` and ends the run. That case wants
  `DrainAll` — see `OPERATOR_DEVELOPMENT.md` "Single-tick override semantics".
- **`set_params` during a run is a silent no-op.** Called on the operator
  handle after `run_async`, it neither throws nor emits a diagnostic, does
  not replay `contract()`, and does not change the running plan's
  behavior. Contract-affecting parameters are build-time configuration;
  retuning one means recompiling. Anything genuinely live must go through
  a `[live]` port (or the parameter control plane) instead.

The residual cost of the gate was that the agreement is manual: an
application had to state which `[live]` ports it wires, instead of that
following from `add_flow` alone. **Settled (L17):** the SDK now tolerates a
declared, unwired non-trigger `Exact{}` + `SafeDefault{}` entry under every
trigger shape, so wiring is the only statement an application makes and the
gate is gone module-wide (§11).

**Re-test list — swept 2026-08-23.** Every operator previously documented as
omitting live Inputs from `contract()` was re-checked against this narrowing
and converted: `ClutterFilterOp`,
`SnapshotGatherOp`, `FMCWRangeProcessorOp`, `FMCWDopplerProcessorOp`,
`FMCWRangeDopplerProcessorOp`, `VirtualArrayAssemblyOp`, `CFARDetectorOp`,
`DetectionReductionOp`, `ClusterDBSCANOp`, `BeamscanEstimatorOp`,
`MVDREstimatorOp`, `MUSICEstimatorOp`, `RootMUSICEstimatorOp`,
`ESPRITEstimatorOp`. Remaining limitations are tracked here and in each
operator's `OPEN_ITEMS.md`: optional *Tensor* inputs (§11), conditional
outputs, and not-yet-implemented descriptor / vector live ports outside the
restored FMCW schema ports (§9). `FeatureAssemblyOp` and
`SubarraySelectionOp` have no POD
lives, so only their optional-input limitations remain.

**Interim.**

- Every ported operator's `contract()` is now its data trigger (`OnEach`, or
  `OnAligned` for the two-tensor ops) plus one unconditional `Exact{}` +
  `SafeDefault{}` entry per override, via `add_live_override_to_contract` in
  `radar_common/sdk/override_param.hpp` — including the two `OnAligned`
  operators (§11). Descriptor ports convert generated
  wire types, and vector ports convert bounded rank-1 host Tensors, before
  override resolution (§9). Optional Tensor inputs and conditional outputs stay
  out of the contract (§11).
- **What is missing now is Graph coverage, not the contract shape.**
  Mid-stream Exact tests cover `FMCWRangeProcessorOp` `nfft`, range/Doppler
  windows, waveform validation, and fused schema overrides. The scalar/enum
  equivalents below still `GTEST_SKIP` pending that harness work.
- **Scalar / bool `[live]` overrides now run** (2026-08-27). The declared-but-
  unwired-port limitation is closed, and the Graph harness pattern from
  `test_fmcw_range_processor.cpp` (`nfft`) ports cleanly to the other
  operators. Landed and passing repeatedly in fresh processes:
  `FMCWDopplerProcessorOpTest.LiveOverrideCenterZeroDopplerMidStream`;
  `ClusterDBSCANOpTest.LiveMinPointsRevertsToParamsAfterMiss` /
  `LiveMaxClustersRevertsToParamsAfterMiss` / `LiveMinPointsSkewedCaptureMissesExact`;
  `MVDREstimatorOpTest.LiveMaxSourcesAppliesForOneTickAndReallocates`; and
  `num_sources` / `source_count_pfa` / `max_sources` on MUSIC, Root-MUSIC, and
  ESPRIT. Prefer a structural observable (an output extent or a reported
  integer) over a float payload comparison — see the MVDR caveat below.
- **Enum `[live]` overrides do not**, and that is now the only blocker for the
  remaining live-override stubs. Tracked separately as **§12** — the failure is
  a payload-identity gap independent of this harness work, so the four skipped
  enum stubs are accounted for there rather than here.
- `SnapshotGatherOpTest.LiveSnapshotWindowAppliesForOneTick` now drives a
  ChannelPod tuple under the production `OnAligned` + Exact contract and
  verifies the following miss reverts to Params. `PhaseShiftBeamformerOp` /
  Beamscan geometry / grid / taper live Graph tests are separate payload cases.
- **Runnable minimal Graph reproducer** (no radar domain logic):
  [`tests/cpp/utils/test_live_override_repro.cpp`](../../tests/cpp/utils/test_live_override_repro.cpp)
  — label `ea_repro`, not part of default `unit`:
  `ctest --test-dir build -L ea_repro --output-on-failure`
  or `./build/tests/cpp/test_live_override_repro`. Control (tensor-only)
  passes; dual-OnClock / co-emit wirings fail with
  compile rejection or runtime diagnostics (see test stdout +
  `/tmp/live_override_repro_*.json`).
- `MVDREstimatorOpTest.LiveMaxSourcesAppliesForOneTickAndReallocates` now passes;
  its geometry / grid / `spatial_smoothing` live Graph tests are active.
  `LiveDiagonalLoadingAppliesForOneTick` stays skipped, but **not** for an SDK
  reason: δ is observable only in float values, and MVDR corrupts later runs
  in the same process, so any two-run value comparison is untrustworthy. See
  `mvdr_estimator_op/OPEN_ITEMS.md` item 6 — an operator-state bug, not a
  port-binding one.
- `MUSICEstimatorOpTest.LiveNumSourcesAppliesForOneTick` /
  `LiveSourceCountPfaAppliesForOneTick` /
  `LiveMaxSourcesAppliesForOneTickAndReallocates` now pass; its geometry / grid /
  `spatial_smoothing` live Graph tests are active. Params also cover selection
  (`NumSourcesConfigurableFromParams` /
  `SourceCountPfaConfigurableFromParams` /
  `MaxSourcesConfigurableFromParams`).
- `RootMUSICEstimatorOpTest` / `ESPRITEstimatorOpTest`
  `LiveNumSourcesAppliesForOneTick` / `LiveSourceCountPfaAppliesForOneTick` /
  `LiveMaxSourcesAppliesForOneTick` now pass; their geometry /
  `spatial_smoothing` live Graph tests are active.
  Params cover the same trio of configurable-from-Params tests.
- `FeatureAssemblyOp` has no `[live]` params. Static `features` selects its
  required aligned data sources; unselected gather Tensor Inputs are
  non-member `Exact` + `SafeDefault` entries under §11. Graph tests cover the
  selected-source joins.
- `DetectionReductionOpTest.LiveRepresentativeAppliesForOneTick` /
  `LiveScoreMethodAppliesForOneTick` and the `LiveFeatureWeights*` suite run.
  Params cover selection and the compiled optional-`points` join; the custom
  `ScoreFn` specialization also runs through the Graph harness from a CUDA TU.
- `CFARDetectorOpTest` covers one-tick `pfa`, `threshold_scale`, `os_rank`,
  `max_detections`, cell-window, `nci_axes`, and `alpha_lut` overrides,
  including revert-to-Params and malformed vector payloads. Params cover the
  corresponding configurable-from-Params paths.
- Graph perf now measures inline POD transitions through the supported shape:
  CFAR `threshold_scale`, DBSCAN `min_points`, MUSIC / Root-MUSIC / ESPRIT
  `max_sources`, MVDR `diagonal_loading`, SnapshotGather `snapshot_window`,
  and VirtualArrayAssembly `wavelength`. `override_ms` /
  `override_delta_ms` are populated in the smoke JSON. Enum live ports and
  vector/descriptor lives remain untimed; unit tests cover their behavior.

**Next.** Generalize the range-processor `nfft` recipe (manual partitions
for the control-plane source, one shared transmit stamp, `Exact{}` +
`SafeDefault{}`) into `sdk_harness.hpp` and unskip the operator Graph
override tests group by group.

**Status:** open — Graph harness coverage only; the contract shape is
settled (L16 / L17, see
[`TEMPORAL_CONTRACTS.md`](TEMPORAL_CONTRACTS.md)). The one wiring probed end
to end is a device Tensor path plus a co-timed POD override from a manually
partitioned control-plane source, which works. Earlier failure tokens
(`RUNTIME_PORT_BINDING_UNAVAILABLE` /
`RUNTIME_EXECUTION_LANE_PLAN_INVALID`) came from the unpartitioned and
co-emit wirings, and the minimal reproducer still shows them; keep it until
GA re-test. Do not reclassify as durable. Perf timing is the exception: POD
`override_ms` is measured for the seven operators listed above through that
same supported shape (§8), so what remains open here is the one-shot
unit tests. Enum transport moved to §12.

---

### 2. Unique port names (inputs and outputs share one namespace)

**Class:** Durable HSDK 5 (confirmed on EA; treat as permanent unless SDK
docs reverse it).

**What changed.** HSDK 5 forbids reusing the same name for an input and
an output on one operator. 4.x pipelines often mirrored the channel name
(`range_profiles` in → `range_profiles` out).

**Interim.** Physical output is `range_profiles_out`; logical channel in
`OPERATOR_API.md` notes the rename (`[EA HSDK5 ONLY]`). Same pattern will
apply to every other operator that previously reused an input name on the
output.

**Next.** When porting each op, pick an explicit `*_out` (or other unique)
name, update `OPERATOR_API.md` in the same change, and call it out as a
contract change. Decide module-wide naming once enough ops are ported
(before dropping the EA fence).

**Status:** open — pattern established; apply consistently on each port.

---

### 3. Device Tensor producers require `bind_tensor_output_device`

**Class:** Durable HSDK 5 (runtime requirement on EA).

**Symptom.** Skipping
`CompileOptions.deployment.bind_tensor_output_device(...)` for a device
Tensor producer yields `RUNTIME_PORT_BINDING_UNAVAILABLE` at run time.
The same diagnostic fires when a bound producer publishes a Tensor that
is **not** from that port’s pool loan — e.g. HSDK 4–style zero-copy
republish of a pre-staged device buffer via `Output::emit(Tensor)`.
Probed 2026-08-17 on `perf_virtual_array_assembly` xl
(`PerfPublishMode::kZeroCopy` in `perf_graph.hpp`): run fails at
warmup; `emit` of `allocate_tensor` + D2D into the loan succeeds.
Reusing one operator-owned buffer across ticks is also disallowed:
`emit(TensorOutputLoan)` consumes the loan each publish, and a foreign
`emit(Tensor)` hits the same binding diagnostic.

**Interim.** Graph harness helper `bind_device_pools` in
`tests/cpp/utils/sdk_harness.hpp` binds `source`/`out` and
`op`/`range_profiles_out`. Dual-IO graphs use
`bind_dual_io_device_pools` for `source0`/`source1` plus both op
outputs. Single-in dual-out graphs use
`bind_single_in_dual_out_device_pools`; optional extra unbound device
outputs (e.g. `SubarraySelectionOp.is_valid_out`) must still be bound
via the harness `extra_op_out_ports` argument or compile fails with
`IMPLICIT_DEVICE_ZERO_REJECTED`. Apps/tests that emit device tensors
must do the same for every such producer path. Perf drivers select the
source publish path via `PerfPublishMode` on `run_perf_graph*`
(default `kCopy` = D2D into the bound pool). `kFillInPlace` (loan +
write; currently zero-fill) is legal and used by
`perf_virtual_array_assembly` because VAA gather/average is
value-agnostic. `kZeroCopy` remains only to re-probe GA.

**Accepted comparison artifact (2026-08-17).** HSDK 4’s perf source
republished one staged device handle every tick with no memcpy
(`perf_source.hpp`). Graph end-to-end medians therefore include a
mandatory per-tick publish write/copy that 4.x did not. Against
`perf_results/perf_virtual_array_assembly.json` (Holoscan 4.3.0) with
`kFillInPlace`: `virtual_dedup_lg` ≈ **1.04×** (+0.025 ms),
`virtual_dedup_xl` ≈ **1.31×** (+1.01 ms). With default `kCopy` the xl
gap was ≈ **2.28×**. This is accepted as a Graph harness / pool
publish-path regression, not a VAA kernel regression — no further chase
unless GA allows foreign-handle publish or loan reuse.

**Status:** open — document per new harness/app as they port; confirm
whether public GA softens discovery or keeps explicit bind; re-probe
zero-copy republish if pool semantics change. VAA 4.x→5.x large-rung
gap accepted as above.

---

### 4. Writable device tensors: do not import raw `DLManagedTensor*`

**Class:** Durable HSDK 5.

**Symptom.** Importing a raw `DLManagedTensor*` is permanently read-only
in HSDK 5; writers need a `DLManagedTensorContext` + owned device memory
(see `radar_common/tensor/dlpack_buffer.hpp`). Fixtures and simulators
must use `mutable_data()` / the writable path.

**Status:** open — pattern landed for the pilot; watch for regressions
when porting remaining test fixtures and apps.

---

### 9. Live override ports cannot carry `std::vector` (POD / Tensor only)

**Class:** Durable HSDK 5 (`PublishedPayload` admission).

**What changed.** Typed `Input<T>` / `Output<T>` admit scalars/enums,
`Tensor`, schema-identity types, or experimental ChannelPod contract
types. `std::vector` (and other owning containers) are rejected at
compile time.

**SDK documentation.**
[Payload type rules](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/payload-type-rules.md)
is the canonical list of admitted families, and states that everything
outside them — "references, cv-qualified types, pointers, arbitrary
classes, `std::string` and string views, backend handles, and raw memory
buffers" — is rejected at the port declaration.
[Experimental raw POD payloads](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/advanced/experimental-pod-payloads.md)
covers the ChannelPod escape hatch we use for `SnapshotWindowOverride`:
padding-free exact tiling, fixed-width scalars only, and "no external
ownership" (no pointer, string, nested object, or Tensor members).

What those pages recommend *instead* is a generated FlatBuffers schema
type, the documented path for messages with string, vector, optional, or
nested fields
([custom FlatBuffer schemas](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/custom-flatbuffer-schemas.md),
[schema payloads](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/schema-payloads.md)).

**Settled policy (L18, 2026-08-23).** Use generated FlatBuffers
descriptor-Tensor schemas for shared owning descriptors, ChannelPod wrappers
for bounded fixed-layout scalar tuples, and rank-1 pageable-host
`Input<Tensor>` payloads for variable-length scalar vectors. Every vector
Tensor has a finite maximum element count captured at operator construction.
The receive boundary validates rank, scalar dtype, pageable-host memory kind,
and extent before converting to operator-owned state. Logical Params/YAML
types remain the descriptor, tuple/vector, or `std::vector<T>` types documented
in `OPERATOR_API.md`; generated objects, ChannelPod wrappers, and Tensor
handles are physical port types only. The host-Tensor vector path is a narrow
control-plane exception to the normal rejection of pageable-host data
Tensors. Exact / SafeDefault and single-tick override semantics are unchanged.

**Measured 2026-08-21 (`tmp-app/fbs/`).** It works for exactly the shapes
we need. `fbs_live_override.cpp` carries this operator's two demoted
overrides — `waveform` (optional scalars + `[Complex64]`) and
`range_window` (enum, optional scalar, `[double]`) — as generated owning
types on ordinary `[live]` ports, reusing
`add_live_override_to_contract` / `receive_optional_value` unchanged,
with `LatestWithin` selection, `SafeDefault`, and per-field round-trip
equality against the descriptor the control plane meant to install. The
costs are all real but bounded: (a) `TRAITS_MAX_SERIALIZED_SIZE` is a
hard per-value cap chosen in CMake — an oversized coefficient vector
fails at `emit()` with `kResourceExhausted`, so every unbounded `[live]`
vector needs a declared bound, the same class of decision as
`MaxTensorExtents`; (b) the port payload is a mutable, unvalidated
generated type, so each descriptor needs a marshaling boundary (~15 lines
per type) which is also where its invariants get enforced; (c) the
generated headers and `holoscan::schema` linkage cannot live in
`radar_common` (Holoscan-free), so the schemas and conversions need a
home; (d) the object-API types are C++-only, so a Python app would drive
these ports through Params. Adopting it is still a build-system and
contract change — raise it with the team; the cheaper alternative to
weigh is stripping the bulk arrays out of the descriptors and routing
them as their own `Input<Tensor>`.

**Measured 2026-08-23: a schema may carry the Tensor itself, which
collapses those two options into one.** `holoscan_add_flatbuffer_schema`
has a second mode, `DESCRIPTOR_TENSOR`, in which a field typed
`holoscan.schema.Tensor` (from the SDK's `tensor.fbs`) becomes a
`std::shared_ptr<holoscan::Tensor>` member of the generated owning type
with a generated `TensorFieldSelector`, its own shape / dtype / memory
kind / allocation bound, and publication through `allocate_tensors()` /
`write_all()` / `emit_tensors()`. The Tensor bytes are never serialized
into the FlatBuffer — the SDK examples (`descriptor_tensor_schema`,
`multi_tensor_schema` under `examples/flatbuffer_schemas/`) state they
move by buffer handle with no staging copy, including across a
process-partitioned graph. A scratch schema mixing our exact field set
(`WindowType` enum, optional `double`, `bool`, `[double]`, plus one
Tensor) generates cleanly, so metadata and bulk array coexist in one
message. Consequences for this decision: cost (a) above mostly
evaporates — `TRAITS_MAX_SERIALIZED_SIZE` and
`TRAITS_DESCRIPTOR_METADATA_CAPACITY` (hard-capped at 65536) then bound
only the metadata, while the array is bounded by the per-field tensor
allocation we already declare via `MaxTensorExtents` — and it is
strictly better than a separate `Input<Tensor>` for the bulk array,
because the descriptor and its array arrive as one sample under one
`capture_time` instead of needing a second join. Two constraints found:
descriptor mode rejects vectors of structs or tables
(`reference_samples: [Complex64]` fails with
`vectors of tables or structs are unsupported by descriptor traits
generation`, so a measured dechirp reference has to *be* the Tensor, or
flatten to an interleaved `[double]`, or take the hand-written-codec
escape hatch) and it forces `--require-explicit-ids` across the schema
closure. Note also that the `descriptor_tensor_schema` README claims
"exactly one Tensor member per schema" while `multi_tensor_schema`
carries three (two device, one host) — verify multi-Tensor before
depending on it.

**Measured 2026-08-23: descriptor Tensor members cannot be null.** Ordinary
publication with a null member is rejected with `kInvalidArgument`. A rank-1
zero-length Tensor allocated and published with `emit_tensor` succeeds and
arrives as a present Tensor with `nbytes() == 0`. The schemas
therefore always carry the Tensor: zero-length is the absent optional-array
representation, nonzero is present, and null is invalid. This is a measured
transport rule, not by itself ratification of the schema payload design.

**Two EA generator workarounds remain.** Descriptor-traits generation emits
invalid C++ for FlatBuffers nullable scalars, so the two schemas encode each
optional double losslessly as a value plus an explicit `has_*` field rather
than reserving a valid numeric sentinel. The generated traits also place their
unqualified `CreateTensor` forwarding block after the generated root header;
`radar_schemas` post-processes the build output to include its forwarding
overload first. Neither workaround changes committed generated files (none are
tracked). Re-probe both against the public SDK and remove or version the schema
workarounds before the EA cleanup.

**Artifact and packaging shape.** The
`.fbs` files are committed source, not generated: schema compatibility is
exact-match on canonical name + version + schema SHA, so the file *is* the
wire contract and every edit is a version decision. Everything flatc
emits (`*_generated.h`, `*_bfbs_generated.h`, the traits header) is build
output under the target's `OUTPUT_DIR`, regenerated on every configure /
build (the schema and its include closure are registered as
`CMAKE_CONFIGURE_DEPENDS`), so nothing generated is tracked. No new build
dependency comes with it: `find_package(holoscan)` already hard-requires
the schema tooling — it `find_dependency(Flatbuffers … EXACT)`,
fatal-errors on a missing `share/holoscan/schemas`, and defines
`holoscan::fbs_tools`; the generator then re-verifies `flatc --version`
against the pinned build. One schema target must own each shared
descriptor (the same two are `[live]` on range, Doppler, range-Doppler,
and MIMO compensation, and duplicate `.fbs` copies would mint two
incompatible identities for one descriptor), which argues for a
`radar_common`-sibling module rather than per-operator schemas. The
sharp edge is the install surface: the payload type appears in the
operator's public header (`Input<radar_fbs::FMCWWaveformT>`), so the
generated headers join the module's installed API next to
`fmcw_range_processor_op.hpp` and must carry the same include spelling in
the build and install trees — a generated header becomes a compatibility
surface for downstream `find_package(holoscan_radar)` users, and the
descriptor's field list then lives in three places (struct, param schema,
`.fbs`) needing a round-trip test to hold them together.

**Current implementation.** `SnapshotGatherOp` keeps
`Params.snapshot_window` as `std::vector<int64_t>` (OPERATOR_API.md / YAML).
The `[live]` port is `Input<SnapshotWindowOverride>` — a fixed
`{doppler, range}` int64 POD with `ChannelTypeIdentity` /
`ChannelPodLayout` ([EA HSDK5 ONLY] in `snapshot_gather_op.hpp`). Normalize
empty/size-1 vectors on the Params path before seeding the POD override.
The same ChannelPod category applies to other bounded fixed tuples, including
CFAR's `(doppler, range)` `guard_cells` and `training_cells`; their logical
Params remain unchanged.

**Every documented `[live]` port now exists.** No operator falls back to
Params-only for a port `OPERATOR_API.md` tags `[live]`. The mapping as
implemented:

| Physical category | Ports |
| --- | --- |
| Generated FlatBuffers descriptor-Tensor schema | `array_geometry` (`Input<WireArrayGeometry>`) on all eight angle operators; `array_taper` (`Input<WireWindowSpec>`) on Beamscan and PhaseShiftBeamformer; `range_window` / `doppler_window` / `waveform` on the three FMCW processing operators and MIMO compensation |
| ChannelPod fixed tuple | SnapshotGather `snapshot_window` (`Input<SnapshotWindowOverride>`); `spatial_smoothing` (`Input<SpatialSmoothingOverride>`, a `{count, first, second}` int64 triple with `count` in 0–2 selecting the 1-D / 2-D counts); CFAR `guard_cells` / `training_cells` |
| Bounded rank-1 pageable-host `Input<Tensor>` | `az_angles` / `el_angles` (bounds `max_az` / `max_el`), `subarray`, DBSCAN `epsilon` (bound `max_features`), CFAR `nci_axes` / `alpha_lut`, DetectionReduction `feature_weights` |
| Plain scalar / enum `Input<T>` | the Exact scalar and enum lives (`nfft`, `pfa`, `max_sources`, `peak_interpolation`, `clutter_filter`, …) |

Vector ports read through `read_host_config_vector<T>`, which validates rank,
scalar dtype, pageable-host memory kind, and extent against the bound before
converting. Where the bound is a `MaxTensorExtents` field, `require_extent`
fails with a named error if the application did not supply it; DBSCAN checks
its `max_features` bound on arrival instead, so a graph that never sends an
`epsilon` override compiles without one. Logical Params stay the types
`OPERATOR_API.md` documents, and single-tick override semantics are unchanged.

The committed-source/generated-output boundary, exact identity and evolution
rules, `MaxTensorExtents` bounds, bridge/install/test sync surface, and
C++-only wire-type limitation are in
[`SCHEMA_PAYLOADS.md`](SCHEMA_PAYLOADS.md). Python continues through Params.

**One connection the categories do not make.** `ArrayGeometry` rides *into* an
operator as a generated wire type but comes *out* of
`VirtualArrayAssemblyOp.array_geometry` as a packed float64 `Output<Tensor>`
(§10, the sanctioned descriptor output). The types differ, so VAA cannot drive
an angle operator's `array_geometry` port directly — that override has to come
from an application-side wire-schema producer. Closing it means either giving
the descriptor output a wire payload or bridging the two representations; it
is not a Params-only fallback and should not be documented as one.

**Next.** Extend Graph Exact coverage over the landed ports (§1), and decide
the VAA descriptor-output representation above.

**Status:** policy settled (L18); implementation landed for every `[live]`
port. Remaining work is test coverage and the VAA descriptor-output type.

---

### 10. Descriptor output ports (`ArrayGeometry`) are not PublishedPayloads

**Class:** Durable HSDK 5 (`PublishedPayload` admission).

**What fails.** `ArrayGeometry` owns `std::vector<double>` and
`std::optional<ArrayGeometryType>` — not a ChannelPod (requires standard
layout + trivially copyable) and not a SchemaIdentity type. Declaring
`Output<ArrayGeometry>` (or `Input<ArrayGeometry>`) fails the
`PublishedPayload` concept at compile time. The same blocker applies to
`VirtualArrayAssemblyOp.array_geometry` when that op ports. SDK
documentation for the admitted families and the POD restrictions:
[payload type rules](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/data-and-payloads/payload-type-rules.md),
[experimental raw POD payloads](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/advanced/experimental-pod-payloads.md)
(§9 has the FlatBuffers-schema alternative these pages point to).

**Contract.** Operators that *compute* geometry at runtime
(`SubarraySelectionOp`, `VirtualArrayAssemblyOp`) expose it on a data-plane
port — Params / side-channel return is not acceptable for runtime-varying
geometry. `VirtualArrayAssemblyOp.array_geometry` is required.
`SubarraySelectionOp.array_geometry` is latched-optional for snapshot-only
consumers; while consumed, it emits on the same ticks as `snapshot_out`.

**Interim.** `SubarraySelectionOp` emits `array_geometry` as a float64
device Tensor of shape `(num_elements + 1, 3)` via
`radar_common/descriptors/array_geometry_tensor.hpp` (positions in rows
`[0, N)`, metadata row `[wavelength, geometry_type_code, 0]`). Documented
in `OPERATOR_API.md` under SubarraySelectionOp. Live *input* overrides of
`ArrayGeometry` ride a generated wire type instead (§9), so a descriptor
output cannot currently feed a descriptor override port.

**Next.** Prefer a durable schema / ChannelPod-compatible wrapper (or SDK
admission for owning containers) shared with VirtualArrayAssembly; restore
typed `Output<ArrayGeometry>` when available. Do not invent a second
descriptor type.

**Status:** open — SubarraySelection + VirtualArrayAssembly Tensor pack
landed; restore typed `Output<ArrayGeometry>` when the SDK admits owning
containers / a durable ChannelPod schema.

---

### 11. No optional input: a contract entry cannot tolerate an unwired port

**Class:** SDK feature request (EA gap; the semantics are already reserved).

**What fails.** An operator with an optional `[live]` override cannot be
written so that connecting the port is the *only* thing an application does.
The contract's input set must match the wiring exactly — a declared entry with
no producer is rejected at compile with `TEMPORAL_INPUT_UNCONNECTED`, and a
wired port absent from the contract kills the second activation with
`RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1). Nothing lets the operator resolve
that itself: its virtual surface is `setup` / `contract` / `start` / `stop` /
`compute`, `contract()` is const with access only to `params()` and its own
port handles, `InputSpecBuilder` offers no `optional()`, `Input<T>` exposes
only compute-time selection state, and the graph-side `InputPortConfigurator`
reaches only `queue_depth` / `expects_tensor` / `memory_conversion` /
`diagnostic_alias` — never a selector. `on_missing` does not help: it governs a
*declared and connected* input that produced no value.

**Requested.** Permit a contract entry for an input with no producer, and
report the reason the SDK already defines for it —
`SelectionReason::kNotConnected` ("No admitted producer is connected to the
input", `activation.hpp`), which is declared and never raised anywhere in the
sources. Routing it through the existing `on_missing` policy needs no new
vocabulary: `Skip{}` suppresses the activation, `SafeDefault{}` runs the
degraded path. An `InputSpecBuilder::optional()` marker would be an acceptable
alternative shape, but the contract-level form composes better with the
`on_missing` policies that already exist.

**Why the workarounds are all unacceptable.** Measured in
`tmp-app/wiring_inference.cpp`; the trade is structural, not a matter of taste:

| Workaround | Cost |
| --- | --- |
| `expect_*` parameter gating the `contract()` entry | The application states the same fact twice (the `add_flow` and the flag) and a disagreement is a compile or runtime failure |
| Application-side derivation from `Graph::view()` | Removes the disagreement but still requires a non-standard pass before `compile()` |
| Wire an idle producer that never emits (making the port mandatory) | Legal and parameter-free, but wiring a deliberately dead operator is not an idiom a Holoscan author would recognize, and it makes "unwired" indistinguishable from "has not published yet" (both `kNeverSeen`) |
| Fan-in a stand-in alongside a real producer | Impossible — `add_flow` throws `GRAPH_INPUT_FANIN_UNSUPPORTED` |

**Answered in the SDK (2026-08-24), not yet shippable here.** SDK branch
`deustice/5x/optional-port-contracts` (CONTRACT-020 / COMPILE-068 /
RUNTIME-050) implements the requested form, and `tmp-app/optional_input_ports.cpp`
measures it against this module's shapes:

| Case | Measured |
| --- | --- |
| `OnEach{data}` + POD override `Exact{}` + `SafeDefault{}`, unwired | Compile **accepted** with warning `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT` (vertex + port scoped); every activation reports `kSafeDefault` / `kNotConnected` and the read returns `kInputEmpty` |
| Same, wired with a co-stamped companion | `kSelected` / `kNone` and the value — unchanged from the `expect_*` form |
| Optional **Tensor** input (`is_valid`-shaped), unwired | Same acceptance and `kNotConnected` — the rule is about the *selector* being scalar `Exact`, not the payload, so the optional-Tensor family is covered too |
| `OnAligned{…}` trigger + unwired `Exact{}` override | **Rejected**, `TEMPORAL_INPUT_UNCONNECTED` — `OnAligned`, `OnWindow`, and predicate inputs excluded (**since fixed**, see below) |
| `OnAligned{…}` trigger + **wired** `Exact{}` override | **Accepted**; every activation reports `kSelected` / `kNone` and delivers the co-stamped value, exactly as under `OnEach` |

Consequences for this module at that point: every `OnEach`-triggered operator
could drop `expect_*` outright, and the optional-Tensor omissions (`is_valid`,
FeatureAssembly gathers) became ordinary contract entries. `SnapshotGatherOp`
and `DetectionReductionOp` trigger `OnAligned`, so they had to keep `expect_*`.

The last two rows together were the precise reason, and it is worth stating
plainly because it is easy to misread as "the override does not work under
`OnAligned`". It works. What was not expressible was its *absence*: an
unconditional entry via `add_live_override_to_contract` would make the override
port **mandatory**, so every application instantiating `SnapshotGatherOp` would
have to wire a `snapshot_window` producer or fail to compile. `expect_*` is what
bought back the optional case, and the same one build still served both wirings.

Coverage caveat, since resolved: the wired-aligned row was originally measured
only by the probe. `DetectionReductionOp`'s enum live-override tests (§12) now
wire an `Exact{}` override under its `OnAligned` trigger and assert the value
lands, so that row is asserted by an operator suite rather than by construction.

**Why the `OnAligned` exclusion was worth asking for.** An override is not an
alignment member and its absence cannot make the alignment unsatisfiable, so
extending the rule to non-member inputs of an `OnAligned` trigger was a strictly
narrowing fix rather than new vocabulary. Until it landed, `SnapshotGatherOp` and
`DetectionReductionOp` were the module's only two `expect_*` holders — recorded
as an exception in [`OPERATOR_API.md`](OPERATOR_API.md)'s parameter override
convention rather than as a module-wide idiom.

**Validated end to end (2026-08-27) against branch tip `906fc1247`.** The
`expect_*`-free `FMCWRangeProcessorOp` from `deustice/optional-input-ports`
builds against that SDK and **all 18 of its tests pass**, including the graph
cases whose overrides are left unwired. The same binary's two representative
cases (`NoRangeBinsWithoutWaveform`, `LiveNfftExactMatchChangesBinCount`) fail
on the `v5.0.0.1` EA SDK with `TEMPORAL_INPUT_UNCONNECTED` on the unwired
`range_window` port and `"admitted": false`. That closes the earlier
measurement gap — "wired override still selects" is now asserted by the
operator's own tests, not only by the probe.

**Adopted (2026-08-27).** The devcontainer base image now pins the SDK to
`deustice/5x/optional-port-contracts`, the module has migrated to that branch's
Tensor spec API, and `expect_*` is gone from every `OnEach`-triggered operator
in the `fmcw`, `angle_estimation`, `detection`, and `clustering` groups along
with the corresponding test wiring (the two `OnAligned` operators followed on
2026-08-28, below). Tests that used to set an `expect_*` flag now pass an
explicit wiring struct to their graph helper, so the fact that used to be
stated twice (the flag and the `add_flow`) is stated once.

**Getting here required the Tensor spec migration.** The branch sits on
`main-5x`, which has reshaped the Tensor spec surface past `v5.0.0.1`, so the
module could not build against the feature until it moved. The mapping is
mechanical and is now applied module-wide behind the `tensor_input` /
`tensor_output` / `pooled_tensor_output` helpers in
`radar_common/sdk/tensor_io.hpp`:

| `v5.0.0.1` (this module) | `main-5x` |
| --- | --- |
| `TensorPortLayout{.memory_kind, .dtype, .rank}` | `TensorRepresentation{…}` nested in `TensorInputSpec` / `TensorOutputSpec` |
| `TensorAllocationBounds{.max_rank, .max_byte_span}` | `TensorBounds{…}` on the spec (pool sizing became `TensorPoolBounds`) |
| `.produces_tensor(layout).tensor_allocation(bounds)` | `.produces_tensor(spec)` with `.storage = TensorOutputStorage::kRuntimePool` |
| `.expects_tensor(layout)` | `.expects_tensor(TensorInputSpec{.representation = …})` |

The migration also surfaced two stricter compile-time checks on the newer SDK:
a producer's pool `max_rank` must agree with the rank its `TensorRepresentation`
declares, or the edge fails `GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT` at compile.
Two test sources were over-declaring rank and were corrected; one CFAR test that
deliberately feeds a malformed rank-2 control tensor now sees it rejected at the
edge rather than inside `compute()`, which is a strictly earlier and better
diagnosis of the same mistake.

**Consequence for the branch.** This is a one-way door: the module no longer
builds against the `v5.0.0.1` EA tag. Cherry-picking the feature back onto
`v5.0.0.1` was tried and conflicts in the diagnostic registry and
`run_session.cpp` (312 commits of drift), so the pin moves forward rather than
the feature moving back. Reasoning and the full measurement set:
[`TEMPORAL_CONTRACTS.md`](TEMPORAL_CONTRACTS.md).

**The `OnAligned` exclusion was fixed too (2026-08-28).** `feat(runtime): allow
optional aligned side inputs` on the same SDK branch (tip `f0cb828d4`) admits an
unwired `Exact{}` + `SafeDefault{}` input that is not a member of the aligned
presence set. Re-run of the probe against that tip:

| Case | Before | Now |
| --- | --- | --- |
| `OnAligned` + unwired `Exact{}` override | rejected, `TEMPORAL_INPUT_UNCONNECTED` | accepted with warning `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT`; reads `kSafeDefault` / `kNotConnected` / `kInputEmpty` |
| `OnAligned` + wired `Exact{}` override | accepted, `kSelected` / `kNone` | unchanged |

That makes the two trigger shapes behave identically, so the reason
`SnapshotGatherOp` and `DetectionReductionOp` kept `expect_*` — that an
unconditional contract entry would make their override ports *mandatory* — no
longer holds.

**Retired in this module (2026-08-28).** `SnapshotGatherOp.expect_snapshot_window`
and `DetectionReductionOp.expect_representative` / `expect_score_method` /
`expect_feature_weights` are gone, and with them the shared
`maybe_add_live_override_to_contract` / `resolve_expected_override` /
`resolve_expected_override_via` helpers, which had no other callers. No operator
in the module carries an `expect_*` parameter now, under any trigger shape. The
`DetectionReductionOp` live suite wires all three override ports in every case
— including cases that never emit on one — and therefore asserts that an
enabled-but-silent aligned side input does not stall the trigger.
`DetectionReductionOp`'s `max_features` check moved from a Params-time
gate to the `feature_weights` converter, so it now fires only for a tick that
actually carries an override.

**Status:** closed, in the SDK and in this module.

---

### 12. Enum-typed `[live]` override edges fail runtime port binding

**Class:** EA limitation — **fixed in the SDK (2026-08-28)**, see Resolution.

**What fails.** A Graph edge whose payload is an `enum class` compiles and then
kills the run. Holoscan labels such a payload `kLocalEnum`
(`holoscan/core/payload_identity.hpp`), admitted "only on local typed Graph
edges" — but a local edge is exactly what fails, while a fixed-width integer of
the *same underlying type* on the *same* edge binds and runs. Connecting the
edge is sufficient: the producer's `Output<Enum>::emit` returns `kNotReady` even
on an activation that publishes nothing on that port, so no application can wire
an enum `[live]` override today.

**Reproducer.** A/B in one binary, graph held constant and only the sideband
type varied (co-emitted CUDA Tensor + scalar sideband, one explicit partition,
`LatestWithin` + `SafeDefault` on the sideband):
[`tests/cpp/utils/hsdk_enum_tensor_bind_repro/`](../../tests/cpp/utils/hsdk_enum_tensor_bind_repro/).
It is drop-in for the SDK tree — copy the directory, it carries its own
`CMakeLists.txt` and needs only Holoscan + CUDA. In this tree:
It is a standalone project rather than a `ctest` entry, so it stays buildable
against any SDK tree: `cmake -S tests/cpp/utils/hsdk_enum_tensor_bind_repro
-B <dir> && cmake --build <dir>`, then run the binary. Originally measured on
`deustice/5x/optional-port-contracts` tip `906fc1247`:

```text
uint32_t:    plan_ok=true captured=3 diagnostics=
enum class:  plan_ok=true captured=0 diagnostics= RUNTIME_PORT_BINDING_UNAVAILABLE
```

So compilation accepted the enum edge and the failure was entirely at bind /
run time, on the first generation, with the diagnostic scoped to a vertex and
port.

**What it blocked.** Every enum-typed `[live]` port in the module — eight of
them, on seven operators. Four surfaced as live-override test stubs; the other
four were latent, since no test wires them:

| Port | Type | Trigger | Test |
| --- | --- | --- | --- |
| `ClutterFilterOp.method` | `ClutterFilterMethod` | `OnEach` | `LiveMethodOverrideAppliesForOneTick`, `LiveMethodRevertsToParamsAfterMiss` — landed |
| `BeamscanEstimatorOp.peak_interpolation` | `PeakInterpolation` | `OnEach` | `LivePeakInterpolationAppliesForOneTick` — landed |
| `DetectionReductionOp.representative` | `RepresentativePolicy` | `OnAligned` | `LiveRepresentativeAppliesForOneTick`, `LiveRepresentativeAppliesForOneTick` — landed |
| `DetectionReductionOp.score_method` | `ScoreMethod` | `OnAligned` | `LiveScoreMethodAppliesForOneTick` — landed |
| `FMCWRangeProcessorOp.fft_norm` | `FftNorm` | `OnEach` | none authored |
| `FMCWDopplerProcessorOp.fft_norm` | `FftNorm` | `OnEach` | none authored |
| `FMCWRangeDopplerProcessorOp.fft_norm` | `FftNorm` | `OnEach` | none authored |
| `FMCWRangeDopplerProcessorOp.clutter_filter` | `ClutterFilterMethod` | `OnEach` | `LiveClutterOverrideChangesShape` stubbed, skipped citing §1 |

The three `fft_norm` ports remain exercised from `Params` only, and the fused
`clutter_filter` stub still skips against the §1 harness item — both are now
ordinary authoring gaps rather than blocked ones. `ClutterFilterOp.method` is
the worked example to copy for the fused port, since it is the same enum.
Graph perf measures `override_ms` for `ClutterFilterOp.method`; remaining enum
lives are not yet wired (§8).

**What rules out the alternatives.** The A/B holds the graph shape fixed, which
eliminates the co-emit, the partition cover, and the device Tensor binding. It
was then reproduced independently on three operators with unrelated fixtures and
both trigger shapes — `ClutterFilterOp` and `BeamscanEstimatorOp` under `OnEach`,
`DetectionReductionOp` under `OnAligned` triple-in — which eliminates the
trigger, the fixture, and the pool binding. Scalar and bool `[live]` overrides on
those same operators pass (§1). The enum type is the only common factor.

**Requested.** Admit `kLocalEnum` on a local typed Graph edge as the identity
docs already describe — or, if enum payloads are intentionally not transportable,
reject the edge at *compile* with a diagnostic naming the payload identity,
rather than accepting the plan and failing at bind.

**Resolution (2026-08-28).** Fixed upstream by `fix(core): support local enum
edges` on `deustice/5x/optional-port-contracts` (this branch's SDK pin, tip
`f0cb828d4`). The reproducer now reads:

```text
uint32_t:    plan_ok=true captured=3 diagnostics=
enum class:  plan_ok=true captured=3 diagnostics=
```

No operator code changed — the enum ports were always declared exactly like
their scalar siblings, so the fix was invisible to this module. What changed is
the test surface: all four skipped stubs are now real tests that wire the enum
edge and assert the override takes effect mid-stream, reverts to Params on the next miss,
and reverts to the `Params` value when ephemeral. Because the two
`DetectionReductionOp` ports sit under an `OnAligned` trigger, they are also the
first tests to assert a wired enum override on that trigger shape.

**Status:** closed. Re-running the reproducer is the whole re-test, so re-run it
at the GA pin to confirm the fix survived.

---

### 13. Optional outputs: an unconsumed emit fails, and is diagnosed generically

Closed 2026-09-01, worked around module-side — see **Module resolution** below.

**Class:** EA limitation — worked around in the module; the residue is
diagnosis quality, not capability.

**What happens.** An operator that conditionally produces an output cannot ask
whether anyone is listening. `Output` exposes `name()`, the `emit` overloads,
and the allocate paths — there is no connection query, and `contract()` has no
output side at all (it declares triggers and inputs; outputs are declared in
`setup()`). So the emit can only be gated on something the operator can see,
such as "a waveform is configured", which is not the same condition as "the
port is wired". Publishing to an unconsumed output then fails.

**What works.** Emitting anyway is safe, provided the result is absorbed. Every
`emit` overload is `[[nodiscard]] expected<void, Error>` precisely so a caller
can decide, and the failure is confined to that call: the activation completes
and the tick's *other* outputs are delivered normally. Verified directly — with
the error swallowed, an undrained `range_bins` run finishes with zero
diagnostics and `range_profiles` intact. The trap is only in the module's own
reflex of wrapping every emit in `HOLOSCAN_TRY`, which propagates the error out
of `compute()` and fails the whole activation, so a graph that merely declined
an optional output produces nothing at all and says nothing about why.

**Module resolution (2026-09-01).** The fix generalized once the survey showed
the exposure was far wider than the conditional emits. Seventeen further ports
were documented `(optional)` in `OPERATOR_API.md` yet emitted unconditionally
under `HOLOSCAN_TRY` — `ClusterDBSCANOp.sizes`, `SnapshotGatherOp.is_valid`,
`BeamscanEstimatorOp.angle_spectrum`, and the `angle_spectrum` / `doa_multi` /
`source_count` / `eigenvalues` sets on MVDR, MUSIC, Root-MUSIC and ESPRIT. An
application that actually declined one of those died on its *first* tick. It
had gone unnoticed only because every app and test harness wires everything.

`OPERATOR_API.md` now carries an **Optional output convention** giving each of
the 27 optional outputs one of two dispositions, both of which absorb the
failed emit so no unwired port can fail a run:

- **`always`** (11 ports) — `(void)emit_prepared_tensor(...)`, silent. Used
  where the value is a byproduct of work another output needs anyway, so
  stopping saves nothing (an `angle_spectrum` is the same scan-grid pass that
  yields `aoa_estimates`; a MUSIC `eigenvalues` re-exports the eig its own
  projector requires).
- **`latched`** (16 ports) — a `LatchedOutput` in `radar_common/sdk/tensor_io.hpp`
  warns once on the first failed emit and then skips everything done solely for
  that port. It latches only on `kFailure`, the code an unconsumed output
  reports, so a transient fault costs one frame rather than the whole run;
  that is sound because wiring is fixed once the plan compiles. The largest
  saving is `MVDREstimatorOp.eigenvalues`, which sheds a batched Hermitian
  `matx::eig` that exists only to fill the port. CFAR's five outputs also latch:
  each skips its pool loan and D2D copy, while `detection_idx` additionally
  skips the exact-count host sync and compaction.

Required outputs keep `HOLOSCAN_TRY`, where a failed emit *is* an error.
`(void)` is the SDK's own idiom for a discarded `[[nodiscard]]` result (its
docs spell `add_flow` the same way).

Two stage kernels now take null output pointers to make the latch meaningful:
the MVDR and MUSIC peak-extraction passes write `doa_multi` and `source_count`
together, so the pass is skipped only once both have latched, and each kernel
null-guards its stores.

`emit_prepared_tensor` / `emit_device_tensor` now carry `[[nodiscard]]`
themselves. They previously did not, and neither does `holoscan::expected`, so
the wrappers were laundering away the protection the SDK put on `Output::emit`:
a bare call compiled silently and "deliberately ignored" was indistinguishable
from "forgot to check" in every operator. Adding it cost nothing — every
required emit site already used `HOLOSCAN_TRY` — and makes the `(void)`
compiler-enforced.

Locked by three tests, each running many ticks because a latched port still
takes a pool loan on the tick it discovers it is unwired, so a slot the runtime
failed to recycle would starve the operator a few frames in rather than on the
first: `UndrainedRangeBinsDoesNotFailTheRun` (conditional + latched),
`MVDREstimatorOpTest.UnwiredOptionalOutputsDoNotFailTheRun` (the previously
fatal class — wires `aoa_estimates` alone and leaves all four optional outputs
unwired), and `ClusterDBSCANOpTest.UnwiredSizesDoesNotFailTheRunOrDisturbLabels`
(also checks the surviving renumbering histogram still labels correctly once
the sizes-only one is gated off).

**What remains — diagnosis, and only if we want it.** Nothing reports an
undrained optional output. The SDK is silent at every phase: no compile
diagnostic, no bind warning, nothing but the bare return value. Its only
"unconnected" tokens (`TEMPORAL_INPUT_UNCONNECTED`,
`TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT`) are input-side, and no SDK log
mentions an unconsumed output. Nor could a report be precise: the unconsumed
emit is the generic `ErrorCode::kFailure` with an empty message, which a
genuine fault also produces. The docs' statement that an unbound output returns
`kNotReady` does not describe this case — `kNotReady` is returned only when the
binding pointer is null, and a bound-but-unwired output is not that. This is
the asymmetry left over from §11: inputs got both a specific `kNotConnected`
and a named warning, outputs got neither.

The tradeoff accepted with the silence: an output wired wrongly (one the author
meant to consume) now fails quietly, surfacing later as missing downstream
data. Judged acceptable — an optional port is optional.

**Simplification trigger.** A distinct error code for the unconsumed emit, or a
wiring query on `Output`. Either would make an undrained-output report precise
enough to be worth adding. Recorded here rather than raised upstream, by
decision.

**Status:** closed — worked around, with the residual diagnosis gap accepted
rather than papered over.

---

## Open — deferred port work

### 5. Allow-list: C++ operator first-pass complete

**Class:** Our deferred work (allow-list retained until EA cleanup).

All 19 operators are on `HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS`. The
allow-list remains until public HSDK 5 / cleanup gate so apps and packaging
stay gated intentionally. Demo apps still need
Holoviz (§6) and Graph-port of remaining app scaffolding. CFAR GO/SO/OS /
alpha_lut Graph unit suites and the `statistical`-labeled MC suite have
landed (phase-2 unit coverage). Perf drivers are on the separate
`HOLOSCAN_RADAR_HSDK5_PORTED_PERF` list (phase 3: all 19 drivers + split_vs_fused; POD `override_ms` restored §8; clutter `method` enum `override_ms` landed; remaining enum/§9 lives remain open).

**Status:** C++ first-pass + CFAR variant Graph units landed; allow-list
kept for EA gating / cleanup.

---

### 6. Python bindings, Holoviz demos, apps, perf packaging

**Class:** Our deferred work.

`HOLOSCAN_RADAR_BUILD_PYTHON` is forced off (`FORCE` cache set, so a stale
cache cannot resurrect it) until a fuller EA/GA Python operator surface +
`holoscan::pybind11` are available (current EA `holoscan.core` is
Tensor-only); `tests/CMakeLists.txt` therefore does not register the
pytest suites, and `python/holoscan/radar/__init__.py` imports the
unported operator modules optionally. Demo apps remain gated on Holoviz
**and** ported dependencies (§5) until Graph-ported; Holoviz itself is
still absent from the EA package.

**Interim live RD path (2026-08-17).** When `libX11` is present,
`HOLOSCAN_RADAR_HAS_X11` enables a Holoviz-free Graph viewer:
`RdMapToImageOp` (LUT indices) → `X11RdViewerOp` (Xlib + turbo).
No Dockerfile change. Smoke: `ctest -R X11RdViewer` (skips without
`DISPLAY`) — a raster-only graph plus an overlay graph wiring all eight
ports (detections, bins, cluster labels/representatives, aoa_estimates,
label_idx).

Overlays (2026-08-17, same day): the viewer now draws the Holoviz demo's
information layers itself, so nothing on the RD path waits on Holoviz.
Window size comes from `visualization.window_width/window_height`
(default 1280x720, resizable) and the frame is composited into a Pixmap
back buffer: nearest-neighbor-scaled turbo raster, framed axes with
physical range / velocity ticks read from the bin tensors, CFAR
detections as outlined dots colored by DBSCAN cluster (noise gray), each
cluster's `rep_idx` crosshaired, a per-cluster range/velocity legend
(fixed-width slots in **range-rank** order — the DBSCAN label is
size-ranked, so a label-ordered strip swapped columns on ~23% of frames
on the SISO scene; counters and readouts are space-padded so a digit
gained or lost cannot shift the line), a
scenario clock + detection/cluster counters, and the rolling
range-vs-time / velocity-vs-time panels under `plots.enable` (detection
dots plus each cluster's representative as a `+`). Cluster
colors use the Holoviz path's rank-based assignment, duplicated into the
Holoviz-free `radar_app/viz/overlay_style.hpp` (fold back at cleanup —
`viz.hpp` cannot be included without Holoviz). Text is `XDrawString`
(Latin-1, ASCII titles only) and presents on the compute thread. The
boresight panel followed (same day): `overlay_boresight` + the `az_lo` /
`az_hi` / `el_lo` / `el_hi` extents add a third right-column panel — an
instantaneous az/el scatter of the wired `aoa_estimates`, degrees on both
axes over the estimator's scan grid, markers colored by the same cluster
buckets as the RD map (SIMO / dense / MIMO; SISO has no angle stage). Panel
titles sit above their frames, since a trace riding the top of an axis draws
straight through a title placed inside. All Holoviz demo chrome is now
reproduced.

A header variant label followed: `overlay_label` + `label_caption` +
`label_names` add a leading status field naming the algorithm variant a
frame came from, fed by a `label_idx` port carrying an int64 `(1,)` index
into `label_names` (`fmcw_aoa_estimators`: `AoaMergeOp` emits the index of
whichever estimator port fired, beside `aoa_estimates`). A device tensor
rather than a scalar port on purpose — mid-stream Graph scalar/enum sources
are EA-open (§1) — and the name is right-padded to the widest configured
name so the rest of the right-aligned status line does not slide when the
variant changes.

Overlay ports are wired per config (`X11RdViewerConfig`): the contract is
`OnAligned`/`Exact` over exactly the wired set (rd_image +
detection_idx/range_bins/velocity_bins, plus labels/rep_idx outside
`peak_only`, plus aoa_estimates with the boresight panel and label_idx with
the variant label), since on EA a
registered-but-unwired port must stay out of `contract()` and a wired one
must be in it (§1). Alignment across `RdMapToImageOp`, CFAR, DBSCAN,
reduction, and beamscan/aoa outputs holds at window 0 — they share the
source activation time, even where members sit at different hop counts from
CFAR.

**EA gap: an operator with no wired port kills auto-partitioning.** The
angle demos created a `cfar_power_sink` `NullSinkOp` and then skipped wiring
it when the viewer consumed `cfar_power` instead. That isolated vertex makes
compile fail with `PARTITION_MANUAL_AUTHORING_REQUIRED` (the
`holoscan.partition.exact_cover/v1` rule declines; the only alternative is
authoring `graph.partition()` sets covering every operator exactly once).
The diagnostic names no vertex, so it reads as an unrelated temporal /
contract failure — it is not. Fix: always drain the port (fan-out to the
sink alongside the viewer) rather than conditionally constructing sinks.
Recheck after EA: auto-partitioning should tolerate an unreachable
operator, or the diagnostic should name it.

**`fmcw_siso` Graph pilot (2026-08-17).** Builds under
`HOLOSCAN_RADAR_HAS_RD_LIVE` (X11 and/or Holoviz). Application scaffolding
→ Graph: `SimAdcSourceOp`, report/null sinks, `run_fmcw_graph_app`, YAML →
`Params` + `MaxTensorExtents` (no `from_config()`). Optional X11 RD window
with the full overlay set when `visualization.enabled`. **EA workaround:**
`FeatureAssemblyOp` now uses a Params-derived `OnAligned` contract for selected
sources, so SISO wires its physical range/velocity axes directly to the library
operator. The earlier app-local required-port stand-in is retired.
Smoke: `./fmcw_siso fmcw_siso_smoke.yaml` (log-only) or enable
visualization under a real/`Xvfb` `DISPLAY`.

**Graph demo wave (2026-08-17, same day).** The same X11 overlay path
extends to the remaining sparse/dense FMCW demos under
`HOLOSCAN_RADAR_HAS_RD_LIVE`:

| App | Notes |
| --- | --- |
| `fmcw_simo` | Sparse SIMO: SnapshotGather → Beamscan; `FeatureAssemblyOp` (4 features); `cfar_power` → RD image; boresight from `beamscan.aoa_estimates` |
| `fmcw_dense_simo` | Dense: PhaseShiftBeamformer + Graph `AnglePeakPick`/`CubeAoaLookup`/`TrailingIdxColumns`/`RdAngleMaxOp`; 4-col `rep_idx` ClusterReport with az/el grids; boresight from `CubeAoaLookupOp` |
| `fmcw_mimo` | Graph `MimoTdm`/`MimoDdm` sources + `DdmReportVelocityBinsOp`; TDM/DDM compose branches; same sparse tail as SIMO; boresight over the virtual-array scan grid |
| `fmcw_aoa_estimators` | Five sparse estimators behind Graph `EstimatorRouterOp` (N snapshot outs) + `AoaMergeOp` (`OnAnyInput`); rotates beamscan/MVDR/MUSIC/Root-MUSIC/ESPRIT; X11 overlays + boresight + the active estimator's name in the header (`AoaMergeOp.estimator_idx` → viewer `label_idx`, also always drained) |

Shared helpers: `radar_app/app/graph_params.hpp` (YAML → Params),
DetectionReport/ClusterReport
`Config` flags (`report_aoa` / `report_detection_idx`) so optional ports
enter `contract()` only when wired. Beamscan `is_valid` stays unwired
(§1). Smoke YAMLs: `*_smoke.yaml` (+ `fmcw_mimo_ddm_smoke.yaml`) — note
these run log-only, so also run each app's default YAML under a real /
`Xvfb` `DISPLAY`: the viz subgraph is where the partitioning gap above
surfaced.

**Deferred:** ~~`fmcw_aoa_estimators` — Application `set_dynamic_flows()`
has no Graph equivalent~~ **landed (Graph-native, 2026-08-17):**
`EstimatorRouterOp` emits on one of five `snapshot_*` outs; idle estimators
never receive (`OnEach`); `AoaMergeOp` uses synthesized `OnAnyInput` over
five `aoa_*` ports and forwards the selected sample. Same X11 overlay path
as the other Graph demos. Holoviz / Python bindings remain deferred.

Shared `performance/cpp/utils/perf_graph.hpp` is the Graph harness;
legacy `perf_source.hpp` / `perf_sink.hpp` remain for reference only.
CMake gates drivers via `HOLOSCAN_RADAR_HSDK5_PORTED_PERF` (all drivers
listed — see §8). Python perf drivers stay deferred with bindings.

**Status:** open — phase 4: Graph demos landed for siso/simo/dense/mimo/aoa
(+ X11 overlays); Holoviz / Python still deferred; C++ Graph perf done.

---

### 7. Graph test harness: generalize to the remaining port shapes

**Class:** Our deferred work (+ depends on item 1 for live-override
coverage).

`tests/cpp/utils/sdk_harness.hpp` provides a WireFn single-IO runner
(`run_single_io_operator_graph_result` — ClutterFilter /
PhaseShiftBeamformer), DualIO (`DualTensorSourceOp` emitting both
inputs on one OnClock tick — required for `OnAligned`+`Exact` without
`PARTITION_MANUAL_AUTHORING_REQUIRED`; `FrameBankB` /
`TensorCaptureBOp` / `run_dual_io_operator_graph_result`) for
SnapshotGather / Beamscan+`is_valid`, and
`run_single_in_dual_out_operator_graph_result` for 1-in/2-out
(Beamscan without `is_valid`). `run_single_in_five_out_operator_graph_result`
covers MVDR / MUSIC five always-emitted Tensor outputs (FrameBank C/D/E).
`run_single_in_four_out_operator_graph_result` covers Root-MUSIC /
ESPRIT’s four always-emitted outs (no `angle_spectrum`; FrameBank B/C/D).
Dual-/triple-in single-out runners
(`run_dual_in_single_out_operator_graph_result` /
`run_triple_in_single_out_operator_graph_result` +
`TripleTensorSourceOp`) cover DetectionReductionOp. Error-path waits poll
`session.diagnostics()` (capture CV alone never wakes on reject). Further
generalize as more multi-port ops land; restore mid-stream live override
coverage once item 1 is fixed.

**Status:** open — runners for every landed port shape exist; remaining
work is the FMCW/MIMO shapes and the §1 live-override cases.

---

### 8. Graph perf: POD live-override timing restored

**Class:** our work, with enum/vector/descriptor EA limits in §§1/9.

All `perf_*.cpp` drivers are Graph-ported on HSDK 5 (setup / cold /
steady, process warmup, smoke via `ctest -L performance`) and listed in
`HOLOSCAN_RADAR_HSDK5_PORTED_PERF`. Shared source/sink/runner lives in
`performance/cpp/utils/perf_graph.hpp`. `run_async` has no HSDK 4
`CountCondition` equivalent, so the runner owns the stop condition; it waits on
*lack of progress* (`kPerfStall`, reset by every captured frame) rather than a
fixed wall-clock budget, since per-frame cost is hardware-dependent and spans
milliseconds to seconds (a fixed 60 s deadline previously aborted
`perf_root_music_estimator --full` at the `xl` rung, which needs ~150 s here).
Use `--stop-after-s` to cap total runtime. Inline mid-session POD `[live]`
changes use the supported single-clock, exact-contract, single-partition shape
from §1 and report `override_ms` / `override_delta_ms`. Enum
`ClutterFilterOp.method` uses production `Exact{}` (the co-emit source already
publishes every tick; do not restack `LatestWithin`). Remaining enum lives are
not yet wired. Vector/descriptor live *timing* is not yet
covered, though the ports exist (§9). Per-tick source publish is a fresh pool loan (§3); vs HSDK 4’s free handle
republish this is an
**accepted** end-to-end comparison artifact on large inputs (VAA
`virtual_dedup_lg`/`_xl` — see §3). Drivers pick `PerfPublishMode`
explicitly; default remains `kCopy`.

**Status:** partially closed — cold/steady landed for all drivers; POD
override timing restored for the applicable consumers. Enum
`ClutterFilterOp.method` `override_ms` landed; remaining enum lives and §9
lives remain open.

---

## Resolved / landed workarounds (keep until EA cleanup)

| Item | Notes |
| --- | --- |
| Params schema | Ported ops use `Operator<Params>` + `HOLOSCAN_PARAM_SCHEMA`; drop `ArgumentSetter` / `ensure_radar_arg_setters` for those |
| Stream | `context.cuda_stream()` only — no `receive_cuda_stream` / `set_cuda_stream` |
| Emit | `prepare_device_tensor` then `emit_prepared_tensor` for multi-out ticks (fill/commit all loans before any publish); single-out may still use `emit_device_tensor` |
| Live override helper | `add_live_override_to_contract` (`Exact{}` + `SafeDefault{}`, unconditional under every trigger shape) + `resolve_live_override`, which returns the matched sample for that tick and the Params seed otherwise (not experimental `runtime_mutable`). No operator-side latch; `LatestWithin` is not used on `[live]` ports |
| Optional receives | Only `kInputUndeclared` / `kInputEmpty` / `kInputNotSelected` / `kNotReady` mean "absent" (`is_absent_input_error`); every other receive error propagates instead of degrading to the Params seed |
| Empty tensors | `receive_tensor` rejects a null `data()` only when `nbytes() > 0` — a zero-element tensor legitimately carries no allocation |
| Static params | Root-MUSIC / ESPRIT reject a mid-session `angle_convention` change rather than silently reusing the seeded device relabeling |
| Harness error-path wait | Poll diagnostics at 25 ms inside the deadline (`sdk_harness.hpp`) — avoids false 15s “hangs” when the sink never captures |

---

## Cleanup gate (before `main`)

Prerequisite: **phase 5 (Close deferred)** is done — every **Open — SDK /
EA** item is closed or moved into durable docs, and intentional EA test
skips are gone or explicitly reclassified as permanent contract.

1. Search the repo for `[EA HSDK5 ONLY]` and remove every marked workaround,
   including the temporary section in [`AGENTS.md`](../../AGENTS.md).
2. Empty or delete this file once all open items are closed or moved to
   durable docs (`OPERATOR_API.md`, `OPERATOR_DEVELOPMENT.md`, per-op
   `OPEN_ITEMS.md`).
3. Confirm allow-list can be deleted (all operators ported or explicitly
   retired) and Python/apps are back on.
