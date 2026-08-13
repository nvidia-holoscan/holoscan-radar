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
| [HSDK 5 preview docs](https://nvidia-preview-v5-0-0-1.docs.buildwithfern.com/holoscan/sdk-user-guide/introduction/overview.md) | SDK API source of truth (prefer `.md`; index via `llms.txt`) — do not infer from HSDK 4.x |
| [`OPERATOR_API.md`](OPERATOR_API.md) | Locked operator contracts (ports, shapes, dtypes, params) |
| [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md) | Done gate once an op is on HSDK 5 (adapt 4.x checklist wording to Graph / `Operator<Params>`) |
| [`operators/common/README.md`](../../operators/common/README.md) | Pilot patterns: Tensor I/O, sticky `[live]`, Graph harness |
| [`STATUS.md`](STATUS.md) | Who owns what; claim / land coordination |
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
6. Close issues deferred on the EA SDK and/or team decisions (sticky live
   Graph, durable port/descriptor API deltas, restore or ratify interim
   Params-only / Tensor-pack workarounds).
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
| **1 — Shared glue** | `radar_common/sdk/` Tensor I/O + sticky override; `sdk_harness.hpp` Graph runners | ClutterFilter pilot compiles |
| **2 — Operator wave** | Rewrite ops to `Operator<Params>` + typed ports; Graph unit tests; allow-list each name | `ctest -L unit` green for ported ops (intentional skips OK) |
| **3 — Perf wave** | Rewrite `perf_*.cpp` to Graph; add names to `HOLOSCAN_RADAR_HSDK5_PORTED_PERF` | `ctest -L performance` smokes pass |
| **4 — Apps / Python** | Demo apps + bindings when Holoviz / Python EA surface exists | App smoke / pytest |
| **5 — Close deferred** | Re-probe EA gaps; settle durable HSDK 5 API deltas with the team; restore Graph sticky / live descriptor ports / typed geometry outs where ratified; unskip deferred suites | Every **Open — SDK / EA** item closed or moved to durable docs; no intentional EA `GTEST_SKIP` left unexplained |
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
| **EA limitation** (e.g. §1 mid-stream sticky Graph, §8 live-override perf) | Re-probe on newer EA / GA; restore `contract()` sticky policy + unskip Graph tests; document the working harness recipe — or reclassify as durable if the SDK will not support it |
| **Durable HSDK 5** (e.g. §2 unique in/out names, §9 vector/descriptor lives, §10 `ArrayGeometry` Tensor pack) | Team decision: keep the interim as the locked contract (update `OPERATOR_API.md`) **or** restore typed live/descriptor ports when the SDK admits them; drop dual “logical vs physical” naming notes once settled |
| **Our deferred work blocked on the above** (optional-mask Graph wiring, Params-only → live Inputs, Graph sticky unskips) | Land only after the matching EA/team item is closed; do not silently pick a contract answer in code |

Phase 5 is **not** for new algorithm contracts (those stay
[`TEAM_BLOCKED.md`](TEAM_BLOCKED.md) / per-op `OPEN_ITEMS.md`). It **is**
for every interim that exists only because of EA friction or an unsettled
HSDK 5 API shape.

---

## How to port an operator

Copy the ClutterFilter / SnapshotGather patterns; do not reintroduce
HSDK 4.x `InputContext` / `TensorMap` / `HOLOSCAN_OPERATOR_FORWARD_ARGS`.

1. **Read the contract** in `OPERATOR_API.md` and any op `OPEN_ITEMS.md`.
   Note unique in/out names (§2), POD vs Params-only live params (§9), and
   whether a descriptor must ride a Tensor pack (§10).
2. **Rewrite the op** to `Operator<Params>` + `HOLOSCAN_PARAM_SCHEMA`;
   typed `Input<Tensor>` / `Output<Tensor>` (or POD live Inputs); receive
   via `receive_tensor`, emit via `emit_device_tensor` /
   `bind_tensor_output_device` on the producer side; stream from
   `context.cuda_stream()`; sticky live via `resolve_sticky_override`.
3. **Allow-list** the CMake target name in
   `HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS`
   ([`cmake/HoloscanRadarOptionalFeatures.cmake`](../../cmake/HoloscanRadarOptionalFeatures.cmake))
   and register its `test_*` binary in `tests/cpp/CMakeLists.txt`.
4. **Graph tests** using `tests/cpp/utils/sdk_harness.hpp` (pick the
   runner that matches port arity). Skip mid-stream sticky live Graph
   cases under §1 until EA supports them; keep Params configurability
   coverage. Prefer stage / kernel tests when optional Inputs cannot be
   wired yet.
5. **Checklist** against [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md)
   (interpret “TensorMap / Application harness” as typed ports + Graph).
6. **Perf (optional same PR or follow-up):** Graph-rewrite `perf_<name>.cpp`,
   append `<name>` to `HOLOSCAN_RADAR_HSDK5_PORTED_PERF` in
   `performance/cpp/CMakeLists.txt`, run the smoke.
7. **Record** progress in the table below, durable deltas under **Open**
   or **Resolved**, and a `STATUS.md` line when claiming or landing.

**C++ operator first-pass:** complete (all 19 operators allow-listed). CFAR
GO/SO/OS/alpha_lut Graph unit suites + statistical MC (`ctest -L statistical`)
landed. Remaining C++ work is Graph perf (phase 3), apps/Python (phase 4),
and EA gaps in §§1/9/10 (phase 5) — see progress table.

---

## Build and verify

```bash
cmake -S . -B build \
  -DHOLOSCAN_RADAR_BUILD_TESTING=ON \
  -DHOLOSCAN_RADAR_BUILD_PERFORMANCE=ON
cmake --build build -j"$(nproc)"

# Ported C++ units (pytest is unregistered while Python is forced off)
ctest --test-dir build -L unit --output-on-failure

# Graph-ported perf smokes only (today: clutter)
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
| Shared Tensor I/O + sticky `[live]` helpers (`radar_common/sdk/`) | landed (ClutterFilter pilot) |
| `ClutterFilterOp` F32/F64 + Graph C++ tests | landed; Graph perf cold/steady landed; live-override unit+perf deferred (§1) |
| `SnapshotGatherOp` F32/F64 + Graph C++ tests (OnAligned 2-in/2-out) | landed; live-override unit deferred (§1); Graph perf deferred |
| `PhaseShiftBeamformerOp` F32/F64 + Graph C++ tests | landed; `[live]` geometry/grid/taper Params-only (§9); Graph live tests skipped (§1) |
| `BeamscanEstimatorOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; `peak_interpolation` live + optional `is_valid` Graph wiring deferred (§1); geometry/grid/taper Params-only (§9); Graph perf deferred |
| `SubarraySelectionOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; outputs `snapshot_out` / `is_valid_out` (§2); `array_geometry` emitted as packed float64 Tensor (§10); `array_geometry`/`subarray` live Params-only (§9); optional `is_valid` Graph wiring deferred (§1); Graph perf deferred |
| `ClusterDBSCANOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; `min_points`/`max_clusters` sticky Inputs (Graph live deferred §1); `epsilon` Params-only (§9 unbounded vector); Graph perf deferred |
| `MVDREstimatorOp` F32/F64 + Graph C++ tests (1-in/5-out) | landed; always emits optional `angle_spectrum`/`doa_multi`/`source_count`/`eigenvalues`; sticky `diagonal_loading`/`max_sources` Inputs (Graph live deferred §1); geometry/grid/`spatial_smoothing` Params-only (§9); optional `is_valid` Graph deferred (§1); `stop()` MatX retire preserved; Graph perf deferred |
| `MUSICEstimatorOp` F32/F64 + Graph C++ tests (1-in/5-out) | landed; always emits optional `angle_spectrum`/`doa_multi`/`source_count`/`eigenvalues`; sticky `num_sources`/`source_count_pfa`/`max_sources` Inputs (Graph live deferred §1); geometry/grid/`spatial_smoothing` Params-only (§9); optional `is_valid` Graph deferred (§1); `stop()` MatX retire preserved; Graph perf deferred |
| `RootMUSICEstimatorOp` F32/F64 + Graph C++ tests (1-in/4-out) | landed; always emits optional `doa_multi`/`source_count`/`eigenvalues` (no `angle_spectrum`); sticky `num_sources`/`source_count_pfa`/`max_sources` Inputs (Graph live deferred §1); geometry/`spatial_smoothing` Params-only (§9); optional `is_valid` Graph deferred (§1); `stop()` MatX/cuSOLVER teardown preserved; Graph perf deferred |
| `ESPRITEstimatorOp` F32/F64 + Graph C++ tests (1-in/4-out) | landed; always emits optional `doa_multi`/`source_count`/`eigenvalues` (no `angle_spectrum`); sticky `num_sources`/`source_count_pfa`/`max_sources` Inputs (Graph live deferred §1); geometry/`spatial_smoothing`/`subarray_spec` Params-only (§9); optional `is_valid` Graph deferred (§1); `stop()` MatX retire preserved; Graph perf deferred |
| `FeatureAssemblyOp` F32/F64 + Graph C++ tests (1-in/1-out; optional gathers) | landed; optional gather Inputs omitted from contract (§1 / Beamscan `is_valid` pattern); Graph optional wiring deferred — stage kernel coverage; no `[live]` params; Graph perf deferred |
| `DetectionReductionOp` F32/F64 + Graph C++ tests (2/3-in/1-out) | landed; `representative`/`score_method` sticky Inputs (Graph live deferred §1); `feature_weights` Params-only (§9 unbounded vector); Graph perf deferred |
| `CFARDetectorOp` F32/F64 + Graph C++ tests (1-in/5-out; CA + GO/SO/OS + alpha_lut) | landed; always emits all 5 outs; sticky POD lives (Graph live deferred §1); vector lives Params-only (§9); statistical MC suite on label `statistical`; Graph perf deferred |
| `MIMODeinterleaveOp` F32/F64 + Graph C++ tests | landed; output `adc_cube_out` (§2); Graph sticky/`TXSchedule` live N/A (Params-only schedule); Graph perf deferred |
| `MIMOCoherentCompensationOp` F32/F64 + Graph C++ tests | landed; output `rd_maps_out` (§2); `[live]` `waveform` Params-only (§9); Graph live tests skipped; Graph perf deferred |
| `VirtualArrayAssemblyOp` F32/F64 + Graph C++ tests (1-in/2-out) | landed; output `rd_maps_out` (§2); `array_geometry` packed float64 Tensor (§10); sticky `wavelength` Input (Graph mid-stream sticky deferred §1); Graph perf deferred |
| `FMCWRangeProcessorOp` F32/F64 + Graph C++ tests (1-in/1–2-out) | landed; optional `range_bins` only with waveform; sticky POD lives; `range_window`/`waveform` Params-only (§9); Graph live window skipped (§1/§9); Graph perf deferred |
| `FMCWDopplerProcessorOp` F32/F64 + Graph C++ tests (1-in/1–2-out) | landed; optional `velocity_bins` only with waveform; sticky POD lives; `doppler_window`/`waveform` Params-only (§9); Graph live skipped (§1/§9); Graph perf deferred |
| `FMCWRangeDopplerProcessorOp` F32/F64 + Graph C++ tests (1-in/1–3-out) | landed; optional axis outs with waveform; sticky POD lives incl. `clutter_filter`; windows/`waveform` Params-only (§9); UlpParity vs split chain preserved; Graph live skipped; Graph perf deferred |
| Python bindings / Holoviz demos / apps / Graph perf (most ops) | deferred |

---

## Open — SDK / EA

### 1. Graph mid-stream sticky `[live]` override harness

**Class:** EA limitation (not a permanent ban on sticky live overrides).

**Contract.** `OPERATOR_API.md` still requires sticky `[live]` parameters
(e.g. `ClutterFilterOp.method`, `SnapshotGatherOp.snapshot_window`). The
operator keeps the typed `Input<>` and resolves via
`resolve_sticky_override` in `compute()`.

**What fails on EA.** Wiring a second OnClock / multi-partition enum
source beside a device-Tensor path fails at runtime with
`RUNTIME_PORT_BINDING_UNAVAILABLE` and/or
`RUNTIME_EXECUTION_LANE_PLAN_INVALID`. Putting an enum `Output` on the
same op as a device `Output<Tensor>` also failed binding. Separate
partitions compiled in some trials but did not run cleanly.

**Interim.**

- `ClutterFilterOp::contract()` triggers only on `range_profiles`
  (`OnEach`); `method` is registered in `setup()` but omitted from the
  temporal contract so unwired graphs compile
  (`TEMPORAL_INPUT_UNCONNECTED` otherwise).
- `SnapshotGatherOp::contract()` uses `OnAligned` on the two data
  tensors only; `snapshot_window` is likewise omitted from the contract.
- `ClutterFilterOpTest.LiveMethodOverrideMidStreamAndSticky` and
  `SnapshotGatherOpTest.LiveSnapshotWindowStickyOverride` are
  `GTEST_SKIP`’d. `BeamscanEstimatorOpTest.LivePeakInterpolationStickyOverride`
  is likewise skipped (enum live Input kept). `PhaseShiftBeamformerOp` /
  Beamscan geometry/grid/taper live Graph tests are also skipped; those
  ops have no live Inputs for non-POD descriptors yet (§9 /
  Params-only). `ClusterDBSCANOpTest.LiveMinPointsStickyOverride` /
  `LiveMaxClustersStickyOverride` are skipped the same way (scalar live
  Inputs kept); `LiveEpsilonStickyOverride` is skipped under §9.
- Selection is still covered via Params
  (`MethodConfigurableFromParams` /
  `SnapshotWindowConfigurableFromParams` /
  `ScanGridConfigurableFromParams` /
  `PeakInterpolationConfigurableFromParams` /
  `EpsilonConfigurableFromParams` /
  `MinPointsConfigurableFromParams` /
  `MaxClustersConfigurableFromParams` /
  `DiagonalLoadingConfigurableFromParams` /
  `MaxSourcesConfigurableFromParams`).
- `MVDREstimatorOpTest.LiveDiagonalLoadingStickyOverride` /
  `LiveMaxSourcesStickyOverrideReallocates` are likewise skipped (POD live
  Inputs kept); geometry/grid/`spatial_smoothing` live Graph tests skipped
  under §9.
- `MUSICEstimatorOpTest.LiveNumSourcesStickyOverride` /
  `LiveSourceCountPfaStickyOverride` /
  `LiveMaxSourcesStickyOverrideReallocates` are likewise skipped (POD live
  Inputs kept); geometry/grid/`spatial_smoothing` live Graph tests skipped
  under §9. Params cover selection
  (`NumSourcesConfigurableFromParams` /
  `SourceCountPfaConfigurableFromParams` /
  `MaxSourcesConfigurableFromParams`).
- `RootMUSICEstimatorOpTest` / `ESPRITEstimatorOpTest`
  `LiveNumSourcesStickyOverride` / `LiveSourceCountPfaStickyOverride` /
  `LiveMaxSourcesStickyOverride` are likewise skipped (POD live Inputs
  kept); geometry/`spatial_smoothing` live Graph tests skipped under §9.
  Params cover the same trio of configurable-from-Params tests.
- `FeatureAssemblyOp` has no `[live]` params; optional gather Tensor Inputs
  are omitted from `contract()` (unconnected apps) and Graph wiring of those
  ports is deferred (`OptionalGatherGraphWiringDeferred` skipped) — same EA
  gap as Beamscan `is_valid`; stage kernel tests cover optional gather paths.
- `DetectionReductionOpTest.LiveRepresentativeStickyOverride` /
  `LiveScoreMethodStickyOverride` are skipped the same way (enum live Inputs
  kept); `LiveFeatureWeightsStickyOverride` is skipped under §9. Params
  cover selection (`RepresentativeConfigurableFromParams` /
  `ScoreMethodConfigurableFromParams` /
  `FeatureWeightsConfigurableFromParams`).
- `CFARDetectorOpTest.LiveThresholdScaleOverrideMidStreamAndSticky` /
  `LivePfaOverrideMidStreamAndSticky` /
  `LiveMaxDetectionsOverrideMidStreamAndSticky` are likewise skipped (POD
  live Inputs kept); vector lives (`guard_cells` / `training_cells` /
  `nci_axes` / `alpha_lut`) skipped under §9. Params cover selection
  (`ThresholdScaleConfigurableFromParams` / `PfaConfigurableFromParams` /
  `MaxDetectionsConfigurableFromParams` /
  `GuardCellsConfigurableFromParams` /
  `TrainingCellsConfigurableFromParams`).

**Next.** Re-probe on a newer EA: restore sticky policy in `contract()`
(`LatestWithin` + `SafeDefault` / reuse semantics from
`override_param.hpp`), unskip the Graph sticky test, and document the
working partition/binding recipe in `sdk_harness.hpp`.

**Status:** open — EA harness gap; sticky contract still required.

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

**Interim.** Graph harness helper `bind_device_pools` in
`tests/cpp/utils/sdk_harness.hpp` binds `source`/`out` and
`op`/`range_profiles_out`. Dual-IO graphs use
`bind_dual_io_device_pools` for `source0`/`source1` plus both op
outputs. Single-in dual-out graphs use
`bind_single_in_dual_out_device_pools`; optional extra unbound device
outputs (e.g. `SubarraySelectionOp.is_valid_out`) must still be bound
via the harness `extra_op_out_ports` argument or compile fails with
`IMPLICIT_DEVICE_ZERO_REJECTED`. Apps/tests that emit device tensors
must do the same for every such producer path.

**Status:** open — document per new harness/app as they port; confirm
whether public GA softens discovery or keeps explicit bind.

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

**Interim.** `SnapshotGatherOp` keeps `Params.snapshot_window` as
`std::vector<int64_t>` (OPERATOR_API.md / YAML). The `[live]` port is
`Input<SnapshotWindowOverride>` — a fixed `{doppler, range}` int64 POD
with `ChannelTypeIdentity` / `ChannelPodLayout` ([EA HSDK5 ONLY] in
`snapshot_gather_op.hpp`). Normalize empty/size-1 vectors on the Params
path before seeding the POD sticky state. `PhaseShiftBeamformerOp`,
`BeamscanEstimatorOp`, and `SubarraySelectionOp` cannot use that
fixed-POD recipe for descriptor / unbounded vectors: `az_angles` /
`el_angles` / `subarray` are unbounded, and `ArrayGeometry` /
`WindowSpec` own vectors — those `[live]` params stay Params-only (see
each op’s `OPEN_ITEMS.md`). Beamscan still declares sticky
`Input<PeakInterpolation>` for the enum live (harness deferred §1).
`ClusterDBSCANOp.epsilon` is likewise Params-only: length `1` or
`num_features` with unbounded `num_features` (caller-assembled
`points`); scalar `min_points` / `max_clusters` keep sticky
`Input<int64_t>` (harness deferred §1). `MVDREstimatorOp` mirrors this:
sticky `Input<double> diagonal_loading` / `Input<int64_t> max_sources`;
`az_angles` / `el_angles` / `spatial_smoothing` / `ArrayGeometry` stay
Params-only. `MUSICEstimatorOp` likewise: sticky `Input<int64_t>
num_sources` / `Input<double> source_count_pfa` / `Input<int64_t>
max_sources`; geometry/grid/`spatial_smoothing` Params-only.
`DetectionReductionOp.feature_weights` is Params-only for the
same unbounded-vector reason (length `num_features`); enum
`representative` / `score_method` keep sticky Inputs (harness deferred §1).
`RootMUSICEstimatorOp` / `ESPRITEstimatorOp` mirror MUSIC’s POD sticky
trio (`num_sources` / `source_count_pfa` / `max_sources`); geometry /
`spatial_smoothing` (and ESPRIT’s STATIC `subarray_spec`) stay
Params-only.
`CFARDetectorOp` mirrors this: sticky `Input<double> pfa` /
`threshold_scale` and `Input<int64_t> os_rank` / `max_detections`;
`guard_cells` / `training_cells` / `nci_axes` / `alpha_lut` stay
Params-only (pair / unbounded vectors).

**Next.** Prefer POD / schema types for other vector-shaped live
overrides as those ops port; for unbounded grids / feature radii prefer
`Input<Tensor>` (or a durable schema) and document any OPERATOR_API
physical-type divergence.

**Status:** open — SnapshotGather POD pattern + PhaseShift/Beamscan/
SubarraySelection/ClusterDBSCAN/MVDR/MUSIC/RootMUSIC/ESPRIT/
DetectionReduction/CFAR Params-only fallback for non-POD / unbounded
vectors.

---

### 10. Descriptor output ports (`ArrayGeometry`) are not PublishedPayloads

**Class:** Durable HSDK 5 (`PublishedPayload` admission).

**What fails.** `ArrayGeometry` owns `std::vector<double>` and
`std::optional<ArrayGeometryType>` — not a ChannelPod (requires standard
layout + trivially copyable) and not a SchemaIdentity type. Declaring
`Output<ArrayGeometry>` (or `Input<ArrayGeometry>`) fails the
`PublishedPayload` concept at compile time. The same blocker applies to
`VirtualArrayAssemblyOp.array_geometry` when that op ports.

**Contract.** Operators that *compute* geometry at runtime
(`SubarraySelectionOp`, `VirtualArrayAssemblyOp`) must still emit it on a
data-plane port each tick — Params / side-channel return is not acceptable
(downstream estimators need the reduced / virtual geometry).

**Interim.** `SubarraySelectionOp` emits `array_geometry` as a float64
device Tensor of shape `(num_elements + 1, 3)` via
`radar_common/descriptors/array_geometry_tensor.hpp` (positions in rows
`[0, N)`, metadata row `[wavelength, geometry_type_code, 0]`). Documented
in `OPERATOR_API.md` under SubarraySelectionOp. Live *input* overrides of
`ArrayGeometry` remain Params-only (§9).

**Next.** Prefer a durable schema / ChannelPod-compatible wrapper (or SDK
admission for owning containers) shared with VirtualArrayAssembly; restore
typed `Output<ArrayGeometry>` when available. Do not invent a second
descriptor type.

**Status:** open — SubarraySelection + VirtualArrayAssembly Tensor pack
landed; restore typed `Output<ArrayGeometry>` when the SDK admits owning
containers / a durable ChannelPod schema.

---

## Open — deferred port work

### 5. Allow-list: C++ operator first-pass complete

**Class:** Our deferred work (allow-list retained until EA cleanup).

All 19 operators are on `HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS`. The
allow-list remains until public HSDK 5 / cleanup gate so apps, packaging,
and unported Graph perf stay gated intentionally. Demo apps still need
Holoviz (§6) and Graph-port of remaining app scaffolding. CFAR GO/SO/OS /
alpha_lut Graph unit suites and the `statistical`-labeled MC suite have
landed (phase-2 unit coverage). Perf drivers remain on the separate
`HOLOSCAN_RADAR_HSDK5_PORTED_PERF` list (today: clutter only).

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
unported operator modules optionally. Demo apps are gated on Holoviz **and**
ported dependencies (§5). Shared `performance/cpp/utils` still has HSDK 4.x
Application helpers for unported ops. CMake gates drivers via
`HOLOSCAN_RADAR_HSDK5_PORTED_PERF` in `performance/cpp/CMakeLists.txt`
(currently only `clutter_filter` — see §8); operator allow-list alone is
not enough because most `perf_*.cpp` files remain 4.x Application /
`TensorMap` code.

**Status:** open — after a critical mass of C++ operators.

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
generalize as more multi-port ops land; restore mid-stream sticky override
coverage once item 1 is fixed.

**Status:** open — runners for every landed port shape exist; remaining
work is the FMCW/MIMO shapes and the §1 live-override cases.

---

### 8. ClutterFilterOp perf: live-override timing still deferred

**Class:** EA limitation (depends on §1) + partial our work.

`perf_clutter_filter` is Graph-ported on HSDK 5 (setup / cold / steady,
process warmup, smoke via `ctest -L performance`). Inline mid-session
`[live] method` `override_ms` is **not** measured until §1 is fixed —
checklist Performance → live-override coverage remains open (see
`operators/fmcw/clutter_filter_op/OPEN_ITEMS.md` §1). Other `perf_*`
binaries stay off the build until rewritten and added to
`HOLOSCAN_RADAR_HSDK5_PORTED_PERF` (avoids linking 4.x drivers against
ported ops when `HOLOSCAN_RADAR_BUILD_PERFORMANCE=ON`).

**Status:** open — cold/steady landed; override timing blocked on §1.

---

## Resolved / landed workarounds (keep until EA cleanup)

| Item | Notes |
| --- | --- |
| Params schema | Ported ops use `Operator<Params>` + `HOLOSCAN_PARAM_SCHEMA`; drop `ArgumentSetter` / `ensure_radar_arg_setters` for those |
| Stream | `context.cuda_stream()` only — no `receive_cuda_stream` / `set_cuda_stream` |
| Emit | `prepare_device_tensor` then `emit_prepared_tensor` for multi-out ticks (fill/commit all loans before any publish); single-out may still use `emit_device_tensor` |
| Sticky helper | Operator-owned `std::optional` + `resolve_sticky_override` (not experimental `runtime_mutable`) |
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
