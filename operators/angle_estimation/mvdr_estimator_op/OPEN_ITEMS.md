# MVDREstimatorOp — open items

Status: **HSDK 5 EA C++ port landed** (checklist not fully closed — live /
`is_valid` Graph coverage deferred). Algorithm contract
(`OPERATOR_API.md` → MVDREstimatorOp) is unchanged:

- sample covariance with `is_valid` masking, diagonal loading (`auto` / fixed),
  batched Hermitian inverse, Capon spectrum, dominant DOA;
- `max_sources` **[live]** + `doa_multi` / `source_count` and `eigenvalues`;
- `spatial_smoothing` (+ `forward_backward`) front-end on filled ULA/URA.

Cross-cut SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream sticky `[live]` POD coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky” for
`diagonal_loading` / `max_sources`.

**Class:** HSDK 5 EA limitation. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Interim.**

- Operator still declares sticky `Input<double> diagonal_loading` and
  `Input<int64_t> max_sources`; `compute()` resolves via
  `resolve_sticky_override`.
- Those ports are omitted from `contract()` so unwired graphs compile.
- `LiveDiagonalLoadingStickyOverride` /
  `LiveMaxSourcesStickyOverrideReallocates` are `GTEST_SKIP`’d.
- Params authoring covers selection
  (`DiagonalLoadingConfigurableFromParams` /
  `MaxSourcesConfigurableFromParams` / fixed-loading DOA test).

**Next.** Restore sticky temporal policy + Graph sticky unit tests on a
newer EA / GA (shared recipe with ClutterFilter §1).

**Status:** open — EA.

---

## 2. Non-POD / variable-length `[live]` params are Params-only

**Blocks checklist:** Tests → live `az_angles` / `el_angles` /
`array_geometry` / `spatial_smoothing` mid-stream sticky overrides.

**Class:** Durable HSDK 5 (`PublishedPayload` admission) + our deferred
work. Same recipe as Beamscan / PhaseShift — see
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**Interim.** Those parameters remain on `MVDRParams` only (`set_params`).
Live Graph override tests are `GTEST_SKIP`’d. `angle_convention` and
`forward_backward` stay Params-only (STATIC / not `[live]`).

**Next.** Prefer Tensor live ports for angle grids / smoothing counts
and/or ChannelPod / schema wrappers for `ArrayGeometry` once a module-wide
recipe lands (§9).

**Status:** open — ported compute path; live descriptor/vector ports
deferred.

---

## 3. Optional `is_valid` Graph wiring

**Blocks checklist:** Tests → optional `is_valid` connected Graph paths.

**Class:** HSDK 5 EA limitation (related to §1). Same as Beamscan OPEN_ITEMS
§3.

**Interim.** Port registered in `setup()`; omitted from `contract()`;
`compute()` uses `receive_optional_value`. Graph tests
`IsValidMaskExcludesWrongDoaSnapshots` /
`AllInvalidMaskBlanksDoaAndSpectrum` are `GTEST_SKIP`’d.

**Status:** open — EA.

---

## 4. Optional outputs always emitted

**Contract.** `OPERATOR_API.md` marks `angle_spectrum`, `doa_multi`,
`source_count`, and `eigenvalues` optional. On HSDK 5 EA the operator
**always allocates and emits** all five Tensor outputs each tick (same
compute cost as before — `eigenvalues` still runs `matx::eig` every tick).
C++ Graph tests wire and `bind_tensor_output_device` all five.

**Apps.** Consumers that previously left optional ports unwired must still
satisfy device-output binding (`IMPLICIT_DEVICE_ZERO_REJECTED` if a device
producer is unbound — HSDK5_PORT §3). Prefer wiring a no-op sink or
explicitly binding unused ports in app compile options. A future SDK
“is output connected” signal could gate `eig` / peak extract (see
Performance follow-ups below).

**Status:** open — document for apps; not a blocker for the C++ port.

---

## 5. MatX cache teardown via `stop()`

**Status:** preserved. HSDK 5 `OperatorBase::stop()` still runs at the
`kStop` lifecycle checkpoint. `MVDREstimatorOp::stop()` calls
`detail::retire_matx_solver_caches()` (full `GetAllocMap()` drain once while
CUDA is valid). Per-tick `clear_matx_solver_plan_caches()` at the end of
`run_mvdr_stage` is unchanged. Public `retire_matx_solver_caches()` remains
for Python / apps.

---

## Representation notes (unchanged)

- **`diagonal_loading` encoding.** `double`: negative → auto
  (`kAutoLoadingFactor · trace(R̂)/C`), non-negative → absolute δ. Default
  `-1` (auto).
- **Auto-loading factor.** `kAutoLoadingFactor = 1e-2` first cut; tune with
  the analytic simulator.

## Performance follow-ups (unchanged)

- Capon reduce is latency-bound (~97% GPU time); warp-cooperative / GEMM
  paths still open.
- `eigenvalues` `matx::eig` every tick even when unused (~2–4% on lg/xl).

## Python bindings / perf harness

**Class:** Our deferred work (HSDK5_PORT §6 / §8). Existing pybind trampoline
and `perf_mvdr_estimator` remain HSDK 4.x and are not built while
`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`. Graph-port perf in a follow-up.

**Status:** open — deferred; not required to land the C++ HSDK 5 port.

## Spatial smoothing (settled — S1)

Unchanged from the HSDK 4.x landing: front-end transform via
`radar_common/dsp/spatial_smoothing`; non-uniform / holey arrays rejected
(upstream S3). See STATUS.md (S1 / L13).
