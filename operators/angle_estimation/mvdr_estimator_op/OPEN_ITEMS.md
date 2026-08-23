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

## 1. Graph mid-stream `[live]` POD coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky” for
`diagonal_loading` / `max_sources`.

**Wiring (settled — L16 / L17).** `contract()` joins
`Input<double> diagonal_loading` and `Input<int64_t> max_sources` to the
data trigger with `Exact{}` and `on_missing SafeDefault{}` when the
matching `expect_*` is set; `compute()` resolves them with
`resolve_expected_override`. `sticky_*` (default true) latches the last
Exact match; false reverts to Params on a miss. The application stamps one
transmit time on the override and the snapshot — see
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Still open (EA test harness).**

- `LiveDiagonalLoadingStickyOverride` /
  `LiveMaxSourcesStickyOverrideReallocates` are `GTEST_SKIP`’d.
- `perf_mvdr_estimator` measures an inline `diagonal_loading` transition with
  the supported single-clock / exact-contract / single-partition shape.
- Params authoring covers selection
  (`DiagonalLoadingConfigurableFromParams` /
  `MaxSourcesConfigurableFromParams` / fixed-loading DOA test).

**Next.** Add Graph mid-stream + sticky tests, modeled on
`FMCWRangeProcessorOp`’s `nfft` Exact cases.

**Status:** open — test coverage only; the wiring decision is settled.
Isolated `max_sources` perf is still open; `diagonal_loading` perf
timing is restored (§8).

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

**Class:** HSDK 5 EA gap — no optional input
([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Same as
Beamscan OPEN_ITEMS §3.

**Interim.** Port registered in `setup()`; omitted from `contract()` so
unwired graphs compile (wiring it anyway is
`RUNTIME_OPERATOR_REPLAY_MISMATCH`, §1); `compute()` uses
`receive_optional_value`. Graph tests
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

**Class:** Our deferred work (HSDK5_PORT §6). Existing pybind trampoline is
still HSDK 4.x and is not built while `HOLOSCAN_RADAR_BUILD_PYTHON=OFF`.
`perf_mvdr_estimator` is Graph-ported and reports `diagonal_loading`
`override_ms` (§8).

**Status:** open — Python deferred; not required to land the C++ HSDK 5 port.

## Spatial smoothing (settled — S1)

Unchanged from the HSDK 4.x landing: front-end transform via
`radar_common/dsp/spatial_smoothing`; non-uniform / holey arrays rejected
(upstream S3). See STATUS.md (S1 / L13).
