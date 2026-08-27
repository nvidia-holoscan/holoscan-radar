# MVDREstimatorOp — open items

Status: **HSDK 5 EA C++ port landed** (checklist not fully closed — scalar
live / `is_valid` Graph coverage deferred; one test is red on a real
correctness bug, item 6). Algorithm contract
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

**Wiring (settled — L16 / L17).** `contract()` unconditionally joins
`Input<double> diagonal_loading` and `Input<int64_t> max_sources` to the
data trigger with `Exact{}` and `on_missing SafeDefault{}`; `compute()`
resolves them with `resolve_live_override`. The application enables an
override by wiring its port and disables it by leaving it unwired.
`sticky_*` (default true) latches the last
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

## 2. `array_geometry` overrides have no in-module producer

**Blocks checklist:** nothing on its own — the descriptor / grid /
smoothing ports and their Graph coverage are in place (see item 6 for the
one red test); this is the transport decision behind the geometry port.

**Class:** Open `ArrayGeometry` transport decision. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9 / §10.

**Ports (settled — L18).** `contract()` unconditionally joins
`Input<WireArrayGeometry> array_geometry`, the rank-1 pageable-host float64
config Tensors `az_angles` / `el_angles` (bounded by `max_az` / `max_el`),
and the bounded `{count, first, second}` int64 POD
`Input<SpatialSmoothingOverride> spatial_smoothing`; `compute()` resolves
each with `resolve_live_override_via`. Logical `MVDRParams` types are
unchanged. `angle_convention` and `forward_backward` stay Params-only
(STATIC / not `[live]`).

**Still open.** `VirtualArrayAssemblyOp` / `SubarraySelectionOp` emit
geometry as a packed float64 `(N + 1, 3)` Tensor (§10), not the
wire-schema type this port consumes (§9). The representations differ, so a
runtime virtual geometry cannot drive this override directly — it has to
come from an app-side wire-schema producer. The generated wire objects are
also C++-only, so Python drives the parameter through Params.

**Next.** Settle one `ArrayGeometry` transport so a descriptor output can
feed a descriptor override port.

**Status:** open — `ArrayGeometry` transport decision.

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

## 6. Second tick diverges from the first on identical input

**Blocks checklist:** Tests → most of the suite is red. See *Measured scope*.

**Measured scope (2026-08-27, `holoscan-radar:optports-baseline`).** This is
much broader than the single sticky-override test this item originally
described. Six consecutive runs of the whole `test_mvdr_estimator` binary:

| Test | Runs failed |
| --- | --- |
| `SpatialSmoothingReducesEigenvalueChannels` | 6/6 |
| `RecoversKnownDoaUlaAzOnly` | 6/6 |
| `MultiSourceRecoversTwoDoa` | 6/6 |
| `FixedDiagonalLoadingRecoversDoa` | 6/6 |
| `FloatDoubleAgreeOnDoa` | 5/6 |
| `MaxSourcesCapacityNanFillsUnusedRows` | 2/6 |
| `LiveArrayGeometryStickyOverride` | 2/6 |

**It is process-state contamination, not a wrong algorithm.** Every one of
those tests passes when run as the only test in its own process
(`--gtest_filter` on each of the four: 0 failures). They fail only when the
whole binary runs them in one process. That is why `ctest` is green — it
discovers each GTest case as a separate test and so isolates them: the full
`ctest -LE performance` run is 764/765. Anyone running
`./test_mvdr_estimator` directly will see five failures and should not read
them as broken DOA math.

So the defect is that MVDR leaves per-process state behind that corrupts
later runs in the same process — consistent with the MatX solver-cache /
cold-path evidence below, and a superset of the original tick-to-tick
symptom. `LiveArrayGeometryStickyOverride`, the test this item was filed
against, is now among the *least* frequent failures.

Pre-existing and independent of the live-override test work: the committed
baseline of the test file fails the same five per whole-binary run, and the
newly added `LiveMaxSourcesStickyOverrideReallocates` passes 8/8 in
isolation.

**Symptom.** In a fresh process, feeding the same snapshot twice makes
`angle_spectrum` differ between tick 1 and tick 2. Tick 1 is the correct
answer (peak at the simulated target, `argmax = 20` for
`u = 0.10` on a `linspace(-0.3, 0.3, 31)` grid, peak `0.98`); tick 2 peaks
one bin over at `7.74`. It is deterministic run to run.

**Not the live-override path.** A throwaway two-tick test with every
`expect_*` false and no override port wired reproduces it, and a probe
showed both ticks resolving the same `ArrayGeometry` and the same steering
cache key. `LiveArrayGeometryStickyOverride` is the only test that
compares spectra across ticks, which is why it is the one that fails; the
sticky resolution itself is correct. The test was `GTEST_SKIP`’d before
the live overrides were restored, so this is newly exposed, not new.

**Blast radius is wider than one tick.** It also breaks run-to-run
comparison, which blocks the `diagonal_loading` live-override test: δ moves
spectrum values only, so the natural assertion is that an override δ produces
the same spectrum as the identical Params δ. Written out, that comparison
fails 8/8 when it is the first graph run in a fresh process and passes once the
process is warm — the first run is the wrong one, matching the cold-path
finding below. The test is therefore left skipped against this item rather than
against any SDK limitation. `LiveMaxSourcesStickyOverrideReallocates` is
unaffected and passes 8/8, because `max_sources` is observable in the
`doa_multi` shape rather than in float values.

**What the evidence points at** — the `matx::eig` / `matx::inv` region of
`run_mvdr_stage`, with a dependence on host timing:

- Same CUDA stream both ticks, and identical device addresses for the
  manifold, covariance, inverse, and all five output loans. No aliasing.
- `cudaStreamSynchronize` plus D2H reads *inside* the stage make both
  ticks agree on tick 1's value. A sync only at the tail of the stage
  changes nothing; a sync only after `matx::inv` yields a third, also
  wrong, tick-2 answer.
- Under `--gtest_repeat=5`, iteration 1 fails and iterations 2–5 pass in
  the same process — the cold path is the broken one.
- Suppressing the per-tick `clear_matx_solver_plan_caches()` changes
  nothing.
- Incidental, possibly related: the `eigenvalues` loan comes back in a
  different address space from the other four outputs, and
  `cudaMemsetAsync` on it fails, though all five declare
  `produces_tensor(kDevice)`.

**Next.** Reduce to a standalone MatX `eig` + `inv` repro (batched
Hermitian, `C = 4`, two back-to-back solves on one stream) so it can be
filed upstream, then re-check against a newer MatX / HSDK 5 build.

**Status:** open — correctness bug, test left failing deliberately so it
is not forgotten.

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
