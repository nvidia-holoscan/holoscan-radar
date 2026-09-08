# MUSICEstimatorOp — open items

Status: **HSDK 5 EA C++ port landed** (checklist not fully closed — scalar
live / `is_valid` Graph coverage deferred). Algorithm contract
(`OPERATOR_API.md` → MUSICEstimatorOp) is unchanged:

- sample covariance with `is_valid` masking, batched Hermitian
  eigendecomposition (`matx::eig`, `EigenMode::VECTOR`), per-item model-order
  resolution (fixed or EFT `auto`), noise-subspace projector, MUSIC
  pseudospectrum, and dominant DOA;
- `num_sources` **[live]** (int or `auto`) + `source_count_pfa` **[live]**;
- `max_sources` **[live]** + `doa_multi` / `source_count` and `eigenvalues`;
- `spatial_smoothing` (+ `forward_backward`) front-end on filled ULA/URA
  (shared with MVDR).

Cross-cut SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream `[live]` POD coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + revert-to-Params” for
`num_sources` / `source_count_pfa` / `max_sources`.

**Wiring (settled — L16 / L17).** `contract()` unconditionally joins
`Input<int64_t> num_sources`, `Input<double> source_count_pfa`, and
`Input<int64_t> max_sources` to the data trigger with `Exact{}` and
`on_missing SafeDefault{}`; `compute()` resolves them with
`resolve_live_override`. The application enables an override by wiring its
port and disables it by leaving it unwired. A matched override applies to
that tick only; a tick with no match uses the Params value. The application stamps one transmit time on the override and the
snapshot — see
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

**Still open (EA test harness).**

- `LiveNumSourcesAppliesForOneTick` / `LiveSourceCountPfaAppliesForOneTick` /
  `LiveMaxSourcesAppliesForOneTickAndReallocates` are `GTEST_SKIP`’d.
- `perf_music_estimator` measures an inline `max_sources` transition with
  the supported single-clock / exact-contract / single-partition shape.
- Params authoring covers selection
  (`NumSourcesConfigurableFromParams` /
  `SourceCountPfaConfigurableFromParams` /
  `MaxSourcesConfigurableFromParams`).

**Next.** Add Graph mid-stream + revert-to-Params tests, modeled on
`FMCWRangeProcessorOp`’s `nfft` Exact cases.

**Status:** open — test coverage only; the wiring decision is settled.
Isolated `num_sources` / `source_count_pfa` perf is still open;
`max_sources` perf timing is restored (§8).

---

## 3. Optional `is_valid` Graph wiring

**Blocks checklist:** Tests → optional `is_valid` connected Graph paths.

**Class:** stale HSDK 5 EA optional-input gap; §11 is closed. This operator has
not yet migrated its mask contract and Graph tests.

**Interim.** Port registered in `setup()`; omitted from `contract()` so
unwired graphs compile (wiring it anyway is
`RUNTIME_OPERATOR_REPLAY_MISMATCH`, §1); `compute()` uses
`receive_optional_value`. Graph tests
`IsValidMaskExcludesWrongDoaSnapshots` /
`AllInvalidMaskBlanksDoaAndSpectrum` are `GTEST_SKIP`’d.

**Status:** open — EA.

---

## 4. Optional outputs (resolved 2026-09-01)

**Contract.** `OPERATOR_API.md` marks `angle_spectrum`, `doa_multi`,
`source_count`, and `eigenvalues` optional, and its **Optional output
convention** now gives each a disposition. `angle_spectrum` and `eigenvalues`
are `always`: the pseudo-spectrum is the same pass that yields
`aoa_estimates` and also feeds peak extraction, and the eigendecomposition is
required for the subspace projector — neither has work to skip, so the port is
a re-export. `doa_multi` and `source_count` are `latched` and share one
peak-extraction pass, skipped once both have latched.

Previously all five were emitted under `HOLOSCAN_TRY`, so declining any of them
killed the run on its first tick.

**Apps.** Unwired device outputs must still be bound
(`IMPLICIT_DEVICE_ZERO_REJECTED` if a device producer is unbound — HSDK5_PORT
§3); binding is not wiring, and an unwired-but-bound port is exactly the case
the convention handles.

**Status:** resolved.

---

## 5. MatX cache teardown via `stop()`

**Status:** preserved. HSDK 5 `OperatorBase::stop()` still runs at the
`kStop` lifecycle checkpoint. `MUSICEstimatorOp::stop()` calls
`detail::retire_music_matx_caches()` (full `GetAllocMap()` drain once while
CUDA is valid). Per-tick `clear_music_matx_caches()` at the end of
`run_music_stage` is unchanged. Public `retire_music_matx_caches()` remains
for Python / apps. MUSIC only ever calls `eig` (VECTOR mode), never `inv`.

---

## Representation notes (unchanged)

- **`num_sources` encoding.** `int64_t`: negative → auto (per-item EFT),
  non-negative → fixed order. Default `-1` (auto).
- **`source_count_pfa`.** Target false-alarm for EFT auto; must be in
  `(0, 1)`. Default `1e-2`. See fidelity note below (EFT thresholds).
- **`doa_multi` targets `num_sources`, not always `max_sources`** — a real
  semantic difference from MVDR (Capon has no model-order concept).

## EFT source-count fidelity (unchanged)

The shared `radar_common/dsp/source_count` helper uses the EFT paper's
closed-form noise-eigenvalue-profile model for prediction but a first-order
Gaussian threshold instead of Monte-Carlo-calibrated η_P. `source_count_pfa`
is therefore a **target**, not a Monte-Carlo-guaranteed false-alarm rate.
Rank-deficient (`K < channels`) falls back to `num_sources = 1` per item on
device; the host "and warns" half is only exact for the uniform (no ragged
`is_valid`) case.

## Performance follow-ups (unchanged)

- `music_reduce_kernel` is L2-bound on per-beam manifold reads after wave-
  gated shared-`Pn` staging; further cross-item manifold reuse deferred.
- `num_sources` is fixed at 1 in the perf harness for determinism; a model-
  order sweep remains open.
- Denominator floor `kMusicDenomFloor<T> = 1e-6` is a first cut.

## Python bindings / perf harness

**Class:** Our deferred work (HSDK5_PORT §6). Existing pybind trampoline is
still HSDK 4.x and is not built while `HOLOSCAN_RADAR_BUILD_PYTHON=OFF`.
`perf_music_estimator` is Graph-ported and reports `max_sources`
`override_ms` (§8).

**Status:** open — Python deferred; not required to land the C++ HSDK 5 port.

## Spatial smoothing (settled — S1)

Unchanged from the HSDK 4.x landing: front-end transform via
`radar_common/dsp/spatial_smoothing`; non-uniform / holey arrays rejected
(upstream S3). See STATUS.md (S1 / L13).
