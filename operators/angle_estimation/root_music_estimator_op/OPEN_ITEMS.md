# RootMUSICEstimatorOp — open items

Status: **HSDK 5 EA C++ port landed** (checklist not fully closed — live /
`is_valid` Graph coverage deferred). Algorithm contract
(`OPERATOR_API.md` → RootMUSICEstimatorOp) is unchanged:

- gridless ULA / filled-URA Root-MUSIC (companion + `cusolverDnXgeev`);
- shared MUSIC covariance / eig / fixed-or-EFT source count front-end;
- `num_sources` / `source_count_pfa` / `max_sources` **[live]** POD sticky Inputs;
- `spatial_smoothing` (+ `forward_backward`) front-end on filled ULA/URA;
- always-emit `aoa_estimates` / `doa_multi` / `source_count` / `eigenvalues`
  (no `angle_spectrum`).

Cross-cut SDK / migration detail lives in
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

---

## 1. Graph mid-stream sticky `[live]` POD coverage

**Blocks checklist:** Tests → “[live] overrides: mid-stream + sticky” for
`num_sources` / `source_count_pfa` / `max_sources`.

**Class:** HSDK 5 EA limitation. See
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1.

**Interim.**

- Operator still declares sticky `Input<int64_t> num_sources`,
  `Input<double> source_count_pfa`, and `Input<int64_t> max_sources`;
  `compute()` resolves via `resolve_sticky_override`.
- Those ports are omitted from `contract()` so unwired graphs compile.
- `LiveNumSourcesStickyOverride` / `LiveSourceCountPfaStickyOverride` /
  `LiveMaxSourcesStickyOverride` are `GTEST_SKIP`’d.
- `perf_root_music_estimator` measures an inline `max_sources` transition
  with the supported single-clock / exact-contract / single-partition shape.
- Params authoring covers selection
  (`NumSourcesConfigurableFromParams` /
  `SourceCountPfaConfigurableFromParams` /
  `MaxSourcesConfigurableFromParams`).

**Next.** Restore sticky temporal policy + Graph sticky unit tests on a
newer EA / GA (shared recipe with ClutterFilter §1).

**Status:** open for sticky unit tests and isolated `num_sources` /
`source_count_pfa` perf; `max_sources` perf timing is restored.

---

## 2. Non-POD / variable-length `[live]` params are Params-only

**Blocks checklist:** Tests → live `array_geometry` / `spatial_smoothing`
mid-stream sticky overrides.

**Class:** Durable HSDK 5 (`PublishedPayload` admission) + our deferred
work. Same recipe as MVDR / Beamscan — see
[`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9.

**Interim.** Those parameters remain on `RootMUSICParams` only
(`set_params`). Live Graph override tests are `GTEST_SKIP`’d.
`angle_convention` and `forward_backward` stay Params-only (STATIC /
not `[live]`).

**Next.** Prefer Tensor live ports for smoothing counts and/or ChannelPod /
schema wrappers for `ArrayGeometry` once a module-wide recipe lands (§9).

**Status:** open — ported compute path; live descriptor/vector ports
deferred.

---

## 3. Optional `is_valid` Graph wiring

**Blocks checklist:** Tests → optional `is_valid` connected Graph paths.

**Class:** HSDK 5 EA limitation (related to §1). Same as MVDR OPEN_ITEMS §3.

**Interim.** Port registered in `setup()`; omitted from `contract()`;
`compute()` uses `receive_optional_value`. Graph tests
`IsValidMaskExcludesWrongSnapshots` / `AllInvalidMaskBlanksDoa` are
`GTEST_SKIP`’d.

**Status:** open — EA.

---

## 4. Optional outputs always emitted

**Contract.** `OPERATOR_API.md` marks `doa_multi`, `source_count`, and
`eigenvalues` optional. On HSDK 5 EA the operator **always allocates and
emits** all four Tensor outputs each tick (no `angle_spectrum`). C++ Graph
tests wire and `bind_tensor_output_device` all four.

**Status:** open — document for apps; not a blocker for the C++ port.

---

## 5. MatX / cuSOLVER teardown via `stop()`

**Status:** preserved. HSDK 5 `OperatorBase::stop()` still runs at the
`kStop` lifecycle checkpoint. `RootMUSICEstimatorOp::stop()` calls
`detail::retire_music_matx_caches()` (full `GetAllocMap()` drain once while
CUDA is valid). Per-item `cusolverDnXgeev` workspaces live in
`RootMUSICSolverState` and are destroyed with the operator.

---

## 6. Batched polynomial root solver (performance)

Unchanged from the HSDK 4.x landing: serial hybrid `cusolverDnXgeev` per
item dominates latency. Postponed by team decision (2026-07-22). See prior
measurements in git history / perf harness. Not a contract gap.

**Status:** postponed — performance follow-up.

---

## 7. `source_count_pfa` contract listing

**Context:** `OPERATOR_API.md` references EFT/auto inside `num_sources` text
for Root-MUSIC/ESPRIT but does not list `source_count_pfa` as its own
parameter row (MUSIC lists it explicitly).

**Impact:** Implementation exposes `source_count_pfa` **[live]** like MUSIC;
no behavioral gap.

**Status:** open — doc harmonization only.

---

## 8. 2-D Root-MUSIC variant

**Status:** landed (2026-07-23) — 2-D URA reduced-dimension + pairing
implemented; joint 2-D polynomial / explicit two-pair specs deferred.

---

## Python bindings / perf harness

**Class:** Our deferred work (HSDK5_PORT §6). Existing pybind trampoline is
still HSDK 4.x and is not built while `HOLOSCAN_RADAR_BUILD_PYTHON=OFF`.
`perf_root_music_estimator` is Graph-ported and reports `max_sources`
`override_ms` (§8).

**Status:** open — Python deferred; not required to land the C++ HSDK 5 port.
