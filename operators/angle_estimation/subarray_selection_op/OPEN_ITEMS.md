# SubarraySelectionOp — open items

Status: **settled — S3 interface resolved; all four team forks ratified (L14).**
HSDK 5 port landed with the EA interim notes below.

Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual array
and re-emits a reduced `(snapshot, array_geometry)` pair, so the gridless
estimators (Root-MUSIC / ESPRIT) and holey-array spatial smoothing (S1) can run.
Selection is host-planned (`radar_common/dsp/subarray_selection`) and cached; the
per-tick channel gather reuses the S1 gather (`run_spatial_smoothing_gather`,
single sub-array, no forward-backward).

- **Auto (default):** largest filled ULA (longest run) / URA (maximal all-filled
  rectangle), objective = max element count, ties broken by larger azimuth
  aperture. A filled array is a no-op pass-through.
- **Explicit:** caller-supplied channel indices, validated to form a filled
  uniform ULA/URA and re-emitted in lattice order.

The four selection forks (auto-vs-explicit, objective, contiguous-vs-sum-set,
standalone-vs-`VirtualArrayAssemblyOp` mode) are settled — see STATUS.md →
Settled decisions (L14).

## Verifiability caveat

S3's headline consumers (Root-MUSIC / ESPRIT) are not yet implemented, so the op
is unit-tested on synthetic holey geometries and validated end-to-end only
through MVDR spatial smoothing on a reduced array; the gridless-estimator wiring
follows when those operators land.

---

## HSDK 5 EA / port follow-ups

Cross-cut detail:
[`docs/planning/HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md).

### 1. Non-POD `[live]` `array_geometry` / `subarray` are Params-only

**Class:** Durable HSDK 5 (`PublishedPayload` admission) — HSDK5_PORT §9.

`Input<ArrayGeometry>` and `Input<std::vector<int64_t>>` are not valid
PublishedPayloads. Those parameters stay on `SubarraySelectionParams` only
(`set_params`), so this operator declares no `expect_*` / `sticky_*` fields.
Params coverage: `SubarrayConfigurableFromParams` + selection Graph tests.

**Next.** Tensor / POD / schema live ports once a module-wide recipe lands.
The wiring they would use is already settled: join the trigger with `Exact{}`
plus `on_missing SafeDefault{}` behind a per-port `expect_*` (default false),
with `sticky_*` (default true) latching the last match — see
[`TEMPORAL_CONTRACTS.md`](../../../docs/planning/TEMPORAL_CONTRACTS.md).

### 2. `array_geometry` output is a packed float64 Tensor

**Class:** Durable HSDK 5 — HSDK5_PORT §10.

`Output<ArrayGeometry>` does not compile (owns `vector` / `optional`). The port
still emits geometry each tick as a float64 Tensor
(`array_geometry_tensor.hpp`). VirtualArrayAssembly faces the same blocker when
ported. Params side-channel is not acceptable.

**Next.** Restore typed descriptor output when SDK admits a durable carrier.

### 3. Optional `is_valid` Graph wiring

**Class:** HSDK 5 EA limitation — no optional input (HSDK5_PORT §11).

Port registered in `setup()`, omitted from `contract()`, read with
`receive_optional_value` and forwarded to `is_valid_out` when present.
Connected Graph paths stay deferred: declaring the entry would fail a
mask-less graph at compile (`TEMPORAL_INPUT_UNCONNECTED`), and wiring the
mask without declaring it kills the second activation
(`RUNTIME_OPERATOR_REPLAY_MISMATCH`). The `expect_*` gate that unblocked the
POD `[live]` overrides is not applied here — the contract calls `is_valid` a
plain optional input, not a `[live]` parameter, so adding a flag would be a
contract change.

### 4. Physical output renames (`snapshot_out` / `is_valid_out`)

**Class:** Durable HSDK 5 unique in/out names — HSDK5_PORT §2.

Documented in `OPERATOR_API.md`. Revisit module-wide naming before dropping
`[EA HSDK5 ONLY]` fences.

### 5. Python bindings / Graph perf

Deferred with the module-wide Python / perf packaging wave (HSDK5_PORT §6 / §8).

## Deferred (algorithm)

- Sum-set (non-contiguous) holey smoothing (`S_b + S_c`; arXiv:2401.06502).
- Multiple simultaneous sub-arrays / co-array DOF expansion.
- 3-D apertures (rejected today, as S1).
