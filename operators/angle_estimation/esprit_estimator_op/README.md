# `ESPRITEstimatorOp`

Gridless least-squares ESPRIT sparse-path DOA: the same covariance / eig / EFT front-end as `MUSICEstimatorOp`, then a closed-form rotation of the signal subspace instead of a scan grid or a polynomial.

It occupies the same pluggable sparse slot after `SnapshotGatherOp`. A ULA (or an explicit `ShiftInvariantSubarrayPair`) resolves one coordinate; a filled URA with `subarray_spec: auto` resolves native `(az, el)` by dual-invariance joint diagonalization. The fact a caller must not miss: there is **no** `angle_spectrum`; an explicit `subarray_spec` and nonempty `spatial_smoothing` are mutually exclusive; `angle_convention` is required and locked after the first successful tick.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Rank-2 rejected |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional; covariance sums only marked slots. Omitted from `contract()` ([EA HSDK5 ONLY] `HSDK5_PORT.md` §11) |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Dominant DOA; always emitted |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Rotation eigenvalues mapped to angles; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | DOAs actually written |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Ascending covariance spectrum |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable |
| live | `spatial_smoothing` | `SpatialSmoothingOverride` | `{count, first, second}` int64 POD | `[live]` Exact override; `count` 0–2 selects 1-D / 2-D counts |
| live | `num_sources` | int64 | scalar | `[live]` Exact override; always declared, wire to enable. Negative → per-item EFT `auto` |
| live | `source_count_pfa` | float64 | scalar | `[live]` Exact override; always declared, wire to enable. EFT target Pfa for `auto` |
| live | `max_sources` | int64 | scalar | `[live]` Exact override; always declared, wire to enable. `doa_multi` capacity and EFT `K_max` |

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → ESPRITEstimatorOp → aoa_estimates
```

A holey MIMO virtual array needs `SubarraySelectionOp` (or spatial smoothing on an already-filled ULA/URA) before this slot.

Contract: [`OPERATOR_API.md` § `ESPRITEstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md) ·
open items: [`OPEN_ITEMS.md`](OPEN_ITEMS.md)

## Algorithm

Covariance, Hermitian `matx::eig` (VECTOR, ascending), and fixed-or-EFT model order match MUSIC (`run_esprit_stage` / `run_esprit_2d_stage` in `esprit_stage.hpp`). The signal subspace is the **last** `d` eigenvector columns (largest eigenvalues). v1 is **LS-ESPRIT** only (TLS / Unitary are deferred, OPEN_ITEMS §6).

**Subarray pair.** `subarray_spec` is structural and static. Empty vectors (`is_auto()`) derive:

- **ULA / line:** maximum-overlap pair `[0…M−2]` / `[1…M−1]` via `build_shift_invariant_subarray_plan` (common displacement validated against `array_geometry`).
- **Filled URA:** `analyze_planar_aperture` then `build_ura_dual_invariance`: y-shift (azimuth) `nz·(ny−1)` pairs and z-shift (elevation) `(nz−1)·ny` pairs, `make_ura_geometry` order `c = iz·ny + iy`.

An explicit (nonempty) pair always stays on the 1-D path and never runs planar analysis. Explicit pair + nonempty `spatial_smoothing` is rejected.

### 1-D LS rotation (`esprit_solve_kernel`)

For each item, with pair size `P` and model order `d ≤ min(P, max_sources)`:

```text
Ψ = (E_s,ref^H E_s,ref)^{-1} (E_s,ref^H E_s,shift)
```

Gram/cross are formed on the device; the inverse is Gaussian elimination with partial pivoting (singular → that item emits NaN / `source_count = 0`). Eigenvalues of `Ψ` come from an in-kernel shifted QR (`esprit_qr_eigenvalues`, max 192 iterations). Selected eigenvalues are those closest to the unit circle (`|1 − |μ||`); phase maps like Root-MUSIC’s ULA embedding:

```text
u = clamp(arg(μ) / (2π · Δ / λ), −1, 1)
dir = u · axis + sqrt(1 − u²) · broadside
```

`axis` / `broadside` / `Δ` come from the validated displacement. `angle_canonical_to_boundary` applies `angle_convention`.

### 2-D dual invariance (`esprit_solve_2d_kernel`)

Two rotations `Ψ_y`, `Ψ_z` are formed the same LS way on the y- and z-shift index sets. In the noiseless model they commute, so the unitary Schur basis `Z` that triangularizes `Ψ_y` also triangularizes `Ψ_z`:

```text
μ_y,i = (Z^H Ψ_y Z)_{ii}
μ_z,i = (Z^H Ψ_z Z)_{ii}
u_i = clamp(arg(μ_y,i) / (k·dy), −1, 1)
v_i = clamp(arg(μ_z,i) / (k·dz), −1, 1)
dir = u · y_axis + v · z_axis + sqrt(max(0, 1−u²−v²)) · normal
```

Pairing is automatic (no assignment step). Failed LU / Schur leaves that item blank. Explicit two-pair `subarray_spec`s are deferred (OPEN_ITEMS §7).

**Spatial smoothing** is the same S1 front-end as MUSIC. When enabled, leave `subarray_spec=auto` so pairs are re-derived on the reduced aperture. Warnings:

```text
ESPRITEstimatorOp: spatial smoothing override changes the ESPRIT aperture and derived subarray pair[s]
ESPRITEstimatorOp: unseen array_geometry override; revalidating the [URA dual-invariance pairs | shift-invariant subarray pair]
```

Rank-deficient `auto` warns once (same condition as Root-MUSIC).

## Shape and dtype rules

Precision is the template `T`. `snapshot` must be rank 3. Extra leading batch dims are not admitted.

- Output shapes match Root-MUSIC (four Tensors; `eigenvalues` use post-smoothing `C_est`).
- `num_items == 0` still emits all four.
- Mid-stream `max_sources` realloc is legal; a geometry / smoothing override re-derives the auto pair on the tick it arrives. `subarray_spec` is static, so an explicit pair never moves mid-stream.

Rejected combinations (operator’s message unless noted):

| Condition | Message |
| --- | --- |
| `snapshot` not rank 3 | `ESPRITEstimatorOp: snapshot must have shape (num_items, num_channels, num_snapshots)` |
| Channel mismatch | `ESPRITEstimatorOp: snapshot num_channels must match array_geometry.num_elements` |
| `max_sources` not in `[1, 64]` | `ESPRITEstimatorOp: max_sources must be in [1, 64]` |
| `source_count_pfa` not in `(0, 1)` | `ESPRITEstimatorOp: source_count_pfa must be in (0, 1)` |
| Explicit pair + smoothing | `ESPRITEstimatorOp: explicit subarray_spec cannot be combined with spatial_smoothing; use subarray_spec=auto on the reduced aperture` |
| Fixed `num_sources > P` (1-D) | `ESPRITEstimatorOp: num_sources must not exceed the number of corresponding subarray elements` |
| Fixed `num_sources > min(y_pairs, z_pairs)` (URA) | `ESPRITEstimatorOp: num_sources must not exceed the URA invariance subarray size` |
| Fixed `num_sources ≥ C_est` | `ESPRITEstimatorOp: num_sources must leave at least one noise-subspace dimension` |
| `num_channels > INT32_MAX` | `ESPRITEstimatorOp: num_channels exceeds index capacity` |
| `angle_convention` changed after first emit | `ESPRITEstimatorOp: angle_convention is static and cannot change after the first successful activation; instantiate a separate operator per convention` |
| Malformed pair / non-uniform auto ULA | `requested_pair.validate()` / `build_shift_invariant_subarray_plan` `invalid_argument` text |
| `is_valid` / pageable / dtype | same wording as MUSIC, with this operator name |
| Missing stream | `ESPRITEstimatorOp: no CUDA compute stream` |

A ULA `∥` canonical `+x` is rejected by the shift-invariant planner (same embedding uniqueness as Root-MUSIC). Tests cover `RejectsUlaParallelToBoresight` and `RejectsNonUniformLinearGeometry`.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override (§9) | Revalidates pair / URA invariances; rebuilds smoothing | Shift-invariant positions + `λ` |
| `angle_convention` | `AngleConvention` | none (required) | static (rejected if changed after first emit) | Seeds device relabeling once | Boundary meaning of `(az, el)` / `(u, v)` |
| `num_sources` | int64 | `-1` (`auto`) | `[live]` Exact | Free | Signal-subspace dimension |
| `source_count_pfa` | double | `1e-2` | `[live]` Exact | Free | Target EFT Pfa; `auto` only |
| `max_sources` | int64 | `4` | `[live]` Exact | Reallocates `doa_multi`; cap `64` | Capacity and EFT `K_max` |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD (§9) | Rebuilds plan + re-derives auto pairs | Per-axis counts; exclusive with explicit `subarray_spec` |
| `forward_backward` | bool | `true` | static | Part of the smoothing-plan key | FB looks; ignored unless smoothing is on |
| `subarray_spec` | `ShiftInvariantSubarrayPair` | auto (empty vectors) | static (Params-only by contract) | Rebuilds index H2D | Explicit corresponding indices, or auto ULA/URA pairs |

Displacement is derived from `array_geometry`, never stored on the descriptor. Same `num_sources` `int64` encoding as MUSIC.

## SDK implementation notes

- **Construction.** `ESPRITEstimatorOp(const MaxTensorExtents&)`. If both `max_items` and `max_sources` are unset, pool spans stay `0`. Otherwise `max_items`, `max_sources`, and `max_channels` must be positive. Same four-output spans as Root-MUSIC. `max_az` / `max_el` unused.
- **Triggering.** `contract()` is `OnEach{snapshot}` plus an unconditional `Exact{}` + `SafeDefault{}` entry for every override port. Wiring is the application's only obligation: an `add_flow` into a port enables that override, and leaving it unwired compiles (warning `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT`) and reports `kNotConnected` on every activation, which `compute()` reads as no override this tick. A wired port that is *not* in the contract still fails the second activation with `RUNTIME_OPERATOR_REPLAY_MISMATCH` (`HSDK5_PORT.md` §1) — which is why optional `is_valid`, omitted from the contract (§11), must stay unwired.
- **Ports and queues.** Device Tensor ports, queue depth `2`. All four outputs declare `max_rank = 3`. Live Inputs also queue depth `2` — wire-object, POD, and scalar payloads only (no config Tensor, since there is no scan grid).
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>`, a generated descriptor-schema type; `spatial_smoothing` is `Input<SpatialSmoothingOverride>`, a bounded `{count, first, second}` int64 POD (`count` must be 0, 1, or 2). Both convert with `from_wire` inside `resolve_live_override_via`, so neither wire storage nor the POD reaches operator state. `num_sources` / `source_count_pfa` / `max_sources` are plain scalar `Input<>` ports resolved with `resolve_live_override`. Logical `ESPRITParams` types are unchanged (`HSDK5_PORT.md` §9); `subarray_spec` has no port at all — it is static by contract.
- **Streams and lifetime.** `context.cuda_stream()`. Inputs retained at end of `compute()`. Rotation eigenproblems run in-kernel (no cuSOLVER). URA index upload **does** `cudaStreamSynchronize` once when the invariance tables are rebuilt. `stop()` calls `retire_music_matx_caches()`.
- **Emission.** Loan → fill → commit → emit (`aoa_estimates`, `doa_multi`, `source_count`, `eigenvalues`). **[EA HSDK5 ONLY]** all four always emitted (OPEN_ITEMS §4).
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.ESPRITParams` version `1`.
- **Precision.** `ESPRITEstimatorOp<float>` and `<double>`.
- **[EA HSDK5 ONLY]** All five override ports are unconditionally in `contract()`; geometry / smoothing Graph tests run. Runtime geometry producers connect directly to `array_geometry` with `WireArrayGeometry`; Python remains Params-only for schema payloads.

## Consumer lifetime notes

Every output is a fresh pool loan. A consumer may keep the Tensor for the usual Holoscan lifetime.

## Performance

**Cold.** First tick: `matx::eig` plan, optional smoothing plan, and (auto URA) dual-invariance index H2D + sync. First unseen geometry revalidates the pair.

**Steady.** Covariance + eig + EFT every tick; then one CUDA block per item (thread 0) for the `d×d` LS solve / QR or Schur. `d` is small (`≤ max_sources ≤ 64`). No per-item host `geev`. Per-tick MatX eig-plan clear matches MUSIC.

**`[live]` cost.** Scalar overrides cheap; `max_sources` reallocates `doa_multi`. A geometry / smoothing override revalidates the pair, rebuilds the smoothing plan, and (auto URA) re-uploads the invariance tables on the tick it arrives.

```bash
ctest --test-dir build -R perf_esprit_estimator_smoke
```

Harness axes: channels, detections, snapshots (ULA `xs`…`xl`); default matrix includes `ura_sm` (6×6); `--full` adds `ura_lg` (8×8). `num_sources` fixed at 1. Live-override timing deferred. No measured numbers are recorded in-repo for this Graph harness.

## Tests and verification

| Check | Proves |
| --- | --- |
| `RecoversKnownDoaUlaFixedOneSource` / `RecoversTwoSources` / `ResolvesCloselySpacedSources` | Gridless ULA DOA ≈ truth (TESTING.md) |
| `RecoversUraSingleSourceAzEl` / `…TwoSourcesAzElPairing` / `…Float` | Native `(az, el)`; joint-diagonalization pairing |
| `ExplicitSubarrayPairWorks` / `…RecoversTwoSources` / `RejectsMalformed…` | `ShiftInvariantSubarrayPair` validate + 1-D path |
| `RejectsExplicitSubarrayWithSpatialSmoothing` | Mutual exclusion |
| `SpatialSmoothing1DRecoversDoa` / `ForwardOnlySmoothingRecoversCoherentSources` | S1 front-end |
| `RejectsNonUniformLinearGeometry` / `RejectsUlaParallelToBoresight` | Geometry constraint |
| Convention / zenith / endfire / NaN pad | Same family as Root-MUSIC |
| `LiveArrayGeometryAppliesForOneTick` / `LiveSpatialSmoothingAppliesForOneTick` | Mid-stream override revalidates / rebuilds, and sticks after the next miss |
| `IsValidMask…` / `AllInvalidMask…` / `LiveNumSources…` / `LiveSourceCountPfa…` / `LiveMaxSources…` | **[EA HSDK5 ONLY]** Graph `GTEST_SKIP` |
| `RejectsSnapshotRank2` | Contract 3-D `snapshot` |

Analytic ground truth is a host steering-vector snapshot. C++ Graph tests bind all four outputs. OPEN_ITEMS §8 still wants an end-to-end `SubarraySelectionOp` → ESPRIT Graph test (not written).

```bash
ctest --test-dir build -R test_esprit_estimator --output-on-failure
```

## Usage

```yaml
esprit_estimator:
  angle_convention:
    preset: radar_boresight
  num_sources: -1
  source_count_pfa: 0.01
  max_sources: 4
  spatial_smoothing: []
  forward_backward: true
  # subarray_spec omitted → auto (empty reference_indices / shifted_indices)
```

```cpp
MaxTensorExtents extents;
extents.max_items = 4096;
extents.max_sources = 4;
extents.max_channels = 16;

auto esprit = graph.op<ESPRITEstimatorOp<float>>("esprit_estimator", extents);
esprit.set_params(params);  // ESPRITParams, including array_geometry

graph.add_flow(gather->snapshot, esprit->snapshot);
graph.add_flow(esprit->aoa_estimates, consumer->aoa_estimates);
graph.add_flow(esprit->doa_multi, doa_sink->in);
graph.add_flow(esprit->source_count, count_sink->in);
graph.add_flow(esprit->eigenvalues, eig_sink->in);
```

**[EA HSDK5 ONLY]** Bind all four device outputs. Python `ESPRITEstimatorOpF32` / `F64` bindings are deferred on this branch.

## Contract conformance

HSDK 5 EA C++ compute path matches the locked LS-ESPRIT ULA/URA algorithm; the scalar Graph override/`is_valid` harness and optional-output Connectivity are EA-deferred.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` in `contract()`, not `Input<ArrayGeometry>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9, OPEN_ITEMS §2) |
| `spatial_smoothing` **[live]** | conforms (payload form differs) | Bounded `{count, first, second}` int64 POD, not `Input<std::vector<int64_t>>` (§9) |
| `num_sources` / `source_count_pfa` / `max_sources` **[live]** | conforms | Unconditional Exact `Input<>` in `contract()`; the app enables each override by wiring the port. Graph mid-stream tests still skipped (§1) |
| Optional `is_valid` connected in Graph | EA-deferred | Omitted from `contract()` (§11); OPEN_ITEMS §3 |
| Optional `doa_multi` / `source_count` / `eigenvalues` | Done | All `always`: fused with `aoa_estimates` in the solve kernel, so unwired sinks are discarded. No `angle_spectrum` (OPEN_ITEMS §4) |
| `max_sources` unbounded except by `doa_multi` capacity | deviates | Rejects values outside `[1, 64]` (`kMaxSourcesCap` in `esprit_estimator_op.cu`) |

`subarray_spec` is **static** in the contract, so having no override port is conformance. TLS/Unitary variants are documented follow-ups, not unimplemented contract modes.

Audit scope (static; no build or test run): `OPERATOR_API.md` § ESPRITEstimatorOp, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md`, `STATUS.md`, group/common READMEs, `OPEN_ITEMS.md`, `esprit_estimator_op.hpp` / `.cu`, `esprit_stage.hpp`, `metadata.json`, C++/Python tests, `perf_esprit_estimator.cpp`, `shift_invariant_subarray.hpp`. Deviations that are team questions are proposed in chat, not appended to `OPEN_ITEMS.md`.
