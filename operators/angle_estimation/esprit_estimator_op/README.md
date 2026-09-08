# `ESPRITEstimatorOp`

Gridless least-squares ESPRIT sparse-path DOA: the same covariance / eig / EFT front-end as `MUSICEstimatorOp`, then a closed-form rotation of the signal subspace instead of a scan grid or a polynomial.

It occupies the same pluggable sparse slot after `SnapshotGatherOp`. A ULA (or an explicit `ShiftInvariantSubarrayPair`) resolves one coordinate; a filled URA with `subarray_spec: auto` resolves native `(az, el)` by dual-invariance joint diagonalization. The fact a caller must not miss: there is **no** `angle_spectrum`; an explicit `subarray_spec` and nonempty `spatial_smoothing` are mutually exclusive; `angle_convention` defaults to `radar_boresight` and is locked after the first successful tick.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Rank-2 rejected |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
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

Covariance, Hermitian `matx::eig` (VECTOR, ascending), and fixed-or-EFT model order match MUSIC (`run_esprit_stage` / `run_esprit_2d_stage` in `esprit_stage.hpp`). The signal subspace is the **last** `d` eigenvector columns (largest eigenvalues). v1 is **LS-ESPRIT** only; TLS and Unitary variants are deferred.

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

The tick shell follows `resolve/plan → optional mask → empty path → prepare effective covariance input → ESPRIT ULA/URA path → emit`. `prepare_smoothed_input` owns smoothing-plan cache updates, index uploads, and gather dispatch; with smoothing off it passes the original geometry, pointers, and dimensions through unchanged.

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
| `angle_convention` | `AngleConvention` | `radar_boresight` | static (rejected if changed after first emit) | Seeds device relabeling once | Boundary meaning of `(az, el)` / `(u, v)` |
| `num_sources` | int64 | `-1` (`auto`) | `[live]` Exact | Free | Signal-subspace dimension |
| `source_count_pfa` | double | `1e-2` | `[live]` Exact | Free | Target EFT Pfa; `auto` only |
| `max_sources` | int64 | `4` | `[live]` Exact | Reallocates `doa_multi`; cap `64` | Capacity and EFT `K_max` |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD (§9) | Rebuilds plan + re-derives auto pairs | Per-axis counts; exclusive with explicit `subarray_spec` |
| `forward_backward` | bool | `true` | static | Part of the smoothing-plan key | FB looks; ignored unless smoothing is on |
| `subarray_spec` | `ShiftInvariantSubarrayPair` | auto (empty vectors) | static (Params-only by contract) | Rebuilds index H2D | Explicit corresponding indices, or auto ULA/URA pairs |

Displacement is derived from `array_geometry`, never stored on the descriptor. Same `num_sources` `int64` encoding as MUSIC.

## SDK implementation notes

- **Construction.** `ESPRITEstimatorOp(const MaxTensorExtents&)`. If both `max_items` and `max_sources` are unset, pool spans stay `0`. Otherwise `max_items`, `max_sources`, and `max_channels` must be positive. Constructor-created runtime state owns stable covariance/eigenspace/count and six bounded small-matrix workspaces.
- **Triggering.** `contract()` is `OnEach{snapshot}`. Optional `is_valid` and every live override are unconditional non-members with `Exact{}` + `SafeDefault{}`; unwired means absent/all-valid or Params for that tick. A producer must publish a same-tick mask before `snapshot`.
- **Ports and queues.** Device Tensor ports, queue depth `2`. All four outputs declare `max_rank = 3`. Live Inputs also queue depth `2` — wire-object, POD, and scalar payloads only (no config Tensor, since there is no scan grid).
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>`, a generated descriptor-schema type; `spatial_smoothing` is `Input<SpatialSmoothingOverride>`, a bounded `{count, first, second}` int64 POD (`count` must be 0, 1, or 2). Both convert with `from_wire` inside `resolve_live_override_via`, so neither wire storage nor the POD reaches operator state. `num_sources` / `source_count_pfa` / `max_sources` are plain scalar `Input<>` ports resolved with `resolve_live_override`. Logical `ESPRITParams` types are unchanged (`HSDK5_PORT.md` §9); `subarray_spec` has no port at all — it is static by contract.
- **Tick flow.** `compute()` is the high-level shell: receive and plan → validate mask / stream / convention → empty path → prepare smoothed input and aperture analysis → dispatch the explicit auto-URA or shift-invariant 1-D helper → publish, seed the convention, and retain. Each variant helper owns its indices, workspace, output preparation, warnings, and execution call. Runtime state retains capacity-managed smoothing gather buffers and grows them only when a larger effective input arrives.
- **Streams and lifetime.** `context.cuda_stream()`. Inputs and smoothing gather buffers remain alive through estimator execution. `ESPRITPreparedCommonWorkspace` and `ESPRITPreparedOutputs` keep scratch pointers, loans, and write guards alive through either variant; all siblings commit before required-first publication. Rotation eigenproblems run in-kernel (no cuSOLVER). URA index upload **does** `cudaStreamSynchronize` once when the invariance tables are rebuilt. `stop()` calls `retire_music_matx_caches()`.
- **Emission.** Required `aoa_estimates` publishes first; optional/always `doa_multi`, `source_count`, and `eigenvalues` absorb declined emits.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.ESPRITParams` version `1`.
- **Precision.** `ESPRITEstimatorOp<float>` and `<double>`.
- **[EA HSDK5 ONLY]** All five override ports are unconditionally in `contract()`; geometry / smoothing Graph tests run. Runtime geometry producers connect directly to `array_geometry` with `WireArrayGeometry`; Python remains Params-only for schema payloads.

## Consumer lifetime notes

Every output is a fresh pool loan. A consumer may keep the Tensor for the usual Holoscan lifetime.

## Performance

**Cold.** First tick: `matx::eig` plan, optional smoothing plan, and (auto URA) dual-invariance index H2D + sync. First unseen geometry revalidates the pair.

**Steady.** Covariance + eig + EFT run on stable constructor-created workspace; then one CUDA block per item (thread 0) handles the `d×d` LS solve / QR or Schur. `d` is small (`≤ max_sources ≤ 64`). No per-item host `geev`.

**`[live]` cost.** Scalar overrides are cheap; `max_sources` reallocates `doa_multi`. A geometry / smoothing override revalidates the pair, rebuilds the smoothing plan, and (auto URA) re-uploads the invariance tables on the tick it arrives. The Graph harness isolates all five live inputs.

```bash
ctest --test-dir build -R perf_esprit_estimator_smoke
```

Harness axes: channels, detections, snapshots (ULA `xs`…`xl`); default matrix includes `ura_sm` (6×6); `--full` adds `ura_lg` (8×8). Each size runs isolated production-Exact scenarios for `array_geometry`, `spatial_smoothing`, `num_sources`, `source_count_pfa`, and `max_sources`. The Pfa scenario uses automatic source count; other steady paths fix one source.

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
| `LiveArrayGeometryAppliesForOneTick` / `LiveSpatialSmoothingAppliesForOneTick` | Mid-stream override applies for one tick; the next miss reverts to Params |
| `IsValidMask…` / `AllInvalidMask…` | Connected Exact mask and all-invalid blanking |
| `LiveNumSources…` / `LiveSourceCountPfa…` / `LiveMaxSources…` | Scalar Exact override and revert |
| `ReducedSubarrayDirectlyDrivesEstimator` | `SubarraySelectionOp` snapshot and runtime geometry directly drive ESPRIT |
| `RejectsSnapshotRank2` | Contract 3-D `snapshot` |

Analytic ground truth is a host steering-vector snapshot. C++ Graph tests bind all four outputs and include the direct reduced-subarray integration path.

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

The HSDK 5 C++ compute path matches the locked LS-ESPRIT ULA/URA algorithm and optional-port convention.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` in `contract()`, not `Input<ArrayGeometry>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9) |
| `spatial_smoothing` **[live]** | conforms (payload form differs) | Bounded `{count, first, second}` int64 POD, not `Input<std::vector<int64_t>>` (§9) |
| `num_sources` / `source_count_pfa` / `max_sources` **[live]** | conforms | Unconditional Exact inputs with active mid-stream/revert tests |
| Optional `is_valid` connected in Graph | conforms | Non-member Exact + SafeDefault; connected mask tests run |
| Optional `doa_multi` / `source_count` / `eigenvalues` | conforms | All `always`: fused with `aoa_estimates`, so unwired sinks are discarded. No `angle_spectrum` |
| `max_sources` unbounded except by `doa_multi` capacity | deviates | Rejects values outside `[1, 64]` (`kMaxSourcesCap` in `esprit_estimator_op.cu`) |

`subarray_spec` is **static** in the contract, so having no override port is conformance. TLS/Unitary variants are documented follow-ups, not unimplemented contract modes.

Audit scope (static; no build or test run): `OPERATOR_API.md` § ESPRITEstimatorOp, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md`, `STATUS.md`, group/common READMEs, `OPEN_ITEMS.md`, `esprit_estimator_op.hpp` / `.cu`, `esprit_stage.hpp`, `metadata.json`, C++/Python tests, `perf_esprit_estimator.cpp`, `shift_invariant_subarray.hpp`. Deviations that are team questions are proposed in chat, not appended to `OPEN_ITEMS.md`.
