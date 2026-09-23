# `ESPRITEstimatorOp`

## Purpose

Gridless least-squares ESPRIT sparse-path DOA: the same covariance / eig / EFT front-end as `MUSICEstimatorOp`, then a closed-form rotation of the signal subspace instead of a scan grid or a polynomial. It fills the gridless sparse-estimator slot after snapshot gathering when valid shift-invariant subarrays are available.

## Responsibilities

- Forms covariance matrices, estimates or accepts source count, and solves the shift-invariant signal-subspace rotation.
- Supports ULA, filled-URA, and explicit compatible subarray-pair geometry, including tick-local geometry and smoothing overrides.
- Emits sparse DOA rows and validity/model-order diagnostics; it does not produce a scan spectrum or repair unsupported geometry.

## When to use

Use after `SnapshotGatherOp` when a ULA, filled URA, or explicit shift-invariant subarray pair supports gridless multi-source estimation. Prefer it over scan-based MUSIC when grid construction and spectrum output are unnecessary. Do not use it with arbitrary holey geometry, when an angle spectrum is required, or with explicit subarray pairs combined with spatial smoothing.

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → ESPRITEstimatorOp → aoa_estimates
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Required trigger/aligned input | Rank-2 rejected |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional Exact side input | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Required output | Dominant DOA; always emitted |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Required output | Rotation eigenvalues mapped to angles; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | Required output | DOAs actually written |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Required output | Ascending covariance spectrum |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable |
| live | `spatial_smoothing` | `SpatialSmoothingOverride` | `{count, first, second}` int64 POD | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; `count` 0–2 selects 1-D / 2-D counts |
| live | `num_sources` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. Negative → per-item EFT `auto` |
| live | `source_count_pfa` | float64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. EFT target Pfa for `auto` |
| live | `max_sources` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. `doa_multi` capacity and EFT `K_max` |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | Omit both `max_items` and `max_sources` for zero-span output bounds, or provide positive `max_items`, `max_sources`, and `max_channels` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override | Shift-invariant positions + `λ` Cost: Revalidates pair / URA invariances; rebuilds smoothing. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static (rejected if changed after first emit) | Boundary meaning of `(az, el)` / `(u, v)` Cost: Seeds device relabeling once. |
| `num_sources` | int64 | `-1` (`auto`) | `[live]` Exact | Signal-subspace dimension Cost: Free. |
| `source_count_pfa` | double | `1e-2` | `[live]` Exact | Target EFT Pfa; `auto` only Cost: Free. |
| `max_sources` | int64 | `4` | `[live]` Exact | Capacity and EFT `K_max` Cost: Reallocates `doa_multi`; cap `64`. |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD | Per-axis counts; exclusive with explicit `subarray_spec` Cost: Rebuilds plan + re-derives auto pairs. |
| `forward_backward` | bool | `true` | static | FB looks; ignored unless smoothing is on Cost: Part of the smoothing-plan key. |
| `subarray_spec` | `ShiftInvariantSubarrayPair` | auto (empty vectors) | static (Params only) | Explicit corresponding indices, or auto ULA/URA pairs Cost: Rebuilds index H2D. |

Contract: [`OPERATOR_API.md` § `ESPRITEstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md) ·

## Algorithm

Covariance, Hermitian `matx::eig` (VECTOR, ascending), and fixed-or-EFT model order match MUSIC (`run_esprit_stage` / `run_esprit_2d_stage` in `esprit_stage.hpp`). The signal subspace is the **last** `d` eigenvector columns (largest eigenvalues). Only **LS-ESPRIT** is supported; TLS and Unitary ESPRIT are not implemented.

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

Pairing is automatic (no assignment step). Failed LU / Schur leaves that item blank. Explicit two-pair `subarray_spec`s are not supported; the current contract does not expose them.

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

## Performance

**Cold.** First tick: `matx::eig` plan, optional smoothing plan, and (auto URA) dual-invariance index H2D + sync. First unseen geometry revalidates the pair.

**Steady.** Covariance + eig + EFT run on stable constructor-created workspace; then one CUDA block per item (thread 0) handles the `d×d` LS solve / QR or Schur. `d` is small (`≤ max_sources ≤ 64`). No per-item host `geev`.

**`[live]` cost.** Scalar overrides are cheap; `max_sources` reallocates `doa_multi`. A geometry / smoothing override revalidates the pair, rebuilds the smoothing plan, and (auto URA) re-uploads the invariance tables on the tick it arrives. The Graph harness isolates all five live inputs.

```bash
ctest --test-dir build -R perf_esprit_estimator_smoke
```

Harness axes: channels, detections, snapshots (ULA `xs`…`xl`); default matrix includes `ura_sm` (6×6); `--full` adds `ura_lg` (8×8). Each size runs isolated production-Exact scenarios for `array_geometry`, `spatial_smoothing`, `num_sources`, `source_count_pfa`, and `max_sources`. The Pfa scenario uses automatic source count; other steady paths fix one source.

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
