# `RootMUSICEstimatorOp`

## Purpose

Gridless Root-MUSIC sparse-path DOA: the same covariance / eig / EFT front-end as `MUSICEstimatorOp`, then polynomial roots instead of a scan grid. It fills the gridless sparse-estimator slot after snapshot gathering when the array is a supported ULA or filled URA.

## Responsibilities

- Forms covariance matrices, selects the signal subspace, and obtains gridless directions from Root-MUSIC polynomials.
- Owns ULA/filled-URA geometry validation, source-count handling, root selection, pairing, and validity diagnostics.
- Does not emit a pseudospectrum or operate directly on arbitrary holey/conformal arrays.

## When to use

Use after `SnapshotGatherOp` for gridless multi-source estimation on a uniform linear array or supported filled URA. Prefer it when avoiding scan-grid cost is important and no spectrum output is needed. Do not use it with arbitrary conformal or holey arrays; select a valid subarray or choose a scan-based estimator instead.

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → RootMUSICEstimatorOp → aoa_estimates
```

Declarations and parameter defaults: [root_music_estimator_op.hpp](root_music_estimator_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Required trigger/aligned input | Rank-2 rejected |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional Exact side input | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Required output | Dominant DOA; always emitted |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Required output | Selected roots / pairs; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | Required output | Roots / pairs actually written |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Required output | Ascending full-aperture covariance spectrum |
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
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override | ULA or filled URA + `λ`; `num_elements` must match channels Cost: Revalidates ULA/URA; rebuilds smoothing plan; may resize cuSOLVER workspace. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static (rejected if changed after first emit) | Boundary meaning of `(az, el)` / `(u, v)` Cost: Seeds device relabeling once. |
| `num_sources` | int64 | `-1` (`auto`) | `[live]` Exact | Signal-subspace dimension / root count Cost: Free (EFT or fill). |
| `source_count_pfa` | double | `1e-2` | `[live]` Exact | Target EFT Pfa; `auto` only. See [CLARAHOLOS-3394](https://jirasw.nvidia.com/browse/CLARAHOLOS-3394). Cost: Free. |
| `max_sources` | int64 | `4` | `[live]` Exact | Capacity and EFT `K_max` Cost: Reallocates `doa_multi`; cap `64`. |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD | Per-axis counts on a filled ULA/URA Cost: Rebuilds plan + H2D; changes polynomial degree. |
| `forward_backward` | bool | `true` | static | FB looks; ignored unless smoothing is on Cost: Part of the smoothing-plan key. |

## Algorithm

Covariance, Hermitian `matx::eig` (VECTOR, ascending), fixed-or-EFT model order, and the noise-subspace projector `P_n` are the MUSIC front-end (`music_covariance_kernel` / `projector_kernel` via `run_root_music_stage` / `run_root_music_2d_stage` in `root_music_stage.hpp`). Geometry is auto-detected with `analyze_planar_aperture`: a filled URA (`ny > 1 && nz > 1`, `make_ura_geometry` order `c = iz * ny + iy`) takes the 2-D path; a line array takes the 1-D cone path.

### 1-D ULA

Polynomial coefficients are the anti-diagonals (lags) of `P_n` (`root_music_coefficients_kernel`):

```text
degree = 2 · (C_est − 1)
c[ℓ + (C−1)] = Σ_row P_n[row, row+ℓ]     for ℓ ∈ [1−C, C−1]
```

The companion matrix of `c` (`root_music_companion_kernel`, last column `−c[row]/c[degree]`) is solved per item with `cusolverDnXgeev` (`CUSOLVER_EIG_MODE_NOVECTOR`). Empty items leave a zero companion; a vanishing leading coefficient writes a NaN matrix, which is detected and blanks that item’s roots before calling `geev`. Finite companions that do not converge (`info > 0`) also blank only that item’s roots. `info < 0` throws:

```text
RootMUSICEstimatorOp: cusolverDnXgeev illegal argument for item {i} (info {info})
```

`root_music_select_kernel` keeps roots with `|z| ≤ 1` and finite magnitude, de-duplicates phases within `1e-4` rad, and picks the `min(d_i, max_sources)` closest to the unit circle (`|1 − |z||`). Spatial cosine and the forward-hemisphere embedding (ULA axis + broadside orthogonal to canonical `+x`):

```text
u = clamp(arg(z) / (2π · Δ / λ), −1, 1)
dir = u · axis + sqrt(1 − u²) · broadside
```

`analyze_ula` requires consecutive, equally spaced positions in snapshot channel order and a unique broadside (`ULA ∦ +x`). `angle_canonical_to_boundary` then applies `angle_convention`. Elevation on a ULA is the degenerate second coordinate of that embedding (≈ 0 under `radar_boresight` when the ULA lies in y).

### 2-D URA

The full `C = ny · nz` covariance still yields one shared `P_n`. `ura_axis_covariance_kernel` forms row- and column-averaged sub-covariances (forward average over the orthogonal axis):

```text
R_y[a,b] = (1/nz) Σ_iz R[iz·ny+a, iz·ny+b]
R_z[a,b] = (1/ny) Σ_iy R[a·ny+iy, b·ny+iy]
```

Each axis is rooted with the 1-D machinery at a model order clamped to that axis’s noise budget (`min(d_i, N_axis−1)`). Candidates are direction cosines `u_i`, `v_k` (same in-circle / phase-dedup rule), at most `kRootMusicPairCandCap = 8` per axis. `root_music_pair_kernel` evaluates the full 2-D MUSIC metric `1 / (a(u,v)^H P_n a(u,v))` at every pair (denominator floored at `kMusicDenomFloor = 1e-6`) and greedily selects the `d` mutually exclusive best pairs. Steering on the lattice:

```text
a_c = exp(j · (iy · k·dy · u + iz · k·dz · v)),   c = iz·ny + iy
dir = u · y_axis + v · z_axis + sqrt(max(0, 1−u²−v²)) · normal
```

There is no canonical single-polynomial 2-D Root-MUSIC; this reduced-dimension + pairing method is the reduced-dimension design documented here. Joint 2-D polynomials and explicit non-ULA per-axis subarray specifications are not supported.

**Spatial smoothing** is the same S1 front-end as MUSIC (per-axis counts, optional forward-backward). Changing an already-built plan logs:

```text
RootMUSICEstimatorOp: spatial smoothing override changes the Root-MUSIC aperture and polynomial degree
```

A newly seen `array_geometry` after the first warns `unseen array_geometry override; revalidating ULA|URA aperture and solver dimensions`. Rank-deficient `auto` (no mask, `K < C_est`) warns once: `num_sources=auto with fewer snapshots than channels; falling back to one source`.

The tick shell follows `resolve/plan → optional mask → empty path → prepare effective covariance input → Root-MUSIC ULA/URA path → emit`. `prepare_smoothed_input` owns smoothing-plan cache updates, index uploads, and gather dispatch; with smoothing off it passes the original geometry, pointers, and dimensions through unchanged.

## Shape and dtype rules

Precision is the template `T` ∈ {`float`, `double`}. `snapshot` must be rank 3; rank 2 and extra leading batch dims are rejected.

- `aoa_estimates`: `(num_items, 2)`.
- `doa_multi`: `(num_items, effective_max_sources, 2)`.
- `source_count`: `(num_items,)`.
- `eigenvalues`: `(num_items, C_est)` with `C_est` the post-smoothing channel count (full `ny·nz` on a URA).
- `num_items == 0` still emits all four (eigenvalues use the pre-smoothing channel count).
- Mid-stream `max_sources` realloc is legal; a geometry / smoothing override that alters `C_est` / polynomial degree takes effect on the tick it arrives.

Rejected combinations (operator’s message unless noted):

| Condition | Message |
| --- | --- |
| `snapshot` not rank 3 | `RootMUSICEstimatorOp: snapshot must have shape (num_items, num_channels, num_snapshots)` |
| Channel mismatch | `RootMUSICEstimatorOp: snapshot num_channels must match array_geometry.num_elements` |
| `max_sources` not in `[1, 64]` | `RootMUSICEstimatorOp: max_sources must be in [1, 64]` |
| `source_count_pfa` not in `(0, 1)` | `RootMUSICEstimatorOp: source_count_pfa must be in (0, 1)` |
| Fixed `num_sources > C_est−1` (ULA) | `RootMUSICEstimatorOp: num_sources must leave at least one noise-subspace dimension` |
| Fixed `num_sources > min(ny,nz)−1` (URA) | `… must leave at least one noise-subspace dimension on each URA aperture axis` |
| URA `ny·nz > 64` | `RootMUSICEstimatorOp: 2-D URA pairing supports at most 64 channels (Ny*Nz); reduce the aperture upstream` |
| Non-uniform / not in channel order | `RootMUSICEstimatorOp: array_geometry must be a uniformly spaced linear array in snapshot channel order; use SubarraySelectionOp upstream for non-uniform geometry` |
| Tagged non-ULA on the 1-D path | `RootMUSICEstimatorOp: array_geometry must be a ULA` |
| `< 2` elements | `RootMUSICEstimatorOp: array_geometry must contain at least two elements` |
| ULA `∥` canonical `+x` | `RootMUSICEstimatorOp: a ULA parallel to canonical boresight has no unique forward-hemisphere 1-D DOA embedding` |
| `angle_convention` changed after first emit | `RootMUSICEstimatorOp: angle_convention is static and cannot change after the first successful activation; instantiate a separate operator per convention` |
| `is_valid` dtype / shape | same wording as MUSIC, with this operator name |
| Pageable / non-contiguous / dtype | shared `require_*` strings with `RootMUSICEstimatorOp` |
| Holey smoothing | `spatial_smoothing: array_geometry is holey …` (S3) |
| Missing stream | `RootMUSICEstimatorOp: no CUDA compute stream` |

`analyze_planar_aperture` also throws when a geometry tagged `kUra` is not a filled orthogonal lattice (helper text, not the operator prefix).

## Performance

**Cold.** The constructor creates both cuSOLVER states and bounded core workspace. First use initializes solver internals, an optional smoothing plan, and the first `matx::eig` plan. A new geometry key revalidates the aperture.

**Steady.** Covariance + eig + projector every tick on stable operator-owned workspace; then one serial `cusolverDnXgeev` per item per rooted polynomial (ULA: one of degree `2(C−1)`; URA: two smaller companions plus pairing). The serial hybrid dominates latency; a batched device root solver is postponed, not a contract gap. The private ULA/URA end-clear sites have been removed (one alternate site per nonempty tick); matching MatX eig plans are retained until eviction/stop, with both live marks and existing stop retirement unchanged. Distinct configuration keys can retain workspace; current serialized device 0 scope does not establish memory bounds across arbitrary CUDA contexts or configuration histories.

**`[live]` cost.** Scalar overrides are cheap. `max_sources` reallocates `doa_multi`. A geometry / smoothing override revalidates the aperture and rebuilds the smoothing plan (and can resize the cuSOLVER workspace) on the tick it arrives.

Harness axes: channels, detections, snapshots (`xs`…`xl` ULA); `--full` also adds URA rungs (`ura_sm` 6×6, `ura_lg` 8×8). Every geometry isolates production Exact timing for `array_geometry`, `spatial_smoothing`, `num_sources`, `source_count_pfa`, and `max_sources`, plus separate fixed model-order scenarios at 1, 2, and `min(4, num_channels − 1)`.

## Usage

```yaml
root_music_estimator:
  angle_convention:
    preset: radar_boresight
  num_sources: -1
  source_count_pfa: 0.01
  max_sources: 4
  spatial_smoothing: []
  forward_backward: true
```

```cpp
MaxTensorExtents extents;
extents.max_items = 4096;
extents.max_sources = 4;
extents.max_channels = 16;

auto root = graph.op<RootMUSICEstimatorOp<float>>(
    "root_music_estimator", extents);
root.set_params(params);  // RootMUSICParams, including array_geometry

graph.add_flow(gather->snapshot, root->snapshot);
graph.add_flow(root->aoa_estimates, consumer->aoa_estimates);
graph.add_flow(root->doa_multi, doa_sink->in);
graph.add_flow(root->source_count, count_sink->in);
graph.add_flow(root->eigenvalues, eig_sink->in);
```
