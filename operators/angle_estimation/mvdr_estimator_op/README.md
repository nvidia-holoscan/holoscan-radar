# `MVDREstimatorOp`

## Purpose

Adaptive MVDR / Capon sparse-path DOA: for each gathered snapshot item it forms a sample covariance, regularizes and inverts it, and scans an `(az, el)` grid for the Capon power peak. It fills the sparse-estimator slot after snapshot gathering when adaptive interference nulling is more important than a conventional scan.

## Responsibilities

- Forms and diagonally loads each covariance matrix, solves the Capon response, and extracts the strongest grid direction.
- Owns scan-grid and geometry validation, optional spatial smoothing, validity propagation, and spectrum output.
- Does not estimate model order, apply an array taper, or gather snapshots from range-Doppler maps.

## When to use

Use after `SnapshotGatherOp` when adaptive null placement and interference rejection matter more than conventional-beamforming simplicity. Supply enough valid snapshots for a useful covariance estimate and use diagonal loading for poorly conditioned cases. Prefer Beamscan when covariance adaptation is unnecessary, and do not use MVDR when model-order estimation or an array taper is required.

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → MVDREstimatorOp → aoa_estimates
```

Declarations and parameter defaults: [mvdr_estimator_op.hpp](mvdr_estimator_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Required trigger/aligned input | Exact rank 3; no opaque leading batch |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional Exact side input | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Required output | Dominant Capon DOA `(az, el)` or `(u, v)` |
| out | `angle_spectrum` | float32 / float64 | `(num_items, num_az, num_el)` | Optional `always` output | Capon spectrum. Optional (`always` disposition): the grid is the same pass that yields `aoa_estimates`, so it is produced regardless and an unwired sink is silently discarded |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Required output | Up to `max_sources` peaks; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | Required output | Peaks actually written into `doa_multi` |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Required output | Ascending **unloaded** covariance spectrum |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable |
| live | `az_coordinates` / `el_coordinates` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; extents bounded by `max_az` / `max_el` |
| live | `spatial_smoothing` | `SpatialSmoothingOverride` | `{count, first, second}` int64 POD | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; `count` 0–2 selects 1-D / 2-D counts |
| live | `diagonal_loading` | float64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. Negative → `auto` |
| live | `max_sources` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. `doa_multi` capacity |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | Omit both `max_items` and `max_az` for zero-span output bounds, or provide positive `max_items`, `max_az`, `max_el`, `max_sources`, and `max_channels` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `cache_budget_bytes` | unsigned bytes | 67108864 (64 MiB) | static | Instance-wide retained cache ceiling; 0 disables retention. Oversized entries execute uncached with a warning. |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override | Element positions + `λ` Cost: Rebuilds steering (+ smoothing plan if enabled). |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Boundary `(az, el)` ↔ unit-vector map Cost: Fixes output meaning for the graph. |
| `az_coordinates` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_az` | First coordinate: azimuth radians for angular conventions, dimensionless `u` for `direction_cosine` Cost: Rebuilds steering manifold. |
| `el_coordinates` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_el` | Second coordinate: elevation radians for angular conventions, dimensionless `v` for `direction_cosine`; singleton gives a 1-D scan Cost: Rebuilds steering manifold. |
| `diagonal_loading` | double | `-1` (`auto`) | `[live]` Exact | `R + δI`; `< 0` → `0.01 · trace(R̂)/C` Cost: Free (kernel reload of `δ`). |
| `max_sources` | int64 | `1` | `[live]` Exact | Capon peaks to report (`aoa_estimates` is always the strongest) Cost: Reallocates `doa_multi`; cap `64`. |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD | Per-axis sub-array **counts** on a filled ULA/URA Cost: Rebuilds plan + index H2D; warns once. |
| `forward_backward` | bool | `true` | static | Conjugate-reversed looks; ignored unless smoothing is on Cost: Part of the smoothing-plan key. |

## Algorithm

Per item `i`, `run_mvdr_stage` (`mvdr_stage.hpp`) applies the following, all enqueued on the runtime stream.

**1. Sample covariance** (same kernel family as MUSIC):

```text
R̂_i = (1/K_i) Σ_{s valid} x_{i,s} x_{i,s}^H
```

`K_i` is the per-item valid count. Empty items yield a zero matrix and are blanked after the scan.

**2. Eigenvalues of the raw (unloaded) `R̂`.** Batched `matx::eig` (ascending). These are the `eigenvalues` output — diagnostics of the sample covariance, not of `R̂ + δI`.

**3. Diagonal loading** (`load_diagonal_kernel`), in place:

```text
R̂ ← R̂ + δ I
δ = kAutoLoadingFactor · trace(R̂)/C     if loading < 0   (kAutoLoadingFactor = 1e-2)
δ = loading                               if loading ≥ 0
```

If auto `δ` is not `> 0` (all-zero / empty covariance), the kernel uses the absolute floor `kAutoLoadingFactor` so `matx::inv` stays defined; empty items are still NaN/zeroed later.

**4. Hermitian inverse.** Batched `matx::inv(R̂)`.

**5. Capon scan** (`capon_reduce_kernel`). Untapered steering `a(θ)` from `build_steering_weights_device` (`WindowSpec{}`, el-major beams, `conjugate=false`). The cached private tensor is channel-major for float with effective `C >= 8` (including smoothing), with matching scan read stride `num_beams`; smaller float apertures and all double use the original beam-major layout/read stride 1:

```text
a_m(θ) = exp(j · k · (p_m · dir(θ))),   k = 2π / λ
P(θ)   = 1 / (a(θ)^H R̂⁻¹ a(θ))
```

Unlike MUSIC, a non-positive denominator is **gated to zero** (`power = denom > 0 ? 1/denom : 0`), not floored — a singular Capon denominator is not a meaningful peak. Invisible `direction_cosine` beams (zero manifold) therefore write `0`. `aoa_estimates` is the argmax; lowest beam index wins ties.

**6. Multi-peak extract** (`peak_extract_kernel`). Target count is always `max_sources` (Capon has no model order). Pass 0 is a global argmax so `doa_multi[i,0,:]` matches `aoa_estimates`. Passes `k ≥ 1` require a strict 2-D local maximum. Unused rows are NaN.

**Spatial smoothing** is the same S1 front-end as MUSIC (per-axis counts, optional forward-backward, holey reject). First enable warns `MVDREstimatorOp: spatial smoothing on — aperture {}->{} channels, {} sub-arrays[ (forward-backward)]`. Steering is rebuilt against `reduced_geometry`. The first manifold build is silent; a later unseen steering key warns once, then caches.

The tick shell follows `resolve/plan → optional mask → empty path → prepare effective covariance input → MVDR path → emit`. `prepare_smoothed_input` owns smoothing-plan cache updates, index uploads, and gather dispatch; with smoothing off it passes the original geometry, pointers, and dimensions through unchanged.

## Shape and dtype rules

Precision is the template `T` ∈ {`float`, `double`}. Named leading axes stop at `num_items`.

- `snapshot` has exact rank 3. Use a singleton trailing axis for one look.
- `num_channels` must equal `array_geometry.num_elements()` (pre-smoothing). After smoothing, estimation and `eigenvalues` use `C_est = L`.
- `aoa_estimates` `(num_items, 2)`; `angle_spectrum` `(num_items, len(az), len(el))`; `doa_multi` `(num_items, effective_max_sources, 2)`; `source_count` `(num_items,)`.
- `num_items == 0` still emits all five (eigenvalues use the pre-smoothing channel count).
- Mid-stream `max_sources` realloc is legal. A scan-grid / geometry / smoothing override reshapes the spectrum on the tick it arrives; downstream re-infers. An override grid longer than `max_az` / `max_el` is rejected on arrival.

Rejected combinations:

| Condition | Message |
| --- | --- |
| Empty `az_coordinates` or `el_coordinates` | `MVDREstimatorOp: az_coordinates and el_coordinates must be non-empty` |
| `max_sources` not in `[1, 64]` | `MVDREstimatorOp: max_sources must be in [1, 64]` |
| `snapshot` not rank 3 | `MVDREstimatorOp: snapshot must have shape (num_items, num_channels, num_snapshots)` |
| Channel mismatch | `MVDREstimatorOp: snapshot num_channels must match array_geometry.num_elements` |
| `is_valid` dtype / shape | same wording as MUSIC, with this operator name |
| Pageable / non-contiguous / dtype | shared `require_*` strings with `MVDREstimatorOp` |
| Holey / non-uniform smoothing | `spatial_smoothing: array_geometry is holey …` or `… non-uniform along an aperture axis …` |
| Missing stream | `MVDREstimatorOp: no CUDA compute stream` |
| Missing ctor extents (when any sizing field is set) | `MVDREstimatorOp: MaxTensorExtents.<field> must be a positive value …` |

`num_items`, scan grids, channels, and `max_sources` are checked against the captured extents before compute.

## Performance

**Cold.** First unseen geometry / grid / smoothing key builds the steering manifold and (if enabled) the smoothing plan. First `matx::eig` and `matx::inv` create plans.

**Steady.** Covariance, load, inv, and Capon scan run every tick using stable constructor-created workspace and retained MatX plans. The steering artifact caches its manifold and device scan grids. Eigen decomposition and peak extraction drop out after their optional outputs latch unwired. MatX 1.0.0's cached single-matrix cuSolver inverse is not repeatable, so a one-item tick duplicates its loaded covariance internally and inverts a two-matrix batch; public and downstream item counts remain one.

For float instances whose eigenvalue output has latched unwired, covariance and diagonal loading share one kernel while preserving the original accumulation and loading order. Double instances retain the separate kernels.

On Thor, `optional-minimal-medium` (8 channels, 128 items, 32 snapshots, 41 × 21 scan grid) measured 0.193560 → 0.184654 ms: a median paired saving of 10.27 µs / 5.31%, with three screen and three fresh pairs, five warmups and 50 frames. All 41 native tests and established exact output policies across seven cases in both precisions passed. This is a scoped steady-state result, with no double-speedup claim.

Startup and tails varied: fresh cached-smoothing double cold time increased 1.830 ms, medium float cold time increased 0.679 ms, and large double median p99 increased 0.246 ms (worst maximum +1.208 ms). One target pair also had slightly worse tails despite a better median. Fifty-frame p99 is descriptive; these results provide no startup, tail, arbitrary-shape or multi-GPU guarantee.

**`[live]` cost.** `diagonal_loading` is free. `max_sources` reallocates `doa_multi`. A geometry / grid / smoothing override costs a steering-manifold rebuild (plus the smoothing plan and its index H2D) on the tick it arrives.

## Usage

```yaml
mvdr_estimator:
  angle_convention:
    preset: radar_boresight
  az_coordinates: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  el_coordinates: [0.0]
  diagonal_loading: -1.0
  max_sources: 1
  spatial_smoothing: []
  forward_backward: true
```

```cpp
MaxTensorExtents extents;
extents.max_items = 4096;
extents.max_az = 101;
extents.max_el = 51;
extents.max_sources = 4;
extents.max_channels = 16;

auto mvdr = graph.op<MVDREstimatorOp<float>>("mvdr_estimator", extents);
mvdr.set_params(params);  // MVDRParams, including array_geometry

graph.add_flow(gather->snapshot, mvdr->snapshot);
graph.add_flow(mvdr->aoa_estimates, consumer->aoa_estimates);
graph.add_flow(mvdr->angle_spectrum, spectrum_sink->in);
graph.add_flow(mvdr->doa_multi, doa_sink->in);
graph.add_flow(mvdr->source_count, count_sink->in);
graph.add_flow(mvdr->eigenvalues, eig_sink->in);
```
