# `RootMUSICEstimatorOp`

Gridless Root-MUSIC sparse-path DOA: the same covariance / eig / EFT front-end as `MUSICEstimatorOp`, then polynomial roots instead of a scan grid.

It occupies the same pluggable sparse slot after `SnapshotGatherOp`. A ULA (or single row/column) resolves one coordinate (azimuth; elevation degenerate); a filled URA resolves native `(az, el)` by reduced-dimension per-axis rooting plus 2-D MUSIC-metric pairing. The fact a caller must not miss: there is **no** `angle_spectrum` and **no** scan-grid parameter; `array_geometry` must be a ULA or a filled URA (`(iz` outer, `iy` inner)), and `angle_convention` defaults to `radar_boresight` and is locked after the first successful tick.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Rank-2 rejected |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Dominant DOA; always emitted |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Selected roots / pairs; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | Roots / pairs actually written |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Ascending full-aperture covariance spectrum |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable |
| live | `spatial_smoothing` | `SpatialSmoothingOverride` | `{count, first, second}` int64 POD | `[live]` Exact override; `count` 0–2 selects 1-D / 2-D counts |
| live | `num_sources` | int64 | scalar | `[live]` Exact override; always declared, wire to enable. Negative → per-item EFT `auto` |
| live | `source_count_pfa` | float64 | scalar | `[live]` Exact override; always declared, wire to enable. EFT target Pfa for `auto` |
| live | `max_sources` | int64 | scalar | `[live]` Exact override; always declared, wire to enable. `doa_multi` capacity and EFT `K_max` |

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → RootMUSICEstimatorOp → aoa_estimates
```

A holey MIMO virtual array needs `SubarraySelectionOp` (or spatial smoothing on an already-filled ULA/URA) before this slot.

Contract: [`OPERATOR_API.md` § `RootMUSICEstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md) ·
open items: [`OPEN_ITEMS.md`](OPEN_ITEMS.md)

## Algorithm

Covariance, Hermitian `matx::eig` (VECTOR, ascending), fixed-or-EFT model order, and the noise-subspace projector `P_n` are the MUSIC front-end (`music_covariance_kernel` / `projector_kernel` via `run_root_music_stage` / `run_root_music_2d_stage` in `root_music_stage.hpp`). Geometry is auto-detected with `analyze_planar_aperture`: a filled URA (`ny > 1 && nz > 1`, `make_ura_geometry` order `c = iz * ny + iy`) takes the 2-D path; a line array takes the 1-D cone path.

### 1-D ULA

Polynomial coefficients are the anti-diagonals (lags) of `P_n` (`root_music_coefficients_kernel`):

```text
degree = 2 · (C_est − 1)
c[ℓ + (C−1)] = Σ_row P_n[row, row+ℓ]     for ℓ ∈ [1−C, C−1]
```

The companion matrix of `c` (`root_music_companion_kernel`, last column `−c[row]/c[degree]`) is solved per item with `cusolverDnXgeev` (`CUSOLVER_EIG_MODE_NOVECTOR`). Empty items leave a zero companion; a vanishing leading coefficient writes a NaN matrix so `geev` non-convergence (`info > 0`) blanks that item’s roots rather than aborting the tick. `info < 0` throws:

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

There is no canonical single-polynomial 2-D Root-MUSIC; this reduced-dimension + pairing method is the documented `OPERATOR_API.md` design. A joint 2-D polynomial and explicit non-ULA-per-axis subarray specs are deferred.

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

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override (§9) | Revalidates ULA/URA; rebuilds smoothing plan; may resize cuSOLVER workspace | ULA or filled URA + `λ`; `num_elements` must match channels |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static (rejected if changed after first emit) | Seeds device relabeling once | Boundary meaning of `(az, el)` / `(u, v)` |
| `num_sources` | int64 | `-1` (`auto`) | `[live]` Exact | Free (EFT or fill) | Signal-subspace dimension / root count |
| `source_count_pfa` | double | `1e-2` | `[live]` Exact | Free | Target EFT Pfa; `auto` only. Same EFT fidelity note as MUSIC |
| `max_sources` | int64 | `4` | `[live]` Exact | Reallocates `doa_multi`; cap `64` | Capacity and EFT `K_max` |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD (§9) | Rebuilds plan + H2D; changes polynomial degree | Per-axis counts on a filled ULA/URA |
| `forward_backward` | bool | `true` | static | Part of the smoothing-plan key | FB looks; ignored unless smoothing is on |

No `az_angles` / `el_angles`. `num_sources` encoding is the same `int64` negative-`auto` convention as MUSIC.

## SDK implementation notes

- **Construction.** `RootMUSICEstimatorOp(const MaxTensorExtents&)`. If both `max_items` and `max_sources` are unset, pool spans stay `0`. Otherwise `max_items`, `max_sources`, and `max_channels` must be positive. Constructor-created runtime state owns stable covariance/eigenspace and ULA polynomial workspace plus both cuSOLVER states.
- **Triggering.** `contract()` is `OnEach{snapshot}`. Optional `is_valid` and every live override are unconditional non-members with `Exact{}` + `SafeDefault{}`; unwired means absent/all-valid or Params for that tick. A producer must publish a same-tick mask before `snapshot`.
- **Ports and queues.** Device Tensor ports, queue depth `2`. All four outputs declare `max_rank = 3` (the fat-cube bound; actual ranks are 1–3). Live Inputs also queue depth `2` — wire-object, POD, and scalar payloads only (no config Tensor, since there is no scan grid).
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>`, a generated descriptor-schema type; `spatial_smoothing` is `Input<SpatialSmoothingOverride>`, a bounded `{count, first, second}` int64 POD (`count` must be 0, 1, or 2). Both convert with `from_wire` inside `resolve_live_override_via`, so neither wire storage nor the POD reaches operator state. `num_sources` / `source_count_pfa` / `max_sources` are plain scalar `Input<>` ports resolved with `resolve_live_override`. Logical `RootMUSICParams` types are unchanged (`HSDK5_PORT.md` §9).
- **Tick flow.** `compute()` resolves overrides and builds the plan, validates the optional mask, handles the empty path, asks `prepare_smoothed_input` for the effective covariance input, runs the ULA or URA estimator path, and emits. Smoothing gather and remaining URA transient storage are still allocated per applicable tick (tracked in `OPEN_ITEMS.md`).
- **Streams and lifetime.** Work uses `context.cuda_stream()`. Inputs are retained at the end of `compute()`. `solve_companion_roots` synchronizes to read per-item `geev` info codes. `stop()` calls `retire_music_matx_caches()`; cuSOLVER workspaces die with the runtime state.
- **Emission.** Required `aoa_estimates` publishes first; optional/always `doa_multi`, `source_count`, and `eigenvalues` absorb declined emits.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.RootMUSICParams` version `1`.
- **Precision.** `RootMUSICEstimatorOp<float>` and `<double>`.
- **[EA HSDK5 ONLY]** All five override ports are unconditionally in `contract()`; geometry / smoothing Graph tests run. Runtime geometry producers connect directly to `array_geometry` with `WireArrayGeometry`; Python remains Params-only for schema payloads.

## Consumer lifetime notes

Every output is a fresh pool loan. A consumer may keep the Tensor for the usual Holoscan lifetime.

## Performance

**Cold.** The constructor creates both cuSOLVER states and bounded core workspace. First use initializes solver internals, an optional smoothing plan, and the first `matx::eig` plan. A new geometry key revalidates the aperture.

**Steady.** Covariance + eig + projector every tick; then one serial `cusolverDnXgeev` per item per rooted polynomial (ULA: one of degree `2(C−1)`; URA: two smaller companions plus pairing). The serial hybrid dominates latency; a batched device root solver is postponed, not a contract gap. Per-tick MatX eig-plan clear matches MUSIC (`cudaDeviceSynchronize` inside `clear_music_matx_caches`).

**`[live]` cost.** Scalar overrides are cheap. `max_sources` reallocates `doa_multi`. A geometry / smoothing override revalidates the aperture and rebuilds the smoothing plan (and can resize the cuSOLVER workspace) on the tick it arrives.

```bash
ctest --test-dir build -R perf_root_music_estimator_smoke
```

Harness axes: channels, detections, snapshots (`xs`…`xl` ULA); `--full` also adds URA rungs (`ura_sm` 6×6, `ura_lg` 8×8). `num_sources` is fixed at 1. The harness measures production Exact `max_sources` timing.

## Tests and verification

| Check | Proves |
| --- | --- |
| `RecoversKnownDoaUlaFixedOneSource` / `RecoversTwoSources` / `ResolvesCloselySpacedSources` | Gridless ULA DOA ≈ truth; two-source + close spacing (TESTING.md) |
| `RecoversUraSingleSourceAzEl` / `…TwoSourcesAzElPairing` / `…Float` | Native `(az, el)`; pairing with sources distinct in both coordinates |
| `RejectsNonUniformLinearGeometry` / `RejectsUlaParallelToBoresight` | Geometry constraint; predictable fail |
| `SpatialSmoothing1DRecoversDoa` | S1 on a ULA; eigenvalues shrink with `L` |
| `DirectionCosineConventionMapsUlaProjection` / `CustomConvention*` / `…Zenith…` / `DirectionCosineNearEndfirePreservesSign` | Convention-free kernel + boundary relabel; endfire sign |
| `MaxSourcesCapsReportedRoots` / `UnusedDoaRowsAreNan` | `doa_multi` capacity + NaN fill |
| `LiveArrayGeometryAppliesForOneTick` / `LiveSpatialSmoothingAppliesForOneTick` | Mid-stream override applies for one tick; the next miss reverts to Params |
| `IsValidMask…` / `AllInvalidMask…` | Connected Exact mask and all-invalid blanking |
| `LiveNumSources…` / `LiveSourceCountPfa…` / `LiveMaxSources…` | Scalar Exact override and revert |
| `RejectsSnapshotRank2` | Contract 3-D `snapshot` |
| Memory-space / precision / determinism / rejects | TESTING.md cross-cuts |

Analytic ground truth is a host steering-vector snapshot in the test file. C++ Graph tests bind all four outputs.

```bash
ctest --test-dir build -R test_root_music_estimator --output-on-failure
```

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

**[EA HSDK5 ONLY]** Bind all four device outputs. Python `RootMUSICEstimatorOpF32` / `F64` bindings are deferred on this branch.

## Contract conformance

The HSDK 5 C++ compute path matches the locked gridless ULA/URA algorithm and optional-port convention.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` in `contract()`, not `Input<ArrayGeometry>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9) |
| `spatial_smoothing` **[live]** | conforms (payload form differs) | Bounded `{count, first, second}` int64 POD, not `Input<std::vector<int64_t>>` (§9) |
| `num_sources` / `source_count_pfa` / `max_sources` **[live]** | conforms | Unconditional Exact inputs with active mid-stream/revert tests |
| Optional `is_valid` connected in Graph | conforms | Non-member Exact + SafeDefault; connected mask tests run |
| Optional `doa_multi` / `source_count` / `eigenvalues` | conforms | All `always`: fused with `aoa_estimates`, so unwired sinks are discarded. No `angle_spectrum` |
| `max_sources` unbounded except by `doa_multi` capacity | deviates | Rejects values outside `[1, 64]` (`kMaxSourcesCap` in `root_music_estimator_op.cu`) |
| URA of any filled size | deviates | Pairing kernel caps `ny·nz` at 64 (`kRootMusicPairChannelCap`, `root_music_stage.hpp`) and per-axis candidates at 8 (`kRootMusicPairCandCap`) |

Audit scope (static; no build or test run): `OPERATOR_API.md` § RootMUSICEstimatorOp, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md`, `STATUS.md`, `operators/common/README.md`, `operators/angle_estimation/README.md`, `OPEN_ITEMS.md`, `root_music_estimator_op.hpp` / `.cu`, `root_music_stage.hpp`, `metadata.json`, Python bindings, C++/Python tests, `perf_root_music_estimator.cpp`, `planar_aperture.hpp`. Deviations that are team questions are proposed in chat, not appended to `OPEN_ITEMS.md`.
