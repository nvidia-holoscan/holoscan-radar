# `MVDREstimatorOp`

Adaptive MVDR / Capon sparse-path DOA: for each gathered snapshot item it forms a sample covariance, regularizes and inverts it, and scans an `(az, el)` grid for the Capon power peak.

It sits in the pluggable sparse slot after `SnapshotGatherOp`, the same wiring as Beamscan / MUSIC. It solves “one (or a few) DOAs per detection/cluster” with data-adaptive nulls that a conventional scan cannot place. The fact a caller must not miss: `angle_convention` defaults to `radar_boresight`, the scan grid is required, and there is no array taper (the beam is shaped from `R̂`).

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Exact rank 3; no opaque leading batch |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Dominant Capon DOA `(az, el)` or `(u, v)` |
| out | `angle_spectrum` | float32 / float64 | `(num_items, num_az, num_el)` | Capon spectrum. Optional (`always` disposition): the grid is the same pass that yields `aoa_estimates`, so it is produced regardless and an unwired sink is silently discarded |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Up to `max_sources` peaks; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | Peaks actually written into `doa_multi` |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Ascending **unloaded** covariance spectrum |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable |
| live | `az_angles` / `el_angles` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | `[live]` Exact override; extents bounded by `max_az` / `max_el` |
| live | `spatial_smoothing` | `SpatialSmoothingOverride` | `{count, first, second}` int64 POD | `[live]` Exact override; `count` 0–2 selects 1-D / 2-D counts |
| live | `diagonal_loading` | float64 | scalar | `[live]` Exact override; always declared, wire to enable. Negative → `auto` |
| live | `max_sources` | int64 | scalar | `[live]` Exact override; always declared, wire to enable. `doa_multi` capacity |

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → MVDREstimatorOp → aoa_estimates
```

`num_snapshots` should be ≳ `num_channels` for a well-conditioned inverse; a singleton snapshot axis (`K = 1`) still runs because diagonal loading regularizes `R̂`.

Contract: [`OPERATOR_API.md` § `MVDREstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md) ·
open items: [`OPEN_ITEMS.md`](OPEN_ITEMS.md)

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

**5. Capon scan** (`capon_reduce_kernel`). Untapered steering `a(θ)` from `build_steering_weights_device` (`WindowSpec{}`, el-major, beam-major, `conjugate=false`):

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
| Empty `az_angles` or `el_angles` | `MVDREstimatorOp: az_angles and el_angles must be non-empty` |
| `max_sources` not in `[1, 64]` | `MVDREstimatorOp: max_sources must be in [1, 64]` |
| `snapshot` not rank 3 | `MVDREstimatorOp: snapshot must have shape (num_items, num_channels, num_snapshots)` |
| Channel mismatch | `MVDREstimatorOp: snapshot num_channels must match array_geometry.num_elements` |
| `is_valid` dtype / shape | same wording as MUSIC, with this operator name |
| Pageable / non-contiguous / dtype | shared `require_*` strings with `MVDREstimatorOp` |
| Holey / non-uniform smoothing | `spatial_smoothing: array_geometry is holey …` or `… non-uniform along an aperture axis …` |
| Missing stream | `MVDREstimatorOp: no CUDA compute stream` |
| Missing ctor extents (when any sizing field is set) | `MVDREstimatorOp: MaxTensorExtents.<field> must be a positive value …` |

`num_items`, scan grids, channels, and `max_sources` are checked against the captured extents before compute.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override (§9) | Rebuilds steering (+ smoothing plan if enabled) | Element positions + `λ` |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Fixes output meaning for the graph | Boundary `(az, el)` ↔ unit-vector map |
| `az_angles` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_az` (§9) | Rebuilds steering manifold | Azimuth (or `u`) scan grid |
| `el_angles` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_el` (§9) | Rebuilds steering manifold | Elevation (or `v`); singleton → 1-D az scan |
| `diagonal_loading` | double | `-1` (`auto`) | `[live]` Exact | Free (kernel reload of `δ`) | `R + δI`; `< 0` → `0.01 · trace(R̂)/C` |
| `max_sources` | int64 | `1` | `[live]` Exact | Reallocates `doa_multi`; cap `64` | Capon peaks to report (`aoa_estimates` is always the strongest) |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD (§9) | Rebuilds plan + index H2D; warns once | Per-axis sub-array **counts** on a filled ULA/URA |
| `forward_backward` | bool | `true` | static | Part of the smoothing-plan key | Conjugate-reversed looks; ignored unless smoothing is on |

No `num_sources` / EFT — Capon has no model-order concept. `diagonal_loading` encoding is `double`: negative → `auto`, non-negative → absolute `δ` (the app YAML `diagonal_loading: -1.0` is `auto`).

## SDK implementation notes

- **Construction.** `MVDREstimatorOp(const MaxTensorExtents&)`. If both `max_items` and `max_az` are unset, pool spans stay `0`. Otherwise `max_items`, `max_az`, `max_el`, `max_sources`, and `max_channels` must be positive. Constructor-created runtime state owns stable covariance/inverse/count workspace and the steering cache.
- **Triggering.** `contract()` is `OnEach{snapshot}`. Optional `is_valid` and every live override are unconditional non-members with `Exact{}` + `SafeDefault{}`; unwired means absent/all-valid or Params for that tick. A producer must publish a same-tick mask before `snapshot`.
- **Ports and queues.** Device Tensor ports, queue depth `2`. Output ranks: `aoa_estimates` 2, `angle_spectrum` 3, `doa_multi` 3, `source_count` 1, `eigenvalues` 2. Live Inputs also queue depth `2`; `az_angles` / `el_angles` are rank-1 pageable-host float64 config Tensors, the rest are wire-object / POD / scalar payloads.
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>`, a generated descriptor-schema type; `spatial_smoothing` is `Input<SpatialSmoothingOverride>`, a bounded `{count, first, second}` int64 POD (`count` must be 0, 1, or 2). Both convert with `from_wire` inside `resolve_live_override_via`, so neither wire storage nor the POD reaches operator state. `az_angles` / `el_angles` ride rank-1 pageable-host float64 config Tensors validated on arrival (rank, host memory, contiguity, dtype, `max_az` / `max_el` extent). `diagonal_loading` / `max_sources` are plain scalar `Input<>` ports resolved with `resolve_live_override`. Logical Params types are unchanged (`HSDK5_PORT.md` §9).
- **Tick flow.** `compute()` is the high-level shell: receive and plan → validate mask / stream → empty path → prepare smoothed input → prepare steering, workspace, and outputs → execute MVDR → publish and retain. Operator-local helpers own the cache, Tensor-loan, execution-call, and optional-output plumbing. Smoothing gather storage is still allocated per smoothed tick (tracked in `OPEN_ITEMS.md`).
- **Streams and lifetime.** `context.cuda_stream()`. Inputs are retained at end of `compute()`. `MVDRPreparedWorkspace` points into stable covariance/inverse/count buffers; `MVDRPreparedOutputs` keeps every active loan and write guard alive through execution and commits all siblings before required-first publication. The MatX plan clear remains; `stop()` drains `GetAllocMap()` via `retire_matx_solver_caches()`.
- **Emission.** Required `aoa_estimates` publishes first. `angle_spectrum` is optional/always; `doa_multi` and `source_count` share a latched peak pass; `eigenvalues` is independently latched.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.MVDRParams` version `1`.
- **Precision.** `MVDREstimatorOp<float>` and `<double>`.
- **[EA HSDK5 ONLY]** All six override ports are unconditionally in `contract()`; the descriptor / grid / smoothing Graph tests run. Runtime geometry producers connect directly to `array_geometry` with `WireArrayGeometry`; Python remains Params-only for schema payloads.

## Consumer lifetime notes

Every output is a fresh pool loan. A consumer may keep the Tensor for the usual Holoscan lifetime.

## Performance

**Cold.** First unseen geometry / grid / smoothing key builds the steering manifold and (if enabled) the smoothing plan. First `matx::eig` and `matx::inv` create plans.

**Steady.** Covariance, load, inv, and Capon scan run every tick using stable constructor-created workspace. The steering artifact caches its manifold and device scan grids. Eigen decomposition and peak extraction drop out after their optional outputs latch unwired. Per-tick plan clear does not drain the global MatX allocator.

**`[live]` cost.** `diagonal_loading` is free. `max_sources` reallocates `doa_multi`. A geometry / grid / smoothing override costs a steering-manifold rebuild (plus the smoothing plan and its index H2D) on the tick it arrives.

```bash
ctest --test-dir build -R perf_mvdr_estimator_smoke
```

Size ladder axes: channels, detections, snapshots, `num_az × num_el` (`xs`…`xl`; default smoke `xs`/`sm`/`md`). The Graph harness measures the production Exact `diagonal_loading` override. Do not treat historical kernel notes as current Graph results.

## Tests and verification

| Check | Proves |
| --- | --- |
| `RecoversKnownDoaUra` / `RecoversKnownDoaUlaAzOnly` | Single-source `aoa_estimates ≈ θ` |
| `FixedDiagonalLoadingRecoversDoa` / `DiagonalLoadingConfigurableFromParams` | Absolute `δ` vs `auto` |
| `MultiSourceRecoversTwoDoa` / `MaxSourcesCapacityNanFillsUnusedRows` | `doa_multi` always targets `max_sources`; NaN pad |
| `EigenvaluesRankOneSourceDominant` | Unloaded spectrum; one dominant eigenvalue |
| `SpatialSmoothing*` | S1 1-D/2-D; eigenvalues shrink; holey reject |
| `Rejects2DSingleSnapshot` | Locked exact-rank snapshot contract |
| `DirectionCosineMasksInvisibleRegion` | Invisible `(u, v)` → zero Capon power |
| `LiveAzGridAppliesForOneTick` / `LiveSpatialSmoothingAppliesForOneTick` | Mid-stream override applies for one tick; the next miss reverts to Params |
| `LiveArrayGeometryAppliesForOneTick` | Same for `array_geometry`; active and passing |
| `LiveSpatialSmoothingRejectsMalformedPod` / `LiveAzGridRejectsExtentAboveMaxAz` | Out-of-range `count` and an over-`max_az` grid are diagnosed, not consumed |
| `IsValidMask…` / `AllInvalidMask…` | Connected Exact mask and all-invalid blanking |
| `LiveMaxSources…` | Scalar Exact override and revert; loading value comparison remains blocked by the documented MVDR solver bug |
| Rejects / precision / multi-item | TESTING.md cross-cuts |

Analytic ground truth is a host steering-vector snapshot. C++ Graph tests bind all five outputs.

TESTING.md also asks that multi-snapshot MVDR **null an interferer better than Beamscan**. No C++ test in `test_mvdr_estimator.cpp` performs that comparison (coverage gap, not an algorithm fork).

```bash
ctest --test-dir build -R test_mvdr_estimator --output-on-failure
```

## Usage

```yaml
mvdr_estimator:
  angle_convention:
    preset: radar_boresight
  az_angles: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  el_angles: [0.0]
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

**[EA HSDK5 ONLY]** Bind all five device outputs. Python `MVDREstimatorOpF32` / `F64` bindings are deferred on this branch.

## Contract conformance

The HSDK 5 C++ compute path matches the locked Capon algorithm and optional-port convention.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` in `contract()`, not `Input<ArrayGeometry>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9) |
| `az_angles` / `el_angles` **[live]** | conforms (payload form differs) | Rank-1 host config Tensors bounded by `max_az` / `max_el`, not `Input<std::vector<double>>` (§9) |
| `spatial_smoothing` **[live]** | conforms (payload form differs) | Bounded `{count, first, second}` int64 POD, not `Input<std::vector<int64_t>>` (§9) |
| `diagonal_loading` / `max_sources` **[live]** | conforms | Unconditional Exact `Input<>`; production-contract Graph performance and `max_sources` revert coverage |
| Optional `is_valid` connected in Graph | conforms | Non-member Exact + SafeDefault; connected mask tests run |
| Optional `angle_spectrum` / `doa_multi` / `source_count` / `eigenvalues` | conforms | `angle_spectrum` always (fused with `aoa`); the other three latched, and `eigenvalues` sheds its batched eig |
| `max_sources` unbounded except by `doa_multi` capacity | deviates | Rejects values outside `[1, 64]` (`kMaxSourcesCap` in `mvdr_estimator_op.cu`) |

Audit scope (static; no build or test run): `OPERATOR_API.md` § MVDREstimatorOp, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md`, `STATUS.md`, group/common READMEs, `OPEN_ITEMS.md`, `mvdr_estimator_op.hpp` / `.cu`, `mvdr_stage.hpp`, `metadata.json`, C++ tests, `perf_mvdr_estimator.cpp`. Deviations that are team questions are proposed in chat, not appended to `OPEN_ITEMS.md`.
