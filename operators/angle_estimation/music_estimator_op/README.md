# `MUSICEstimatorOp`

Subspace (MUSIC) sparse-path DOA: for each gathered snapshot item it forms a sample covariance, splits signal from noise, and scans an `(az, el)` grid for the MUSIC pseudospectrum peak.

It sits in the pluggable sparse slot after `SnapshotGatherOp` (SIMO physical RX or MIMO virtual channels). It solves the same “one DOA per detection/cluster” problem as Beamscan, but resolves closely spaced sources that a conventional scan cannot. The fact a caller must not miss: `angle_convention` defaults to `radar_boresight` and the scan grid is required.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Exact rank 3; no opaque leading batch |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Dominant DOA `(az, el)` or `(u, v)` |
| out | `angle_spectrum` | float32 / float64 | `(num_items, num_az, num_el)` | MUSIC pseudospectrum; always emitted |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Up to `min(d_i, max_sources)` peaks; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | Peaks actually written into `doa_multi` |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Ascending covariance spectrum; `C_est` is post-smoothing |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable |
| live | `az_coordinates` / `el_coordinates` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | `[live]` Exact override; extents bounded by `max_az` / `max_el` |
| live | `spatial_smoothing` | `SpatialSmoothingOverride` | `{count, first, second}` int64 POD | `[live]` Exact override; `count` 0–2 selects 1-D / 2-D counts |
| live | `num_sources` | int64 | scalar | `[live]` Exact override; always declared, wire to enable. Negative → per-item EFT `auto` |
| live | `source_count_pfa` | float64 | scalar | `[live]` Exact override; always declared, wire to enable. EFT target Pfa for `auto` |
| live | `max_sources` | int64 | scalar | `[live]` Exact override; always declared, wire to enable. `doa_multi` capacity and EFT `K_max` |

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → MUSICEstimatorOp → aoa_estimates
```

Contract: [`OPERATOR_API.md` § `MUSICEstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md) ·
open items: [`OPEN_ITEMS.md`](OPEN_ITEMS.md)

## Algorithm

Per item `i`, `run_music_stage` (`music_stage.hpp`) applies the following, all enqueued on the runtime stream.

**1. Sample covariance** (`music_covariance_kernel`). Over valid snapshots (all `K` slots, or those with `is_valid != 0`):

```text
R_i = (1/K_i) Σ_{s valid} x_{i,s} x_{i,s}^H
```

`K_i` is written to `k_counts[i]`. An empty item (`K_i == 0`) yields a zero matrix and is blanked later. Layout of `x` is `[num_items, num_channels, num_snapshots]` with the snapshot index innermost.

**2. Hermitian eigendecomposition.** Batched `matx::eig(R_i, EigenMode::VECTOR)`: eigenvalues **ascending**, eigenvectors as **columns**. Those eigenvalues are the `eigenvalues` output.

**3. Model order `d_i`.** `num_sources >= 0` broadcasts that integer to every item (`fill_model_order_kernel`). `num_sources < 0` (`auto`) runs the shared device EFT (`run_source_count_device`) on the ascending spectrum and per-item `K_i`, clamped to `[1, max_sources]`. Rank-deficient items (`K_i < C_est`) fall back to `d_i = 1` inside that kernel. A uniform host-visible deficiency (`auto`, no `is_valid`, `num_snapshots < C_est`) also logs once:

```text
MUSICEstimatorOp: num_sources=auto with num_snapshots ({}) < num_channels ({}) - covariance is rank-deficient; falling back to num_sources=1 for all items
```

EFT uses Quinlan et al. 2007’s closed-form noise-eigenvalue profile for prediction and a first-order Gaussian threshold rather than the paper’s Monte-Carlo `η_P`. `source_count_pfa` is therefore a **target**, not a guaranteed false-alarm rate (see `radar_common/dsp/source_count.hpp` and the [shared angle-estimation open items](../OPEN_ITEMS.md)). Seeding from the CFAR/cluster detection count is rejected (STATUS.md S2).

**4. Noise-subspace projector** (`projector_kernel`):

```text
P_n,i = Σ_{j=0}^{C_est - d_i - 1} e_j e_j^H
```

using the `C_est - d_i` smallest-eigenvalue eigenvectors.

**5. MUSIC scan** (`music_reduce_kernel`). Untapered steering `a(θ)` is built by `build_steering_weights_device` with `WindowSpec{}`, `BeamOrder::kElMajor` (`beam = ie * num_az + ia`), `SteeringLayout::kBeamMajor`, `conjugate=false`. The kernel computes

```text
a_m(θ) = exp(j · k · (p_m · dir(θ))),   k = 2π / λ
P(θ)   = 1 / (a(θ)^H P_n a(θ))
```

`dir(θ)` is the unit vector after `AngleConvention` boundary relabeling; the kernel itself only dots positions with that vector. Invisible `direction_cosine` beams (`u²+v² > 1`) carry an all-zero manifold entry; those cells are written as `0` (guarded **before** the denominator floor so `0/ε` cannot become a spurious peak). A near-zero denominator is a genuine MUSIC peak and is **floored**, not gated:

```text
kMusicDenomFloor<T> = 1e-6
power = 1 / (denom > floor ? denom : floor)
```

`aoa_estimates` is the argmax over the grid; lowest beam index wins ties. An all-non-positive spectrum writes NaN.

**6. Multi-peak extract** (`music_peak_extract_kernel`). Target count is `min(d_i, max_sources)` (floored at 1). Pass 0 is a global argmax so `doa_multi[i,0,:]` matches `aoa_estimates`. Passes `k >= 1` require a strict 2-D local maximum (4-neighbor, `>=` rejects plateaus) so one lobe is not double-counted. Unused rows are NaN. `source_count[i]` is the number actually found. Empty items are then forced to a zero spectrum, NaN DOAs, and `source_count = 0` (`music_blank_empty_items_kernel`).

**Spatial smoothing** (optional front-end, shared `radar_common/dsp/spatial_smoothing`). A nonempty `spatial_smoothing` vector is per-aperture-axis **counts** `P` on a filled ULA/URA: `[P]` or `[P_az, P_el]`. Sub-array length is `L_axis = M_axis − P_axis + 1`. The gather expands looks to `est_snapshots = looks_per_snapshot() * num_snapshots` (`P` or `2P` with `forward_backward`) on the reduced aperture `C_est = L`. Steering is rebuilt against `reduced_geometry`. Holey / non-uniform geometry is rejected (reduce upstream with `SubarraySelectionOp`). First enable warns:

```text
MUSICEstimatorOp: spatial smoothing on — aperture {}->{} channels, {} sub-arrays[ (forward-backward)]
```

`forward_backward` applies only when smoothing is on. Empty `spatial_smoothing` is off.

The tick shell follows `resolve/plan → optional mask → empty path → prepare effective covariance input → MUSIC path → emit`. `prepare_smoothed_input` owns smoothing-plan cache updates, index uploads, and gather dispatch; with smoothing off it passes the original geometry, pointers, and dimensions through unchanged.

## Shape and dtype rules

Precision is the template argument `T` ∈ {`float`, `double`}: inputs are `complex<T>`, real outputs are `T`, `source_count` is int32. No runtime dtype flag.

- `snapshot` has exact rank 3. Use a singleton trailing axis for one look.
- `num_channels` must equal `array_geometry.num_elements()` (pre-smoothing). After smoothing, estimation and `eigenvalues` use `C_est = L`.
- `aoa_estimates` is always `(num_items, 2)`.
- `angle_spectrum` is `(num_items, len(az_coordinates), len(el_coordinates))`.
- `doa_multi` is `(num_items, effective_max_sources, 2)`.
- `source_count` is `(num_items,)`.
- Named leading axes stop at `num_items`. Extra leading batch dims are not admitted.
- `num_items == 0` still emits all five outputs at the shapes above (eigenvalues use the pre-smoothing channel count).
- Mid-stream shape changes from a `[live]` `max_sources` realloc are legal; downstream re-infers. A scan-grid / geometry / smoothing override reshapes the spectrum on the tick it arrives; an override grid longer than `max_az` / `max_el` is rejected on arrival.

Rejected combinations (operator’s own message unless noted):

| Condition | Message |
| --- | --- |
| Empty `az_coordinates` or `el_coordinates` | `MUSICEstimatorOp: az_coordinates and el_coordinates must be non-empty` |
| `max_sources` not in `[1, 64]` | `MUSICEstimatorOp: max_sources must be in [1, 64]` |
| `source_count_pfa` not in `(0, 1)` | `MUSICEstimatorOp: source_count_pfa must be in (0, 1)` |
| `snapshot` not rank 3 | `MUSICEstimatorOp: snapshot must have shape (num_items, num_channels, num_snapshots)` |
| Channel mismatch | `MUSICEstimatorOp: snapshot num_channels must match array_geometry.num_elements` |
| Fixed `num_sources > C_est - 1` | `MUSICEstimatorOp: num_sources must leave at least one noise-subspace dimension (num_sources <= num_channels - 1, using the post-smoothing channel count)` |
| `is_valid` not bool/uint8 | `MUSICEstimatorOp: is_valid must have dtype bool or uint8` |
| `is_valid` shape mismatch | `MUSICEstimatorOp: is_valid must have shape (num_items, num_snapshots) matching snapshot` |
| Pageable-host `snapshot` | `MUSICEstimatorOp: tensor received on input port 'snapshot' is not CUDA-accessible (device, managed, or pinned-host memory required)` |
| Non-contiguous `snapshot` | `MUSICEstimatorOp: tensor received on input port 'snapshot' must be contiguous (compact row-major)` |
| Dtype mismatch | `MUSICEstimatorOp: tensor received on input port 'snapshot' has dtype …, expected …` |
| Holey / non-uniform smoothing | `spatial_smoothing: array_geometry is holey …` or `… non-uniform along an aperture axis …` (reduce via S3) |
| Missing compute stream | `MUSICEstimatorOp: no CUDA compute stream` (`kNotSupported`) |
| Missing ctor extents (when any sizing field is set) | `MUSICEstimatorOp: MaxTensorExtents.<field> must be a positive value (required for Tensor pool sizing)` |

`num_items`, scan grids, channels, and `max_sources` are checked against the captured extents before compute.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override (§9) | Rebuilds steering (+ smoothing plan if enabled); warn-once on smoothing enable | Element positions + `λ`; `num_elements` must match `snapshot` channels |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Fixes output meaning for the graph | Boundary `(az, el)` ↔ unit-vector map; `direction_cosine` is `(u, v)` |
| `az_coordinates` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_az` (§9) | Rebuilds steering manifold | First coordinate: azimuth radians for angular conventions, dimensionless `u` for `direction_cosine` |
| `el_coordinates` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_el` (§9) | Rebuilds steering manifold | Second coordinate: elevation radians for angular conventions, dimensionless `v` for `direction_cosine`; singleton gives a 1-D scan |
| `num_sources` | int64 | `-1` (`auto`) | `[live]` Exact | Free (device EFT or a fill); no FFT | Signal-subspace dimension; `< 0` → per-item EFT |
| `source_count_pfa` | double | `1e-2` | `[live]` Exact | Free (EFT threshold only) | Target EFT over-estimation probability; used only for `auto` |
| `max_sources` | int64 | `4` | `[live]` Exact | Reallocates `doa_multi`; implementation cap `64` | `doa_multi` / EFT `K_max`; MUSIC targets `min(d_i, max_sources)` peaks |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD (§9) | Rebuilds plan + index H2D; warns once | Per-axis sub-array **counts** `[P]` / `[P_az, P_el]` on a filled ULA/URA |
| `forward_backward` | bool | `true` | static | Part of the smoothing-plan key | Conjugate-reversed looks; ignored unless smoothing is on |

Application YAML accepts explicit arrays or inclusive `{min: ..., max: ..., step: ...}` mappings for `az_coordinates` and `el_coordinates`. Range mappings require finite values, `step > 0`, `min <= max`, and endpoint alignment within floating-point tolerance. They expand to the same vector Params before construction; live ports remain rank-1 float64 Tensors. Every effective grid coordinate must be finite. Under `direction_cosine`, each authored axis value must lie in `[-1, 1]`; square-grid corners outside `u² + v² <= 1` remain legal and are masked per beam.

`num_sources` encoding is `int64_t`: negative → `auto`, non-negative → fixed order. There is no YAML `auto` token. `array_geometry.validate()` runs every tick; `geometry_type` is informational.

## SDK implementation notes

- **Construction.** `MUSICEstimatorOp(const MaxTensorExtents&)`. If both `max_items` and `max_az` are unset, pool spans stay `0`. Otherwise all of `max_items`, `max_az`, `max_el`, `max_sources`, and `max_channels` must be positive. Constructor-created runtime state owns stable covariance/eigenvector/projector/count workspace and the steering cache.
- **Triggering.** `contract()` is `OnEach{snapshot}`. Optional `is_valid` and every live override are unconditional non-members with `Exact{}` + `SafeDefault{}`; unwired means absent/all-valid or Params for that tick. A producer must publish a same-tick mask before `snapshot`.
- **Ports and queues.** The data Tensor ports declare `MemoryKind::kCudaDevice`, queue depth `2`. Dtype/rank are enforced in `compute()`. Pooled output ranks: `aoa_estimates` 2, `angle_spectrum` 3, `doa_multi` 3, `source_count` 1, `eigenvalues` 2. Live Inputs also queue depth `2`; `az_coordinates` / `el_coordinates` are rank-1 pageable-host float64 config Tensors, the rest are wire-object / POD / scalar payloads.
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>`, a generated descriptor-schema type; `spatial_smoothing` is `Input<SpatialSmoothingOverride>`, a bounded `{count, first, second}` int64 POD (`count` must be 0, 1, or 2). Both convert with `from_wire` inside `resolve_live_override_via`, so neither wire storage nor the POD reaches operator state. `az_coordinates` / `el_coordinates` ride rank-1 pageable-host float64 config Tensors validated on arrival (rank, host memory, contiguity, dtype, `max_az` / `max_el` extent). `num_sources` / `source_count_pfa` / `max_sources` are plain scalar `Input<>` ports resolved with `resolve_live_override`. Logical `MUSICParams` types are unchanged (`HSDK5_PORT.md` §9).
- **Tick flow.** `compute()` is the high-level shell: receive and plan → validate mask / stream → empty path → prepare smoothed input → validate effective aperture → prepare steering, workspace, and outputs → execute MUSIC → publish and retain. Operator-local helpers own the cache, Tensor-loan, execution-call, and optional-output plumbing.
- **Streams and lifetime.** Work uses `context.cuda_stream()` only. Inputs and smoothing gather buffers remain alive through estimator execution. `MUSICPreparedWorkspace` points into stable eigenspace/count buffers; `MUSICPreparedOutputs` keeps every loan and write guard alive through execution and commits all siblings before required-first publication. The MatX plan clear remains; `stop()` drains the MatX `MemTracker`.
- **Emission.** Required `aoa_estimates` publishes first. `angle_spectrum` and `eigenvalues` are optional/always; `doa_multi` and `source_count` share a latched peak pass.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.MUSICParams` version `1`. YAML / `from_config` / `set_params` field names match the table above.
- **Precision.** Explicit instantiations `MUSICEstimatorOp<float>` and `MUSICEstimatorOp<double>`.
- **[EA HSDK5 ONLY]** All seven override ports join the contract unconditionally; descriptor / grid / smoothing Graph tests run. Runtime geometry producers connect directly to `array_geometry` with `WireArrayGeometry`; Python remains Params-only for schema payloads.

## Consumer lifetime notes

Every output is a fresh device Tensor from the operator’s output pool (loan → fill → commit). A consumer may keep the buffer for the usual Holoscan Tensor lifetime; nothing is an in-place ping-pong workspace.

## Performance

**Cold.** First tick (or first unseen geometry / grid / smoothing key) builds one steering artifact containing the untapered manifold and device scan grids and, if enabled, the spatial-smoothing plan plus index / FB-flip H2D. First `matx::eig` also creates a plan.

**Steady.** Covariance, eig, EFT, projector, and scan run every tick on stable constructor-created workspace. Device scan grids are reused with the cached manifold; the peak pass drops out once both associated optional outputs latch unwired.

**`[live]` cost.** Scalar overrides are cheap. `max_sources` reallocates `doa_multi`. A geometry / grid / smoothing override costs a steering-manifold rebuild (plus the smoothing plan and its index H2D) on the tick it arrives. The Graph harness isolates all seven live inputs. The reduce kernel stages `P_n` in shared memory only when it fits a 40 KiB budget **and** the item grid covers at least one occupancy wave; otherwise it reads `P_n` from global memory.

Harness (Graph, isolated production-Exact timing for every live input):

```bash
ctest --test-dir build -R perf_music_estimator_smoke
```

Size ladder axes: channels, detections (`num_items`), snapshots, and `num_az × num_el` (`xs`…`xl`; default smoke is `xs`/`sm`/`md`). Each size runs `array_geometry`, both scan grids, `spatial_smoothing`, `num_sources`, `source_count_pfa`, and `max_sources` scenarios plus a separate fixed model-order sweep at 1, 2, and `min(4, num_channels − 1)`. The Pfa scenario uses automatic source count; other live-override steady paths fix one source. Recorded HSDK 4.x `ncu` notes in the group README / `OPEN_ITEMS.md` are historical kernel work, not this Graph harness.

## Tests and verification

| Check | Proves |
| --- | --- |
| `RecoversKnownDoaUlaAzOnlyFixedOneSource` / `…Ura…` | Single-source `aoa_estimates ≈ θ` on ULA (az) and URA |
| `ResolvesCloselySpacedSourcesBeamscanCannot` / `AutoNumSourcesResolvesTwo…` | TESTING.md super-resolution vs Beamscan; fixed and EFT `auto` |
| `SharedSparseSlotInterchangesWithBeamscan` | Same `snapshot` → `aoa_estimates` slot |
| `MultiSourceRecoversTwoDoa` / `MaxSourcesCapsBelowNumSourcesNanFillsRest` | `doa_multi` / `source_count` target `min(d, max_sources)`; NaN pad |
| `SpatialSmoothingResolvesCoherentSourcesUla` / `…2DUra…` / `…ForwardOnly…` / `…RejectsNonUniform…` | S1 front-end; eigenvalues shrink with `L`; holey reject |
| `EigenvaluesAscendingAndNonNegative` | `matx::eig` convention on the output port |
| `RankDeficientAutoFallsBackToOneSource` | `K < C` → `d = 1` |
| `IsValidMask…` / `AllInvalidMask…` | Connected Exact mask and all-invalid blanking |
| `LiveAzGridAppliesForOneTick` / `LiveElGridAppliesForOneTick` / `LiveArrayGeometryAppliesForOneTick` / `LiveSpatialSmoothingAppliesForOneTick` | Mid-stream override applies for one tick; the next miss reverts to Params |
| `LiveNumSources…` / `LiveSourceCountPfa…` / `LiveMaxSources…` | Scalar Exact override and revert |
| `NumSourcesConfigurableFromParams` (and Pfa / `max_sources`) | Params authoring of the three scalar lives |
| `Rejects2DSingleSnapshot` / `EmptyItems…` / `MultipleItems…` | Exact rank, empty emit, independent items |
| `DirectionCosineMasksInvisibleRegion` | Invisible `(u, v)` → zero spectrum / no bogus peak |
| Rejects (rank, dtype, pageable, channels, empty grid, `num_sources`, `max_sources`, Pfa) | Error strings above |
| `FloatInstantiationSmoke` / `FloatDoubleAgreeOnDoa` / `DeterministicAcrossRuns` | TESTING.md precision + determinism |

Analytic ground truth is a host steering-vector snapshot (`steering_vector` / two-source block in the test file), not `point_target_sim`. DOA match tolerances in the closely-spaced test are `0.015` rad on a 16-element ULA. C++ Graph tests always bind all five outputs.

Python behavioral tests exist under `tests/python/angle_estimation/test_music_estimator.py` but are deferred with the bindings.

```bash
ctest --test-dir build -R test_music_estimator --output-on-failure
```

## Usage

`array_geometry` is usually built by the caller and applied with `set_params`. Other fields match `MUSICParams` / YAML `from_config`.

```yaml
music_estimator:
  angle_convention:
    preset: radar_boresight
  az_coordinates: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  el_coordinates: [0.0]
  num_sources: -1
  source_count_pfa: 0.01
  max_sources: 4
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

auto music = graph.op<MUSICEstimatorOp<float>>("music_estimator", extents);
music.set_params(params);  // MUSICParams; or from_config("music_estimator")

graph.add_flow(gather->snapshot, music->snapshot);
graph.add_flow(music->aoa_estimates, consumer->aoa_estimates);
graph.add_flow(music->angle_spectrum, spectrum_sink->in);
graph.add_flow(music->doa_multi, doa_sink->in);
graph.add_flow(music->source_count, count_sink->in);
graph.add_flow(music->eigenvalues, eig_sink->in);
```

**[EA HSDK5 ONLY]** Bind every device output (`bind_tensor_output_device` / compile options). Every optional/live input is in the contract whether or not it is wired. Python `MUSICEstimatorOpF32` / `F64` bindings are deferred on this branch.

## Contract conformance

The HSDK 5 C++ compute path matches the locked algorithm contract and optional-port convention. No undocumented algorithm gap.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` in `contract()`, not `Input<ArrayGeometry>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9) |
| `az_coordinates` / `el_coordinates` **[live]** | conforms (payload form differs) | Rank-1 host config Tensors bounded by `max_az` / `max_el`, not `Input<std::vector<double>>` (§9) |
| `spatial_smoothing` **[live]** | conforms (payload form differs) | Bounded `{count, first, second}` int64 POD, not `Input<std::vector<int64_t>>` (§9) |
| `num_sources` / `source_count_pfa` / `max_sources` **[live]** | conforms | Unconditional Exact inputs with active mid-stream/revert tests |
| Optional `is_valid` connected in Graph | conforms | Non-member Exact + SafeDefault; connected mask tests run |
| Optional `angle_spectrum` / `doa_multi` / `source_count` / `eigenvalues` | conforms | Always vs. latched dispositions match the locked convention |
| `max_sources` unbounded except by `doa_multi` capacity | deviates | Implementation rejects values outside `[1, 64]` (`music_estimator_op.cu` `kMaxSourcesCap`, used for peak-extract shared memory) |

Audit scope (static; no build or test run): `docs/planning/OPERATOR_API.md` § Sparse angle estimators / `MUSICEstimatorOp`, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md`, `STATUS.md`, `operators/common/README.md`, `operators/angle_estimation/README.md`, `OPEN_ITEMS.md`, `music_estimator_op.hpp` / `.cu`, `music_stage.hpp`, `metadata.json`, Python bindings, `tests/cpp/angle_estimation/test_music_estimator.cpp`, `tests/python/angle_estimation/test_music_estimator.py`, `performance/cpp/perf_music_estimator.cpp`, plus `source_count.hpp`, `spatial_smoothing.hpp`, and `steering_weights_device.cu` for quoted math. Deviations that are team questions are proposed in chat, not appended to `OPEN_ITEMS.md`.
