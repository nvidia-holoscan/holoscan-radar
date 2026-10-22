# `MUSICEstimatorOp`

## Purpose

Subspace (MUSIC) sparse-path DOA: for each gathered snapshot item it forms a sample covariance, splits signal from noise, and scans an `(az, el)` grid for the MUSIC pseudospectrum peak. It fills the sparse-estimator slot after snapshot gathering and can resolve multiple sources while also emitting the scan spectrum.

## Responsibilities

- Forms covariance matrices, separates signal/noise subspaces, and evaluates the configured scan grid.
- Owns source-count selection, optional spatial smoothing, peak extraction, and validity/model-order outputs.
- Does not gather snapshots, cluster detections, or infer a scan grid from array geometry.

## When to use

Use after `SnapshotGatherOp` when closely spaced sources must be resolved and a pseudospectrum or multiple DOAs is required. It is appropriate when enough valid snapshots exist to estimate covariance and the array is directly supported or has been made suitable by subarray selection/smoothing. Prefer Beamscan for a cheaper dominant-direction estimate; do not use MUSIC when no scan grid can be supplied.

```text
… → rd_maps → CFARDetectorOp
  → SnapshotGatherOp → snapshot → MUSICEstimatorOp → aoa_estimates
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Required trigger/aligned input | Exact rank 3; no opaque leading batch |
| in | `is_valid` | bool or uint8 (`0`/`1`) | `(num_items, num_snapshots)` | Optional Exact side input | Optional same-tick mask; non-member Exact + SafeDefault. Publish before `snapshot` |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Required output | Dominant DOA `(az, el)` or `(u, v)` |
| out | `angle_spectrum` | float32 / float64 | `(num_items, num_az, num_el)` | Required output | MUSIC pseudospectrum; always emitted |
| out | `doa_multi` | float32 / float64 | `(num_items, max_sources, 2)` | Required output | Up to `min(d_i, max_sources)` peaks; unused rows NaN |
| out | `source_count` | int32 | `(num_items,)` | Required output | Peaks actually written into `doa_multi` |
| out | `eigenvalues` | float32 / float64 | `(num_items, C_est)` | Required output | Ascending covariance spectrum; `C_est` is post-smoothing |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable |
| live | `az_coordinates` / `el_coordinates` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; extents bounded by `max_az` / `max_el` |
| live | `spatial_smoothing` | `SpatialSmoothingOverride` | `{count, first, second}` int64 POD | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; `count` 0–2 selects 1-D / 2-D counts |
| live | `num_sources` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. Negative → per-item EFT `auto` |
| live | `source_count_pfa` | float64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. EFT target Pfa for `auto` |
| live | `max_sources` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. `doa_multi` capacity and EFT `K_max` |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | Omit both `max_items` and `max_az` for zero-span output bounds, or provide positive `max_items`, `max_az`, `max_el`, `max_sources`, and `max_channels` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override | Element positions + `λ`; `num_elements` must match `snapshot` channels Cost: Rebuilds steering (+ smoothing plan if enabled); warn-once on smoothing enable. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Boundary `(az, el)` ↔ unit-vector map; `direction_cosine` is `(u, v)` Cost: Fixes output meaning for the graph. |
| `az_coordinates` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_az` | First coordinate: azimuth radians for angular conventions, dimensionless `u` for `direction_cosine` Cost: Rebuilds steering manifold. |
| `el_coordinates` | `vector<double>` | none (required, non-empty) | `[live]` Exact host config Tensor, extent bounded by `max_el` | Second coordinate: elevation radians for angular conventions, dimensionless `v` for `direction_cosine`; singleton gives a 1-D scan Cost: Rebuilds steering manifold. |
| `num_sources` | int64 | `-1` (`auto`) | `[live]` Exact | Signal-subspace dimension; `< 0` → per-item EFT Cost: Free (device EFT or a fill); no FFT. |
| `source_count_pfa` | double | `1e-2` | `[live]` Exact | Target EFT over-estimation probability; used only for `auto` Cost: Free (EFT threshold only). |
| `max_sources` | int64 | `4` | `[live]` Exact | `doa_multi` / EFT `K_max`; MUSIC targets `min(d_i, max_sources)` peaks Cost: Reallocates `doa_multi`; implementation cap `64`. |
| `spatial_smoothing` | `vector<int64_t>` | empty (off) | `[live]` Exact `{count, first, second}` POD | Per-axis sub-array **counts** `[P]` / `[P_az, P_el]` on a filled ULA/URA Cost: Rebuilds plan + index H2D; warns once. |
| `forward_backward` | bool | `true` | static | Conjugate-reversed looks; ignored unless smoothing is on Cost: Part of the smoothing-plan key. |

Contract: [`OPERATOR_API.md` § `MUSICEstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md) ·

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

EFT uses Quinlan et al. 2007’s closed-form noise-eigenvalue profile for prediction and a first-order Gaussian threshold rather than the paper’s Monte-Carlo `η_P`. `source_count_pfa` is therefore a **target**, not a guaranteed false-alarm rate (see `radar_common/dsp/source_count.hpp` and [CLARAHOLOS-3394](https://jirasw.nvidia.com/browse/CLARAHOLOS-3394)). Seeding from the CFAR/cluster detection count is rejected.

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

## Performance

**Cold.** First tick (or first unseen geometry / grid / smoothing key) builds one steering artifact containing the untapered manifold and device scan grids and, if enabled, the spatial-smoothing plan plus index / FB-flip H2D. First `matx::eig` also creates a plan.

**Steady.** Covariance, eig, EFT, projector, and scan run every tick on stable constructor-created workspace. Device scan grids are reused with the cached manifold; the peak pass drops out once both associated optional outputs latch unwired. MatX eig plans are retained across matching configurations instead of being cleared each tick; existing live markers and stop-time retirement remain. Plan workspaces can persist until eviction/stop.

Float steering artifacts use a channel-major layout so neighboring beam threads read neighboring manifold entries; double retains the beam-major layout. Each beam keeps the same ordered scalar reductions and output ordering.

A paired Thor measurement of the existing `baseline-large` recipe (256 items, 16 channels, 48 snapshots, 61 × 31 grid) reduced float steady medians from 6.347 to 1.739 ms, with a 4.609 ms / 72.64% median paired saving. Three fresh pairs followed three screening pairs, each with five warmups and 50 frames. Medium float improved about 0.132 ms / 24.09%; double was approximately flat. All 47 native tests and exact five-output policies across seven cases in both precisions passed. These results cover the selected deployment and workloads.

Startup and tails are separate tradeoffs: cached-smoothing float cold time increased 1.057 ms in the fresh check, while its steady time improved 23.92%. Cached-smoothing double p99 increased 0.116 ms; small float and large double p99 increased 0.016 and 0.022 ms. Other cold/setup diagnostics were mixed. Fifty-frame p99 is descriptive; this is a steady-state improvement without a startup, tail, arbitrary-shape or multi-GPU guarantee.

**`[live]` cost.** Scalar overrides are cheap. `max_sources` reallocates `doa_multi`. A geometry / grid / smoothing override costs a steering-manifold rebuild (plus the smoothing plan and its index H2D) on the tick it arrives. The Graph harness isolates all seven live inputs. The reduce kernel stages `P_n` in shared memory only when it fits a 40 KiB budget **and** the item grid covers at least one occupancy wave; otherwise it reads `P_n` from global memory.

Harness (Graph, isolated production-Exact timing for every live input):

Run this performance smoke **inside a development container**, after
configuring the named build tree with performance enabled (see the
[developer guide](../../../DEVELOPER.md#performance-characterization)):

```bash
ctest --test-dir build -R perf_music_estimator_smoke
```

Size ladder axes: channels, detections (`num_items`), snapshots, and `num_az × num_el` (`xs`…`xl`; default smoke is `xs`/`sm`/`md`). Each size runs `array_geometry`, both scan grids, `spatial_smoothing`, `num_sources`, `source_count_pfa`, and `max_sources` scenarios plus a separate fixed model-order sweep at 1, 2, and `min(4, num_channels − 1)`. The Pfa scenario uses automatic source count; other live-override steady paths fix one source. Run the harness on the target GPU for current measurements.

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

Bind every device output (`bind_tensor_output_device` / compile options). Every optional/live input is declared whether or not it is wired.
