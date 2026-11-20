# `FeatureAssemblyOp`

## Purpose

Range–Doppler clustering prep: turn CFAR cells into the `points` matrix `ClusterDBSCANOp` / `DetectionReductionOp` consume. It is a **lookup / gather bridge**, not a general feature bus and not FMCW-specific — it never sees a chirp, beat frequency, or waveform. Anything that emits this `detection_idx` layout plus the optional tables can feed it. Mixed-unit scaling lives downstream (`ClusterDBSCANOp.epsilon`, `DetectionReductionOp.feature_weights`).

## Responsibilities

- Gathers configured physical axes, scores, angle values, or raw indices into a fixed-order row for every detection.
- Derives the required aligned input set from the immutable constructor `features` list and validates every lookup.
- Does not normalize mixed units, concatenate unrelated row sets, or own any live parameters.

## When to use

Use after CFAR and optional angle estimation to build the row-wise feature matrix consumed by clustering. Configure `features` at construction so required aligned inputs and output-column order are explicit. Do not use it as an arbitrary tensor concatenator: every row is indexed by `detection_idx`, and physical coordinates require their matching axis or angle inputs.

```text
CFARDetectorOp.detection_idx (+ detection_score) → FeatureAssemblyOp → points → ClusterDBSCANOp
⟨sparse estimator⟩.aoa_estimates → FeatureAssemblyOp.aoa_estimates   (tokens az_estimate / el_estimate)
⟨scan-grid tensor source⟩.az_grid / el_grid → FeatureAssemblyOp  (dense RDA; tokens az / el)
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `detection_idx` | int64 | `(num_points, 2 [+ n_preserved])` | Required trigger/aligned input | Sparse cells. Last two columns are always `(doppler_bin, range_bin)`. Rank 2; no opaque leading batch |
| in (selected) | `range_bins` | float32 / float64 | `(num_range,)` | Required aligned input when selected; otherwise optional non-member Exact | Dense 1-D axis. Required when token `range` is selected; indexed with `range_bin` |
| in (selected) | `velocity_bins` | float32 / float64 | `(num_doppler,)` | Required aligned input when selected; otherwise optional non-member Exact | Dense 1-D axis. Required when token `velocity` is selected; indexed with `doppler_bin` |
| in (selected) | `detection_score` | float32 / float64 | `(num_doppler, num_range)` plus optional leading matching preserved columns | Required aligned input when selected; otherwise optional non-member Exact | Dense map, gathered at the cell. Required when `features` includes `score` |
| in (selected) | `aoa_estimates` | float32 / float64 | `(num_points, 2)` | Required aligned input when selected; otherwise optional non-member Exact | Sparse companion, row-aligned with `detection_idx`. Tokens `az_estimate` / `el_estimate` |
| in (selected) | `az_grid` | float32 / float64 | `(num_az,)` | Required aligned input when selected; otherwise optional non-member Exact | Dense 1-D scan grid. Token `az` indexes the preserved `az_bin` column |
| in (selected) | `el_grid` | float32 / float64 | `(num_el,)` | Required aligned input when selected; otherwise optional non-member Exact | Dense 1-D scan grid. Token `el` indexes the preserved `el_bin` column |
| out | `points` | float32 / float64 | `(num_points, num_features)` | Required output | Sparse. Columns in `features` order; one row per detection |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_points` and `max_features` required | Sizes Tensor pools and bounded workspaces before `arm()`. |
| `features` | `std::vector<AssemblyFeature>` | Required immutable column list | Selects output columns and input-port requiredness. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| None | — | — | — | This operator has no Params; behavior is set by constructor captures and inputs. |

Contract: [`OPERATOR_API.md` § `FeatureAssemblyOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/detection/README.md`](../README.md)

## Feature reference

`detection_idx` is always sparse. Inputs are either 1-D axes, a dense score
map, or a row-aligned companion. For each row, the last two columns are
`(doppler_bin, range_bin)`. On the dense RDA path the two preserved columns
immediately before that pair are `(az_bin, el_bin)`.

Every row below is a value accepted by the constructor `features` list. The
list order is the output-column order.

| Config token | C++ enum | Emitted value | Required input |
| --- | --- | --- | --- |
| `range` | `AssemblyFeature::kRange` | `range_bins[range_bin]` | `range_bins` |
| `velocity` | `AssemblyFeature::kVelocity` | `velocity_bins[doppler_bin]` | `velocity_bins` |
| `score` | `AssemblyFeature::kScore` | `detection_score[..., doppler_bin, range_bin]` | `detection_score` |
| `az` | `AssemblyFeature::kAz` | `az_grid[az_bin]` | `az_grid` |
| `el` | `AssemblyFeature::kEl` | `el_grid[el_bin]` | `el_grid` |
| `az_estimate` | `AssemblyFeature::kAzEstimate` | `aoa_estimates[row, 0]` | `aoa_estimates` |
| `el_estimate` | `AssemblyFeature::kElEstimate` | `aoa_estimates[row, 1]` | `aoa_estimates` |
| `doppler_bin` | `AssemblyFeature::kDopplerBin` | Raw trailing Doppler index | none |
| `range_bin` | `AssemblyFeature::kRangeBin` | Raw trailing range index | none |
| `az_bin` | `AssemblyFeature::kAzBin` | Raw preserved azimuth index | none |
| `el_bin` | `AssemblyFeature::kElBin` | Raw preserved elevation index | none |

Selecting a token whose source table is missing is an error. Select
`range_bin` / `doppler_bin` / `az_bin` / `el_bin` explicitly when raw indices
are wanted.

## Usage

Static `features` is the whole configuration. It fixes both the output layout
and the compile-time temporal join; instantiate a separate operator or
recompile the graph to vary it. Only source ports used by the selected features
must be wired. Unselected gather ports may remain unwired.

Pass the final `features` list to `graph.op()` so it is fixed before `setup()`
declares required and optional ports. It is constructor-only and cannot be set
through Params. Ctor extents must set `max_points` and `max_features`.

### Pre-angle (cluster-then-angle, range–Doppler only)

`detection_idx` is `(N, 2)`. No angle ports.

```yaml
feature_assembly:
  features: [range, velocity, score]
```

```cpp
std::vector<AssemblyFeature> features{AssemblyFeature::kRange, AssemblyFeature::kVelocity,
                   AssemblyFeature::kScore};
auto feat = graph.op<FeatureAssemblyOp<float>>(
    "feature_assembly", extents, std::move(features));

graph.add_flow(cfar->detection_idx, feat->detection_idx, q);
graph.add_flow(range->range_bins, feat->range_bins, q);
graph.add_flow(doppler->velocity_bins, feat->velocity_bins, q);
graph.add_flow(cfar->detection_score, feat->detection_score, q);
graph.add_flow(feat->points, cluster->points, q);
```

### Sparse post-angle (angle-then-cluster)

Same RD gathers as pre-angle, plus a sparse estimator’s `(N, 2)` AoA.
`aoa_estimates` rows must match `detection_idx`.

```yaml
feature_assembly:
  features: [range, velocity, score, az_estimate, el_estimate]
```

```cpp
std::vector<AssemblyFeature> features{AssemblyFeature::kRange, AssemblyFeature::kVelocity,
                   AssemblyFeature::kScore,
                   AssemblyFeature::kAzEstimate,
                   AssemblyFeature::kElEstimate};
auto feat = graph.op<FeatureAssemblyOp<float>>(
    "feature_assembly", extents, std::move(features));

graph.add_flow(estimator->aoa_estimates, feat->aoa_estimates, q);
```

### Dense RDA (CFAR on the beam cube)

`detection_idx` is `(N, 4)`: `(az_bin, el_bin, doppler_bin, range_bin)`. Same
RD gathers as pre-angle. Angle columns are grid lookups, not `aoa_estimates`.

```yaml
feature_assembly:
  features: [range, velocity, score, az, el]
```

```cpp
std::vector<AssemblyFeature> features{AssemblyFeature::kRange, AssemblyFeature::kVelocity,
                   AssemblyFeature::kScore, AssemblyFeature::kAz,
                   AssemblyFeature::kEl};
auto feat = graph.op<FeatureAssemblyOp<float>>(
    "feature_assembly", extents, std::move(features));

graph.add_flow(grid_source->az_grid, feat->az_grid, q);
graph.add_flow(grid_source->el_grid, feat->el_grid, q);
```

The grid source publishes the same values configured on the beamformer.
`PhaseShiftBeamformerOp.az_coordinates` / `el_coordinates` are `[live]`
inputs; its `az_grid` / `el_grid` outputs expose the effective values for a
tick-coherent connection. An application that already resolves row-aligned `aoa_estimates` may
instead use `az_estimate` / `el_estimate`, as the current dense demo does.

### Raw RD bins (no physical axes)

Legal when integer Doppler/range coordinates are the intended feature units.

```yaml
feature_assembly:
  features: [doppler_bin, range_bin]
```

```cpp
std::vector<AssemblyFeature> features{AssemblyFeature::kDopplerBin,
                   AssemblyFeature::kRangeBin};
auto feat = graph.op<FeatureAssemblyOp<float>>(
    "feature_assembly", extents, std::move(features));
graph.add_flow(cfar->detection_idx, feat->detection_idx, q);
```

### Raw dense bins (no physical axes or scan-coordinate grids)

For `(az_bin, el_bin, doppler_bin, range_bin)` detection rows, all four raw
coordinates can be emitted without any gather inputs.

```yaml
feature_assembly:
  features: [az_bin, el_bin, doppler_bin, range_bin]
```

```cpp
std::vector<AssemblyFeature> features{AssemblyFeature::kAzBin,
                   AssemblyFeature::kElBin,
                   AssemblyFeature::kDopplerBin,
                   AssemblyFeature::kRangeBin};
auto feat = graph.op<FeatureAssemblyOp<float>>(
    "feature_assembly", extents, std::move(features));
graph.add_flow(cfar->detection_idx, feat->detection_idx, q);
```

## Algorithm

For each detection row, the operator interprets the final two columns as
`(doppler_bin, range_bin)` and any preceding columns as preserved coordinates.
It evaluates the constructor-selected feature tokens in order, gathering values
from aligned axis, score, grid, or AoA inputs. Physical range and velocity
never silently fall back to raw indices; select the explicit `*_bin` token when
an integer coordinate is wanted.

Out-of-range indices (when the corresponding axis/grid/score map is present)
set a device error flag; the host maps it after a stream sync **before**
`points` is emitted:

| Flag | Message |
| --- | --- |
| `kErrBatchCoord` | `FeatureAssemblyOp: detection_idx preserved-batch coordinate out of range for detection_score leading dims` |
| `kErrRangeBin` | `FeatureAssemblyOp: range_bin out of range for range_bins` |
| `kErrDopplerBin` | `FeatureAssemblyOp: doppler_bin out of range for velocity_bins` |
| `kErrScoreCoord` | `FeatureAssemblyOp: (doppler_bin, range_bin) out of range for detection_score` |
| `kErrAzBin` | `FeatureAssemblyOp: az_bin out of range for az_grid` |
| `kErrElBin` | `FeatureAssemblyOp: el_bin out of range for el_grid` |

## Shape and dtype rules

- **`points`.** `num_points` = `detection_idx` rows (0 is legal → `(0, num_features)`). `num_features` = `features.size()`.
- **`detection_idx`.** Rank 2, at least 2 columns. Extra columns are preserved-batch indices. Extra leading axes fail closed in `parse_detection_idx_layout`. Dense physical and raw angle-bin tokens require `n_preserved >= 2`.
- **Selected sources.** `range`, `velocity`, `score`, `az`/`el`, `az_bin`, and `el_bin` require `range_bins`, `velocity_bins`, `detection_score`, `aoa_estimates`, `az_grid`, and `el_grid`, respectively. Duplicate columns share one source port.
- **`detection_score` leading rank** must equal `n_preserved` when `score` is gathered (`FeatureAssemblyOp: detection_score leading rank must match detection_idx preserved batch columns`). Preserved rank is capped at `kMaxPreservedBatchDims` (8). Those leading axes are named CFAR leftovers (NCI-preserved or dense `az`/`el` bins), not a module batch axis.
- **Precision.** Template `T ∈ {float, double}`. Bin axes, score, AoA, and scan-coordinate grids must be real `T`. `detection_idx` is int64.
- **Named leading / batch.** No opaque leading batch on `points` or `detection_idx`. Tick row count may grow up to `MaxTensorExtents.max_points`.
- **Legal mid-stream shape change.** `num_points` (and score-map geometry) may change per tick; `points` is re-loaned to the new shape. `features` is static and determines the compiled contract; changing it requires graph recompilation.
- **Rejected combinations:**

  | Condition | Message |
  | --- | --- |
  | Empty `features` | `FeatureAssemblyOp: features must be a non-empty list` |
  | `range` without `range_bins` | graph compile rejects the missing required aligned source |
  | `velocity` without `velocity_bins` | graph compile rejects the missing required aligned source |
  | Bad `detection_idx` rank/cols | `FeatureAssemblyOp: detection_idx must have shape (num_points, 2 [+ n_batch])` |
  | Rows exceed ctor max | `FeatureAssemblyOp: detection_idx rows (N) exceed MaxTensorExtents max_points (M)` |
  | Feature count exceeds ctor max | `FeatureAssemblyOp: features size (N) exceeds MaxTensorExtents max_features (M)` |
  | `score` without `detection_score` | `FeatureAssemblyOp: features includes 'score' but detection_score input is absent` |
  | `az_estimate`/`el_estimate` without `aoa_estimates` | `FeatureAssemblyOp: features includes 'az_estimate'/'el_estimate' but aoa_estimates input is absent` |
  | `az` without `az_grid` | `FeatureAssemblyOp: features includes 'az' but az_grid input is absent` (same pattern for `el` / `el_grid`) |
  | Dense tokens, `< 2` preserved cols | `FeatureAssemblyOp: dense angle features require detection_idx with at least two preserved-batch columns (…, az_bin, el_bin, doppler_bin, range_bin)` |
  | `aoa_estimates` not `(N, 2)` or `N ≠ num_points` | `FeatureAssemblyOp: aoa_estimates must have shape (num_points, 2)` / `… row count must match detection_idx` |
  | Rank-1 ports not rank-1 | `FeatureAssemblyOp: range_bins must be rank-1 (num_range,)` (same for `velocity_bins`, `az_grid`, `el_grid`) |
  | `features` changed after first emit | `FeatureAssemblyOp: features is static and cannot change after the first successful activation; instantiate a separate operator per column layout` |
  | Non-device / non-contiguous / wrong dtype | `FeatureAssemblyOp: tensor received on input port '…' is not CUDA-accessible …` / `… must be contiguous (compact row-major)` / `… has dtype …, expected …` |
  | Missing ctor extent | `FeatureAssemblyOp: MaxTensorExtents.max_points must be a positive value (required for Tensor pool sizing)` (same for `max_features`) |

## Performance

**Cold.** First non-empty tick: feature-id H2D into the ctor-sized table; score leading-dims upload when `score` is gathered with preserved batch. **Steady.** Offset kernel (if preserved score batch) + assemble kernel + error-flag sync + one `points` loan. Feature ids reuse after the first successful non-empty emit. Score leading dims re-upload only when that layout changes. A cache miss does not `cudaMalloc`.

No `[live]` overrides. `features` cannot change mid-stream.
