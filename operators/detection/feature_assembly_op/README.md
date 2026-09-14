# `FeatureAssemblyOp`

Range–Doppler clustering prep: turn CFAR cells into the `points` matrix
`ClusterDBSCANOp` / `DetectionReductionOp` consume. It is a **lookup / gather
bridge**, not a general feature bus and not FMCW-specific — it never sees a
chirp, beat frequency, or waveform. Anything that emits this `detection_idx`
layout plus the optional tables can feed it. Mixed-unit scaling lives
downstream (`ClusterDBSCANOp.epsilon`, `DetectionReductionOp.feature_weights`).

The fact a caller must not miss: each `points` row is exactly one
`detection_idx` row. **Tokens are output columns, not Inputs.** `range` is
`range_bins[range_bin]`; there is no `range` port and no dense range map.

It lives under `operators/detection/` because it speaks CFAR’s `detection_idx`
(trailing `(doppler_bin, range_bin)`, optional preserved-batch columns).
`ClusterDBSCANOp` is radar-agnostic and only sees `points`. Do not move this
operator into `clustering/` — that group is the clusterer, not the RD adapter.

| Kind | Port | Dtype | Shape | Indexing |
| --- | --- | --- | --- | --- |
| in | `detection_idx` | int64 | `(num_points, 2 [+ n_preserved])` | Sparse cells. Last two columns are always `(doppler_bin, range_bin)`. Rank 2; no opaque leading batch |
| in | `range_bins` | float32 / float64 | `(num_range,)` | Dense 1-D axis. Required when token `range` is selected; indexed with `range_bin` |
| in | `velocity_bins` | float32 / float64 | `(num_doppler,)` | Dense 1-D axis. Required when token `velocity` is selected; indexed with `doppler_bin` |
| in | `detection_score` | float32 / float64 | `(num_doppler, num_range)` plus optional leading matching preserved columns | Dense map, gathered at the cell. Required when `features` includes `score` |
| in | `aoa_estimates` | float32 / float64 | `(num_points, 2)` | Sparse companion, row-aligned with `detection_idx`. Tokens `az_estimate` / `el_estimate` |
| in | `az_angles` | float32 / float64 | `(num_az,)` | Dense 1-D scan grid. Token `az` indexes the preserved `az_bin` column |
| in | `el_angles` | float32 / float64 | `(num_el,)` | Dense 1-D scan grid. Token `el` indexes the preserved `el_bin` column |
| out | `points` | float32 / float64 | `(num_points, num_features)` | Sparse. Columns in `features` order; one row per detection |

`n_preserved = idx_cols - 2`: leading integers CFAR copied from axes in front of
the (Doppler, range) plane (NCI leftovers, or dense-path `(az_bin, el_bin)`).
They are **columns**, not a tensor batch axis.

```text
CFARDetectorOp.detection_idx (+ detection_score) → FeatureAssemblyOp → points → ClusterDBSCANOp
⟨sparse estimator⟩.aoa_estimates → FeatureAssemblyOp.aoa_estimates   (tokens az_estimate / el_estimate)
⟨scan-grid tensor source⟩.az_angles / el_angles → FeatureAssemblyOp  (dense RDA; tokens az / el)
```

The six gathers are **data-plane** Inputs, not `[live]` overrides. Static
`features` determines the compile-time join: selected source ports are required
zero-window `OnAligned` members; raw-bin-only layouts use
`OnEach{detection_idx}`. Unselected gathers are non-member `Exact{}` +
`SafeDefault{}` inputs, so they may remain unwired.

Contract: [`OPERATOR_API.md` § `FeatureAssemblyOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/detection/README.md`](../README.md)

## Scope

**In:** Detect on a (Doppler, range) plane — or an RDA cube whose extra beam
axes show up as preserved `detection_idx` columns — then cluster. Three
placements, one operator, no variant. This module’s FMCW SISO / SIMO /
MIMO-TDM / MIMO-DDM apps are the `rd_maps` / bin-axis producers we ship
today; a pulse-Doppler chain would occupy the same ports.

**Out:** Waveform math (range/Doppler processors); SAR / image-cell
indices; an open user-defined column list; Cartesian `(x, y)` from
range+angle; sub-bin interpolation; gathering `cfar_power` or
`noise_estimate` (use `score` — CFAR’s noise-normalized margin). Extra
preserved columns (TX/RX before NCI) index the score plane but are not
emit-able feature tokens except dense `(az_bin, el_bin)`, which can be emitted
as raw indices or looked up on angle grids. A per-app assembler is the
documented hatch for anything else. Pulse-Doppler is out of **module** scope
this wave; the same RD-cell gather would apply if a later chain reused this
`detection_idx` contract.

There is no dense az/el *value cube*. On the RDA path the cell’s angle is the
scan-grid coordinate: tokens `az` / `el` plus `az_angles` / `el_angles`.
Sparse row-aligned estimator results use `az_estimate` / `el_estimate`.

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
| `az` | `AssemblyFeature::kAz` | `az_angles[az_bin]` | `az_angles` |
| `el` | `AssemblyFeature::kEl` | `el_angles[el_bin]` | `el_angles` |
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

graph.add_flow(grid_source->az_angles, feat->az_angles, q);
graph.add_flow(grid_source->el_angles, feat->el_angles, q);
```

The grid source publishes the same values configured on the beamformer.
`PhaseShiftBeamformerOp.az_angles` / `el_angles` are `[live]` inputs, not
outputs. An application that already resolves row-aligned `aoa_estimates` may
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

### Raw dense bins (no physical axes or angle grids)

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

`compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous
namespace. It receives `detection_idx` (device + contiguous + int64 in the same
try), receives the optional gathers, and binds the static `features` list to
this input (`build_assembly_stage_plan` → `AssemblyStagePlan` with layout,
gather flags, and `out_shape`). There is no `resolve_params`: no `[live]`
ports, and the empty-`features` sentinel is rejected in the plan builder rather
than mapped to an optional. Trailing rank is `(num_points, 2 [+ n_preserved])`,
not a `(chirps, samples)` ADC cube, so the plan builder uses
`parse_detection_idx_layout` instead of `collapse_named_tx_rx_batch`. Rank must
be exactly 2.

Token evaluation is the table in **Feature reference**. Physical
range/velocity never silently change units to raw indices. No scaling. Custom
CUDA (MatX is a poor fit for sparse irregular gathers). A shared gather
primitive with `SnapshotGatherOp` was rejected: snapshot gathering
uses complex channel vectors, 2-D neighborhoods, edge policies, and a different
execution shape. This operator intentionally owns its scalar assemble kernel.

Out-of-range indices (when the corresponding axis/grid/score map is present)
set a device error flag; the host maps it after a stream sync **before**
`points` is emitted:

| Flag | Message |
| --- | --- |
| `kErrBatchCoord` | `FeatureAssemblyOp: detection_idx preserved-batch coordinate out of range for detection_score leading dims` |
| `kErrRangeBin` | `FeatureAssemblyOp: range_bin out of range for range_bins` |
| `kErrDopplerBin` | `FeatureAssemblyOp: doppler_bin out of range for velocity_bins` |
| `kErrScoreCoord` | `FeatureAssemblyOp: (doppler_bin, range_bin) out of range for detection_score` |
| `kErrAzBin` | `FeatureAssemblyOp: az_bin out of range for az_angles` |
| `kErrElBin` | `FeatureAssemblyOp: el_bin out of range for el_angles` |

## Shape and dtype rules

- **`points`.** `num_points` = `detection_idx` rows (0 is legal → `(0, num_features)`). `num_features` = `features.size()`.
- **`detection_idx`.** Rank 2, at least 2 columns. Extra columns are preserved-batch indices. Extra leading axes fail closed in `parse_detection_idx_layout`. Dense physical and raw angle-bin tokens require `n_preserved >= 2`.
- **Selected sources.** `range`, `velocity`, `score`, `az`/`el`, `az_bin`, and `el_bin` require `range_bins`, `velocity_bins`, `detection_score`, `aoa_estimates`, `az_angles`, and `el_angles`, respectively. Duplicate columns share one source port.
- **`detection_score` leading rank** must equal `n_preserved` when `score` is gathered (`FeatureAssemblyOp: detection_score leading rank must match detection_idx preserved batch columns`). Preserved rank is capped at `kMaxPreservedBatchDims` (8). Those leading axes are named CFAR leftovers (NCI-preserved or dense `az`/`el` bins), not a module batch axis.
- **Precision.** Template `T ∈ {float, double}`. Bin axes, score, AoA, and angle grids must be real `T`. `detection_idx` is int64.
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
  | `az` without `az_angles` | `FeatureAssemblyOp: features includes 'az' but az_angles input is absent` (same pattern for `el` / `el_angles`) |
  | Dense tokens, `< 2` preserved cols | `FeatureAssemblyOp: dense angle features require detection_idx with at least two preserved-batch columns (…, az_bin, el_bin, doppler_bin, range_bin)` |
  | `aoa_estimates` not `(N, 2)` or `N ≠ num_points` | `FeatureAssemblyOp: aoa_estimates must have shape (num_points, 2)` / `… row count must match detection_idx` |
  | Rank-1 ports not rank-1 | `FeatureAssemblyOp: range_bins must be rank-1 (num_range,)` (same for `velocity_bins`, `az_angles`, `el_angles`) |
  | `features` changed after first emit | `FeatureAssemblyOp: features is static and cannot change after the first successful activation; instantiate a separate operator per column layout` |
  | Non-device / non-contiguous / wrong dtype | `FeatureAssemblyOp: tensor received on input port '…' is not CUDA-accessible …` / `… must be contiguous (compact row-major)` / `… has dtype …, expected …` |
  | Missing ctor extent | `FeatureAssemblyOp: MaxTensorExtents.max_points must be a positive value (required for Tensor pool sizing)` (same for `max_features`) |

## Constructor configuration

| Field | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `features` | ordered list of {range, velocity, score, az, el, az_estimate, el_estimate, doppler_bin, range_bin, az_bin, el_bin} | none (required, non-empty) | constructor-only | Feature-id vector uploaded once into a ctor-sized int32 table | Column set and order; fixes `num_features` and the compiled temporal join |

There are no Params or `[live]` parameters. `features` has no Exact port and is not declared as `spec.input`. Pass it directly to `graph.op()`; `setup()` and `contract()` read the immutable constructor capture. Do not invent gather-port `[live]` tags: those Inputs are data sources, and their requiredness follows from `features`.

YAML tokens match `to_string(AssemblyFeature)`: `range`, `velocity`, `score`, `az`, `el`, `az_estimate`, `el_estimate`, `doppler_bin`, `range_bin`, `az_bin`, `el_bin`.

## SDK implementation notes

- **Construction.** `FeatureAssemblyOp(const MaxTensorExtents& extents, std::vector<AssemblyFeature> features)`. The constructor stores the immutable column list because it controls port requiredness. Required extents: `max_points`, `max_features`. The `points` pool is `max_rank = 2`, `max_byte_span` = real-`T` `(max_points, max_features)`. A tick with more rows than `max_points`, or more columns than `max_features`, is rejected in the plan builder.
- **Workspace.** Four ctor-sized device Tensors (`make_device_tensor`), not `ConfigCache` and not a first-tick `cudaMalloc`: `feature_ids` (`max_features` int32), `score_offsets` (`max_points` int64 — worst-case source count, not the collapsed output width), `score_leading` (`kMaxPreservedBatchDims` int64), `error_flag` (one int32). Tensor pools are per output port only; there is no scratch-loan API for these tables. Feature-id H2D is keyed on the first successful non-empty emit; score leading dims re-upload when that layout changes. `DLDataType{kDLInt, 64, 1}` / `{kDLInt, 32, 1}` are DLPack encodings of scalar `int64_t` / `int32_t` (signed, 64 or 32 bits, 1 lane), independent of template `T`.
- **Triggering.** `contract()` reads constructor-captured `features`. Raw-bin-only layouts use `OnEach{detection_idx}`. Any selected source adds its port to a zero-window `OnAligned` trigger with `detection_idx`; every member selects `Exact{}`. The source set is deduplicated, so `az_estimate` + `el_estimate` add `aoa_estimates` once.
- **Unselected gathers.** Every gather not selected by `features` is still declared as a non-member `Exact{}` + `SafeDefault{}` input. The pinned SDK admits such an unwired Tensor input ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11), reporting `kNotConnected`; `receive_optional_value` maps that to absence. This keeps one fixed `setup()` port set while preventing a wired-but-undeclared replay mismatch.
- **Ports and queues.** Every Input: `queue_depth(2)`, device tensors. `points`: device, allocation bounds above. `setup()` registers data-plane I/O only (trigger, then optional gathers, then `points`). There is no live block. Dtype is not pinned on the port (enforced in `compute()`).
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. Every received Tensor is retained until the next tick. The hot path **does** synchronize: a scalar error-flag D2H + `cudaStreamSynchronize` after the assemble kernel, **before** emit, so an OOB index fails the tick rather than publishing a partial `points`. That sync is accepted operator-owned workspace (Holoscan `Error` is host-side); it dominates small-N steady time and is separate from the `points` alloc question.
- **Emission.** Single required output. Empty `detection_idx` is a no-op fill. Non-empty ticks `prepare_device_tensor` (stream-ordered kernels only), inspect the error flag, then `emit_prepared_tensor` — the prepare/emit split exists here so a device OOB can drop the loan before publish, not because there is a latched optional sibling. `emit_device_tensor`'s fill must not host-sync, so the D2H sits between prepare and emit.
- **Configuration authoring.** There is no Params schema. Apps pass the selected `features` vector directly to `graph.op()` as a replayable constructor capture.
- **Precision.** `FeatureAssemblyOp<float>`, `FeatureAssemblyOp<double>`.
- **[EA HSDK5 ONLY]** No port renames. The contract relies on the optional non-member Tensor support in the pinned SDK ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Graph live tests are N/A (no `[live]` params).

## Consumer lifetime notes

`points` is a fresh pool-loaned device buffer each tick. A consumer may retain it. `points` is `(num_detections, num_features)` (typically small).

## Performance

**Cold.** First non-empty tick: feature-id H2D into the ctor-sized table; score leading-dims upload when `score` is gathered with preserved batch. **Steady.** Offset kernel (if preserved score batch) + assemble kernel + error-flag sync + one `points` loan. Feature ids reuse after the first successful non-empty emit. Score leading dims re-upload only when that layout changes. A cache miss does not `cudaMalloc`.

No `[live]` overrides. `features` cannot change mid-stream.

Harness: `performance/cpp/perf_feature_assembly.cpp` is a Graph rewrite. Axes: map-size ladder (`xs`…`xlw`) × detection-count. The driver times the physical `{range, velocity}` pack with aligned axis inputs; score / AoA / dense paths are unit-tested. No measured numbers are recorded in-repo.

```bash
ctest --test-dir build -R perf_feature_assembly_smoke
```

## Tests and verification

Graph tests cover raw-bin-only `OnEach`, selected-source `OnAligned`, unwired
unselected gathers, and physical / score / sparse-AoA / dense-grid paths.
Direct `launch_assemble_features` tests retain focused kernel and OOB coverage.

| Check | Proves |
| --- | --- |
| `LooksUpPhysicalRangeVelocityAndScore` | `range = range_bins[range_bin]`, `velocity = velocity_bins[doppler_bin]`, score gathered (`TESTING.md`) |
| `RejectsPhysicalFeaturesWithoutAxes` / `AssemblesRawBinEscapeHatches` | Physical sources are mandatory; raw indices require explicit tokens |
| `AppendsAoaEstimates` | Sparse `az_estimate`/`el_estimate` from `aoa_estimates` |
| `DenseAzElGridLookup` / `DenseAzElWithRangeVelocity` | Dense `az`/`el` grid lookup from preserved bins |
| `AssemblesRawDenseAngleBinIndices` | Raw dense azimuth/elevation indices need only `detection_idx` |
| `GathersScoreAcrossPreservedBatch` | Score-plane offsets for leading batch |
| `EmptyDetectionIdxEmitsEmptyPoints` | `(0, num_features)` |
| `RejectsScoreFeatureWithoutScoreInput` / `RejectsAzEstimateWithoutAoaInput` / `RejectsElEstimateWithoutAoaInput` / `RejectsAzWithoutGrid` | Selecting a token whose required input is absent (`TESTING.md`) |
| `RejectsOob*` / `RejectsAzWithTooFewPreservedColumns` / `RejectsScoreLeadingRankMismatch` | Bounds and shape errors |
| `RejectsOpaqueLeadingBatch` | Extra leading dim is not a named axis; `parse_detection_idx_layout` fail-closes (Part B) |
| `FeaturesConfiguredAtConstruction` / `FloatDoubleAgreeOnPhysicalLookup` | Constructor field + both precisions |
| selected-source Graph joins | Constructor-derived trigger requires exactly the source ports used by `features` |

`TESTING.md` row-alignment is implied by the gather tests (one output row per `detection_idx` row). Mid-stream optional Graph wiring is skipped. Python tests exist but are not built (HSDK5_PORT §6).

```bash
ctest --test-dir build -R test_feature_assembly --output-on-failure
```

## Contract conformance

Conforms on ports, dtypes, explicit physical-versus-raw token semantics,
Constructor-derived temporal joins, row alignment, rank-2 `detection_idx`
(fail-closed extra leading axes), and dense az/el lookup. The operator is
checklist-complete with no operator-local open items.

Header comment block matches the implementation (constructor-derived required
sources, no `[live]` params, explicit raw tokens, ctor-sized workspace).
