# `DetectionReductionOp`

Reduces labeled detections to one **representative detection** and, optionally, one reduced feature row per non-noise cluster. `rep_idx` always selects an actual row of `detection_idx`; synthetic weighted means are confined to `reduced_points`.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `labels` | int64 | `(num_points,)` | Cluster id per detection; noise = `-1` (no representative). Row-aligned with `detection_idx` |
| in | `detection_idx` | int64 | `(num_points, 2 [+ n_preserved])` | Detection coordinates; row-aligned with `labels` |
| in | `points` | float32 / float64 | `(num_points, num_features)` | Row-aligned reduction features; may augment the clustered matrix. Always required |
| out | `rep_idx` | int64 | `(num_clusters, 2 [+ n_preserved])` | One `detection_idx` row per non-noise cluster; row count *is* `num_clusters` |
| out | `reduced_points` | float32 / float64 | `(num_clusters, num_features)` | Optional/latched; selected input rows or synthetic weighted means |
| live | `representative` | enum | scalar | Exact override; the `contract()` entry is unconditional and wiring enables it |
| live | `score_method` | enum | scalar | Exact override; the `contract()` entry is unconditional and wiring enables it |
| live | `feature_weights` | float64 config Tensor | `(num_features,)`, rank-1 pageable host | Exact override; the `contract()` entry is unconditional and wiring enables it |

```text
labels + detection_idx + points
  → DetectionReductionOp → rep_idx
                         → reduced_points (optional)
```

Contract: [`OPERATOR_API.md` § `DetectionReductionOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/detection/README.md`](../README.md)

## Algorithm

Thin operator over the public `radar_common/reduce/` group-by-label primitive (`detection_reduction_stage.hpp` only includes those launch wrappers). `compute()` resolves the enabled `[live]` overrides, validates ports, finds `max_label`, runs the selected policy, then compact-gathers occupied slots into exact-sized outputs.

1. **Group by `labels`.** Noise (`-1`, `kNoiseLabel`) is excluded. Non-noise labels must lie in `[0, max_label]`; empty ids in that range are skipped when compacting. `max_label + 1 > 2^20` is rejected. Exact output sizing requires scalar D2H reads of `max_label` and the occupied-cluster count.

2. **Select one row per occupied label** via `holoscan::radar_common_reduce`:

   - **`first`** — `atomicMin` of the row index per label (`launch_select_first`). Lowest `detection_idx` row wins. Ignores `points` and `feature_weights`.
   - **`peak`** — per-row scalar score, then `atomicMax` of ordered bits, then `atomicMin` of the row among score ties (lowest index). Built-in scores from `peak_score` in `group_by_label.cu`:

     ```text
     weighted_sum :  Σ_f  w[f] * row[f]
     weighted_l2  :  sqrt( Σ_f (w[f] * row[f])^2 )     // ‖w ⊙ feature‖₂
     weighted_max :  max_f w[f] * row[f]
     custom       :  ScoreFn{}(row, num_features)     // compile-time device functor
     ```

     Default `DetectionReductionOp<T>` uses `ScoreFn = NoCustomScore` and rejects `score_method=custom` with `DetectionReductionOp: score_method=custom requires a compile-time ScoreFn template argument (DetectionReductionOp<T, ScoreFn>)`. A real functor must be instantiated in a CUDA TU (`detection_reduction_op_impl.cuh`).
   - **`medoid`** — two segmented passes. Centroid per label: sum `w[f] * row[f]` then divide by the cluster count (`finalize_centroid_kernel`: `inv = 1 / counts[lab]`). Then argmin of scaled Euclidean distance, quoted from `medoid_neg_dist2`:

     ```text
     d = w[f] * row[f] - cen[f]
     dist2 = Σ d²
     score = -dist2          // maximize → nearest
     ```

     Equal distance then `atomicMin` of the row (lowest index). The centroid is never emitted.
   - **`weighted_mean`** — empty `feature_weights` gives uniform row weights `α[i] = 1`; otherwise the configured `score_method` produces each row's scalar `α[i]`. Per cluster:

     ```text
     reduced_points = Σ_i α[i] * points[i] / Σ_i α[i]
     index_centroid = Σ_i α[i] * detection_idx[i] / Σ_i α[i]
     ```

     Every row weight must be finite and non-negative, and each occupied cluster must have a finite positive sum. `rep_idx` is the actual input row nearest `index_centroid` over all index columns; ties choose the lowest input row.

3. **Emit.** Compaction walks labels `0 … max_label` in ascending label-id order. `rep_idx` copies each selected real index row. If wired, `reduced_points` gathers the selected feature row for `first` / `peak` / `medoid`, or the synthetic feature centroid for `weighted_mean`. Noise yields no row; empty/all-noise inputs emit matching zero-row shapes.

## Shape and dtype rules

- **`rep_idx`.** Same column count as `detection_idx`. Rows = occupied non-noise labels (`≤ max_label + 1`, after skipping gaps).
- **`reduced_points`.** Same column count and template precision as `points`; rows match `rep_idx`. Optional/latched.
- **Alignment.** `labels` rank 1; `detection_idx` rank 2 with `≥ 2` columns; `labels.shape[0] == detection_idx.shape[0]`. `points` is `(num_points, num_features)` with the same row count.
- **Precision.** Template `T ∈ {float, double}` for `points`. Index ports are int64.
- **Extents.** Ctor requires `max_clusters` and `max_features`. The `rep_idx` pool span is `(max_clusters, 10) · sizeof(int64)`; the `reduced_points` pool span is `(max_clusters, max_features) · sizeof(T)`.
- **Legal mid-stream change.** `num_points`, `num_features`, and `num_clusters` are inferred per tick. A `[live]` `representative` / `score_method` / `feature_weights` change switches policy or metric without reallocating the output rank (row count still follows the scene).
- **Rejected combinations:**

  | Condition | Message |
  | --- | --- |
  | `peak` + `custom` + default `ScoreFn` | `DetectionReductionOp: score_method=custom requires a compile-time ScoreFn template argument (DetectionReductionOp<T, ScoreFn>)` |
  | `labels` not rank 1 | `DetectionReductionOp: labels must have shape (num_points,)` |
  | Bad `detection_idx` | `DetectionReductionOp: detection_idx must have shape (num_points, 2 [+ n_batch])` |
  | More than eight preserved-index columns | `DetectionReductionOp: detection_idx has too many columns for the rep_idx pool` |
  | Row-count mismatch | `DetectionReductionOp: labels and detection_idx row counts must match` |
  | Missing `points` connection | Graph compilation rejects the missing required input |
  | Bad `points` shape | `DetectionReductionOp: points must have shape (num_points, num_features)` |
  | `feature_weights` length ≠ `num_features` | `DetectionReductionOp: feature_weights length must equal num_features` |
  | Invalid weighted-mean row/cluster weight | `DetectionReductionOp: weighted_mean requires finite non-negative row weights and a positive finite weight sum per cluster` |
  | `max_label + 1 > 2^20` | `DetectionReductionOp: max cluster label exceeds supported range (expected DBSCAN-style dense labels)` |
  | Missing ctor extent | `DetectionReductionOp: MaxTensorExtents.max_clusters/max_features must be a positive value (required for Tensor pool sizing)` |
  | No CUDA stream | `DetectionReductionOp: no CUDA compute stream` |

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `representative` | enum {peak, medoid, first, weighted_mean} | `peak` | `[live]` Exact | Selects a different launch | Member or aggregate policy |
| `score_method` | enum {weighted_sum, weighted_l2, weighted_max, custom} | `weighted_sum` | `[live]` Exact | Free | Peak score or weighted-mean row-weight collapse |
| `feature_weights` | float64 vector, length `num_features` | none | `[live]` Exact host config Tensor | H2D when used | Peak collapse, medoid scaling, or weighted-mean row weights; empty means uniform mean only for `weighted_mean` |
| `score_fn` | compile-time device functor | none (`NoCustomScore`) | static (template argument) | — | `T operator()(const T* row, int64_t num_features) const`; only when `score_method = custom` |

`feature_weights` has no universal default for member policies because features may use mixed units. One-hot `w` reduces `weighted_sum` to one column. For `weighted_mean`, that scalar becomes the row weight rather than an argmax score; empty weights request an ordinary mean.

`points` is always part of the zero-window aligned trigger. Consequently every live `representative` value is legal without changing required input membership; `first` simply ignores the aligned matrix values.

## SDK implementation notes

- **Construction.** `DetectionReductionOp(const MaxTensorExtents& extents)`. Required extents: `max_clusters`, `max_features`. Pools are rank 2 and bounded for `(max_clusters, 10)` int64 index rows and `(max_clusters, max_features)` real feature rows.
- **Triggering.** `OnAligned{labels, detection_idx, points}` always selects all three required data inputs at zero window. The three `[live]` overrides are unconditional non-members with `Exact + SafeDefault`; wiring is the application's only override statement.
- **Ports and queues.** `labels`, `detection_idx`, and `points`: `queue_depth(2)`, device. Live enums and the `feature_weights` config Tensor (rank-1 pageable host): `queue_depth(2)`.
- **Streams and lifetime.** `context.cuda_stream()`. Inputs retained until the next tick. The hot path **does** synchronize twice: max-label D2H, then cluster-count D2H, so the contract-required `rep_idx` loan can be exact-sized.
- **Emission.** Prepare optional `reduced_points` before required `rep_idx`, then publish `rep_idx` last as the trigger boundary. An unwired reduced output warns once and latches off its output-only compact/loan work; weighted-mean accumulation remains necessary for `rep_idx`.
- **Scratch.** Per-tick `make_device_tensor` temps (`best_rows`, score bits, centroid sums, …) are accepted operator-owned scratch. Their slot count follows runtime label ids, so ctor extents do not provide a safe fixed capacity when sparse label gaps are admitted.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.DetectionReductionParams` version 1. Fields: `representative`, `score_method`, `feature_weights`. There are no latch or override-wiring flags.
- **Precision.** Explicit instantiations: `DetectionReductionOp<float>`, `DetectionReductionOp<double>` (default `NoCustomScore`). Custom `ScoreFn` specializations are caller-provided in a CUDA TU.
- **Live override payloads.** Exact enum `Input<>` ports for `representative` / `score_method` and a rank-1 pageable-host float64 config Tensor for `feature_weights` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). All three entries are unconditional and resolved into one tick-local stage config; the Tensor payload is validated on arrival (rank, host memory, contiguity, dtype, `max_features` extent).

## Consumer lifetime notes

Both outputs are fresh pool-loaned device buffers each tick. A consumer may retain them.

## Performance

**Cold.** First-tick device temps and weight H2D.

**Steady.** Find-max-label + policy kernels + compact; two scalar syncs; per-tick scratch allocs. Weight vector is re-uploaded whenever peak/medoid needs it.

**Live rebuilds.** `representative` / `score_method` are free policy switches (no cached kernel). A `feature_weights` override re-uploads the weight vector on the tick it arrives.

Harness: `performance/cpp/perf_detection_reduction.cpp` (Graph triple-in, both outputs consumed). Axes: detection count, cluster count, policy {first, peak, medoid, weighted_mean}. Live-override timing is not measured. No measured numbers are recorded in-repo.

```bash
ctest --test-dir build -R perf_detection_reduction_smoke
```

## Tests and verification

| Check | Proves |
| --- | --- |
| `FirstSelectsLowestIndexPerCluster` | `first` = lowest row (`TESTING.md`) |
| `PeakSelectsMaxWeightedSum` / `PeakSelectsMaxWeightedL2` / `PeakSelectsMaxWeightedMax` | Built-in peak scorers (`TESTING.md`) |
| `PeakEqualScorePicksLowestIndex` / `MedoidEqualDistancePicksLowestIndex` | Exact ties choose the lowest input row |
| `MedoidRespectsFeatureWeights` | Medoid uses `w`-scaled centroid distance (`TESTING.md`) |
| `WeightedMeanUsesScoreAndNearestRealIndex` / `WeightedMeanEmptyWeightsMeansUniformMean` | One-hot row weighting and ordinary-mean fallback |
| `WeightedMeanNearestUsesAllIndexColumnsAndTies` | Every index column participates; ties choose the lowest input row |
| `MemberPoliciesGatherReducedPoints` / `ReducedPointsCompactSparseLabelsInOrder` | Optional feature output gathers or compacts in label order |
| `WeightedMeanRejectsInvalidRowWeights` | Negative/non-finite rows and non-positive cluster totals reject |
| `ExcludesNoiseAndEmitsOnePerCluster` / `SparseLabelGapsSkipped` | One real cell per occupied label; noise excluded; gaps skipped |
| `WidenedDetectionIdxCols` | Preserved-batch columns copied through |
| `RejectsOpaqueLeadingBatch` / `RejectsTooManyDetectionIdxCols` | Fail-closed rank and preserved-index bounds |
| `EmptyInputEmitsZeroClusters` / `AllNoiseEmitsZeroClusters` | `(0, idx_cols)` |
| `RejectsMismatchedFeatureWeights` / `RejectsCustomWithoutScoreFn` | Weight length and default-`ScoreFn` reject |
| `RepresentativeConfigurableFromParams` / `ScoreMethodConfigurableFromParams` / `FeatureWeightsConfigurableFromParams` | Params field names |
| `FloatDoubleAgreeOnPeak` | Both precisions |
| `FloatDoubleAgreeOnWeightedMean` | Weighted aggregate and nearest real index agree across precisions |
| `LiveFeatureWeights*` (Exact match, skewed capture, revert-to-Params, malformed, bound overflow) | Config-Tensor override: Exact join, revert-to-Params, payload rejects |
| `RejectsMissingRequiredPoints` | The graph rejects an unwired required `points` input |
| `LiveRepresentativeAppliesForOneTick` / `LiveScoreMethodAppliesForOneTick` | Exact enum overrides apply for one tick and then revert to Params |
| `LiveWeightedMeanAppliesForOneTick` | Weighted mean is legal as a tick-local representative override |
| `test_detection_reduction_custom.cu` | Graph coverage for a non-default `ScoreFn` instantiated in a CUDA TU |

`TESTING.md` “every `rep_idx` is a subset of `detection_idx`” is covered by the policy tests (they assert the expected cell). Python tests exist but are not built.

```bash
ctest --test-dir build -R test_detection_reduction --output-on-failure
```

## Usage

```yaml
reduction:
  representative: weighted_mean
  score_method: weighted_sum
  feature_weights: [0.0, 0.0, 1.0]
```

```cpp
auto reduction = graph.op<DetectionReductionOp<float>>(
    "detection_reduction", extents);
reduction.set_params(from_config<DetectionReductionParams>("reduction"));

graph.add_flow(cluster->labels, reduction->labels, q);
graph.add_flow(cfar->detection_idx, reduction->detection_idx, q);
graph.add_flow(feat->points, reduction->points, q);
graph.add_flow(reduction->rep_idx, snap->detection_idx, q);
graph.add_flow(reduction->reduced_points, display->reduced_points, q);
```

`extents` must set `max_clusters` and `max_features`. Python bindings are the HSDK 4.x trampoline and are **not built** on this branch.

## Contract conformance

Conforms on ports, dtypes, all four policies, built-in scorers, compile-time `ScoreFn`, required aligned `points`, exact-sized outputs, optional-output latching, and “`rep_idx` is a real cell.”

`feature_weights` uses the module's bounded rank-1 host config-Tensor representation for variable-length `[live]` vectors ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). The reduction primitive currently uses custom CUDA atomics and compact kernels; CUB/MatX may replace that implementation without changing the operator contract.

Medoid centroid sums use floating-point atomics, so accumulation order can perturb a near-tie. Once distances have identical encoded values, `medoid_min_row_kernel` deterministically chooses the lowest input row with `atomicMin`, matching the peak tie rule.
