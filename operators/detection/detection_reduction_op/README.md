# `DetectionReductionOp`

## Purpose

Reduces labeled detections to one **representative detection** and, optionally, one reduced feature row per non-noise cluster. `rep_idx` always selects an actual row of `detection_idx`; synthetic weighted means are confined to `reduced_points`.

## Responsibilities

- Validates label, detection, point, and optional score alignment, then applies the selected representative policy per cluster.
- Always emits an actual detection row in `rep_idx` and emits aggregate feature rows only through optional `reduced_points`.
- Does not create labels, preserve noise rows, or reinterpret feature units.

## When to use

Use after `ClusterDBSCANOp` to collapse each non-noise cluster into one representative detection and, optionally, one reduced feature row. Select `peak`, `medoid`, `first`, or `weighted_mean` according to whether the application needs a real member or an aggregate. Do not use it before labels exist or when every cluster member must remain available downstream.

```text
labels + detection_idx + points
  → DetectionReductionOp → rep_idx
                         → reduced_points (optional)
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `labels` | int64 | `(num_points,)` | Required trigger/aligned input | Cluster id per detection; noise = `-1` (no representative). Row-aligned with `detection_idx` |
| in | `detection_idx` | int64 | `(num_points, 2 [+ n_preserved])` | Required trigger/aligned input | Detection coordinates; row-aligned with `labels` |
| in | `points` | float32 / float64 | `(num_points, num_features)` | Required trigger/aligned input | Row-aligned reduction features; may augment the clustered matrix. Always required |
| out | `rep_idx` | int64 | `(num_clusters, 2 [+ n_preserved])` | Required output | One `detection_idx` row per non-noise cluster; row count *is* `num_clusters` |
| out (optional, latched) | `reduced_points` | float32 / float64 | `(num_clusters, num_features)` | Optional latched output | Optional/latched; selected input rows or synthetic weighted means |
| live | `representative` | enum | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; the `contract()` entry is unconditional and wiring enables it |
| live | `score_method` | enum | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; the `contract()` entry is unconditional and wiring enables it |
| live | `feature_weights` | float64 config Tensor | `(num_features,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | Exact override; the `contract()` entry is unconditional and wiring enables it |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_clusters`, `max_features` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `representative` | enum {peak, medoid, first, weighted_mean} | `peak` | `[live]` Exact | Member or aggregate policy Cost: Selects a different launch. |
| `score_method` | enum {weighted_sum, weighted_l2, weighted_max, custom} | `weighted_sum` | `[live]` Exact | Peak score or weighted-mean row-weight collapse Cost: Free. |
| `feature_weights` | float64 vector, length `num_features` | none | `[live]` Exact host config Tensor | Peak collapse, medoid scaling, or weighted-mean row weights; empty means uniform mean only for `weighted_mean` Cost: H2D when used. |
| `score_fn` | compile-time device functor | none (`NoCustomScore`) | static (template argument) | `T operator()(const T* row, int64_t num_features) const`; only when `score_method = custom` |

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

## Performance

**Cold.** First-tick device temps and weight H2D.

**Steady.** Find-max-label + policy kernels + compact; two scalar syncs; per-tick scratch allocs. Final representative/point compaction uses a 256-thread integer prefix scan over label tiles, preserving ascending label order. Selection arithmetic and floating-point accumulation are unchanged. Weight vector is re-uploaded whenever peak/medoid needs it.

**Live rebuilds.** `representative` / `score_method` are free policy switches (no cached kernel). A `feature_weights` override re-uploads the weight vector on the tick it arrives.

A paired Thor HSDK 5 check (5 warmups, 50 measured frames, three fresh pairs) measured large float/double paired median reductions of 111.181µs/37.848% and 109.894µs/37.217%. All three target pairs and all selected endpoint rules passed. All 150 fresh compute samples per endpoint, including outliers, summed to lower time at every selected endpoint (4.52–37.67% reductions). Both builds passed 43 native tests, including a three-tile sparse-label oracle, and nine-case both-precision input-derived coupled output validation.

These are scoped results with observed costs: individual small/optional processes were 6–9% (about 13–18µs) slower, including an optional-double slowdown in both screening and fresh checks in opposite execution orders. Screen optional-double also had a 2.113ms frame; fresh float setup increased about 0.227ms on the large case and 0.240ms on the small case. The larger target benefit was accepted with those costs; individual-process latency, tails, startup and live events are not guaranteed to improve.

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

`extents` must set `max_clusters` and `max_features`.
