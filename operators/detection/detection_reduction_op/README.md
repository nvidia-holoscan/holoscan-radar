# `DetectionReductionOp`

Reduces labeled detections to one **representative detection** per non-noise cluster. Every policy (`peak` / `medoid` / `first`) selects an actual row of `detection_idx`; a synthetic centroid is never emitted. The output is therefore an exact-sized index table that callers can use wherever one real detection per cluster is needed.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `labels` | int64 | `(num_points,)` | Cluster id per detection; noise = `-1` (no representative). Row-aligned with `detection_idx` |
| in | `detection_idx` | int64 | `(num_points, 2 [+ n_preserved])` | Detection coordinates; row-aligned with `labels` |
| in | `points` | float32 / float64 | `(num_points, num_features)` | Same matrix clustered on. Always required; `first` ignores its values |
| out | `rep_idx` | int64 | `(num_clusters, 2 [+ n_preserved])` | One `detection_idx` row per non-noise cluster; row count *is* `num_clusters` |
| live | `representative` | enum | scalar | Exact override; the `contract()` entry is unconditional and wiring enables it |
| live | `score_method` | enum | scalar | Exact override; the `contract()` entry is unconditional and wiring enables it |
| live | `feature_weights` | float64 config Tensor | `(num_features,)`, rank-1 pageable host | Exact override; the `contract()` entry is unconditional and wiring enables it |

```text
labels + detection_idx + points
  → DetectionReductionOp → rep_idx
```

Contract: [`OPERATOR_API.md` § `DetectionReductionOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/detection/README.md`](../README.md)

## Algorithm

Thin operator over the public `radar_common/reduce/` group-by-label primitive (`detection_reduction_stage.hpp` only includes those launch wrappers). `compute()` resolves the enabled `[live]` enum overrides, validates ports, finds `max_label`, runs the selected policy, then compact-gathers occupied slots into exact-sized `rep_idx`.

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

3. **Emit.** `compact_and_gather_kernel` walks labels `0 … max_label` and copies each occupied `detection_idx` row into `rep_idx` in **ascending label-id order** (which, after DBSCAN’s descending-size numbering, is largest-cluster-first). Noise yields no row. All-noise or empty input → `(0, idx_cols)`.

## Shape and dtype rules

- **`rep_idx`.** Same column count as `detection_idx`. Rows = occupied non-noise labels (`≤ max_label + 1`, after skipping gaps).
- **Alignment.** `labels` rank 1; `detection_idx` rank 2 with `≥ 2` columns; `labels.shape[0] == detection_idx.shape[0]`. `points` is `(num_points, num_features)` with the same row count.
- **Precision.** Template `T ∈ {float, double}` for `points`. Index ports are int64.
- **Extents.** Ctor requires `max_clusters` (pool rows). The `rep_idx` pool span is `(max_clusters, 10) · sizeof(int64)`, covering the two cell coordinates plus the module's eight preserved-index columns. Wider `detection_idx` inputs are rejected explicitly; too many output rows fail the bounded loan.
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
  | `max_label + 1 > 2^20` | `DetectionReductionOp: max cluster label exceeds supported range (expected DBSCAN-style dense labels)` |
  | Missing ctor extent | `DetectionReductionOp: MaxTensorExtents.max_clusters must be a positive value (required for Tensor pool sizing)` |
  | A `feature_weights` override arrives without `max_features` | `DetectionReductionOp: MaxTensorExtents.max_features must be positive to accept a feature_weights override` |
  | No CUDA stream | `DetectionReductionOp: no CUDA compute stream` |

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `representative` | enum {peak, medoid, first} | `peak` | `[live]` Exact | Free (selects a different launch) | Which real member to keep |
| `score_method` | enum {weighted_sum, weighted_l2, weighted_max, custom} | `weighted_sum` | `[live]` Exact | Free; used only by `peak` | How weighted columns collapse to the peak score |
| `feature_weights` | float64 vector, length `num_features` | none (required for `peak` non-custom and `medoid`) | `[live]` Exact host config Tensor | H2D of the weight vector each tick that needs it | `w` for peak collapse / medoid scaled distance |
| `score_fn` | compile-time device functor | none (`NoCustomScore`) | static (template argument) | — | `T operator()(const T* row, int64_t num_features) const`; only when `score_method = custom` |

`feature_weights` has no universal default because features may use mixed units. One-hot `w` reduces `weighted_sum` to “max of a single column.” When labels were produced with a weighted feature-space metric, matching those weights keeps medoid selection consistent with that metric.

`points` is always part of the zero-window aligned trigger. Consequently every live `representative` value is legal without changing required input membership; `first` simply ignores the aligned matrix values.

## SDK implementation notes

- **Construction.** `DetectionReductionOp(const MaxTensorExtents& extents)`. Required extent: `max_clusters`. `max_features` is required whenever an application wires the `feature_weights` override — it bounds the payload, and a sample arriving without it is rejected with the message above. `rep_idx` pool: `max_rank = 2`, span for `(max_clusters, 10)` int64 rows.
- **Triggering.** `OnAligned{labels, detection_idx, points}` always selects all three required data inputs at zero window. The three `[live]` overrides are unconditional non-members with `Exact + SafeDefault`; wiring is the application's only override statement.
- **Ports and queues.** `labels`, `detection_idx`, and `points`: `queue_depth(2)`, device. Live enums and the `feature_weights` config Tensor (rank-1 pageable host): `queue_depth(2)`.
- **Streams and lifetime.** `context.cuda_stream()`. Inputs retained until the next tick. The hot path **does** synchronize twice: max-label D2H, then cluster-count D2H, so the contract-required `rep_idx` loan can be exact-sized.
- **Emission.** One required output through `emit_device_tensor`; its fill launches `compact_and_gather` after the exact row count is known. Empty scenes use the same helper with a no-op fill.
- **Scratch.** Per-tick `make_device_tensor` temps (`best_rows`, score bits, centroid sums, …) are accepted operator-owned scratch. Their slot count follows runtime label ids, so ctor extents do not provide a safe fixed capacity when sparse label gaps are admitted.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.DetectionReductionParams` version 1. Fields: `representative`, `score_method`, `feature_weights`. There are no latch or override-wiring flags.
- **Precision.** Explicit instantiations: `DetectionReductionOp<float>`, `DetectionReductionOp<double>` (default `NoCustomScore`). Custom `ScoreFn` specializations are caller-provided in a CUDA TU.
- **Live override payloads.** Exact enum `Input<>` ports for `representative` / `score_method` and a rank-1 pageable-host float64 config Tensor for `feature_weights` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). All three entries are unconditional and resolved into one tick-local stage config; the Tensor payload is validated on arrival (rank, host memory, contiguity, dtype, `max_features` extent).

## Consumer lifetime notes

`rep_idx` is a fresh pool-loaned device buffer each tick. A consumer may retain it.

## Performance

**Cold.** First-tick device temps and weight H2D.

**Steady.** Find-max-label + policy kernels + compact; two scalar syncs; per-tick scratch allocs. Weight vector is re-uploaded whenever peak/medoid needs it.

**Live rebuilds.** `representative` / `score_method` are free policy switches (no cached kernel). A `feature_weights` override re-uploads the weight vector on the tick it arrives.

Harness: `performance/cpp/perf_detection_reduction.cpp` (Graph triple-in). Axes: detection count, cluster count, policy {first, peak, medoid}. Live-override timing is not measured. No measured numbers are recorded in-repo.

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
| `ExcludesNoiseAndEmitsOnePerCluster` / `SparseLabelGapsSkipped` | One real cell per occupied label; noise excluded; gaps skipped |
| `WidenedDetectionIdxCols` | Preserved-batch columns copied through |
| `RejectsOpaqueLeadingBatch` / `RejectsTooManyDetectionIdxCols` | Fail-closed rank and preserved-index bounds |
| `EmptyInputEmitsZeroClusters` / `AllNoiseEmitsZeroClusters` | `(0, idx_cols)` |
| `RejectsMismatchedFeatureWeights` / `RejectsCustomWithoutScoreFn` | Weight length and default-`ScoreFn` reject |
| `RepresentativeConfigurableFromParams` / `ScoreMethodConfigurableFromParams` / `FeatureWeightsConfigurableFromParams` | Params field names |
| `FloatDoubleAgreeOnPeak` | Both precisions |
| `LiveFeatureWeights*` (Exact match, skewed capture, revert-to-Params, malformed, bound overflow) | Config-Tensor override: Exact join, revert-to-Params, payload rejects |
| `RejectsMissingRequiredPoints` | The graph rejects an unwired required `points` input |
| `LiveRepresentativeAppliesForOneTick` / `LiveScoreMethodAppliesForOneTick` | Exact enum overrides apply for one tick and then revert to Params |
| `test_detection_reduction_custom.cu` | Graph coverage for a non-default `ScoreFn` instantiated in a CUDA TU |

`TESTING.md` “every `rep_idx` is a subset of `detection_idx`” is covered by the policy tests (they assert the expected cell). Python tests exist but are not built.

```bash
ctest --test-dir build -R test_detection_reduction --output-on-failure
```

## Usage

```yaml
reduction:
  representative: medoid
  score_method: weighted_sum
  feature_weights: [0.3333, 0.5]
```

```cpp
auto reduction = graph.op<DetectionReductionOp<float>>(
    "detection_reduction", extents);
reduction.set_params(from_config<DetectionReductionParams>("reduction"));

graph.add_flow(cluster->labels, reduction->labels, q);
graph.add_flow(cfar->detection_idx, reduction->detection_idx, q);
graph.add_flow(feat->points, reduction->points, q);
graph.add_flow(reduction->rep_idx, snap->detection_idx, q);
```

`extents` must set `max_clusters`. Python bindings are the HSDK 4.x trampoline and are **not built** on this branch.

## Contract conformance

Conforms on ports, dtypes, all three policies, built-in peak scorers, compile-time `ScoreFn`, required aligned `points`, exact-sized `rep_idx`, and “representative is a real cell.”

`feature_weights` uses the module's bounded rank-1 host config-Tensor representation for variable-length `[live]` vectors ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). The reduction primitive currently uses custom CUDA atomics and compact kernels; CUB/MatX may replace that implementation without changing the operator contract.

Medoid centroid sums use floating-point atomics, so accumulation order can perturb a near-tie. Once distances have identical encoded values, `medoid_min_row_kernel` deterministically chooses the lowest input row with `atomicMin`, matching the peak tie rule.
