# Clustering operators

GPU operators for the clustering stage of the radar chain. Clustering is
orthogonal to modality — it attaches at the detection / angle stage over a
caller-assembled feature matrix (see
[`docs/planning/WORKFLOWS.md`](../../docs/planning/WORKFLOWS.md)). Each operator
is templated on precision (`ClusterDBSCANOp<float>` / `<double>`); module-wide
port, batching, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's header
carries its full port/parameter reference.

Typical placement (angle-then-cluster; the `points` matrix is assembled by the
detection-group `FeatureAssemblyOp` bridge):

```text
… → CFARDetectorOp → detection_idx → FeatureAssemblyOp → points → ClusterDBSCANOp → labels (+ sizes) → …
```

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| `ClusterDBSCANOp<T>` | `points` (num_points, num_features) → `labels` (num_points,), `sizes` (num_clusters,) | Density clustering (DBSCAN) over an N-D feature matrix; per-axis `epsilon` neighborhood, `min_points` core threshold, optional `max_clusters` cap; noise labeled `-1` |

## Implementation status

`ClusterDBSCANOp` is landing against its locked contract in
[`docs/planning/OPERATOR_API.md`](../../docs/planning/OPERATOR_API.md). The
header is the full contract.

- **Implemented:** anisotropic per-axis `epsilon` (scalar → isotropic, or a
  `(num_features,)` vector giving the ellipsoidal neighborhood
  `Σ_i ((p_i − q_i) / ε_i)² ≤ 1`); `min_points` core threshold (default
  `2·num_features`); ids numbered by **descending cluster size** (cluster 0 =
  largest, tie-break ascending root); optional `max_clusters` cap (drops the
  **smallest** surplus clusters to noise with a warn-once); `labels`
  (noise = `-1`, row-aligned with `points`) and per-tick reallocated `sizes`
  (non-increasing); all three parameters `[live]`-overridable with sticky
  semantics; float and double; empty `points` → empty `labels` / `sizes`. Unit
  tests cover the [`TESTING.md`](../../docs/planning/TESTING.md) clustering
  checks and a `perf_cluster_dbscan` characterization harness measures the
  live-`epsilon` override.
- **Staged follow-ups (not contract changes):** the neighbor graph is a
  brute-force `O(N²)` sweep with iterative min-label connected components —
  correct and appropriate for the sparse post-CFAR detection counts this stage
  sees, but a uniform-grid spatial index / atomic union-find upgrade for large
  point clouds is tracked in
  [`cluster_dbscan_op/OPEN_ITEMS.md`](cluster_dbscan_op/OPEN_ITEMS.md), along
  with a deferred `cluster_order` parameter (the descending-size numbering /
  `max_clusters` surplus policy is now settled — STATUS.md L7 / L10) and the
  `epsilon` scalar-encoding ergonomics.

## Shared kernel conventions

- **Stage function.** The operator's math is a stage in its own directory
  (`cluster_dbscan_stage.hpp`, a CUDA-TU-only header); `compute()` is a thin
  shell that resolves overrides, validates ports, allocates outputs, and drives
  the DBSCAN kernels on the Holoscan CUDA stream.
- **Custom CUDA, not MatX.** DBSCAN is an irregular neighbor-graph /
  connected-components algorithm with no MatX expression, so the neighbor,
  core-marking, label-propagation, border-assignment, relabel, and histogram
  steps are custom kernels; CUB provides the exclusive-scan used to compact
  cluster roots into dense `[0, num_clusters)` ids.
- **Streams.** Work is enqueued on the stream from `receive_cuda_stream()`; the
  `points` input is retained until the next tick as a stream-ordered lifetime
  guard.
