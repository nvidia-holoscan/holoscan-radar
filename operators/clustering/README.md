# Clustering operators

GPU operators for the clustering stage of the radar chain. Clustering is
orthogonal to modality — it attaches at the detection / angle stage over a
caller-assembled feature matrix (see
[`docs/planning/WORKFLOWS.md`](../../docs/planning/WORKFLOWS.md)). Each operator
is templated on precision (`ClusterDBSCANOp<float>` / `<double>`); module-wide
port, batching, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's header
carries its full port/parameter reference.

Typical placement (angle-then-cluster; the `points` matrix is assembled by
[`FeatureAssemblyOp`](../detection/feature_assembly_op/README.md) in the
detection group). That bridge stays under `detection/` because it speaks CFAR
`detection_idx`; this group is the radar-agnostic clusterer.

```text
… → CFARDetectorOp → detection_idx → FeatureAssemblyOp → points → ClusterDBSCANOp → labels (+ sizes) → …
```

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| [`ClusterDBSCANOp<T>`](cluster_dbscan_op/README.md) | `points` (num_points, num_features) → `labels` (num_points,), optional `sizes` (num_clusters,) | Density clustering (DBSCAN) over an N-D feature matrix; per-axis `epsilon` neighborhood, `min_points` core threshold, optional `max_clusters` cap; noise labeled `-1` |

## Implementation status

`ClusterDBSCANOp` implements its locked contract in
[`docs/planning/OPERATOR_API.md`](../../docs/planning/OPERATOR_API.md). The
header is the full contract.

- **Implemented:** anisotropic per-axis `epsilon` (length-1 list `[ε]` →
  isotropic, or a `(num_features,)` vector giving the ellipsoidal neighborhood
  `Σ_i ((p_i − q_i) / ε_i)² ≤ 1`); `min_points` core threshold (default
  `2·num_features`); ids numbered by **descending cluster size** (cluster 0 =
  largest, tie-break ascending root); optional `max_clusters` cap (drops the
  **smallest** surplus clusters to noise with a warn-once); `labels`
  (noise = `-1`, row-aligned with `points`) and optional/latched, exact-sized
  `sizes` (non-increasing); all three parameters `[live]`-overridable per tick;
  float and double; empty `points` → empty `labels` / `sizes`. Unit
  tests cover the [`TESTING.md`](../../docs/planning/TESTING.md) clustering
  checks and a `perf_cluster_dbscan` characterization harness measures an
  inline live `min_points` override.
- **Scaling boundary:** the neighbor graph is a brute-force `O(N²)` sweep with
  iterative min-label connected components. It is correct and appropriate for
  sparse post-CFAR detection counts; a spatial index or union-find upgrade is
  profiling-driven future work. Descending-size numbering, the
  `max_clusters` surplus policy, and the `epsilon` length-1-list encoding are
  settled (STATUS.md L7 / L8 / L10).

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
- **Streams.** Work is enqueued on `context.cuda_stream()`; the
  `points` input is retained until the next tick as a stream-ordered lifetime
  guard.
