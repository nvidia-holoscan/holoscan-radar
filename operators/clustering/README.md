# Clustering operators

GPU clustering for detections represented as an ordinary feature matrix. The
clustering stage is independent of radar modality; `FeatureAssemblyOp` converts
CFAR indices and optional physical axes into the matrix it consumes.

```text
CFAR -> FeatureAssemblyOp -> points -> ClusterDBSCANOp -> labels
```

## Operators

| Operator | Input → output | Function |
| --- | --- | --- |
| [`ClusterDBSCANOp<T>`](cluster_dbscan_op/README.md) | `points (N,F)` → `labels (N)` and optional `sizes` | Anisotropic DBSCAN with per-feature scales, deterministic size-ranked cluster ids, and `-1` noise labels |

Use this operator for sparse post-detection point sets. Its neighbor search is
quadratic in the number of points, so it is not intended for dense image-sized
clustering. `epsilon` may be one isotropic value or one value per feature; a
zero value excludes that feature from distance calculations.

See [`FeatureAssemblyOp`](../detection/feature_assembly_op/README.md) for
constructing `points` and
[`DetectionReductionOp`](../detection/detection_reduction_op/README.md) for
selecting or synthesizing one representative per cluster. Shared precision,
tensor, and live-override conventions are in
[`operators/common/README.md`](../common/README.md).
