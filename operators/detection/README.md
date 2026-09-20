# Detection operators

GPU operators that convert range-Doppler or range-Doppler-angle data into
compact detections and downstream feature rows.

```text
rd_maps -> CFARDetectorOp -> detection_idx -> FeatureAssemblyOp -> points
labels + detection_idx + points -> DetectionReductionOp -> representatives
```

## Operators

| Operator | Input → output | Function |
| --- | --- | --- |
| [`CFARDetectorOp<T>`](cfar_detector_op/README.md) | `rd_maps` → dense CFAR products and compact `detection_idx` | CA, GO, SO, or OS 2-D CFAR over Doppler/range, with optional non-coherent integration over named leading axes |
| [`FeatureAssemblyOp<T>`](feature_assembly_op/README.md) | indices plus selected axes/scores/angles → `points` | Builds a caller-selected clustering feature matrix from bin or physical coordinates |
| [`DetectionReductionOp<T>`](detection_reduction_op/README.md) | labels, indices, and points → representatives | Selects `first`, `peak`, or `medoid` cells, or computes a weighted mean |
| [`MultiSourceExpansionOp<T>`](multi_source_expansion_op/README.md) | detections plus baseline and selected-estimator angles → expanded rows | Merges selective multi-source estimates with deterministic baseline fallback |

## CFAR data model

Complex input is converted to square-law power; real input is treated as an
already formed power map. CFAR always operates on the trailing Doppler/range
plane. `nci_axes` may integrate named leading axes before detection, and any
preserved leading coordinates appear before `(doppler_bin, range_bin)` in each
`detection_idx` row.

The compact index output is appropriate for sparse downstream processing.
Dense score, mask, noise, and power outputs are independently optional; wire
only the products the graph needs. See the operator README for threshold
calibration, edge behavior, peak suppression, and output timing.

Shared precision, tensor, and live-override conventions are in
[`operators/common/README.md`](../common/README.md). Deferred feature work is
tracked in the repository [`OPEN_ITEMS.md`](../../OPEN_ITEMS.md).
