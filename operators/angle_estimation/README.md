# Angle-estimation operators

GPU operators for the sparse (and dense) angle paths. Module-wide port,
batching, and parameter-override conventions are in
[`operators/common/README.md`](../common/README.md); each operator header
carries its full port/parameter reference.

Typical sparse-path placement:

```text
… → FMCWDopplerProcessorOp → rd_maps (per-channel)
  → CFARDetectorOp (NCI over channel)
  → SnapshotGatherOp → snapshot → ⟨sparse angle estimator⟩ → aoa_estimates
```

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| `SnapshotGatherOp<T>` | `rd_maps` + `detection_idx` → `snapshot`, `is_valid` | Per-detection channel-vector gather from the RD neighborhood around each CFAR cell |
| `BeamscanEstimatorOp<T>` | `snapshot` [+ `is_valid`] → `aoa_estimates`, `angle_spectrum` | Conventional (Bartlett) sparse-path DOA on a scan grid |

## Implementation status

- **`SnapshotGatherOp` (checklist-complete — dpc):** RD-neighborhood gather for the sparse
  angle path; sticky `[live] snapshot_window`; static `doppler_edge` /
  `range_edge`; float/double complex `rd_maps`; unit tests (edge modes, batch
  dims, error paths, live override); `performance/cpp/perf_snapshot_gather.cpp`
  characterizes gather cost + the `[live] snapshot_window` override spike. HSDK
  4.3 bool transport caveat for `is_valid` in `snapshot_gather_op/OPEN_ITEMS.md`
  §1 (aligned with CFAR `detection_mask`).
- **`BeamscanEstimatorOp` (checklist-complete — dpc):** conventional sparse-path
  Bartlett DOA on a scan grid; lands `ArrayGeometry` + `AngleConvention`
  descriptors in `operators/common/`; sticky `[live]` overrides for geometry,
  scan grid, taper, and peak interpolation; float/double; unit tests (2-D DOA
  recovery on a planar URA, 3-D single-look input, direction-cosine masking,
  error paths, live `az_angles` override); `performance/cpp/perf_beamscan_estimator.cpp`
  characterizes scan cost + the `[live] az_angles` override spike. Open items in
  `beamscan_estimator_op/OPEN_ITEMS.md` (`custom` angle preset, bool `is_valid`
  transport caveat aligned with SnapshotGather).
