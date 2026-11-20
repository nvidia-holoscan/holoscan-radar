# `RankOneResidualScreenOp`

## Purpose

`RankOneResidualScreenOp` is a cheap routing stage for sparse angle pipelines. For each item, it projects every valid array snapshot onto the steering vector named by a row-aligned single-source baseline angle. Items whose normalized unexplained energy reaches `residual_threshold` are compacted for a more expensive multi-source estimator.

## Responsibilities

- Computes normalized energy left after the supplied one-source steering model for every valid item.
- Compacts threshold-crossing snapshots, source-row indices, and optional residual diagnostics without losing row alignment.
- Does not calibrate a source-count probability or perform the downstream multi-source estimate.

## When to use

Use between a baseline single-source angle estimate and an expensive multi-source estimator to route only items that poorly fit one steering vector. The baseline rows, geometry, convention, and timestamps must match the snapshots exactly. Do not treat the residual as a calibrated source count: noise, calibration error, multipath, and extended targets can also trigger it.

```text
snapshot + baseline_aoa [+ is_valid] [+ live array_geometry]
  → RankOneResidualScreenOp
  → snapshot_out + selected_idx [+ diagnostics]
```

Declarations and parameter defaults: [rank_one_residual_screen_op.hpp](rank_one_residual_screen_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels, num_snapshots)` | Required trigger/aligned input | Exact rank 3; positive trailing extents. |
| in | `baseline_aoa` | float32 / float64 | `(num_items, 2)` | Required trigger/aligned input | Single-source `(az, el)` or `(u, v)`, row-aligned with `snapshot`. |
| in | `is_valid` | bool or uint8 | `(num_items, num_snapshots)` | Optional Exact side input | Optional Exact mask; missing means every look is valid. |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | Optional Exact override; a miss uses Params for that tick. |
| out | `snapshot_out` | complex64 / complex128 | `(num_selected, num_channels, num_snapshots)` | Required output | Selected rows, copied in original order. Required. |
| out | `selected_idx` | int64 | `(num_selected,)` | Required output | Strictly ascending input-row indices. Required. |
| out | `is_valid_out` | uint8 | `(num_selected, num_snapshots)` | Optional `always` output | Compacted mask; all ones when the input mask was absent. Optional/always. |
| out | `residual_score` | float32 / float64 | `(num_items,)` | Optional `always` output | Row-aligned diagnostic score. Optional/always. |
| out | `selection_mask` | uint8 | `(num_items,)` | Optional `always` output | Row-aligned `score >= threshold` decision. Optional/always. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_items`, `max_channels`, `max_snapshots` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required unless every tick supplies a live value) | `[live]` Exact wire override | Positions and wavelength for the snapshot channel axis. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Boundary convention used by `baseline_aoa`. |
| `residual_threshold` | double | `0.10` | static | Select finite scores greater than or equal to this value; constrained to `[0, 1]`. |

## Algorithm

For input item `i`, valid-look set `V_i`, and steering vector `a_i` derived from `baseline_aoa[i]`, the operator computes

```text
                         Σ(k ∈ V_i) |a_iᴴ x_ik|²
residual[i] = 1 - -----------------------------------------------
                    (a_iᴴ a_i) Σ(k ∈ V_i) ||x_ik||²
```

Every steering-vector element has unit magnitude, so `a_iᴴa_i` is `num_channels`. The result is clamped to `[0, 1]` for floating-point roundoff. A row is selected when `residual[i] >= residual_threshold`; equality selects. Stable compaction copies the selected snapshot and validity rows and emits their original indices in ascending order.

Projection is performed independently per look and summed in power. The stage does not form a covariance, estimate its rank, or infer a source count. One look is legal and measures steering-model mismatch for one vector; multiple looks generally make routing more robust.

Rows with no valid looks, zero total energy, a non-finite baseline angle, or an invalid direction-cosine pair receive a NaN score, a zero decision, and no compacted row. An empty item axis emits correctly shaped zero-row outputs.

## Shape, dtype, and alignment rules

- Precision is fixed by the template instantiation: complex input and matching real baseline/score (`float` or `double`).
- Tensor inputs must be contiguous and CUDA-accessible. There is no opaque leading batch axis.
- `array_geometry.num_elements()` must equal `num_channels`; wavelength and positions determine the steering vector.
- `angle_convention` is static and must describe `baseline_aoa`. Presets, custom conventions, and `direction_cosine` are supported.
- Required data inputs use a zero-window Exact join. The mask and live geometry are Exact + SafeDefault sidebands.
- Every output preserves the triggering snapshot timestamp. `num_selected` may vary from zero through `num_items` each tick.

Rejected combinations and messages include:

| Condition | Message |
| --- | --- |
| snapshot not rank 3 or invalid trailing extent | `snapshot must have shape (num_items, num_channels, num_snapshots) with positive trailing extents` |
| baseline shape mismatch | `baseline_aoa must have shape (num_items, 2) matching snapshot` |
| captured bound exceeded | `snapshot exceeds MaxTensorExtents` |
| geometry/channel mismatch | `snapshot num_channels must match array_geometry.num_elements` |
| threshold non-finite or outside `[0, 1]` | `residual_threshold must be within [0, 1]` |
| invalid validity mask | shared `is_valid` dtype, shape, memory, or contiguity diagnostic |
| invalid geometry or convention | shared descriptor/convention diagnostic |
| missing stream | `RankOneResidualScreenOp: no CUDA stream` |

## Performance

Steady work is `O(num_items × num_channels × num_snapshots)` plus a linear selection scan and copies proportional to `num_selected`. A new geometry pays one position upload. Exact-size compaction incurs one synchronization per activation.

## Usage

```yaml
rank_one_residual_screen:
  angle_convention:
    preset: radar_boresight
  residual_threshold: 0.10
  # array_geometry may be authored here or supplied by a live producer
```

```cpp
auto screen = graph.op<RankOneResidualScreenOp<float>>(
    "rank_one_residual_screen",
    MaxTensorExtents{.max_items = max_detections,
                     .max_channels = max_virtual_channels,
                     .max_snapshots = max_snapshot_looks});
screen.set_params(screen_params);

graph.add_flow(compensation->snapshot, screen->snapshot, q);
graph.add_flow(beamscan->aoa_estimates, screen->baseline_aoa, q);
graph.add_flow(compensation->is_valid, screen->is_valid, q);
graph.add_flow(compensation->array_geometry, screen->array_geometry, q);
graph.add_flow(screen->snapshot_out, multi_source_estimator->snapshot, q);
```
