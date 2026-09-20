# `SubarraySelectionOp`

## Purpose

Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual array and re-emits a reduced `(snapshot, array_geometry)` pair so gridless estimators and holey-array spatial smoothing can run. It sits between `SnapshotGatherOp` and `RootMUSICEstimatorOp` / `ESPRITEstimatorOp` (or in front of MVDR/MUSIC `spatial_smoothing` on a holey virtual array). Selection is host-planned and cached; the per-tick work is a channel gather. A filled array is a no-op pass-through of the snapshot.

## Responsibilities

- Selects an explicit uniform subset or discovers a filled ULA/URA subset, then caches the channel-gather plan.
- Gathers snapshots and emits geometry that exactly describes the selected channel order.
- Does not synthesize missing elements, increase aperture, or choose an estimator.

## When to use

Use after snapshot gathering when a holey virtual array must be reduced to a filled ULA or URA for Root-MUSIC, ESPRIT, or spatial smoothing. It is also useful when an explicit stable channel subset is part of the application design. Do not use it merely to reorder a geometry already accepted by the chosen estimator, because discarded channels reduce aperture and sensitivity.

```text
SnapshotGatherOp.snapshot [+ is_valid]
  → SubarraySelectionOp → snapshot_out + array_geometry
  → RootMUSICEstimatorOp / ESPRITEstimatorOp
    (or MVDREstimatorOp / MUSICEstimatorOp with spatial_smoothing)
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, N_virtual)` or `(num_items, N_virtual, num_snapshots)` | Required trigger/aligned input | Holey virtual-array snapshot from `SnapshotGatherOp`. `N_virtual` must match `array_geometry.num_elements`. |
| in | `is_valid` | bool or uint8 | `(num_items, num_snapshots)` | Optional Exact side input | Optional Exact sideband. Forwarded unchanged when present. |
| out | `snapshot_out` | complex64 / complex128 | `(num_items, L)` or `(num_items, L, num_snapshots)` | Required output | Selected-channel snapshot (`L` = sub-array size). unique in/out name (logical `snapshot`). |
| out (cond.) | `array_geometry` | `WireArrayGeometry` | descriptor-Tensor | Optional latched output | Optional (`latched`) reduced logical geometry. When wired, emitted on the same tick as `snapshot_out`; an unwired first emit latches off future packing, allocation, and publication |
| out (cond.) | `is_valid_out` | same as `is_valid` | `(num_items, num_snapshots)` | Optional latched output | Emitted before `snapshot_out` only when `is_valid` arrived. Optional (`latched`): the first emit to an unwired port warns once, after which the forwarding copy is skipped. unique in/out name. |
| live | `array_geometry_in` | `WireArrayGeometry` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. `_in` suffix — the output owns the `array_geometry` name. |
| live | `subarray` | int64 config Tensor | `(L,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; extent bounded by `max_subarray_channels` (else `max_virtual_channels` / `max_channels`, else the tick's channel count). |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | Omit all three fields for zero-span output bounds, or provide positive `max_items`, `max_subarray_channels`, and `max_snapshots` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override on `array_geometry_in` | Holey virtual positions + wavelength. `num_elements` must match `snapshot` channels. Cost: Rebuilds the host plan and H2D channel-index table; logs INFO once per new key. |
| `subarray` | `std::vector<int64_t>` | `{}` (auto) | `[live]` Exact host config Tensor, extent bounded by `max_subarray_channels` | Explicit channel indices; empty → largest filled ULA/URA. Re-emitted in lattice order. Cost: Same rebuild. |
| `min_subarray_size` | int64 | `2` | static | Auto (and explicit) selection errors below this many channels. Cost: Rebuilds the plan (part of the cache key). |

Contract: [`OPERATOR_API.md` § `SubarraySelectionOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md)

## Algorithm

The operator shell (`subarray_selection_op.cu`) receives and classifies the snapshot, resolves the two Exact overrides into one tick-local `SubarraySelectionStageConfig`, derives named axes in `SubarraySelectionStagePlan`, validates the optional mask, prepares the cached selection, then prepares and publishes outputs. `prepare_selection` rebuilds `SubarraySelection` via `radar_common/dsp/subarray_selection.cpp` only when geometry, `subarray`, or `min_subarray_size` changes. Per tick, selected channels are gathered with the S1 helper `run_spatial_smoothing_gather` at `num_subarrays = 1`, `forward_backward = false` (no mask rewrite). An identity selection Device-to-Device copies the input snapshot.

**Auto** (`subarray` empty). Infer the (possibly holey) integer lattice from element positions: unique coordinates per Cartesian axis, unit spacing = real-GCD of gaps (`fgcd` / `1e-6 * span` tolerance). Occupancy is a row-major grid of input channel indices (`-1` = hole). Then:

- 1-D: longest run of filled cells.
- 2-D: maximal all-filled rectangle (histogram / largest-rectangle sweep). Objective is **max element count**; ties break by larger first-aperture-axis extent (azimuth)..

**Explicit** (`subarray` non-empty). Validate range and uniqueness, rebuild a lattice from the chosen positions, require a filled ULA/URA (occupancy cell count equals the index count — no holes), and re-emit the indices in lattice row-major order.

`L < min_subarray_size` is an error. Pass-through requires the selected indices to be exactly `[0, 1, …, N_virtual-1]`; equal cardinality alone is insufficient because a filled geometry may arrive in non-lattice channel order. Reduced/reordered geometry copies the selected positions, keeps `wavelength`, and sets `geometry_type` to `kUla` (1-D) or `kUra` (2-D).

The `array_geometry` output uses `WireArrayGeometry`. Its pageable-host
descriptor-Tensor member keeps the shared bridge's internal packing:

```text
shape (L + 1, 3) float64
rows [0, L):  element_positions (x, y, z) meters
row  L:       [wavelength, geometry_type_code, 0]
              code 0=ULA, 1=URA, 2=UCA, 3=CONFORMAL; NaN if unset
```

On a reduced/reordered cache miss the operator logs `SubarraySelectionOp: selected L/N channels (uniform sub-array)` (INFO). 3-D apertures, off-lattice positions, colocated-only arrays, and sum-set or non-contiguous “Harnessing Holes” selections are not supported.

## Shape and dtype rules

- `snapshot_out` keeps `num_items` and `num_snapshots`; the channel axis becomes `L` (or stays `N_virtual` on pass-through). A 2-D input stays 2-D; a 3-D input stays 3-D.
- `array_geometry.packed_geometry` is rank-2 `(L + 1, 3)` float64, independent of snapshot precision.
- `array_geometry` is published only while its independent optional-output latch remains active. A snapshot-only consumer may instead own a matching static reduced geometry; live geometry/subarray changes then remain the caller's coherence responsibility.
- `is_valid_out` is a byte-identical copy of `is_valid` when that input arrived and the port has not latched off; otherwise it is not published.
- No opaque leading batch. `snapshot` rank must be 2 or 3.
- Precision is the template instantiation. `is_valid` accepts `bool` or `uint8` (8-bit, 1 lane).
- Inputs must be CUDA-accessible and compact row-major. Outputs are device tensors.
- A geometry / `subarray` change (override on `array_geometry_in` / `subarray`, or `set_params`) misses the plan cache key and rebuilds on that tick. Downstream re-infers `L`.

Rejected combinations and the operator's own messages (selection-helper messages are forwarded as `kInvalidArgument`):

| Condition | Message |
| --- | --- |
| `snapshot` rank not 2 or 3 | `SubarraySelectionOp: snapshot must have shape (num_items, num_channels) or (num_items, num_channels, num_snapshots)` |
| `num_channels ≠ array_geometry.num_elements` | `SubarraySelectionOp: snapshot num_channels must match array_geometry.num_elements` |
| `is_valid` not bool/uint8 | `SubarraySelectionOp: is_valid must have dtype bool or uint8` |
| `is_valid` shape mismatch | `SubarraySelectionOp: is_valid must have shape (num_items, num_snapshots) matching snapshot` |
| empty / non-positive / non-finite geometry | `ArrayGeometry: num_elements must be positive` / `ArrayGeometry: wavelength must be a positive finite scalar` / `ArrayGeometry: element_positions length must be a multiple of 3` |
| all elements colocated | `subarray_selection: array_geometry has no aperture (all elements colocated)` |
| 3-D aperture | `subarray_selection: 3-D element apertures are not supported` |
| degenerate lattice / off-grid / duplicate positions | `subarray_selection: degenerate lattice spacing` / `… element off the inferred lattice (array_geometry is not on a consistent uniform grid)` / `… duplicate element positions` |
| explicit index out of range / duplicate | `subarray_selection: explicit channel index out of range` / `… duplicate explicit channel index` |
| explicit set is holey | `subarray_selection: explicit channel indices do not form a filled ULA/URA (holey)` |
| auto or explicit `L < min_subarray_size` | `subarray_selection: no filled uniform sub-array of the requested minimum size exists in this geometry` |
| selected `L` exceeds the captured pool bound | `SubarraySelectionOp: selected channels exceed MaxTensorExtents.max_subarray_channels` |
| non-contiguous / pageable-host / dtype mismatch / no stream | same `require_*` / `no CUDA compute stream` wording as other operators, prefixed `SubarraySelectionOp` |

## Performance

Cold / first new configuration: host `build_subarray_selection` + H2D of an `int32` channel-index table of length `L`, plus the INFO log when selection is non-identity. Steady: one S1 gather (`P = 1`, no forward-backward) or a D2D pass-through, plus host wire preparation while geometry remains consumed. The replace-in-place cache key uses exact `ArrayGeometry`, explicit-index vector, and `min_subarray_size` equality.

`performance/cpp/perf_subarray_selection.cpp` reports cold and steady Graph timings over an item/channel/snapshot ladder on a holey ULA. Cold includes plan construction and the index-table upload; steady includes gather plus schema-aware geometry publication. The smoke has no timing threshold.

## Usage

No example app currently instantiates this operator. YAML / `set_params` shape:

```yaml
subarray_selection:
  # array_geometry: set in C++ from the virtual array (or YAML if authored)
  subarray: []          # empty → auto
  min_subarray_size: 2
```

```cpp
auto sel = graph.op<SubarraySelectionOp<float>>(
    "subarray_selection",
    MaxTensorExtents{/* max_items, max_subarray_channels, max_snapshots */});
sel.set_params(cfg.subarray_params);

graph.add_flow(gather->snapshot, sel->snapshot, q);
graph.add_flow(sel->snapshot_out, estimator->snapshot, q);
graph.add_flow(sel->array_geometry, estimator->array_geometry, q);
// bind is_valid_out even if is_valid is unwired.
```
