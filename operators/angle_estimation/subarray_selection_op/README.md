# `SubarraySelectionOp`

Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual array and re-emits a reduced `(snapshot, array_geometry)` pair so gridless estimators and holey-array spatial smoothing can run. It sits between `SnapshotGatherOp` and `RootMUSICEstimatorOp` / `ESPRITEstimatorOp` (or in front of MVDR/MUSIC `spatial_smoothing` on a holey virtual array). Selection is host-planned and cached; the per-tick work is a channel gather. A filled array is a no-op pass-through of the snapshot.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, N_virtual)` or `(num_items, N_virtual, num_snapshots)` | Holey virtual-array snapshot from `SnapshotGatherOp`. `N_virtual` must match `array_geometry.num_elements`. |
| in | `is_valid` | bool or uint8 | `(num_items, num_snapshots)` | Optional. Forwarded unchanged when present. [EA HSDK5 ONLY] omitted from `contract()` — a contract entry cannot tolerate an unwired port ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). |
| out | `snapshot_out` | complex64 / complex128 | `(num_items, L)` or `(num_items, L, num_snapshots)` | Selected-channel snapshot (`L` = sub-array size). [EA HSDK5 ONLY] unique in/out name (logical `snapshot`). |
| out (cond.) | `array_geometry` | `WireArrayGeometry` | descriptor-Tensor | Optional (`latched`) reduced logical geometry. When wired, emitted on the same tick as `snapshot_out`; an unwired first emit latches off future packing, allocation, and publication |
| out (cond.) | `is_valid_out` | same as `is_valid` | `(num_items, num_snapshots)` | Emitted only when `is_valid` arrived. Optional (`latched`): the first emit to an unwired port warns once, after which the forwarding copy is skipped. [EA HSDK5 ONLY] unique in/out name. |
| live | `array_geometry_in` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable. [EA HSDK5 ONLY] `_in` suffix — the output owns the `array_geometry` name (§2). |
| live | `subarray` | int64 config Tensor | `(L,)`, rank-1 pageable host | `[live]` Exact override; extent bounded by `max_subarray_channels` (else `max_virtual_channels` / `max_channels`, else the tick's channel count). |

```text
SnapshotGatherOp.snapshot [+ is_valid]
  → SubarraySelectionOp → snapshot_out + array_geometry
  → RootMUSICEstimatorOp / ESPRITEstimatorOp
    (or MVDREstimatorOp / MUSICEstimatorOp with spatial_smoothing)
```

Contract: [`OPERATOR_API.md` § `SubarraySelectionOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md)

## Algorithm

The operator shell (`subarray_selection_op.cu`) receives and classifies the snapshot, resolves the two Exact overrides into one tick-local `SubarraySelectionStageConfig`, derives named axes in `SubarraySelectionStagePlan`, validates the optional mask, prepares the cached selection, then prepares and publishes outputs. `prepare_selection` rebuilds `SubarraySelection` via `radar_common/dsp/subarray_selection.cpp` only when geometry, `subarray`, or `min_subarray_size` changes. Per tick, selected channels are gathered with the S1 helper `run_spatial_smoothing_gather` at `num_subarrays = 1`, `forward_backward = false` (no mask rewrite). An identity selection Device-to-Device copies the input snapshot.

**Auto** (`subarray` empty). Infer the (possibly holey) integer lattice from element positions: unique coordinates per Cartesian axis, unit spacing = real-GCD of gaps (`fgcd` / `1e-6 * span` tolerance). Occupancy is a row-major grid of input channel indices (`-1` = hole). Then:

- 1-D: longest run of filled cells.
- 2-D: maximal all-filled rectangle (histogram / largest-rectangle sweep). Objective is **max element count**; ties break by larger first-aperture-axis extent (azimuth). STATUS.md L14.

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

On a reduced/reordered cache miss the operator logs `SubarraySelectionOp: selected L/N channels (uniform sub-array)` (INFO). 3-D apertures, off-lattice positions, colocated-only arrays, and sum-set / non-contiguous “Harnessing Holes” selections are rejected or deferred (STATUS.md L14 fork 3).

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

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required) | `[live]` Exact wire-schema override on `array_geometry_in` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9) | Rebuilds the host plan and H2D channel-index table; logs INFO once per new key | Holey virtual positions + wavelength. `num_elements` must match `snapshot` channels. |
| `subarray` | `std::vector<int64_t>` | `{}` (auto) | `[live]` Exact host config Tensor, extent bounded by `max_subarray_channels` (§9) | Same rebuild | Explicit channel indices; empty → largest filled ULA/URA. Re-emitted in lattice order. |
| `min_subarray_size` | int64 | `2` | static | Rebuilds the plan (part of the cache key) | Auto (and explicit) selection errors below this many channels. |

Contract marks `array_geometry` and `subarray` **[live]**, and each has an override port whose match applies for one tick before the effective value reverts to Params. The logical Params types stay `ArrayGeometry` / `std::vector<int64_t>` — the wire-object and host-Tensor payloads are a port detail (HSDK5_PORT.md §9). YAML / `from_config` / `set_params` use the table names.

## SDK implementation notes

- **Construction.** `SubarraySelectionOp(const MaxTensorExtents&)`. Runtime state is created in the constructor. If none of `max_items`, `max_subarray_channels`, `max_snapshots` is set, pool spans stay 0. If any is set, all three are required. `snapshot_out` is sized as a rank-3 complex cube `{max_items, max_subarray_channels, max_snapshots}`; the host wire geometry member as `(max_subarray_channels + 1) × 3` float64; `is_valid_out` as rank-2 `max_items × max_snapshots` bytes. The private int32 channel-index table is allocated once at `max_subarray_channels`.
- **Triggering.** `contract()` is `OnEach{snapshot}` plus one unconditional `Exact{}` + `SafeDefault{}` entry per `[live]` override (`array_geometry_in`, `subarray`); the application enables an override by `add_flow`-ing its port. `is_valid` is registered in `setup()` but stays out of the contract, and a wired-but-undeclared port fails the second activation with `RUNTIME_OPERATOR_REPLAY_MISMATCH`, so the unwired case is the one the single build supports ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11).
- **Ports and queues.** `snapshot` / `is_valid` `queue_depth(2)`, `MemoryKind::kCudaDevice`, rank-agnostic. Required outputs precede the optional output in `setup()`; Exact overrides follow in Params order. Live Inputs use `queue_depth(2)`; `subarray` is a rank-1 pageable-host int64 config Tensor, `array_geometry_in` is a wire object. `snapshot_out` and `is_valid_out` are bounded device outputs; `array_geometry` is a bounded pageable-host schema output. [EA HSDK5 ONLY] Graph still has to bind the unwired device `is_valid_out`.
- **Live override payloads.** `array_geometry_in` is `Input<WireArrayGeometry>`, a generated descriptor-schema type converted with `from_wire` inside `resolve_live_override_via` so wire storage never reaches operator state. `subarray` rides a rank-1 pageable-host int64 config Tensor validated on arrival (rank, host memory, contiguity, dtype, extent) and converted per tick, with no state kept ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). The index bound is `max_subarray_channels`, else `max_virtual_channels`, else `max_channels`; with none of them set it falls back to the tick's channel count.
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. `snapshot` (and `is_valid` when present) are retained across ticks. The hot path does not synchronize.
- **Emission.** Required `snapshot_out` is prepared first. When geometry is active, its wire payload is prepared and published before `snapshot_out` so a downstream Exact join selects it when the snapshot triggers; an unwired emit is absorbed and latched off. `is_valid_out` remains an independent optional after the snapshot.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.SubarraySelectionParams`, version 1.
- **Precision.** `SubarraySelectionOp<float>`, `SubarraySelectionOp<double>`. The geometry tensor is always float64.
- **[EA HSDK5 ONLY]** Physical names `snapshot_out` / `is_valid_out` / `array_geometry_in` (HSDK5_PORT.md §2). Geometry input and output share `WireArrayGeometry` (§9/§10), so the reduced geometry directly feeds a downstream angle override. Optional `is_valid` Graph wiring remains deferred (§11).

## Consumer lifetime notes

Every published output is a fresh pool loan filled and committed this tick. A consumer may retain it. Convert `WireArrayGeometry` through `from_wire` when a host `ArrayGeometry` is needed.

## Performance

Cold / first new configuration: host `build_subarray_selection` + H2D of an `int32` channel-index table of length `L`, plus the INFO log when selection is non-identity. Steady: one S1 gather (`P = 1`, no forward-backward) or a D2D pass-through, plus host wire preparation while geometry remains consumed. The replace-in-place cache key uses exact `ArrayGeometry`, explicit-index vector, and `min_subarray_size` equality.

`performance/cpp/perf_subarray_selection.cpp` reports cold and steady Graph timings over an item/channel/snapshot ladder on a holey ULA. Cold includes plan construction and the index-table upload; steady includes gather plus schema-aware geometry publication. The smoke has no timing threshold.

## Tests and verification

Graph operator tests: `tests/cpp/angle_estimation/test_subarray_selection_op.cpp`; `is_valid_out` is bound but unwired. Host helper tests: `tests/cpp/common/test_subarray_selection.cpp`. Ground truth is synthetic holey lattices and channel-marked or boresight snapshots.

| Check | Proves |
| --- | --- |
| `HoleyUlaSelectsLongestRun` | Auto 1-D: `{0,1,2,3,5,6}` → channels `{0,1,2,3}`, `L = 4` |
| `FilledUlaPassesThrough` | Filled ULA: shape and wavelength unchanged |
| `FilledUnorderedUlaIsGatheredToLatticeOrder` | Keeping all channels does not bypass the required lattice-order permutation |
| `OptionalGeometryMayBeLeftUnwired` | Snapshot publication continues after the geometry output discovers it has no consumer and latches off |
| `ReducedGeometryDirectlyDrivesBeamscan` | Snapshot and same-tick wire geometry satisfy the downstream Exact join and steer at boresight |
| `ExplicitSubarray` / `SubarrayConfigurableFromParams` | Explicit indices gather those channels; Params author `subarray` |
| `LiveOverridesRequireExactCaptureTime` / `LiveOverridesRevertToParamsAfterExactMiss` | `Exact{}` join, and the revert to Params on `array_geometry_in` / `subarray` after a miss |
| `RejectsMalformedLivePayloads` / `RejectsLiveSubarrayBeyondCapturedBound` | Malformed wire geometry and an over-bound index Tensor are diagnosed, not consumed |
| `RejectsSelectionBeyondCapturedOutputBound` | A selected `L` above `max_subarray_channels` fails before requesting an undersized output loan |
| `PackUnpackRoundTrip` | `pack_array_geometry` / `unpack_array_geometry` identity (helper, not a Graph emit) |
| Helper `HoleyUraLargestRectangle` | 3×3 minus a corner → area-6 rectangle; azimuth-extent tie-break |
| Helper `ExplicitSelectionReordersToLattice` | Scrambled explicit indices come back lattice-ordered |
| Helper `ExplicitRejectsHoleySet` / `…OutOfRangeAndDuplicate` / `RejectsWhenBelowMinSize` | Named selection errors |

Not covered in the Graph suite: `is_valid` forward, 2-D snapshot (no `K` axis), float instantiation, holey URA through the operator, or the error paths (those last live on the helper). Optional-mask Graph wiring is deferred by HSDK5_PORT.md §11.

```bash
ctest --test-dir build -R 'test_subarray_selection' --output-on-failure
```

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
// [EA HSDK5 ONLY] bind is_valid_out even if is_valid is unwired.
```

Python bindings are not present for this operator (module-wide HSDK5_PORT.md §6 follow-up).

## Contract conformance

Matches the locked S3 contract (auto / explicit selection, `min_subarray_size`, forwarded mask, same-tick geometry emission when consumed) with the [EA HSDK5 ONLY] physical-type and naming interims the contract itself records. No undocumented algorithm divergence.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| Output port name `snapshot` | renamed | Physical name `snapshot_out` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §2); already noted in `OPERATOR_API.md`. |
| Output port name `is_valid` | renamed | Physical name `is_valid_out`. |
| `array_geometry` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` in `contract()` under the physical name `array_geometry_in`, not `Input<ArrayGeometry>` (HSDK5_PORT.md §2 / §9). |
| `subarray` **[live]** | conforms (payload form differs) | Rank-1 host config Tensor bounded by `max_subarray_channels`, not `Input<std::vector<int64_t>>` (HSDK5_PORT.md §9). |
| `array_geometry` output is `ArrayGeometry` | conforms (wire payload) | `Output<WireArrayGeometry>` preserves the logical descriptor and directly matches downstream live inputs (HSDK5_PORT.md §10). |
| Optional `is_valid` Graph wiring | EA-deferred | Receive + forward implemented; the connected Graph path is deferred because a contract entry cannot tolerate an unwired port (HSDK5_PORT.md §11). |
