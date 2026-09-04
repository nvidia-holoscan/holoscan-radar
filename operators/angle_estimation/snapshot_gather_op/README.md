# `SnapshotGatherOp`

Gathers, for each detection, the per-channel complex snapshot(s) from the range-Doppler maps — the channel vector at the detected cell plus a configurable RD neighborhood — into the `snapshot` matrix the sparse angle estimators consume. It sits on the sparse path after an RD-map producer and `CFARDetectorOp` (or after `DetectionReductionOp` on the cluster-then-angle path) and before `⟨sparse angle estimator⟩`. Wire the **per-channel** `rd_maps`, not the NCI-integrated map CFAR thresholded on: the channel axis must stay intact. The gather is pure indexed copy — no arithmetic.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(num_channels, num_doppler, num_range)`, optional leading named batch axes | Channel axis immediately before the `(doppler, range)` plane. Leading rank must match `detection_idx` preserved-batch columns. |
| in | `detection_idx` | int64 | `(num_detections, 2 [+ n_batch])` | Last two columns are `(doppler_bin, range_bin)`; leading columns are preserved batch coordinates. Row count *is* `num_detections`. |
| out | `snapshot` | complex64 / complex128 | `(num_detections, num_channels, num_snapshots)` | Row *i* is detection *i*'s `(num_channels, num_snapshots)` block. Always emitted, including the empty (`num_detections == 0`) tick. |
| out | `is_valid` | uint8 (`0`/`1`) | `(num_detections, num_snapshots)` | Per-slot mask of gathered cells. Always emitted on this branch ([EA HSDK5 ONLY] device-output binding). All `1` except `clamp`-truncated slots. |
| live | `snapshot_window` | `SnapshotWindowOverride` `{doppler, range}` int64 POD | scalar POD | `[live]` Exact override; the entry is unconditional and wiring the port enables it — see Triggering. [EA HSDK5 ONLY] Params stay `std::vector<int64_t>`; the live Input is this fixed two-int64 POD. |

```text
rd_maps (per-channel) + CFARDetectorOp.detection_idx
  → SnapshotGatherOp → ⟨sparse angle estimator⟩.snapshot  [+ is_valid]
```

In-tree `rd_maps` today: `FMCWDopplerProcessorOp`.

On the cluster-then-angle path, `DetectionReductionOp.rep_idx` feeds `detection_idx` (one cell per cluster) instead of CFAR's full list.

Contract: [`OPERATOR_API.md` § `SnapshotGatherOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md)

## Algorithm

The operator shell (`snapshot_gather_op.cu`) receives and classifies both data tensors, resolves one tick-local `SnapshotGatherStageConfig`, builds the shape-derived `SnapshotGatherStagePlan`, prepares reusable validation workspace, loans both outputs, runs the gather, then publishes required `snapshot` followed by optional/always `is_valid`. The stage (`snapshot_gather_stage.hpp`) does the device work: `batch_linear_kernel` validates all detection coordinates and folds preserved-batch coordinates into a linear slice index, then `gather_snapshots_kernel` walks the neighborhood. There is no MatX plan, window, or FFT.

`num_snapshots = (2·doppler_half + 1)·(2·range_half + 1)` from the effective window. Every detection gets that many slots, regardless of edge mode. Neighborhood offsets run Doppler-outer, range-inner (`od` from `-d_half` to `+d_half`, then `orng` from `-r_half` to `+r_half`). Each axis index is resolved independently:

- **`wrap`** — `idx = ((center + offset) % n + n) % n`; always valid. Periodic — bin 0 is adjacent to bin `num_doppler − 1`. Default on Doppler (aliased spectrum). Offered on range for symmetry; rarely physical.
- **`clamp`** — `idx = center + offset`; if `idx < 0` or `idx >= n` the slot is invalid (`is_valid = 0`) and that channel vector is left as the pre-kernel zero fill.
- **`shift`** — slide the window inward so it stays inside `[0, n)` and keeps `num_snapshots` real cells, asymmetric about the detection near an edge. Start is `shift_window_start(center, half, n)`:

```text
win_len = 2 * half + 1
start = max(center - half, 0)
if start + win_len - 1 >= n:
    start = max((n - 1) - win_len + 1, 0)
idx = start + (offset + half)     # always marked valid
```

Slot *k* of row *i* is written channel-major, snapshot-minor — `out[c * num_snapshots + snap_slot]` — matching the estimators' `xi[c * num_snapshots + s]` indexing. This explicitly avoids the prior `(N, K, C)` write-order bug.

`is_valid` is written and emitted as `uint8` (`0`/`1`), the NumPy-visible DLPack type (STATUS.md L1). Consumers still accept `bool` or `uint8`.

Spatial smoothing is **not** applied here — that is a downstream estimator parameter on the gathered channel axis. Multi-frame / temporal snapshots would need cross-tick state and are out of scope.

## Shape and dtype rules

- `snapshot` axis 0 = `detection_idx` row count (`num_detections`).
- `snapshot` axis 1 = `rd_maps` trailing-3 (`num_channels`).
- `snapshot` axis 2 = `(2·doppler_half + 1)·(2·range_half + 1)` from the effective window.
- `is_valid` = `(num_detections, num_snapshots)`.
- `rd_maps` must end with `(num_channels, num_doppler, num_range)`; those three extents must be positive. Optional leading axes are admitted only as named preserved-batch dims: their rank must equal `detection_idx`'s column count minus 2, and each preserved coordinate must lie in `[0, leading_dim)`. More than 8 preserved dims is rejected (`kMaxSnapshotPreservedBatchDims`).
- Every Doppler/range coordinate in `detection_idx` must address a real `rd_maps` cell. A `shift` window must fit its corresponding axis; unlike `wrap` and `clamp`, it promises a full window of distinct real cells.
- Precision is the template instantiation (`float` → complex64, `double` → complex128). Real-IF `rd_maps` is rejected. `detection_idx` must be int64.
- Inputs must be CUDA-accessible (device, managed, or pinned-host) and compact row-major. Outputs are device tensors.
- Mid-stream, a `[live] snapshot_window` change is a legal output-shape change (`num_snapshots` moves); downstream re-infers. Edge modes cannot move — they are static.

Rejected combinations and the operator's own messages:

| Condition | Message |
| --- | --- |
| `snapshot_window` length not in `{0, 1, 2}` | `SnapshotGatherOp: snapshot_window must have exactly 2 entries (doppler, range)` |
| either half-extent `< 0` | `SnapshotGatherOp: snapshot_window half-extents must be >= 0` |
| `detection_idx` rank ≠ 2 or columns `< 2` | `SnapshotGatherOp: detection_idx must have shape (num_detections, 2 [+ n_batch])` |
| `detection_idx` rows exceed `MaxTensorExtents` `max_detections` / `max_items` | `SnapshotGatherOp: detection_idx rows (N) exceed MaxTensorExtents max_detections/max_items (C)` |
| `rd_maps` rank `< 3` | `SnapshotGatherOp: rd_maps must end with (num_channels, num_doppler, num_range)` |
| a trailing RD extent `<= 0` | `SnapshotGatherOp: rd_maps trailing axes must be positive` |
| leading rank ≠ preserved-batch columns | `SnapshotGatherOp: rd_maps leading rank must match detection_idx preserved batch columns` |
| preserved-batch rank `> 8` | `SnapshotGatherOp: too many preserved batch dims in detection_idx` |
| a preserved-batch coordinate out of range | `SnapshotGatherOp: detection_idx preserved-batch coordinate out of range for rd_maps leading dims` |
| a Doppler/range coordinate out of range | `SnapshotGatherOp: detection_idx doppler/range coordinate out of range for rd_maps` |
| a `shift` window longer than its axis | `SnapshotGatherOp: shift window length must not exceed its rd_maps axis` |
| real-valued `rd_maps` | `rd_maps: real-valued input <dtype> is not supported here; provide <complex dtype>` |
| `rd_maps` / `detection_idx` dtype mismatch | `SnapshotGatherOp: tensor received on input port '<port>' has dtype <got>, expected <want>` |
| non-contiguous or pageable-host input | `SnapshotGatherOp: tensor received on input port '<port>' must be contiguous (compact row-major)` / `… is not CUDA-accessible (device, managed, or pinned-host memory required)` |
| no CUDA compute stream | `SnapshotGatherOp: no CUDA compute stream` |

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `snapshot_window` | `std::vector<int64_t>` (YAML); live POD `{doppler, range}` | `{0, 0}` | `[live]` Exact | Free — a gather, no plan or coefficient rebuild | Per-axis `(doppler, range)` neighborhood half-extent. `{0, 0}` → one snapshot (the detection cell). Empty → `{0, 0}`; size-1 → isotropic `{w, w}`. |
| `doppler_edge` | `SnapshotEdgeMode` `{wrap, clamp, shift}` | `wrap` | static | — | Neighborhood behavior at the Doppler-axis edges. |
| `range_edge` | `SnapshotEdgeMode` `{wrap, clamp, shift}` | `shift` | static | — | Same three behaviors on range. `wrap` is rarely physical for range bins. |

YAML / `from_config` / `set_params` field names are the table's Parameter column. The live Input is named `snapshot_window` and carries `SnapshotWindowOverride`, not the Params vector ([EA HSDK5 ONLY] — HSDK5_PORT.md §9). Override resolution: Params seed the effective value; the port joins the trigger with `Exact{}`, so a sample is selected only if its `capture_time` matches the tick's data. A match applies to that tick only; a miss reverts to Params. Leaving the port unwired is what disables the override — no parameter states the wiring.

## SDK implementation notes

- **Construction.** `SnapshotGatherOp(const MaxTensorExtents&)`. Constructor-created state owns fixed leading-dimension and error-flag buffers. If none of `max_detections` / `max_items`, `max_channels`, or `max_snapshots` is set, pool byte spans stay 0 (non-pooled / Python path) and the batch-linear table grows only when needed. If any is set, the constructor requires `max_detections` or `max_items`, plus `max_channels` and `max_snapshots`; it preallocates the batch-linear table and sizes `snapshot` as rank 3 and `is_valid` as rank 2. The plan builder rejects ticks beyond any captured extent with an operator-owned error before pool allocation.
- **Triggering.** `contract()` triggers `OnAligned` on `rd_maps` and `detection_idx` with `window = 0`, both taken `Exact{}`, plus an unconditional `Exact{}` + `SafeDefault{}` entry for `snapshot_window`. The entry stands whether or not the app wires the port: an unwired non-trigger `Exact{}` entry compiles under `OnAligned` exactly as it does under `OnEach` and reports `kNotConnected` every activation ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11).
- **Ports and queues.** Both data Inputs use `queue_depth(2)`, `MemoryKind::kCudaDevice`, and rank-agnostic declarations (dtype enforced in `compute()`). `snapshot` declares pooled rank 3; `is_valid` declares pooled rank 2. The live Input is `queue_depth(2)` with no tensor layout.
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. `rd_maps` and `detection_idx` are retained across ticks as a stream-ordered lifetime guard. The hot path **does** synchronize: after `batch_linear`, a D2H of a one-int error flag plus `cudaStreamSynchronize` validates preserved-batch coordinates before the gather kernel. Empty-detection ticks skip that sync.
- **Emission.** Both loans are allocated before either is published because one gather kernel fills both buffers. Required `snapshot` is emitted first; optional/always `is_valid` absorbs an unwired emit without failing the tick. Zero-detection ticks use `prepare_device_tensor` no-ops in the same order.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.SnapshotGatherParams`, version 1. Live POD identity `com.nvidia.holoscan.radar.SnapshotWindowOverride`, version 1.
- **Precision.** Explicit instantiations: `SnapshotGatherOp<float>`, `SnapshotGatherOp<double>`.
- **[EA HSDK5 ONLY]** The `snapshot_window` live payload is `SnapshotWindowOverride` rather than the Params vector (HSDK5_PORT.md §9); the Exact `Input<>` is declared unconditionally. The Graph test drives a matched middle tick and verifies the next miss reverts to Params.

## Consumer lifetime notes

Both `snapshot` and `is_valid` are fresh pool loans filled and committed each tick. A consumer may retain them; they are not operator-owned ping-pong workspace.

## Performance

Constructor-created state reuses the leading-dimension vector, batch-linear table, and one-int error flag. With constructor extents the tables never allocate on the hot path; the non-pooled path grows the batch-linear table only when a larger detection count arrives. Each non-empty tick H2D-copies the used leading extents and host-syncs the validation flag. A `[live] snapshot_window` change rebuilds no cached artifact; it only changes gather and output extents.

Harness: `performance/cpp/perf_snapshot_gather.cpp` (Graph dual-input runner). Size ladder `xs`…`xl` (channels × RD map × detection count) and windows `{0,0}`, `{1,1}`, `{2,2}`. It records cold, steady, and inline Exact `snapshot_window` override timing. No measured numbers are recorded in-repo for this operator; run the harness:

```bash
ctest --test-dir build -R perf_snapshot_gather_smoke
```

## Tests and verification

C++: `tests/cpp/angle_estimation/test_snapshot_gather.cpp` (Graph dual-IO harness). Python: `tests/python/angle_estimation/test_snapshot_gather.py` (legacy HSDK 4.x bindings; module-wide HSDK 5 Python work is deferred in `HSDK5_PORT.md` §6). Ground truth is a synthetic RD cube whose cell `(c, d, r)` is the deterministic complex `c·100 + d·10 + r + j(c+d+r)` — not `radar_common/sim/`.

| Check | Proves |
| --- | --- |
| `SingleCellGather` | Default `(0,0)` window → shape `(N, C, 1)` and the cell's channel vector |
| `WindowGatherPreservesChannelAxis` / `SnapshotWindowConfigurableFromParams` | `num_snapshots = (2d+1)(2r+1)`; Params author `snapshot_window`; channel-major layout |
| `DopplerWrapAtBoundary` | `wrap` at Doppler bin 0 yields `{n−1, 0, 1}` |
| `ClampMarksInvalidSlots` | `clamp` at `(0,0)` with window `(1,1)` marks 4 of 9 slots valid |
| `ShiftKeepsSnapshotCount` | `shift` at `(0,0)` keeps 9 valid slots and slides to real cells |
| `PreservedBatchSelectsCorrectSlice` | Leading batch column selects the matching `rd_maps` slice |
| `EmptyDetections` | Zero-row `detection_idx` emits zero-row outputs |
| `OptionalValidityMayBeLeftUnwired` | Declining optional/always `is_valid` does not suppress required `snapshot` |
| `FloatInstantiationSmoke` / `FloatAndDoubleAgree` | Both precisions; values agree to `1e-5` |
| `RejectsRdMapsDtypeMismatch` / `RejectsNarrowDetectionIdx` / `RejectsNegativeSnapshotWindow` / `RejectsNegativeIsotropicSnapshotWindow` | Dtype, shape, and both vector normalization error paths |
| `RejectsBatchCoordOutOfRange` / `RejectsCellCoordOutOfRange` / `RejectsOversizedShiftWindow` | All index columns are bounded and `shift` cannot read past an axis |
| `LiveSnapshotWindowAppliesForOneTick` | A matched Exact override changes one output shape; the following miss reverts to Params |

TESTING.md also asks that the channel axis stay intact (covered) and that a non-trivial window gather the correct neighborhood (covered). Memory-space and determinism cross-cuts are not dedicated tests here.

```bash
ctest --test-dir build -R test_snapshot_gather --output-on-failure
```

## Usage

```yaml
snapshot_gather:
  snapshot_window: [0, 0]
  # doppler_edge: wrap    # default
  # range_edge: shift     # default
```

```cpp
auto gather = graph.op<SnapshotGatherOp<float>>(
    "snapshot_gather",
    MaxTensorExtents{/* max_detections or max_items, max_channels, max_snapshots */});
gather.set_params(cfg.gather_params);

graph.add_flow(doppler->rd_maps, gather->rd_maps, q);
graph.add_flow(cfar->detection_idx, gather->detection_idx, q);
graph.add_flow(gather->snapshot, estimator->snapshot, q);
// Optional: wire the validity mask when clamp edges need to be excluded.
graph.add_flow(gather->is_valid, estimator->is_valid, q);
```

Python bindings exist at `python/_snapshot_gather_op_bindings.cpp` as a legacy HSDK 4.x `Fragment` / `Arg` trampoline and are not built on this branch (`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`; see `HSDK5_PORT.md` §6).

## Contract conformance

Matches the locked contract on ports, shapes, edge-mode math, `snapshot_window` semantics, defaults, and optional-output disposition, with the [EA HSDK5 ONLY] live payload form below.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `snapshot_window` **[live]** as a vector | conforms (payload form differs) | Params stay `std::vector<int64_t>`; the live Input is the fixed `{doppler, range}` `SnapshotWindowOverride` POD ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). The unconditional Exact contract entry, single-tick resolve, and mid-stream Graph test are implemented. |
| `is_valid` optional; emit when either axis uses `clamp` | Done | `always` disposition: the validity bits are written inside the existing gather kernel, so it is produced regardless (all-`1` when neither axis is `clamp`) and an unwired sink is discarded |
| Custom CUDA gather | conforms | `batch_linear_kernel` validates/folds named leading coordinates; `gather_snapshots_kernel` performs the channel-neighborhood copy. |
| `shift` keeps `num_snapshots` using real distinct cells | conforms | The plan rejects a shift window longer than its axis before launching the gather. |

Verified in the HSDK 5 container with all `SnapshotGatherOpTest` cases and `perf_snapshot_gather_smoke`.
