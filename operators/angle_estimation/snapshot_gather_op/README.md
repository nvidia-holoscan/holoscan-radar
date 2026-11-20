# `SnapshotGatherOp`

## Purpose

Gathers, for each detection, the per-channel complex snapshot(s) from the range-Doppler maps — the channel vector at the detected cell plus a configurable RD neighborhood — into the `snapshot` matrix the sparse angle estimators consume. It sits on the sparse path after an RD-map producer and `CFARDetectorOp` (or after `DetectionReductionOp` on the cluster-then-angle path) and before `⟨sparse angle estimator⟩`. Wire the **per-channel** `rd_maps`, not the NCI-integrated map CFAR thresholded on: the channel axis must stay intact. The gather is pure indexed copy — no arithmetic.

## Responsibilities

- Validates detection rows against the per-channel range-Doppler layout and gathers the configured neighborhood for each row.
- Owns edge handling, output row order, validity emission, and tick-local snapshot-window overrides.
- Does not perform TDM/DDM compensation, estimate angles, or gather from an NCI-collapsed map that lacks channels.

## When to use

Use in detect-then-angle workflows to gather per-channel RD neighborhoods around CFAR detections or reduced cluster representatives. It feeds Beamscan, MUSIC, MVDR, ESPRIT, or Root-MUSIC with row-aligned snapshots and edge validity. Do not place it after a dense beamforming cube, and use the fused TDM/DDM sparse bridge operators when compensation or demultiplexing must occur during gathering.

```text
rd_maps (per-channel) + CFARDetectorOp.detection_idx
  → SnapshotGatherOp → ⟨sparse angle estimator⟩.snapshot  [+ is_valid]
```

Declarations and parameter defaults: [snapshot_gather_op.hpp](snapshot_gather_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(num_channels, num_doppler, num_range)`, optional leading named batch axes | Required trigger/aligned input | Channel axis immediately before the `(doppler, range)` plane. Leading rank must match `detection_idx` preserved-batch columns. |
| in | `detection_idx` | int64 | `(num_detections, 2 [+ n_batch])` | Required trigger/aligned input | Last two columns are `(doppler_bin, range_bin)`; leading columns are preserved batch coordinates. Row count *is* `num_detections`. |
| out | `snapshot` | complex64 / complex128 | `(num_detections, num_channels, num_snapshots)` | Required output | Row *i* is detection *i*'s `(num_channels, num_snapshots)` block. Always emitted, including the empty (`num_detections == 0`) tick. |
| out (optional, always) | `is_valid` | uint8 (`0`/`1`) | `(num_detections, num_snapshots)` | Optional `always` output | Per-slot mask of gathered cells. Publication is attempted every tick. All `1` except `clamp`-truncated slots. |
| live | `snapshot_window` | `SnapshotWindowOverride` `{doppler, range}` int64 POD | scalar POD | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; the entry is unconditional and wiring the port enables it — see Triggering. Params stay `std::vector<int64_t>`; the live Input is this fixed two-int64 POD. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | all unset is allowed; otherwise `max_detections` or `max_items`, plus `max_channels` and `max_snapshots`, are required | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `snapshot_window` | `std::vector<int64_t>` (YAML); live POD `{doppler, range}` | `{0, 0}` | `[live]` Exact | Per-axis `(doppler, range)` neighborhood half-extent. `{0, 0}` → one snapshot (the detection cell). Empty → `{0, 0}`; size-1 → isotropic `{w, w}`. Cost: Free — a gather, no plan or coefficient rebuild. |
| `doppler_edge` | `SnapshotEdgeMode` `{wrap, clamp, shift}` | `wrap` | static | Neighborhood behavior at the Doppler-axis edges. |
| `range_edge` | `SnapshotEdgeMode` `{wrap, clamp, shift}` | `shift` | static | Same three behaviors on range. `wrap` is rarely physical for range bins. |

## Algorithm

The operator shell (`snapshot_gather_op.cu`) receives and classifies both data tensors, resolves one tick-local `SnapshotGatherStageConfig`, builds the shape-derived `SnapshotGatherStagePlan`, prepares reusable validation workspace, loans both outputs, runs the gather, then publishes optional/always `is_valid` before required `snapshot` so downstream Exact mask inputs see the sideband when the snapshot triggers. A declined mask emit is absorbed and cannot suppress the snapshot. The stage (`snapshot_gather_stage.hpp`) does the device work: `batch_linear_kernel` validates all detection coordinates and folds preserved-batch coordinates into a linear slice index, then `gather_snapshots_kernel` walks the neighborhood. There is no MatX plan, window, or FFT.

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

`is_valid` is written and emitted as `uint8` (`0`/`1`), the NumPy-visible DLPack type. Consumers still accept `bool` or `uint8`.

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

## Performance

Constructor-created state reuses the leading-dimension vector, batch-linear table, and one-int error flag. With constructor extents the tables never allocate on the hot path; the non-pooled path grows the batch-linear table only when a larger detection count arrives. Each non-empty tick H2D-copies the used leading extents and host-syncs the validation flag. A `[live] snapshot_window` change rebuilds no cached artifact; it only changes gather and output extents.

The gather explicitly writes complex zero to each invalid slot and copies every valid cell, so it fills the whole output without a preceding full-output memset. Validation synchronization, channel-major output order and optional validity-mask behavior are unchanged.

On Thor, `baseline-large` (32 channels, 512 × 1024 RD cells, 4096 detections, nine slots) measured these fresh steady results. Each precision used three screen and three fresh pairs, five warmups and 50 frames; all 23 native tests and established exact two-output policies across six cases in both precisions passed.

| Precision | Original median | Fused median | Median paired saving |
| --- | --- | --- | --- |
| Float | 1.713638 ms | 1.646456 ms | 67.18 µs / 3.92% |
| Double | 3.136229 ms | 2.985085 ms | 151.14 µs / 4.82% |

These are scoped steady gains with tradeoffs. Fresh small-double p99 increased 30.43 µs and optional-output-minimal float p99 increased 35.56 µs; both also worsened in the screen. One large-double fresh frame reached 5.177 ms versus its original pair's maximum 3.195 ms despite a useful median. Fresh medium-double setup increased 0.835 ms and optional-minimal float cold time increased 18.41 µs. Fifty-frame p99 is descriptive; no startup, tail, arbitrary-shape, multi-GPU or direct consumer-chain guarantee is implied.

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
