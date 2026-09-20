# `TDMSparseCompensationOp`

## Purpose

Fuses detection-centered snapshot gathering, TDM motion-phase compensation, and virtual-array assembly. It replaces three stages when detection occurs before compensation.

## Responsibilities

- Owns per-cell TDM correction, virtual ordering/overlap policy, edge masks, and coherent geometry.
- Publishes geometry and validity before snapshots for downstream Exact joins.
- Does not run CFAR, estimate direction, or support DDM coding.

## When to use

Use after CFAR on uncompensated TDM maps to gather, phase-correct, and assemble only detected neighborhoods. It replaces dense compensation, virtual assembly, and snapshot gathering when downstream consumers need sparse angle snapshots rather than every RD cell. Use the dense path for full-cube consumers and `DDMSparseDemultiplexOp` for DDM.

```text
rd_maps + detection_idx -> TDMSparseCompensationOp -> snapshot + geometry -> angle estimator
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(M_tx,N_rx,D,R)` | Required trigger/aligned input | Uncompensated center-zero TDM maps. |
| in | `detection_idx` | int64 | `(N,2)` | Required trigger/aligned input | In-bounds Doppler/range rows. |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | PRI for correction and carrier for geometry. |
| live | `snapshot_window` | `TDMSparseWindowOverride` | POD `{doppler, range}` | Optional Exact one-tick override; Params fallback on miss | Neighborhood half-extents. |
| out | `snapshot` | complex64 / complex128 | `(N,num_virtual,num_snapshots)` | Required output | Corrected virtual snapshots. |
| out (optional, always) | `is_valid` | uint8 | `(N,num_snapshots)` | Optional `always` output | Edge-validity mask. |
| out | `array_geometry` | `WireArrayGeometry` | descriptor-Tensor | Required output | Virtual positions and wavelength. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins`, `max_detections` or `max_items`, `max_snapshots`; optional `max_virtual_channels` | Sizes Tensor pools and bounded device workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `waveform` | `FMCWWaveformDescriptor` | none (required) | `[live]` Exact | PRI and carrier frequency. |
| `tx_schedule` | `TXSchedule` | identity round-robin | static | Uniform firing-slot mapping. |
| `tx_positions` | `vector<double>` | none (required) | static | Row-major TX positions. |
| `rx_positions` | `vector<double>` | none (required) | static | Row-major RX positions. |
| `overlap_policy` | `OverlapPolicy` | `dedup` | static | Virtual collision handling. |
| `snapshot_window` | `vector<int64_t>` | `{0,0}` | `[live]` Exact | Doppler/range half-extents. |
| `doppler_edge` | `SnapshotEdgeMode` | `wrap` | static | Doppler edge policy. |
| `range_edge` | `SnapshotEdgeMode` | `shift` | static | Range edge policy. |

Contract: [`OPERATOR_API.md` section `TDMSparseCompensationOp`](../../../docs/planning/OPERATOR_API.md) - group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

The planner resolves each detection neighborhood. Every valid cell is corrected by firing slot and Doppler bin, then passed through the shared virtual gather/average plan. Geometry uses `p_virtual = p_tx + p_rx` and the wavelength derived from the same effective waveform.

## Shape, dtype, and timing rules

Inputs are exact rank four and two; detection coordinates must be in bounds. `num_snapshots` follows the resolved window. Live waveform/window ports join `detection_idx` with `Exact{}` and never latch. Geometry and mask publish before snapshot.

## Configuration and caching

One cache owns correction tables keyed by waveform, schedule, and Doppler shape; another owns virtual assembly keyed by positions and overlap policy. Borrowed inputs remain retained across asynchronous work.
