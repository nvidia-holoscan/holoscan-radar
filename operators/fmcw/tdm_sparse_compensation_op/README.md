# `TDMSparseCompensationOp`

## Purpose

Fuses detection-centered snapshot gathering, TDM motion-phase compensation, and virtual-array assembly. It replaces three stages when detection occurs before compensation.

## Responsibilities

- Owns per-cell TDM and externally supplied fold correction, virtual ordering/overlap policy, edge masks, and coherent geometry.
- Publishes geometry and validity before snapshots for downstream Exact joins.
- Does not run CFAR, estimate direction, or support DDM coding.

## When to use

Use after CFAR on uncompensated TDM maps to gather, phase-correct, and assemble only detected neighborhoods. It replaces dense compensation, virtual assembly, and snapshot gathering when downstream consumers need sparse angle snapshots rather than every RD cell. Use the dense path for full-cube consumers and `DDMSparseDemultiplexOp` for DDM.

```text
rd_maps + detection_idx + optional fold_idx -> TDMSparseCompensationOp -> snapshot + geometry -> angle estimator
```

Declarations and parameter defaults: [tdm_sparse_compensation_op.hpp](tdm_sparse_compensation_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(M_tx,N_rx,D,R)` | Required trigger/aligned input | Uncompensated center-zero TDM maps. |
| in | `detection_idx` | int64 | `(N,2)` | Required trigger/aligned input | In-bounds Doppler/range rows. |
| in | `fold_idx` | int64 | `(N)` | Optional Exact side input; `SafeDefault{0}` on miss | External TDM Doppler-ambiguity class in `[0,M_tx)`. |
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
| `cache_budget_bytes` | unsigned bytes | 67108864 (64 MiB) | static | Instance-wide retained cache ceiling; 0 disables retention. Oversized entries execute uncached with a warning. |
| `waveform` | `FMCWWaveformDescriptor` | none (required) | `[live]` Exact | PRI and carrier frequency. |
| `tx_schedule` | `TXSchedule` | identity round-robin | static | Uniform firing-slot mapping. |
| `tx_positions` | `vector<double>` | none (required) | static | Row-major TX positions. |
| `rx_positions` | `vector<double>` | none (required) | static | Row-major RX positions. |
| `overlap_policy` | `OverlapPolicy` | `dedup` | static | Virtual collision handling. |
| `snapshot_window` | `vector<int64_t>` | `{0,0}` | `[live]` Exact | Doppler/range half-extents. |
| `doppler_edge` | `SnapshotEdgeMode` | `wrap` | static | Doppler edge policy. |
| `range_edge` | `SnapshotEdgeMode` | `shift` | static | Range edge policy. |

## Algorithm

The planner resolves each detection neighborhood. Every valid cell is corrected by firing slot, Doppler bin, and the detection's external fold class, then passed through the shared virtual gather/average plan. A Doppler-window cell crossing the reduced-axis wrap adjusts the class modulo `M_tx`. Geometry uses `p_virtual = p_tx + p_rx` and the wavelength derived from the same effective waveform.

## Shape, dtype, and timing rules

Inputs are exact rank four and two; detection coordinates must be in bounds. A connected rank-one `fold_idx` must have one `[0,M_tx)` entry per detection; an unwired port means fold zero. `num_snapshots` follows the resolved window. Fold and live waveform/window ports join `detection_idx` with `Exact{}` and never latch. Fold correction restores spatial coherence but does not estimate ambiguity or unfold velocity. Geometry and mask publish before snapshot.

## Configuration and caching

One cache owns correction tables keyed by waveform, schedule, and Doppler shape; another owns virtual assembly keyed by positions and overlap policy. Borrowed inputs remain retained across asynchronous work.

## Performance

Single-snapshot windows pack detection and virtual-channel pairs into active threads of 128-thread blocks. Other windows and extents beyond the new flat-grid bounds retain the original generic kernel. Coordinate/fold checks, channel ordering, correction multiplication, average accumulation order, masks, and publication are unchanged.

A bounded Thor run on the pinned HSDK 5 deployment measured `baseline-large`: fresh float latency medians were 2.7221 and 2.4883 ms before/after, with median paired savings of 0.2449 ms (9.00%). Double medians were 5.1883 and 4.8916 ms, with paired savings of 0.3021 ms (5.81%). Three paired screening runs and one fresh three-pair confirmation used five warmups and 50 measured frames per process. Cached live-window cases measure Params steady frames separately from the changed frame; their steady gains do not establish a window-kernel gain.

Small cases have costs: one fresh small-float process measured 70.935 versus 64.750 microseconds for its paired original, while the other five candidate processes across screening/confirmation did not repeat that high regime. Optional-minimal small float was about 1.125 microseconds slower in fresh confirmation, within the declared max(3 microseconds, 3%) allowance. Its sampled p99 rose by 12.73 microseconds in float and 7.72 in double. These scoped measurements do not guarantee zero slowdown or population-tail behavior.

Twelve stage/public tests and six public recipe exports matched the declared checks in both precisions, with all three exported ports exact. An independent direct-stage oracle checks 33 detections, 132 distinct point values, supplied fold classes, validity, and errors across the flat-block tail. A separate positive-fold public fixture encountered an unresolved SDK/runtime binding issue on the unchanged original with grouped and separate publication; the public recipe exports use default fold and do not resolve that coverage gap. No contract or SDK behavior was changed to address it.
