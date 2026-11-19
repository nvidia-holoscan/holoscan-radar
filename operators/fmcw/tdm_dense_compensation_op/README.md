# `TDMDenseCompensationOp`

## Purpose

Applies Doppler-bin-dependent inter-transmit phase correction to dense TDM range-Doppler maps. It preserves the explicit `(M_tx, N_rx, D, R)` layout so virtual-array assembly can follow without changing detector coordinates.

## Responsibilities

- Owns firing-slot-aware TDM motion-phase correction for every dense RD cell.
- Validates the uniform transmit schedule and uses the effective waveform for the current tick.
- Does not deinterleave ADC data, assemble virtual channels, or detect targets.

## When to use

Use after TDM deinterleave plus range/Doppler processing when the complete dense cube must be phase-corrected before virtual-array assembly or detection. Choose `TDMSparseCompensationOp` when detection occurs first and only detected cells need correction. Do not use it for DDM, data without an explicit TX axis, or pass-through.

```text
... -> FMCWDopplerProcessorOp -> TDMDenseCompensationOp -> VirtualArrayAssemblyOp
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(M_tx, N_rx, D, R)` | Required trigger/aligned input | Uncompensated center-zero TDM maps with explicit TX axis. |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Per-TX PRI and carrier metadata for this tick. |
| out | `rd_maps_out` | complex64 / complex128 | same as `rd_maps` | Required output | Phase-corrected maps with layout preserved. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins` | Sizes Tensor pools and bounded device workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `cache_budget_bytes` | unsigned bytes | 67108864 (64 MiB) | static | Instance-wide retained cache ceiling; 0 disables retention. Oversized entries execute uncached with a warning. |
| `waveform` | `FMCWWaveformDescriptor` | none (required) | `[live]` Exact wire-schema override | Per-TX PRI for Doppler-dependent correction. |
| `tx_schedule` | `TXSchedule` | empty identity round-robin | static | Maps TX identity to firing slot; must be uniform and match `M_tx`. |

Contract: [`OPERATOR_API.md` section `TDMDenseCompensationOp`](../../../docs/planning/OPERATOR_API.md) - group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

The shared coherent-compensation planner converts each Doppler bin and per-TX PRI into a unit-magnitude correction indexed by firing slot. The CUDA stage multiplies every `(tx, rx, d, r)` sample, preserving shape, relative magnitude, and RX ordering. Empty `tx_schedule` maps TX id directly to firing slot.

## Shape, dtype, and timing rules

Input rank is exactly four; opaque leading batch is rejected. Every named extent must fit the constructor bounds. The waveform override joins `rd_maps` with `Exact{}` and `SafeDefault{}`; a miss uses Params for that tick and never latches.

## Configuration and caching

Correction tables are keyed by waveform, tensor shape, and firing-slot mapping. New keys build a device table; repeated keys reuse it. Schedule structure remains static.
