# `DDMDenseDemultiplexOp`

## Purpose

Separates simultaneous DDM transmitters from a dense pre-demultiplex range-Doppler cube. Linear codes gather integer-offset Doppler bands; nonlinear phase codes correlate across the full Doppler axis.

## Responsibilities

- Owns DDM code validation, TX separation, initial-code-phase removal, and velocity-axis transformation.
- Publishes a coherent physical velocity sideband before the demultiplexed cube when available.
- Does not resolve velocity ambiguity, assemble virtual channels, suppress clutter, or run CFAR.

## When to use

Use after Doppler processing when a dense MIMO-DDM pipeline needs every TX-separated map. Prefer `DDMPowerFoldOp` plus `DDMSparseDemultiplexOp` for detect-before-demultiplex workflows. Do not use it for TDM or to resolve the reduced-interval velocity ambiguity.

```text
FMCWDopplerProcessorOp -> DDMDenseDemultiplexOp -> [DopplerNotchOp] -> VirtualArrayAssemblyOp
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(N_rx, D, R)` | Required trigger/aligned input | Pre-demultiplex center-zero DDM maps. |
| in (optional) | `velocity_bins` | float32 / float64 | `(D,)` | Optional Exact side input | Physical axis aligned with `rd_maps`. |
| out | `rd_maps_out` | complex64 / complex128 | `(M_tx, N_rx, D_out, R)` | Required output | TX-separated maps. |
| out (optional, latched) | `velocity_bins_out` | float32 / float64 | `(D_out,)` | Optional latched output | Transformed axis emitted only when the input sideband is present. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins` | Sizes Tensor pools and bounded device workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `ddm_code` | `DDMCodeDescriptor` | none (required) | static | Defines TX count and exactly one linear-offset or nonlinear phase-code layout. |

Contract: [`OPERATOR_API.md` section `DDMDenseDemultiplexOp`](../../../docs/planning/OPERATOR_API.md) - group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

Linear mode validates integer-bin offsets, builds a band-gather table, gathers the centered usable bins for every TX/RX/range cell, and removes initial code phase. Nonlinear mode builds correlation weights and retains the full Doppler axis. The optional velocity output takes the same centered slice or full-axis copy.

## Shape, dtype, and timing rules

`rd_maps` is exact rank three. A present axis is rank one, template-matching real data of length `D`. Linear output has per-band `D_out`; nonlinear output keeps `D`. The optional axis joins by `Exact{}` with `SafeDefault{}` and publishes before `rd_maps_out`.

## Configuration and caching

Linear gather/phase tables and nonlinear correlation weights use separate caches keyed by validated code and Doppler extent. Static `ddm_code` means rotating layouts require separate instances.
