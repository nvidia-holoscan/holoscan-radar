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
| `cache_budget_bytes` | unsigned bytes | 134217728 (128 MiB) | static | Instance-wide retained cache ceiling; 0 disables retention. Oversized entries execute uncached with a warning. |
| `ddm_code` | `DDMCodeDescriptor` | none (required) | static | Defines TX count and exactly one linear-offset or nonlinear phase-code layout. |

Contract: [`OPERATOR_API.md` section `DDMDenseDemultiplexOp`](../../../docs/planning/OPERATOR_API.md) - group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

Linear mode validates integer-bin offsets, builds a band-gather table, gathers the centered usable bins for every TX/RX/range cell, and removes initial code phase. Nonlinear mode builds correlation weights and retains the full Doppler axis. The optional velocity output takes the same centered slice or full-axis copy.

## Shape, dtype, and timing rules

`rd_maps` is exact rank three. A present axis is rank one, template-matching real data of length `D`. Linear output has per-band `D_out`; nonlinear output keeps `D`. The optional axis joins by `Exact{}` with `SafeDefault{}` and publishes before `rd_maps_out`.

## Configuration and caching

Linear gather/phase tables and nonlinear correlation weights use separate caches keyed by validated code and Doppler extent. Static `ddm_code` means rotating layouts require separate instances.

## Nonlinear packing performance

Nonlinear correlation uses a shared-memory transpose for packing and scattering 32×32 tiles. Original scalar kernels remain the fallback beyond the tiled grid limits; FFT dimensions, correlation weights, sample arithmetic and output layout are unchanged. Both precisions pass 32 public DDM tests, including RX 65535/65536 fallback boundaries, and the shared TDM suite. Exact exports cover seven existing cases, including full nonlinear-large outputs and optional velocity axes.

On the current Thor deployment, a fresh three-pair comparison of `nonlinear-large` (`M_tx=8`, `N_rx=32`, `D=256`, `R=1024`) measured the following steady compute latencies after five warmups with 50 frames/run:

| Precision | Original median (ms) | Tiled median (ms) | Median paired reduction | Paired range |
| --- | --- | --- | --- | --- |
| float | 24.924 | 16.612 | 33.27% (8.293 ms) | 32.95–33.80% |
| double | 92.998 | 74.691 | 19.68% (18.305 ms) | 19.48–20.68% |

The independent screen also passed. Smaller and axis-bearing selected cases stayed within the predeclared median allowance of max(0.01 ms, 3%); medium nonlinear cases improved. Small-case timing can vary by microseconds, and setup/cold results vary across runs. These are scoped engineering observations, not a guarantee for other shapes, startup or population tails.
