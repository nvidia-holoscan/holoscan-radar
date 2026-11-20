# `DDMPowerFoldOp`

## Purpose

Builds the reduced real detection surface for sparse DDM. It folds every usable TX band onto one Doppler interval and non-coherently integrates power across transmit and receive looks without materializing a demultiplexed cube.

## Responsibilities

- Owns the linear band mapping shared with sparse demultiplexing and sums `|x|^2` over TX/RX looks.
- Produces an exact-sized CFAR-ready map.
- Does not support nonlinear or fractional-bin codes, resolve folds, estimate angle, or preserve complex phase.

## When to use

Use before `CFARDetectorOp` in detect-before-demultiplex DDM with a linear integer-bin code. It avoids materializing the dense TX-separated cube when detection can operate on non-coherently integrated power. Do not use it for nonlinear/fractional-bin codes, physical-velocity notching, or workflows that need complex per-TX samples before detection.

```text
pre-demux rd_maps -> DDMPowerFoldOp -> CFARDetectorOp -> DDMSparseDemultiplexOp
```

Declarations and parameter defaults: [ddm_power_fold_op.hpp](ddm_power_fold_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(N_rx, D, R)` | Required trigger/aligned input | Pre-demultiplex center-zero DDM maps. |
| out | `power_map` | float32 / float64 | `(D/M_tx - guard_bands, R)` | Required output | Folded power summed over `M_tx * N_rx` looks. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_rx`, `max_doppler_bins`, `max_range_bins` | Sizes Tensor pools and bounded device workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `cache_budget_bytes` | unsigned bytes | 67108864 (64 MiB) | static | Instance-wide retained cache ceiling; 0 disables retention. Oversized entries execute uncached with a warning. |
| `ddm_code` | `DDMCodeDescriptor` | none (required) | static | Defines linear integer-bin TX bands and discarded guards. |

## Algorithm

The shared DDM planner maps each folded Doppler bin and TX to a source bin. For each output cell the CUDA stage sums squared complex magnitude over all mapped TX bands and RX channels. CFAR should use `nci_axes: []` and `preintegrated_looks: M_tx * N_rx`; correlated looks may require empirical calibration.

## Shape, dtype, and timing rules

Input is exact rank three complex data and output is exact rank two in matching real precision. `D` must divide by TX count, offsets must be integer-bin aligned, and guard removal must leave a positive width. Folded output is not a physical velocity axis and must not feed `DopplerNotchOp`.

## Configuration and caching

`ddm_code` is static. The integer band map is built for a new code/shape key and reused thereafter. There are no live inputs or optional outputs.
