# `DopplerNotchOp`

## Purpose

Suppresses a physical radial-velocity stopband on an already formed range-Doppler tensor. It applies the same hard or raised-cosine gain to every named channel while preserving shape, dtype, and relative channel phase.

## Responsibilities

- Owns physical-axis validation, cyclic or ordinary distance, and notch gain application.
- Always nulls the nearest bin to the configured center, even at zero width.
- Does not create a velocity axis, fold DDM bands, or provide a pass-through `none` mode.

## When to use

Use after Doppler processing or dense DDM demultiplexing when a known physical radial-velocity interval must be suppressed across all channels. Omit the operator for pass-through. Do not feed it folded DDM power, because one folded bin represents multiple physical velocities.

```text
rd_maps + velocity_bins -> DopplerNotchOp -> rd_maps_out
```

Declarations and parameter defaults: [doppler_notch_op.hpp](doppler_notch_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(D,R)`, `(N_rx,D,R)`, or `(M_tx,N_rx,D,R)` | Required trigger/aligned input | Maps with Doppler as the second-to-last axis. |
| in | `velocity_bins` | float32 / float64 | `(D,)` | Required trigger/aligned input | Finite strictly monotonic physical velocity in m/s, ascending or descending. |
| live | `center_velocity` | float64 scalar | scalar | Optional Exact one-tick override; Params fallback on miss | Notch center. |
| live | `stop_half_width` | float64 scalar | scalar | Optional Exact one-tick override; Params fallback on miss | Full-rejection half-width. |
| live | `transition_width` | float64 scalar | scalar | Optional Exact one-tick override; Params fallback on miss | Raised-cosine transition width. |
| out | `rd_maps_out` | complex64 / complex128 | same as `rd_maps` | Required output | Notched maps with phase and layout preserved. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_doppler_bins`, `max_range_bins`; optional `max_rx` and `max_tx` admit named leading axes | Sizes Tensor pools and bounded device workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `center_velocity` | float64 | `0.0` m/s | `[live]` Exact | Center of the rejected interval. |
| `stop_half_width` | float64 | `0.0` m/s | `[live]` Exact | Non-negative full-rejection half-width. |
| `transition_width` | float64 | `0.0` m/s | `[live]` Exact | Non-negative raised-cosine width; zero is hard. |
| `cyclic_velocity_axis` | bool | `true` | static | Wrap distance over the aliased interval. |

## Algorithm

Distance from each velocity bin to the center is optionally wrapped over `D * abs(bin_spacing)`. Signed spacing locates the nearest bin without reordering the velocity axis or RD maps. The stop interval receives zero gain, a positive transition receives a raised-cosine ramp, and remaining bins receive unity. The CUDA stage broadcasts gain across channel and range axes.

## Shape, dtype, and timing rules

The velocity axis length equals `D`, matches template precision, and is finite and strictly monotonic (ascending or descending, matching the RD map order); cyclic mode also requires uniform spacing. No opaque leading batch is accepted. Live scalars join the map trigger with `Exact{}` and revert to Params after each tick.

## Configuration and caching

The plan contains validated layout and scalar notch values; no coefficient cache is required. Constructor extents determine admitted named leading rank and output pool capacity.

## Performance

The CUDA stage computes the original gain once per contiguous range row, casts it to the selected precision, and shares it within the block. Row-grid striding preserves larger extents, with barriers protecting reuse between rows. Per-cell multiplication and live-value resolution are unchanged; no persistent coefficient cache or device allocation was added. The original generic private stage remains available for partial rows.

A bounded Thor run on the pinned HSDK 5 deployment measured `noncyclic-mimo-large` (4 TX, 4 RX, 256 Doppler bins, 2048 range bins): fresh median latency fell from 5.8773 to 1.2034 ms in float and 6.2586 to 2.4958 ms in double. Median paired savings were 79.53% and 60.13%, respectively. Three paired screening runs and one fresh three-pair confirmation used five warmups and 50 frames per process. Six native recipe exports matched exactly in both precisions; public analytic tests cover capped row-grid striding, odd ranges, changing axes, and live overrides followed by Params fallback.

Small-case medians stayed within the declared max(1 microsecond, 3%) risk allowance with at least two of three pairs passing per phase. One fresh small-float process had a 48.898 microsecond median versus 40.949 microseconds for its paired original; this observation is retained rather than discarded. Setup, cold, and sampled tail results vary, so these scoped engineering measurements do not guarantee zero slowdown or population-tail behavior on every shape or deployment.
