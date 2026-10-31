# `BackprojectionOp`

## Purpose

Accumulate complex SAR images from weighted range profiles and 3-D pulse geometry.

## Responsibilities

- Accumulate phase-corrected pulse contributions on the declared image geometry.
- Emit a complex image on control, resetting or scrolling retained pixels as configured.

## When to use

Use after aperture management to form a monostatic image on a SampleGeometry lattice or explicit sample points. Reset geometry gives independent images; scroll geometry retains overlapping image rows as the window advances. Choose a backend statically and provide the first fast-time sample frequency.

## Ports

`T=float` means complex64/float32; `T=double` means complex128/float64.

A second parameter sets the geometry precision: `BackprojectionOp<T, TGeom>`
defaults `TGeom` to `T`, and the geometry ports take `TGeom` while samples and
the image take `T`. `<float, double>` halves the retained aperture while
keeping positions in float64, which they need: ECEF coordinates do not survive
float32, where one ulp at a 6.4e6 m radius is 0.76 m -- some 25 wavelengths at
X band. `range_bins` and `weight` stay on `T`, because their producers are
sample-precision. `TGeom` may not be coarser than `T`.
The three pulse tensors are required and aligned. All optional `[live]`
inputs are per-tick `Exact{}`/`SafeDefault{}` joins; `control` is the optional
emit trigger and is normally supplied by `ApertureManagerOp.control_out`.

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | `range_profiles` | complex `T` | `(num_pulses,num_range_bins)` | Required, aligned | Pulse data |
| In | `positions` | real `TGeom` | `(num_pulses,3)` | Required, aligned | Antenna phase-centre coordinates |
| In | `reference_position` | real `TGeom` | `(num_pulses,3)` | Required, aligned | Per-pulse scene reference point |
| In | `control` | `uint8` Tensor | `(1)` | Optional `[live]` | emit/reset command |
| In | `range_bins` | real `T` | `(num_range_bins)` | Optional `[live]` | uniform explicit axis |
| In | `sample_points` | real `TGeom` | `(num_v,num_u,3)` | Optional `[live]` | explicit pixel positions |
| In | `weight` | real `T` | `(num_pulses)` | Optional `[live]` | signed pulse weights |
| In | `initial_image` | complex `T` | `(num_v,num_u)` | Declared `[live]` but rejected | Reserved, unsupported initializer |
| In | `sample_geometry` | `SampleGeometry` descriptor | descriptor | Optional `[live]` | regular lattice |
| In | `range_bin_spacing` | `double`, metres | scalar | Optional `[live]` | required without range bins |
| In | `range_bin_origin` | `double`, metres | scalar | Optional `[live]` | centred default if unset |
| In | `reference_frequency` | `double`, Hz | scalar | Optional `[live]` | positive value required |
| In | `interpolation` | `RangeInterpolation` | scalar | Optional `[live]` | `linear` supported |
| Out | `image` | complex `T` | `(num_v,num_u)` | On emit command only | Formed complex image |
| Out | `sample_geometry_out` | `SampleGeometry` descriptor | descriptor | Optional, latched | on lattice image emit |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `BackprojectionOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

Set `max_pulses`, `max_range_bins`, `max_image_v`, and `max_image_u` for the largest pulse block and image lattice (including later collection geometries). These are pool bounds captured at `graph.op()`, not runtime sizing parameters.

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `coordinate_frame` | `CoordinateFrame` | required | Static | Cartesian frame declaration |
| `sample_geometry` | `SampleGeometry` | required for lattice | [live] | Image grid |
| `range_bin_spacing` | `optional<double>` | unset | [live] | Differential-range bin spacing |
| `range_bin_origin` | `optional<double>` | unset | [live] | Differential-range first bin |
| `reference_frequency` | `double` | required positive | [live] | First fast-time sample frequency |
| `interpolation` | `RangeInterpolation` | linear | [live] | Range lookup method |
| `backend` | `BackprojectionBackend` | NVRadar LUT vector | Static | CUDA arithmetic strategy |
| `geometry_change` | `GeometryChange` | reset | Static | Reset or scroll lattice |

## Formation and accumulation

Each pulse contributes a phase-adjusted sample of its differential-range
profile to each image point. The stage computes the path from antenna phase
centre to sample point relative to the pulse reference point, finds the
corresponding range bin, linearly interpolates its complex value, applies
`reference_frequency` phase and optional pulse `weight`, and accumulates.
Positions and sample points must share the declared Cartesian frame and
metre units. `reference_frequency` is the **first fast-time sample frequency**
for this range convention; the SAR apps derive it from CPHD `SC0`/`SCSS`
for their chosen range-compression setup.

`control` drives image emission and reset/continuation. An unwired or missing
control means the stage accumulates without emitting; stopping with pending
pulses warns. A full `image` queue fails `compute()`; an
`emit_and_reset` tick resets the accumulator only after that emit succeeds. With `geometry_change: reset`, a changed lattice discards the
old accumulation. With `scroll`, a forward shift along the lattice v axis
retains overlapping rows and clears newly exposed rows. Scroll disallows
signed `weight`. `range_bins` can supply a uniform explicit range axis;
otherwise spacing/origin come from `[live]` inputs or Params. The optional
`sample_geometry_out` is emitted only when an image is emitted, a lattice
was used, and the output is connected.

`T` fixes the complex profile/image types; `TGeom` fixes the real geometry
types on `positions`, `reference_position` and `sample_points`.
All `[live]` inputs are joined per tick with `Exact{}` and a Params fallback
when absent. `sample_geometry`, range spacing/origin, and frequency changes
on a closing `emit_and_reset` tick are staged for the next aperture, so that
collection's image uses its own geometry and frequency. The optional
`sample_geometry_out` describes the emitted image, even when the next lattice
is staged on that closing tick; mosaic consumers pair it with `image`.

## Supported choices and failures

The shipped backend is
`nvradar_block_anchored_expanded_throughput_phase_lut_vec`; see the contract
for the other accepted backends. `linear` interpolation is implemented.
`nearest`, `sinc`, `initial_image`, and `matx_taylor_fast` are declared but
rejected. The operator validates positive reference frequency and range
spacing, uniform `range_bins`, matching pulse/position axes, sample-points
shape `(num_v,num_u,3)`, and image bounds. A missing geometry or CUDA stream
is an error. Numerical backend and radiometric parity questions remain in
the [SAR open-items roll-up](../../../docs/planning/SAR/OPEN_ITEMS.md).

## Use and verification

```yaml
backprojection:
  coordinate_frame: ecef
  geometry_change: reset
```

An app must also provide `sample_geometry` or sample points, range spacing
(or uniform `range_bins`), and a positive reference frequency. The C++ tests
compare analytic point targets and exercise reset, scroll, backend selection,
control values, and invalid axes. See the
[SAR operator contract](../../../docs/planning/SAR/OPERATOR_API.md).
