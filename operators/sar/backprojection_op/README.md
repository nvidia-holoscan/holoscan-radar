# `BackprojectionOp`

## Purpose

Accumulate complex SAR images from weighted range profiles and 3-D pulse geometry.

## Responsibilities

- Accumulate phase-corrected pulse contributions on the declared image geometry.
- Emit a complex image on control, resetting or scrolling retained pixels as configured.

## When to use

Use after aperture management to form a monostatic image on a SampleGeometry lattice or explicit sample points. Reset geometry gives independent images; scroll geometry retains overlapping image rows as the window advances. Choose a backend statically and provide the phase-reference frequency matching range compression.

Declarations and parameter defaults: [backprojection_op.hpp](backprojection_op.hpp). Behavioral requirements are documented below.

## Ports

`T=float` means complex64/float32; `T=double` means complex128/float64.

A second template parameter sets geometry precision:
`BackprojectionOp<T, TGeom>` defaults `TGeom` to `T`. `positions`,
`reference_position`, and `sample_points` use `TGeom`; profiles, images,
`range_bins`, and `weight` use `T`.

`<float, double>` keeps ECEF geometry in float64 while using single-precision
samples/images. At a coordinate magnitude of 6.4e6 m, adjacent float32 values
are 0.5 m apart, many wavelengths at X band. `TGeom` may not be coarser than `T`.

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
| `reference_frequency` | `double` | required positive | [live] | Phase-reference frequency matching range compression |
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
metre units. `reference_frequency` is the frequency rotated to IFFT DC by range compression: **stored sample `floor(num_samples / 2)`** for the supplied FX path. The SAR apps derive it as `SC0 + floor(num_samples / 2) * SCSS`, matching the range stage. The unrotated first-sample SC0 convention is not interchangeable with it.

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
the [qualification notes](#remaining-qualification-and-contract-questions).

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
[shared architecture](../../../docs/ARCHITECTURE.md).

## Remaining qualification and contract questions

All backend names being dispatched does not establish numerical parity: qualify precision, even-length origin/phase, and real-data focus for each selected NVRadar/MatX variant. The z0 backend is for planar scenes. Do not compensate origin differences with an ad hoc R0 or profile shift.

Explicit-points mode is remembered on first matching `sample_points` arrival rather than inferred from wiring; a missing first sample can therefore select lattice mode. A scroll tick missing its live lattice falls back to the original Params lattice and can request an invalid backward scroll. Whether a scrolling miss should mean no change requires contract review; this cleanup does not change Exact/SafeDefault semantics. Publish the current lattice for each tick.

The public image is rank 2 and named axes admit no opaque batch. Historical internal per-batch accumulator code remains; it is not a supported multi-image API. Review that unreachable path before removing it or proposing collection batching. The old R11 discussion is available at Git revision `6c261de7c153daa0c8a6be9d6829b149cfbdf41c`, path `operators/sar/backprojection_op/OPEN_ITEMS.md`. `initial_image`, nearest/sinc interpolation and `matx_taylor_fast` remain declared but rejected as documented above.
