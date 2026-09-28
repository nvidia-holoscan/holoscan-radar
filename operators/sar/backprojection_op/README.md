# `BackprojectionOp`

## Purpose

Accumulate complex SAR images from weighted range profiles and 3-D pulse geometry.

## Responsibilities

- Apply the public SAR operator contract and report invalid shapes or configurations.
- Preserve capture-time alignment and output the documented tensor or descriptor payloads.

## When to use

Use after aperture management to form a monostatic image on a SampleGeometry lattice or explicit sample points. Reset geometry gives independent images; scroll geometry retains overlapping image rows as the window advances. Choose a backend statically and provide the first fast-time sample frequency.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | range_profiles | complex T | (num_pulses,num_range_bins) | required | Compressed history |
| In | positions, reference_position | real T | (num_pulses,3) | required | APC and SRP |
| In | control, range_bins, sample_points, weight, initial_image | Tensor | named axes | optional [live] | Formation controls |
| In | sample_geometry, range_bin_spacing, range_bin_origin, reference_frequency, interpolation | descriptor / scalar | scalar | optional [live] | Tick-local geometry and range |
| Out | image | complex T | (num_v,num_u) | on emit | Formed image |
| Out | sample_geometry_out | descriptor | scalar | optional on emit | Output lattice |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `BackprojectionOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `coordinate_frame` | `CoordinateFrame` | required | Static | Cartesian frame declaration |
| `sample_geometry` | `SampleGeometry` | required for lattice | [live] | Image grid |
| `range_bin_spacing` | `optional<double>` | unset | [live] | Differential-range bin spacing |
| `range_bin_origin` | `optional<double>` | unset | [live] | Differential-range first bin |
| `reference_frequency` | `double` | required positive | [live] | First fast-time sample frequency |
| `interpolation` | `RangeInterpolation` | linear | [live] | Range lookup method |
| `backend` | `BackprojectionBackend` | NVRadar LUT vector | Static | CUDA arithmetic strategy |
| `geometry_change` | `GeometryChange` | reset | Static | Reset or scroll lattice |

The full shapes, defaults, numerical conventions, and open decisions are in
[SAR OPERATOR_API.md](../../../docs/planning/SAR/OPERATOR_API.md).
