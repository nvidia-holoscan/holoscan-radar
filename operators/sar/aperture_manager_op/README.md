# `ApertureManagerOp`

## Purpose

Manage pulse membership, retention, emit cadence and collection-boundary flush.

## Responsibilities

- Apply the public SAR operator contract and report invalid shapes or configurations.
- Preserve capture-time alignment and output the documented tensor or descriptor payloads.

## When to use

Use between range compression and backprojection when apertures span ticks or need a typed boundary flush. Retained video frames use signed weights for evictions; independent images reset on each collection. It owns pulse history and emits control, not an image.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | range_profiles | complex T | (num_pulses,num_range_bins) | required | Incoming profiles |
| In | positions, reference_position | real T | (num_pulses,3) | required | Pulse geometry |
| In | stream_boundary | uint8 | scalar | required Exact | Collection seam |
| In | sample_geometry, sample_points, control, evict | descriptor / Tensor | named axes | optional [live] | Geometry and app control |
| Out | range_profiles_out, positions_out, reference_position_out | complex / real T | matching pulse axes | every tick | Managed pulse block |
| Out | weight, sample_geometry_out, sample_points_out, control_out | real T / descriptor / uint8 | named axes | optional by wire | Weights, geometry and control |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `ApertureManagerOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `max_aperture_pulses` | `int64` | 0 | Static | Retention capacity |
| `emit_interval_pulses` | `int64` | 0 | Static | Automatic emit cadence |
| `emit_mode` | `ApertureEmitMode` | continue | Static | Continue, reset, or reform |
| `aperture_window` | `WindowSpec` | rectangular | Static | Pulse weights for an aperture |

The full shapes, defaults, numerical conventions, and open decisions are in
[SAR OPERATOR_API.md](../../../docs/planning/SAR/OPERATOR_API.md).
