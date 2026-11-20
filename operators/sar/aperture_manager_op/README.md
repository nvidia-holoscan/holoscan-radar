# `ApertureManagerOp`

## Purpose

Manage pulse membership, retention, emit cadence and collection-boundary flush.

## Responsibilities

- Track aperture membership and emit cadence across pulse batches and collection seams.
- Publish aligned pulse groups, signed eviction weights, staged geometry, and formation control.

## When to use

Use between range compression and backprojection when apertures span ticks or need a typed boundary flush. Retained video frames use signed weights for evictions; independent images reset on each collection. It owns pulse history and emits control, not an image.

Declarations and parameter defaults: [aperture_manager_op.hpp](aperture_manager_op.hpp). Behavioral requirements are documented below.

## Ports

`T=float` means complex64/float32; `T=double` means complex128/float64.

A second template parameter sets geometry precision:
`ApertureManagerOp<T, TGeom>` defaults `TGeom` to `T`. Position, reference,
and sample-point tensors use `TGeom`; retained profiles and amplitude
`weight` use `T`.

`<float, double>` halves retained sample storage compared with
`<double, double>` while keeping ECEF geometry in float64. At a coordinate
magnitude of 6.4e6 m, adjacent float32 values are 0.5 m apart, many wavelengths
at X band. `TGeom` may not be coarser than `T`.

The four required inputs join with `Exact{}` at the same capture time.
Optional `[live]` inputs also use same-tick `Exact{}`/`SafeDefault{}`.

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | `range_profiles` | complex `T` | `(num_pulses,num_range_bins)` | Required, aligned | Pulse data |
| In | `positions` | real `TGeom` | `(num_pulses,3)` | Required, aligned | Antenna phase-centre coordinates |
| In | `reference_position` | real `TGeom` | `(num_pulses,3)` | Required, aligned | Per-pulse scene reference point |
| In | `stream_boundary` | `uint8` | scalar | Required, aligned | `0`, `1`, or `2` |
| In | `sample_geometry` | `SampleGeometry` descriptor | descriptor | Optional `[live]` | staged lattice |
| In | `sample_points` | real `TGeom` | `(num_v,num_u,3)` | Optional `[live]` | staged points |
| In | `control` | `uint8` Tensor | `(1)` | Optional `[live]` | formation request |
| In | `evict` | `uint32` Tensor | scalar | Optional `[live]` | pulse count |
| Out | `range_profiles_out` | complex `T` | `(num_output_pulses,num_range_bins)` | Managed additions/departures or replay | Managed profiles in output membership order |
| Out | `positions_out` | real `TGeom` | `(num_output_pulses,3)` | Aligned with output profiles | Managed antenna positions |
| Out | `reference_position_out` | real `TGeom` | `(num_output_pulses,3)` | Aligned with output profiles | Managed scene reference points |
| Out | `weight` | real `T` | `(num_output_pulses)` | Optional, latched | signed contribution |
| Out | `sample_geometry_out` | `SampleGeometry` descriptor | descriptor | Optional, latched | staged geometry |
| Out | `sample_points_out` | real `TGeom` | `(num_v,num_u,3)` | Optional, latched | staged points |
| Out | `control_out` | `uint8` Tensor | `(1)` | Optional, latched | only on emit |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `ApertureManagerOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Positive `max_pulses`, `max_range_bins`, `max_image_v`, and `max_image_u` | Bound fat output tensors before arm |

Set `max_pulses` for the **largest managed output batch**, including arrivals
plus evictions or a retained replay, and `max_range_bins` for the profiles.
`max_image_v` and `max_image_u` bound `sample_points_out` and are required
during setup even when that output is unwired. All four bounds are captured
at `graph.op()` before arm. For example, a graph with at most 2048 managed
pulses, 4096 range bins, and a 512-by-512 points grid can use:

```cpp
MaxTensorExtents extents{.max_pulses = 2048,
                         .max_image_u = 512,
                         .max_image_v = 512,
                         .max_range_bins = 4096};
auto manager = graph.op<ApertureManagerOp<float, double>>("aperture_manager", extents);
```

Size these values from the application's actual maximum output and image
extents; this is a construction fragment for an existing Graph.

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `max_aperture_pulses` | `int64` | 0 | Static | Retention capacity |
| `emit_interval_pulses` | `int64` | 0 | Static | Automatic emit cadence |
| `emit_mode` | `ApertureEmitMode` | continue | Static | Continue, reset, or reform |
| `aperture_window` | `WindowSpec` | none (rectangular, all +1) | Static | Pulse weights for an aperture |

## Membership and emit rules

The required `range_profiles`, `positions`, `reference_position`, and
`stream_boundary` join at the same capture time. `stream_boundary` is the
reader's scalar `0` (none), `1` (collection boundary), or `2` (end of
stream). A nonzero boundary forces the closing aperture to emit and reset.
An explicit `control` value or `emit_interval_pulses` can also request an
image; the manager emits `control_out` only on an emit tick. Backprojection
uses that signal to publish an image.

With `max_aperture_pulses: 0`, incoming pulses pass through without retention.
A positive capacity retains pulse groups in manager-owned CUDA storage,
releasing the upstream reader and range-compression pool slots after each
tick. In `continue` mode, admission of a new group can evict old pulses;
the output batch contains the new pulses
followed by evictions. The optional `weight` is `+1` for additions and `-1`
for evictions, so a continuing backprojector can remove their contribution.
`reset` closes the aperture after emission. `reform` replays its retained
aperture at emission, optionally with the configured aperture window.
`aperture_window` is legal only where final membership is known: under
`reform`, or stamped at admission with `max_aperture_pulses: 0`,
`emit_mode: reset` and `emit_interval_pulses > 0`. Anywhere else, including
every sliding window, `start()` rejects it. A stamped aperture that closes
short of `emit_interval_pulses` warns that its window was truncated.
An explicit `evict` input requests a pulse count to evict; a count above
occupancy is rejected. A tick-local `sample_geometry` or `sample_points`
may be staged and passed through to connected outputs.
A full required-output queue fails `compute()`. Descriptor, points, control,
and weight outputs stay latched.

`sample_geometry`, `sample_points`, `control`, and `evict` are optional
same-tick `[live]` inputs; a missing tick uses the static/default behavior.
The descriptor and points outputs are latched after an unwired first
emit, avoiding later dedicated output work. The output
pulse tensors can be larger than the incoming batch when eviction or replay
is enabled; size `MaxTensorExtents.max_pulses` for that output bound.

## Validation and use

The input pulse axes must agree, with `(num_pulses, num_range_bins)` profiles
and `(num_pulses, 3)` positions and references. Tensors must be contiguous,
CUDA accessible, and match the template precision. The manager rejects an
unknown boundary/control value, a batch larger than retention capacity,
invalid capacity/cadence, `reform` without retention, and output pulse counts
above the pool bound. Example for one image at each collection seam:

```yaml
aperture_manager:
  max_aperture_pulses: 0
  emit_interval_pulses: 0
  emit_mode: reset
```

The C++ tests cover partial final batches, collection and EOS boundaries,
retention, explicit evictions, signed weights, and reset/reform behavior.
A generated 20-pulse video app smoke exercises retention beyond the upstream
Tensor-pool depth and checks five emitted images.
Shared rules are in [architecture](../../../docs/ARCHITECTURE.md).

## Remaining validation and implementation limits

Image-level reform equivalence against direct formation and more boundary-preservation tests through splitting/coalescing stages remain useful qualification work. Historical metadata-disabled cases must be recast for the required typed `stream_boundary` input; this operator no longer reads a string metadata key.

Constructor checks require pulse/range bounds, while setup also dereferences image extents for `sample_points_out`; callers must supply both image bounds. Scalar control/evict reads check shape and byte size but do not fully validate dtype, and the optional points path lacks the geometry dtype check used for positions. Supply the declared dtypes; these validation gaps remain follow-up work. The cadence counter resets inside the active `control_out` branch; with that output unwired the continuing-mode internal counter is not reset on cadence. This does not affect the supplied graphs, which wire control. Preserve these behaviors pending a focused fix and tests.
