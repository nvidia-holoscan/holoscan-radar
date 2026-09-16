# `DopplerNotchOp`

Post-Doppler physical-velocity stopband for already formed complex
range-Doppler tensors. It is generic across SISO, SIMO, TDM, and dense DDM,
although the first application integration is dense DDM after
`DDMDenseDemultiplexOp`.

```text
DDMDenseDemultiplexOp → DopplerNotchOp → VirtualArrayAssemblyOp
```

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(D,R)`, `(N_rx,D,R)`, or `(M_tx,N_rx,D,R)` | Required aligned data |
| in | `velocity_bins` | float32 / float64 | `(D,)` | Required physical m/s axis, matching precision |
| out | `rd_maps_out` | input dtype | input shape | Same gain for every channel |
| live | `center_velocity` | float64 | scalar | Exact one-tick override |
| live | `stop_half_width` | float64 | scalar | Exact one-tick override |
| live | `transition_width` | float64 | scalar | Exact one-tick override |

`center_velocity`, `stop_half_width`, and `transition_width` are in m/s.
Every bin inside the stop half-width is zeroed, and the bin nearest the center
is always zeroed. A zero transition width is a hard notch. A positive
transition width applies the gain
`0.5 * (1 - cos(pi * distance_into_transition / transition_width))`.
When `cyclic_velocity_axis` is true, distance wraps over the uniformly spaced
aliased velocity interval.

The operation multiplies each TX/RX channel by the same real gain at a given
Doppler bin, so surviving cells retain relative channel phase. There is no
pass-through method; omit the operator from the graph to disable it.

This is not a replacement for pre-Doppler `ClutterFilterOp`, which acts on
slow-time range profiles. It also must not consume folded sparse-DDM power:
one folded bin represents multiple physical velocities.

## Implementation

The HSDK 5 shell validates memory, dtype, named rank, axis length, and extents,
then launches one stream-ordered CUDA kernel over the same-shape output pool.
The kernel derives one cyclic physical distance per Doppler bin and broadcasts
its gain over named channel and range axes. Both inputs are retained across the
next tick for asynchronous lifetime safety.

The required inputs meet under a zero-window `OnAligned` trigger with
`Exact{}` timestamp matching. Live scalar overrides use `Exact{} +
SafeDefault{}` and revert to Params after a miss. No notch-mask output exists
in v1; its statistical interaction with CFAR is tracked in
[`OPEN_ITEMS.md`](OPEN_ITEMS.md).

Contract: [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) · tests:
[`TESTING.md`](../../../docs/planning/TESTING.md) · pool sizing:
[`MAX_TENSOR_EXTENTS.md`](../../../docs/planning/MAX_TENSOR_EXTENTS.md)
