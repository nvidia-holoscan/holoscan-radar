# `radar_common`

Shared types and utilities used by every Holoscan Radar operator. The library
contains configuration descriptors, host-side DSP and simulation helpers,
tensor/SDK adapters, and reusable reductions in the `holoscan::radar`
namespace.

## Layout

| Folder | User-facing role |
| --- | --- |
| `descriptors/` | Shared descriptors, parameter enums, wire conversions, and `MaxTensorExtents` |
| `dsp/` | Host-side window, taper, grid, and source-count helpers |
| `sim/` | Independent analytic SISO/SIMO/TDM/DDM signal models and array builders |
| `tensor/` | DLPack, CUDA, and MatX tensor utilities |
| `sdk/` | Typed Holoscan Tensor I/O, live-override, output, and configuration-cache helpers |
| `reduce/` | Reusable GPU group-by-label reductions |

Headers under any `detail/` directory, such as `tensor/detail/matx_views.hpp`,
are private: they are not installed and may include only from CUDA translation
units (`.cu`) and `*_stage.hpp` headers. Consumers of the compiled
reduction kernels should link `holoscan::radar_common_reduce` rather than
compiling `group_by_label.cu` again.

## Shared descriptor types

The public configuration plane has seven descriptors:

- `ArrayGeometry`
- `ShiftInvariantSubarrayPair`
- `AngleConvention`
- `WindowSpec`
- `FMCWWaveformDescriptor`
- `DDMCodeDescriptor`
- `TXSchedule`

They are ordinary parameters and may also use same-named live input ports where
an operator documents a live override. `MaxTensorExtents` is different: it is a
constructor sizing object used to bound tensor pools before graph activation;
it never travels on a port.

## `MaxTensorExtents`

Fat-output operators capture `MaxTensorExtents` at construction so HSDK 5 can
size Tensor pools before `arm()`. The field list in
`radar_common/descriptors/max_tensor_extents.hpp` is the public sizing surface;
each operator README Construction table identifies the fields that operator
requires. Applications may construct the value directly or decode the same
`max_*` keys with `MaxTensorExtentsFromYaml` from
`radar_common/sdk/max_tensor_extents_yaml.hpp`.

The values are maxima for the graph lifetime, not fixed per-tick shapes.
Operators continue to infer named extents from each input and accept shape
changes within the captured bounds. An absent required field, a non-positive
value, or a tick exceeding a bound fails closed. There is deliberately no
opaque leading-batch extent; application-level loop or fan-out owns multi-item
throughput.

Size every field from the real application or test scenario. Pool dimensions
multiply, so generous unrelated defaults can request many GiB; on unified-
memory systems that can exhaust host RAM and make the machine unavailable.
Keep concurrent GPU test jobs low enough that their combined pools fit physical
memory.

## Operator conventions

- Precision is selected by the C++ template (`float` or `double`), never by a
  runtime flag. Real and complex ports use the matching precision.
- Data ports carry plain `holoscan::Tensor` values. Descriptor sidebands use
  their documented wire-schema types; the public logical type remains the
  descriptor named in the parameter table.
- Inputs use the contract's named axes. Operators infer their extents each tick
  but do not accept arbitrary opaque leading batch axes unless the individual
  operator explicitly documents them.
- Input and output names are unique within an operator. Shape-preserving
  operators therefore commonly use an `_out` suffix on the output.
- A `[live]` parameter may be replaced for one tick through its same-named
  optional input. The side input joins the data trigger with `Exact{}`; a miss
  uses the configured parameter and never latches the prior override.
- Expensive configuration changes may build and cache FFT plans, steering
  weights, lookup tables, or window coefficients. Repeated keys reuse cached
  artifacts.
- Optional outputs use either `always` or `latched` behavior as stated in the
  port table. An `always` output attempts publication every tick; a `latched`
  output stops doing dedicated work after the runtime reports it unwired.
- GPU work runs on the activation's CUDA stream. Inputs may be retained after
  `compute()` returns when asynchronous kernels still read them.

The locked interface definitions are in
[`OPERATOR_API.md`](../../docs/planning/OPERATOR_API.md), while each operator
README gives the compact user-facing port, construction, and parameter tables.
