# radar_common

Shared library for the radar operator groups: the module's config-plane
vocabulary, host-side signal math, tensor plumbing, and Holoscan-integration
glue. Everything lives in the `holoscan::radar` namespace and is included as
`radar_common/<layer>/<header>.hpp`.

## Layout (by dependency layer)

Each folder depends only on the layers above it:

| Folder | Contents | Depends on |
| --- | --- | --- |
| `radar_common/descriptors/` | Shared config-plane types (`WindowSpec`, `FMCWWaveformDescriptor`, …), `MaxTensorExtents` (ctor-capture pool sizing; not a descriptor port type), and the contract enums (`WindowType`, `FftNorm`, `ClutterFilterMethod`) | STL only |
| `radar_common/dsp/` | Host-side signal math (window/taper generation) | descriptors |
| `radar_common/sim/` | Analytic FMCW signal simulators — single-TX `point_target_sim.hpp` (multi-RX `SimScene`) and MIMO TDM/DDM `mimo_sim.hpp` — plus antenna-array geometry builders `array_builders.hpp` (input synthesis for applications; analytic ground truth for tests) | descriptors |
| `radar_common/tensor/` | Rank-agnostic batching, DLPack device buffers, MatX view/upload helpers | holoscan::Tensor, CUDA |
| `radar_common/sdk/` | Operator-authoring glue: HSDK 5 Tensor I/O + Exact `[live]` helpers, warn-once config cache (4.x argument-setter path retained only until remaining ops port) | holoscan core |
| `radar_common/reduce/` | Public group-by-label CUDA primitives (headers + `radar_common_reduce` lib) | CUDA, descriptors/enums |

`tensor/matx_utils.hpp` includes `<matx.h>` and must only be included from
CUDA translation units (`.cu` files and the stage headers they include).

`reduce/group_by_label.hpp` declares host launch wrappers; non-template
kernels are compiled once into `holoscan::radar_common_reduce`. Link that
target rather than recompiling `group_by_label.cu` in a second library
(ODR). Template custom-peak launches live in the header (`__CUDACC__`).

## Module conventions

These conventions apply to every operator in the module and are implemented
by this library:

- **Precision by template instantiation.** Operators are
  `template <typename T>` with `T` ∈ {`float`, `double`}; an instance's
  inputs and outputs match its precision (complex64/float32 or
  complex128/float64). Never a runtime dtype flag. The descriptor structs
  themselves are precision-agnostic (`double` scalars); operators cast when
  building device-side artifacts.
- **Plain-tensor data ports.** [EA HSDK5 ONLY] Ports are typed
  `Input<Tensor>` / `Output<Tensor>` carrying a plain `holoscan::Tensor`
  (no `TensorMap`). Port names must be unique within an operator — inputs
  and outputs share one namespace, so a same-named in/out pair is illegal
  (e.g. `ClutterFilterOp` emits `range_profiles_out`). No bespoke message
  types — shared descriptors stay parameters and ride input ports only as
  `[live]` overrides.
- **Rank-agnostic batching.** Operators act on the trailing axes and treat
  leading `...` axes as opaque batch dimensions (`tensor/tensor_utils.hpp`:
  `collapse_batch` / `full_rank_shape`). Shapes are inferred from the
  tensors each tick, so shape-changing overrides are legal — downstream
  re-infers. Inputs must be compact row-major in this wave.
- **Overridable ("live") parameters.** [EA HSDK5 ONLY] An overridable
  parameter may be changed in-flight via a same-named typed
  `Input<T>` (`sdk/override_param.hpp`). Every such port joins the data
  trigger with `Exact{}` + `SafeDefault{}` unconditionally; `sticky_*`
  (default true) latches the last Exact match in `compute()` via
  `resolve_live_override`. An unwired port reports `kNotConnected` and reads
  as no override, so one build serves both wirings. Costly reconfigurations
  still warn once and cache via `sdk/config_cache.hpp`; the cache decides
  that itself (a miss against a non-empty cache), so callers pass no warn
  flag. Container-owning FMCW window/waveform lives use generated
  `holoscan::radar::wire` descriptor-Tensor payloads and validated bridges;
  Python remains Params-only for those physical ports. See
  [`SCHEMA_PAYLOADS.md`](../../docs/planning/SCHEMA_PAYLOADS.md).
- **Params schema replaces ArgumentSetter.** [EA HSDK5 ONLY] Ported
  operators use `Operator<Params>` + `HOLOSCAN_PARAM_SCHEMA` (YAML /
  `set_params` / `from_config`). Drop `HOLOSCAN_OPERATOR_FORWARD_ARGS`
  and `ensure_radar_arg_setters()` for those ops. Unported operators keep
  the 4.x setter path until they are allow-listed.

## HSDK 5 pilot patterns ([EA HSDK5 ONLY])

Patterns established by the `ClutterFilterOp` port — copy these for the next
operators; remove every `[EA HSDK5 ONLY]` fence before promoting the HSDK 5
branch to `main`. Open EA / migration issues live in
[`docs/planning/HSDK5_PORT.md`](../../docs/planning/HSDK5_PORT.md).

| Concern | Pattern |
| --- | --- |
| Allow-list | `cmake/HoloscanRadarOptionalFeatures.cmake` → `HOLOSCAN_RADAR_HSDK5_PORTED_OPERATORS`; groups call `holoscan_radar_add_operator()` |
| Receive | `receive_tensor(Input<Tensor>&, who)` → `expected<Tensor, Error>` |
| Emit | `emit_device_tensor(Output<Tensor>&, context, shape, dtype, fill)` — pool loan → write on `context.cuda_stream()` → commit → emit |
| Stream | `context.cuda_stream()` only (no `receive_cuda_stream` / `set_cuda_stream`) |
| Exact `[live]` | `add_live_override_to_contract` + `resolve_live_override`; the entry is unconditional (`maybe_add_live_override_to_contract` + `expect_*` only under an `OnAligned` trigger) |
| Writable test tensors | `dlpack_buffer.hpp` builds a `DLManagedTensorContext` + device alloc — raw `DLManagedTensor*` import is permanently read-only in HSDK 5 |
| C++ tests | `tests/cpp/utils/sdk_harness.hpp`: `TensorSourceOp` / `TensorCaptureOp` + `Graph` compile/`run_async`; device producers need `bind_tensor_output_device` |

Device Tensor producers that skip `bind_tensor_output_device` fail at run
time with `RUNTIME_PORT_BINDING_UNAVAILABLE`. Graph mid-stream enum
override sources (separate clock / multi-partition) are deferred until EA
partition + device-binding behavior is settled; unit coverage for sticky
method change on ClutterFilter is skipped with that note.
