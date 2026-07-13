# radar_common

Shared library for the radar operator groups: the module's config-plane
vocabulary, host-side signal math, tensor plumbing, and Holoscan-integration
glue. Everything lives in the `holoscan::radar` namespace and is included as
`radar_common/<layer>/<header>.hpp`.

## Layout (by dependency layer)

Each folder depends only on the layers above it:

| Folder | Contents | Depends on |
| --- | --- | --- |
| `radar_common/descriptors/` | Shared config-plane types (`WindowSpec`, `FMCWWaveformDescriptor`, …) and the contract enums (`WindowType`, `FftNorm`, `ClutterFilterMethod`) | STL only |
| `radar_common/dsp/` | Host-side signal math (window/taper generation) | descriptors |
| `radar_common/sim/` | Analytic point-target FMCW signal simulator (input synthesis for applications; analytic ground truth for tests) | STL only |
| `radar_common/tensor/` | Rank-agnostic batching, DLPack device buffers, MatX view/upload helpers | holoscan::Tensor, CUDA |
| `radar_common/sdk/` | Operator-authoring glue: parameter override ports, warn-once config cache, argument-setter registration | holoscan core |

`tensor/matx_utils.hpp` includes `<matx.h>` and must only be included from
CUDA translation units (`.cu` files and the stage headers they include).

## Module conventions

These conventions apply to every operator in the module and are implemented
by this library:

- **Precision by template instantiation.** Operators are
  `template <typename T>` with `T` ∈ {`float`, `double`}; an instance's
  inputs and outputs match its precision (complex64/float32 or
  complex128/float64). Never a runtime dtype flag. The descriptor structs
  themselves are precision-agnostic (`double` scalars); operators cast when
  building device-side artifacts.
- **Plain-tensor data ports.** Ports carry a single-entry
  `holoscan::TensorMap`; the port name is the contract, the map key is
  accepted leniently. No bespoke message types — the shared descriptor
  structs are parameters, riding input ports only as overrides.
- **Rank-agnostic batching.** Operators act on the trailing axes and treat
  leading `...` axes as opaque batch dimensions (`tensor/tensor_utils.hpp`:
  `collapse_batch` / `full_rank_shape`). Shapes are inferred from the
  tensors each tick, so shape-changing overrides are legal — downstream
  re-infers. Inputs must be compact row-major in this wave.
- **Overridable ("live") parameters.** An overridable parameter may be
  changed in-flight via a same-named non-blocking input port
  (`sdk/override_param.hpp`). Overrides are **sticky**: the parameter seeds
  the initial effective value and an arriving override remains effective
  until the next override. Where an override incurs a genuine cost (FFT
  plan, window coefficients), the operator warns once per new configuration
  and caches prior ones (`sdk/config_cache.hpp`) so alternating values do
  not thrash; the initial parameter-driven build is silent.
- **Custom parameter types need registered argument setters.** Holoscan's
  `ArgumentSetter` only knows built-in types. Every operator therefore
  calls `holoscan::radar::ensure_radar_arg_setters()`
  (`sdk/arg_setters.hpp`) from an `initialize()` override, which registers
  the SDK's default setter for each shared descriptor type. Both native
  C++ Args and YAML-sourced Args (`from_config()`) work; the YAML side
  decodes through the `YAML::convert<T>` specializations in
  `sdk/yaml_converters.hpp`, using the canonical spellings from
  `descriptors/enums.hpp` (e.g. `method: mean_subtraction`,
  `range_window: hann` or a full `WindowSpec` map).
