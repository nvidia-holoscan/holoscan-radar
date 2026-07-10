# Signal Processing Operators — Implementation Plan

Implementation plan for the four FMCW signal-processing operators on branch
`deustice/sigproc-ops`. Contracts are locked in
[`OPERATOR_API.md`](OPERATOR_API.md) (Signal Processing Operators + Supporting
data structures); code must match them exactly. Test requirements come from
[`TESTING.md`](TESTING.md).

**Status (2026-07-06): all four phases complete** on `deustice/sigproc-ops`.
Notable deltas from the plan as written: an SDK 5.x migration was attempted
and rolled back mid-effort (the pre-release lacks the operator framework;
the module stays on 4.3); [live] overrides were clarified as sticky in
`OPERATOR_API.md` (contract clarification); the split-vs-fused parity
measured bitwise-identical (delta exactly 0.0); the YAML-converter
deferral was later lifted on this branch (`YAML::convert<>`
specializations in `radar_common/sdk/yaml_converters.hpp`, registered
through the SDK default setter in `radar_common/sdk/arg_setters.hpp`, so
custom parameter types work through both native Args and `from_config()`;
the SISO app consumes the per-operator YAML blocks natively).

## Scope and sequencing

1. **Phase 0 — Foundation** (sequential): shared `operators/common/` library,
   MatX/CUDA build wiring, `operators/fmcw/` skeletons with complete `setup()`
   and stub `compute()`, analytic point-target simulator + test harness. All
   shared/contended files land here and are then frozen.
2. **Phase 1 — Parallel implementation**: three independent agents implement
   `FMCWRangeProcessorOp`, `ClutterFilterOp`, `FMCWDopplerProcessorOp` with
   strictly disjoint file ownership.
3. **Phase 2 — Fused operator**: `FMCWRangeDopplerProcessorOp`, built on the
   split operators' stage functions; parity test per `TESTING.md`.
4. **Phase 3 — Integration**: remove the `radar_op` placeholder, rewire the
   example app/tests/packaging, cross-cutting tests, final status update.

Decisions already applied to the planning docs (2026-07-06):

- Layout: `operators/fmcw/<op>/` for signal processing + MIMO marshaling;
  future `operators/detection/`, `operators/clustering/`,
  `operators/angle_estimation/`; shared types in `operators/common/`.
- Python bindings deferred to a later phase — C++ operators + tests only.
- Contract clarification: `apply_dechirp = true` requires complex-IF input
  (real-IF input errors). Flag as a contract change in the PR.

## Verified environment facts

- The Holoscan v4.3 container (`holoscan-radar:ngc-v4.3.0-cuda13`, already
  pulled locally) ships MatX v0.9.4; `find_package(holoscan)` chains
  `find_dependency(MATX)` and provides the `matx::matx` target. No
  FetchContent needed.
- No host Holoscan SDK; builds/tests run in-container. Host has CUDA 13 and an
  RTX 3500 Ada GPU (`docker run --gpus all` works), so GPU tests run locally.
- SDK 4.3 provides `receive_cuda_stream()` / `set_cuda_stream()` and
  non-blocking ports via `IOSpec::condition(ConditionType::kNone)`. The SDK
  MatX example shows zero-copy in
  (`matx::make_tensor<T>(tensor->data(), ...)`) and out
  (`holoscan::Tensor(matx_tensor.ToDlPack())`).

## Architecture decisions

1. **Precision**: `template <typename T>` operator classes (`T` in
   {float, double}, `static_assert`ed), explicit instantiation in the `.cu`,
   `extern template` in the header. C++ users write
   `make_operator<FMCWRangeProcessorOp<float>>(...)`. Public `.hpp` headers
   include only Holoscan + `radar_common` (no MatX); all MatX code lives in
   `.cu` files and `*_stage.hpp` headers included only from `.cu`.
2. **Real-IF vs complex-IF**: auto-selected per tick from the received DLPack
   dtype (complex matching `T` → C2C; real matching `T` → R2C with
   `nfft/2 + 1` bins); wrong precision errors, naming the instantiation.
3. **Descriptors**: plain value structs in `operators/common/`
   (precision-agnostic, scalars `double`): `WindowSpec`,
   `FMCWWaveformDescriptor` (stores `bandwidth`/`chirp_slope` as optionals,
   derives the other), plus `enum class WindowType, FftNorm,
   ClutterFilterMethod`. Used directly as `Parameter<T>`; YAML converters
   provided for `from_config()`.
4. **[live] overrides**: same-named non-blocking input port
   (`.condition(ConditionType::kNone)`); `compute()` checks
   `op_input.receive<T>(name)` and falls back to the parameter. Wrapped in a
   small `LiveParam<T>` helper so all four operators do it identically.
5. **Warn-once-and-cache**: `ConfigCache<Key, Value>` in common — warns once
   per new key, only on override-triggered rebuilds (never the initial
   parameter build); caches window-coefficient tensors and dechirp references.
   FFT plan reuse comes free from MatX's internal plan cache; operators keep a
   seen-key set purely for the warn rule.
6. **Fused-op parity by construction**: each split operator's math is a pure
   stage function (`range_stage.hpp`, `clutter_stage.hpp`,
   `doppler_stage.hpp` — template over MatX views + config + state + stream)
   living in that operator's own directory; `compute()` is a thin shell. The
   fused operator includes the three stage headers and calls them
   back-to-back — identical instantiated code, so the `TESTING.md` parity
   test holds by construction. Stage signatures are frozen in Phase 0.
7. **Batching**: validate rank ≥ 2 and compact row-major, collapse leading
   dims to one batch dim, zero-copy 3-D MatX view, emit with original leading
   dims restored. Device/managed/pinned memory used in place; host (`kDLCPU`)
   staged H2D; outputs are always stream-ordered device memory. Strided
   (non-compact) inputs error in this wave.
8. **Streams**: `stream = op_input.receive_cuda_stream("<in port>")`; all MatX
   ops run on `matx::cudaExecutor(stream)`;
   `op_output.set_cuda_stream(stream, port)` before emit; the input tensor
   `shared_ptr` is retained until the next tick (async lifetime guard). No
   hot-path synchronizes.
9. **Port payload**: `holoscan::TensorMap` with a single entry (lenient on key
   at receive) — the SDK-idiomatic currency; recorded as a module convention.

## Kernel design (per contract)

- **Range** (`fmcw_range_processor_op`): optional dechirp
  (`in * conj(ref)`, reference cached from `waveform.reference_samples` or
  synthesized; requires waveform + complex input), window multiply (cached
  coefficients, coherent-gain normalized), FFT along the last axis with
  `nfft` zero-pad and `fft_norm` mapped to
  `matx::FFTNorm::{BACKWARD, ORTHO, FORWARD}`, no fftshift.
  `range_bins[i] = propagation_speed * (i * fs / nfft) / (2 * slope)`,
  emitted only when a waveform is present.
- **Clutter** (`clutter_filter_op`, stateless): `mean_subtraction` =
  `in - mean(in, slow_time)` (shape preserved); `single_delay` = first
  difference (`num_chirps - 1`); `double_delay` =
  `x[k] - 2x[k+1] + x[k+2]` (`num_chirps - 2`).
- **Doppler** (`fmcw_doppler_processor_op`): permute slow-time innermost,
  window, FFT (`nfft`, norm), fftshift when `center_zero_doppler` (default
  true; odd-N semantics verified against `numpy.fft.fftshift`), permute back.
  `velocity_bins`: `v = -f_d * lambda / 2` with
  `PRF = 1 / chirp_repetition_interval`, emitted only when a waveform is
  present.
- **Fused** (`fmcw_range_doppler_processor_op`): range stage → optional
  clutter stage (`clutter_filter` enum incl. `none`) → Doppler stage on async
  intermediates; union of parameters with `nfft_range` / `nfft_doppler`.
- Window generation (all types: Hann / Hamming / Blackman / Blackman-Harris /
  Kaiser / Chebyshev / Taylor / flat_top / Tukey / custom) is host-side in
  `common/window.cpp`, uploaded to a device tensor by the caller.

## Phase 0 — Foundation

New files (as landed):

- `operators/common/`: `CMakeLists.txt` (STATIC lib `radar_common` +
  `holoscan::radar_common` alias, links `holoscan::core`), with
  `radar_common/` organized by dependency layer:
  `descriptors/{enums, window_spec, fmcw_waveform_descriptor}.hpp` (STL
  only), `dsp/window.{hpp,cpp}` (host math), `tensor/{tensor_utils.{hpp,cpp},
  dlpack_buffer.hpp, matx_utils.hpp}` (tensor/DLPack/MatX plumbing), and
  `sdk/{override_param, config_cache, arg_setters}.hpp` (Holoscan glue).
  `matx_utils.hpp` is MatX-side (include from `.cu` only);
  `dlpack_buffer.hpp` wraps a cudaMalloc device buffer as a full-rank
  `holoscan::Tensor` (dynamic-rank outputs — MatX tensors are
  compile-time-rank, so kernels use collapsed 3-D views over the same
  memory). YAML converters initially deferred, later landed on this branch
  (`sdk/yaml_converters.hpp`).
- `operators/fmcw/CMakeLists.txt` (fused op declared FIRST so its DEPENDS
  force-set precedes its dependencies' option() binding) + four operator dirs
  `operators/fmcw/{fmcw_range_processor_op, clutter_filter_op,
  fmcw_doppler_processor_op, fmcw_range_doppler_processor_op}/`, each with:
  `<name>.hpp` (templated class declaration), `<name>.cu` (complete
  `setup()` — every locked port/param/[live] input — stub `compute()`
  erroring "not implemented", explicit instantiations for float/double),
  `<name>_stage.hpp` (FROZEN stage signatures with full semantic contracts,
  incl. the dechirp reference convention `r[s] = exp(+j*pi*slope*t_s^2)`
  shared with the simulator), `metadata.json`, `CMakeLists.txt` (modeled on
  `radar_op`'s, plus `matx::matx` + `holoscan::radar_common`, no `python/`
  subdir).
- `tests/cpp/utils/{point_target_sim.hpp, bin_model.hpp, tensor_fixtures.hpp,
  sdk_harness.hpp}`: analytic point-target simulator per `TESTING.md`
  (host-only, `std::complex<double>`, convention-free Cartesian scene,
  stop-and-hop, complex-IF / real-IF / pre-mixer outputs, seeded noise,
  ground truth out; no operator/MatX includes — independence guards against
  tautological tests); forward bin model; host↔device Tensor fixtures +
  `RADAR_TEST_SKIP_IF_NO_GPU()`; radar-flavored glue over the SDK-native
  operator test harness (`holoscan::test::OperatorTestHarness` +
  `TestHarnessSourceOp`/`TestHarnessSinkOp`): TensorMap capture-to-host
  validators and leading-tick [live]-override wiring. (An initial custom
  `OpHarness` was replaced by the SDK harness;
  `holoscan::ops::OperatorRunner` was evaluated and rejected — it requires
  a running application's executor.)
- `tests/cpp/test_common.cpp` (window generation incl. sidelobe checks,
  cache semantics, waveform descriptor, simulator cross-checks — fully
  implemented in Phase 0) + `tests/cpp/test_{fmcw_range_processor,
  clutter_filter, fmcw_doppler_processor, fmcw_range_doppler_processor}.cpp`
  as compiling `GTEST_SKIP()` stubs.
- `radar_op` placeholder KEPT until Phase 3 (still compiles on 4.3), along
  with its bindings, python package exports, and `test_operators.cpp`.

Modified shared files (frozen after this phase):

- `CMakeLists.txt`: `LANGUAGES CXX CUDA`, `CMAKE_CUDA_STANDARD 17`,
  `CMAKE_CUDA_ARCHITECTURES` default `native` floored at 70 (SDK example
  pattern).
- `operators/CMakeLists.txt`: `add_subdirectory(common)`,
  `add_subdirectory(fmcw)`; `operators/fmcw/CMakeLists.txt` carries the four
  `add_holohub_operator()` calls (fused one `DEPENDS OPERATORS` the other
  three).
- `tests/cpp/CMakeLists.txt`: utils interface lib + five test executables
  (`gtest_discover_tests`, LABELS unit).

Exit gate (in-container): configure + build green, `ctest -L unit` green
(stubs skip), `pre-commit run --all-files` green.

## Phase 1 — Parallel implementation (three agents)

Same working tree; strictly disjoint file ownership:

| Agent | Exclusively owns |
| --- | --- |
| Range | `operators/fmcw/fmcw_range_processor_op/**`, `tests/cpp/test_fmcw_range_processor.cpp` |
| Clutter | `operators/fmcw/clutter_filter_op/**`, `tests/cpp/test_clutter_filter.cpp` |
| Doppler | `operators/fmcw/fmcw_doppler_processor_op/**`, `tests/cpp/test_fmcw_doppler_processor.cpp` |

Hard rules: `operators/common/`, `tests/cpp/utils/`, all CMake outside the
owned dir, and planning docs are read-only for agents — defects in shared code
get reported back, never patched inline. Stage signatures are frozen; escalate
if a change seems needed.

Test assertions are physical-domain (map peaks through the operator's own
`range_bins` / `velocity_bins`), per `TESTING.md`: peak within one bin of
ground truth, both IF paths, dechirp round-trip (pre-mixer simulator data),
window sidelobe reduction with coherent gain preserved under each `fft_norm`,
`nfft` interpolates without adding resolution, leading-batch-axes
independence, live-override warn-once + cache reuse, float/double agreement,
memory-space handling.

## Phase 2 — Fused operator

Owns `operators/fmcw/fmcw_range_doppler_processor_op/**` +
`tests/cpp/test_fmcw_range_doppler_processor.cpp`. `compute()` = range stage →
optional clutter stage → Doppler stage. Tests: the `TESTING.md` parity test
(`max|rd_fused - rd_split| / max|rd_split|` < 1e-5 float / 1e-12 double, bins
exact) across every `clutter_filter` value and `nfft_range` / `nfft_doppler`
overrides, plus its own parameter handling.

## Phase 3 — Integration and radar_op removal

- Delete `operators/radar_op/`, `tests/cpp/test_operators.cpp`,
  `tests/python/test_radar_op.py` per the settled decision.
- Rewire `applications/fmcw_siso/cpp` to a sim-source → range → clutter →
  Doppler chain; the Python app variant has no operators to import (bindings
  deferred) — reduce to a placeholder or remove, noted in the PR.
- Update `python/holoscan/radar/__init__.py` (no operator exports until
  bindings land), `pyproject.toml` (drop `-DOP_radar_op=ON`), `pkg/` (package
  the four new C++ operator libs/headers, drop radar_op), root
  `metadata.json` (`operator_names`).
- Cross-cutting tests (float/double agreement matrix, determinism, end-to-end
  SISO chain), final `STATUS.md` update, full verification pass.

## Verification (every phase)

```bash
docker run --rm --gpus all --ipc=host \
  -v "$PWD":/workspace/holoscan-radar -w /workspace/holoscan-radar \
  holoscan-radar:ngc-v4.3.0-cuda13 bash -lc '
    cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_TESTING=ON &&
    cmake --build build -j"$(nproc)" &&
    ctest --test-dir build --output-on-failure -L unit'

pre-commit run --all-files   # host; exactly what CI runs
```

Every new file gets the two-line SPDX header; per-operator `metadata.json` is
schema-validated by pre-commit. Commits follow `Area: Summary` (e.g.
`Operators: Add FMCW range processor`).

## Open notes

- CI impact: enabling CUDA at configure time means CPU-host `cmake` configure
  no longer works; builds are container-only from here on (flag in the PR).
- Warn-once scope interpretation: only override-triggered rebuilds warn, never
  the initial parameter-driven build.
