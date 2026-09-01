# `FMCWDopplerProcessorOp`

Doppler compression: a windowed FFT along slow-time that turns range profiles into range-Doppler maps. It sits after `FMCWRangeProcessorOp` (and optional `ClutterFilterOp`) on every split FMCW chain. Zero-Doppler is centered by default (`fftshift`, numpy semantics, including odd lengths); radial velocity uses `v = −f_d·λ/2` (`v_r > 0` = receding). `velocity_bins` is emitted only when a valid `waveform` is set.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `range_profiles` | complex64 / complex128 | `(num_chirps, num_range_bins)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Slow-time FFT along `num_chirps`. Real IF is rejected |
| out | `rd_maps` | complex64 / complex128 | same leading + `(num_doppler, num_range_bins)` | Always emitted. `num_doppler = nfft` (default `num_chirps`). Zero-Doppler centered when `center_zero_doppler` |
| out (cond.) | `velocity_bins` | float32 / float64 | `(num_doppler,)` | Emitted only when a valid `waveform` is available. Optional (`latched`): the first emit to an unwired port warns once, after which the axis is no longer built |
| live | `doppler_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |
| live | `center_zero_doppler` | bool | scalar | Exact override; wire to enable, unwired = miss |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |
| live | `fft_norm` | `FftNorm` | scalar | Exact override; wire to enable, unwired = miss |
| live | `nfft` | int64 | scalar | Exact override; wire to enable, unwired = miss |

```text
FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp → rd_maps
```

`doppler_window` and `waveform` remain logical `WindowSpec` / `FMCWWaveformDescriptor` Params. Their HSDK 5 physical ports use the L18 generated `holoscan::radar::wire` descriptor-Tensor encoding because the SDK does not admit container-owning C++ types directly. The Tensor member is always present: rank-1 zero-length represents an absent optional array, while null is invalid. See [`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md).

Contract: [`OPERATOR_API.md` § `FMCWDopplerProcessorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

All math lives in `doppler_stage.hpp`. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype and named-axis shape (`collapse_named_tx_rx_batch`), merges this tick's Exact matches onto pinned Params (`resolve_params` → `DopplerStageConfig`, sentinels mapped), and binds that config to this input (`build_doppler_stage_plan` → `DopplerStagePlan` with `num_doppler`). Required `rd_maps` is published first (`emit_prepared_tensor`); optional `velocity_bins` is a later gated block (`LatchedOutput::emit`). The fused `FMCWRangeDopplerProcessorOp` feeds this same entry point (optionally with a clutter expression as `in_op`), which is why split-vs-fused Doppler is identical by construction.

Per tick, the stage:

1. **Window.** If `doppler_window.type != none`, multiply along slow-time by the coherent-gain-normalized taps from `generate_window` (cached). The cube is permuted so chirps are innermost: `permute(in, {0, 2, 1}) * clone(window, {batch, num_range_bins, keep})`.
2. **Doppler FFT.** `matx::fft(pre, num_doppler, norm)` along that last axis. `num_doppler = nfft` when set, else `num_chirps`. A larger `nfft` zero-pads. `fft_norm` maps to MatX (`backward` / `ortho` / `forward`); `backward` is the radar-standard unnormalized forward.
3. **Center zero-Doppler (default).** When `center_zero_doppler` is true, `matx::fftshift1D` then permute back `{0, 2, 1}`. The shift is numpy `fftshift`: `out[i] = in[(i + ceil(n/2)) % n]` (MatX: `fftshift1D` maps output index `i` to input `(i + (n + 1) / 2) % n`). When false, DC stays at bin 0; only the permute runs.
4. **`velocity_bins` (conditional).** When a valid `waveform` is present, fill `v = −f_d·λ/2` with `PRF = 1 / chirp_repetition_interval` and `λ = waveform.wavelength()` (`fill_velocity_bins`). Per-bin Doppler frequency comes from MatX `fftfreq` (unshifted signed DFT order, Nyquist folded), then `fftshift1D` when `center_zero_doppler` so the axis matches `rd_maps`. `v_r > 0` is receding.

Invariants: leading named axes are processed independently (collapsed to a batch of size `N_rx` or `M_tx·N_rx`); `num_chirps` is the post-clutter length when a delay canceller ran upstream.

## Shape and dtype rules

- **`rd_maps` rank** = input rank. Trailing axes are `(num_doppler, num_range_bins)` with `num_range_bins` copied and `num_doppler = doppler_num_bins(num_chirps, nfft)` (`nfft` or `num_chirps`).
- **`velocity_bins`** is rank-1, length `num_doppler`.
- **Named leading axes.** None, `N_rx`, or `(M_tx, N_rx)`, admitted only up to `named_tx_rx_leading_rank(extents)` (0 / 1 / 2 from whether `max_tx` and `max_rx`/`max_channels` were set). Extra leading axes fail with `FMCWDopplerProcessorOp: unexpected leading axes (opaque batch is not supported); got leading rank <got>, max named leading <max>`. A leading product above `max_tx * max_rx` fails with `FMCWDopplerProcessorOp: leading product <n> exceeds max_tx*max_rx from MaxTensorExtents`.
- **Complex only.** `classify_input_dtype<T>(…, allow_real=false)`. Real input: `range_profiles: real-valued input <dtype> is not supported here; provide <complex>`.
- **`nfft`.** When non-zero it must be `≥ num_chirps`; otherwise `FMCWDopplerProcessorOp: nfft (<n>) must be >= num_chirps (<c>) on input port 'range_profiles'`. Downstream re-infers `num_doppler` after a `[live]` `nfft` change (or after a delay-canceller chirp loss upstream).
- **Layout.** `range_profiles` must be CUDA-accessible and compact row-major: `FMCWDopplerProcessorOp: tensor received on input port 'range_profiles' is not CUDA-accessible (device, managed, or pinned-host memory required)` / `… must be contiguous (compact row-major)`.
- **No CUDA stream.** `FMCWDopplerProcessorOp: no CUDA compute stream`.
- **`center_zero_doppler`.** Mid-stream flip changes bin order, not extents; `velocity_bins` (when emitted) is rebuilt to match.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `doppler_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Rebuilds cached window taps on an unseen spec, warning once | Slow-time window before the Doppler FFT; coherent-gain normalized when `coherent_gain_normalization` is true (WindowSpec default) |
| `center_zero_doppler` | bool | `true` | `[live]` Exact | Free (permute / `fftshift1D` choice) | Center DC on the Doppler axis; else bin 0 is DC |
| `waveform` | `FMCWWaveformDescriptor` | none (`is_valid() == false`) | `[live]` Exact via wire schema | Free for the FFT; required for `velocity_bins` | Carrier → `λ`, PRI → PRF for the velocity axis |
| `fft_norm` | `FftNorm` {`backward`, `ortho`, `forward`} | `backward` | `[live]` Exact | Free at the operator (MatX applies the convention on the FFT) | FFT normalization; `backward` keeps coherent-integration gain visible |
| `nfft` | int64 | `0` (= `num_chirps`) | `[live]` Exact | Unseen length builds a MatX FFT plan; the stage keys `seen_fft_keys` and warns once per new length | Doppler FFT length; larger interpolates the velocity grid over the same span |

`waveform` is required only when a consumer needs `velocity_bins`. A default-constructed descriptor is treated as absent. YAML / `set_params` field names match the table (`doppler_window: hann` or a `WindowSpec` map; `fft_norm: backward`).

There is no Params-only field (unlike range `propagation_speed`).

## SDK implementation notes

- **Construction.** `FMCWDopplerProcessorOp(const MaxTensorExtents&)` requires `max_chirps` and `max_range_bins`. Optional `max_rx` / `max_channels` admit a leading `N_rx`; `max_tx` plus an RX/channel field admits `(M_tx, N_rx)`. Those fields size `rd_maps` (`max_rank` 2–4, `max_byte_span` of a complex cube `{max_tx, max_rx, max_chirps, max_range_bins}`) and `velocity_bins` (`max_rank` 1, real `{max_chirps}`). The Doppler extent is **`max_chirps`**, not `max_doppler_bins` (that field is unused). A `[live]` `nfft` larger than `max_chirps` therefore exceeds the pool span; callers must size `max_chirps` to the largest Doppler FFT they will run. A tick that exceeds the admitted leading rank or `max_tx * max_rx` is rejected (messages above). `max_rx`/`max_tx` must be positive when set.
- **Triggering.** `contract()` is `OnEach{range_profiles}` plus an `Exact{}` + `SafeDefault{}` entry for every override, wired or not. An unwired override compiles to a `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT` warning and reads as `kNotConnected` on every tick, so wiring the port is the application's only obligation. This requires an SDK carrying CONTRACT-020 ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Wiring a port `contract()` does not declare is still `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1). Optional `velocity_bins` stays omitted so an unwired sink does not gate scheduling.
- **Ports and queues.** `range_profiles` and the override inputs use `queue_depth(2)`. Data Tensor ports are `MemoryKind::kCudaDevice`. Fat / bins outs carry ctor-derived `TensorBounds` via `pooled_tensor_output`. `setup()` registers data-plane I/O first (`range_profiles`, then `rd_maps`, then latched `velocity_bins`), then every `[live]` Exact input in Params field order. Schema-backed lives (`doppler_window`, `waveform`) sit with the POD lives, not next to the cube; they are `Input<wire::WindowSpecT>` / `Input<wire::FMCWWaveformT>` and do not take `expects_tensor`.
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. After emit, `range_profiles` is moved into `retained_input_` so the FFT can finish reading it after `compute()` returns. That is a different buffer from the published loans: emit never publishes the input (HSDK 5 rejects a foreign Tensor on a pooled output). Neither the Doppler FFT path nor `fill_velocity_bins` synchronizes: bins are written into the loaned device buffer by MatX `fftfreq` / `fftshift1D` on the compute stream, with no host staging. The only remaining stream sync is cold-path — uploading regenerated window coefficients on a cache miss ([`OPEN_ITEMS.md`](OPEN_ITEMS.md) item 1).
- **Emission.** `rd_maps` is prepared and published first (`prepare_device_tensor` → `emit_prepared_tensor`). Holding that required emit until the optional bins loan is ready would drop maps if bins allocate failed. When `velocity_bins` is produced, that work runs afterward in one gated `if (plan.stage.waveform.has_value() && velocity_bins_out_.active())` (allocate, fill, `LatchedOutput::emit`). An unwired bins emit is absorbed and latches the port off. The required emit is not duplicated across branches.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.FMCWDopplerProcessorParams`, version 1. Apps on this branch load YAML into `FMCWDopplerProcessorParams` and call `set_params` ([EA HSDK5 ONLY] Graph has no `from_config()`).
- **Precision.** Explicit instantiations `FMCWDopplerProcessorOp<float>` and `FMCWDopplerProcessorOp<double>` in `fmcw_doppler_processor_op.cu`.
- **Cache warn-once.** The stage's `ConfigCache`s decide on their own when to warn: the first configuration builds silently, a later unseen one (new `nfft`, new window key, new chirp count) builds and warns once, then is cached. `compute()` passes no warn flag.
- **[EA HSDK5 ONLY]** Exact `Input<>` ports are declared unconditionally (see Triggering). `doppler_window` / `waveform` use generated schema wire types and convert back to the logical descriptor before override resolution; scalar / enum ports use their logical type directly. Graph tests cover the window, waveform, and `center_zero_doppler` schema / Exact behavior.

## Consumer lifetime notes

Every output is a fresh Tensor-pool buffer for that tick. A consumer may retain `rd_maps` or `velocity_bins` without copying. `retained_input_` is the input-side counterpart and is not visible to consumers.

## Performance

**Cold.** First tick (and first unseen window / FFT-length key) builds `DopplerStageState`: `windows` and `seen_fft_keys`. MatX owns the cuFFT plan cache. `velocity_bins` is filled directly on the compute stream by MatX `fftfreq` / `fftshift1D`.

**Steady.** Cached window taps are reused; each tick loans `rd_maps` (and `velocity_bins` when a waveform is set) and enqueues the permute + FFT + optional shift. `[live]` `nfft` to an unseen length builds a new MatX plan. `[live]` `doppler_window` rebuilds taps on an unseen key. `center_zero_doppler` and `fft_norm` do not invalidate those caches.

Harness: `performance/cpp/perf_fmcw_doppler_processor.cpp` (Python mirror exists but is not built on this branch). Axes: size ladder `{xs, sm, md, lg, xl}` = `(16×128) … (256×2048)` chirps × range bins; window `{rect, hann}`; optional channel batch (`md` × 4). Default matrix: rect `{xs, md, lg, xl}` plus hann/`md` and rect/`md`/4 channels. `--full` crosses windows × sizes and adds channel batches on `md`/`lg`. Notes field records `live_override=deferred_ea` because mid-session override timing remains deferred under §1 / §8, not because the port is absent. No measured numbers are recorded in-tree; run the harness for timings.

GPU-native construction of window taps is a performance follow-up; see [`OPEN_ITEMS.md`](OPEN_ITEMS.md) §1.

```bash
ctest --test-dir build -R perf_fmcw_doppler_processor_smoke
```

## Tests and verification

C++ Graph suite: `tests/cpp/fmcw/test_fmcw_doppler_processor.cpp`. Ground truth is analytic Doppler tones plus `radar_common/sim/` (`SimulatorTargetMapsToTruthVelocity`) and `tests/cpp/utils/bin_model.hpp` (`expected_velocity_axis`). Python `tests/python/fmcw/test_fmcw_doppler_processor.py` is not registered ([EA HSDK5 ONLY] `HOLOSCAN_RADAR_BUILD_PYTHON` forced off; [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6).

| Check | Proves |
| --- | --- |
| `VelocityPeakMatchesTruthCentered` | Peak maps via `velocity_bins` to injected `v_r` within one bin; axis matches `v = −f_d·λ/2` with centering (`EXPECT_FLOAT_EQ` vs `expected_velocity_axis`) |
| `VelocityAxisUncentered` / `CenterZeroDopplerPlacesDc` | DC at bin 0 when `center_zero_doppler` is false; DC centered when true |
| `OddChirpCountFftshiftMatchesNumpy` | Odd-`num_chirps` shift matches numpy `fftshift` |
| `HannWindowPreservesOnBinPeak` / `HannWindowLowersDopplerSidelobes` | Windowing as in the range operator (peak kept; PSL down) |
| `NfftInterpolatesVelocityGrid` | Larger `nfft` interpolates; peak stays on the same velocity |
| `NamedRxAxisProcessedIndependently` | Leading `N_rx` preserved; tones do not leak across RX |
| `VelocityBinsOnlyEmittedWithWaveform` | Conditional emit: no waveform → profiles-only graph still runs; waveform set → both outs |
| `RejectsRealInput` / `NfftSmallerThanChirpCountRejected` | Contract-illegal inputs fail closed |
| `FloatAndDoubleAgree` | Magnitudes within `1e-3 · num_chirps` |
| `SimulatorTargetMapsToTruthVelocity` | Sim target at known `v_r` peaks on the matching bin |
| `LiveOverrideDopplerWindow` | Schema-backed Hann + narrower-waveform Exact tick lowers sidelobes and halves velocity-axis spacing; the following miss tick reverts both to Params |
| `LiveOverrideCenterZeroDopplerMidStream` | Exact `true` centers a DC tone (and the 0 m/s `velocity_bins` slot); the following miss reverts to Params `false` |

`TESTING.md` also asks for a `tx×rx` leading-axis case; only `N_rx` is covered. There is no dedicated unwired-`velocity_bins`-with-waveform latch test (range has `UndrainedRangeBinsDoesNotFailTheRun`).

```bash
ctest --test-dir build -R test_fmcw_doppler_processor --output-on-failure
```

## Usage

```yaml
doppler_processor:
  waveform: *waveform
  doppler_window: hann
  # center_zero_doppler: true
  # fft_norm: backward
  # nfft: 512
```

```cpp
auto doppler = graph.op<FMCWDopplerProcessorOp<float>>(
    "doppler_processor", extents);
doppler.set_params(cfg.doppler_params);

graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, q);
graph.add_flow(doppler->rd_maps, cfar->rd_maps, q);
graph.add_flow(doppler->velocity_bins, sink->velocity_bins, q);  // only if waveform is set
```

Python bindings (`FMCWDopplerProcessorOpF32` / `F64`) are deferred on this branch ([EA HSDK5 ONLY] [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6); the 4.x Fragment trampoline in `python/` is not built.

## Contract conformance

The locked [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) surface is implemented. `doppler_window` and `waveform` `[live]` ports use the L18 generated `holoscan::radar::wire` descriptor-Tensor encoding ([`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md)); that is the contract physical form, not a gap.

Audit scope (static; no build or test run): `docs/planning/OPERATOR_API.md` § `FMCWDopplerProcessorOp`, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md` §1/§6/§8/§9, `STATUS.md`, `operators/common/README.md`, `operators/fmcw/README.md`, `fmcw_doppler_processor_op.hpp` / `.cu`, `doppler_stage.hpp`, `metadata.json`, `python/_fmcw_doppler_processor_op_bindings.cpp`, `tests/cpp/fmcw/test_fmcw_doppler_processor.cpp`, `tests/python/fmcw/test_fmcw_doppler_processor.py`, `performance/cpp/perf_fmcw_doppler_processor.cpp`, and `OPEN_ITEMS.md`. The GPU coefficient-construction question is tracked in `OPEN_ITEMS.md` §1.
