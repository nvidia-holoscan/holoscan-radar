# `FMCWRangeProcessorOp`

Range compression: optional software dechirp, then a windowed FFT along fast-time that turns an ADC cube into range profiles. It is the first node of the split FMCW chain (SISO / SIMO / MIMO-TDM after de-interleave; MIMO-DDM on the raw cube). The input dtype selects the path — complex IF keeps a full-length FFT; real IF takes the R2C half-spectrum — and bin 0 is always zero range (no `fftshift`). `range_bins` is emitted only when a valid `waveform` is set.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `adc_cube` | complex64 / complex128, or float32 / float64 | `(num_chirps, samples_per_chirp)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Path auto-selected by dtype: complex IF or real IF |
| out | `range_profiles` | complex64 / complex128 | same leading + `(num_chirps, num_range_bins)` | Always emitted. `num_range_bins = nfft` (complex IF) or `nfft/2+1` (real IF); `nfft` defaults to `samples_per_chirp` |
| out (cond.) | `range_bins` | float32 / float64 | `(num_range_bins,)` | Emitted only when a valid `waveform` is available. Optional (`latched`): the first emit to an unwired port warns once, after which the axis is no longer built |
| live | `range_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |
| live | `apply_dechirp` | bool | scalar | Exact override; wire to enable, unwired = miss |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |
| live | `fft_norm` | `FftNorm` | scalar | Exact override; wire to enable, unwired = miss |
| param | `propagation_speed` | float64 | scalar | Params-only (no Exact port). Wave speed for `range_bins`; defaults to the speed of light |
| live | `nfft` | int64 | scalar | Exact override; wire to enable, unwired = miss |

```text
adc_cube → FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp
```

`range_window` and `waveform` remain logical `WindowSpec` / `FMCWWaveformDescriptor` Params. Their HSDK 5 physical ports use the L18 generated `holoscan::radar::wire` descriptor-Tensor encoding because the SDK does not admit container-owning C++ types directly. The Tensor member is always present: rank-1 zero-length represents an absent optional array, while null is invalid. See [`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md).

Contract: [`OPERATOR_API.md` § `FMCWRangeProcessorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

All math lives in `range_stage.hpp`. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype and named-axis shape (`collapse_named_tx_rx_batch`), merges this tick's Exact matches onto pinned Params (`resolve_params` → `RangeStageConfig`, sentinels mapped), and binds that config to this input (`build_range_stage_plan` → `RangeStagePlan` with `num_bins`). Required `range_profiles` is published first (`emit_prepared_tensor`); optional `range_bins` is a later gated block (`LatchedOutput::emit`). The fused `FMCWRangeDopplerProcessorOp` calls this same stage, which is why split-vs-fused range compression is identical by construction.

Per tick, the stage:

1. **Path selection.** `classify_input_dtype<T>(adc_cube, …, allow_real=true)`: matching complex dtype → C2C FFT of length `nfft`; matching real dtype → R2C FFT of length `nfft` producing `nfft/2+1` bins. Precision is the template `T`, never a runtime flag.
2. **Optional software dechirp** (`apply_dechirp = true`, complex IF only). Conjugate-multiply along fast-time against a per-chirp reference. When `waveform.reference_samples` is non-empty those samples are used (length must equal `samples_per_chirp`); otherwise the reference is synthesized as
   `r[s] = exp(+j·π·slope·t_s²)` with `t_s = s / sample_rate`, and the dechirped sample is `in[s] * conj(r[s])` (`range_stage.hpp`). This is the same sign convention as `radar_common/sim/`.
3. **Window.** If `range_window.type != none`, multiply along fast-time by the coherent-gain-normalized taps from `generate_window`. On the complex path the window and `conj(r)` are folded into one cached per-sample premultiplier; the real path caches plain window taps (dechirp is illegal there).
4. **Range FFT.** `matx::fft(pre, nfft, norm)` along the last axis. `nfft` defaults to `samples_per_chirp`; a larger value zero-pads. `fft_norm` maps to MatX (`backward` = unnormalized forward — radar-standard, coherent-integration gain stays visible; `ortho`; `forward`). **No `fftshift`**: bin 0 is zero beat / zero range.
5. **`range_bins` (conditional).** When a valid `waveform` is present, fill
   `range_bins[i] = propagation_speed · (i · fs / nfft) / (2 · slope)`
   (`fill_range_bins`, MatX `range` on the compute stream). Units are metres relative to zero beat. Default `propagation_speed` is `kDefaultPropagationSpeed` (`kSpeedOfLight` = 299792458.0).

Invariants: leading named axes are processed independently (collapsed to a batch of size `N_rx` or `M_tx·N_rx`); the stage does not reorder chirps or range bins.

## Shape and dtype rules

- **`range_profiles` rank** = input rank. Trailing axes are `(num_chirps, num_range_bins)` with `num_chirps` copied from the input and `num_range_bins = range_num_bins(complex_input, samples_per_chirp, nfft)`: `nfft` when complex, `nfft/2+1` when real (`nfft` defaults to `samples_per_chirp`).
- **`range_bins`** is rank-1, length `num_range_bins`.
- **Named leading axes.** None, `N_rx`, or `(M_tx, N_rx)`, admitted only up to `named_tx_rx_leading_rank(extents)` (0 / 1 / 2 from whether `max_tx` and `max_rx`/`max_channels` were set). Extra leading axes fail with `FMCWRangeProcessorOp: unexpected leading axes (opaque batch is not supported); got leading rank <got>, max named leading <max>`. A leading product above `max_tx * max_rx` fails with `FMCWRangeProcessorOp: leading product <n> exceeds max_tx*max_rx from MaxTensorExtents`.
- **Real IF vs complex IF.** Selected only by `adc_cube` dtype vs the instantiation. A mismatched dtype fails: `adc_cube: received dtype <got> does not match this operator's float|double-precision instantiation (expected <complex> or <real>)`.
- **`nfft`.** When non-zero it must be `≥ samples_per_chirp`; otherwise `FMCWRangeProcessorOp: nfft (<n>) must be >= samples_per_chirp (<s>) on port adc_cube`. Downstream re-infers `num_range_bins` after a `[live]` `nfft` change.
- **`apply_dechirp = true`** without a valid waveform: `FMCWRangeProcessorOp: apply_dechirp = true requires a waveform descriptor (dechirp reference source)`. With real-IF input: `FMCWRangeProcessorOp: apply_dechirp = true requires complex-IF input on port adc_cube (received real-IF samples; software dechirp produces complex samples, inconsistent with the real-IF output size)`. Stage-level: `waveform.reference_samples` length must match `samples_per_chirp`.
- **Layout.** `adc_cube` must be CUDA-accessible and compact row-major: `FMCWRangeProcessorOp: tensor received on input port 'adc_cube' is not CUDA-accessible (device, managed, or pinned-host memory required)` / `… must be contiguous (compact row-major)`.
- **No CUDA stream.** `FMCWRangeProcessorOp: no CUDA compute stream`.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `range_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Rebuilds cached premultiplier / window taps on an unseen spec, warning once | Fast-time window before the range FFT; coherent-gain normalized when `coherent_gain_normalization` is true (WindowSpec default) |
| `apply_dechirp` | bool | `false` | `[live]` Exact | Rebuilds the cached premultiplier when combined with the waveform / window | Software dechirp; requires a valid `waveform` and complex-IF input |
| `waveform` | `FMCWWaveformDescriptor` | none (`is_valid() == false`) | `[live]` Exact via wire schema | Rebuilds dechirp reference / premultiplier when `apply_dechirp` is true; required for `range_bins` | Dechirp reference (`reference_samples`, or `slope` + `sample_rate`) and the `range_bins` axis |
| `fft_norm` | `FftNorm` {`backward`, `ortho`, `forward`} | `backward` | `[live]` Exact | Free at the operator (MatX applies the convention on the FFT) | FFT normalization; `backward` keeps coherent-integration gain visible |
| `propagation_speed` | float64 | `299792458.0` m/s | Params-only | Free (device fill of `range_bins` only) | Wave speed for beat frequency → range |
| `nfft` | int64 | `0` (= `samples_per_chirp`) | `[live]` Exact | Unseen length builds a MatX FFT plan; the stage keys `seen_fft_keys` and warns once per new length | Fast-time FFT length; larger zero-pads for a finer interpolated grid over the same span |

`waveform` is required when `apply_dechirp` is true and whenever a consumer needs `range_bins`. A default-constructed descriptor is treated as absent. YAML / `set_params` field names match the table (`range_window: hann` or a `WindowSpec` map; `fft_norm: backward`).

`propagation_speed` has no `[live]` port and is not declared in `setup()` / `contract()`. `resolve_params` copies `pinned.propagation_speed` into the stage config. It is used only by `fill_range_bins`.

## SDK implementation notes

- **Construction.** `FMCWRangeProcessorOp(const MaxTensorExtents&)` requires `max_chirps` and `max_range_bins`. Optional `max_rx` / `max_channels` admit a leading `N_rx`; `max_tx` plus an RX/channel field admits `(M_tx, N_rx)`. Those fields size `range_profiles` (`max_rank` 2–4, `max_byte_span` of a complex cube `{max_tx, max_rx, max_chirps, max_range_bins}`) and `range_bins` (`max_rank` 1, real `{max_range_bins}`). A tick that exceeds the admitted leading rank or `max_tx * max_rx` is rejected (messages above). `max_rx`/`max_tx` must be positive when set.
- **Triggering.** `contract()` is `OnEach{adc_cube}` plus an `Exact{}` + `SafeDefault{}` entry for every override, wired or not. An unwired override compiles to a `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT` warning and reads as `kNotConnected` on every tick, so wiring the port is the application's only obligation. This requires an SDK carrying CONTRACT-020 ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Wiring a port `contract()` does not declare is still `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1). Optional `range_bins` stays omitted so an unwired sink does not gate scheduling.
- **Ports and queues.** `adc_cube` and the override inputs use `queue_depth(2)`. Data Tensor ports are `MemoryKind::kCudaDevice`. Fat / bins outs carry ctor-derived `TensorBounds` via `pooled_tensor_output`. `setup()` registers data-plane I/O first (`adc_cube`, then `range_profiles`, then latched `range_bins`), then every `[live]` Exact input in Params field order. Schema-backed lives (`range_window`, `waveform`) sit with the POD lives, not next to the cube; they are `Input<wire::WindowSpecT>` / `Input<wire::FMCWWaveformT>` and do not take `expects_tensor`.
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. After emit, `adc_cube` is moved into `retained_input_` so the FFT can finish reading it after `compute()` returns. That is a different buffer from the published loans: emit never publishes the input (HSDK 5 rejects a foreign Tensor on a pooled output). Neither the range FFT path nor `fill_range_bins` synchronizes: bins are written into the loaned device buffer by a MatX `range` generator on the compute stream, with no host staging. The only remaining stream sync is cold-path — uploading regenerated dechirp / window coefficients on a cache miss ([`OPEN_ITEMS.md`](OPEN_ITEMS.md) item 1).
- **Emission.** `range_profiles` is prepared and published first (`prepare_device_tensor` → `emit_prepared_tensor`). Holding that required emit until the optional bins loan is ready would drop profiles if bins allocate failed. When `range_bins` is produced, that work runs afterward in one gated `if (plan.stage.waveform.has_value() && range_bins_out_.active())` (allocate, fill, `LatchedOutput::emit`). An unwired bins emit is absorbed and latches the port off. The required emit is not duplicated across branches.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.FMCWRangeProcessorParams`, version 1. Apps on this branch load YAML into `FMCWRangeProcessorParams` and call `set_params` ([EA HSDK5 ONLY] Graph has no `from_config()`).
- **Precision.** Explicit instantiations `FMCWRangeProcessorOp<float>` and `FMCWRangeProcessorOp<double>` in `fmcw_range_processor_op.cu`.
- **Cache warn-once.** The stage's `ConfigCache`s decide on their own when to warn: the first configuration builds silently, a later unseen one (new `nfft`, new window / dechirp key, new chirp length) builds and warns once, then is cached. `compute()` passes no warn flag.
- **[EA HSDK5 ONLY]** Exact `Input<>` ports are declared unconditionally (see Triggering). `range_window` / `waveform` use generated schema wire types and convert back to the logical descriptor before override resolution; scalar / enum ports use their logical type directly. Graph tests cover `nfft`, window, waveform, and the schema wire behavior.

## Consumer lifetime notes

Every output is a fresh Tensor-pool buffer for that tick. A consumer may retain `range_profiles` or `range_bins` without copying. `retained_input_` is the input-side counterpart and is not visible to consumers.

## Performance

**Cold.** First tick (and first unseen window / dechirp-reference / FFT-length key) builds `RangeStageState`: `premultipliers` (complex path), `windows` (real path), and `seen_fft_keys`. MatX owns the cuFFT plan cache. `range_bins` is filled directly on the compute stream by a MatX range generator.

**Steady.** Cached coefficients are reused; each tick loans `range_profiles` (and `range_bins` when a waveform is set) and enqueues the FFT. `[live]` `nfft` to an unseen length builds a new MatX plan. `[live]` `apply_dechirp` / `range_window` / `waveform` rebuild the premultiplier or window taps on an unseen key. `fft_norm` and the parameter-only `propagation_speed` do not invalidate those caches.

Harness: `performance/cpp/perf_fmcw_range_processor.cpp` (Python mirror exists but is not built on this branch). Axes: size ladder `{xs, sm, md, lg, xl}` = `(16×128) … (256×2048)` chirps × samples; window `{rect, hann}`; optional channel batch (`md` × 4). Default matrix: rect `{xs, md, lg, xl}` plus hann/`md` and rect/`md`/4 channels. `--full` crosses windows × sizes and adds channel batches on `md`/`lg`. Notes field records `live_override=deferred_ea` because mid-session override timing remains deferred under §1 / §8, not because the port is absent. No measured numbers are recorded in-tree; run the harness for timings.

GPU-native construction of synthesized dechirp references, window taps, and
premultipliers is a performance follow-up; see [`OPEN_ITEMS.md`](OPEN_ITEMS.md)
§1.

```bash
ctest --test-dir build -R perf_fmcw_range_processor_smoke
```

## Tests and verification

C++ Graph suite: `tests/cpp/fmcw/test_fmcw_range_processor.cpp`. Ground truth is `radar_common/sim/` (`sim::simulate` / `to_real_if`). Python `tests/python/fmcw/test_fmcw_range_processor.py` is not registered ([EA HSDK5 ONLY] `HOLOSCAN_RADAR_BUILD_PYTHON` forced off; [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6).

| Check | Proves |
| --- | --- |
| `SingleTargetComplexIfPeakAndRangeAxis` | Peak maps via `range_bins` to the injected range within one bin; axis matches `propagation_speed·(i·fs/nfft)/(2·slope)` (tolerance `1e-3 + 1e-5·abs(expected)`) |
| `RealIfPathPeakAtRange` | Real IF → `samples_per_chirp/2+1` bins; peak still at the target range |
| `NoRangeBinsWithoutWaveform` | `range_bins` not emitted when `waveform` is absent; profiles-only graph still runs |
| `UndrainedRangeBinsDoesNotFailTheRun` | Waveform present but `range_bins` unwired: 128 ticks still deliver `range_profiles`; the first failed emit latches the port off |
| `ApplyDechirpMatchesHardwareDechirp` | Software dechirp of pre-mixer data matches hardware-dechirped sim (≤ `2e-3` of peak magnitude) |
| `ApplyDechirpWithoutWaveformErrors` / `ApplyDechirpWithRealInputErrors` | The two contract-illegal dechirp combinations raise diagnostics |
| `HannWindowLowersSidelobesKeepsPeak` | Hann PSL < 50% of rectangular (bins `peak+4…peak+16`); on-bin peak preserved within 3% under `fft_norm = backward` |
| `WindowConfigurableFromParams` / `WaveformConfigurableFromParams` | Params authoring of `WindowSpec` / descriptor |
| `NfftOversamplingInterpolatesGrid` | `nfft = 2·samples` doubles bin count; bin spacing halves; peak stays on the same range |
| `NamedRxAxisRoundTrip` | Leading `N_rx = 2` preserved; per-RX magnitudes agree to `1e-3` |
| `LiveNfftExactMatchChangesBinCount` | Wired `nfft` override + matching `capture_time` changes `num_range_bins` |
| `LiveNfftRevertsToParamsAfterMiss` | Silent miss after a match reverts to Params `nfft` (= `samples_per_chirp`) |
| `LiveNfftSkewedCaptureTimeMissesExact` | 1 ms-skewed override is not selected; cube uses Params |
| `LiveWindowOverrideAppliesPerTick` | Schema-backed Hann + narrower-waveform Exact tick lowers sidelobes and changes range-axis spacing; the following miss tick reverts both to Params |
| `InvalidLiveWaveformRejectedBeforeDsp` | Malformed live waveform is rejected at the wire bridge; no DSP output that tick |
| `FloatAndDoubleInstantiationsAgree` | F32 vs F64 peak index match; magnitudes within `1e-4` of the F64 peak |

`TESTING.md` also asks for a `tx×rx` leading-axis case; only `N_rx` is covered.

```bash
ctest --test-dir build -R test_fmcw_range_processor --output-on-failure
```

## Usage

```yaml
range_processor:
  waveform: *waveform
  range_window: hann
  # apply_dechirp: false
  # fft_norm: backward
  # nfft: 512
  # propagation_speed: 299792458.0
```

```cpp
auto range = graph.op<FMCWRangeProcessorOp<float>>(
    "range_processor", extents);
range.set_params(cfg.range_params);

graph.add_flow(source->adc_cube, range->adc_cube, q);
graph.add_flow(range->range_profiles, clutter->range_profiles, q);
graph.add_flow(range->range_bins, sink->range_bins, q);  // only if waveform is set
```

Python bindings (`FMCWRangeProcessorOpF32` / `F64`) are deferred on this branch ([EA HSDK5 ONLY] [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6); the 4.x Fragment trampoline in `python/` is not built.

## Contract conformance

The locked [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) surface is implemented. `range_window` and `waveform` `[live]` ports use the L18 generated `holoscan::radar::wire` descriptor-Tensor encoding ([`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md)); that is the contract physical form, not a gap.

Audit scope (static; no build or test run): `docs/planning/OPERATOR_API.md` § `FMCWRangeProcessorOp`, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md` §1/§6/§8/§9, `STATUS.md`, `operators/common/README.md`, `operators/fmcw/README.md`, `fmcw_range_processor_op.hpp` / `.cu`, `range_stage.hpp`, `metadata.json`, `python/_fmcw_range_processor_op_bindings.cpp`, `tests/cpp/fmcw/test_fmcw_range_processor.cpp`, `tests/python/fmcw/test_fmcw_range_processor.py`, `performance/cpp/perf_fmcw_range_processor.cpp`, and `OPEN_ITEMS.md`. The GPU coefficient-construction question is tracked in `OPEN_ITEMS.md` §1.
