# `FMCWRangeProcessorOp`

## Purpose

Range compression: optional software dechirp, then a windowed FFT along fast-time that turns an ADC cube into range profiles. It is the first node of the split FMCW chain (SISO / SIMO / MIMO-TDM after de-interleave; MIMO-DDM on the raw cube). The input dtype selects the path — complex IF keeps a full-length FFT; real IF takes the R2C half-spectrum — and bin 0 is always zero range (no `fftshift`). `range_bins` is emitted only when a valid `waveform` is set.

## Responsibilities

- Optionally dechirps ADC samples, applies the fast-time window and range FFT, and normalizes the spectrum.
- Owns coefficient/FFT caching, tick-local range settings, real-versus-complex output sizing, and optional physical range-axis generation.
- Does not suppress slow-time clutter, compute Doppler, or infer a waveform when none is supplied.

## When to use

Use as the first spectral stage for real or complex FMCW ADC data when range profiles are needed by later clutter, Doppler, or diagnostic consumers. Supply a waveform for software dechirp or a physical range axis. Choose the fused range-Doppler operator when no graph consumer needs range profiles; do not use this operator on already range-compressed data.

```text
adc_cube → FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `adc_cube` | complex64 / complex128, or float32 / float64 | `(num_chirps, samples_per_chirp)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Required trigger/aligned input | Path auto-selected by dtype: complex IF or real IF |
| out | `range_profiles` | complex64 / complex128 | same leading + `(num_chirps, num_range_bins)` | Required output | Always emitted. `num_range_bins = nfft` (complex IF) or `nfft/2+1` (real IF); `nfft` defaults to `samples_per_chirp` |
| out (cond.) | `range_bins` | float32 / float64 | `(num_range_bins,)` | Optional latched output | Emitted only when a valid `waveform` is available. Optional (`latched`): the first emit to an unwired port warns once, after which the axis is no longer built |
| live | `range_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `apply_dechirp` | bool | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `fft_norm` | `FftNorm` | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `nfft` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_chirps` and `max_range_bins` required; channel/TX bounds optional | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `range_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Fast-time window before the range FFT; coherent-gain normalized when `coherent_gain_normalization` is true (WindowSpec default) Cost: Rebuilds cached premultiplier / window taps on an unseen spec, warning once. |
| `apply_dechirp` | bool | `false` | `[live]` Exact | Software dechirp; requires a valid `waveform` and complex-IF input Cost: Rebuilds the cached premultiplier when combined with the waveform / window. |
| `waveform` | `FMCWWaveformDescriptor` | none (`is_valid() == false`) | `[live]` Exact via wire schema | Dechirp reference (`reference_samples`, or `slope` + `sample_rate`) and the `range_bins` axis Cost: Rebuilds dechirp reference / premultiplier when `apply_dechirp` is true; required for `range_bins`. |
| `fft_norm` | `FftNorm` {`backward`, `ortho`, `forward`} | `backward` | `[live]` Exact | FFT normalization; `backward` keeps coherent-integration gain visible Cost: Free at the operator (MatX applies the convention on the FFT). |
| `propagation_speed` | float64 | `299792458.0` m/s | Params-only | Wave speed for beat frequency → range Cost: Free (device fill of `range_bins` only). |
| `nfft` | int64 | `0` (= `samples_per_chirp`) | `[live]` Exact | Fast-time FFT length; larger zero-pads for a finer interpolated grid over the same span Cost: Unseen length builds a MatX FFT plan; the stage keys `seen_fft_keys` and warns once per new length. |

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

## Performance

**Cold.** First tick (and first unseen window / dechirp-reference / FFT-length key) builds `RangeStageState`: `premultipliers` (complex path), `windows` (real path), and `seen_fft_keys`. MatX owns the cuFFT plan cache. `range_bins` is filled directly on the compute stream by a MatX range generator.

**Steady.** Cached coefficients are reused; each tick loans `range_profiles` (and `range_bins` when a waveform is set) and enqueues the FFT. `[live]` `nfft` to an unseen length builds a new MatX plan. `[live]` `apply_dechirp` / `range_window` / `waveform` rebuild the premultiplier or window taps on an unseen key. `fft_norm` and the parameter-only `propagation_speed` do not invalidate those caches.

Harness: `performance/cpp/perf_fmcw_range_processor.cpp` (Python mirror exists but is not built in the current build). Axes: size ladder `{xs, sm, md, lg, xl}` = `(16×128) … (256×2048)` chirps × samples; window `{rect, hann}`; optional channel batch (`md` × 4). Default matrix: rect `{xs, md, lg, xl}` plus hann/`md` and rect/`md`/4 channels. `--full` crosses windows × sizes and adds channel batches on `md`/`lg`. No measured numbers are recorded in-tree; run the harness for timings.

```bash
ctest --test-dir build -R perf_fmcw_range_processor_smoke
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
