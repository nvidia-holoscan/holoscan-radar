# `FMCWDopplerProcessorOp`

## Purpose

Doppler compression: a windowed FFT along slow-time that turns range profiles into range-Doppler maps. It sits after `FMCWRangeProcessorOp` (and optional `ClutterFilterOp`) on every split FMCW chain. Zero-Doppler is centered by default (`fftshift`, numpy semantics, including odd lengths); radial velocity uses `v = −f_d·λ/2` (`v_r > 0` = receding). `velocity_bins` is emitted only when a valid `waveform` is set.

## Responsibilities

- Applies the slow-time window, Doppler FFT, optional centering, and configured normalization.
- Owns FFT/window caching, tick-local processing overrides, and optional physical velocity-axis generation.
- Does not range-compress ADC samples, suppress clutter, or demultiplex TDM/DDM channels.

## When to use

Use after range compression and optional clutter filtering to transform slow-time range profiles into range-Doppler maps. Supply a waveform when downstream consumers need a physical velocity axis, and choose the fused range-Doppler operator when intermediate profiles are unnecessary. Do not use it on raw ADC fast-time samples or on data already in the Doppler domain.

```text
FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp → rd_maps
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `range_profiles` | complex64 / complex128 | `(num_chirps, num_range_bins)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Required trigger/aligned input | Slow-time FFT along `num_chirps`. Real IF is rejected |
| out | `rd_maps` | complex64 / complex128 | same leading + `(num_doppler, num_range_bins)` | Required output | Always emitted. `num_doppler = nfft` (default `num_chirps`). Zero-Doppler centered when `center_zero_doppler` |
| out (cond.) | `velocity_bins` | float32 / float64 | `(num_doppler,)` | Optional latched output | Emitted only when a valid `waveform` is available. Optional (`latched`): the first emit to an unwired port warns once, after which the axis is no longer built |
| live | `doppler_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `center_zero_doppler` | bool | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
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
| `doppler_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Slow-time window before the Doppler FFT; coherent-gain normalized when `coherent_gain_normalization` is true (WindowSpec default) Cost: Rebuilds cached window taps on an unseen spec, warning once. |
| `center_zero_doppler` | bool | `true` | `[live]` Exact | Center DC on the Doppler axis; else bin 0 is DC Cost: Free (permute / `fftshift1D` choice). |
| `waveform` | `FMCWWaveformDescriptor` | none (`is_valid() == false`) | `[live]` Exact via wire schema | Carrier → `λ`, PRI → PRF for the velocity axis Cost: Free for the FFT; required for `velocity_bins`. |
| `fft_norm` | `FftNorm` {`backward`, `ortho`, `forward`} | `backward` | `[live]` Exact | FFT normalization; `backward` keeps coherent-integration gain visible Cost: Free at the operator (MatX applies the convention on the FFT). |
| `nfft` | int64 | `0` (= `num_chirps`) | `[live]` Exact | Doppler FFT length; larger interpolates the velocity grid over the same span Cost: Unseen length builds a MatX FFT plan; the stage keys `seen_fft_keys` and warns once per new length. |

Contract: [`OPERATOR_API.md` § `FMCWDopplerProcessorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md) ·

## Algorithm

All math lives in `doppler_stage.hpp`. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype and named-axis shape (`collapse_named_tx_rx_batch`), merges this tick's Exact matches onto pinned Params (`resolve_params` → `DopplerStageConfig`, sentinels mapped), and binds that config to this input (`build_doppler_stage_plan` → `DopplerStagePlan` with `num_doppler`). Required `rd_maps` is prepared first; when active, optional `velocity_bins` is prepared and published before `rd_maps` so it can be an Exact causal sideband for immediate DDM demultiplexing. The fused operator feeds this same stage entry point (optionally with a clutter expression as `in_op`), which is why split-vs-fused Doppler is identical by construction.

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
- **Complex only.** `require_complex<T>`. Real input: `range_profiles: real-valued input <dtype> is not supported here; provide <complex>`.
- **`nfft`.** When non-zero it must be `≥ num_chirps`; otherwise `FMCWDopplerProcessorOp: nfft (<n>) must be >= num_chirps (<c>) on input port 'range_profiles'`. Downstream re-infers `num_doppler` after a `[live]` `nfft` change (or after a delay-canceller chirp loss upstream).
- **Layout.** `range_profiles` must be CUDA-accessible and compact row-major: `FMCWDopplerProcessorOp: tensor received on input port 'range_profiles' is not CUDA-accessible (device, managed, or pinned-host memory required)` / `… must be contiguous (compact row-major)`.
- **No CUDA stream.** `FMCWDopplerProcessorOp: no CUDA compute stream`.
- **`center_zero_doppler`.** Mid-stream flip changes bin order, not extents; `velocity_bins` (when emitted) is rebuilt to match.

## Performance

**Cold.** First tick (and first unseen window / FFT-length key) fills the ctor-allocated `DopplerStageState` caches: `windows` and `seen_fft_keys`. MatX owns the cuFFT plan cache. `velocity_bins` is filled directly on the compute stream by MatX `fftfreq` / `fftshift1D`.

**Steady.** Cached window taps are reused; each tick loans `rd_maps` (and `velocity_bins` when a waveform is set) and enqueues the permute + FFT + optional shift. `[live]` `nfft` to an unseen length builds a new MatX plan. `[live]` `doppler_window` rebuilds taps on an unseen key. `center_zero_doppler` and `fft_norm` do not invalidate those caches.

Harness: `performance/operators/fmcw/fmcw_doppler_processor/recipe.yaml` defines small, medium, and large chirps × range bins profiles, curated parameter cases, and one-frame `[live]` transitions, including cached and unseen FFT lengths. Its colocated `adapter.cpp` supplies contract-specific values. No measured numbers are recorded in-tree; run the harness for timings.

Run this performance smoke **inside a development container**, after
configuring the named build tree with performance enabled (see the
[developer guide](../../../DEVELOPER.md#performance-characterization)):

```bash
ctest --test-dir build -R perf_fmcw_doppler_processor_smoke
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
