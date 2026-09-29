# `SARRangeCompressionOp`

## Purpose

Window and inverse-transform CPHD phase history into centered differential-range profiles.

## Responsibilities

- Resolve the tick-local window, waveform, FFT length, and normalization.
- Produce centered differential-range profiles and optional physical range/window outputs.

## When to use

Use after CPHD ingress or any producer of frequency-domain SAR phase history. The inverse transform, leading ifftshift, and output fftshift preserve the centred range convention needed by backprojection; a waveform supplies the optional physical range axis.

## Ports

`T=float` means complex64/float32; `T=double` means complex128/float64.
The five `[live]` inputs are declared unconditionally and use same-tick
`Exact{}`/`SafeDefault{}`; an absent value uses Params for this tick.

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | `phase_history` | complex `T` | `(num_pulses,num_samples)` | Required | triggers each tick |
| In | `range_window` | `WindowSpec` descriptor | descriptor | Optional `[live]` | Tick-local fast-time taper |
| In | `waveform` | `LFMWaveformDescriptor` descriptor | descriptor | Optional `[live]` | Tick-local range-axis description |
| In | `fft_norm` | `FftNorm` | scalar | Optional `[live]` | Inverse FFT scaling |
| In | `nfft` | `int64` | scalar | Optional `[live]` | Tick-local transform length |
| In | `propagation_speed` | `double`, m/s | scalar | Optional `[live]` | Range scale in m/s |
| Out | `range_profiles` | complex `T` | `(num_pulses,num_range_bins)` | Every input tick | Pulse data |
| Out | `range_bins` | real `T` | `(num_range_bins)` | Optional, latched | only with valid waveform |
| Out | `range_window_out` | `WindowSpec` descriptor | descriptor | Optional, latched | applied window |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `SARRangeCompressionOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

Set `max_pulses`, `max_samples_per_pulse`, and `max_range_bins` for the largest tick and padded transform length. A tick whose inferred extent exceeds a bound fails rather than resizing a live Tensor pool.

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `range_window` | `WindowSpec` | rectangular | [live] | Fast-time taper |
| `waveform` | `LFMWaveformDescriptor` | unset | [live] | Physical range axis |
| `fft_norm` | `FftNorm` | backward | [live] | Inverse FFT scaling |
| `nfft` | `int64` | observed samples | [live] | Zero-padded transform length |
| `propagation_speed` | `double` | 299792458 m/s | [live] | Frequency-to-range scale |

## Transform and axis conventions

For each pulse, the stage generates the selected fast-time window, rotates
the frequency samples with `ifftshift`, zero-pads to `nfft` if requested,
runs an inverse FFT, and centres the result with `fftshift`. Thus
`num_range_bins = nfft` when positive, otherwise `num_samples`. `nfft` must
be at least the observed sample count; padding is symmetric around the
frequency-domain centre. `fft_norm=backward` is the shipped inverse-FFT
normalization. Window coefficients and FFT configuration are cached; a new
costly `[live]` configuration warns once.

`range_bins` is attempted only when a valid `LFMWaveformDescriptor` is
available; a connected output receives it. It describes the centred
differential-range axis from waveform sample spacing and
`propagation_speed`. The descriptor output `range_window_out` reports the
applied window when connected. An unwired optional output is latched after
its first declined emit, avoiding later dedicated output work. The main `range_profiles` tensor emits
on every input tick. Each `[live]` port uses same-tick `Exact{}` with Params
as the safe default; no earlier override persists.

## Validation and use

Input must be a contiguous CUDA-accessible complex tensor of rank two with
positive named extents. Output is contiguous complex `T` with the same pulse
count. The operator rejects non-positive propagation speed, negative `nfft`,
undersized `nfft`, and any input or output exceeding `MaxTensorExtents`.
Both `float` and `double` instances are supported; the instance dtype is
fixed at construction. Example static configuration:

```yaml
range_compression:
  range_window: taylor
  propagation_speed: 299792458.0
```

Applications may pass an app-derived `nfft` at graph construction. The C++
range-compression tests compare impulse and analytic point-target responses,
FFT shift, padding, and axis spacing. See the
[SAR operator contract](../../../docs/planning/SAR/OPERATOR_API.md) for the
locked port definitions and `OPEN_ITEMS.md` for the radiometric-default review.
