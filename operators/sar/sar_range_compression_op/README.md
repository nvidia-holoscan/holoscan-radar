# `SARRangeCompressionOp`

## Purpose

Window and inverse-transform CPHD phase history into centered differential-range profiles.

## Responsibilities

- Apply the public SAR operator contract and report invalid shapes or configurations.
- Preserve capture-time alignment and output the documented tensor or descriptor payloads.

## When to use

Use after CPHD ingress or any producer of frequency-domain SAR phase history. The inverse transform, leading ifftshift, and output fftshift preserve the centred range convention needed by backprojection; a waveform supplies the optional physical range axis.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | phase_history | complex T | (num_pulses,num_samples) | required | Frequency-domain history |
| In | range_window, waveform, fft_norm, nfft, propagation_speed | descriptor / scalar | scalar | optional [live] | Tick-local configuration |
| Out | range_profiles | complex T | (num_pulses,num_range_bins) | every tick | Compressed samples |
| Out | range_bins | real T | (num_range_bins) | optional | Differential range axis |
| Out | range_window_out | descriptor | scalar | optional | Applied window |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `SARRangeCompressionOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- |
| `range_window` | `WindowSpec` | rectangular | [live] | Fast-time taper |
| `waveform` | `LFMWaveformDescriptor` | unset | [live] | Physical range axis |
| `fft_norm` | `FftNorm` | backward | [live] | Inverse FFT scaling |
| `nfft` | `int64` | observed samples | [live] | Zero-padded transform length |
| `propagation_speed` | `double` | 299792458 m/s | [live] | Frequency-to-range scale |

The full shapes, defaults, numerical conventions, and open decisions are in
[SAR OPERATOR_API.md](../../../docs/planning/SAR/OPERATOR_API.md).
