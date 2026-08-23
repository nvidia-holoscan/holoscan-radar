# FMCW signal-processing operators

GPU (MatX) operators for the FMCW radar chain. Each operator is templated on
precision (`FMCWRangeProcessorOp<float>` / `<double>`); module-wide port,
batching, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's
header carries its full port/parameter reference.

Typical chains:

```text
split:  adc_cube → FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp → rd_maps
fused:  adc_cube → FMCWRangeDopplerProcessorOp → rd_maps
```

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| [`FMCWRangeProcessorOp<T>`](fmcw_range_processor_op/README.md) | `adc_cube` (…, chirps, samples; complex or real IF) → `range_profiles` (…, chirps, range bins) + optional `range_bins` axis | Range compression: optional software dechirp, windowed fast-time FFT (`nfft` zero-pad, `fft_norm`); real IF takes the R2C path (`nfft/2+1` bins); bin 0 = zero range (no fftshift) |
| [`ClutterFilterOp<T>`](clutter_filter_op/README.md) | `range_profiles` → `range_profiles_out` (chirp count preserved / −1 / −2) | Slow-time clutter suppression: `mean_subtraction` (zero-Doppler nulled, no chirp loss) or 1st/2nd-order MTI delay cancellers. **[EA HSDK5 ONLY]** ported on HSDK 5; see [`HSDK5_PORT.md`](../../docs/planning/HSDK5_PORT.md) + operator `OPEN_ITEMS.md` |
| [`FMCWDopplerProcessorOp<T>`](fmcw_doppler_processor_op/README.md) | `range_profiles` → `rd_maps` (…, doppler, range bins) + optional `velocity_bins` axis | Doppler compression: windowed slow-time FFT (`nfft` zero-pad), zero-Doppler centered by default (numpy-style fftshift) |
| [`FMCWRangeDopplerProcessorOp<T>`](fmcw_range_doppler_processor_op/README.md) | `adc_cube` → `rd_maps` + optional `range_bins`/`velocity_bins` | Fused range + optional inline clutter + Doppler in one node; per-axis `nfft_range`/`nfft_doppler` |
| [`MIMODeinterleaveOp<T>`](mimo_deinterleave_op/README.md) | `adc_cube` (…, N_rx, M·L, samples; complex or real IF) → `adc_cube` (…, M_tx, N_rx, L, samples) | TDM marshaling: split interleaved slow-time into an explicit TX axis via `TXSchedule` gather. **Uniform round-robin TDM only** — non-uniform / staggered schedules are rejected (staggered TDM deferred; STATUS.md L12) |
| [`VirtualArrayAssemblyOp<T>`](virtual_array_assembly_op/README.md) | `rd_maps` (…, M_tx, N_rx, D, R) → `rd_maps` (…, num_virtual, D, R) + `array_geometry` | MIMO marshaling: flatten separated TX×RX channels into a virtual axis (`p_tx + p_rx`) |
| [`MIMOCoherentCompensationOp<T>`](mimo_coherent_compensation_op/README.md) | TDM: `rd_maps` (…, M_tx, N_rx, D, R) → same shape. DDM: `rd_maps` (…, N_rx, D, R) → (…, M_tx, N_rx, D/M−guard, R) | MIMO marshaling: TDM per-Doppler-bin inter-TX phase correction; DDM Doppler-band demux into an explicit TX axis (standard linear-phase; arbitrary codes deferred) |

MIMO marshaling operators document **imposed decisions** (contract pins) and
deferred slices in each operator's `OPEN_ITEMS.md`.

The optional axis outputs are emitted only when a waveform descriptor
(`FMCWWaveformDescriptor`) is available; consumers map peaks back to
physical units through them.

## Shared kernel conventions

- **Stage functions.** Each operator's math is a pure stage function in its
  own directory (`range_stage.hpp`, `clutter_stage.hpp`,
  `doppler_stage.hpp` — CUDA-TU-only headers); `compute()` is a thin shell.
  The fused operator calls the same stage functions back-to-back, which is
  why its output is bitwise-identical to the split chain for matching
  parameters (verified by the parity test suite).
- **Dechirp reference convention.** `apply_dechirp` conjugate-multiplies
  the input against the reference chirp
  `r[s] = exp(+j·π·slope·t_s²)`, `t_s = s / sample_rate` (or
  `waveform.reference_samples` when provided). Software dechirp requires
  complex-IF input — dechirping real IF would produce complex samples,
  inconsistent with the real-IF output size.
- **Sign conventions.** Radial velocity `v_r > 0` = receding; the velocity
  axis satisfies `v = -f_d·λ/2`; the range axis is
  `range_bins[i] = propagation_speed·(i·fs/nfft)/(2·slope)`.
- **Streams.** [EA HSDK5 ONLY] Ported operators enqueue on
  `context.cuda_stream()` (no hot-path synchronization) and retain the
  received input until the next tick as a stream-ordered lifetime guard.
  Unported operators still use the HSDK 4.x `receive_cuda_stream()` path.
