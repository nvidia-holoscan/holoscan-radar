# FMCW signal-processing operators

GPU (MatX) operators for the FMCW radar chain. Each operator is templated on
precision (`FMCWRangeProcessorOp<float>` / `<double>`); module-wide port,
tensor, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's
header carries its full port/parameter reference.

Typical chains (split form; `FMCWRangeDopplerProcessorOp` drops in for
range + optional clutter + Doppler wherever no MIMO marshaling step
interposes). Downstream detection and clustering are not shown.
`⟨AoA operator⟩` is the angle-estimation slot — not a type in this group.

```text
SISO / SIMO (split):
  adc_cube → FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp → rd_maps
SISO / SIMO (fused):
  adc_cube → FMCWRangeDopplerProcessorOp → rd_maps

MIMO TDM (split):
  adc_cube → MIMODeinterleaveOp → FMCWRangeProcessorOp → [ClutterFilterOp]
           → FMCWDopplerProcessorOp → TDMDenseCompensationOp
           → VirtualArrayAssemblyOp → ⟨AoA operator⟩
MIMO TDM (fused):
  adc_cube → MIMODeinterleaveOp → FMCWRangeDopplerProcessorOp
           → TDMDenseCompensationOp → VirtualArrayAssemblyOp → ⟨AoA operator⟩

MIMO DDM (split):
  adc_cube → FMCWRangeProcessorOp → FMCWDopplerProcessorOp
           → DDMDenseDemultiplexOp → [DopplerNotchOp] → VirtualArrayAssemblyOp
           → ⟨AoA operator⟩
MIMO DDM (fused):
  adc_cube → FMCWRangeDopplerProcessorOp → DDMDenseDemultiplexOp
           → [DopplerNotchOp] → VirtualArrayAssemblyOp → ⟨AoA operator⟩
MIMO DDM sparse (linear exact-bin only):
  rd_maps → DDMPowerFoldOp → CFARDetectorOp
          → DDMSparseDemultiplexOp → ⟨sparse AoA operator⟩
```

SIMO continues from `rd_maps` to an AoA operator on the physical RX axis;
SISO has no angle stage. DDM has no deinterleave (TX fire together) and cannot
use ordinary pre-FFT slow-time clutter filtering. Dense DDM may apply a
physical-velocity `DopplerNotchOp` after demultiplexing. Linear-phase gather
and non-linear correlate share that chain.

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| [`FMCWRangeProcessorOp<T>`](fmcw_range_processor_op/README.md) | `adc_cube` (…, chirps, samples; complex or real IF) → `range_profiles` (…, chirps, range bins) + optional `range_bins` axis | Range compression: optional software dechirp, windowed fast-time FFT (`nfft` zero-pad, `fft_norm`); real IF takes the R2C path (`nfft/2+1` bins); bin 0 = zero range (no fftshift) |
| [`ClutterFilterOp<T>`](clutter_filter_op/README.md) | `range_profiles` → `range_profiles_out` (chirp count preserved / −1 / −2) | Slow-time clutter suppression: uniform or `doppler_window`-weighted `mean_subtraction` (zero-Doppler nulled, no chirp loss), or 1st/2nd-order MTI delay cancellers. |
| [`FMCWDopplerProcessorOp<T>`](fmcw_doppler_processor_op/README.md) | `range_profiles` → `rd_maps` (…, doppler, range bins) + optional `velocity_bins` axis | Doppler compression: windowed slow-time FFT (`nfft` zero-pad), zero-Doppler centered by default (numpy-style fftshift) |
| [`DopplerNotchOp<T>`](doppler_notch_op/README.md) | complex `(D,R)`, `(N_rx,D,R)`, or `(M_tx,N_rx,D,R)` + physical `(D,)` velocity axis → same RD shape | Hard or raised-cosine physical-velocity stopband applied identically across channels |
| [`FMCWRangeDopplerProcessorOp<T>`](fmcw_range_doppler_processor_op/README.md) | `adc_cube` → `rd_maps` + optional `range_bins`/`velocity_bins` | Fused range + optional inline clutter + Doppler in one node; `doppler_window` supplies both weighted mean projection and analysis taper under `mean_subtraction` |
| [`MIMODeinterleaveOp<T>`](mimo_deinterleave_op/README.md) | `adc_cube` (`N_rx`, `M·L`, samples; complex or real IF) → `adc_cube_out` (`M_tx`, `N_rx`, `L`, samples) | TDM marshaling: split interleaved slow-time into an explicit TX axis via `TXSchedule` gather. **Uniform repeating TDM only** — non-uniform or staggered schedules are rejected |
| [`TDMDenseCompensationOp<T>`](tdm_dense_compensation_op/README.md) | `(M_tx, N_rx, D, R)` → same shape | Firing-slot-aware TDM inter-TX motion-phase correction |
| [`DDMDenseDemultiplexOp<T>`](ddm_dense_demultiplex_op/README.md) | `(N_rx, D, R)` + optional `(D,)` velocity axis → `(M_tx, N_rx, D_out, R)` + optional `(D_out,)` axis | Linear DDM gather or nonlinear full-axis correlation with matching physical-axis transformation |
| [`DDMPowerFoldOp<T>`](ddm_power_fold_op/README.md) | `(N_rx,D,R)` → real `(D/M-G,R)` | Fold all linear DDM bands and RX looks into detector power |
| [`DDMSparseDemultiplexOp<T>`](ddm_sparse_demultiplex_op/README.md) | pre-demux maps + folded detections + optional fold index → snapshots + mask + geometry | Sparse linear DDM gather, fold recovery, and virtual assembly |
| [`TDMSparseCompensationOp<T>`](tdm_sparse_compensation_op/README.md) | uncompensated `rd_maps` + detections + optional fold index → snapshots + mask + geometry | Sparse TDM bridge: gather detected cells, apply firing-slot-aware Doppler/fold correction, and assemble virtual snapshots without processing the full cube |
| [`VirtualArrayAssemblyOp<T>`](virtual_array_assembly_op/README.md) | `rd_maps` (`M_tx`, `N_rx`, D, R) → `rd_maps_out` (`num_virtual`, D, R) + `WireArrayGeometry` | MIMO marshaling: flatten separated TX×RX channels into a virtual axis (`p_tx + p_rx`) |

The optional axis outputs are emitted only when a waveform descriptor
(`FMCWWaveformDescriptor`) is available; consumers map peaks back to
physical units through them.

## Signal conventions

- `FMCWRangeProcessorOp` uses bin zero for zero range and does not shift the
  range axis. `FMCWDopplerProcessorOp` centers zero Doppler by default.
- Positive radial velocity means receding, with `v = -f_d * wavelength / 2`.
  Range uses `propagation_speed * beat_frequency / (2 * slope)`.
- Software dechirp conjugate-multiplies complex IF against the configured or
  synthesized reference chirp. Real IF cannot use software dechirp because the
  product would no longer match the real-input FFT shape.
- Under `mean_subtraction`, the fused range-Doppler operator uses its Doppler
  window for both weighted clutter projection and Doppler analysis. A split
  chain must give `ClutterFilterOp` and `FMCWDopplerProcessorOp` the same window
  to reproduce that result.
- Range and velocity axis outputs require a waveform descriptor. Connect them
  when downstream operators must translate bin indices to physical units.

Shared precision, tensor, and live-override behavior is documented in
[`operators/common/README.md`](../common/README.md). Deferred waveform and
algorithm extensions are tracked in the repository
[`OPEN_ITEMS.md`](../../OPEN_ITEMS.md).
