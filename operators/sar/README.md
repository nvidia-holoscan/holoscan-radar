# SAR formation operators

GPU operators for synthetic-aperture image formation by time-domain
backprojection. The chain runs phase history → range profiles → image, with
optional autofocus, and is specified in
[`docs/planning/SAR/OPERATOR_API.md`](../../docs/planning/SAR/OPERATOR_API.md) —
the locked contract for every port, shape, dtype pair and parameter. Each
operator is templated on precision (`<float>` / `<double>`); module-wide port,
batching, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's header
carries its full port/parameter reference.

The standard-format readers and writers that bracket this chain are **not** in
this group: they live in [`operators/io/`](../io/README.md), so a build that
wants only formation pays nothing for the XML and NITF dependencies.

Typical placement (wave one, autofocus off — see
[`docs/planning/SAR/WORKFLOWS.md`](../../docs/planning/SAR/WORKFLOWS.md) for
the full wiring, including where `ApertureManagerOp` sits):

```text
CPHDReaderOp → phase_history → SARRangeCompressionOp → range_profiles → ApertureManagerOp → BackprojectionOp → image → …
```

## Operators

`SARRangeCompressionOp` has landed; `BackprojectionOp` is in flight as a
mode-1 slice (not checklist-complete), in the build order set by
[`docs/planning/SAR/SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md):

| Operator | In → Out | Function |
| --- | --- | --- |
| `SARRangeCompressionOp<T>` | `phase_history` → `range_profiles` (+ `range_bins`) | Inverse transform of deramped phase history to the range axis |
| `ApertureManagerOp<T>` | `range_profiles`, `positions` → same + `weight` + `control` | Aperture membership, emit cadence, geometry staging; no kernel |
| `BackprojectionOp<T>` | `range_profiles`, `positions`, `control` → `image` | Time-domain backprojection onto a `SampleGeometry` lattice or an explicit `sample_points` one, accumulating across ticks |
| `PhaseCorrectionOp<T>` | `phase_history`, `phase_correction` → `phase_history` | Broadcast multiply applying an autofocus estimate |
| `AutofocusOp<T>` | `image` → `phase_correction` | Phase-gradient autofocus estimate over a conjugate grid |
| `RangeReferenceMultiplyOp<T>` | `pulses` → `phase_history` | Direct-sampled branch entry: forward FFT and multiply by the range reference function |

## Implementation status

The contract was accepted on 2026-08-07 and implementation is sequenced by the
minimum-per-demo rule in
[`SIGPROC_PLAN.md`](../../docs/planning/SAR/SIGPROC_PLAN.md), so this group
fills in demo order rather than in the table order above:
`SARRangeCompressionOp` then `BackprojectionOp` for the mode 1 spotlight demo,
with autofocus and the direct-sampled branch later.

**`ApertureManagerOp` — landed, not checklist-complete.** CPU-only
membership / emit / D22 flush / geometry staging. Modes 1 and 5 of
`sar_formation` wire the spotlight row (`max_aperture_pulses = 0`,
`emit_mode = reset`). Retention, application-owned `evict`, and
`emit_mode = reform` are implemented and unit-tested; signed `weight`
stays unconnected on those demos. No perf harness (no GPU hot path).
Follow-ups in
[`aperture_manager_op/OPEN_ITEMS.md`](aperture_manager_op/OPEN_ITEMS.md).

**`BackprojectionOp` — in flight, mode-1 slice (not checklist-complete).**
Required ports, four control phases, `kUpdate`, `stop()` warning, linear
interpolation, `geometry_change = reset`. Shipped backends: default
`nvradar_block_anchored_expanded_throughput_phase_lut_vec`, golden
`nvradar_fp64_reference`, Taylor
`nvradar_block_anchored_refined_throughput`,
naive/HQ `nvradar_cuda_fp32{,_hq,_hq_fast}`, every remaining `nvradar_*`
copy, and the `matx_*` wrap (except `matx_taylor_fast`) when the bundled
header is present.
`initial_image` and `scroll` are declared and rejected. I6/I7 closed.
Follow-ups in
[`backprojection_op/OPEN_ITEMS.md`](backprojection_op/OPEN_ITEMS.md).

**`SARRangeCompressionOp` — landed, contract-complete.** All three output
ports, all five `[live]` parameters, `nfft` zero-padding (spectral edges of
the rotated row), a leading `ifftshift` so IFFT DC is stored sample `N/2`,
and the centred range axis. The transform is the **inverse** one — phase
history is the frequency-domain form — which is the opposite direction from
FMCW's `FMCWRangeProcessorOp`, so `fft_norm = backward` applies the 1/N
*here* while the same spelling leaves the FMCW stage unscaled.

Three things about it are worth knowing before reading the code:

- **The centring `fftshift` is a correctness requirement, not presentation.**
  Without it every sample nearer than the reference point wraps to the top of
  the array, where `BackprojectionOp`'s bounds check drops it — roughly half the
  swath, silently. It costs a second pass over the cube (the shift is a gather,
  so it cannot be done in place) into a shape-cached scratch buffer. A leading
  `ifftshift` is a separate convention: it makes `range_profiles` complex
  baseband (IFFT DC = stored sample `N/2`) so formation's `reference_frequency`
  is that sample, not `SC0`.
- **The tests assert phase, not just peak position.** A point target's
  compressed profile is a Dirichlet kernel whose complex value at every bin is
  predictable in closed form, and asserting against that is what makes an
  inverted transform, a missing shift, an `(N−1)/2` centring and a wrong
  `fft_norm` all visible — each of which survives a magnitude-only test. The
  input comes from `radar_common/sim/sar_phase_history_sim.hpp`, the SAR
  counterpart to the FMCW point-target simulator, which carries the same phase
  and frequency-axis convention as `tools/sar_sim`.
- **`performance/cpp/perf_sar_range_compression.cpp`** prices the axes the FMCW
  range stage does not have: the shift pass, the optional range axis, and the
  `[live]` `nfft` override, which is the only override here that rebuilds
  anything costly and so gets four separated transitions sampled per session
  rather than one spike frame. Three findings are in
  [`sar_range_compression_op/OPEN_ITEMS.md`](sar_range_compression_op/OPEN_ITEMS.md):
  a ~0.2 ms step in per-tick cost when the output crosses ~2 MiB, independent of
  the work done (item 4 — the same per-tick `cudaMalloc` `CPHDReaderOp` found
  from the other direction); a **prime `nfft` costing ~170 ms on a cold CUDA JIT
  cache** and ~5 ms warm, against 0.35 ms for a power of two (item 5); and the
  range-axis emission costing ~18 µs a tick, most of it a host-blocking
  `cudaStreamSynchronize` (item 6).

I5 — whether `backward` is the right normalization default — **remains open**;
the operator implements the contract's stated default, which does not settle it.

Changes to the SAR contract require the developer's approval in session before
the edit — see [`AGENTS.md`](../../AGENTS.md) → *Docs first*.
