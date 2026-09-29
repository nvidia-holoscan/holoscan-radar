# `FMCWRangeDopplerProcessorOp`

## Purpose

Fused range compression, optional inline clutter, and Doppler compression in one graph node — a drop-in for the split `FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp` chain when intermediate range profiles are not needed. It calls the same stage functions as the split operators (range FFT materialized internally; clutter composed into the Doppler pre-FFT expression). There is no `range_profiles` output, and `range_bins` / `velocity_bins` are emitted only when a valid `waveform` is set.

## Responsibilities

- Runs the shared range, optional clutter, and Doppler stages inside one operator while preserving split-chain numerical semantics.
- Owns both FFT/window caches, intermediate workspace, tick-local overrides, and optional physical range/velocity axes.
- Does not expose intermediate range profiles or replace TDM/DDM compensation and demultiplexing.

## When to use

Use when range compression, optional clutter suppression, and Doppler processing can be one graph node and no consumer needs intermediate range profiles. It can replace the split chain for SISO/SIMO and after TDM deinterleave; TDM compensation still follows. Prefer the split operators for intermediate inspection or independently wired stages, and do not enable ordinary clutter filtering on coded pre-demultiplex DDM chirps.

```text
adc_cube → FMCWRangeDopplerProcessorOp → rd_maps
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `adc_cube` | complex64 / complex128, or float32 / float64 | `(num_chirps, samples_per_chirp)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Required trigger/aligned input | Same path selection as `FMCWRangeProcessorOp` |
| out | `rd_maps` | complex64 / complex128 | same leading + `(num_doppler, num_range_bins)` | Required output | Always emitted. `num_range_bins` as the range operator; `num_doppler` as the Doppler operator after optional clutter chirp loss |
| out (cond.) | `range_bins` | float32 / float64 | `(num_range_bins,)` | Optional latched output | Emitted only when a valid `waveform` is available. Optional (`latched`): the first emit to an unwired port warns once, after which the axis is no longer built |
| out (cond.) | `velocity_bins` | float32 / float64 | `(num_doppler,)` | Optional latched output | Emitted only when a valid `waveform` is available. Optional (`latched`); declining this port does not stop `range_bins` |
| live | `range_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `doppler_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `apply_dechirp` | bool | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `clutter_filter` | `ClutterFilterMethod` | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable. `none` is legal here (unlike standalone `ClutterFilterOp`) |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `fft_norm` | `FftNorm` | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable. One convention for both FFTs |
| live | `center_zero_doppler` | bool | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `nfft_range` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `nfft_doppler` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_chirps` and `max_range_bins` required; channel/TX bounds optional | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `range_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Fast-time window Cost: Rebuilds range premultiplier / taps, warning once. |
| `doppler_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Slow-time analysis window and, for `mean_subtraction`, clutter-projection weights Cost: Rebuilds Doppler/projection taps, warning once. |
| `apply_dechirp` | bool | `false` | `[live]` Exact | Software dechirp; needs waveform + complex IF Cost: Rebuilds range premultiplier. |
| `clutter_filter` | `ClutterFilterMethod` {`none`, `mean_subtraction`, `single_delay`, `double_delay`} | `none` | `[live]` Exact | Inline clutter; mean subtraction uses `doppler_window`, other methods ignore it for projection Cost: None (expression choice). Delay cancellers shrink `num_doppler`. |
| `waveform` | `FMCWWaveformDescriptor` | none (`is_valid() == false`) | `[live]` Exact via wire schema | Required for dechirp and/or `range_bins` / `velocity_bins` Cost: Dechirp reference + both axis fills. |
| `fft_norm` | `FftNorm` {`backward`, `ortho`, `forward`} | `backward` | `[live]` Exact | Shared FFT normalization Cost: Free. |
| `center_zero_doppler` | bool | `true` | `[live]` Exact | Center DC on the Doppler axis Cost: Free. |
| `propagation_speed` | float64 | `299792458.0` m/s | Params-only | Beat → range Cost: Free (device fill of `range_bins` only). |
| `nfft_range` | int64 | `0` (= `samples_per_chirp`) | `[live]` Exact | Range FFT length Cost: Unseen length builds a range FFT plan, warning once. |
| `nfft_doppler` | int64 | `0` (= post-clutter chirp count) | `[live]` Exact | Doppler FFT length Cost: Unseen length builds a Doppler FFT plan, warning once. |

Contract: [`OPERATOR_API.md` § `FMCWRangeDopplerProcessorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md) ·

## Algorithm

There is no fused-specific stage header. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype and named-axis shape (`collapse_named_tx_rx_batch`), merges this tick's Exact matches onto pinned Params (`resolve_params` → `FusedStageConfig`, sentinels mapped, `propagation_speed` copied from pinned, one `waveform` copied onto both stage configs), and binds that config to this input (`build_fused_stage_plan` → `FusedStagePlan`). Internal `profiles` is written first (not a port). Required `rd_maps` is the first published output (`emit_prepared_tensor`); optional `range_bins` / `velocity_bins` are later gated blocks (`LatchedOutput::emit`). One `waveform` feeds both fills; the latches are independent.

Per tick the shell sequences the split stages:

1. **Range.** `run_range_stage` into a stream-ordered MatX tensor `profiles` of shape `(batch, num_chirps, num_range_bins)` (`MATX_ASYNC_DEVICE_MEMORY`). Same dechirp / window / C2C-or-R2C FFT / no-`fftshift` rules as [`FMCWRangeProcessorOp`](../fmcw_range_processor_op/README.md). This cube never leaves the operator.
2. **Inline clutter.** `with_clutter_expr(clutter_filter, profiles, num_chirps, …)` — the same expression as [`ClutterFilterOp`](../clutter_filter_op/README.md). `none` passes `profiles` through (zero added kernels). Delay cancellers shrink the chirp axis; `mean_subtraction` does not and uses the resolved `doppler_window` for its weighted projection. The expression is **not** materialized to its own cube.
3. **Doppler.** That expression is the `in_op` of `run_doppler_stage` (expression overload). Same window / FFT / `fftshift` / `v = −f_d·λ/2` rules as [`FMCWDopplerProcessorOp`](../fmcw_doppler_processor_op/README.md). `num_chirps` seen by Doppler is the post-clutter count. MatX evaluates the clutter expression inside the fused pass that materializes the FFT input.
4. **Axes (conditional).** When a valid `waveform` is present, `fill_range_bins` and `fill_velocity_bins` run after the required `rd_maps` emit, each gated on its own `LatchedOutput` (`range_bins_out_.active()` / `velocity_bins_out_.active()`). The two `has_value()` checks are the same optional: `resolve_params` copies one descriptor onto both stage configs. Both fills are MatX expressions on the compute stream (no host staging).

`fft_norm` is one value applied to both transforms. When `clutter_filter = mean_subtraction`, `doppler_window` intentionally supplies both the clutter-projection weights and the Doppler analysis taper. This corrects the prior fused behavior, which projected with a uniform mean before applying a tapered window. Split chains require matching `ClutterFilterOp.doppler_window` and `FMCWDopplerProcessorOp.doppler_window` values for parity. Invariants of the split operators (bin 0 = zero range; zero-Doppler centered by default; `v_r > 0` receding) hold.

The fused `rd_maps` follow the split-chain numerical contract. The shared stages make that true for delay cancellers and the no-clutter path to within a few ULP of the split Graph (host capture / re-upload between split nodes). `mean_subtraction` is **not** bit-exact: MatX `mean` reduction order can differ between a materialized clutter write and the fused pre-FFT expression. The accepted parity bound is **8 ULP** (`kParityUlps`), not bitwise equality.

## Shape and dtype rules

- **`rd_maps` rank** = input rank. Trailing axes are `(num_doppler, num_range_bins)`:
  - `num_range_bins = range_num_bins(complex_input, samples_per_chirp, nfft_range)`
  - `chirps_out = clutter_num_chirps_out(clutter_filter, num_chirps)`
  - `num_doppler = doppler_num_bins(chirps_out, nfft_doppler)`
- **`range_bins` / `velocity_bins`** are rank-1, lengths `num_range_bins` / `num_doppler`.
- **Named leading axes.** None, `N_rx`, or `(M_tx, N_rx)`, admitted only up to `named_tx_rx_leading_rank(extents)`. Extra leading axes fail with `FMCWRangeDopplerProcessorOp: unexpected leading axes (opaque batch is not supported); got leading rank <got>, max named leading <max>`. A leading product above `max_tx * max_rx` fails with `FMCWRangeDopplerProcessorOp: leading product <n> exceeds max_tx*max_rx from MaxTensorExtents`.
- **Real IF vs complex IF.** Same dtype auto-select and mismatch messages as the range operator (`adc_cube: …`).
- **`nfft_range`.** When non-zero, `≥ samples_per_chirp` or `FMCWRangeDopplerProcessorOp: nfft_range (<n>) must be >= samples_per_chirp (<s>) on port adc_cube`.
- **`nfft_doppler`.** When non-zero, `≥` the **post-clutter** chirp count or `FMCWRangeDopplerProcessorOp: nfft_doppler (<n>) must be >= the post-clutter chirp count (<c>)`.
- **`apply_dechirp`.** Same two errors as the range operator (missing waveform / real IF), prefixed with `FMCWRangeDopplerProcessorOp`.
- **Delay-canceller length.** `FMCWRangeDopplerProcessorOp: clutter_filter 'single_delay' needs num_chirps >= 2, got <n> on port adc_cube` / `… 'double_delay' needs num_chirps >= 3, got <n> on port adc_cube`.
- **Layout.** `adc_cube` must be CUDA-accessible and compact row-major: `FMCWRangeDopplerProcessorOp: tensor received on input port 'adc_cube' is not CUDA-accessible …` / `… must be contiguous (compact row-major)`.
- **No CUDA stream.** `FMCWRangeDopplerProcessorOp: no CUDA compute stream`.
- **`[live]` shape changes.** `nfft_range`, `nfft_doppler`, and a delay-canceller `clutter_filter` change output extents; downstream re-infers.

## Performance

**Cold.** First tick (and first unseen window / dechirp-reference / FFT-length key) fills the ctor-allocated `RangeStageState` and `DopplerStageState` caches. Each tick also allocates the internal `profiles` cube (async device memory). Axis fills are MatX generators on the compute stream.

**Steady.** Cached coefficients reused; each tick loans `rd_maps` (and the two axes when a waveform is set), writes `profiles`, and enqueues clutter-into-Doppler. Versus the split chain this saves one full-cube write+read, one kernel launch, and one async alloc/free per tick **plus** the graph-level cost of the extra split nodes. The FFTs still dominate. Compare whole workflows when deciding whether fusion helps an application.

Harness: `performance/operators/fmcw/fmcw_range_doppler_processor/recipe.yaml` defines small, medium, and large profiles plus explicit parameter and one-frame live cases; its colocated `adapter.cpp` supplies contract-specific values. The tool times this operator alone. No measured numbers are recorded in-tree.

```bash
ctest --test-dir build -R perf_fmcw_range_doppler_processor_smoke
```

## Usage

The example applications use the split chain. A fused node is constructed the same way:

```yaml
range_doppler_processor:
  waveform: *waveform
  range_window: hann
  doppler_window: hann
  clutter_filter: mean_subtraction
  # apply_dechirp: false
  # fft_norm: backward
  # center_zero_doppler: true
  # nfft_range: 512
  # nfft_doppler: 256
  # propagation_speed: 299792458.0
```

```cpp
auto rd = graph.op<FMCWRangeDopplerProcessorOp<float>>(
    "range_doppler_processor", extents);
rd.set_params(cfg.rd_params);

graph.add_flow(source->adc_cube, rd->adc_cube, q);
graph.add_flow(rd->rd_maps, cfar->rd_maps, q);
graph.add_flow(rd->range_bins, sink->range_bins, q);      // only if waveform is set
graph.add_flow(rd->velocity_bins, sink->velocity_bins, q);
```
