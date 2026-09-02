# `FMCWRangeDopplerProcessorOp`

Fused range compression, optional inline clutter, and Doppler compression in one graph node — a drop-in for the split `FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp` chain when intermediate range profiles are not needed. It calls the same stage functions as the split operators (range FFT materialized internally; clutter composed into the Doppler pre-FFT expression). There is no `range_profiles` output. `range_bins` / `velocity_bins` are emitted only when a valid `waveform` is set. Whether the fused node earns its keep is an open team decision ([`OPEN_ITEMS.md`](OPEN_ITEMS.md) §1).

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `adc_cube` | complex64 / complex128, or float32 / float64 | `(num_chirps, samples_per_chirp)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Same path selection as `FMCWRangeProcessorOp` |
| out | `rd_maps` | complex64 / complex128 | same leading + `(num_doppler, num_range_bins)` | Always emitted. `num_range_bins` as the range operator; `num_doppler` as the Doppler operator after optional clutter chirp loss |
| out (cond.) | `range_bins` | float32 / float64 | `(num_range_bins,)` | Emitted only when a valid `waveform` is available. Optional (`latched`): the first emit to an unwired port warns once, after which the axis is no longer built |
| out (cond.) | `velocity_bins` | float32 / float64 | `(num_doppler,)` | Emitted only when a valid `waveform` is available. Optional (`latched`); declining this port does not stop `range_bins` |
| live | `range_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |
| live | `doppler_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |
| live | `apply_dechirp` | bool | scalar | Exact override; wire to enable, unwired = miss |
| live | `clutter_filter` | `ClutterFilterMethod` | scalar | Exact override; wire to enable. `none` is legal here (unlike standalone `ClutterFilterOp`) |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |
| live | `fft_norm` | `FftNorm` | scalar | Exact override; wire to enable. One convention for both FFTs |
| live | `center_zero_doppler` | bool | scalar | Exact override; wire to enable, unwired = miss |
| param | `propagation_speed` | float64 | scalar | Params-only (no Exact port). Wave speed for `range_bins`; defaults to the speed of light |
| live | `nfft_range` | int64 | scalar | Exact override; wire to enable, unwired = miss |
| live | `nfft_doppler` | int64 | scalar | Exact override; wire to enable, unwired = miss |

```text
adc_cube → FMCWRangeDopplerProcessorOp → rd_maps
```

On MIMO-TDM the fused node may replace the range/Doppler pair **after** `MIMODeinterleaveOp`; compensation and virtual-array assembly still follow. MIMO-DDM cannot use the fused clutter stage in place of post-demux `ClutterFilterOp`.

`range_window`, `doppler_window`, and `waveform` remain logical `WindowSpec` / `FMCWWaveformDescriptor` Params. Their HSDK 5 physical ports use the L18 generated `holoscan::radar::wire` descriptor-Tensor encoding because the SDK does not admit container-owning C++ types directly. The Tensor member is always present: rank-1 zero-length represents an absent optional array, while null is invalid. See [`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md).

Contract: [`OPERATOR_API.md` § `FMCWRangeDopplerProcessorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md) ·
open items: [`OPEN_ITEMS.md`](OPEN_ITEMS.md)

## Algorithm

There is no fused-specific stage header. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype and named-axis shape (`collapse_named_tx_rx_batch`), merges this tick's Exact matches onto pinned Params (`resolve_params` → `FusedStageConfig`, sentinels mapped, `propagation_speed` copied from pinned, one `waveform` copied onto both stage configs), and binds that config to this input (`build_fused_stage_plan` → `FusedStagePlan`). Internal `profiles` is written first (not a port). Required `rd_maps` is the first published output (`emit_prepared_tensor`); optional `range_bins` / `velocity_bins` are later gated blocks (`LatchedOutput::emit`). One `waveform` feeds both fills; the latches are independent.

Per tick the shell sequences the split stages:

1. **Range.** `run_range_stage` into a stream-ordered MatX tensor `profiles` of shape `(batch, num_chirps, num_range_bins)` (`MATX_ASYNC_DEVICE_MEMORY`). Same dechirp / window / C2C-or-R2C FFT / no-`fftshift` rules as [`FMCWRangeProcessorOp`](../fmcw_range_processor_op/README.md). This cube never leaves the operator.
2. **Inline clutter.** `with_clutter_expr(clutter_filter, profiles, num_chirps, …)` — the same expression as [`ClutterFilterOp`](../clutter_filter_op/README.md). `none` passes `profiles` through (zero added kernels). Delay cancellers shrink the chirp axis; `mean_subtraction` does not. The expression is **not** materialized to its own cube.
3. **Doppler.** That expression is the `in_op` of `run_doppler_stage` (expression overload). Same window / FFT / `fftshift` / `v = −f_d·λ/2` rules as [`FMCWDopplerProcessorOp`](../fmcw_doppler_processor_op/README.md). `num_chirps` seen by Doppler is the post-clutter count. MatX evaluates the clutter expression inside the fused pass that materializes the FFT input.
4. **Axes (conditional).** When a valid `waveform` is present, `fill_range_bins` and `fill_velocity_bins` run after the required `rd_maps` emit, each gated on its own `LatchedOutput` (`range_bins_out_.active()` / `velocity_bins_out_.active()`). The two `has_value()` checks are the same optional: `resolve_params` copies one descriptor onto both stage configs. Both fills are MatX expressions on the compute stream (no host staging).

`fft_norm` is one value applied to both transforms. Invariants of the split operators (bin 0 = zero range; zero-Doppler centered by default; `v_r > 0` receding) hold.

The locked contract says the fused `rd_maps` are **bit-identical** to the split chain. The shared stages make that true for delay cancellers and the no-clutter path to within a few ULP of the split Graph (host capture / re-upload between split nodes). `mean_subtraction` is **not** bit-exact: MatX `mean` reduction order can differ between a materialized clutter write and the fused pre-FFT expression ([`OPEN_ITEMS.md`](OPEN_ITEMS.md) §1). Tests gate **8 ULP** (`kParityUlps`), not bitwise equality.

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

## Parameters

Union of the split operators; semantics match those READMEs.

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `range_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Rebuilds range premultiplier / taps, warning once | Fast-time window |
| `doppler_window` | `WindowSpec` | `none` (rectangular) | `[live]` Exact via wire schema | Rebuilds Doppler taps, warning once | Slow-time window |
| `apply_dechirp` | bool | `false` | `[live]` Exact | Rebuilds range premultiplier | Software dechirp; needs waveform + complex IF |
| `clutter_filter` | `ClutterFilterMethod` {`none`, `mean_subtraction`, `single_delay`, `double_delay`} | `none` | `[live]` Exact | None (expression choice). Delay cancellers shrink `num_doppler` | Inline clutter; `none` is pass-through |
| `waveform` | `FMCWWaveformDescriptor` | none (`is_valid() == false`) | `[live]` Exact via wire schema | Dechirp reference + both axis fills | Required for dechirp and/or `range_bins` / `velocity_bins` |
| `fft_norm` | `FftNorm` {`backward`, `ortho`, `forward`} | `backward` | `[live]` Exact | Free | Shared FFT normalization |
| `center_zero_doppler` | bool | `true` | `[live]` Exact | Free | Center DC on the Doppler axis |
| `propagation_speed` | float64 | `299792458.0` m/s | Params-only | Free (device fill of `range_bins` only) | Beat → range |
| `nfft_range` | int64 | `0` (= `samples_per_chirp`) | `[live]` Exact | Unseen length builds a range FFT plan, warning once | Range FFT length |
| `nfft_doppler` | int64 | `0` (= post-clutter chirp count) | `[live]` Exact | Unseen length builds a Doppler FFT plan, warning once | Doppler FFT length |

`nfft_doppler` is compared to the post-clutter chirp count, not the raw `num_chirps`. YAML / `set_params` field names match the table.

`propagation_speed` has no `[live]` port and is not declared in `setup()` / `contract()`. `resolve_params` copies `pinned.propagation_speed` into the range stage config. It is used only by `fill_range_bins`.

## SDK implementation notes

- **Construction.** `FMCWRangeDopplerProcessorOp(const MaxTensorExtents&)` requires `max_chirps` and `max_range_bins`. Optional `max_rx` / `max_channels` admit a leading `N_rx`; `max_tx` plus an RX/channel field admits `(M_tx, N_rx)`. Those fields size `rd_maps` (`max_rank` 2–4, `max_byte_span` of a complex cube `{max_tx, max_rx, max_chirps, max_range_bins}`), `range_bins` (real `{max_range_bins}`), and `velocity_bins` (real `{max_chirps}`). The Doppler extent is **`max_chirps`**, not `max_doppler_bins` (that field is unused). An `nfft_range` above `max_range_bins` or `nfft_doppler` above `max_chirps` therefore exceeds the pool span. A tick that exceeds the admitted leading rank or `max_tx * max_rx` is rejected (messages above). `max_rx`/`max_tx` must be positive when set. Both stage states are allocated in the ctor.
- **Triggering.** `contract()` is `OnEach{adc_cube}` plus an `Exact{}` + `SafeDefault{}` entry for every override, wired or not. An unwired override compiles to a `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT` warning and reads as `kNotConnected` on every tick, so wiring the port is the application's only obligation. This requires an SDK carrying CONTRACT-020 ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Wiring a port `contract()` does not declare is still `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1). Both optional axis outputs stay omitted so unwired sinks do not gate scheduling.
- **Ports and queues.** `adc_cube` and the override inputs use `queue_depth(2)` (operator default; a graph may raise it per edge with `ConnectionOptions`). Data Tensor ports are `MemoryKind::kCudaDevice`. Fat / bins outs carry ctor-derived `TensorBounds` via `pooled_tensor_output`. `setup()` registers data-plane I/O first (`adc_cube`, then `rd_maps`, then latched `range_bins` / `velocity_bins`), then every `[live]` Exact input in Params field order. Schema-backed lives (`range_window`, `doppler_window`, `waveform`) sit with the POD lives, not next to the cube; they are `Input<wire::T>` and do not take `expects_tensor`.
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. After emit, `adc_cube` is moved into `retained_input_` so the range FFT can finish reading it after `compute()` returns. That is a different buffer from the published loans: emit never publishes the input (HSDK 5 rejects a foreign Tensor on a pooled output). Neither FFT path nor `fill_range_bins` / `fill_velocity_bins` synchronizes. Internal `profiles` is stream-ordered MatX memory freed at `compute()` scope exit after the enqueued kernels. The only remaining stream sync is cold-path coefficient upload on a cache miss (range [`OPEN_ITEMS.md`](../fmcw_range_processor_op/OPEN_ITEMS.md) §1, Doppler [`OPEN_ITEMS.md`](../fmcw_doppler_processor_op/OPEN_ITEMS.md) §1).
- **Emission.** `rd_maps` is prepared and published first (`prepare_device_tensor` → `emit_prepared_tensor`). Holding that required emit until an optional bins loan is ready would drop maps if that allocate failed. Each axis then runs in its own gated `if (waveform present && *_out_.active())`. `plan.stage.range.waveform` and `plan.stage.doppler.waveform` are the same optional; the gates differ only by latch. An unwired bins emit is absorbed and latches that port off; declining one axis does not stop the other. The required emit is not duplicated across branches.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.FMCWRangeDopplerProcessorParams`, version 1. Example apps on this branch wire the **split** chain (`set_params` on the three operators); the fused node is constructed the same way in the perf / parity harnesses ([EA HSDK5 ONLY] Graph has no `from_config()`).
- **Precision.** Explicit instantiations `FMCWRangeDopplerProcessorOp<float>` and `FMCWRangeDopplerProcessorOp<double>` in `fmcw_range_doppler_processor_op.cu`.
- **Cache warn-once.** Both stages' `ConfigCache`s decide when to warn themselves (first configuration silent, later unseen ones warn once then cache); `compute()` passes no warn flag.
- **[EA HSDK5 ONLY]** Exact `Input<>` ports are declared unconditionally (see Triggering). The two windows and `waveform` use generated schema wire types and convert back to the logical descriptor before override resolution; scalar / enum ports use their logical type directly. Graph tests cover the window and waveform descriptors; the mid-stream `clutter_filter` case remains skipped ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1).

## Consumer lifetime notes

Every output is a fresh Tensor-pool buffer for that tick. A consumer may retain `rd_maps`, `range_bins`, or `velocity_bins` without copying. `retained_input_` is the input-side counterpart and is not visible to consumers. The internal range-profile cube is not a port.

## Performance

**Cold.** First tick (and first unseen window / dechirp-reference / FFT-length key) fills the ctor-allocated `RangeStageState` and `DopplerStageState` caches. Each tick also allocates the internal `profiles` cube (async device memory). Axis fills are MatX generators on the compute stream.

**Steady.** Cached coefficients reused; each tick loans `rd_maps` (and the two axes when a waveform is set), writes `profiles`, and enqueues clutter-into-Doppler. Versus the split chain this saves one full-cube write+read, one kernel launch, and one async alloc/free per tick **plus** the graph-level cost of the extra split nodes ([`OPEN_ITEMS.md`](OPEN_ITEMS.md) §1). The FFTs still dominate. Do **not** decide keep/kill from a sum of the three per-operator harnesses — use `perf_fmcw_split_vs_fused`.

Harness: `performance/cpp/perf_fmcw_range_doppler_processor.cpp` times this node alone (Python mirror exists but is not built on this branch). Axes: size ladder `{xs, sm, md, lg, xl}`; clutter `{none, mean_subtraction}` (default) / `--full` adds `single_delay` across the ladder and channel batches `{4, 16}` on `md`/`lg`. Default matrix: none `{xs, md, lg, xl}` plus mean-sub/`md` and none/`md`/4 channels. Notes field records `live_override=deferred_ea` because mid-session override timing remains deferred under §1 / §8, not because the port is absent. End-to-end comparison: `performance/cpp/perf_fmcw_split_vs_fused`. No measured numbers are recorded in-tree.

```bash
ctest --test-dir build -R perf_fmcw_range_doppler_processor_smoke
ctest --test-dir build -R perf_fmcw_split_vs_fused_smoke
```

## Tests and verification

C++ Graph suite: `tests/cpp/fmcw/test_fmcw_range_doppler_processor.cpp`. The split chain is three Graph runs with host capture / re-upload between stages. Ground truth for the peak test is `radar_common/sim/` plus `bin_model.hpp`. Python `tests/python/fmcw/test_fmcw_range_doppler_processor.py` is not registered ([EA HSDK5 ONLY] `HOLOSCAN_RADAR_BUILD_PYTHON` forced off; [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6).

| Check | Proves |
| --- | --- |
| `UlpParityAllClutterFiltersFloat` / `Double` | `rd_maps` within 8 ULP of the split chain for `{none, mean_subtraction, single_delay, double_delay}` × {no window, Hamming Doppler}; axes **exactly** equal |
| `UlpParityNfftDoppler` / `UlpParityRealIf` | Same ULP gate for `nfft_doppler = 2×` and real-IF input |
| `ParityAllClutterFiltersFloat` / `Double` | Looser relative gate (`1e-5` / `1e-12`) plus exact axes |
| `ParityNfftOverrides` / `ParityWindows` / `ParityRealIfInput` / `ParityApplyDechirp` | Matching parameters produce matching shapes / axes; `nfft_range`/`nfft_doppler` = 2× doubles both axes |
| `InlineClutterReducesDopplerLength` | Delay cancellers shrink `num_doppler` consistently with standalone clutter |
| `PeakMapsToTruthRangeAndVelocity` | Injected target maps via both axes within one bin |
| `OptionalOutputsOnlyWithWaveform` | Axis ports silent without a waveform; `rd_maps`-only graph still runs |
| `ApplyDechirpWithoutWaveformErrors` / `ApplyDechirpWithRealInputErrors` / `NfftRangeTooSmallErrors` / `NfftDopplerTooSmallErrors` | Contract-illegal combinations fail closed (`nfft_doppler` also checked against post-clutter length) |
| `FloatAndDoubleAgree` | F32 vs F64 fused magnitudes agree under mean-subtraction |
| `LiveWindowAndWaveformOverride` | Schema-backed range / Doppler window plus waveform Exact tick matches the fused reference; the following miss tick reverts to Params |
| `LiveClutterOverrideChangesShape` | `GTEST_SKIP` — Graph live harness ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1) |

`TESTING.md` parity + inline-clutter + axis-match clauses are covered. There is no dedicated `N_rx` / `M_tx×N_rx` fused test (the split range/Doppler suites cover those axes), and no unwired-axis latch test.

```bash
ctest --test-dir build -R FMCWRangeDopplerProcessorOpTest --output-on-failure
```

## Usage

Example apps on this branch wire the split chain. A fused node is constructed the same way:

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

Python bindings (`FMCWRangeDopplerProcessorOpF32` / `F64`) are deferred on this branch ([EA HSDK5 ONLY] [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6); the 4.x Fragment trampoline in `python/` is not built.

## Contract conformance

The locked [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) surface is implemented. `range_window`, `doppler_window`, and `waveform` `[live]` ports use the L18 generated `holoscan::radar::wire` descriptor-Tensor encoding ([`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md)); that is the contract physical form, not a gap. Tests require 8 ULP against the split chain (`kParityUlps`), not bitwise equality — [`OPEN_ITEMS.md`](OPEN_ITEMS.md) §1 records the `mean_subtraction` summation-order caveat.

Audit scope (static; no build or test run): `docs/planning/OPERATOR_API.md` § `FMCWRangeDopplerProcessorOp`, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md` §1/§6/§8/§9, `STATUS.md`, `operators/common/README.md`, `operators/fmcw/README.md`, `OPEN_ITEMS.md`, `fmcw_range_doppler_processor_op.hpp` / `.cu`, the three stage headers it includes, `metadata.json`, `python/_fmcw_range_doppler_processor_op_bindings.cpp`, `tests/cpp/fmcw/test_fmcw_range_doppler_processor.cpp`, `tests/python/fmcw/test_fmcw_range_doppler_processor.py`, `performance/cpp/perf_fmcw_range_doppler_processor.cpp`, `performance/cpp/perf_fmcw_split_vs_fused.cpp`.
