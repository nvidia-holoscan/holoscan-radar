# `ClutterFilterOp`

Slow-time static-clutter / zero-Doppler suppression on range profiles, between range compression and the Doppler FFT. It nulls DC-across-chirps (`mean_subtraction`, no chirp loss) or applies a 1st/2nd-order MTI delay canceller (`single_delay` / `double_delay`, dropping 1–2 chirps). Switching method mid-stream is allowed: `num_chirps_out` follows the method and the downstream Doppler stage re-infers it. **[EA HSDK5 ONLY]** the physical output port is `range_profiles_out` (HSDK 5 unique in/out names).

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `range_profiles` | complex64 / complex128 | `(num_chirps, num_range_bins)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Slow-time axis is second-to-last. Real IF is rejected |
| out | `range_profiles_out` | complex64 / complex128 | same leading + `(num_chirps_out, num_range_bins)` | `num_chirps_out = num_chirps` / `num_chirps−1` / `num_chirps−2` by method. **[EA HSDK5 ONLY]** renamed from logical `range_profiles` |
| live | `method` | `ClutterFilterMethod` | scalar | Exact override; wire to enable, unwired = miss |

```text
FMCWRangeProcessorOp → ClutterFilterOp → FMCWDopplerProcessorOp
```

On MIMO-DDM the same ports sit **after** `MIMOCoherentCompensationOp` demux (the input is still named `range_profiles` but carries the demuxed block). The fused `FMCWRangeDopplerProcessorOp` inlines this stage and does not emit intermediate profiles.

Contract: [`OPERATOR_API.md` § `ClutterFilterOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

All math lives in `clutter_stage.hpp` (`with_clutter_expr`). `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype and named-axis shape (`collapse_named_tx_rx_batch`), merges this tick's Exact match onto pinned Params (`resolve_params` → `ClutterStageConfig`; standalone `none` is rejected here), and binds that config to this input (`build_clutter_stage_plan` → `ClutterStagePlan` with `chirps_out`). Required `range_profiles_out` is the only published tensor (`emit_device_tensor`). There is no latched optional axis. The fused operator composes the same expression into the Doppler pre-FFT pass (no extra cube write), which is why split-vs-fused parity is at expression granularity.

`method` selects one of three kernels (plus `none`, which this standalone operator rejects):

- **`mean_subtraction`.** Subtract the per-(batch, range-bin) slow-time mean, broadcast back over chirps. Shape preserved: `out = in − clone(mean(in, {1}), {keep, num_chirps, keep})`. Nulls zero-Doppler; a moving tone is kept.
- **`single_delay`.** First-order MTI: `out[k] = in[k+1] − in[k]` for `k ∈ [0, num_chirps−1)`. `num_chirps_out = num_chirps − 1`. Needs `num_chirps ≥ 2`.
- **`double_delay`.** Second-order MTI: `out[k] = in[k] − 2·in[k+1] + in[k+2]` for `k ∈ [0, num_chirps−2)`. `num_chirps_out = num_chirps − 2`. Needs `num_chirps ≥ 3`.
- **`none`.** Pass-through in `with_clutter_expr` (fused inline stage only). Standalone `resolve_params` returns `ClutterFilterOp: method 'none' is not supported by the standalone operator (it exists only for the fused operator's inline clutter stage); use one of {mean_subtraction, single_delay, double_delay}`.

Stateless: no FFT plan, no coefficient cache. Work is enqueued on the compute stream with no synchronization.

## Shape and dtype rules

- **`range_profiles_out` rank** = input rank. Trailing axes become `(num_chirps_out, num_range_bins)` with `num_range_bins` copied and `num_chirps_out = clutter_num_chirps_out(method, num_chirps)`.
- **Named leading axes.** Same rule as the range operator: none, `N_rx`, or `(M_tx, N_rx)`, capped by `MaxTensorExtents`. Extra leading axes: `ClutterFilterOp: unexpected leading axes (opaque batch is not supported); got leading rank <got>, max named leading <max>`. Leading product above `max_tx * max_rx`: `ClutterFilterOp: leading product <n> exceeds max_tx*max_rx from MaxTensorExtents`.
- **Extents ceiling.** Unlike the range operator, a tick whose `num_chirps` or `num_range_bins` exceeds the ctor fields fails with `ClutterFilterOp: input chirps/range_bins exceed MaxTensorExtents`. That check lives in `build_clutter_stage_plan`, on top of `collapse_named_tx_rx_batch`.
- **Complex only.** `classify_input_dtype<T>(…, allow_real=false)`. Real input: `range_profiles: real-valued input <dtype> is not supported here; provide <complex>`. Mismatched precision uses the same `received dtype … does not match this operator's float|double-precision instantiation` form (complex expected only).
- **Delay-canceller length.** `ClutterFilterOp: method 'single_delay' needs num_chirps >= 2, got <n>` / `… 'double_delay' needs num_chirps >= 3, got <n>`.
- **Layout.** `range_profiles` must be CUDA-accessible and compact row-major: `ClutterFilterOp: tensor received on input port 'range_profiles' is not CUDA-accessible (device, managed, or pinned-host memory required)` / `… must be contiguous (compact row-major)`.
- **No CUDA stream.** `ClutterFilterOp: no CUDA compute stream`.
- **`[live]` `method`.** Switching to/from a delay canceller changes `num_chirps_out` mid-stream; downstream re-infers.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `method` | `ClutterFilterMethod` {`mean_subtraction`, `single_delay`, `double_delay`} | `mean_subtraction` | `[live]` Exact | None (stateless). Shape-changing when a delay canceller is selected | Slow-time clutter suppression. `none` is rejected here |

YAML spelling is the canonical `to_string` form (`mean_subtraction`, `single_delay`, `double_delay`).

There is no Params-only field (unlike range `propagation_speed`).

## SDK implementation notes

- **Construction.** `ClutterFilterOp(const MaxTensorExtents&)` **requires** `max_chirps` and `max_range_bins` (contract `Ctor:` line). Optional `max_rx` / `max_channels` / `max_tx` admit named leading axes via `require_tx_rx_pool_extents`. Those fields size `range_profiles_out` (`max_rank` 2–4, complex cube `{max_tx, max_rx, max_chirps, max_range_bins}`). A tick above the admitted leading rank, leading product, or chirp/bin ceilings is rejected. There is no stage-state allocation (stateless).
- **Triggering.** `contract()` is `OnEach{range_profiles}` plus an `Exact{}` + `SafeDefault{}` entry for `method`, wired or not. An unwired override compiles to a `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT` warning and reads as `kNotConnected` on every tick, so wiring the port is the application's only obligation. This requires an SDK carrying CONTRACT-020 ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Wiring a port `contract()` does not declare is still `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1).
- **Ports and queues.** `range_profiles` and `method` use `queue_depth(2)` (operator default; a graph may raise it per edge with `ConnectionOptions`). Data Tensor ports are `MemoryKind::kCudaDevice`. The fat out carries ctor-derived `TensorBounds` via `pooled_tensor_output`. `setup()` registers data-plane I/O first (`range_profiles`, then `range_profiles_out`), then the `[live]` Exact input in Params field order. `method` is `Input<ClutterFilterMethod>` and does not take `expects_tensor`. Dtype is not pinned on the port (enforced in `compute()`).
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. After emit, `range_profiles` is moved into `retained_input_` so the stage can finish reading it after `compute()` returns. That is a different buffer from the published loan: emit never publishes the input (HSDK 5 rejects a foreign Tensor on a pooled output). The MatX expression does not synchronize.
- **Emission.** Single required output: `emit_device_tensor` (loan → `run_clutter_stage` → commit → emit). There is no latched optional, so the prepare-then-`emit_prepared_tensor` split used by range/Doppler is not needed here.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.ClutterFilterParams`, version 1. Apps on this branch `set_params` a loaded `ClutterFilterParams` ([EA HSDK5 ONLY] Graph has no `from_config()`).
- **Precision.** Explicit instantiations `ClutterFilterOp<float>` and `ClutterFilterOp<double>`.
- **[EA HSDK5 ONLY]** Output port `range_profiles_out` (unique in/out names; [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §2). A POD Exact `Input<>` is declared unconditionally for `method` (see Triggering). Graph mid-stream `method` tests run since the enum-edge fix (§12); Graph perf reports `override_ms` for a mid-session method switch. This operator was the HSDK 5 Tensor I/O + live-override pilot.

## Consumer lifetime notes

`range_profiles_out` is a fresh Tensor-pool buffer for that tick. A consumer may retain it without copying. `retained_input_` is the input-side counterpart and is not visible to consumers.

## Performance

**Cold / steady.** No operator-owned cache: first tick and later ticks enqueue the same MatX expression. `mean_subtraction` materializes a small stream-ordered mean temp during the consuming expression’s PreRun; delay cancellers are slices plus an elementwise combo. Each tick loans `range_profiles_out`. A `[live]` `method` change has no setup cost (it can change `num_chirps_out`).

Harness: `performance/cpp/perf_clutter_filter.cpp`. Axes: size ladder `{xs, sm, md, lg, xl}` = `(16×128) … (256×2048)` chirps × range bins; method `{msub, sdel, ddel}`; optional channel batch. Default matrix: msub `{xs, md, lg, xl}` plus sdel/ddel on `md` and msub/`md`/4 channels. `--full` crosses methods × sizes and adds channel batches on `md`/`lg`. Every compute run inlines a mid-session `[live] method` flip and reports `override_ms` / `override_delta_ms`. No measured numbers are recorded in-tree.

```bash
ctest --test-dir build -R perf_clutter_filter_smoke
```

## Tests and verification

C++ Graph suite: `tests/cpp/fmcw/test_clutter_filter.cpp`. Ground truth is `radar_common/sim/` (static / moving point targets) plus analytic Doppler tones for the binomial MTI gain. Python `tests/python/fmcw/test_clutter_filter.py` is not registered ([EA HSDK5 ONLY] [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6).

| Check | Proves |
| --- | --- |
| `MeanSubtractionNullsStaticTarget` | Zero-Doppler sim cube → residual `< 1e-4` of input peak; shape preserved |
| `MeanSubtractionPreservesMovingTarget` | Energy after mean-subtraction matches a moving-only scene within 5% |
| `SingleDelayShapeAndResponse` / `DoubleDelayShapeAndResponse` | Chirp count −1 / −2; per-cell gain matches `\|1−e^{j2πf}\|` and its square (tol `1e-4`) |
| `DelayCancellersNotchDC` | DC tone residual `< 1e-5` for both delay methods |
| `NamedRxAxisPreservedAndIndependent` | `N_rx = 2` preserved; per-RX mean-sub matches the host mean; single-delay drops chirps on the chirp axis only |
| `RejectsOpaqueLeadingBatchWhenExtentsAreSiso` | Extra leading axis with SISO extents raises diagnostics |
| `RejectsMethodNone` / `RejectsRealInput` / `RejectsTooFewChirps` | The three contract-illegal combinations fail closed |
| `MethodConfigurableFromParams` | Params authoring of `method` |
| `FloatAndDoubleAgree` | F32 vs F64 within `1e-5` of the F64 peak for all three methods |
| `LiveMethodOverrideAppliesForOneTick` | A mid-stream `method` override changes the output chirp count for that tick only |
| `LiveMethodRevertsToParamsAfterMiss` | The tick after a present override reverts to the Params method |

`TESTING.md` leading-axis clause is covered for `N_rx`; there is no dedicated `M_tx×N_rx` case.

```bash
ctest --test-dir build -R ClutterFilterOpTest --output-on-failure
```

## Usage

```yaml
clutter_filter:
  method: mean_subtraction
```

```cpp
auto clutter = graph.op<ClutterFilterOp<float>>("clutter_filter", extents);
clutter.set_params(cfg.clutter_params);

graph.add_flow(range->range_profiles, clutter->range_profiles, q);
graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, q);
```

Python bindings (`ClutterFilterOpF32` / `F64`) are deferred on this branch ([EA HSDK5 ONLY] [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6); the 4.x Fragment trampoline in `python/` is not built.

## Contract conformance

The locked [`OPERATOR_API.md`](../../../docs/planning/OPERATOR_API.md) surface is implemented. The physical output port is `range_profiles_out` ([EA HSDK5 ONLY] unique in/out names; already locked in `OPERATOR_API.md` and [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §2). Graph mid-stream `[live] method` is covered; Graph perf reports `override_ms` for a mid-session method switch.

Audit scope (static; no build or test run): `docs/planning/OPERATOR_API.md` § `ClutterFilterOp`, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md` §1/§2/§6/§8/§11/§12, `STATUS.md`, `operators/common/README.md`, `operators/fmcw/README.md`, `clutter_filter_op.hpp` / `.cu`, `clutter_stage.hpp`, `metadata.json`, `python/_clutter_filter_op_bindings.cpp`, `tests/cpp/fmcw/test_clutter_filter.cpp`, `tests/python/fmcw/test_clutter_filter.py`, `performance/cpp/perf_clutter_filter.cpp`.
