# `ClutterFilterOp`

## Purpose

Slow-time static-clutter / zero-Doppler suppression on range profiles, between range compression and the Doppler FFT. It can pass through unchanged (`none`), null DC across chirps (`mean_subtraction`, no chirp loss), or apply a 1st/2nd-order MTI delay canceller (`single_delay` / `double_delay`, dropping 1–2 chirps). `mean_subtraction` uses a uniform mean by default or a `doppler_window`-weighted mean when configured. Switching method mid-stream is allowed: `num_chirps_out` follows the method and the downstream Doppler stage re-infers it; the physical output port is `range_profiles_out` because Holoscan requires unique input/output names.

## Responsibilities

- Applies pass-through, weighted/unweighted mean subtraction, or first/second-order slow-time delay cancellation.
- Owns method-specific output length, window validation/caching, and tick-local method/window overrides.
- Does not perform Doppler-domain notching, FFT processing, or code-aware DDM clutter suppression.

## When to use

Use between range compression and the Doppler FFT for SISO, SIMO, or deinterleaved TDM data when stationary slow-time clutter should be suppressed. Omit it for pass-through, or use the fused range-Doppler operator when intermediate range profiles are unnecessary. Do not apply it to coded pre-demultiplex DDM chirps or to data already in the Doppler domain; use `DopplerNotchOp` after dense DDM demultiplexing instead.

```text
FMCWRangeProcessorOp → ClutterFilterOp → FMCWDopplerProcessorOp
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `range_profiles` | complex64 / complex128 | `(num_chirps, num_range_bins)`, optional named leading `N_rx` / `(M_tx, N_rx)` | Required trigger/aligned input | Slow-time axis is second-to-last. Real IF is rejected |
| out | `range_profiles_out` | complex64 / complex128 | same leading + `(num_chirps_out, num_range_bins)` | Required output | `num_chirps_out = num_chirps` / `num_chirps−1` / `num_chirps−2` by method. renamed from logical `range_profiles` |
| live | `method` | `ClutterFilterMethod` | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |
| live | `doppler_window` | `holoscan::radar::wire::WindowSpecT` | schema metadata + descriptor Tensor | Optional Exact one-tick override; Params fallback on miss | Exact override; one tick only, then a miss reverts to Params |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_chirps` and `max_range_bins` required; channel/TX bounds optional | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `method` | `ClutterFilterMethod` {`none`, `mean_subtraction`, `single_delay`, `double_delay`} | `mean_subtraction` | `[live]` Exact | Slow-time clutter suppression or materialized pass-through Cost: Shape-changing when a delay canceller is selected. |
| `doppler_window` | `WindowSpec` | `none` | `[live]` Exact via wire schema | Weighting for `mean_subtraction`; ignored by other methods Cost: Unseen windows rebuild and cache taps, warning once. |

Contract: [`OPERATOR_API.md` § `ClutterFilterOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

All math lives in `clutter_stage.hpp` (`with_clutter_expr`). `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype and named-axis shape (`collapse_named_tx_rx_batch`), merges this tick's Exact match onto pinned Params (`resolve_params` → `ClutterStageConfig`), and binds that config to this input (`build_clutter_stage_plan` → `ClutterStagePlan` with `chirps_out`). Required `range_profiles_out` is the only published tensor (`emit_device_tensor`). There is no latched optional axis. The fused operator composes the same expression into the Doppler pre-FFT pass (no extra cube write), which is why split-vs-fused parity is at expression granularity.

`method` selects one of three filters plus pass-through:

- **`mean_subtraction`.** Static clutter removal by subtracting the weighted
  slow-time mean and broadcasting it over chirps. This is not a pulse
  canceller. `doppler_window: none` uses a uniform mean; matching the
  projection taps to the downstream window guarantees a zero DC bin.
- **`single_delay`.** Two-pulse canceller (chirp-to-chirp here), first-order
  MTI: `out[k] = in[k+1] − in[k]`. It drops one chirp.
- **`double_delay`.** Three-pulse canceller (chirp-to-chirp here), second-order
  MTI: `out[k] = in[k] − 2·in[k+1] + in[k+2]`. It drops two chirps.
- **`none`.** Identity expression in `with_clutter_expr`. The standalone operator materializes it into its output pool; the fused operator keeps it inline without an intermediate cube.

`none`, `single_delay`, and `double_delay` ignore `doppler_window`; there are no new method enum values or YAML method spellings. Weighted projection taps use the shared `WindowSpec` generation and cache conventions. An unseen configuration rebuilds taps and warns once; prior configurations are reused.

## Shape and dtype rules

- **`range_profiles_out` rank** = input rank. Trailing axes become `(num_chirps_out, num_range_bins)` with `num_range_bins` copied and `num_chirps_out = clutter_num_chirps_out(method, num_chirps)`.
- **Named leading axes.** Same rule as the range operator: none, `N_rx`, or `(M_tx, N_rx)`, capped by `MaxTensorExtents`. Extra leading axes: `ClutterFilterOp: unexpected leading axes (opaque batch is not supported); got leading rank <got>, max named leading <max>`. Leading product above `max_tx * max_rx`: `ClutterFilterOp: leading product <n> exceeds max_tx*max_rx from MaxTensorExtents`.
- **Extents ceiling.** Unlike the range operator, a tick whose `num_chirps` or `num_range_bins` exceeds the ctor fields fails with `ClutterFilterOp: input chirps/range_bins exceed MaxTensorExtents`. That check lives in `build_clutter_stage_plan`, on top of `collapse_named_tx_rx_batch`.
- **Complex only.** `require_complex<T>`. Real input: `range_profiles: real-valued input <dtype> is not supported here; provide <complex>`. Mismatched precision uses the same `received dtype … does not match this operator's float|double-precision instantiation` form (complex expected only).
- **Delay-canceller length.** `ClutterFilterOp: method 'single_delay' needs num_chirps >= 2, got <n>` / `… 'double_delay' needs num_chirps >= 3, got <n>`.
- **Layout.** `range_profiles` must be CUDA-accessible and compact row-major: `ClutterFilterOp: tensor received on input port 'range_profiles' is not CUDA-accessible (device, managed, or pinned-host memory required)` / `… must be contiguous (compact row-major)`.
- **No CUDA stream.** `ClutterFilterOp: no CUDA compute stream`.
- **`[live]` `method`.** Switching to/from a delay canceller changes `num_chirps_out` mid-stream; downstream re-infers.
- **`[live]` `doppler_window`.** The generated-wire `WindowSpec` joins the data trigger with `Exact{}` + `SafeDefault{}`. A match applies to one tick; a miss uses Params and never latches.

## Performance

**Cold / steady.** The weighted kernel caches generated projection taps. Its first configured window and later unseen overrides incur generation/upload; seen windows reuse cached taps. `doppler_window: none` retains the current uniform reduction. Delay cancellers remain slices plus an elementwise combo and ignore the window. Each tick loans `range_profiles_out`; a `[live]` `method` change can change `num_chirps_out`.

Harness: `performance/operators/fmcw/clutter_filter/adapter.cpp`. Axes: size ladder `{xs, sm, md, lg, xl}` = `(16×128) … (256×2048)` chirps × range bins; method `{msub, sdel, ddel}`; optional channel batch. Default matrix: msub `{xs, md, lg, xl}` plus sdel/ddel on `md` and msub/`md`/4 channels. `--full` crosses methods × sizes and adds channel batches on `md`/`lg`. Every compute run inlines a mid-session `[live] method` flip and reports `override_ms` / `override_delta_ms`. No measured numbers are recorded in-tree.

```bash
ctest --test-dir build -R perf_clutter_filter_smoke
```

## Usage

```yaml
clutter_filter:
  method: mean_subtraction
  doppler_window: hann
```

```cpp
auto clutter = graph.op<ClutterFilterOp<float>>("clutter_filter", extents);
clutter.set_params(cfg.clutter_params);

graph.add_flow(range->range_profiles, clutter->range_profiles, q);
graph.add_flow(clutter->range_profiles_out, doppler->range_profiles, q);
```
