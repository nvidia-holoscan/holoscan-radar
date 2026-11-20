# `MIMODeinterleaveOp`

## Purpose

TDM front-of-chain marshaling: gather an interleaved slow-time ADC cube into an explicit TX axis. It sits at the start of the MIMO-TDM chain so range, clutter, and Doppler see `(M_tx, N_rx, L, samples)` rather than a packed `M·L` chirp axis. The operator is a pure strided gather per `TXSchedule` — no FFT, window, or phase math — and it rejects non-uniform / staggered schedules because downstream TDM compensation models a fixed `PRI/M` inter-TX offset.

## Responsibilities

- Validates a uniform TDM transmit schedule and gathers packed chirps into explicit transmitter, receiver, chirp, and sample axes.
- Owns schedule-index caching, output sizing, and tick-local schedule overrides.
- Does not compensate inter-transmitter phase, process DDM codes, or support staggered/non-uniform timing.

## When to use

Use at the front of a uniform TDM-MIMO chain to convert the packed slow-time chirp sequence into explicit TX and RX axes. Pass the same `TXSchedule` to downstream TDM compensation so correction uses firing slot rather than TX id. Do not use it for simultaneous-transmit DDM or non-uniform/staggered schedules.

```text
adc_cube → MIMODeinterleaveOp → FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp → TDMDenseCompensationOp → VirtualArrayAssemblyOp
```

Declarations and parameter defaults: [mimo_deinterleave_op.hpp](mimo_deinterleave_op.hpp). Behavioral requirements are documented below.

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `adc_cube` | complex64 / complex128, or float32 / float64 | `(N_rx, M·L, samples_per_chirp)` | Required trigger/aligned input | Slow-time interleaves the M TX per `tx_schedule`. Complex IQ IF or real IF; dtype preserved. Rank 3; no opaque leading batch |
| out | `adc_cube_out` | same as input | `(M_tx, N_rx, L, samples_per_chirp)` | Required output | Always emitted. `L = (M·L) / num_tx`. physical name `adc_cube_out` (logical channel `adc_cube`) |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_tx`, `max_rx`, `max_chirps`, and `max_samples_per_chirp` required | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `tx_schedule` | `TXSchedule` | none (required) | static (Params-only) | TDM firing pattern. Carries `num_tx`; optional `tx_order` (length M, default round-robin) or `chirp_tx_index` (length `M·L`, overrides `tx_order`). `L` is inferred as `(M·L)/num_tx`, not a separate parameter Cost: Changing it (not a `[live]` port) rebuilds the gather map: host expand + async H2D into the ctor-sized table. Same `(slow_time, schedule)` is a cache hit. |

## Algorithm

The operator is data marshaling only. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype (complex or real IF), and binds the static `tx_schedule` to this input (`build_deinterleave_stage_plan` → `DeinterleaveStagePlan` with layout, `num_tx`, `chirps_per_tx`, and `out_shape`). There is no `resolve_params`: no `[live]` ports, and the empty-schedule sentinel is rejected in the plan builder rather than mapped to an optional. Required `adc_cube_out` is the only published tensor (`emit_device_tensor`). There is no latched optional axis.

Trailing rank is `(N_rx, M·L, samples)`, not an FMCW `(chirps, bins)` cube, so the plan builder uses `parse_adc_cube_layout` instead of `collapse_named_tx_rx_batch`. Rank must be exactly 3.

**Schedule expansion.** `TXSchedule` describes how the interleaved `M·L` slow-time axis splits. `num_tx` is M. Empty `tx_order` means round-robin `[0, 1, …, M−1]`; a nonempty `tx_order` is a permutation of that set and is the firing cycle. Optional `chirp_tx_index` (length `M·L`) overrides `tx_order` when present. The cycle repeats `L` times. Expansion lives in `TXSchedule::expand_chirp_tx_index` (`operators/common/radar_common/descriptors/tx_schedule.cpp`). Pass the same `TXSchedule` to `TDMDenseCompensationOp / DDMDenseDemultiplexOp` so TDM phase uses firing slot, not TX id (`firing_slot_by_tx()`).

**Uniformity (v1 scope).** First-wave TDM assumes uniform per-TX spacing. After expansion, the pattern must be a length-M cycle that repeats identically across all `L` repetitions, and each TX must appear exactly `chirps_per_tx = slow_time / num_tx` times. Custom permutations of `[0, …, M−1]` are allowed when they still repeat uniformly. Non-uniform / staggered schedules are rejected with `"TXSchedule: non-uniform / non-repeating schedule is rejected for first-wave TDM"` (or `"TXSchedule: each TX must appear exactly chirps_per_tx times"`). Staggered TDM is not supported.

**Gather map.** For each interleaved chirp index `k`, with TX `tx = pattern[k]` and `l` the number of times that TX has already fired:

```text
gather[tx * chirps_per_tx + l] = k
```

So output slot `(tx, l)` gathers from interleaved slow-time index `k`. Built on the host by `TXSchedule::build_gather_map`, copied into a ctor-sized device table, and indexed in the kernel as `gather_map[tx * chirps_per_tx + l]`. A new `(slow_time, schedule)` pair rebuilds the host map and async-H2Ds it; a previously seen pair is a cache hit. Uniformity is checked in `expand_chirp_tx_index` on that rebuild, not on every tick.

**Row copy.** One CUDA block per output row `(tx, rx, l)` cooperatively copies the contiguous `samples` vector. Input layout is `(N_rx, slow_time, samples)`; output layout is `(M_tx, N_rx, L, samples)` with channel-major `(tx, rx)` ordering. The innermost `samples` axis is copied 1:1. Element size selects a 4-, 8-, or 16-byte word copy (`uint32_t` / `uint64_t` / `uint4`) so complex64, complex128, float32, and float64 all take the same path.

**PRI is not applied here.** This operator does not read `FMCWWaveformDescriptor`. The caller must set `waveform.chirp_repetition_interval` on downstream operators to the **per-TX effective PRI** (`M ×` the physical inter-chirp spacing). See the [MIMO TDM application](../../../applications/fmcw/mimo_tdm/README.md) for the simulator split between physical vs effective PRI.

TDM only. There is no DDM path — DDM has no slow-time interleave (see the [workflow guide](../../../docs/WORKFLOWS.md)).

## Shape and dtype rules

Output extents are inferred each tick from `adc_cube` and `tx_schedule.num_tx`:

- `M_tx` = `tx_schedule.num_tx`
- `N_rx` = input axis `shape[0]`
- `L` (`chirps_per_tx`) = `slow_time / num_tx`, with `slow_time = shape[1]`
- `samples_per_chirp` = input axis `shape[2]`

**Named axes.** The contract input is exactly `(N_rx, M·L, samples_per_chirp)` (rank 3). There is no named leading batch. Constructor pool bounds pin `max_rank = 4` on `adc_cube_out` (rank-3 in → rank-4 out). Extra leading axes fail closed in `parse_adc_cube_layout`.

**Dtype.** Precision is the template parameter `T` ∈ {`float`, `double`}. `classify_input_dtype<T>(…, allow_real=true)` accepts complex (`complex64`/`complex128`) or real IF (`float32`/`float64`) matching `T`. The output is emitted with the input's `DLDataType` — no conversion, no real-IF vs complex-IF bin-count change.

**Rejected combinations** (operator / descriptor messages, quoted):

- Input rank ≠ 3: `"adc_cube: expected rank 3 (N_rx, slow_time, samples), got <n>"`
- Non-positive trailing axis: `"adc_cube: trailing axes must be positive"`
- Dtype mismatch for this instantiation: `"adc_cube: received dtype … does not match this operator's float-precision instantiation (expected complex64 or float32)"` (double analogously)
- Non-CUDA-accessible memory: `"MIMODeinterleaveOp: tensor received on input port 'adc_cube' is not CUDA-accessible (device, managed, or pinned-host memory required)"`
- Non-contiguous input: `"MIMODeinterleaveOp: tensor received on input port 'adc_cube' must be contiguous (compact row-major)"`
- Empty / invalid `tx_schedule`: `"TXSchedule: num_tx must be positive"`
- `tx_order` length ≠ `num_tx`, out of range, or not a permutation: `"TXSchedule: tx_order length must equal num_tx"` / `"TXSchedule: tx_order entry out of range [0, num_tx)"` / `"TXSchedule: tx_order must be a permutation of [0, num_tx)"`
- `slow_time` not divisible by `num_tx`: `"MIMODeinterleaveOp: slow_time must be divisible by tx_schedule.num_tx"` (also `"TXSchedule: slow_time must be a positive multiple of num_tx"` inside gather-map build)
- Output axes above ctor extents: `"MIMODeinterleaveOp: output (M_tx, N_rx, L, samples) exceeds MaxTensorExtents"`
- `chirp_tx_index` length ≠ `slow_time`: `"TXSchedule: chirp_tx_index length must equal slow_time"`
- Non-uniform / non-repeating pattern: `"TXSchedule: non-uniform / non-repeating schedule is rejected for first-wave TDM"`
- Unequal per-TX counts: `"TXSchedule: each TX must appear exactly chirps_per_tx times"`
- TX index out of range: `"TXSchedule: chirp TX index out of range [0, num_tx)"`
- Missing compute stream: `"MIMODeinterleaveOp: no CUDA compute stream"`

**Mid-stream shape changes.** `tx_schedule` is static, so `M_tx` does not change via a `[live]` port. `N_rx`, `L`, and `samples_per_chirp` are re-inferred from the input each tick. A new `slow_time` (hence a new `L`) rebuilds the gather map; previously seen `(slow_time, schedule)` pairs reuse the device table. Downstream operators re-infer their trailing axes from `adc_cube_out`. A tick whose product exceeds the ctor-captured `MaxTensorExtents` is rejected in the plan builder.

## Performance

**Cold.** First tick (or a new `(slow_time, TXSchedule)` pair): host `expand_chirp_tx_index` / `build_gather_map`, async H2D of the gather indices into the ctor-sized table, then the row-copy kernel. **Steady.** Same schedule and slow-time: kernel only, gather map reused. No FFT plans, no window coefficients. Per tick the operator loans one output tensor from the Graph pool.

## Usage

```yaml
mimo_deinterleave:
  tx_schedule:
    num_tx: 2
    # tx_order: [0, 1]          # optional; default round-robin
    # chirp_tx_index: [0, 1, 0, 1, ...]  # optional; overrides tx_order
```

```cpp
auto deinterleave = graph.op<holoscan::radar::MIMODeinterleaveOp<float>>(
    "mimo_deinterleave", extents);
deinterleave.set_params(cfg.deinterleave_params);
graph.add_flow(source->adc_cube, deinterleave->adc_cube, q);
graph.add_flow(deinterleave->adc_cube_out, range->adc_cube, q);
```

`extents` must fill `max_tx`, `max_rx`, `max_chirps` (per-TX `L`), and `max_samples_per_chirp`. Device Tensor producers also need `CompileOptions.deployment.bind_tensor_output_device` on `"adc_cube_out"`. Set downstream `waveform.chirp_repetition_interval` to the per-TX effective PRI (`M ×` physical spacing). Copy the same `tx_schedule` onto `TDMDenseCompensationOp / DDMDenseDemultiplexOp` under TDM.
