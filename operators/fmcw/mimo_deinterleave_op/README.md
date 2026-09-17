# `MIMODeinterleaveOp`

TDM front-of-chain marshaling: gather an interleaved slow-time ADC cube into an explicit TX axis. It sits at the start of the MIMO-TDM chain so range, clutter, and Doppler see `(M_tx, N_rx, L, samples)` rather than a packed `M·L` chirp axis. The operator is a pure strided gather per `TXSchedule` — no FFT, window, or phase math — and it rejects non-uniform / staggered schedules because downstream TDM compensation models a fixed `PRI/M` inter-TX offset.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `adc_cube` | complex64 / complex128, or float32 / float64 | `(N_rx, M·L, samples_per_chirp)` | Slow-time interleaves the M TX per `tx_schedule`. Complex IQ IF or real IF; dtype preserved. Rank 3; no opaque leading batch |
| out | `adc_cube_out` | same as input | `(M_tx, N_rx, L, samples_per_chirp)` | Always emitted. `L = (M·L) / num_tx`. **[EA HSDK5 ONLY]** physical name `adc_cube_out` (logical channel `adc_cube`) |

```text
adc_cube → MIMODeinterleaveOp → FMCWRangeProcessorOp → [ClutterFilterOp] → FMCWDopplerProcessorOp → TDMDenseCompensationOp → VirtualArrayAssemblyOp
```

Contract: [`OPERATOR_API.md` § `MIMODeinterleaveOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

The operator is data marshaling only. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies dtype (complex or real IF), and binds the static `tx_schedule` to this input (`build_deinterleave_stage_plan` → `DeinterleaveStagePlan` with layout, `num_tx`, `chirps_per_tx`, and `out_shape`). There is no `resolve_params`: no `[live]` ports, and the empty-schedule sentinel is rejected in the plan builder rather than mapped to an optional. Required `adc_cube_out` is the only published tensor (`emit_device_tensor`). There is no latched optional axis.

Trailing rank is `(N_rx, M·L, samples)`, not an FMCW `(chirps, bins)` cube, so the plan builder uses `parse_adc_cube_layout` instead of `collapse_named_tx_rx_batch`. Rank must be exactly 3.

**Schedule expansion.** `TXSchedule` describes how the interleaved `M·L` slow-time axis splits. `num_tx` is M. Empty `tx_order` means round-robin `[0, 1, …, M−1]`; a nonempty `tx_order` is a permutation of that set and is the firing cycle. Optional `chirp_tx_index` (length `M·L`) overrides `tx_order` when present. The cycle repeats `L` times. Expansion lives in `TXSchedule::expand_chirp_tx_index` (`operators/common/radar_common/descriptors/tx_schedule.cpp`). Pass the same `TXSchedule` to `TDMDenseCompensationOp / DDMDenseDemultiplexOp` so TDM phase uses firing slot, not TX id (`firing_slot_by_tx()`).

**Uniformity (v1 scope).** First-wave TDM assumes uniform per-TX spacing. After expansion, the pattern must be a length-M cycle that repeats identically across all `L` repetitions, and each TX must appear exactly `chirps_per_tx = slow_time / num_tx` times. Custom permutations of `[0, …, M−1]` are allowed when they still repeat uniformly. Non-uniform / staggered schedules are rejected with `"TXSchedule: non-uniform / non-repeating schedule is rejected for first-wave TDM"` (or `"TXSchedule: each TX must appear exactly chirps_per_tx times"`). Staggered TDM is deferred (STATUS.md L12 / Roadmap).

**Gather map.** For each interleaved chirp index `k`, with TX `tx = pattern[k]` and `l` the number of times that TX has already fired:

```text
gather[tx * chirps_per_tx + l] = k
```

So output slot `(tx, l)` gathers from interleaved slow-time index `k`. Built on the host by `TXSchedule::build_gather_map`, copied into a ctor-sized device table, and indexed in the kernel as `gather_map[tx * chirps_per_tx + l]`. A new `(slow_time, schedule)` pair rebuilds the host map and async-H2Ds it; a previously seen pair is a cache hit. Uniformity is checked in `expand_chirp_tx_index` on that rebuild, not on every tick.

**Row copy.** One CUDA block per output row `(tx, rx, l)` cooperatively copies the contiguous `samples` vector. Input layout is `(N_rx, slow_time, samples)`; output layout is `(M_tx, N_rx, L, samples)` with channel-major `(tx, rx)` ordering. The innermost `samples` axis is copied 1:1. Element size selects a 4-, 8-, or 16-byte word copy (`uint32_t` / `uint64_t` / `uint4`) so complex64, complex128, float32, and float64 all take the same path.

**PRI is not applied here.** This operator does not read `FMCWWaveformDescriptor`. The caller must set `waveform.chirp_repetition_interval` on downstream operators to the **per-TX effective PRI** (`M ×` the physical inter-chirp spacing). See `TXSchedule` notes in `OPERATOR_API.md` and `applications/fmcw/mimo_tdm` for the simulator split between physical vs effective PRI.

TDM only. There is no DDM path — DDM has no slow-time interleave (`WORKFLOWS.md`).

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

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `tx_schedule` | `TXSchedule` | none (required) | static (Params-only) | Changing it (not a `[live]` port) rebuilds the gather map: host expand + async H2D into the ctor-sized table. Same `(slow_time, schedule)` is a cache hit | TDM firing pattern. Carries `num_tx`; optional `tx_order` (length M, default round-robin) or `chirp_tx_index` (length `M·L`, overrides `tx_order`). `L` is inferred as `(M·L)/num_tx`, not a separate parameter |

`TXSchedule` YAML / `set_params` fields: `num_tx` (int, required), `tx_order` (int vector, optional), `chirp_tx_index` (int vector, optional). Decode runs `TXSchedule::validate()`. Empty `tx_order` is round-robin; nonempty `chirp_tx_index` wins over `tx_order`.

There are no `[live]` parameters. `tx_schedule` has no Exact port and is not declared as `spec.input`. A live schedule on this producer would be a new contract, and would have to land with a live consumer (`TDMDenseCompensationOp / DDMDenseDemultiplexOp`) at the same time — do not add one here to "complete" the Params surface. HSDK 5 Graph live-override harness notes in [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1 / §8 / §9 do not apply to this operator.

## SDK implementation notes

- **Construction.** `MIMODeinterleaveOp(const MaxTensorExtents& extents)` requires `max_tx`, `max_rx`, `max_chirps`, and `max_samples_per_chirp` (each a positive value, else `"MIMODeinterleaveOp: MaxTensorExtents.<field> must be a positive value (required for Tensor pool sizing)"`). `max_chirps` sizes **per-TX** `L`, not the interleaved `M·L` input axis. Other `MaxTensorExtents` fields are unused. `max_rank` is pinned to 4; `max_byte_span` is `byte_span_complex_cube<T>({max_tx, max_rx, max_chirps, max_samples})` — the real-IF path is therefore over-sized to the complex element width.
- **Gather-map workspace.** The gather-index table is allocated in the ctor as a rank-1 int64 device Tensor of length `max_tx * max_chirps` (`make_device_tensor`) — operator-owned workspace, the same class as compensation's DDM-NL tiles. Tensor pools are per output port only; there is no scratch-loan API for this table.
- **Triggering.** `contract()` is `OnEach{adc_cube}` only. There are no optional or `[live]` ports to omit.
- **Ports and queues.** `adc_cube` is `expects_tensor` with `MemoryKind::kCudaDevice`, `queue_depth(2)`. `adc_cube_out` is `produces_tensor` with the same memory kind and ctor-derived `TensorBounds` via `pooled_tensor_output`. `setup()` registers data-plane I/O only (`adc_cube`, then `adc_cube_out`). Dtype is not pinned on the port (enforced in `compute()`).
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. After emit, `adc_cube` is moved into `retained_input_` so the gather kernel can finish reading it after `compute()` returns. That is a different buffer from the published loan: emit never publishes the input (HSDK 5 rejects a foreign Tensor on a pooled output). The hot path does not synchronize. Gather-map H2D uses `cudaMemcpyAsync` on that stream; a cache miss does not `cudaMalloc`.
- **Emission.** Single required output: `emit_device_tensor` (loan → `run_deinterleave_stage` → commit → emit). There is no latched optional, so the prepare-then-`emit_prepared_tensor` split used by range/Doppler is not needed here.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.MIMODeinterleaveParams`, version 1. YAML / `from_config` / `set_params` field: `tx_schedule`. Apps on this branch `set_params` a loaded `MIMODeinterleaveParams` ([EA HSDK5 ONLY] Graph has no `from_config()`).
- **Precision.** Explicit instantiations `MIMODeinterleaveOp<float>` and `MIMODeinterleaveOp<double>`.
- **[EA HSDK5 ONLY]** Physical output name is `adc_cube_out` because inputs and outputs share one name namespace ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §2). The logical channel in the contract remains `adc_cube`. Ported and allow-listed; Graph C++ tests and Graph perf cold/steady have landed. Graph live tests are N/A (no `[live]` params).

## Consumer lifetime notes

`adc_cube_out` is a fresh Tensor-pool buffer for that tick. A consumer may retain it without copying. `retained_input_` is the input-side counterpart and is not visible to consumers.

## Performance

**Cold.** First tick (or a new `(slow_time, TXSchedule)` pair): host `expand_chirp_tx_index` / `build_gather_map`, async H2D of the gather indices into the ctor-sized table, then the row-copy kernel. **Steady.** Same schedule and slow-time: kernel only, gather map reused. No FFT plans, no window coefficients. Per tick the operator loans one output tensor from the Graph pool.

There are no `[live]` overrides. Size-ladder axes in `performance/cpp/perf_mimo_deinterleave.cpp`: `(M_tx, N_rx, L, samples)` at rungs `xs` (2, 4, 8, 64), `sm` (4, 8, 16, 128), `md` (4, 16, 32, 256), `lg` (8, 32, 64, 512), `xl` (8, 64, 128, 1024), variant `rr` (round-robin). No leading-batch rung (opaque batch is out of scope). No measured numbers are recorded in-tree; run the harness:

```bash
ctest --test-dir build -R perf_mimo_deinterleave_smoke
```

## Tests and verification

C++ Graph suite: `tests/cpp/fmcw/test_mimo_deinterleave.cpp`. Ground truth is a host-built interleaved cube whose samples are TX-tagged markers (`marker_value(batch, tx, l, rx, sample)`), not `radar_common/sim/`. After gather, each output slot `(tx, rx, l, s)` must match the marker for that TX's `l`-th firing.

| Check | What it proves |
| --- | --- |
| `RoundRobinComplexFloat` / `CustomTxOrderComplexFloat` / `ExplicitChirpTxIndexUniformPattern` | Interleaved cube splits to `(M_tx, N_rx, L, samples)` for default round-robin, custom `tx_order = {1, 0}`, and uniform `chirp_tx_index` |
| `RealIfPreserved` / `ComplexDoublePreserved` / `RealDoublePreserved` | Dtype preserved (real IF and both precisions) |
| `RejectsNonUniformSchedule` | `{0,1,0,1,0,0}` `chirp_tx_index` fails the tick (TESTING.md: non-uniform rejected) |
| `RejectsIndivisibleSlowTime` | `slow_time` not a multiple of `num_tx` fails |
| `RejectsOpaqueLeadingBatch` | Extra leading dim is not a named axis; `parse_adc_cube_layout` fail-closes (Part B) |
| `FloatDoubleAgreement` | F32 vs F64 markers agree to `1e-4` |
| `FiringSlotByTxIdentityAndPermutation` | `TXSchedule::firing_slot_by_tx()` identity / empty / permuted cycle (compensation consumer) |

Python: `tests/python/fmcw/test_mimo_deinterleave.py` ports the marker tests. **[EA HSDK5 ONLY]** Python operator Graph is still phase-4 deferred ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §6); that file still wires output port `"adc_cube"` and still asserts leading-batch preservation (`test_LeadingBatchAxesPreservedAndIndependent`), which disagrees with the C++ Graph suite.

Owed by TESTING.md and covered in C++: round-robin + custom order split, non-uniform rejection, dtype preservation, opaque leading-batch rejection. Not covered: a Graph test that quotes the exact non-uniform error string.

```bash
ctest --test-dir build -R MIMODeinterleaveOp --output-on-failure
```

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

`extents` must fill `max_tx`, `max_rx`, `max_chirps` (per-TX `L`), and `max_samples_per_chirp`. Device Tensor producers also need `CompileOptions.deployment.bind_tensor_output_device` on `"adc_cube_out"` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §3). Set downstream `waveform.chirp_repetition_interval` to the per-TX effective PRI (`M ×` physical spacing). Copy the same `tx_schedule` onto `TDMDenseCompensationOp / DDMDenseDemultiplexOp` under TDM.

Python bindings exist (`MIMODeinterleaveOpF32` / `MIMODeinterleaveOpF64`). **[EA HSDK5 ONLY]** the public Python Graph / Holoviz path is still deferred (HSDK5_PORT §6); the C++ Graph construction above is the ported surface.

```python
from holoscan.radar import MIMODeinterleaveOpF32, TXSchedule

schedule = TXSchedule()
schedule.num_tx = 2
op = MIMODeinterleaveOpF32(
    fragment, max_tx=8, max_rx=16, max_chirps=512, max_samples=4096,
    tx_schedule=schedule, name="mimo_deinterleave")
```

## Contract conformance

Conforms on ports (including the documented physical rename), dtypes, `tx_schedule` as a required static parameter, uniformity rejection, dtype preservation, TDM-only marshaling, and rank-3 input (fail-closed extra leading axes). Exceptions:

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| Output name `adc_cube` | renamed | **[EA HSDK5 ONLY]** Physical port `adc_cube_out`. Already recorded in the contract and [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §2 |

Audit scope (static, no build or test run): `docs/planning/OPERATOR_API.md`, `WORKFLOWS.md`, `TESTING.md`, `HSDK5_PORT.md`, `STATUS.md`, `MAX_TENSOR_EXTENTS.md`; `operators/common/README.md`, `operators/fmcw/README.md`; operator `mimo_deinterleave_op.hpp` / `.cu`, `mimo_deinterleave_stage.hpp` / `.cu`, `metadata.json`, Python bindings; `radar_common/descriptors/tx_schedule.hpp` / `.cpp`; `tests/cpp/fmcw/test_mimo_deinterleave.cpp`, `tests/python/fmcw/test_mimo_deinterleave.py`; `performance/cpp/perf_mimo_deinterleave.cpp`, `performance/python/perf_mimo_deinterleave.py`.
