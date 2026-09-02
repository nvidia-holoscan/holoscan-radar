# `MIMOCoherentCompensationOp`

The MIMO compute step: TDM per-Doppler-bin inter-TX phase correction, or DDM Doppler-band demux (linear-phase gather or non-linear conjugate correlation) into an explicit TX axis. It sits after Doppler (TDM) or after Doppler and before optional clutter (DDM), and emits a separated, phase-aligned `(tx, rx)` block for `VirtualArrayAssemblyOp`. `multiplexing_scheme` is static and required — instantiate separate operators to switch TDM vs DDM — and the Doppler axis must match `FMCWDopplerProcessorOp` with `center_zero_doppler = true`.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | TDM: `(M_tx, N_rx, num_doppler, num_range_bins)`. DDM: `(N_rx, num_doppler, num_range_bins)` | Rank/interpretation set by `multiplexing_scheme`. Complex only |
| out | `rd_maps_out` | complex64 / complex128 | `(M_tx, N_rx, num_doppler_out, num_range_bins)` | TDM and DDM non-linear correlate: `num_doppler_out = num_doppler`. DDM linear-phase gather: `num_doppler_out = num_doppler / num_tx − guard_bands`. **[EA HSDK5 ONLY]** physical name `rd_maps_out` (logical channel `rd_maps`) |
| live | `waveform` | `holoscan::radar::wire::FMCWWaveformT` | schema metadata + descriptor Tensor | Exact override; wire to enable, unwired = miss |

`waveform` remains a logical `FMCWWaveformDescriptor` Param. Its HSDK 5 physical port uses the generated `holoscan::radar::wire` descriptor-Tensor encoding because the SDK does not admit container-owning C++ types directly. The Tensor member is always present: rank-1 zero-length represents an absent optional array, while null is invalid. See [`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md).

```text
TDM:  … → FMCWDopplerProcessorOp → MIMOCoherentCompensationOp → VirtualArrayAssemblyOp
DDM:  … → FMCWDopplerProcessorOp → MIMOCoherentCompensationOp → [ClutterFilterOp] → VirtualArrayAssemblyOp
```

Contract: [`OPERATOR_API.md` § `MIMOCoherentCompensationOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

Math lives in `mimo_coherent_compensation_stage.hpp` / `.cu`. `compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, classifies complex dtype, merges this tick's Exact `waveform` onto pinned Params (`resolve_params` → `CompensationStageConfig`; empty waveform and `unspecified` scheme are rejected there), and binds that config to this input (`build_compensation_stage_plan` → `CompensationStagePlan` with `out_shape` and the TDM or DDM layout). Required `rd_maps_out` is the only published tensor (`emit_device_tensor`). There is no latched optional axis.

TDM trailing rank is `(M_tx, N_rx, D, R)`; DDM is `(N_rx, D, R)`. Neither is an FMCW `(chirps, bins)` cube, so the plan builder uses `parse_tdm_rd_maps_layout` / `parse_ddm_rd_maps_layout` instead of `collapse_named_tx_rx_batch`. TDM reads `M_tx` from the input axis; DDM reads `num_tx` from `ddm_code`. No `ArrayGeometry` — this is phase / demux, not spatial. Private `run_tdm_compute` / `run_ddm_compute` / `run_ddm_correlate` split the real kernels.

### TDM — per-Doppler-bin inter-TX phase correction

A moving target accrues Doppler phase between TX firings. `waveform.chirp_repetition_interval` is the **per-TX effective PRI** (`M ×` the on-wire inter-chirp spacing). Inter-TX time is `slot × (PRI / M)`, where `slot` is the firing position of that TX id in the TDM cycle (`tx_schedule`; empty = identity, `slot == TX id`):

```text
PRF = 1 / PRI
t_inter_chirp = PRI / num_tx
phase(tx, d) = −2π · f_d · slot[tx] · t_inter_chirp
table[tx * num_doppler + d] = cos(phase) + j·sin(phase)
```

`f_d` is the aliased Doppler frequency (Hz) of center-zero bin `d` (`build_tdm_correction_table` / `doppler_frequency_hz` in `mimo_coherent_compensation_stage.cu`):

```text
idx = (d − num_doppler/2 + num_doppler) % num_doppler
f = idx / num_doppler * PRF
if f >= PRF/2: f -= PRF
```

The kernel complex-multiplies each RD cell by `table[tx * num_doppler + d]` (unit magnitude). Output shape equals input. The TX that fires first in the cycle is identity (TX 0 when `tx_schedule` is empty). `num_tx` comes from the input `M_tx` axis. The model is inter-TX timing only — not per-chirp TX codes on the RD map. Uniform per-TX spacing is required; non-uniform schedules are rejected upstream (`MIMODeinterleaveOp` / `TXSchedule`). Pass the same `tx_schedule` as deinterleave so a uniform `tx_order` permutation uses firing slot, not TX id. A nonempty schedule's `num_tx` must match input `M_tx`.

### DDM — path selection

`ddm_classify_codes`: nonempty `doppler_offsets` → linear gather. Else `phase_codes` are tested for a constant per-chirp phase step (`num_chirps ≥ 2` per TX): `arg(code[k+1]·conj(code[k]))` must match the first pair within `1e-3` rad (wrapped) for every TX. Pass → reduce to offsets `arg(step)/2π` folded to `[0, 1)` and take the gather path. Fail → conjugate-correlation path. `phase_codes` and `doppler_offsets` are mutually exclusive (`DDMCodeDescriptor::validate`).

### DDM linear-phase — Doppler-bin gather

Standard linear-phase DDM only **shifts** TX bands. Demux is a pure gather (no multiply by `phase_codes` on the RD data). `doppler_offsets[m]` are fractions of PRF in `[0, 1)` (folded with `−= floor(o)`); per tick, fft-order start bin is `round(offset · num_doppler) mod D`. `num_tx` comes from `ddm_code`, not from an input axis. Equal-width bands:

- `num_doppler` must be divisible by `num_tx`; band width = `num_doppler / num_tx`
- `num_doppler_per_tx = band_width − guard_bands`; the **top `guard_bands` fft-order bins** of each band are discarded
- Output Doppler axis is center-zero over width `W = num_doppler_per_tx`; the demux table encodes offset shift + fftshift bookkeeping so every TX lane shares that reduced axis

`build_ddm_demux_table` (with `c = (D+1)/2`, `cW = (W+1)/2`):

```text
start = round(offsets[m] * D) mod D
for o in 0 .. W-1:
  p = (o + cW) % W          # output center-zero → reduced fft-order
  j = (start + p) % D       # → absolute input fft-order
  in_cz = (j - c) mod D     # → input center-zero index
  table[m * W + o] = in_cz
```

The kernel copies the range row at input Doppler `table[tx * W + out_d]` into output slot `(tx, rx, out_d)`. Waveform is required and validated (TDM contract parity) but **does not affect** this gather. A `[live]` waveform under DDM is a no-op.

### DDM non-linear — conjugate correlation

Codes that fail the 1e-3 rad test use FFT circular correlation. Output keeps the **full** Doppler axis. `guard_bands` must be 0; `phase_codes.size() / num_tx` must equal input `num_doppler`. Weights (`build_ddm_corr_weights`):

```text
W_m[p] = conj(c_m[(-p) mod D])
```

Stage (`run_ddm_correlate_stage`): pack (ifftshift to fft-order) → cuFFT forward → per-TX `Ŷ * W_m * 1/D` + cuFFT inverse → scatter (fftshift to center-zero). Equivalent, in fft-order:

```text
out[d] = (1/D) * sum_k Y[k] * conj(C_m[(k-d) mod D]),   C_m = DFT(c_m)
```

Range-tiled (~32 MiB persistent workspaces). Waveform is required but unused, same as the gather path.

### Shared conventions

Input Doppler must be center-zero (`FMCWDopplerProcessorOp` default). There is no `[live]` toggle here for another ordering. Unambiguous velocity is reduced by `M` (TDM per-TX PRI; DDM shared `[0, PRF)` split across bands). This operator does not unfold that aliasing; downstream sees the per-TX Doppler axis from `FMCWDopplerProcessorOp`.

## Shape and dtype rules

**TDM** (`multiplexing_scheme: tdm`):

- Input trailing axes `(M_tx, N_rx, num_doppler, num_range_bins)` (rank 4)
- Output same; `M_tx` from the input axis

**DDM linear** (`doppler_offsets` or linear `phase_codes`):

- Input trailing `(N_rx, num_doppler, num_range_bins)` (rank 3)
- `M_tx` = `ddm_code.num_tx`
- `num_doppler_out` = `num_doppler / num_tx − guard_bands`
- `N_rx`, `num_range_bins` copied

**DDM non-linear** (`phase_codes` failing the linear-phase test):

- Same input rank as DDM linear
- Output `(M_tx, N_rx, num_doppler, num_range_bins)` — full Doppler preserved

**Named axes.** Contract shapes are exact ranks (no opaque leading batch). Layout parsers still admit `rank >=` the trailing count and collapse extras as batch; constructor `max_rank = 4` makes a leading dim fail at emit. See Contract conformance.

**Dtype.** Template `T` ∈ {`float`, `double`}; complex only (`allow_real=false`). Output dtype equals input.

**Rejected combinations** (quoted):

- Unspecified scheme: `"MIMOCoherentCompensationOp: multiplexing_scheme is required (tdm or ddm)"`
- Unknown scheme: `"MIMOCoherentCompensationOp: unknown multiplexing_scheme"`
- Invalid / missing waveform: `"MIMOCoherentCompensationOp: waveform must be a valid FMCWWaveformDescriptor (build-time or [live] override)"`
- TDM rank `< 4`: `"rd_maps: expected rank >= 4 (..., M_tx, N_rx, num_doppler, num_range)"`
- DDM rank `< 3`: `"rd_maps: expected rank >= 3 (..., N_rx, num_doppler, num_range) for DDM"`
- Real IF: `"rd_maps: real-valued input float32 is not supported here; provide complex64"` (double analogously)
- Dtype mismatch: `"rd_maps: received dtype … does not match this operator's float-precision instantiation (expected complex64)"`
- Non-CUDA-accessible / non-contiguous: same `require_cuda_accessible` / `require_contiguous` wording as other ops, port `'rd_maps'`, who `MIMOCoherentCompensationOp`
- DDM empty code: `"MIMOCoherentCompensationOp: DDM requires ddm_code with doppler_offsets or phase_codes"`
- Both code forms: `"DDMCodeDescriptor: provide phase_codes or doppler_offsets, not both"`
- `num_doppler` not divisible by `num_tx`: `"MIMOCoherentCompensationOp: DDM requires num_doppler (D) divisible by num_tx (M)"`
- Guard too large: `"MIMOCoherentCompensationOp: DDM guard_bands (G) leaves no Doppler bins in a band of width B"`
- Non-linear `guard_bands != 0`: `"MIMOCoherentCompensationOp: non-linear DDM phase_codes require guard_bands = 0 (got G)"`
- Non-linear chirp/Doppler mismatch: `"MIMOCoherentCompensationOp: non-linear DDM phase_codes imply num_chirps = C but input num_doppler = D; these must match for the correlate path"`
- `tx_schedule.num_tx` ≠ input `M_tx`: `"MIMOCoherentCompensationOp: tx_schedule.num_tx (N) must match input M_tx (M)"`
- Missing compute stream: `"MIMOCoherentCompensationOp: no CUDA compute stream"`

**Mid-stream.** `multiplexing_scheme`, `ddm_code`, and `tx_schedule` are static. `waveform` is `[live]` Exact: a `capture_time`-matched override replaces the Params value for that tick only; a tick without a match uses the Params waveform. A new effective descriptor rebuilds the TDM correction table and warns once (`"reconfigured to an unseen TDM correction table configuration; building it now and caching for reuse"`); seen `(num_tx, num_doppler, carrier, PRI, slots)` keys reuse the cache. DDM gather/correlate tables key on `ddm_code` + `num_doppler` and do not rebuild on waveform. Output Doppler length can change if input `num_doppler` or (DDM) `guard_bands` in a new `ddm_code` cache entry changes; downstream re-infers. Extents that exceed ctor pool bounds fail at loan allocation.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `multiplexing_scheme` | enum `{tdm, ddm}` | none (required) | static (Params-only) | Selects the kernel and input-rank parse. Default in `Params` is sentinel `unspecified`, rejected in `resolve_params` | TDM phase correction vs DDM demux |
| `waveform` | `FMCWWaveformDescriptor` | none (required) | `[live]` Exact via wire schema | TDM: new PRI/carrier builds a correction table and warns once; seen keys reuse `ConfigCache`. DDM: validated, unused | Per-TX PRI (`chirp_repetition_interval`) and carrier for TDM Doppler-phase math |
| `ddm_code` | `DDMCodeDescriptor` | none (required for DDM; unused for TDM) | static (Params-only) | New code / `num_doppler` builds a demux table or correlate weights (first cache build is silent) | `num_tx`, either `doppler_offsets` `(num_tx,)` in `[0, 1)` or `phase_codes` `(num_tx, num_chirps)` row-major, and `guard_bands` (default 0) |
| `tx_schedule` | `TXSchedule` | empty (identity round-robin) | static (Params-only) | New firing-slot map rebuilds the TDM correction table (same cache as waveform). Unused for DDM | Same descriptor as `MIMODeinterleaveOp`; `num_tx` must match input `M_tx` when set |

YAML field names: `multiplexing_scheme` (`tdm` / `ddm`), `waveform` (standard FMCW descriptor keys), `ddm_code.num_tx`, `ddm_code.doppler_offsets` **or** `ddm_code.phase_codes` (flat `[re, im, re, im, …]`), `ddm_code.guard_bands`, `tx_schedule` (`num_tx`, optional `tx_order` / `chirp_tx_index`). TDM callers may leave `ddm_code` default-empty. DDM callers may leave `tx_schedule` default-empty.

`multiplexing_scheme`, `ddm_code`, and `tx_schedule` have no Exact ports and are not declared as `spec.input`. `resolve_params` copies them from `pinned`.

## SDK implementation notes

- **Construction.** Requires `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins`. `max_rank = 4`; `max_byte_span = byte_span_complex_cube<T>({max_tx, max_rx, max_doppler, max_range})`. Size the Doppler extent for the **output** axis (full `num_doppler` for TDM / DDM-NL; `num_doppler_per_tx` is smaller, so the full-D bound still covers linear DDM). Table `ConfigCache`s are default-constructed. DDM-NL keeps a persistent cuFFT plan plus two ~32 MiB tiles (`yhat` / `work`) reused across ticks — operator-owned workspace, the same class as range/Doppler MatX FFT plan caches. Tensor pools are per output port only.
- **Triggering.** `contract()` is `OnEach{rd_maps}` plus an `Exact{}` + `SafeDefault{}` entry for `waveform`, wired or not. An unwired override compiles to a `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT` warning and reads as `kNotConnected` on every tick, so wiring the port is the application's only obligation. This requires an SDK carrying CONTRACT-020 ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Wiring a port `contract()` does not declare is still `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1).
- **Ports and queues.** `rd_maps` and `waveform` use `queue_depth(2)` (operator default; a graph may raise it per edge with `ConnectionOptions`). Data Tensor ports are `MemoryKind::kCudaDevice`. The fat out carries ctor-derived `TensorBounds` via `pooled_tensor_output`. `setup()` registers data-plane I/O first (`rd_maps`, then `rd_maps_out`), then the `[live]` Exact input in Params field order. `waveform` is `Input<wire::FMCWWaveformT>` and does not take `expects_tensor`. Dtype is not pinned on the port (enforced in `compute()`).
- **Streams and lifetime.** Work is enqueued on `context.cuda_stream()`. After emit, `rd_maps` is moved into `retained_input_` so TDM/DDM kernels and DDM-NL cuFFT can finish reading it after `compute()` returns. That is a different buffer from the published loan: emit never publishes the input (HSDK 5 rejects a foreign Tensor on a pooled output). No hot-path device sync. Table H2D is `cudaMemcpyAsync` on that stream. DDM-NL cuFFT plans are `cufftSetStream`'d; destructor calls `release_ddm_correlate_scratch(this)`.
- **Emission.** Single required output: `emit_device_tensor` (loan → stage → commit → emit). There is no latched optional, so the prepare-then-`emit_prepared_tensor` split used by range/Doppler is not needed here.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.MIMOCoherentCompensationParams`, version 1. Fields `multiplexing_scheme`, `waveform`, `ddm_code`, `tx_schedule`. Apps on this branch `set_params` a loaded `MIMOCoherentCompensationParams` ([EA HSDK5 ONLY] Graph has no `from_config()`).
- **Precision.** Explicit instantiations `MIMOCoherentCompensationOp<float>` and `<double>`.
- **Caches.** TDM `ConfigCache` keyed by `num_tx|num_doppler|carrier|PRI|slots`. DDM gather keyed by code + `num_doppler`. DDM-NL keyed by full `phase_codes` + `num_doppler`. The first build of each cache is silent; a later unseen key (e.g. a changed effective waveform, which changes the TDM key) warns once, then caches.
- **[EA HSDK5 ONLY]** `rd_maps_out` unique-name rename ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §2). The Exact `Input<>` for `waveform` is declared unconditionally (see Triggering) and uses a generated schema wire type, converted back to the logical descriptor in `resolve_params`. Ported; Graph C++ tests landed, including the live waveform cases; Graph perf cold/steady landed. Schema-valued `waveform` `override_ms` is deferred (§9), same as range/Doppler windows.

## Consumer lifetime notes

`rd_maps_out` is a fresh Tensor-pool buffer for that tick. A consumer may retain it without copying. `retained_input_` is the input-side counterpart and is not visible to consumers.

## Performance

**Cold (TDM).** First `(num_tx, num_doppler, carrier, PRI)`: host correction table + device tensor + H2D, then the multiply kernel. **Steady.** Cache hit, kernel only. **Cold (DDM linear).** Host demux table + H2D, then gather. **Cold (DDM-NL).** Correlate weights + H2D, cuFFT `PlanMany`, ~32 MiB tile workspace; then tiled pack / FFT / mul / IFFT / scatter. **Steady.** Plans, tiles, and weights reused.

A `[live]` or `set_params` waveform change rebuilds TDM tables for unseen keys. DDM output does not depend on waveform.

Harness `performance/cpp/perf_mimo_coherent_compensation.cpp`: size rungs `xs`–`xl` over `(M_tx, N_rx, num_doppler, num_range)`, variants `tdm` / `ddm` / `ddm_nl`. Cold/steady Graph is current. Schema-valued `waveform` `override_ms` is deferred (`HSDK5_PORT.md` §9), same as range/Doppler windows — do not wrap the production `Exact{}` port in `PerfLiveContractOp`. No in-tree measured medians. July 2026 nsys (HSDK 4, per-tick `make_device_tensor`) showed malloc/free filling the gap between kernel and tick time; those numbers are not a current gap — HSDK 5 `rd_maps_out` is a Graph pool loan and DDM-NL tiles persist. Re-profile before treating allocation as leftover work.

```bash
ctest --test-dir build -R perf_mimo_coherent_compensation_smoke
```

## Tests and verification

C++ Graph suite: `tests/cpp/fmcw/test_mimo_coherent_compensation.cpp`. TDM ground truth is a host RD cube with a known inter-TX motion phase; after compensation, TX lanes must align (TESTING.md key regression). DDM uses a coded multi-TX cube and checks each TX lands in its band (gather) or that correlation recovers peaks (non-linear). Stage helpers `build_tdm_correction_table` are also asserted (unit magnitude; TX 0 identity).

| Check | What it proves |
| --- | --- |
| `TdmRemovesInterTxPhaseAcrossDopplerBins` / `FourTxLanesAlignAfterCompensation` / `PermutedTxOrderAlignsAfterCompensation` | TDM correction removes inter-TX motion phase, including a non-identity `tx_order` |
| `TdmCorrectionTableIsUnitMagnitude` / `TdmCorrectionTableTxZeroIsIdentity` / `TdmCorrectionTableUsesFiringSlot` / `SingleTxIsIdentity` | Table convention (identity TX 0; first-firing TX when permuted) |
| `DdmDemuxRecoversEachTxIntoBand` / `DdmLinearPhaseCodesMatchOffsets` / `DdmNonUniformOffsetPermutation` | Linear demux + code→offset reduction |
| `DdmGuardBandsReduceWidth` / `DdmGuardBandDiscardsEdgeBin` | Imposed guard layout |
| `DdmNonLinearMultiTxSeparatesPeaks` (+ sibling NL tests) | Correlate path; `guard_bands=0` and chirp/Doppler match |
| dtype / F32–F64 agreement / determinism | Precision and stability |
| `RejectsUnspecifiedMultiplexingScheme` / `RejectsInvalidWaveform` / `RejectsEmptyDdmCode` / `DdmRejectsNonDivisibleDoppler` / `DdmRejectsGuardTooLarge` / `DdmRejectsBothOffsetForms` / `RejectsTxScheduleNumTxMismatch` | Error paths |
| `RejectsOpaqueLeadingBatch` / `DdmRejectsOpaqueLeadingBatch` | Extra leading dim fails Graph emit |
| `LiveWaveformOverrideAppliesPerTick` / `LiveWaveformUsesExactCaptureTime` / `RejectsInvalidLiveWaveform` | Schema-backed Exact override applies for its tick and reverts to Params on the next miss, ignores a skewed `capture_time`, and fails closed on an invalid descriptor — TX lanes stay aligned in each admitted case |
| pageable / pinned / managed / TensorMap / non-contiguous | **[EA HSDK5 ONLY]** skipped; Graph only delivers contiguous device loans |

Python `tests/python/fmcw/test_mimo_coherent_compensation.py` ports a subset, including a host-side formula check `test_TdmCorrectionTableLayoutMatchesFormula`. **[EA HSDK5 ONLY]** Python Graph is phase-4 deferred; that file still asserts leading-batch preservation.

Owed by TESTING.md: TDM unbiased-DOA vs skip-compensation is the *pipeline* regression (end-to-end MIMO); this unit suite asserts lane alignment after the multiply, which is the operator-local form of that claim. `num_tx` from TDM input axis vs DDM `ddm_code` is exercised by the two layout parsers.

```bash
ctest --test-dir build -R test_mimo_coherent_compensation --output-on-failure
```

## Usage

```yaml
mimo_coherent_compensation:
  multiplexing_scheme: tdm
  waveform:
    carrier_frequency: 77.0e9
    bandwidth: 150.0e6
    chirp_duration: 20.0e-6
    sample_rate: 12.8e6
    chirp_repetition_interval: 50.0e-6   # per-TX effective PRI
  # tx_schedule:                         # TDM: same as mimo_deinterleave; omit = identity
  #   num_tx: 2
  #   tx_order: [0, 1]
  # ddm_code:                            # required for ddm; omit on tdm
  #   num_tx: 2
  #   doppler_offsets: [0.0, 0.5]
  #   guard_bands: 0
```

```cpp
auto compensation = graph.op<holoscan::radar::MIMOCoherentCompensationOp<float>>(
    "mimo_coherent_compensation", extents);
compensation.set_params(cfg.compensation_params);
graph.add_flow(doppler->rd_maps, compensation->rd_maps, q);
graph.add_flow(compensation->rd_maps_out, vaa->rd_maps, q);
```

`extents` must fill `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins`. Bind `"rd_maps_out"` when this op is a Graph device producer ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §3).

Python bindings exist (`MIMOCoherentCompensationOpF32` / `F64`). **[EA HSDK5 ONLY]** public Python Graph is deferred (HSDK5_PORT §6).

```python
from holoscan.radar import MIMOCoherentCompensationOpF32, MultiplexingScheme

op = MIMOCoherentCompensationOpF32(
    fragment, multiplexing_scheme=MultiplexingScheme.TDM,
    waveform=waveform, name="mimo_coherent_compensation")
```

## Contract conformance

Conforms on TDM/DDM rank split, complex dtypes, `multiplexing_scheme` required/static, `ddm_code` required-for-DDM, `tx_schedule` TDM firing-slot index, linear vs correlate `num_doppler_out`, and the documented physical rename. Exceptions:

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| Output name `rd_maps` | renamed | **[EA HSDK5 ONLY]** Physical `rd_maps_out` (`mimo_coherent_compensation_op.hpp` 110–115). Already in the contract / HSDK5_PORT §2 |
| `waveform` **[live]** | conforms | Generated `holoscan::radar::wire` descriptor-Tensor payload ([`SCHEMA_PAYLOADS.md`](../../../docs/planning/SCHEMA_PAYLOADS.md)); logical Param remains `FMCWWaveformDescriptor` |
| Input shapes without leading `…` | deviates | Header documents `(..., M_tx, …)` / `(..., N_rx, …)` (`mimo_coherent_compensation_op.hpp` 37–59). Parsers admit `rank >=` 4 (TDM) / 3 (DDM) as collapsed batch (`mimo_coherent_compensation_stage.cu` 97–124, 206–231). Graph `max_rank = 4` (`mimo_coherent_compensation_op.cu` 177) rejects a leading dim at emit, not with a named extra-axis error |

Aliased ÷M velocity with no unfold is the documented v1 product, not an unimplemented mode.

Audit scope (static, no build or test run): planning docs listed for the deinterleave audit plus this operator's `*.hpp` / `.cu`, bindings, `tests/cpp/fmcw/test_mimo_coherent_compensation.cpp`, `tests/python/fmcw/test_mimo_coherent_compensation.py`, `performance/cpp/perf_mimo_coherent_compensation.cpp`, `radar_common/descriptors/ddm_code_descriptor.*`, `radar_common/descriptors/tx_schedule.*`.
