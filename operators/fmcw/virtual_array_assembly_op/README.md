# `VirtualArrayAssemblyOp`

MIMO marshaling: flatten a separated `(tx, rx)` RD block into one virtual channel axis and emit the matching virtual `ArrayGeometry`. It sits after `MIMOCoherentCompensationOp` (and after optional post-demux clutter on DDM) and feeds CFAR plus the angle path. Virtual positions are `p_virtual = p_tx + p_rx` in the array-local frame; coincident elements are handled by `overlap_policy` so `num_virtual ≤ M·N`.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(M_tx, N_rx, num_doppler, num_range_bins)` | Separated `(tx, rx)` block. TX/RX positions are parameters, not ports. Rank 4; no opaque leading batch |
| out | `rd_maps_out` | complex64 / complex128 | `(num_virtual, num_doppler, num_range_bins)` | Channel flatten `c = tx·N_rx + rx`, then overlap policy. Doppler/range unchanged. **[EA HSDK5 ONLY]** physical name `rd_maps_out` |
| out | `array_geometry` | float64 | `(num_virtual + 1, 3)` | Packed `ArrayGeometry` (`array_geometry_tensor.hpp`). Rows `[0, num_virtual)`: `p_virtual` (m). Last row: `[wavelength, geometry_type_code, 0]` with `geometry_type` unset (NaN). **[EA HSDK5 ONLY]** `Output<ArrayGeometry>` is not a PublishedPayload |
| live | `wavelength` | float64 scalar | scalar | Exact override; wire to enable, unwired = miss |

```text
… → MIMOCoherentCompensationOp → [ClutterFilterOp (DDM only)] → VirtualArrayAssemblyOp → CFARDetectorOp / SnapshotGatherOp / ⟨angle op⟩
```

Contract: [`OPERATOR_API.md` § `VirtualArrayAssemblyOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

Assembly is a channel gather (or average), not a spatial beamformer.
`compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, `require_complex<T>` in the same try as device + contiguous, merges this tick's Exact `wavelength` onto pinned Params (`resolve_params` → `VirtualAssemblyStageConfig`; the non-positive sentinel is rejected here), and binds that config to this input (`build_assembly_stage_plan` → `VirtualAssemblyStagePlan` with layout, gather plan, and both output shapes). Trailing rank is `(M_tx, N_rx, D, R)`, not an FMCW `(chirps, bins)` cube, so the plan builder uses `parse_rd_maps_layout` instead of `collapse_named_tx_rx_batch`. Rank must be exactly 4. Both outputs are required (`prepare` both loans, then `emit_prepared_tensor`). There is no latched optional axis.

**Virtual position**, meters, array-local frame:

```text
p_virtual[tx, rx] = p_tx[tx] + p_rx[rx]
```

**Channel index**, TX outer, RX inner — same order as the input `(M_tx, N_rx, …)` layout:

```text
c = tx · N_rx + rx
```

**Coincidence**: two positions match when all three components agree within `kPositionTol = 1e-9` m (`virtual_array_assembly_stage.cu`). Not scaled by wavelength. Near-duplicates closer than this are merged; farther apart are distinct virtual elements. A wavelength-relative / λ/2-grid overlap rule is out of scope.

**Overlap policies**, scan order `(tx = 0..M−1, rx = 0..N−1)`:

| Policy | `num_virtual` | RD samples | Geometry |
| --- | --- | --- | --- |
| `dedup` (default) | unique coincident groups | copy the **first** `(tx, rx)` slot in scan order (`rep_channel`) | one row per group |
| `average` | unique coincident groups | complex mean `sum(members) / count` | one row per group |
| `keep_all` | `M_tx · N_rx` | identity gather `src_channel[c] = c` | all `M·N` positions, duplicates retained |

`dedup` and `keep_all` share `gather_single_source_kernel` (one source channel per virtual index). `average` uses `gather_average_kernel` over CSR-style `member_offsets` / `member_indices`. Host grouping lives in `build_virtual_plan`; a new `(tx_positions, rx_positions, overlap_policy, M_tx, N_rx)` pair rebuilds the host tables and async-H2Ds them into ctor-sized device buffers. A previously seen key is a cache hit. Wavelength is not part of that key.

**Emitted geometry.** `make_array_geometry` copies post-policy `virtual_positions` and the effective `wavelength`. `geometry_type` is left unset. The operator then `pack_array_geometry`s into a float64 Tensor (HSDK5_PORT §10):

```text
rows [0, num_virtual):  (x, y, z) meters
row  num_virtual:       [wavelength, geometry_type_code, 0]
  geometry_type_code: 0=ULA, 1=URA, 2=UCA, 3=CONFORMAL; NaN if unset
```

Consumers unpack with `unpack_array_geometry`. Virtual positions do not depend on wavelength; only the metadata row does. `num_doppler` and `num_range` pass through unchanged. This is the one sanctioned descriptor *output* port (design invariant 2). Angle operators' `[live]` `array_geometry` Inputs carry the generated `WireArrayGeometry` schema type, a different representation from this packed Tensor, so it cannot currently plug into `BeamscanEstimatorOp.array_geometry` as a typed override — the MIMO app mirrors positions into both Params blocks and drains this port to a sink (HSDK5_PORT §9 / §10).

## Shape and dtype rules

- `num_virtual` = `M_tx · N_rx` for `keep_all`, else the number of unique `p_tx+p_rx` groups (`≤ M·N`)
- `num_doppler` = input `shape[2]`
- `num_range_bins` = input `shape[3]`
- `array_geometry` rows = `num_virtual + 1`, cols = 3

**Named axes.** The contract input is exactly `(M_tx, N_rx, num_doppler, num_range_bins)` (rank 4). There is no named leading batch. Constructor pool bounds pin `rd_maps_out` `max_rank = 3` (the flatten drops two named axes and inserts `num_virtual`). Extra leading axes fail closed in `parse_rd_maps_layout`.

`tx_positions.size()` must equal `num_tx * 3`; `rx_positions.size()` must equal `num_rx * 3` (row-major). Lengths are checked against the tick's `M_tx` / `N_rx`.

**Dtype.** Template `T` ∈ {`float`, `double`}; `rd_maps` complex only (`require_complex<T>`). `rd_maps_out` preserves that dtype. `array_geometry` is always float64.

**Rejected combinations** (quoted):

- Input rank ≠ 4: `"rd_maps: expected rank 4 (M_tx, N_rx, num_doppler, num_range), got <n>"`
- Non-positive trailing axis: `"rd_maps: trailing axes must be positive"`
- Real IF: `"rd_maps: real-valued input float32 is not supported here; provide complex64"` (double analogously)
- Dtype mismatch / non-CUDA-accessible / non-contiguous: same helpers as other ops, port `'rd_maps'`, who `VirtualArrayAssemblyOp`
- Input axes above ctor extents: `"VirtualArrayAssemblyOp: input (M_tx, N_rx, D, R) exceeds MaxTensorExtents"`
- `num_virtual` above the virtual-channel bound: `"VirtualArrayAssemblyOp: num_virtual exceeds MaxTensorExtents"`
- `tx_positions` length: `"VirtualArrayAssemblyOp: tx_positions length must be num_tx * 3"`
- `rx_positions` length: `"VirtualArrayAssemblyOp: rx_positions length must be num_rx * 3"`
- Non-positive / non-finite wavelength: `"VirtualArrayAssemblyOp: wavelength must be a positive finite scalar (build-time or [live] override)"`
- Missing compute stream: `"VirtualArrayAssemblyOp: no CUDA compute stream"`
- Packed-geometry size mismatch / H2D: `"VirtualArrayAssemblyOp: array_geometry size mismatch"` / `"VirtualArrayAssemblyOp: array_geometry H2D failed"`

**Mid-stream.** `overlap_policy`, `tx_positions`, and `rx_positions` are static. `wavelength` is `[live]` Exact: Params seeds the effective value, and an override whose `capture_time` matches the cube's replaces it for that tick only; a tick with no match uses the Params wavelength. Changing wavelength does **not** rebuild the gather plan (positions are wavelength-independent); the next tick's packed metadata row updates. A new `M_tx`/`N_rx` (or Params positions/policy) rebuilds the host plan and re-uploads index tables. Downstream re-infers `num_virtual`. Extents past ctor pool bounds fail in the plan builder.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `tx_positions` | float vector, row-major `(num_tx, 3)` | none (required) | static (Params-only) | Rebuilds the virtual plan + H2D into the ctor-sized index tables | TX element positions (m). Length must match input `M_tx` |
| `rx_positions` | float vector, row-major `(num_rx, 3)` | none (required) | static (Params-only) | Same as `tx_positions` | RX element positions (m). Length must match `N_rx` |
| `wavelength` | float | none (required) | `[live]` Exact | Free for the gather; packed `array_geometry` metadata row is rewritten each tick anyway | Operating wavelength (m) for emitted geometry (`c / carrier_frequency`) |
| `overlap_policy` | enum `{dedup, average, keep_all}` | `dedup` | static (Params-only) | Rebuilds the plan (gather vs average kernel) | Coincident `p_virtual` handling; sets `num_virtual` |

YAML field names: `tx_positions`, `rx_positions`, `wavelength`, `overlap_policy` (`dedup` / `average` / `keep_all`). The MIMO app typically fills positions and wavelength from `tx_array` / `rx_array` / `waveform` rather than duplicating them under this block.

`wavelength` is live because it describes the chirp that produced this cube (carrier → λ) and a producer can stamp it onto the same tick. Positions and `overlap_policy` describe the aperture, not a per-cube encoding, and stay Params-only — do not invent live ports to complete the surface. Schema-valued `override_ms` for this POD Exact is not deferred the same way as §9 descriptors; Graph perf already inlines a mid-session wavelength flip. Graph unit mid-stream coverage remains skipped (HSDK5_PORT §1).

## SDK implementation notes

- **Construction.** Requires `max_tx`, `max_rx`, `max_doppler_bins`, and `max_range_bins`. Virtual-channel bound is `max_virtual_channels` when set, else `max_tx * max_rx`. `rd_maps_out`: `max_rank = 3`, `max_byte_span = byte_span_complex_cube<T>({max_virtual, max_doppler, max_range})`. `array_geometry`: `max_rank = 2`, `max_byte_span = (max_virtual + 1) * 3 * sizeof(double)`.
- **Index-table workspace.** `src_channel` (`max_virtual` int64s), `member_offsets` (`max_virtual + 1`), and `member_indices` (`max_tx * max_rx`) are allocated in the ctor as rank-1 int64 device Tensors (`make_device_tensor`) — operator-owned workspace, the same class as deinterleave's gather map. Tensor pools are per output port only; there is no scratch-loan API for these tables. A cache miss does not `cudaMalloc`.
- **Triggering.** `contract()` is `OnEach{rd_maps}` plus an `Exact{}` + `SafeDefault{}` entry for `wavelength`, wired or not. An unwired override compiles to a `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT` warning and reads as `kNotConnected` on every tick, so wiring the port is the application's only obligation. This requires an SDK carrying CONTRACT-020 ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §11). Wiring a port `contract()` does not declare is still `RUNTIME_OPERATOR_REPLAY_MISMATCH` (§1). Both outputs are always emitted — unwired `array_geometry` still needs a sink or it is an isolated vertex (the MIMO app uses `NullSinkOp`).
- **Ports and queues.** `rd_maps` and `wavelength`: `queue_depth(2)`. Both outputs `produces_tensor` device with the bounds above. `setup()` registers data-plane I/O first (`rd_maps`, then `rd_maps_out`, then `array_geometry`), then the `[live]` Exact input in Params field order. `wavelength` is a scalar Input, not a Tensor, so it does not take `expects_tensor`.
- **Streams and lifetime.** `context.cuda_stream()`. Index-table and packed-geometry H2D are `cudaMemcpyAsync` on that stream. After emit, `rd_maps` is moved into `retained_input_` so the gather kernel can finish reading it after `compute()` returns. That is a different buffer from the published loans: emit never publishes the input (HSDK 5 rejects a foreign Tensor on a pooled output). The hot path does not synchronize.
- **Emission.** Prepare **both** loans (`rd_maps_out` then `array_geometry`) before either emit, then `emit_prepared_tensor` in that order, so a late fill cannot leave one port published. Both are required; the latched-optional split used by range/Doppler does not apply.
- **Parameter authoring.** Schema `com.nvidia.holoscan.radar.VirtualArrayAssemblyParams`, version 1. Apps on this branch `set_params` a loaded `VirtualArrayAssemblyParams` ([EA HSDK5 ONLY] Graph has no `from_config()`).
- **Precision.** `VirtualArrayAssemblyOp<float>` and `<double>` (`rd_maps` path). Packed geometry is always float64.
- **[EA HSDK5 ONLY]** `rd_maps_out` unique-name rename (HSDK5_PORT §2). `array_geometry` packed Tensor (HSDK5_PORT §10). A POD Exact `Input<double> wavelength` is declared unconditionally (see Triggering); Graph mid-stream override *unit* tests remain skipped (HSDK5_PORT §1). Graph perf `wavelength` `override_ms` is restored. Ported; Graph C++ tests and Graph perf cold/steady landed; lg/xl vs 4.3 gap accepted as Graph publish-path artifact (HSDK5_PORT §3).

## Consumer lifetime notes

Both outputs are fresh pool loans each tick. The consumer may retain them. Packed `array_geometry` is a device Tensor; host-side `unpack_array_geometry` needs a host copy of the `(num_virtual+1, 3)` buffer. Module-wide ping-pong reuse of emitted outputs is [`operators/OPEN_ITEMS.md`](../../OPEN_ITEMS.md) §1 — not implemented here.

## Performance

**Cold.** First `(tx_positions, rx_positions, overlap_policy, M_tx, N_rx)`: host plan (`p_tx+p_rx`, grouping) + async H2D of index tables into the ctor-sized buffers, then gather/average kernel + geometry pack H2D. **Steady.** Same plan key: kernel + per-tick geometry pack H2D (wavelength may change without a plan rebuild). No FFT. A cache miss does not `cudaMalloc`.

Harness `performance/cpp/perf_virtual_array_assembly.cpp`: size rungs `xs`–`xl` over `(M_tx, N_rx, num_doppler, num_range)`, overlap-policy variants. `array_geometry` is drained by a null sink. No leading-batch rung (opaque batch is out of scope). No in-tree measured medians. Remaining lg/xl gap vs Holoscan 4.3 is recorded as a Graph publish-path artifact (HSDK5_PORT §3), not an operator-kernel number. Every compute run inlines a mid-session `[live] wavelength` flip and reports `override_ms` / `override_delta_ms`.

```bash
ctest --test-dir build -R perf_virtual_array_assembly_smoke
```

## Tests and verification

C++ Graph suite: `tests/cpp/fmcw/test_virtual_array_assembly.cpp`. Ground truth is analytic `p_tx + p_rx` on host (TESTING.md), plus marker RD cubes that prove which source channel survived `dedup` / `average` / `keep_all`. Emitted geometry is unpacked from the packed Tensor.

| Check | What it proves |
| --- | --- |
| `EmittedGeometryMatchesAnalyticLayout` | `p_virtual = p_tx + p_rx`; packed Tensor unpacks |
| `KeepAllPreservesChannels` | `num_virtual = M·N` |
| `DedupCollapsesCoincidentPositions` / `DedupKeepsFirstRepresentativeChannel` / `DedupIsDefaultPolicy` | unique count; first scan-order representative; default |
| `AverageCombinesCoincidentChannels` / `AverageCombinesAllCollidingChannels` | complex mean |
| `WavelengthConfigurableFromParams` | Params wavelength in metadata row |
| `RejectsMismatchedTxPositions` / `RejectsMismatchedRxPositions` / `RejectsInvalidWavelength` / `RejectsRankTooSmall` | error paths |
| `RejectsOpaqueLeadingBatch` | extra leading dim is not a named axis; `parse_rd_maps_layout` fail-closes (Part B) |
| `FloatDoubleAgreement` / `ComplexDoublePreservesValues` | precision |
| `LiveWavelengthOverrideMidStream` | **[EA HSDK5 ONLY]** `GTEST_SKIP` (HSDK5_PORT §1) |
| `RejectsPageableHostInput` | **[EA HSDK5 ONLY]** skipped (device-bound loans) |

Python `tests/python/fmcw/test_virtual_array_assembly.py` ports the layout/policy tests. **[EA HSDK5 ONLY]** Python Graph is phase-4 deferred; `test_LeadingBatchAxesPreservedAndIndependent` still asserts leading-batch preservation, which disagrees with the C++ Graph suite.

Owed by TESTING.md and covered in C++: virtual layout, all three `overlap_policy` values, `num_virtual ≤ M·N`, opaque leading-batch rejection.

```bash
ctest --test-dir build -R VirtualArrayAssemblyOp --output-on-failure
```

## Usage

```yaml
virtual_array_assembly:
  overlap_policy: keep_all
  # tx_positions / rx_positions / wavelength are required; the MIMO app
  # copies them from tx_array / rx_array / waveform into Params.
```

```cpp
auto vaa = graph.op<holoscan::radar::VirtualArrayAssemblyOp<float>>(
    "virtual_array_assembly", extents);
vaa.set_params(cfg.vaa_params);
graph.add_flow(compensation->rd_maps_out, vaa->rd_maps, q);
graph.add_flow(vaa->rd_maps_out, cfar->rd_maps, q);
graph.add_flow(vaa->rd_maps_out, gather->rd_maps, q);
graph.add_flow(vaa->array_geometry, geom_sink->input, q);  // drain if unused
```

`extents` must fill `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins`, and optionally `max_virtual_channels` (else the virtual bound is `max_tx * max_rx`). Bind `"rd_maps_out"` and `"array_geometry"` when this op is a Graph device producer (HSDK5_PORT §3). Optional: `graph.add_flow(src->wavelength, vaa->wavelength, q)` for a live scalar override; leaving that port unwired keeps the Params wavelength.

Python bindings exist (`VirtualArrayAssemblyOpF32` / `F64`). **[EA HSDK5 ONLY]** public Python Graph is deferred (HSDK5_PORT §6).

```python
from holoscan.radar import VirtualArrayAssemblyOpF32, OverlapPolicy

op = VirtualArrayAssemblyOpF32(
    fragment, tx_positions=tx, rx_positions=rx, wavelength=0.0039,
    overlap_policy=OverlapPolicy.KEEP_ALL, name="virtual_array_assembly")
```

## Contract conformance

Conforms on flatten order, `p_tx+p_rx`, the three overlap policies, complex `rd_maps`, required positions/wavelength, the Exact `wavelength` override Input, rank-4 input (fail-closed extra leading axes), and the documented packed-geometry Tensor. Exceptions:

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| Output name `rd_maps` | renamed | **[EA HSDK5 ONLY]** Physical `rd_maps_out`. Already in the contract / HSDK5_PORT §2 |
| `array_geometry` as packed float64 Tensor | EA-deferred | Matches the HSDK 5 contract text. Typed `Output<ArrayGeometry>` is still the cleanup target (HSDK5_PORT §10) |

Audit scope (static, no build or test run): planning docs plus this operator's `*.hpp` / `*.cu`, bindings, `array_geometry_tensor.hpp`, `tests/cpp/fmcw/test_virtual_array_assembly.cpp`, `tests/python/fmcw/test_virtual_array_assembly.py`, `performance/cpp/perf_virtual_array_assembly.cpp`.
