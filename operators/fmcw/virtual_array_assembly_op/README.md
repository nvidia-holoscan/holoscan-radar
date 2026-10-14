# `VirtualArrayAssemblyOp`

## Purpose

MIMO marshaling: flatten a separated `(tx, rx)` RD block into one virtual channel axis and emit the matching virtual `ArrayGeometry`. It sits after `TDMDenseCompensationOp / DDMDenseDemultiplexOp` and feeds CFAR plus the angle path. Virtual positions are `p_virtual = p_tx + p_rx` in the array-local frame; coincident elements are handled by `overlap_policy` so `num_virtual ≤ M·N`.

## Responsibilities

- Builds the virtual-position and gather/average plan from transmitter and receiver geometry.
- Flattens dense TX/RX maps into channel maps and publishes geometry with identical ordering and overlap policy.
- Does not perform TDM phase compensation, DDM separation, sparse gathering, or angle estimation.

## When to use

Use after dense TDM compensation or dense DDM demultiplexing to flatten TX/RX maps into virtual channels and publish matching geometry. It is the boundary between MIMO separation/compensation and channel-based CFAR or angle processing. Do not use it before TDM phase correction, as a sparse gather, or as an angle estimator.

```text
… → TDMDenseCompensationOp / DDMDenseDemultiplexOp → VirtualArrayAssemblyOp → CFARDetectorOp / SnapshotGatherOp / ⟨angle op⟩
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(M_tx, N_rx, num_doppler, num_range_bins)` | Required trigger/aligned input | Separated `(tx, rx)` block. TX/RX positions are parameters, not ports. Rank 4; no opaque leading batch |
| out | `rd_maps_out` | complex64 / complex128 | `(num_virtual, num_doppler, num_range_bins)` | Required output | Channel flatten `c = tx·N_rx + rx`, then overlap policy. Doppler/range unchanged. physical name `rd_maps_out` |
| out | `array_geometry` | `WireArrayGeometry` | descriptor-Tensor | Required output | Logical `ArrayGeometry`: post-policy virtual positions, effective wavelength, and optional geometry type. Directly connects to angle operators' same-schema live inputs |
| live | `wavelength` | float64 scalar | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable, unwired = miss |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_tx`, `max_rx`, `max_doppler_bins`, and `max_range_bins` required; `max_virtual_channels` optional | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `tx_positions` | float vector, row-major `(num_tx, 3)` | none (required) | static (Params-only) | TX element positions (m). Length must match input `M_tx` Cost: Rebuilds the virtual plan + H2D into the ctor-sized index tables. |
| `rx_positions` | float vector, row-major `(num_rx, 3)` | none (required) | static (Params-only) | RX element positions (m). Length must match `N_rx` Cost: Same as `tx_positions`. |
| `wavelength` | float | none (required) | `[live]` Exact | Operating wavelength (m) for emitted geometry (`c / carrier_frequency`) Cost: Free for the gather; wire geometry metadata is rewritten each tick. |
| `overlap_policy` | enum `{dedup, average, keep_all}` | `dedup` | static (Params-only) | Coincident `p_virtual` handling; sets `num_virtual` Cost: Rebuilds the plan (gather vs average kernel). |

Contract: [`OPERATOR_API.md` § `VirtualArrayAssemblyOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/fmcw/README.md`](../README.md)

## Algorithm

Assembly is a channel gather (or average), not a spatial beamformer.
`compute()` is a thin Holoscan shell; tick helpers stay in the `.cu` anonymous namespace. It receives the cube, `require_complex<T>` in the same try as device + contiguous, merges this tick's Exact `wavelength` onto pinned Params (`resolve_params` → `VirtualAssemblyStageConfig`; the non-positive sentinel is rejected here), and binds that config to this input (`build_assembly_stage_plan` → `VirtualAssemblyStagePlan` with layout, gather plan, and output shape). Trailing rank is `(M_tx, N_rx, D, R)`, not an FMCW `(chirps, bins)` cube, so the plan builder uses `parse_rd_maps_layout` instead of `collapse_named_tx_rx_batch`. Rank must be exactly 4. Both outputs are required; their loans are prepared before publication. There is no latched optional axis.

**Virtual position**, meters, array-local frame:

```text
p_virtual[tx, rx] = p_tx[tx] + p_rx[rx]
```

**Channel index**, TX outer, RX inner — same order as the input `(M_tx, N_rx, …)` layout:

```text
c = tx · N_rx + rx
```

**Coincidence**: two positions match when all three components agree within `kPositionTol = 1e-9` m (`radar_common/dsp/virtual_array.cpp`). Not scaled by wavelength. Near-duplicates closer than this are merged; farther apart are distinct virtual elements. A wavelength-relative / λ/2-grid overlap rule is out of scope.

**Overlap policies**, scan order `(tx = 0..M−1, rx = 0..N−1)`:

| Policy | `num_virtual` | RD samples | Geometry |
| --- | --- | --- | --- |
| `dedup` (default) | unique coincident groups | copy the **first** `(tx, rx)` slot in scan order (`rep_channel`) | one row per group |
| `average` | unique coincident groups | complex mean `sum(members) / count` | one row per group |
| `keep_all` | `M_tx · N_rx` | identity gather `src_channel[c] = c` | all `M·N` positions, duplicates retained |

`dedup` and `keep_all` share `gather_single_source_kernel` (one source channel per virtual index). `average` uses `gather_average_kernel` over CSR-style `member_offsets` / `member_indices`. Host grouping lives in the CUDA-free `radar_common/dsp/virtual_array` planner shared with the fused sparse operators; this public operator retains only rank-4 dense layout handling and CUDA gather/average execution. A new `(tx_positions, rx_positions, overlap_policy, M_tx, N_rx)` pair rebuilds the host tables and async-H2Ds them into ctor-sized device buffers. A previously seen key is a cache hit. Wavelength is not part of that key.

**Emitted geometry.** `make_array_geometry` copies post-policy `virtual_positions` and the effective `wavelength`; `geometry_type` is left unset. `prepare_wire_array_geometry` publishes that logical value as `WireArrayGeometry`. Its pageable-host descriptor-Tensor member uses the schema bridge's internal float64 packing:

```text
rows [0, num_virtual):  (x, y, z) meters
row  num_virtual:       [wavelength, geometry_type_code, 0]
  geometry_type_code: 0=ULA, 1=URA, 2=UCA, 3=CONFORMAL; NaN if unset
```

Consumers receive the coherent descriptor with `from_wire`; direct angle operators do that in their live resolvers. Virtual positions do not depend on wavelength; only metadata does. `num_doppler` and `num_range` pass through unchanged. This is the public dense virtual-array boundary; fused sparse TDM/DDM operators emit the same sanctioned descriptor beside their final snapshots. C++ graphs connect it directly to angle operators' same-schema live inputs. Python remains on equivalent static Params because generated schema payloads are not exposed there.

## Shape and dtype rules

- `num_virtual` = `M_tx · N_rx` for `keep_all`, else the number of unique `p_tx+p_rx` groups (`≤ M·N`)
- `num_doppler` = input `shape[2]`
- `num_range_bins` = input `shape[3]`
- `array_geometry.packed_geometry` rows = `num_virtual + 1`, cols = 3

**Named axes.** The contract input is exactly `(M_tx, N_rx, num_doppler, num_range_bins)` (rank 4). There is no named leading batch. Constructor pool bounds pin `rd_maps_out` `max_rank = 3` (the flatten drops two named axes and inserts `num_virtual`). Extra leading axes fail closed in `parse_rd_maps_layout`.

`tx_positions.size()` must equal `num_tx * 3`; `rx_positions.size()` must equal `num_rx * 3` (row-major). Lengths are checked against the tick's `M_tx` / `N_rx`.

**Dtype.** Template `T` ∈ {`float`, `double`}; `rd_maps` complex only (`require_complex<T>`). `rd_maps_out` preserves that dtype. The wire geometry's Tensor member is float64.

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

**Mid-stream.** `overlap_policy`, `tx_positions`, and `rx_positions` are static. `wavelength` is `[live]` Exact: Params seeds the effective value, and an override whose `capture_time` matches the cube's replaces it for that tick only; a tick with no match uses the Params wavelength. Changing wavelength does **not** rebuild the gather plan (positions are wavelength-independent); the next tick's wire metadata updates. A new `M_tx`/`N_rx` (or Params positions/policy) rebuilds the host plan and re-uploads index tables. Downstream re-infers `num_virtual`. Extents past ctor pool bounds fail in the plan builder.

## Performance

**Cold.** First `(tx_positions, rx_positions, overlap_policy, M_tx, N_rx)`: host plan (`p_tx+p_rx`, grouping) + async H2D of index tables into the ctor-sized buffers, then gather/average kernel + host wire preparation. **Steady.** Same plan key: kernel + per-tick host wire preparation (wavelength may change without a plan rebuild). No FFT. A cache miss does not `cudaMalloc`.

Harness `performance/operators/fmcw/virtual_array_assembly/adapter.cpp`: size rungs `xs`–`xl` over `(M_tx, N_rx, num_doppler, num_range)`, overlap-policy variants. `array_geometry` is drained by a typed schema sink. No leading-batch rung. Every compute run inlines a mid-session `[live] wavelength` flip and reports `override_ms` / `override_delta_ms`.

Run this performance smoke **inside a development container**, after
configuring the named build tree with performance enabled (see the
[developer guide](../../../DEVELOPER.md#performance-characterization)):

```bash
ctest --test-dir build -R perf_virtual_array_assembly_smoke
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
graph.add_flow(vaa->array_geometry, beamscan->array_geometry, q);
```

`extents` must fill `max_tx`, `max_rx`, `max_doppler_bins`, `max_range_bins`, and optionally `max_virtual_channels` (else the virtual bound is `max_tx * max_rx`). Bind only the device `"rd_maps_out"`; the wire geometry pool declares pageable-host placement. Optional: wire a live scalar `wavelength`; leaving it unwired keeps Params.
