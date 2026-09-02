# `PhaseShiftBeamformerOp`

Forms a conventional (Bartlett) beam for every range-Doppler cell, replacing the channel axis with the `(num_az, num_el)` scan grid to build the Range-Doppler-Angle cube. It is the dense angle path: per-channel `rd_maps` → this operator → `CFARDetectorOp.rd_maps` (angle-then-cluster only). A full MVDR/MUSIC cube over every RD cell is prohibitive, so this is the only dense estimator. `angle_convention` is static and required — it fixes the meaning of the beam axes for the whole graph.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(num_channels, num_doppler, num_range_bins)` | Channel axis immediately before the `(doppler, range)` plane; `num_channels` must match `array_geometry`. See Shape for named-leading admission. |
| out | `rda_cube` | complex64 / complex128 | `(num_az, num_el, num_doppler, num_range_bins)` | Channel axis replaced by the `az × el` beam grid (`u × v` under `direction_cosine`). Always emitted. |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable. |
| live | `az_angles` / `el_angles` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | `[live]` Exact override; extents bounded by `max_az` / `max_el`. |
| live | `array_taper` | `WireWindowSpec` | descriptor wire object | `[live]` Exact override; always declared, wire to enable. |

```text
rd_maps (per-channel) → PhaseShiftBeamformerOp → rda_cube → CFARDetectorOp.rd_maps
```

In-tree `rd_maps` today: `FMCWDopplerProcessorOp`. A pulse-Doppler Doppler
stage would plug the same port.

Contract: [`OPERATOR_API.md` § `PhaseShiftBeamformerOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md) ·
open items: [`OPEN_ITEMS.md`](OPEN_ITEMS.md)

## Algorithm

The shell (`phase_shift_beamformer_op.cu`) validates the cube and Params, then fetches conjugated steering weights from a `ConfigCache` keyed by `steering_cache_key`. Weights are built on device (`build_steering_weights_device`) as `(num_beams, num_channels)` with `BeamOrder::kAzMajor` (`beam = ia * num_el + ie`), `SteeringLayout::kBeamMajor`, `conjugate = true`. The stage is one broadcast complex GEMM (`phase_shift_beamformer_stage.hpp`):

```text
Y[b, beam, n] = Σ_c Wᶜ[beam, c] · X[b, c, n]
Wᶜ = conj(w)     [num_beams, num_channels]   (az-major; invalid beams are zero)
X  = rd_maps     [batch, num_channels, num_cells]
Y  = rda_cube    [batch, num_beams,    num_cells]
num_beams = num_az * num_el
num_cells = num_doppler * num_range_bins
```

`Wᶜ` is `matx::clone`'d with a zero batch stride so one table is reused across every leading slice. The contiguous `Y` buffer **is** the `(…, num_az, num_el, D, R)` cube — no post-permute (OPEN_ITEMS.md §3).

Each weight is (`steering_weights_device.cu`):

```text
k = 2π / wavelength
phase = k · (p_m · dir_b)     # p_m = element m; dir_b = unit vector for beam b
w = taper[m] · (cos(phase) + j sin(phase))
Wᶜ = taper[m] · (cos(phase) − j sin(phase))   # conjugate=true
```

`dir_b` comes from `boundary_to_canonical(convention, az, el)` — kernels only dot positions with canonical direction vectors. `AngleConvention` is a boundary relabel. Under `direction_cosine`, a grid point with `u² + v² > 1` is unphysical: `beam_valid = 0` and that beam's weights (hence cube power) are zero. Named presets and `custom` share the same path (STATUS.md L11).

`array_taper` is `generate_window(spec, num_channels)` across elements (`WindowType::kNone` → all-ones / rectangular). Taylor / Chebyshev are the usual spatial sidelobe tapers.

## Shape and dtype rules

- `rda_cube` `num_az` / `num_el` = `az_angles.size()` / `el_angles.size()`.
- `rda_cube` `num_doppler` / `num_range_bins` copy the input trailing pair.
- A size-1 `el_angles` (e.g. `[0]`) is a 1-D azimuth scan: shape `(num_az, 1, D, R)`.
- Precision is the template instantiation. Real-IF `rd_maps` is rejected.
- Inputs must be CUDA-accessible and compact row-major. Output is a device tensor.
- Contract shape is rank-3 `(num_channels, D, R)`. `compute()` also admits named leading `N_rx` or `(M_tx, N_rx)` when `MaxTensorExtents` sets `max_rx` / `max_channels` or `max_tx` (`named_tx_rx_leading_rank`); any extra leading rank is rejected as opaque batch. The constructor pool is **not** sized for those leading axes (see Contract conformance).
- A scan grid / geometry / taper change (override or `set_params`) that misses the steering cache rebuilds weights; downstream re-infers `num_az` / `num_el`. An override grid longer than `max_az` / `max_el` is rejected on arrival.

Rejected combinations and the operator's own messages:

| Condition | Message |
| --- | --- |
| empty `az_angles` or `el_angles` | `PhaseShiftBeamformerOp: az_angles and el_angles must be non-empty` |
| `rd_maps` rank `< 3` | `PhaseShiftBeamformerOp: rd_maps must have shape (..., num_channels, num_doppler, num_range_bins) with rank >= 3` |
| leading rank above the named max | `PhaseShiftBeamformerOp: unexpected leading axes (opaque batch is not supported); got leading rank G, max named leading M` |
| `num_channels ≠ array_geometry.num_elements` | `PhaseShiftBeamformerOp: rd_maps num_channels must match array_geometry.num_elements` |
| invalid geometry | `ArrayGeometry: …` (same validate messages as other geometry consumers) |
| real-valued / wrong-precision `rd_maps` | `rd_maps: real-valued input …` / `rd_maps: received dtype … does not match this operator's …-precision instantiation` |
| non-contiguous / pageable-host | `PhaseShiftBeamformerOp: tensor received on input port 'rd_maps' must be contiguous (compact row-major)` / `… is not CUDA-accessible (device, managed, or pinned-host memory required)` |
| no CUDA compute stream | `PhaseShiftBeamformerOp: no CUDA compute stream` |

A tick whose `num_az × num_el × D × R` exceeds the captured `max_byte_span` fails at `allocate_tensor`, not with a named operator message. Dense-cube memory is the caller's footgun (OPEN_ITEMS.md §2).

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required; empty fails `validate`) | `[live]` Exact wire-schema override ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9) | Rebuilds cached steering weights | Virtual/physical element positions + wavelength. `num_elements` must match `rd_maps` channels. |
| `angle_convention` | `AngleConvention` | contract: none (required). Params default-construct to `radar_boresight` — see conformance | static | Rebuilds weights (part of the cache key) | How `az_angles` / `el_angles` and the output beam axes map to unit vectors, or `direction_cosine` for `(u, v)`. |
| `az_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_az` (§9) | Rebuilds weights | Azimuth scan grid (`u` under `direction_cosine`). |
| `el_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_el` (§9) | Rebuilds weights | Elevation scan grid (`v` under `direction_cosine`). |
| `array_taper` | `WindowSpec` | `WindowType::kNone` (rectangular) | `[live]` Exact wire-schema override (§9) | Rebuilds weights | Amplitude taper across elements. |

Contract marks geometry / grids / taper **[live]** (warn-once, cache), and each has an override port whose match applies for one tick before the effective value reverts to Params. The logical Params types stay `ArrayGeometry` / `std::vector<double>` / `WindowSpec` — the wire-object and host-Tensor payloads are a port detail (HSDK5_PORT.md §9). `ConfigCache` dedupes identical keys and warns once per unseen key after the first. YAML / `from_config` / `set_params` use the table names. `angle_convention` is never live.

## SDK implementation notes

- **Construction.** `PhaseShiftBeamformerOp(const MaxTensorExtents&)`. **Requires** `max_az`, `max_el`, `max_doppler_bins`, `max_range_bins` (no empty-span Python path). Sizes `rda_cube` as a rank-4 complex cube of those extents. There is no `max_channels` in the output span — beamforming collapses channels. A tick that exceeds the span fails at loan allocate.
- **Triggering.** `contract()` is `OnEach{rd_maps}` plus one unconditional `Exact{}` + `SafeDefault{}` entry per `[live]` override (`array_geometry`, `az_angles`, `el_angles`, `array_taper`). There are no optional ports. The application enables an override by `add_flow`-ing its port (an unwired entry reports `kNotConnected` and reads as no override; a wired port missing from the contract is `RUNTIME_OPERATOR_REPLAY_MISMATCH` — [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1).
- **Ports and queues.** `rd_maps` `queue_depth(2)`, `MemoryKind::kCudaDevice`, rank-agnostic (dtype in `compute()`). Live Inputs `queue_depth(2)`; `az_angles` / `el_angles` are rank-1 pageable-host float64 config Tensors, `array_geometry` / `array_taper` are wire objects. Output `tensor_allocation` `{max_rank = 4, max_byte_span = …}`.
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>` and `array_taper` is `Input<WireWindowSpec>` — generated descriptor-schema types converted with `from_wire` inside `resolve_live_override_via`, so wire storage never reaches operator state. `az_angles` / `el_angles` are validated on arrival (rank, host memory, contiguity, dtype, `max_az` / `max_el` extent) and converted per tick, with no state kept ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9).
- **Streams and lifetime.** GEMM and weight builds use `context.cuda_stream()`. `rd_maps` is retained across ticks. The hot path does not synchronize. Weight-builder scratch is `cudaMallocAsync` / `cudaFreeAsync` on that stream.
- **Emission.** `emit_device_tensor` on `rda_cube`: loan → fill (GEMM) → commit → emit.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.PhaseShiftBeamformerParams`, version 1.
- **Precision.** `PhaseShiftBeamformerOp<float>`, `PhaseShiftBeamformerOp<double>`.
- **[EA HSDK5 ONLY]** An upstream descriptor output cannot drive `array_geometry`: `VirtualArrayAssemblyOp` / `SubarraySelectionOp` emit a packed float64 Tensor (§10) while this port takes the wire-schema type, so the override has to come from an application-side wire producer (§9 / §10). Pageable-host rejection is not exercisable via the device-bound Graph harness.

## Consumer lifetime notes

`rda_cube` is a fresh pool loan filled and committed each tick. A consumer may retain it. The cube is large — see OPEN_ITEMS.md §2.

## Performance

Cold / first unseen steering key: host taper + per-beam `boundary_to_canonical`, H2D scratch, and the weight kernel; then the GEMM. Steady: cache hit + one broadcast `matx::matmul` (cuBLAS). The key is `steering_cache_key` (preset, wavelength, positions, both grids, window key), so an override costs one weight rebuild on the tick it arrives. Inline live-override timing is deferred (HSDK5_PORT.md §1; perf packaging §8).

Harness: `performance/cpp/perf_phase_shift_beamformer.cpp`. Size ladder `xs`…`xl` (channels × `num_az` × `num_el` × RD extent). Live `az_angles` override measurement is deferred. No in-repo measured numbers for this operator; run:

```bash
ctest --test-dir build -R perf_phase_shift_beamformer_smoke
```

STATUS.md records a historical host-vs-device weight-build note from the 4.x landing (host build ≈ 0.97 ms at the `lg` override grid); that is not a Graph-harness number on this branch.

## Tests and verification

C++: `tests/cpp/angle_estimation/test_phase_shift_beamformer.cpp`. Python: `tests/python/angle_estimation/test_phase_shift_beamformer.py` (HSDK 4.x bindings; not built — OPEN_ITEMS.md §4). Ground truth is `make_steering_vector` / `make_ula_geometry` / `make_planar_ura_geometry` from `radar_common` (analytic array response), not the point-target ADC simulator.

| Check | Proves |
| --- | --- |
| `RecoversKnownBeamBin2D` | URA steering vector → cube peak within 0.02 rad of truth |
| `RecoversKnownBeamAzOnly1DScan` | ULA + `el_angles = [0]` → azimuth peak |
| `ConventionRelabelsSamePhysicalDirection` | `radar_boresight` and `aerospace_ned` report the same labels |
| `DirectionCosineMasksInvisibleRegion` | `u²+v² > 1` → zero power; center `(0,0)` nonzero |
| `ArrayTaperLowersSidelobes` / `ArrayTaperConfigurableFromParams` | Hann peak-sidelobe ratio `<` rectangular |
| `PreservesRangeDopplerPlanePerCell` | Two RD cells keep distinct DOA peaks; empty cells ≈ 0 |
| `RejectsOpaqueLeadingBatchWhenExtentsOmitRxTx` | Extra leading dim rejected when named leading rank is 0 |
| `FloatDoubleAgreeOnPeakBeam` | Peak beam index matches across precisions |
| `ScanGridConfigurableFromParams` | Params set `num_az` / `num_el` |
| `RejectsRankTooLow` / `RejectsChannelMismatch` / `RejectsEmptyScanGrid` / `RejectsDtypeMismatch` | Named error paths |
| `LiveAzGridAppliesForOneTick` / `LiveElGridAppliesForOneTick` / `LiveArrayGeometryAppliesForOneTick` / `LiveArrayTaperAppliesForOneTick` | Mid-stream override reshapes / rebuilds the cube, and sticks after the next miss |
| `LiveOverridesRejectMalformedAndBounds` | Malformed wire payload and an over-`max_az` / `max_el` grid are diagnosed, not consumed |
| `RejectsPageableHostInput` | [EA HSDK5 ONLY] `GTEST_SKIP`'d |

```bash
ctest --test-dir build -R test_phase_shift_beamformer --output-on-failure
```

## Usage

```yaml
phase_shift_beamformer:
  angle_convention:
    preset: radar_boresight
  az_angles: [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2,
    0.25, 0.3]
  el_angles: [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2,
    0.25, 0.3]
  array_taper:
    type: taylor
    sidelobe_attenuation_db: 35.0
  # array_geometry is set in C++ from the RX array
```

```cpp
auto beamformer = graph.op<PhaseShiftBeamformerOp<float>>(
    "phase_shift_beamformer",
    MaxTensorExtents{/* max_az, max_el, max_doppler_bins, max_range_bins */});
beamformer.set_params(cfg.beamformer_params);

graph.add_flow(doppler->rd_maps, beamformer->rd_maps, q);
graph.add_flow(beamformer->rda_cube, cfar->rd_maps, q);
```

Python bindings exist at `python/_phase_shift_beamformer_op_bindings.cpp` as an HSDK 4.x trampoline and are not built (`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

## Contract conformance

Matches the dense Bartlett contract (ports, GEMM, convention boundary, taper, required grids). Three undocumented divergences below.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` / `array_taper` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` / `Input<WireWindowSpec>` in `contract()`, not `Input<ArrayGeometry>` / `Input<WindowSpec>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9, OPEN_ITEMS.md §1). |
| `az_angles` / `el_angles` **[live]** | conforms (payload form differs) | Rank-1 host config Tensors bounded by `max_az` / `max_el`, not `Input<std::vector<double>>` (§9). |
| `angle_convention` required, no default | deviates | `PhaseShiftBeamformerParams` default-constructs `AngleConvention{}` → `kRadarBoresight` (`angle_convention.hpp` line 14; `phase_shift_beamformer_op.hpp` line 59). No reject-if-unset. |
| `rd_maps` shape `(num_channels, D, R)` | deviates | `compute()` admits named leading axes via `named_tx_rx_leading_rank` (`phase_shift_beamformer_op.cu` lines 130–136). Constructor pool is rank-4 `{max_az, max_el, max_doppler, max_range}` with no leading product (lines 47–59). Header documents `...`; contract does not. |
| Steering cache identity | deviates | `steering_cache_key` (`steering_weights.cpp` lines 23–24) hashes only `convention.preset`, not the signed-axis / `elevation_reference` fields. Two `kCustom` conventions can reuse the wrong table. |

Static audit of `phase_shift_beamformer_op.hpp`, `phase_shift_beamformer_op.cu`, `phase_shift_beamformer_stage.hpp`, `metadata.json`, `python/_phase_shift_beamformer_op_bindings.cpp`, `OPEN_ITEMS.md`, `radar_common/dsp/steering_weights*.{hpp,cpp,cu}`, `tests/cpp/angle_estimation/test_phase_shift_beamformer.cpp`, `performance/cpp/perf_phase_shift_beamformer.cpp`, and the planning docs listed in the skill. No build or test run.
