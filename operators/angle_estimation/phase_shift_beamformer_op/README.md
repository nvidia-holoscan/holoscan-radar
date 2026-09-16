# `PhaseShiftBeamformerOp`

Forms a conventional (Bartlett) beam for every range-Doppler cell, replacing the channel axis with the `(num_az, num_el)` scan grid to build the Range-Doppler-Angle cube. It is the dense angle path: per-channel `rd_maps` → this operator → `CFARDetectorOp.rd_maps` (angle-then-cluster only). A full MVDR/MUSIC cube over every RD cell is prohibitive, so this is the only dense estimator. `angle_convention` is static and defaults to `radar_boresight` — it fixes the meaning of the beam axes for the whole graph.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(num_channels, num_doppler, num_range_bins)` | Channel axis immediately before the `(doppler, range)` plane; `num_channels` must match `array_geometry`. See Shape for the exact-rank rule. |
| out | `rda_cube` | complex64 / complex128 | `(num_az, num_el, num_doppler, num_range_bins)` | Channel axis replaced by the `az × el` beam grid (`u × v` under `direction_cosine`). Always emitted. |
| out | `az_grid` | float32 / float64 | `(num_az,)` | Optional/latched effective azimuth (or `u`) grid for this tick; includes a matched live override. |
| out | `el_grid` | float32 / float64 | `(num_el,)` | Optional/latched effective elevation (or `v`) grid for this tick; includes a matched live override. |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable. |
| live | `az_angles` / `el_angles` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | `[live]` Exact override; extents bounded by `max_az` / `max_el`. |
| live | `array_taper` | `WireWindowSpec` | descriptor wire object | `[live]` Exact override; always declared, wire to enable. |

```text
rd_maps → PhaseShiftBeamformerOp → rda_cube → CFARDetectorOp.rd_maps
                              └──→ az_grid / el_grid → FeatureAssemblyOp
```

In-tree `rd_maps` producers are the FMCW Doppler paths and virtual-array
assembly.

Contract: [`OPERATOR_API.md` § `PhaseShiftBeamformerOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md)

## Algorithm

The shell (`phase_shift_beamformer_op.cu`) receives and validates the rank-3 cube, resolves the four live values into a tick-local `PhaseShiftBeamformerStageConfig`, and builds a `PhaseShiftBeamformerStagePlan` with captured-extent checks and output geometry. It then obtains the runtime stream and fetches conjugated steering weights plus device-resident effective grids from one `ConfigCache`. The required cube is prepared first; active grid sidebands are published before the cube so downstream Exact joins see the axes for that activation. Weights are built on device (`build_steering_weights_device`) as `(num_beams, num_channels)` with `BeamOrder::kAzMajor` (`beam = ia * num_el + ie`), `SteeringLayout::kBeamMajor`, `conjugate = true`. The stage is one broadcast complex GEMM (`phase_shift_beamformer_stage.hpp`):

```text
Y[b, beam, n] = Σ_c Wᶜ[beam, c] · X[b, c, n]
Wᶜ = conj(w)     [num_beams, num_channels]   (az-major; invalid beams are zero)
X  = rd_maps     [batch, num_channels, num_cells]
Y  = rda_cube    [batch, num_beams,    num_cells]
num_beams = num_az * num_el
num_cells = num_doppler * num_range_bins
```

The stage keeps a size-one batch dimension for the broadcast GEMM. The contiguous `Y` buffer **is** the `(num_az, num_el, D, R)` cube — no post-permute.

Each weight is (`steering_weights_device.cu`):

```text
k = 2π / wavelength
phase = k · (p_m · dir_b)     # p_m = element m; dir_b = unit vector for beam b
w = taper[m] · (cos(phase) + j sin(phase))
Wᶜ = taper[m] · (cos(phase) − j sin(phase))   # conjugate=true
```

`dir_b` comes from `boundary_to_canonical(convention, az, el)` — kernels only dot positions with canonical direction vectors. `AngleConvention` is a boundary relabel. Under `direction_cosine`, a grid point with `u² + v² > 1` is unphysical: `beam_valid = 0` and that beam's weights (hence cube power) are zero. Named presets and `custom` share the same path (STATUS.md L11).

`array_taper` is an element-wise window on ULAs and unrecognized geometries. On a recognized filled URA in `(iz, iy)` channel order, non-custom windows use the separable product `w_y[iy] * w_z[iz]`; custom coefficients remain explicit per-element values. `WindowType::kNone` is all-ones / rectangular. Taylor / Chebyshev are the usual spatial sidelobe tapers.

## Shape and dtype rules

- `rda_cube` `num_az` / `num_el` = `az_angles.size()` / `el_angles.size()`.
- `az_grid` / `el_grid` contain those same effective values and lengths in template precision.
- `rda_cube` `num_doppler` / `num_range_bins` copy the input trailing pair.
- A size-1 `el_angles` (e.g. `[0]`) is a 1-D azimuth scan: shape `(num_az, 1, D, R)`.
- Precision is the template instantiation. Real-IF `rd_maps` is rejected.
- Inputs must be CUDA-accessible and compact row-major. Output is a device tensor.
- Contract shape is exactly rank-3 `(num_channels, D, R)`; any leading axis is rejected as opaque module batch.
- A scan grid / geometry / taper change (override or `set_params`) that misses the steering cache rebuilds weights; downstream re-infers `num_az` / `num_el`. An override grid longer than `max_az` / `max_el` is rejected on arrival.
- Input Doppler/range extents are checked against captured `max_doppler_bins` / `max_range_bins` before the output loan.

Rejected combinations and the operator's own messages:

| Condition | Message |
| --- | --- |
| empty `az_angles` or `el_angles` | `PhaseShiftBeamformerOp: az_angles and el_angles must be non-empty` |
| `rd_maps` rank other than 3 | `PhaseShiftBeamformerOp: rd_maps must have shape (num_channels, num_doppler, num_range_bins)` |
| non-positive input axis | `PhaseShiftBeamformerOp: rd_maps axes must be positive` |
| `num_channels ≠ array_geometry.num_elements` | `PhaseShiftBeamformerOp: rd_maps num_channels must match array_geometry.num_elements` |
| input or scan extent above captured bounds | `PhaseShiftBeamformerOp: input or scan grid exceeds MaxTensorExtents` |
| invalid geometry | `ArrayGeometry: …` (same validate messages as other geometry consumers) |
| real-valued / wrong-precision `rd_maps` | `rd_maps: real-valued input …` / `rd_maps: received dtype … does not match this operator's …-precision instantiation` |
| non-contiguous / pageable-host | `PhaseShiftBeamformerOp: tensor received on input port 'rd_maps' must be contiguous (compact row-major)` / `… is not CUDA-accessible (device, managed, or pinned-host memory required)` |
| no CUDA compute stream | `PhaseShiftBeamformerOp: no CUDA compute stream` |

Dense-cube memory is an accepted caller-owned trade-off, but each named extent is rejected before allocation when it exceeds the constructor capture.

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required; empty fails `validate`) | `[live]` Exact wire-schema override ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9) | Rebuilds cached steering weights | Virtual/physical element positions + wavelength. `num_elements` must match `rd_maps` channels. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Rebuilds weights (part of the cache key) | How `az_angles` / `el_angles` and the output beam axes map to unit vectors, or `direction_cosine` for `(u, v)`. |
| `az_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_az` (§9) | Rebuilds weights | Azimuth scan grid (`u` under `direction_cosine`). |
| `el_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_el` (§9) | Rebuilds weights | Elevation scan grid (`v` under `direction_cosine`). |
| `array_taper` | `WindowSpec` | `WindowType::kNone` (rectangular) | `[live]` Exact wire-schema override (§9) | Rebuilds weights | Amplitude taper across elements. |

Application YAML accepts explicit arrays or inclusive `{min: ..., max: ..., step: ...}` mappings for `az_angles` and `el_angles`. Range mappings require finite values, `step > 0`, `min <= max`, and endpoint alignment within floating-point tolerance. They expand to the same `std::vector<double>` Params before construction; live ports remain rank-1 float64 Tensors. Every effective grid coordinate must be finite. Under `direction_cosine`, each authored axis value must lie in `[-1, 1]`; square-grid corners outside `u² + v² <= 1` remain legal and are masked per beam.

Contract marks geometry / grids / taper **[live]** (warn-once, cache), and each has an override port whose match applies for one tick before the effective value reverts to Params. The logical Params types stay `ArrayGeometry` / `std::vector<double>` / `WindowSpec` — the wire-object and host-Tensor payloads are a port detail (HSDK5_PORT.md §9). `ConfigCache` dedupes identical keys and warns once per unseen key after the first. YAML / `from_config` / `set_params` use the table names. `angle_convention` is never live.

## SDK implementation notes

- **Construction.** `PhaseShiftBeamformerOp(const MaxTensorExtents&)`. **Requires** `max_az`, `max_el`, `max_doppler_bins`, `max_range_bins` (no empty-span Python path). Sizes `rda_cube` as a rank-4 complex cube and the two rank-1 grid pools from those extents. There is no `max_channels` in the output span — beamforming collapses channels. A tick that exceeds the span fails at loan allocate.
- **Triggering.** `contract()` is `OnEach{rd_maps}` plus one unconditional `Exact{}` + `SafeDefault{}` entry per `[live]` override (`array_geometry`, `az_angles`, `el_angles`, `array_taper`). Optional outputs stay out of the contract so unwired grids do not gate scheduling. The application enables an override by `add_flow`-ing its port (an unwired entry reports `kNotConnected` and reads as no override; a wired port missing from the contract is `RUNTIME_OPERATOR_REPLAY_MISMATCH` — [`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1).
- **Ports and queues.** Data plane first: `rd_maps`, required rank-4 `rda_cube`, then optional rank-1 `az_grid` / `el_grid`; all use device memory and template precision. The live block follows in Params order; all Inputs use `queue_depth(2)`. `az_angles` / `el_angles` overrides are rank-1 pageable-host float64 config Tensors and `array_geometry` / `array_taper` are wire objects.
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>` and `array_taper` is `Input<WireWindowSpec>` — generated descriptor-schema types converted with `from_wire` inside `resolve_live_override_via`, so wire storage never reaches operator state. `az_angles` / `el_angles` are validated on arrival (rank, host memory, contiguity, dtype, `max_az` / `max_el` extent) and converted per tick, with no state kept ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9).
- **Streams and lifetime.** GEMM and artifact builds use `context.cuda_stream()`. `rd_maps` is retained across ticks; cached device grids remain alive across their stream-ordered D2D copies. The hot path does not synchronize. Weight-builder scratch is `cudaMallocAsync` / `cudaFreeAsync` on that stream.
- **Emission.** Prepare the required `rda_cube` loan first, publish active latched grid sidebands from cached device tensors, then publish the cube trigger. An unwired grid is attempted once and latched off without suppressing `rda_cube`.
- **Parameter authoring.** Schema id `com.nvidia.holoscan.radar.PhaseShiftBeamformerParams`, version 1.
- **Precision.** `PhaseShiftBeamformerOp<float>`, `PhaseShiftBeamformerOp<double>`.
- **[EA HSDK5 ONLY]** Runtime geometry producers connect directly to `array_geometry` with `WireArrayGeometry`; Python remains Params-only for schema payloads. Pageable-host rejection is not exercisable via the device-bound Graph harness.

## Consumer lifetime notes

Every emitted tensor is a fresh pool loan. Consumers may retain the cube and grids. The full dense cube is intentionally caller-sized through the scan grid and captured RD extents.

## Performance

Cold / first unseen steering key: host taper + per-beam `boundary_to_canonical`, H2D scratch, the weight kernel, and device-grid materialization; then the GEMM. Steady with grid outputs wired: cache hit + one broadcast `matx::matmul` (cuBLAS) + two small D2D grid copies. The key covers the complete `AngleConvention` (including custom signed axes and elevation reference), wavelength, positions, both grids, and window key, so an override costs one artifact rebuild on the tick it arrives.

Harness: `performance/cpp/perf_phase_shift_beamformer.cpp`. Size ladder `xs`…`xl` (channels × `num_az` × `num_el` × RD extent). Every selected size runs four isolated inline-override scenarios: `array_geometry`, `az_angles`, `el_angles`, and `array_taper`. The source co-emits the complete live bundle before the triggering `rd_maps`; one value changes mid-session, and the report records `override_ms` / `override_delta_ms` against pre-override steady state. No current HSDK 5 Graph results are checked in; run:

```bash
ctest --test-dir build -R perf_phase_shift_beamformer_smoke
```

STATUS.md records a historical host-vs-device weight-build note from the 4.x landing (host build ≈ 0.97 ms at the `lg` override grid); that is not a Graph-harness number on this branch.

## Tests and verification

C++: `tests/cpp/angle_estimation/test_phase_shift_beamformer.cpp`. Python: `tests/python/angle_estimation/test_phase_shift_beamformer.py` (legacy HSDK 4.x bindings; not built on this branch). Ground truth is `make_steering_vector` / `make_ula_geometry` / `make_planar_ura_geometry` from `radar_common` (analytic array response), not the point-target ADC simulator.

| Check | Proves |
| --- | --- |
| `DefaultsToRadarBoresightConvention` | Default-constructed Params select the documented `radar_boresight` convention |
| `PublishesEffectiveGridsFloat` / `PublishesEffectiveGridsDouble` | Optional grid outputs preserve configured values, shapes, and template precision |
| `OptionalGridsMayRemainUnwired` | Cube-only consumers run without draining either grid |
| `RecoversKnownBeamBin2D` | URA steering vector → cube peak within 0.02 rad of truth |
| `RecoversKnownBeamAzOnly1DScan` | ULA + `el_angles = [0]` → azimuth peak |
| `ConventionRelabelsSamePhysicalDirection` | `radar_boresight` and `aerospace_ned` report the same labels |
| `DirectionCosineMasksInvisibleRegion` | `u²+v² > 1` → zero power; center `(0,0)` nonzero |
| `ArrayTaperLowersSidelobes` / `ArrayTaperConfigurableFromParams` | Hann peak-sidelobe ratio `<` rectangular |
| `PreservesRangeDopplerPlanePerCell` | Two RD cells keep distinct DOA peaks; empty cells ≈ 0 |
| `RejectsOpaqueLeadingBatch` | Any extra leading dim is rejected by the rank-3 contract |
| `FloatDoubleAgreeOnPeakBeam` | Peak beam index matches across precisions |
| `ScanGridConfigurableFromParams` | Params set `num_az` / `num_el` |
| `RejectsRankTooLow` / `RejectsChannelMismatch` / `RejectsEmptyScanGrid` / `RejectsDtypeMismatch` / `RejectsInputBeyondCapturedExtents` | Named error paths and constructor bounds |
| `CacheKeyIncludesCustomConventionAxes` | Distinct custom frames and close floating-point grids cannot alias one steering-table cache entry |
| `RejectsDegenerateCustomConvention` | Invalid custom signed-axis frames fail in plan construction |
| `LiveAzGridAppliesForOneTick` / `LiveElGridAppliesForOneTick` / `LiveArrayGeometryAppliesForOneTick` / `LiveArrayTaperAppliesForOneTick` | Mid-stream override reshapes / rebuilds the cube and corresponding grid output, then the following miss reverts to Params |
| `LiveOverridesRejectMalformedAndBounds` | Malformed wire payload and an over-`max_az` / `max_el` grid are diagnosed, not consumed |

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
graph.add_flow(beamformer->az_grid, features->az_angles, q);
graph.add_flow(beamformer->el_grid, features->el_angles, q);
```

Python bindings exist at `python/_phase_shift_beamformer_op_bindings.cpp` as an HSDK 4.x trampoline and are not built (`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`).

## Contract conformance

Matches the dense Bartlett contract for ports, rank, GEMM, convention boundary, taper, live overrides, and coherent optional grid outputs.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` / `array_taper` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` / `Input<WireWindowSpec>` in `contract()`, not `Input<ArrayGeometry>` / `Input<WindowSpec>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). |
| `az_angles` / `el_angles` **[live]** | conforms (payload form differs) | Rank-1 host config Tensors bounded by `max_az` / `max_el`, not `Input<std::vector<double>>` (§9). |
| `az_grid` / `el_grid` optional outputs | conforms | Rank-1 device Tensors in template precision publish the effective tick-local grids and latch off when unwired. |
| `rd_maps` shape `(num_channels, D, R)` | conforms | The stage plan requires rank 3; no opaque leading batch is admitted. |
| Steering cache identity | conforms | The shared key includes every `AngleConvention` field, geometry, grids, and taper. |

The C++ Graph suite, four-scenario live-override performance smoke, and focused lint commands are the verification surface for this cleanup.
