# `PhaseShiftBeamformerOp`

## Purpose

Forms a conventional (Bartlett) beam for every range-Doppler cell, replacing the channel axis with the `(num_az, num_el)` scan grid to build the Range-Doppler-Angle cube. It is the dense angle path: per-channel `rd_maps` → this operator → `CFARDetectorOp.rd_maps` (angle-then-cluster only). A full MVDR/MUSIC cube over every RD cell is prohibitive, so this is the only dense estimator. `angle_convention` is static and defaults to `radar_boresight` — it fixes the meaning of the beam axes for the whole graph.

## Responsibilities

- Builds and caches conventional steering weights for the configured geometry, convention, window, and scan grid.
- Applies the dense channel-to-angle projection at every range-Doppler cell and publishes the effective coordinate grids.
- Does not detect peaks, estimate source count, or provide adaptive MVDR/MUSIC processing.

## When to use

Use on per-channel range-Doppler maps when detection must run on a dense range-Doppler-angle cube. It is the dense angle path before `CFARDetectorOp`; sparse detect-then-angle workflows should use `SnapshotGatherOp` plus a sparse estimator instead. Do not use it when the cost of scanning every RD cell is unnecessary.

```text
rd_maps → PhaseShiftBeamformerOp → rda_cube → CFARDetectorOp.rd_maps
                              └──→ az_grid / el_grid → FeatureAssemblyOp
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128 | `(num_channels, num_doppler, num_range_bins)` | Required trigger/aligned input | Channel axis immediately before the `(doppler, range)` plane; `num_channels` must match `array_geometry`. See Shape for the exact-rank rule. |
| out | `rda_cube` | complex64 / complex128 | `(num_az, num_el, num_doppler, num_range_bins)` | Required output | Channel axis replaced by the `az × el` beam grid (`u × v` under `direction_cosine`). Always emitted. |
| out (optional, latched) | `az_grid` | float32 / float64 | `(num_az,)` | Optional latched output | Optional/latched effective azimuth (or `u`) grid for this tick; includes a matched live override. |
| out (optional, latched) | `el_grid` | float32 / float64 | `(num_el,)` | Optional latched output | Optional/latched effective elevation (or `v`) grid for this tick; includes a matched live override. |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. |
| live | `az_coordinates` / `el_coordinates` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; extents bounded by `max_az` / `max_el`. |
| live | `array_taper` | `WireWindowSpec` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_az`, `max_el`, `max_doppler_bins`, `max_range_bins` required | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required; empty fails `validate`) | `[live]` Exact wire-schema override | Virtual/physical element positions + wavelength. `num_elements` must match `rd_maps` channels. Cost: Rebuilds cached steering weights. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | How `az_coordinates` / `el_coordinates` and the output beam axes map to unit vectors, or `direction_cosine` for `(u, v)`. Cost: Rebuilds weights (part of the cache key). |
| `az_coordinates` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_az` | First coordinate: azimuth radians for angular conventions, dimensionless `u` for `direction_cosine`. Cost: Rebuilds weights. |
| `el_coordinates` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_el` | Second coordinate: elevation radians for angular conventions, dimensionless `v` for `direction_cosine`. Cost: Rebuilds weights. |
| `array_taper` | `WindowSpec` | `WindowType::kNone` (rectangular) | `[live]` Exact wire-schema override | Amplitude taper across elements. Cost: Rebuilds weights. |

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

`dir_b` comes from `boundary_to_canonical(convention, az, el)` — kernels only dot positions with canonical direction vectors. `AngleConvention` is a boundary relabel. Under `direction_cosine`, a grid point with `u² + v² > 1` is unphysical: `beam_valid = 0` and that beam's weights (hence cube power) are zero. Named presets and `custom` share the same path.

`array_taper` is an element-wise window on ULAs and unrecognized geometries. On a recognized filled URA in `(iz, iy)` channel order, non-custom windows use the separable product `w_y[iy] * w_z[iz]`; custom coefficients remain explicit per-element values. `WindowType::kNone` is all-ones / rectangular. Taylor / Chebyshev are the usual spatial sidelobe tapers.

## Shape and dtype rules

- `rda_cube` `num_az` / `num_el` = `az_coordinates.size()` / `el_coordinates.size()`.
- `az_grid` / `el_grid` contain those same effective values and lengths in template precision.
- `rda_cube` `num_doppler` / `num_range_bins` copy the input trailing pair.
- A size-1 `el_coordinates` (e.g. `[0]`) is a 1-D azimuth scan: shape `(num_az, 1, D, R)`.
- Precision is the template instantiation. Real-IF `rd_maps` is rejected.
- Inputs must be CUDA-accessible and compact row-major. Output is a device tensor.
- Contract shape is exactly rank-3 `(num_channels, D, R)`; any leading axis is rejected as opaque module batch.
- A scan grid / geometry / taper change (override or `set_params`) that misses the steering cache rebuilds weights; downstream re-infers `num_az` / `num_el`. An override grid longer than `max_az` / `max_el` is rejected on arrival.
- Input Doppler/range extents are checked against captured `max_doppler_bins` / `max_range_bins` before the output loan.

Rejected combinations and the operator's own messages:

| Condition | Message |
| --- | --- |
| empty `az_coordinates` or `el_coordinates` | `PhaseShiftBeamformerOp: az_coordinates and el_coordinates must be non-empty` |
| `rd_maps` rank other than 3 | `PhaseShiftBeamformerOp: rd_maps must have shape (num_channels, num_doppler, num_range_bins)` |
| non-positive input axis | `PhaseShiftBeamformerOp: rd_maps axes must be positive` |
| `num_channels ≠ array_geometry.num_elements` | `PhaseShiftBeamformerOp: rd_maps num_channels must match array_geometry.num_elements` |
| input or scan extent above captured bounds | `PhaseShiftBeamformerOp: input or scan grid exceeds MaxTensorExtents` |
| invalid geometry | `ArrayGeometry: …` (same validate messages as other geometry consumers) |
| real-valued / wrong-precision `rd_maps` | `rd_maps: real-valued input …` / `rd_maps: received dtype … does not match this operator's …-precision instantiation` |
| non-contiguous / pageable-host | `PhaseShiftBeamformerOp: tensor received on input port 'rd_maps' must be contiguous (compact row-major)` / `… is not CUDA-accessible (device, managed, or pinned-host memory required)` |
| no CUDA compute stream | `PhaseShiftBeamformerOp: no CUDA compute stream` |

Dense-cube memory is an accepted caller-owned trade-off, but each named extent is rejected before allocation when it exceeds the constructor capture.

## Performance

Cold / first unseen steering key: host taper + per-beam `boundary_to_canonical`, H2D scratch, the weight kernel, and device-grid materialization; then the GEMM. Steady with grid outputs wired: cache hit + one broadcast `matx::matmul` (cuBLAS) + two small D2D grid copies. The key covers the complete `AngleConvention` (including custom signed axes and elevation reference), wavelength, positions, both grids, and window key, so an override costs one artifact rebuild on the tick it arrives.

Harness: `performance/operators/angle_estimation/phase_shift_beamformer/adapter.cpp`. Size ladder `xs`…`xl` (channels × `num_az` × `num_el` × RD extent). Every selected size runs four isolated inline-override scenarios: `array_geometry`, `az_coordinates`, `el_coordinates`, and `array_taper`. The source co-emits the complete live bundle before the triggering `rd_maps`; one value changes mid-session, and the report records `override_ms` / `override_delta_ms` against pre-override steady state. No reference results are checked in; run:

Run this performance smoke **inside a development container**, after
configuring the named build tree with performance enabled (see the
[developer guide](../../../DEVELOPER.md#performance-characterization)):

```bash
ctest --test-dir build -R perf_phase_shift_beamformer_smoke
```

## Usage

```yaml
phase_shift_beamformer:
  angle_convention:
    preset: radar_boresight
  az_coordinates: [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2,
    0.25, 0.3]
  el_coordinates: [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2,
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
graph.add_flow(beamformer->az_grid, features->az_grid, q);
graph.add_flow(beamformer->el_grid, features->el_grid, q);
```
