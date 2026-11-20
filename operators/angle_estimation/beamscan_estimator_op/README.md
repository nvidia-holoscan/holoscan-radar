# `BeamscanEstimatorOp`

## Purpose

Conventional (Bartlett) sparse-path DOA: scan a grid and emit the dominant `(az, el)` (or `(u, v)`). It fills the shared sparse-estimator slot after `SnapshotGatherOp` (or `SubarraySelectionOp`). A 2-D snapshot is one look; a 3-D snapshot averages Bartlett power over valid looks before peak selection. `angle_convention` is static and defaults to `radar_boresight`.

## Responsibilities

- Computes a Bartlett power scan and its dominant direction for every input item.
- Owns scan-grid steering, optional snapshot masking, peak selection, and optional interpolation.
- Does not estimate covariance model order or separate multiple coherent sources.

## When to use

Use as the default sparse angle estimator after `SnapshotGatherOp` when one dominant direction per item and a scan spectrum are useful at conventional-beamforming cost. Choose MUSIC, MVDR, ESPRIT, or Root-MUSIC when closely spaced sources, adaptive interference rejection, or gridless estimation justify their extra assumptions and cost. Do not use it to build a dense angle cube; that is `PhaseShiftBeamformerOp`.

```text
SnapshotGatherOp.snapshot [+ is_valid] → BeamscanEstimatorOp
  → aoa_estimates [+ angle_spectrum] → FeatureAssemblyOp
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels)` or `(num_items, num_channels, num_snapshots)` | Required trigger/aligned input | One or more looks per item. `num_channels` must match `array_geometry`. |
| in | `is_valid` | bool or uint8 | `(num_items, num_snapshots)` | Optional Exact side input | Optional same-tick mask. All-zero rows are blanked to NaN AOA / zero spectrum. |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Required output | Dominant DOA `(az, el)` or `(u, v)`, row-aligned with items. Always emitted. |
| out | `angle_spectrum` | float32 / float64 | `(num_items, num_az, num_el)` | Optional `always` output | Bartlett power grid. Optional (`always` disposition): the grid is the same pass that yields `aoa_estimates`, so it is produced regardless and an unwired sink is silently discarded. No `doa_multi` / `eigenvalues`. |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. |
| live | `az_coordinates` / `el_coordinates` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; extents bounded by `max_az` / `max_el`. |
| live | `array_taper` | `WireWindowSpec` | descriptor wire object | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. |
| live | `peak_interpolation` | `PeakInterpolation` `{none, parabolic}` | scalar enum | Optional Exact one-tick override; Params fallback on miss | `[live]` Exact override; always declared, wire to enable. |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_items`, `max_az`, `max_el` | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `cache_budget_bytes` | unsigned bytes | 67108864 (64 MiB) | static | Instance-wide retained cache ceiling; 0 disables retention. Oversized entries execute uncached with a warning. |
| `array_geometry` | `ArrayGeometry` | none (required; empty fails `validate`) | `[live]` Exact wire-schema override | Element positions + wavelength. Cost: Rebuilds cached steering weights. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Boundary labeling of the grid and `aoa_estimates`, or `direction_cosine`. Cost: Rebuilds weights (part of the cache key). |
| `az_coordinates` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_az` | First coordinate: azimuth radians for angular conventions, dimensionless `u` for `direction_cosine`. Cost: Rebuilds weights. |
| `el_coordinates` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_el` | Second coordinate: elevation radians for angular conventions, dimensionless `v` for `direction_cosine`; `[0]` gives a 1-D scan. Cost: Rebuilds weights. |
| `array_taper` | `WindowSpec` | `kNone` (rectangular) | `[live]` Exact wire-schema override | Spatial amplitude taper; non-custom windows are separable on recognized filled URAs, while custom coefficients stay per-element. Cost: Rebuilds weights. |
| `peak_interpolation` | `PeakInterpolation` | `none` | `[live]` Exact | `none` = argmax cell; `parabolic` = quadratic refinement (2-D interior peaks only). Cost: Free — no weight rebuild. |

Contract: [`OPERATOR_API.md` § `BeamscanEstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md)

## Algorithm

The shell (`beamscan_estimator_op.cu`) receives and validates the snapshot, resolves all five `[live]` values into one tick-local `BeamscanStageConfig`, builds a `BeamscanStagePlan`, resolves cached steering artifacts, prepares both output loans, runs the stage, publishes required `aoa_estimates` before optional/always `angle_spectrum`, and retains borrowed inputs. The cached artifact holds conjugated steering weights plus device azimuth/elevation grids; weights use `BeamOrder::kElMajor` (`beam = ie * num_az + ia`), `SteeringLayout::kChannelMajor` (`(num_channels, num_beams)`), and `conjugate = true` — the transpose of PhaseShift's az-major / beam-major table so the GEMM is `X · Wᶜ`.

The one-look stage (`beamscan_stage.hpp`):

```text
P[i, b] = | Σ_c conj(w[b, c]) · x[i, c] |²
Y = X · Wᶜ
X  = snapshots   [num_items, num_channels]
Wᶜ = conj(w)ᵀ    [num_channels, num_beams]   (invalid beams are zero)
Y  = beamformed  [num_items, num_beams]
```

`beamscan_reduce_kernel` (one block per item) writes `|Y|²` into spectrum layout `spec[ia * num_el + ie]`, then block-reduces the argmax. Tie-break: **lowest beam index** (matching an ascending serial scan). All-zero / no-peak items write NaN AOA (`T{0}/T{0}`).

`none` returns the argmax cell's `(az_grid[ia], el_grid[ie])`. `parabolic` fits a 1-D quadratic on each axis around that cell:

```text
parabolic_offset(left, center, right) =
    0.5 * (left - right) / (left - 2·center + right)     # 0 if |denom| < 1e-12
az += off_az * (az_step_left + az_step_right) / 2
el += off_el * (el_step_left + el_step_right) / 2
```

Each axis refines independently when it has at least three grid points and the peak is interior on that axis. A 1-D azimuth scan (`el_coordinates` size 1) therefore refines azimuth while leaving elevation at its sole grid value.

For `num_snapshots > 1`, `beamscan_multilook_kernel` forms the same beam output for every valid look and writes their mean power before the same deterministic peak selection and interpolation. Steering-weight sign and `direction_cosine` masking match PhaseShift (`k = 2π/λ`, `phase = k (p·dir)`, invalid `u²+v² > 1` → zero weights). On the one-look path, `is_valid` does not change the GEMM; `blank_empty_beamscan_items` overwrites items whose mask row is all zero.

## Shape and dtype rules

- `aoa_estimates` rows = `snapshot` axis 0 (`num_items`).
- `angle_spectrum` = `(num_items, az_coordinates.size(), el_coordinates.size())`.
- A 2-D snapshot aliases the one-look layout; a 3-D snapshot requires a positive snapshot extent.
- No leading batch beyond `(num_items, …)`.
- Precision is the template instantiation (complex in, real out).
- `is_valid` accepts `bool` or `uint8` (8-bit, 1 lane), shape `(num_items, num_snapshots)`.
- Inputs must be CUDA-accessible and compact row-major. Outputs are device tensors.
- `[live] peak_interpolation` does not change shapes. A grid / geometry / taper change (override or `set_params`) misses the steering cache, rebuilds weights, and can change `num_az` / `num_el`. Downstream re-infers.

Rejected combinations and the operator's own messages:

| Condition | Message |
| --- | --- |
| empty `az_coordinates` or `el_coordinates` | `BeamscanEstimatorOp: az_coordinates and el_coordinates must be non-empty` |
| input or scan grid beyond captured maxima | `BeamscanEstimatorOp: input or scan grid exceeds MaxTensorExtents` |
| `snapshot` rank not 2 or 3 | `BeamscanEstimatorOp: snapshot must have shape (num_items, num_channels) or (num_items, num_channels, num_snapshots)` |
| 3-D `num_snapshots <= 0` | `BeamscanEstimatorOp: snapshot num_snapshots must be positive` |
| channel count mismatch | `BeamscanEstimatorOp: snapshot num_channels must match array_geometry.num_elements` |
| `is_valid` not bool/uint8 | `BeamscanEstimatorOp: is_valid must have dtype bool or uint8` |
| `is_valid` shape mismatch | `BeamscanEstimatorOp: is_valid must have shape (num_items, num_snapshots) matching snapshot` |
| invalid geometry / non-contiguous / pageable / dtype / no stream | same `ArrayGeometry` / `require_*` / `no CUDA compute stream` wording, prefixed `BeamscanEstimatorOp` |

## Performance

Cold / first unseen steering key: device weight/grid build (same weight helper as PhaseShift). One-look steady processing uses a cache hit, `X · Wᶜ` GEMM, reduction and optional blanking. Multiple looks use the ordered per-beam mean-power kernel. Float stages the snapshot plane in block-local shared memory when `num_channels * num_snapshots <= 2048` and total dynamic shared storage fits 32 KiB; larger planes and double use global loads. The channel/look accumulation and deterministic peak rules stay the same. `peak_interpolation` does not rebuild the artifact; a grid / geometry / taper override does.

A local Thor HSDK 5 engineering check used five warmups, 50 steady frames, three screen pairs and three fresh pairs. For baseline-large float (16 channels, 1024 items, nine looks, 61×31 beams), fresh compute medians changed 2.746724 → 2.545900 ms: paired saving 0.199546 ms (7.268%), with pair gains 7.247–7.406%. All eleven other case/precision endpoints in the six-case check met the declared median/pair risk allowance of max(5 microseconds, 5%). Both builds passed 47 native tests and exact comparisons under both public-output policies across seven cases, in both precisions, including minimal optional wiring. These results describe the measured workloads.

Startup and tails are separate tradeoffs: fresh medium-float cold latency increased 0.712 ms, optional-minimal double setup increased 1.217 ms, large-double p99 increased 0.080 ms and small-float p99 increased 0.008 ms. The cached interpolation event changed 2.757 → 2.556 ms in float and 63.189 → 63.196 ms in double. Fifty-frame tails are descriptive; the retained benefit targets streaming large float workloads and establishes no startup or tail guarantee.

Harness: `performance/operators/angle_estimation/beamscan_estimator/adapter.cpp` (Graph). Size ladder `xs`…`xl` (channels × detections × scan grid). Every selected size runs five isolated inline-override scenarios and reports `override_ms` / `override_delta_ms` for geometry, azimuth grid, elevation grid, taper, and interpolation. Optional `is_valid` is omitted from performance runs. No reference results are checked in; run:

Run this performance smoke **inside a development container**, after
configuring the named build tree with performance enabled (see the
[developer guide](../../../DEVELOPER.md#performance-characterization)):

```bash
ctest --test-dir build -R perf_beamscan_estimator_smoke
```

## Usage

```yaml
beamscan_estimator:
  angle_convention:
    preset: radar_boresight
  az_coordinates: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  el_coordinates: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  peak_interpolation: parabolic
  # array_geometry set in C++ from the RX / virtual array
```

```cpp
auto beamscan = graph.op<BeamscanEstimatorOp<float>>(
    "beamscan_estimator",
    MaxTensorExtents{/* max_items, max_az, max_el */});
beamscan.set_params(cfg.beamscan_params);

graph.add_flow(gather->snapshot, beamscan->snapshot, q);
graph.add_flow(gather->is_valid, beamscan->is_valid, q);  // optional
graph.add_flow(beamscan->aoa_estimates, feature_assembly->aoa_estimates, q);
// Optional angle_spectrum may be wired to a consumer or left unwired.
```
