# `BeamscanEstimatorOp`

Conventional (Bartlett) sparse-path DOA: one look per item, scan a grid, emit the dominant `(az, el)` (or `(u, v)`). It fills the shared sparse-estimator slot after `SnapshotGatherOp` (or `SubarraySelectionOp`). Beamscan is single-snapshot — a 3-D `snapshot` with `num_snapshots ≠ 1` is an error; the 2-D form is canonical. `angle_convention` is static and defaults to `radar_boresight`.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels)` or `(num_items, num_channels, 1)` | One `(num_channels,)` look per item. `num_channels` must match `array_geometry`. |
| in | `is_valid` | bool or uint8 | `(num_items, num_snapshots)` | Optional same-tick mask. All-zero rows are blanked to NaN AOA / zero spectrum. |
| out | `aoa_estimates` | float32 / float64 | `(num_items, 2)` | Dominant DOA `(az, el)` or `(u, v)`, row-aligned with items. Always emitted. |
| out | `angle_spectrum` | float32 / float64 | `(num_items, num_az, num_el)` | Bartlett power grid. Optional (`always` disposition): the grid is the same pass that yields `aoa_estimates`, so it is produced regardless and an unwired sink is silently discarded. No `doa_multi` / `eigenvalues`. |
| live | `array_geometry` | `WireArrayGeometry` | descriptor wire object | `[live]` Exact override; always declared, wire to enable. |
| live | `az_angles` / `el_angles` | float64 config Tensor | `(num_az,)` / `(num_el,)`, rank-1 pageable host | `[live]` Exact override; extents bounded by `max_az` / `max_el`. |
| live | `array_taper` | `WireWindowSpec` | descriptor wire object | `[live]` Exact override; always declared, wire to enable. |
| live | `peak_interpolation` | `PeakInterpolation` `{none, parabolic}` | scalar enum | `[live]` Exact override; always declared, wire to enable. |

```text
SnapshotGatherOp.snapshot [+ is_valid] → BeamscanEstimatorOp
  → aoa_estimates [+ angle_spectrum] → FeatureAssemblyOp
```

Contract: [`OPERATOR_API.md` § `BeamscanEstimatorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/angle_estimation/README.md`](../README.md)

## Algorithm

The shell (`beamscan_estimator_op.cu`) receives and validates the snapshot, resolves all five `[live]` values into one tick-local `BeamscanStageConfig`, builds a `BeamscanStagePlan`, resolves cached steering artifacts, prepares both output loans, runs the stage, publishes required `aoa_estimates` before optional/always `angle_spectrum`, and retains borrowed inputs. The cached artifact holds conjugated steering weights plus device azimuth/elevation grids; weights use `BeamOrder::kElMajor` (`beam = ie * num_az + ia`), `SteeringLayout::kChannelMajor` (`(num_channels, num_beams)`), and `conjugate = true` — the transpose of PhaseShift's az-major / beam-major table so the GEMM is `X · Wᶜ`.

The stage (`beamscan_stage.hpp`):

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

Each axis refines independently when it has at least three grid points and the peak is interior on that axis. A 1-D azimuth scan (`el_angles` size 1) therefore refines azimuth while leaving elevation at its sole grid value.

Steering-weight sign and `direction_cosine` masking match PhaseShift (`k = 2π/λ`, `phase = k (p·dir)`, invalid `u²+v² > 1` → zero weights). `is_valid` does not change the GEMM; `blank_empty_beamscan_items` then overwrites items whose mask row is all zero.

## Shape and dtype rules

- `aoa_estimates` rows = `snapshot` axis 0 (`num_items`).
- `angle_spectrum` = `(num_items, az_angles.size(), el_angles.size())`.
- 2-D snapshot is canonical; 3-D is legal only with `num_snapshots == 1` (aliases the same contiguous layout).
- No leading batch beyond `(num_items, …)`.
- Precision is the template instantiation (complex in, real out).
- `is_valid` accepts `bool` or `uint8` (8-bit, 1 lane), shape `(num_items, num_snapshots)`.
- Inputs must be CUDA-accessible and compact row-major. Outputs are device tensors.
- `[live] peak_interpolation` does not change shapes. A grid / geometry / taper change (override or `set_params`) misses the steering cache, rebuilds weights, and can change `num_az` / `num_el`. Downstream re-infers.

Rejected combinations and the operator's own messages:

| Condition | Message |
| --- | --- |
| empty `az_angles` or `el_angles` | `BeamscanEstimatorOp: az_angles and el_angles must be non-empty` |
| input or scan grid beyond captured maxima | `BeamscanEstimatorOp: input or scan grid exceeds MaxTensorExtents` |
| `snapshot` rank not 2 or 3 | `BeamscanEstimatorOp: snapshot must have shape (num_items, num_channels) or (num_items, num_channels, num_snapshots)` |
| 3-D `num_snapshots ≠ 1` | `BeamscanEstimatorOp: snapshot num_snapshots must be 1 for single-snapshot Beamscan` |
| channel count mismatch | `BeamscanEstimatorOp: snapshot num_channels must match array_geometry.num_elements` |
| `is_valid` not bool/uint8 | `BeamscanEstimatorOp: is_valid must have dtype bool or uint8` |
| `is_valid` shape mismatch | `BeamscanEstimatorOp: is_valid must have shape (num_items, num_snapshots) matching snapshot` |
| invalid geometry / non-contiguous / pageable / dtype / no stream | same `ArrayGeometry` / `require_*` / `no CUDA compute stream` wording, prefixed `BeamscanEstimatorOp` |

## Parameters

| Parameter | Type | Default | Override | Cost | Meaning |
| --- | --- | --- | --- | --- | --- |
| `array_geometry` | `ArrayGeometry` | none (required; empty fails `validate`) | `[live]` Exact wire-schema override | Rebuilds cached steering weights | Element positions + wavelength. |
| `angle_convention` | `AngleConvention` | `radar_boresight` | static | Rebuilds weights (part of the cache key) | Boundary labeling of the grid and `aoa_estimates`, or `direction_cosine`. |
| `az_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_az` | Rebuilds weights | Azimuth (`u`) scan grid. |
| `el_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_el` | Rebuilds weights | Elevation (`v`) scan grid; `[0]` is a 1-D azimuth scan. |
| `array_taper` | `WindowSpec` | `kNone` (rectangular) | `[live]` Exact wire-schema override | Rebuilds weights | Spatial amplitude taper; non-custom windows are separable on recognized filled URAs, while custom coefficients stay per-element. |
| `peak_interpolation` | `PeakInterpolation` | `none` | `[live]` Exact | Free — no weight rebuild | `none` = argmax cell; `parabolic` = quadratic refinement (2-D interior peaks only). |

Application YAML accepts explicit arrays or inclusive `{min: ..., max: ..., step: ...}` mappings for `az_angles` and `el_angles`. Range mappings require finite values, `step > 0`, `min <= max`, and endpoint alignment within floating-point tolerance. They expand to the same `std::vector<double>` Params before construction; live ports remain rank-1 float64 Tensors.

Schema id `com.nvidia.holoscan.radar.BeamscanParams`, version 1. `peak_interpolation` resolution: Params seed; an Exact match on the override port wins for that tick, and any tick without a match uses the Params value.

## SDK implementation notes

- **Construction.** `BeamscanEstimatorOp(const MaxTensorExtents&)`. **Requires** `max_items`, `max_az`, `max_el`. Sizes `angle_spectrum` as `{max_items, max_az, max_el}` real, `aoa_estimates` as `{max_items, 2}` real, and a reusable complex GEMM workspace as `{max_items, max_az, max_el}`. The plan rejects a tick beyond those bounds before requesting an output loan.
- **Triggering.** `contract()` is `OnEach{snapshot}`. Optional `is_valid` and each `[live]` override (`array_geometry`, `az_angles`, `el_angles`, `array_taper`, `peak_interpolation`) are unconditional non-member `Exact{}` + `SafeDefault{}` entries. Producers publish a same-stamped mask before `snapshot`, so Exact sees it when the trigger arrives. Wiring enables a sideband for one tick; a miss uses all-valid behavior or Params.
- **Ports and queues.** Inputs `queue_depth(2)`, device, rank-agnostic. `aoa_estimates` `max_rank = 2`; `angle_spectrum` `max_rank = 3`. Live Inputs `queue_depth(2)`; `az_angles` / `el_angles` are rank-1 pageable-host config Tensors, the rest are wire-object / enum payloads.
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>` and `array_taper` is `Input<WireWindowSpec>` — generated descriptor-schema types, converted with `from_wire` inside `resolve_live_override_via` so wire storage never reaches operator state. `az_angles` / `el_angles` ride rank-1 pageable-host float64 config Tensors validated on arrival (rank, host memory, contiguity, dtype, `max_az` / `max_el` extent) and converted per tick, with no state kept ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). `peak_interpolation` is a plain enum `Input<>` resolved with `resolve_live_override`. The logical Params types are unchanged — the wire / Tensor form is a port detail.
- **Streams and lifetime.** GEMM, reduce, blanking, and steering-artifact builds use `context.cuda_stream()`. `snapshot` and `is_valid` (when present) are retained. The hot path does not synchronize. The constructor-created GEMM workspace is reused; device scan grids are cached with the steering weights, so unchanged ticks do not allocate scratch or repeat grid H2D copies.
- **Emission.** Both loans are prepared before the shared stage writes either buffer; required `aoa_estimates` publishes before optional/always `angle_spectrum`. Zero-item ticks use no-op preparations in the same order.
- **Precision.** `BeamscanEstimatorOp<float>`, `BeamscanEstimatorOp<double>`.
- **[EA HSDK5 ONLY]** All five override ports are unconditionally in `contract()`, and Graph mid-stream tests cover each override plus reversion. Runtime geometry producers connect directly to `array_geometry` with `WireArrayGeometry`; Python remains Params-only for schema payloads. Optional `is_valid` Graph wiring is active. `angle_spectrum` is an `always` optional output (unwired sinks discarded). The steering-artifact `ConfigCache` warns once per unseen key after the first.

## Consumer lifetime notes

Both outputs are fresh pool loans filled and committed this tick. A consumer may retain them.

## Performance

Cold / first unseen steering key: device weight/grid build (same weight helper as PhaseShift). Steady: cache hit + `X · Wᶜ` GEMM + reduce kernel + optional blanking. `peak_interpolation` does not rebuild the artifact; a grid / geometry / taper override does.

Harness: `performance/cpp/perf_beamscan_estimator.cpp` (Graph). Size ladder `xs`…`xl` (channels × detections × scan grid). Every selected size runs five isolated inline-override scenarios and reports `override_ms` / `override_delta_ms` for geometry, azimuth grid, elevation grid, taper, and interpolation. Optional `is_valid` is omitted from performance runs. No current HSDK 5 Graph results are checked in; run:

```bash
ctest --test-dir build -R perf_beamscan_estimator_smoke
```

## Tests and verification

C++: `tests/cpp/angle_estimation/test_beamscan_estimator.cpp`. Python: `tests/python/angle_estimation/test_beamscan_estimator.py` (legacy HSDK 4.x bindings; not built on this branch under the module-wide Python deferral). Ground truth is `make_steering_snapshot` / ULA / URA builders — an analytic `a(θ)`, not the point-target ADC simulator. Tolerances 0.01–0.02 rad on DOA recovery; `FloatDoubleAgreeOnDoa` uses `2e-3`.

| Check | Proves |
| --- | --- |
| `DefaultsToRadarBoresightConvention` | Default-constructed Params select the documented `radar_boresight` convention |
| `RecoversKnownDoa2D` / `RecoversKnownDoaAzOnly1DScan` | Single-source recovery on URA / ULA |
| `Accepts3DSnapshotWithOneLook` / `RejectsMultiSnapshotInput` | 3-D `K=1` accepted; `K≠1` rejected |
| `RecoversKnownDoaPhysicsSpherical` / `RecoversKnownDoaAerospaceNed` | Convention relabeling |
| `DirectionCosineMasksInvisibleRegion` | Invisible `(u, v)` → zero spectrum |
| `PeakInterpolationNoneReturnsGridCell` / `PeakInterpolationConfigurableFromParams` | `none` = grid node; `parabolic` moves off-grid (2-D `el` size 3) |
| `ArrayTaperConfigurableFromParams` | Taper changes the spectrum |
| `MultipleItemsRecoverIndependentDoa` / `EmptyItemsProducesEmptyOutputs` / `SpectrumShapeMatchesScanGrid` / `OptionalSpectrumMayBeLeftUnwired` | Batch, empty, shape, optional-output disposition |
| `FloatInstantiationSmoke` / `FloatDoubleAgreeOnDoa` | Both precisions |
| `RejectsChannelMismatch` / `RejectsEmptyScanGrid` / `RejectsInvalidSnapshotRank` / `RejectsSnapshotDtypeMismatch` / `RejectsItemsBeyondCapturedExtent` | Named error and captured-bound paths |
| `CheckIsValidMask*` | Mask dtype/shape/pageable validation (direct helper, not Graph) |
| `LiveAzGridAppliesForOneTick` / `LiveElGridAppliesForOneTick` / `LiveArrayGeometryAppliesForOneTick` / `LiveArrayTaperAppliesForOneTick` | Mid-stream override reshapes / rebuilds, then the next miss reverts to Params |
| `LiveOverridesRejectMalformedAndBounds` | Malformed wire payload and an over-`max_el` grid are diagnosed, not consumed |
| `LivePeakInterpolationAppliesForOneTick` | Enum override applies for one tick and the next miss reverts |
| `AcceptsValidIsValidMaskOn3DSnapshot` / `RejectsInvalidIsValidMaskOn3DSnapshot` | Connected optional mask blanks an excluded item and rejects malformed shape |

```bash
ctest --test-dir build -R test_beamscan_estimator --output-on-failure
```

## Usage

```yaml
beamscan_estimator:
  angle_convention:
    preset: radar_boresight
  az_angles: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  el_angles: [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
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

Python bindings exist at `python/_beamscan_estimator_op_bindings.cpp` as an HSDK 4.x trampoline and are not built.

## Contract conformance

The implementation matches the locked single-snapshot sparse-estimator contract. Physical live payloads follow the module-wide HSDK 5 schema/host-Tensor policy; the logical Params types are unchanged. Optional `is_valid` and `angle_spectrum` may be left unwired. Parabolic interpolation works independently on each scan axis, including 1-D azimuth grids, and the shared steering key covers every custom `AngleConvention` field.
