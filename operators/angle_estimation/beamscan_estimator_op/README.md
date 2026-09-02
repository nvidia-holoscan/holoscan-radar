# `BeamscanEstimatorOp`

Conventional (Bartlett) sparse-path DOA: one look per item, scan a grid, emit the dominant `(az, el)` (or `(u, v)`). It fills the shared sparse-estimator slot after `SnapshotGatherOp` (or `SubarraySelectionOp`). Beamscan is single-snapshot — a 3-D `snapshot` with `num_snapshots ≠ 1` is an error; the 2-D form is canonical. `angle_convention` is static and required.

| Kind | Port | Dtype | Shape | Notes |
| --- | --- | --- | --- | --- |
| in | `snapshot` | complex64 / complex128 | `(num_items, num_channels)` or `(num_items, num_channels, 1)` | One `(num_channels,)` look per item. `num_channels` must match `array_geometry`. |
| in | `is_valid` | bool or uint8 | `(num_items, num_snapshots)` | Optional. Omitted from `contract()` ([EA HSDK5 ONLY] no optional input — `HSDK5_PORT.md` §11). All-zero rows are blanked to NaN AOA / zero spectrum. |
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
group: [`operators/angle_estimation/README.md`](../README.md) ·
open items: [`OPEN_ITEMS.md`](OPEN_ITEMS.md)

## Algorithm

The shell (`beamscan_estimator_op.cu`) validates the snapshot (and optional mask), resolves the `[live]` `peak_interpolation`, and fetches conjugated steering weights from a `ConfigCache` (`steering_cache_key`). Weights are `build_steering_weights_device` with `BeamOrder::kElMajor` (`beam = ie * num_az + ia`), `SteeringLayout::kChannelMajor` (`(num_channels, num_beams)`), `conjugate = true` — the transpose of PhaseShift's az-major / beam-major table so the GEMM is `X · Wᶜ`.

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

Refinement runs only when `num_az >= 3 && num_el >= 3` and the peak is strictly interior on **both** axes. A 1-D azimuth scan (`el_angles` size 1) therefore never refines — see Contract conformance.

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
| `angle_convention` | `AngleConvention` | contract: none (required). Params default-construct to `radar_boresight` — see conformance | static | Rebuilds weights (part of the cache key) | Boundary labeling of the grid and `aoa_estimates`, or `direction_cosine`. |
| `az_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_az` | Rebuilds weights | Azimuth (`u`) scan grid. |
| `el_angles` | `std::vector<double>` | none (required; empty rejected) | `[live]` Exact host config Tensor, extent bounded by `max_el` | Rebuilds weights | Elevation (`v`) scan grid; `[0]` is a 1-D azimuth scan. |
| `array_taper` | `WindowSpec` | `kNone` (rectangular) | `[live]` Exact wire-schema override | Rebuilds weights | Spatial amplitude taper. |
| `peak_interpolation` | `PeakInterpolation` | `none` | `[live]` Exact | Free — no weight rebuild | `none` = argmax cell; `parabolic` = quadratic refinement (2-D interior peaks only). |

Schema id `com.nvidia.holoscan.radar.BeamscanParams`, version 1. `peak_interpolation` resolution: Params seed; an Exact match on the override port wins for that tick, and any tick without a match uses the Params value.

## SDK implementation notes

- **Construction.** `BeamscanEstimatorOp(const MaxTensorExtents&)`. **Requires** `max_items`, `max_az`, `max_el`. Sizes `angle_spectrum` as `{max_items, max_az, max_el}` real and `aoa_estimates` as `{max_items, 2}` real. A tick that exceeds those spans fails at loan allocate.
- **Triggering.** `contract()` is `OnEach{snapshot}` plus one unconditional `Exact{}` + `SafeDefault{}` entry per `[live]` override (`array_geometry`, `az_angles`, `el_angles`, `array_taper`, `peak_interpolation`). Wiring is the application's only obligation: an `add_flow` into the port enables the override, and leaving it unwired compiles (warning `TEMPORAL_INPUT_UNCONNECTED_SAFE_DEFAULT`) and reports `kNotConnected` on every activation, which `compute()` reads as no override this tick. A wired port that is *not* in the contract still fails the second activation with `RUNTIME_OPERATOR_REPLAY_MISMATCH` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §1) — which is why optional `is_valid`, omitted from the contract (§11), must stay unwired (OPEN_ITEMS.md §3).
- **Ports and queues.** Inputs `queue_depth(2)`, device, rank-agnostic. `aoa_estimates` `max_rank = 2`; `angle_spectrum` `max_rank = 3`. Live Inputs `queue_depth(2)`; `az_angles` / `el_angles` are rank-1 pageable-host config Tensors, the rest are wire-object / enum payloads.
- **Live override payloads.** `array_geometry` is `Input<WireArrayGeometry>` and `array_taper` is `Input<WireWindowSpec>` — generated descriptor-schema types, converted with `from_wire` inside `resolve_live_override_via` so wire storage never reaches operator state. `az_angles` / `el_angles` ride rank-1 pageable-host float64 config Tensors validated on arrival (rank, host memory, contiguity, dtype, `max_az` / `max_el` extent) and converted per tick, with no state kept ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9). `peak_interpolation` is a plain enum `Input<>` resolved with `resolve_live_override`. The logical Params types are unchanged — the wire / Tensor form is a port detail.
- **Streams and lifetime.** GEMM, reduce, blanking, and weight builds use `context.cuda_stream()`. `snapshot` and `is_valid` (when present) are retained. The hot path does not synchronize. Per non-empty tick the operator also allocates a `beamformed` scratch cube and H2D-copies both scan grids.
- **Emission.** Both loans allocated before either publish; emit `aoa_estimates` then `angle_spectrum`. Zero-item ticks use `prepare_device_tensor` no-ops in the same order.
- **Precision.** `BeamscanEstimatorOp<float>`, `BeamscanEstimatorOp<double>`.
- **[EA HSDK5 ONLY]** All five override ports are unconditionally in `contract()`, and the `peak_interpolation` Graph mid-stream test runs since the enum-edge fix (§12), alongside the descriptor / grid ones. An upstream descriptor output cannot drive `array_geometry`: `VirtualArrayAssemblyOp` / `SubarraySelectionOp` emit a packed float64 Tensor (HSDK5_PORT.md §10) while this port takes the wire-schema type, so the override has to come from an application-side wire producer (§9 / §10). `angle_spectrum` is an `always` optional output (unwired sinks discarded). The steering `ConfigCache` warns once per unseen key after the first (decided inside the cache).

## Consumer lifetime notes

Both outputs are fresh pool loans filled and committed this tick. A consumer may retain them.

## Performance

Cold / first unseen steering key: device weight build (same helper as PhaseShift). Steady: cache hit + `X · Wᶜ` GEMM + reduce kernel + optional blanking. `peak_interpolation` is free; a grid / geometry / taper override costs one steering-weight rebuild on the tick it arrives. Inline live-override timing is deferred (HSDK5_PORT.md §1 / §8), which now also covers `peak_interpolation` — its edge works, only the perf harness skips enum sidebands.

Harness: `performance/cpp/perf_beamscan_estimator.cpp` (Graph; OPEN_ITEMS.md §5 is stale). Size ladder `xs`…`xl` (channels × detections × scan grid). Optional `is_valid` omitted. No in-repo measured numbers; run:

```bash
ctest --test-dir build -R perf_beamscan_estimator_smoke
```

## Tests and verification

C++: `tests/cpp/angle_estimation/test_beamscan_estimator.cpp`. Python: `tests/python/angle_estimation/test_beamscan_estimator.py` (HSDK 4.x bindings; not built — OPEN_ITEMS.md §4). Ground truth is `make_steering_snapshot` / ULA / URA builders — an analytic `a(θ)`, not the point-target ADC simulator. Tolerances 0.01–0.02 rad on DOA recovery; `FloatDoubleAgreeOnDoa` uses `2e-3`.

| Check | Proves |
| --- | --- |
| `RecoversKnownDoa2D` / `RecoversKnownDoaAzOnly1DScan` | Single-source recovery on URA / ULA |
| `Accepts3DSnapshotWithOneLook` / `RejectsMultiSnapshotInput` | 3-D `K=1` accepted; `K≠1` rejected |
| `RecoversKnownDoaPhysicsSpherical` / `RecoversKnownDoaAerospaceNed` | Convention relabeling |
| `DirectionCosineMasksInvisibleRegion` | Invisible `(u, v)` → zero spectrum |
| `PeakInterpolationNoneReturnsGridCell` / `PeakInterpolationConfigurableFromParams` | `none` = grid node; `parabolic` moves off-grid (2-D `el` size 3) |
| `ArrayTaperConfigurableFromParams` | Taper changes the spectrum |
| `MultipleItemsRecoverIndependentDoa` / `EmptyItemsProducesEmptyOutputs` / `SpectrumShapeMatchesScanGrid` | Batch, empty, shape |
| `FloatInstantiationSmoke` / `FloatDoubleAgreeOnDoa` | Both precisions |
| `RejectsChannelMismatch` / `RejectsEmptyScanGrid` / `RejectsInvalidSnapshotRank` / `RejectsSnapshotDtypeMismatch` | Named error paths |
| `CheckIsValidMask*` | Mask dtype/shape/pageable validation (direct helper, not Graph) |
| `LiveAzGridAppliesForOneTick` / `LiveElGridAppliesForOneTick` / `LiveArrayGeometryAppliesForOneTick` / `LiveArrayTaperAppliesForOneTick` | Mid-stream override reshapes / rebuilds, and sticks after the next miss |
| `LiveOverridesRejectMalformedAndBounds` | Malformed wire payload and an over-`max_el` grid are diagnosed, not consumed |
| `LivePeakInterpolationAppliesForOneTick` / `AcceptsValidIsValidMask*` / `RejectsPageableHostSnapshot` | [EA HSDK5 ONLY] `GTEST_SKIP`'d |

TESTING.md's connected `is_valid` exclude-masked-snapshots check is owed and skipped on Graph (helper covers transport only).

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
graph.add_flow(beamscan->aoa_estimates, feature_assembly->aoa_estimates, q);
// [EA HSDK5 ONLY] angle_spectrum must be bound; is_valid into Beamscan
// is deferred — apps drain gather->is_valid to a NullSink.
graph.add_flow(beamscan->angle_spectrum, spectrum_sink->input, q);
```

Python bindings exist at `python/_beamscan_estimator_op_bindings.cpp` as an HSDK 4.x trampoline and are not built.

## Contract conformance

Matches the shared sparse slot plus Beamscan's single-snapshot / spectrum-only surface, with the [EA HSDK5 ONLY] interims below. Two algorithm notes diverge from the locked prose.

| Contract claim | Verdict | Detail |
| --- | --- | --- |
| `array_geometry` / `array_taper` **[live]** | conforms (payload form differs) | Unconditional Exact `Input<WireArrayGeometry>` / `Input<WireWindowSpec>` in `contract()`, not `Input<ArrayGeometry>` / `Input<WindowSpec>` ([`HSDK5_PORT.md`](../../../docs/planning/HSDK5_PORT.md) §9, OPEN_ITEMS.md §2). Graph mid-stream + revert-to-Params covered. |
| `az_angles` / `el_angles` **[live]** | conforms (payload form differs) | Rank-1 host config Tensors bounded by `max_az` / `max_el`, not `Input<std::vector<double>>` (§9). Graph mid-stream + revert-to-Params covered. |
| `peak_interpolation` **[live]** | conforms | Unconditional Exact `Input<>` in `contract()`; the app enables the override by wiring the port. Covered by `LivePeakInterpolationAppliesForOneTick` since the enum-edge fix (§12). |
| `is_valid` optional Graph wiring | EA-deferred | `receive_optional_value` + blanking implemented; port omitted from `contract()` (§11); connected Graph tests skipped (OPEN_ITEMS.md §3). |
| `angle_spectrum` optional | Done | `always` disposition: fused with the argmax that yields `aoa_estimates`, so an unwired sink costs nothing and is discarded |
| `angle_convention` required, no default | deviates | `BeamscanParams` default-constructs `AngleConvention{}` → `kRadarBoresight` (`beamscan_estimator_op.hpp` line 51). |
| `parabolic` sub-grid refinement | deviates | Requires `num_az >= 3 && num_el >= 3` and an interior peak on both axes (`beamscan_stage.hpp` lines 135–137). 1-D scans (`el_angles = [0]`) never refine. |
| Steering cache identity | deviates | Shared `steering_cache_key` hashes only `convention.preset` (`steering_weights.cpp` lines 23–24) — same as PhaseShift. |

Static audit of `beamscan_estimator_op.hpp`, `beamscan_estimator_op.cu`, `beamscan_stage.hpp`, `is_valid_mask.{hpp,cpp}`, `metadata.json`, `python/_beamscan_estimator_op_bindings.cpp`, `OPEN_ITEMS.md`, `tests/cpp/angle_estimation/test_beamscan_estimator.cpp`, `performance/cpp/perf_beamscan_estimator.cpp`, and the planning docs listed in the skill. No build or test run.
