# Angle-estimation operators

GPU operators for the sparse (and dense) angle paths. Module-wide port,
batching, and parameter-override conventions are in
[`operators/common/README.md`](../common/README.md); each operator header
carries its full port/parameter reference.

Typical sparse-path placement:

```text
… → FMCWDopplerProcessorOp → rd_maps (per-channel)
  → CFARDetectorOp (NCI over channel)
  → SnapshotGatherOp → snapshot → ⟨sparse angle estimator⟩ → aoa_estimates
```

Typical dense-path placement (angle-then-cluster — angle is formed before
detection; the `(az, el)` axes ride as leading batch dims on CFAR's `rd_maps`):

```text
… → FMCWDopplerProcessorOp → rd_maps (per-channel)
  → PhaseShiftBeamformerOp → rda_cube (num_az, num_el, D, R)
  → CFARDetectorOp (on the RDA cube) → detections
```

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| `SnapshotGatherOp<T>` | `rd_maps` + `detection_idx` → `snapshot`, `is_valid` | Per-detection channel-vector gather from the RD neighborhood around each CFAR cell |
| `BeamscanEstimatorOp<T>` | `snapshot` [+ `is_valid`] → `aoa_estimates`, `angle_spectrum` | Conventional (Bartlett) sparse-path DOA on a scan grid |
| `MVDREstimatorOp<T>` | `snapshot` [+ `is_valid`] → `aoa_estimates`, `angle_spectrum`, `doa_multi`, `source_count`, `eigenvalues` | Adaptive (MVDR / Capon) sparse-path DOA on a scan grid (multi-snapshot covariance); multi-source + eigenvalue diagnostics |
| `PhaseShiftBeamformerOp<T>` | `rd_maps` → `rda_cube` | Conventional (Bartlett) dense-path beamformer: a beam for every RD cell → Range-Doppler-Angle cube for CFAR's dense path |

## Implementation status

- **`SnapshotGatherOp` (checklist-complete — dpc):** RD-neighborhood gather for the sparse
  angle path; sticky `[live] snapshot_window`; static `doppler_edge` /
  `range_edge`; float/double complex `rd_maps`; unit tests (edge modes, batch
  dims, error paths, live override); `performance/cpp/perf_snapshot_gather.cpp`
  characterizes gather cost + the `[live] snapshot_window` override spike.
  `is_valid` emits `uint8` (0/1), NumPy-visible on HSDK 4.3, aligned with CFAR
  `detection_mask` (settled — STATUS.md L1).
- **`BeamscanEstimatorOp` (checklist-complete — dpc):** conventional sparse-path
  Bartlett DOA on a scan grid; lands `ArrayGeometry` + `AngleConvention`
  descriptors in `operators/common/`; sticky `[live]` overrides for geometry,
  scan grid, taper, and peak interpolation; float/double; unit tests (2-D DOA
  recovery on a planar URA, 3-D single-look input, direction-cosine masking,
  error paths, live `az_angles` override); `performance/cpp/perf_beamscan_estimator.cpp`
  characterizes scan cost + the `[live] az_angles` override spike. The `custom`
  angle preset (STATUS.md L11) and `uint8` `is_valid` transport (STATUS.md L1)
  are both settled.
- **`MVDREstimatorOp` (NOT checklist-complete — one blocked param — dpc):**
  adaptive (MVDR / Capon) sparse-path DOA. Implements the per-item sample
  covariance with `is_valid` masking, diagonal loading (`auto` = a small
  multiple of the `trace(R̂)/C` noise-floor proxy, or a fixed absolute δ), a
  batched Hermitian inverse (`matx::inv`), the Capon spectrum, and the dominant
  DOA; plus the full multi-source surface — `max_sources` **[live]** with the
  `doa_multi` / `source_count` outputs (pass 0 = Capon argmax matching
  `aoa_estimates`; further peaks = strict 2-D local maxima, NaN-padded) — and the
  `eigenvalues` output
  (ascending spectrum of the raw covariance via batched `matx::eig`). Reuses the
  untapered `build_steering_weights_device` manifold and the `ArrayGeometry` /
  `AngleConvention` machinery from Beamscan; sticky `[live]` overrides for
  geometry, scan grid, `diagonal_loading`, and `max_sources`; static
  `angle_convention`; float/double; the four estimator-specific outputs carry
  `kNone` so a consumer wanting only `aoa_estimates` leaves them unwired. Unit
  tests (analytic URA/ULA DOA recovery, two-source recovery, `max_sources`
  capacity + NaN fill, rank-one eigenvalue dominance, `is_valid` exclusion /
  all-invalid blanking, live
  overrides incl. `max_sources` realloc, error paths) +
  `performance/cpp/perf_mvdr_estimator.cpp`. **The one remaining contract
  parameter, `spatial_smoothing`, is BLOCKED on a covariance-estimator team
  decision** (not declared); see `mvdr_estimator_op/OPEN_ITEMS.md` (also the
  `diagonal_loading` float-or-`auto` encoding note and the eigenvalues per-tick
  cost).
- **`PhaseShiftBeamformerOp` (checklist-complete — dpc):** conventional
  (Bartlett) dense-path beamformer; forms a beam for every range-Doppler cell,
  replacing the channel axis with the `(num_az, num_el)` grid to build the
  Range-Doppler-Angle cube CFAR detects on directly (the *only* dense
  estimator — a dense MVDR/MUSIC cube is prohibitive). Reuses the
  `ArrayGeometry` / `AngleConvention` / `array_taper` steering machinery from
  Beamscan (one broadcast complex GEMM per tick); sticky `[live]` overrides for
  geometry, scan grid, and taper; static `angle_convention`; float/double;
  rank-agnostic leading batch dims; unit tests (dense DOA-bin recovery,
  convention relabeling + direction-cosine masking, taper sidelobes, batch,
  live overrides, error paths); `performance/cpp/perf_phase_shift_beamformer.cpp`
  characterizes dense beamforming cost + the `[live] az_angles` override spike.
  Open items in `phase_shift_beamformer_op/OPEN_ITEMS.md` (`custom` angle
  preset, dense-cube memory footprint).
