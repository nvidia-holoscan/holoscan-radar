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
| `SubarraySelectionOp<T>` | `snapshot` + `array_geometry` → `snapshot`, `array_geometry` | Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual array (S3) — unblocks the gridless subspace estimators and holey-array spatial smoothing |
| `BeamscanEstimatorOp<T>` | `snapshot` [+ `is_valid`] → `aoa_estimates`, `angle_spectrum` | Conventional (Bartlett) sparse-path DOA on a scan grid |
| `MVDREstimatorOp<T>` | `snapshot` [+ `is_valid`] → `aoa_estimates`, `angle_spectrum`, `doa_multi`, `source_count`, `eigenvalues` | Adaptive (MVDR / Capon) sparse-path DOA on a scan grid (multi-snapshot covariance); multi-source + eigenvalue diagnostics |
| `MUSICEstimatorOp<T>` | `snapshot` [+ `is_valid`] → `aoa_estimates`, `angle_spectrum`, `doa_multi`, `source_count`, `eigenvalues` | Subspace (MUSIC) sparse-path DOA on a scan grid; per-item model order fixed or auto (EFT); noise-subspace pseudospectrum |
| `RootMUSICEstimatorOp<T>` | `snapshot` [+ `is_valid`] → `aoa_estimates`, `doa_multi`, `source_count`, `eigenvalues` | Gridless Root-MUSIC on a ULA; polynomial roots (no scan grid / no spectrum) |
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
- **`SubarraySelectionOp` (checklist-complete — deustice):** selects a uniform
  ULA/URA sub-array from a (possibly holey) MIMO virtual array (S3) — auto
  picks the largest filled sub-block (holey-lattice detection + maximal
  rectangle) or an explicit index set is validated; re-emits a reduced
  `(snapshot, array_geometry)` pair. Unblocks the gridless subspace
  estimators (Root-MUSIC/ESPRIT) and the holey-array path for spatial
  smoothing. Four forks left to ratify (see `TEAM_BLOCKED.md` L14); gridless
  consumers not yet built, so verified end-to-end only through MVDR on a
  reduced array. See `subarray_selection_op/OPEN_ITEMS.md`.
- **`MVDREstimatorOp` (checklist-complete — dpc):**
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
  `performance/cpp/perf_mvdr_estimator.cpp`. `spatial_smoothing` /
  `forward_backward` are implemented as a front-end transform (settled — S1,
  STATUS.md L13); see `mvdr_estimator_op/OPEN_ITEMS.md` for the residual notes
  (the `diagonal_loading` float-or-`auto` encoding note and the eigenvalues
  per-tick cost).
- **`MUSICEstimatorOp` (checklist-complete, C++ and Python — dpc):** subspace
  (MUSIC) sparse-path DOA. Forms the per-item sample covariance, a batched
  Hermitian eigendecomposition (`matx::eig`, `EigenMode::VECTOR` — ascending
  eigenvalues, column eigenvectors), resolves a per-item model order
  (`num_sources`: a fixed integer, or `auto` = **EFT** on the eigenvalues via
  the shared `radar_common/dsp/source_count` helper, `source_count_pfa`
  **[live]** sets the target rate), builds the noise-subspace projector, and
  scans for the MUSIC pseudospectrum peak (structurally the same `aᴴMa`
  reduce as MVDR's Capon kernel, with the projector standing in for `R̂⁻¹` —
  but a near-zero denominator is MUSIC's genuine sharp peak, so it is
  floored rather than gated to zero, unlike Capon). `max_sources` **[live]**
  - `doa_multi`/`source_count` target `min(num_sources_i, max_sources)`
  peaks (not always `max_sources`, since MUSIC — unlike MVDR — has an actual
  model order) + `eigenvalues`; `spatial_smoothing`/`forward_backward`
  reused verbatim from MVDR; float/double; 40 C++ unit tests (analytic DOA
  recovery on ULA/URA, super-resolution vs. Beamscan per `TESTING.md`,
  `num_sources` fixed/auto, multi-source + `max_sources` capping, spatial
  smoothing (1-D/2-D, forward/forward-backward), rank-deficient auto
  fallback, `is_valid` masking, multi-item batches, live overrides
  (all live parameters), CUDA memory spaces, determinism, and error paths)
  - 38 Python behavioral tests (the pinned/managed DLPack transport checks
  remain C++-only; `is_valid` input works via `uint8`);
  `performance/cpp/perf_music_estimator.cpp` +
  `performance/python/perf_music_estimator.py` (same size ladder as MVDR's,
  inline `[live]` `az_angles` override). Two profile-driven `music_reduce_kernel`
  fixes: (1) it was missing MVDR's 4-way-accumulator ILP unrolling — porting it
  gave -25.8% at `xl` (26.9→19.9 ms); (2) `ncu` then showed the kernel is
  L1-bandwidth-bound (not latency-bound as MVDR's inherited note assumed), so
  `Pn` is now staged in shared memory, gated to grids of ≥1 occupancy wave —
  another -18.7% at `xl` (21.0→17.2 ms) with no `lg` regression, leaving MUSIC
  faster than MVDR at every measured size. See
  `music_estimator_op/OPEN_ITEMS.md` (EFT fidelity note, denominator-floor
  rationale, the full ncu-driven perf writeup incl. a recorded negative result,
  and the `nsys`-in-a-sandbox workaround).
- **`RootMUSICEstimatorOp` (checklist-complete, C++ and Python — dpc):**
  gridless Root-MUSIC on a **ULA** (1-D uniform line in snapshot channel order).
  Reuses the MUSIC covariance / Hermitian eig / EFT / noise-projector front-end;
  forms the Root-MUSIC polynomial, solves roots via companion +
  `cusolverDnXgeev` (one matrix per item), and maps in-unit-circle roots to
  continuous DOAs (no `angle_spectrum`). `spatial_smoothing` /
  `forward_backward` match MVDR/MUSIC; float/double; 36 C++ + 33 Python
  behavioral tests; `performance/cpp/perf_root_music_estimator.cpp` +
  `performance/python/perf_root_music_estimator.py` (inline live overrides for
  geometry, auto/EFT, Pfa, capacity, smoothing). Custom batched device root
  solver tracked in `root_music_estimator_op/OPEN_ITEMS.md`.
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
  Open items in `phase_shift_beamformer_op/OPEN_ITEMS.md` (dense-cube memory
  footprint).
