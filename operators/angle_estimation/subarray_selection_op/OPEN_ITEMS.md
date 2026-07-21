# SubarraySelectionOp — open items

Status: **first pass — resolves the S3 interface; four team forks open.**

Selects a uniform ULA/URA sub-array from a (possibly holey) MIMO virtual array
and re-emits a reduced `(snapshot, array_geometry)` pair, so the gridless
estimators (Root-MUSIC / ESPRIT) and holey-array spatial smoothing (S1) can run.
Selection is host-planned (`radar_common/dsp/subarray_selection`) and cached; the
per-tick channel gather reuses the S1 gather (`run_spatial_smoothing_gather`,
single sub-array, no forward-backward).

- **Auto (default):** largest filled ULA (longest run) / URA (maximal all-filled
  rectangle), objective = max element count, ties broken by larger azimuth
  aperture. A filled array is a no-op pass-through.
- **Explicit:** caller-supplied channel indices, validated to form a filled
  uniform ULA/URA and re-emitted in lattice order.

## BLOCKED — team forks (do not resolve unilaterally)

1. **Auto vs. explicit** as the primary interface (auto default, explicit escape
   hatch — mirrors the S1 count-vs-length ergonomics fork).
2. **Selection objective** — max element count (DOF) vs. max aperture
   (resolution) when sub-blocks compete. **Implemented:** element count,
   azimuth-aperture tie-break.
3. **Contiguous ULA now vs. sum-set "Harnessing Holes"** (keep holes for
   aperture when the array is sum-set smoothable). **Deferred.**
4. **Standalone op vs. a mode inside `VirtualArrayAssemblyOp`.** **Implemented:**
   standalone.

## Verifiability caveat

S3's headline consumers (Root-MUSIC / ESPRIT) are not yet implemented, so the op
is unit-tested on synthetic holey geometries and validated end-to-end only
through MVDR spatial smoothing on a reduced array; the gridless-estimator wiring
follows when those operators land.

## Deferred

- Sum-set (non-contiguous) holey smoothing (`S_b + S_c`; arXiv:2401.06502).
- Multiple simultaneous sub-arrays / co-array DOF expansion.
- 3-D apertures (rejected today, as S1).
