# SubarraySelectionOp — open items

Status: **settled — S3 interface resolved; all four team forks ratified (L14).**

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

The four selection forks (auto-vs-explicit, objective, contiguous-vs-sum-set,
standalone-vs-`VirtualArrayAssemblyOp` mode) are settled — see STATUS.md →
Settled decisions (L14).

## Verifiability caveat

S3's headline consumers (Root-MUSIC / ESPRIT) are not yet implemented, so the op
is unit-tested on synthetic holey geometries and validated end-to-end only
through MVDR spatial smoothing on a reduced array; the gridless-estimator wiring
follows when those operators land.

## Deferred

- Sum-set (non-contiguous) holey smoothing (`S_b + S_c`; arXiv:2401.06502).
- Multiple simultaneous sub-arrays / co-array DOF expansion.
- 3-D apertures (rejected today, as S1).
