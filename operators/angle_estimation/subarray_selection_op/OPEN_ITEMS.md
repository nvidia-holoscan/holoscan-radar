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

## Team forks (ratified — L14, 2026-07-22)

All four ratified as the implemented choices (docs-only; no consumer for any
alternative). See STATUS.md → Settled decisions (L14).

1. **Auto vs. explicit** — ratified **auto default** (largest filled sub-block)
   with an **explicit** channel-index escape hatch (mirrors the S1
   count-vs-length ergonomics fork, L13).
2. **Selection objective** — ratified **max element count** (DOF), ties broken
   by larger **azimuth aperture** — chosen over max-aperture-first.
3. **Contiguous ULA vs. sum-set "Harnessing Holes"** — ratified **contiguous
   only**. Sum-set (non-contiguous, keep holes for aperture when the array is
   sum-set smoothable) is a **deferred future enhancement** (STATUS.md →
   Roadmap).
4. **Standalone op vs. a mode inside `VirtualArrayAssemblyOp`** — ratified
   **standalone** (both operators exist independently: assembly builds the
   virtual array, selection reduces it — clean separation).

## Verifiability caveat

S3's headline consumers (Root-MUSIC / ESPRIT) are not yet implemented, so the op
is unit-tested on synthetic holey geometries and validated end-to-end only
through MVDR spatial smoothing on a reduced array; the gridless-estimator wiring
follows when those operators land.

## Deferred

- Sum-set (non-contiguous) holey smoothing (`S_b + S_c`; arXiv:2401.06502).
- Multiple simultaneous sub-arrays / co-array DOF expansion.
- 3-D apertures (rejected today, as S1).
