# ESPRITEstimatorOp — open items

## 1. TLS / Unitary ESPRIT variants (performance / accuracy)

**Status:** deferred follow-up, not a contract gap.

v1 implements **LS-ESPRIT** in a per-item CUDA block: form the signal
subspace from the largest eigenvectors, split it with the resolved
`ShiftInvariantSubarrayPair`, solve `Phi = pinv(U1) U2`, and take the
phases of the `d x d` eigenvalues. **TLS-ESPRIT** and **Unitary ESPRIT**
are better at low SNR / correlated sources but add an SVD or real-valued
preprocessing stage per item. Track as an accuracy/perf follow-up once the
LS baseline is characterized.

## 2. URA 2-D az/el and explicit non-ULA `subarray_spec`

**Status:** 2-D URA landed (2026-07-23, contract change); explicit two-pair 2-D
specs deferred.

The operator is now geometry-driven. A **1-D ULA** with `subarray_spec: auto`
resolves azimuth only via the single-shift LS path (unchanged). A **filled URA**
with `subarray_spec: auto` (detected via `analyze_planar_aperture`, `(iz` outer,
`iy` inner) channel order) resolves native (az, el) via **dual-invariance 2-D
ESPRIT**: two orthogonal max-overlap invariances (`build_ura_dual_invariance`)
give LS rotation operators `Psi_y`, `Psi_z`; the in-kernel QR is extended to
accumulate Schur vectors `Z` so both operators are *simultaneously* (joint-)
diagonalized, and the paired diagonals of `Z^H Psi_y Z` and `Z^H Psi_z Z` yield
automatically-paired `(u, v)` → boundary (az, el). Verified on analytic URA
snapshots (single + two-source distinct-az-AND-el, float/double).

An **explicit** `ShiftInvariantSubarrayPair` always stays on the 1-D LS path and
never triggers planar-aperture analysis (the global descriptor still supports
arbitrary corresponding index sets, one common displacement, mapped through the
1-D embedding shared with Root-MUSIC).

**Deferred (documented, not blockers):** explicit *two-pair* 2-D subarray specs
(auto derivation covers the filled-URA case), TLS/Unitary real-valued 2-D
speedups (see item 1), and pairing robustness when sources share an az or el
(bounded by tests). URA scenarios (`ura_sm` 6x6, `ura_lg` 8x8) are in the perf
harness — characterized, not gated.

## 3. `SubarraySelectionOp` → ESPRIT integration test

**Status:** follow-on after the operator lands.

S3 is settled and Root-MUSIC documents the holey-array rejection path.
Add an end-to-end test that feeds a reduced ULA from `SubarraySelectionOp`
into ESPRIT once the sparse-path integration harness exists.

## 4. Small-matrix eigen solver choice

**Status:** v1 uses an in-kernel shifted QR iteration for the `d x d`
rotation eigenproblem (`d <= max_sources`, default cap 4). This avoids
per-item `cusolverDnXgeev` launch overhead. If profiling shows numerical
fragility at specific `(M, d, SNR)` corners, consider batched
`cusolverDnXgeev` on `(num_items, d, d)` as a drop-in backend.
