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

## 2. Explicit non-ULA `subarray_spec` on general geometries

**Status:** contract-supported, implementation scope is ULA auto + explicit
pairs validated against `ArrayGeometry`.

The global `ShiftInvariantSubarrayPair` descriptor supports arbitrary
corresponding index sets. v1 validates one common displacement and maps
DOAs through the same 1-D ULA embedding used by Root-MUSIC. Native 2-D
az/el (Unitary / 2-D ESPRIT) remains out of scope for this wave.

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
