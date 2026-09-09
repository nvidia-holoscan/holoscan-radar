# DDMSparseDemultiplexOp

`DDMSparseDemultiplexOp<T>` is the detect-before-demultiplex bridge for sparse
linear DDM. It gathers only folded CFAR neighborhoods, removes each TX code's
initial phase, resolves an externally supplied Doppler fold, applies virtual
overlap policy, and emits estimator-ready snapshots.

## Contract

- `rd_maps`: device complex `(N_rx,D,R)`, before DDM demultiplexing.
- `detection_idx`: device int64 `(N,2)` folded `(doppler,range)` rows.
- `fold_idx`: optional Exact device int64 `(N)`; missing means zero.
- `snapshot`: device complex `(N,num_virtual,num_snapshots)`.
- `is_valid`: optional/always device uint8 `(N,num_snapshots)`.
- `array_geometry`: required pageable-host `WireArrayGeometry`, emitted before
  `snapshot` for Exact downstream coherence.

`ddm_code`, TX/RX positions, overlap policy, and edge modes are static.
`wavelength` and the POD `snapshot_window` are one-tick Exact live overrides.
The same neighborhood semantics and virtual grouping as
`TDMSparseCompensationOp` are used, but Doppler edges apply to folded width
`W=D/M-guard_bands`.

Sparse v1 accepts only a complete linear exact-bin M-band code. `fold_idx=k`
in `[0,M)` undoes the cyclic TX-band permutation caused by velocity aliasing.
The default `k=0` is correct only in the principal reduced interval; an
unresolved nonzero fold can produce plausible but angle-wrong ordering.
Fold resolution belongs in a future `DDMFoldResolverOp` that inspects the
pre-demux cube, or in an external staggered-CRI/code-rotation observation.
The static `ddm_code` parameter does not support frame-to-frame rotation in
one instance. Nonlinear/fractional layouts and clutter-before-demux are
unsupported.
