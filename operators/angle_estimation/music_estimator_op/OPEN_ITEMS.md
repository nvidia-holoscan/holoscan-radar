# MUSICEstimatorOp — open items

## Cross-item steering-manifold reuse

After the noise-subspace projector was conditionally staged in shared memory,
profiling identified per-beam steering-manifold reads as the remaining
`music_reduce_kernel` L2-bandwidth pressure. The manifold is immutable for a
cached geometry and scan grid and is reused by every item, but the current
kernel independently reads it while evaluating each item.

Further optimization is deferred because sharing manifold tiles across items
changes the kernel mapping and may trade lower L2 traffic for additional
shared-memory use, synchronization, register pressure, or reduced occupancy.
Before changing the kernel, re-profile current HSDK 5 builds on representative
small and large scenarios. Compare candidate item/beam tilings against the
existing occupancy-gated projector staging, and retain the change only if it
improves steady-state latency without regressing smaller grids or either
precision.
