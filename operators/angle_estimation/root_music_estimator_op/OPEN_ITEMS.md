# RootMUSICEstimatorOp — open items

The first HSDK 5 cleanup pass covers optional `is_valid`, scalar live
overrides, optional outputs, constructor-created core workspace, captured
bounds, encapsulated effective-input preparation, and production Graph
performance timing.

## Remaining workspace and performance cleanup

- Move the remaining URA per-axis scratch allocations into bounded reusable
  runtime state.
- Reuse spatial-smoothing gather buffers rather than allocating them on each
  smoothed tick.
- Add isolated production-contract timing for geometry, spatial smoothing,
  `num_sources`, and `source_count_pfa`; only `max_sources` is timed now.

## Batched polynomial root solver

Serial per-item `cusolverDnXgeev` dominates latency. A batched or
device-resident replacement remains postponed by team decision (2026-07-22).

## Two-dimensional variants

The locked design is the landed reduced-dimension filled-URA algorithm with
pairing. A joint 2-D polynomial and explicit two-pair non-ULA subarray
specifications remain deferred.

## Python bindings

The HSDK 4.x pybind trampoline is not built while
`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`. Porting it remains deferred under
`docs/planning/HSDK5_PORT.md`.
