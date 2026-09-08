# MVDREstimatorOp — open items

The first HSDK 5 cleanup pass covers optional `is_valid`, `max_sources`,
optional outputs, constructor-created core workspace, cached steering
artifacts, captured bounds, encapsulated effective-input preparation, and
production Graph performance timing.

## Remaining workspace and performance cleanup

- Reuse spatial-smoothing gather buffers rather than allocating them on each
  smoothed tick.
- Add isolated production-contract timing for geometry, both scan grids,
  spatial smoothing, and `max_sources`; only `diagonal_loading` is timed now.

## Process-state contamination in MatX solve path

Several DOA tests fail when the whole `test_mvdr_estimator` binary runs in one
process but pass when GTest/CTest isolates each test in a fresh process. A
two-tick identical-input probe also produces different spectra. Evidence points
to the `matx::eig` / `matx::inv` cold path rather than live-override resolution:
geometry, cache keys, streams, and device addresses agree.

This blocks a trustworthy value-comparison test for the live
`diagonal_loading` override, so
`LiveDiagonalLoadingAppliesForOneTick` remains skipped. The active
`max_sources`, geometry, grid, smoothing, and mask tests are unaffected when
run in isolation.

Next: reduce the issue to a standalone batched Hermitian MatX `eig` + `inv`
reproducer and retest against a newer MatX/HSDK build.

## Algorithm and performance follow-ups

- Auto diagonal loading uses
  `1e-2 * trace(covariance) / num_channels`; the factor remains a first cut.
- Capon reduction remains latency-bound; warp-cooperative or GEMM paths are
  deferred.

## Python bindings

The HSDK 4.x pybind trampoline is not built while
`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`. Porting it remains deferred under
`docs/planning/HSDK5_PORT.md`.
