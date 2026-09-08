# MUSICEstimatorOp — open items

The first HSDK 5 cleanup pass covers optional `is_valid`, scalar live
overrides, optional outputs, constructor-created core workspace, complete
steering artifacts, encapsulated effective-input preparation, and production
Graph performance timing.

## Remaining workspace and performance cleanup

- Reuse spatial-smoothing gather buffers rather than allocating them on each
  smoothed tick.
- Add isolated production-contract timing for geometry, both scan grids,
  spatial smoothing, `num_sources`, and `source_count_pfa`; only
  `max_sources` is timed now.

## Algorithm and performance follow-ups

- The shared EFT source-count helper uses a first-order Gaussian threshold
  rather than Monte-Carlo-calibrated η_P. `source_count_pfa` is a target, not
  a guaranteed false-alarm rate. Rank-deficient inputs fall back to one source.
- `music_reduce_kernel` remains L2-bound on per-beam manifold reads; further
  cross-item manifold reuse is deferred.
- The performance harness fixes `num_sources = 1` for determinism. A model-order
  sweep remains open.
- `kMusicDenomFloor<T> = 1e-6` remains a first-cut numerical floor.

## Python bindings

The HSDK 4.x pybind trampoline is not built while
`HOLOSCAN_RADAR_BUILD_PYTHON=OFF`. Porting it remains deferred under
`docs/planning/HSDK5_PORT.md`.
