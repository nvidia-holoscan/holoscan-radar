# Angle-estimation operators

GPU operators for dense and sparse direction-of-arrival estimation. Sparse
workflows detect range-Doppler cells first and estimate angles only for those
cells; dense workflows beamform every range-Doppler cell before detection.

```text
Sparse: rd_maps -> CFAR -> SnapshotGatherOp -> estimator -> aoa_estimates
Dense:  rd_maps -> PhaseShiftBeamformerOp -> RDA cube -> CFAR
```

## Choosing an estimator

| Need | Operator | Trade-off |
| --- | --- | --- |
| Fast, robust grid scan | [`BeamscanEstimatorOp`](beamscan_estimator_op/README.md) | Conventional resolution; emits a spectrum |
| Adaptive grid scan | [`MVDREstimatorOp`](mvdr_estimator_op/README.md) | Better interference rejection; covariance inversion and diagonal loading required |
| Subspace super-resolution on a grid | [`MUSICEstimatorOp`](music_estimator_op/README.md) | Resolves multiple sources; requires a model order and sufficient snapshots |
| Gridless ULA or filled-URA estimation | [`RootMUSICEstimatorOp`](root_music_estimator_op/README.md) | Avoids scan-grid quantization; polynomial rooting is comparatively expensive |
| Gridless shift-invariant estimation | [`ESPRITEstimatorOp`](esprit_estimator_op/README.md) | Efficient continuous estimates; geometry must provide the required invariance |
| Full range-Doppler-angle cube | [`PhaseShiftBeamformerOp`](phase_shift_beamformer_op/README.md) | Computes every beam for every cell; use only when dense angular output is needed |

## Supporting operators

| Operator | Role |
| --- | --- |
| [`SnapshotGatherOp`](snapshot_gather_op/README.md) | Gathers channel vectors around detected range-Doppler cells |
| [`SubarraySelectionOp`](subarray_selection_op/README.md) | Selects a valid uniform ULA/URA subset and emits matching geometry |
| [`RankOneResidualScreenOp`](rank_one_residual_screen_op/README.md) | Routes likely multi-source cells to an expensive estimator |

All estimators accept complex64 or complex128 according to their C++ template
precision. Geometry, angle conventions, live-override behavior, and tensor
conventions are described in [`operators/common/README.md`](../common/README.md).
Each operator README is the user-level port, parameter, shape, and algorithm
reference. Deferred research and optimization work is tracked in the repository
[`OPEN_ITEMS.md`](../../OPEN_ITEMS.md).
