# Selective AoA operator stress evaluation

Date: 2026-09-19

## Purpose

This evaluation challenged `RankOneResidualScreenOp` and `MultiSourceExpansionOp` outside the `adaptive_aoa` demonstration before treating them as corpus-ready reusable operators. The tests distinguish API/composability gaps from expected limitations of the residual heuristic, array physics, estimator geometry, and application tuning.

The disposable applications were compiled from an ignored build-tree source directory and were never registered as repository applications. They injected synthetic tensors at standard workflow boundaries so the two operators were tested independently of simulator behavior. The existing producer-side suites separately verified `SnapshotGatherOp`, clustering/reduction, sparse TDM compensation, DDM power folding, and sparse DDM demultiplexing.

## Disposable application matrix

| Scenario | Boundary and variation | Result |
| --- | --- | --- |
| SIMO sparse angle | Physical 8-element ULA, one look, static Params geometry, canonical 2-column detection indices, rank-one and mixed-direction rows | Passed; stable mixed selection, baseline fallback, source expansion, and timestamp preservation |
| Cluster-first MIMO-TDM | Nine-look representative rows shaped like `DetectionReductionOp → TDMSparseCompensationOp`, widened 4-column row identity | Passed; neither operator assumes that its item axis is the raw CFAR row set |
| Sparse MIMO-DDM | Sixteen virtual channels and three looks shaped like `DDMSparseDemultiplexOp` output, widened indices | Passed; virtual-channel size and modality do not affect screen or expansion semantics |
| Estimator interchange | Real one-source `MVDREstimatorOp` baseline → residual screen → common-contract selected estimator output → expansion, with 16 looks | Passed; `baseline_aoa` is not coupled to Beamscan internals |

The boundary applications covered one, three, nine, and sixteen looks; 8- and 16-channel arrays; 2- and 4-column detection indices; mixed selection; confirmed two-source expansion; one-source fallback; compacted diagnostic ordering; and end-to-end Exact timestamp preservation. Focused operator tests supply the complementary empty, no-valid-look, zero/all-selection, malformed-index, source-count, captured-extent, live-geometry fallback, and pool-bound cases.

## Workflow and estimator corroboration

Producer-side Graph suites all passed:

- `SnapshotGatherOp`: 19 tests.
- `ClusterDBSCANOp`: 32 tests.
- `DetectionReductionOp`: 40 tests.
- `TDMSparseCompensationOp`: 3 tests.
- `DDMPowerFoldOp`: 3 tests.
- `DDMSparseDemultiplexOp`: 9 tests.

The real `adaptive_aoa` application also completed its smoke scene with each startup selection: MUSIC, Root-MUSIC, ESPRIT, and MVDR. This verifies that `MultiSourceExpansionOp` consumes every current estimator's common `aoa_estimates` / `doa_multi` / `source_count` outputs without an algorithm-specific adapter.

## Findings

No reusable operator API gap was found.

- The screen composes with physical or virtual channel axes, raw or reduced item rows, static or live-compatible geometry, and a baseline produced by a different estimator.
- Expansion preserves the complete index row, works with canonical and widened indices, maintains original row order, duplicates indices only for confirmed sources, and applies its documented zero/one-source fallback.
- Empty/variable selected sets, optional diagnostics, Exact timestamps, and dynamic exact-sized outputs are already represented by the public contracts and focused Graph coverage.
- No generic row-selection operator, radar-specific record type, estimator adapter, runtime fallback policy, or application-specific compatibility shim is justified by these scenarios.

The real MVDR baseline produced a residual near `0.005` for an isolated off-grid source instead of mathematical zero. It remained below the demo threshold of `0.03`. This is expected scan-grid/model mismatch and reinforces that `residual_threshold` must include baseline quantization, calibration, and noise margin; it is not a screen-contract defect.

Root-MUSIC, ESPRIT, and MVDR can emit unstable angles on low-score noise detections that the demo CFAR admits. This is an estimator/SNR and application-threshold issue. The selective operators correctly preserve those estimator decisions and must not silently reinterpret them.

The temporary HSDK 5 graphs also confirmed two SDK integration details already followed by the maintained code: every produced tensor needs a compile-time device binding even when its optional consumer is absent, and a source must not treat emission to an unwired output as required success. Neither detail requires a public operator change.

## Disposition

Both operators retain their current locked contracts. No follow-up implementation commit is required. The disposable sources, generated configurations, and separate stress build were deleted after the results above were recorded.
