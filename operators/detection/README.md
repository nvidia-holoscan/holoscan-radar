# Detection operators

GPU operators for the detection stage of the radar chain. Each operator is
templated on precision (`CFARDetectorOp<float>` / `<double>`); module-wide
port, batching, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's
header carries its full port/parameter reference.

Typical placement:

```text
… → FMCWDopplerProcessorOp → rd_maps → CFARDetectorOp → detection_idx (+ mask / score / noise) → …
```

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| `CFARDetectorOp<T>` | `rd_maps` (…, doppler, range; complex or real power) → `detection_mask`, `detection_score`, `noise_estimate`, `detection_idx` | 2-D CFAR on the (Doppler, range) plane; complex input → square-law power, real input → thresholded directly; per-cell adaptive threshold from the training-cell noise estimate |
| `DetectionReductionOp<T>` | `labels` + `detection_idx` (+ `points`) → `rep_idx` | One representative CFAR cell per cluster (`first` / `peak` / `medoid`) for cluster-then-angle |
| `FeatureAssemblyOp<T>` | `detection_idx` (+ optional `range_bins` / `velocity_bins` / `detection_score` / `aoa_estimates`) → `points` | Gather / bin→physical lookup into the clustering feature matrix; static `features` list sets column layout |

## Implementation status

`CFARDetectorOp` is landing incrementally against its locked contract in
[`docs/planning/OPERATOR_API.md`](../../docs/planning/OPERATOR_API.md). The
header is the full contract; the current build implements a faithful subset:

- **`DetectionReductionOp` (checklist-complete):** group-by-label reduction
  with `first` / `peak` / `medoid`, built-in peak scorers plus compile-time
  `ScoreFn` for `score_method=custom`, sticky `[live]` overrides
  (`representative`, `score_method`, `feature_weights`), exact-sized
  `rep_idx`, public `radar_common/reduce/` primitives, unit tests, and
  `perf_detection_reduction` smoke. Open items in
  [`detection_reduction_op/OPEN_ITEMS.md`](detection_reduction_op/OPEN_ITEMS.md).

- **Implemented (all four CFAR variants — checklist-complete per variant as landed):**
  CA (cell-averaging), GO (2-D GOGO greatest-of), SO (2-D SOSO smallest-of),
  and OS (order-statistic with `[live] os_rank`), complex and real IF power
  paths, per-cell (border-correct) training-cell count, `pfa` / `guard_cells` /
  `training_cells` / `threshold_scale` / `max_detections` / `nci_axes` (all
  overridable with sticky mid-stream tests), adopted
  `[live] alpha_lut` fast path (precomputed multipliers; takes precedence over
  `pfa`), `blocking_alpha_lut_rebuild` for stale-vs-block LUT rebuild policy,
  non-coherent integration over leading axes with Gamma(`K`) thresholding
  (CA: host incomplete-beta LUT; GO/SO: Monte Carlo GOGO/SOSO LUT; OS: closed
  form for `K = 1`, Monte Carlo for `K > 1`), batch-widened `detection_idx`,
  unit + Monte Carlo statistical tests per variant, and the four output ports.
  `detection_score`, `detection_mask`, and `noise_estimate` are dense maps;
  `detection_idx` is compacted per tick on the GPU. Team open items (bool mask
  transport) live in
  [`cfar_detector_op/OPEN_ITEMS.md`](cfar_detector_op/OPEN_ITEMS.md); the
  `alpha_lut` / calibration contract was adopted 2026-07-21 (STATUS.md L5).
- **Staged follow-ups (not contract changes):**
  - ~~Device-side `detection_idx` compaction~~ **landed** — CUB reduce +
    exclusive-scan scatter builds `detection_idx` on the GPU (raster-stable);
    only a scalar count D2H remains for exact-sized allocation.
  - ~~Per-tick dense buffer alloc~~ **landed** — score/noise/mask/power/
    integrated_power are ping-pong cached and reallocated only on geometry
    change.

`FeatureAssemblyOp` is checklist-complete on `dpc/feature-assembly`:

- **Implemented:** static `features` column layout; required `detection_idx`;
  optional `range_bins` / `velocity_bins` / `detection_score` /
  `aoa_estimates` (non-blocking); raw-bin fallback when physical bin axes are
  absent; score/az/el require their inputs; GPU assemble kernel with
  device-side score-plane offsets for preserved batch dims; float and double;
  empty `detection_idx` → empty `points`; unit tests cover
  [`TESTING.md`](../../docs/planning/TESTING.md) FeatureAssembly checks.
- **Staged follow-ups:** see
  [`feature_assembly_op/OPEN_ITEMS.md`](feature_assembly_op/OPEN_ITEMS.md)
  (shared gather with SnapshotGatherOp; optional `points` ping-pong).

## Shared kernel conventions

- **Stage function.** The operator's math is a stage in its own directory
  (`cfar_stage.hpp`, a CUDA-TU-only header); `compute()` is a thin shell that
  resolves overrides, allocates outputs, and compacts `detection_idx` on the
  device (`cfar_compact.cuh`).
- **MatX where it fits; a fused stencil where it doesn't.** Power formation
  is `matx::abs2` (an element-wise fit). The windowed noise estimate,
  adaptive threshold, score, and mask are a single fused CUDA stencil.
  CFAR's noise estimate is a windowed order/sum statistic, not a signal
  convolution — a `matx::conv2d` formulation benchmarked 2.5-13x slower
  (extra passes, per-tick scratch, and a redundant count convolution), and
  the OS variant (per-cell selection) has no MatX expression, so the stencil
  is the uniform home for all four variants.
- **Dtype-selected power path.** Complex `rd_maps` → square-law power
  `|·|²`; real `rd_maps` → treated as an already-formed power map (mirrors
  `FMCWRangeProcessorOp`'s real/complex handling). No parameter.
- **Non-coherent integration.** `nci_axes` sums power along the listed leading
  axes before CFAR; the cell statistic is Gamma(`K`) with
  `K = ∏ integrated axis sizes` (`K = 1` when `nci_axes = []`). The CA
  multiplier `alpha(N)` is precomputed on the host by inverting
  `Pfa = I_{1/(1 + alpha/N)}(N·K, K)` (regularized incomplete beta) and
  uploaded as a per-training-count lookup table for interior and border
  cells. The GO multiplier uses the same Gamma(`K`) CUT statistic but
  calibrates against the 2-D GOGO noise model (range GO × Doppler GO) via
  Monte Carlo inversion (GPU when available), bracketed from the CA seed. SO
  shares the GOGO partition geometry but uses smallest-of side statistics. OS
  uses a per-cell ascending order statistic indexed by `[live] os_rank`
  (0-based); for `K = 1` α is closed-form, for `K > 1` Monte Carlo. An
  optional external `alpha_lut` bypasses online calibration. For `K = 1` and
  representative `K > 1` NCI cases, all variants are Monte Carlo /
  closed-form validated. For `K > 1` the code is **ahead of the contract** —
  see [`cfar_detector_op/OPEN_ITEMS.md`](cfar_detector_op/OPEN_ITEMS.md) §2.
  Preserved leading axes stay batched and are recorded in `detection_idx` as
  extra columns before `(doppler_bin, range_bin)`.
- **Streams.** Work is enqueued on the stream from `receive_cuda_stream()`;
  the input tensor is retained until the next tick as a stream-ordered
  lifetime guard.
