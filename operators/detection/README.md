# Detection operators

GPU operators for the detection stage of the radar chain. Each operator is
templated on precision (`CFARDetectorOp<float>` / `<double>`); module-wide
port, batching, and parameter-override conventions are documented in
[`operators/common/README.md`](../common/README.md), and each operator's
header carries its full port/parameter reference.

Typical placement:

```text
… → CFARDetectorOp → detection_idx (+ score) → FeatureAssemblyOp → points → …
… → ClusterDBSCANOp.labels + detection_idx + points → DetectionReductionOp → rep_idx
```

## Operators

| Operator | In → Out | Function |
| --- | --- | --- |
| [`CFARDetectorOp<T>`](cfar_detector_op/README.md) | `rd_maps` (…, doppler, range; complex or real power) → `detection_mask`, `detection_score`, `noise_estimate`, `cfar_power`, `detection_idx` | 2-D CFAR on the (Doppler, range) plane; complex input → square-law power, real input → thresholded directly; per-cell adaptive threshold from the training-cell noise estimate |
| [`FeatureAssemblyOp<T>`](feature_assembly_op/README.md) | `detection_idx` (+ optional `range_bins` / `velocity_bins` / `detection_score` / `aoa_estimates` / `az_angles` / `el_angles`) → `points` | Range–Doppler clustering-prep bridge: gather / bin→physical lookup into `points`. Lives here (not under `clustering/`) because it speaks CFAR `detection_idx`; static `features` list sets column layout |
| [`DetectionReductionOp<T>`](detection_reduction_op/README.md) | `labels` + `detection_idx` (+ `points`) → `rep_idx` | One representative CFAR cell per cluster (`first` / `peak` / `medoid`) for cluster-then-angle |

## Implementation status

All three operators implement their locked contracts in
[`docs/planning/OPERATOR_API.md`](../../docs/planning/OPERATOR_API.md).

- **`DetectionReductionOp` (checklist-complete):** group-by-label reduction
  with `first` / `peak` / `medoid`, built-in peak scorers plus compile-time
  `ScoreFn` for `score_method=custom`, `[live]` overrides
  (`representative`, `score_method`, `feature_weights`), exact-sized
  `rep_idx`, public `radar_common/reduce/` primitives, unit tests, and
  `perf_detection_reduction` smoke. `points` always joins the compiled aligned
  trigger so every live representative policy shares one fixed contract.

- **Implemented (all four CFAR variants — checklist-complete per variant as landed):**
  CA (cell-averaging), GO (2-D GOGO greatest-of), SO (2-D SOSO smallest-of),
  and OS (order-statistic with `[live] os_rank`), complex and real IF power
  paths, per-cell (border-correct) training-cell count, `pfa` / `guard_cells` /
  `training_cells` / `threshold_scale` / `max_detections` / `nci_axes` (all
  overridable, with mid-stream override tests), adopted
  `[live] alpha_lut` fast path (precomputed multipliers; takes precedence over
  `pfa`), `blocking_alpha_lut_rebuild` for stale-vs-block LUT rebuild policy,
  non-coherent integration over leading axes with Gamma(`K`) thresholding
  (CA: host incomplete-beta LUT; GO/SO: Monte Carlo GOGO/SOSO LUT; OS: closed
  form for `K = 1`, Monte Carlo for `K > 1`), batch-widened `detection_idx`,
  unit + Monte Carlo statistical tests per variant, and all five output ports.
  `detection_score`, `detection_mask` (uint8 0/1), and `noise_estimate` are
  dense maps; `detection_idx` is compacted per tick on the GPU. The `detection_
  mask` dtype (uint8 rather than bool, for HSDK 4.3 NumPy visibility) was settled
  2026-07-22 (STATUS.md L1); the `alpha_lut` / calibration contract was adopted
  2026-07-21 (STATUS.md L5).
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
  `aoa_estimates`; Params-derived exact-aligned joins for selected sources;
  symmetric physical `range` / `velocity` / `az` / `el`, raw `*_bin`, and
  sparse `az_estimate` / `el_estimate` tokens; all physical features require
  their source inputs; GPU assemble kernel with
  device-side score-plane offsets for preserved batch dims; float and double;
  empty `detection_idx` → empty `points`; unit tests cover
  [`TESTING.md`](../../docs/planning/TESTING.md) FeatureAssembly checks.
- **Status:** checklist-complete; no operator-local open items.

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
  the range processor's real/complex handling). No parameter.
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
  closed-form validated. The `K > 1` closed form (incomplete-beta relation) is
  now pinned in the contract (settled — STATUS.md L2).
  Preserved leading axes stay batched and are recorded in `detection_idx` as
  extra columns before `(doppler_bin, range_bin)`.
- **Streams.** Work is enqueued on `ExecutionContext::cuda_stream()`;
  the input tensor is retained until the next tick as a stream-ordered
  lifetime guard.
