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

## Implementation status

`CFARDetectorOp` is landing incrementally against its locked contract in
[`docs/planning/OPERATOR_API.md`](../../docs/planning/OPERATOR_API.md). The
header is the full contract; the current build implements a faithful subset:

- **Implemented:** CA (cell-averaging) variant, complex and real IF power
  paths, per-cell (border-correct) training-cell count, `pfa` /
  `guard_cells` / `training_cells` / `threshold_scale` / `max_detections`
  / `nci_axes` (all overridable), non-coherent integration over leading
  axes with Gamma(`K`) CA thresholding, batch-widened `detection_idx`, and
  the four output ports. `detection_score`, `detection_mask`, and
  `noise_estimate` are dense maps; `detection_idx` is compacted per tick.
- **Staged follow-ups (declared, not yet implemented — not contract
  changes):**
  - GO / SO / OS variants (`cfar_variant` is static, so each is a clean
    per-instance boundary; non-CA currently raises a clear error).
  - Device-side `detection_idx` compaction (v1 compacts host-side with one
    stream sync per tick).

## Shared kernel conventions

- **Stage function.** The operator's math is a stage in its own directory
  (`cfar_stage.hpp`, a CUDA-TU-only header); `compute()` is a thin shell that
  resolves overrides, allocates outputs, and compacts `detection_idx`.
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
  axes before CA-CFAR; the cell statistic is Gamma(`K`) with
  `K = ∏ integrated axis sizes` (`K = 1` when `nci_axes = []`). The CA
  multiplier is recomputed from `K` each tick. For `K = 1` this matches the
  contract exactly: `α = N · (Pfa^(-1/N) - 1)`. For `K > 1` the code uses the
  power-law generalization `α = N · (Pfa^(-1/(N·K)) - 1)` — sufficient for
  current unit tests but **not** the settled false-alarm calibration; the
  correct CA-CFAR Pfa under Gamma statistics needs incomplete-beta math. See
  [`cfar_detector_op/OPEN_ITEMS.md`](cfar_detector_op/OPEN_ITEMS.md) §2.
  Preserved leading axes stay batched and are recorded in `detection_idx` as
  extra columns before `(doppler_bin, range_bin)`.
- **Streams.** Work is enqueued on the stream from `receive_cuda_stream()`;
  the input tensor is retained until the next tick as a stream-ordered
  lifetime guard.
