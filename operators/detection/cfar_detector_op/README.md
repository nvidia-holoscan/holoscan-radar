# `CFARDetectorOp`

## Purpose

2-D CFAR detector on the (Doppler, range) plane: forms or accepts a power map, estimates local noise from a guard/training window, and emits a per-cell score, mask, noise map, the thresholded power surface, and a compacted list of detected cells. `detection_score > 1` is the detection test (`score = P / (threshold_scale · α · noise_estimate)`); `detection_idx` is exact-sized to the detection count.

## Responsibilities

- Forms or accepts the detection power surface, estimates local noise, applies variant-specific threshold calibration, and optionally suppresses non-maxima.
- Owns dense score/noise/power outputs and exact-size compaction of detected indices and masks.
- Does not cluster detections, estimate angles, or perform physical-velocity clutter notching.

## When to use

Use after range-Doppler formation, dense beamforming, or DDM power folding when adaptive thresholding against local noise is needed. Choose CA for homogeneous backgrounds, GO/SO for asymmetric clutter-edge behavior, and OS for robustness to interfering training cells; configure NCI or peak suppression only when those axes and statistics match the data. Do not use CFAR as a clusterer, angle estimator, or substitute for a physical-velocity notch.

```text
rd_maps / rda_cube / power_map -> CFARDetectorOp
  -> detection_idx + detection_score + noise_estimate [ + detection_mask ]
```

## Ports

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| in | `rd_maps` | complex64 / complex128, or float32 / float64 | `(num_doppler, num_range_bins)`, optional named leading axes | Required trigger/aligned input | Path auto-selected by dtype: complex → square-law abs2; real → already-formed power |
| out (optional, latched) | `detection_mask` | uint8 (`0`/`1`) | same trailing plane; `nci_axes` reduced; other leading axes preserved | Optional latched output | Raw threshold mask for `none`; retained maxima for other peak modes |
| out (optional, latched) | `detection_score` | float32 / float64 | same as `detection_mask` | Optional latched output | `P / (threshold_scale · α · noise_estimate)`; `> 1` ⇔ detected |
| out (optional, latched) | `noise_estimate` | float32 / float64 | same as `detection_mask` | Optional latched output | Per-cell training-window noise |
| out (optional, latched) | `cfar_power` | float32 / float64 | same as `detection_mask` | Optional latched output | Surface that was thresholded: square-law abs2, or post-NCI sum of abs2 |
| out (optional, latched) | `detection_idx` | int64 | `(num_detections, 2 [+ n_preserved])` | Optional latched output | `(…preserved, doppler_bin, range_bin)` per row; row count *is* `num_detections` |
| live | `pfa` | float64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable |
| live | `threshold_scale` | float64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable |
| live | `os_rank` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override (OS only); wire to enable |
| live | `max_detections` | int64 | scalar | Optional Exact one-tick override; Params fallback on miss | Exact override (`<= 0` → uncapped); wire to enable |
| live | `guard_cells` | `CfarCellWindowOverride` POD | int64 `(doppler, range)` pair | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable |
| live | `training_cells` | `CfarCellWindowOverride` POD | int64 `(doppler, range)` pair | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable |
| live | `nci_axes` | int64 config Tensor | `(num_nci_axes,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable; extent bounded by the named leading rank |
| live | `alpha_lut` | float64 config Tensor | `(num_entries,)`, rank-1 pageable host | Optional Exact one-tick override; Params fallback on miss | Exact override; wire to enable; extent bound depends on the variant |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Precision | `float` or `double` template argument | Select at compile time | Fixes accepted and emitted real/complex precision. |
| `extents` | `MaxTensorExtents` | `max_doppler_bins` and `max_range_bins` required; channel/TX and `max_detections` bounds optional | Sizes Tensor pools and bounded workspaces before `arm()`. |

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `cfar_variant` | enum {CA, GO, SO, OS} | `CA` | static | Noise-estimation kernel; instantiate a separate operator to vary it |
| `os_rank` | int64 | none (`-1` sentinel) | `[live]` Exact | 0-based ascending rank into sorted training powers; required for OS, unused otherwise Cost: New rank rebuilds the pfa-derived LUT (OS); cached keys reuse. |
| `guard_cells` | int64[2] `(doppler, range)` | none (required) | `[live]` Exact POD pair | Guard half-extent each side of the CUT Cost: May recalibrate the α LUT; no kernel rebuild. |
| `training_cells` | int64[2] `(doppler, range)` | none (required) | `[live]` Exact POD pair | Training half-extent each side; sets `N` Cost: Same as `guard_cells`. |
| `pfa` | float64 | `1e-6` | `[live]` Exact | Target Pfa when `alpha_lut` is empty; ignored for α while an external LUT is set Cost: Unseen `(variant, pfa, K, window, …)` key builds / queues an α LUT and warns once. |
| `calibration_relative_error` | float64 | `0.10` | static | Relative Pfa error target for direct Monte Carlo; finite in `(0,1)`; not alpha error. |
| `calibration_confidence` | float64 | `0.95` | static | Per-geometry coverage of the relative Pfa band; finite in `(0,1)`; not simultaneous LUT coverage. |
| `alpha_lut` | float64 vector | `[]` | `[live]` Exact host config Tensor | Precomputed α table; non-empty → used directly, `pfa` ignored for α Cost: Validation + H2D upload. |
| `blocking_alpha_lut_rebuild` | bool | `false` | static | `true` blocks the tick on a pfa-LUT miss; `false` serves the previous LUT while a background build runs (first build still blocks) |
| `nci_fold` | int64 | `0` (unset) | static | Optional `K` hint so `start()` can pre-warm when `nci_axes` is set; `<= 0` defers |
| `preintegrated_looks` | int64 | `1` | static | Independent power looks already summed in each real input cell; effective `K = preintegrated_looks · NCI fold` Cost: Changes α calibration. |
| `threshold_scale` | float64 | `1.0` | `[live]` Exact | Empirical multiplier: `threshold = threshold_scale · α · noise`; `1.0` = pure CFAR Cost: Free (kernel argument). |
| `nci_axes` | int64 vector | `[]` | `[live]` Exact host config Tensor | Leading-axis indices to sum in power before CFAR Cost: Changes `K` and output shape; may recalibrate α. |
| `max_detections` | int64 | `0` (uncapped) | `[live]` Exact | `> 0` caps `detection_idx` rows and warns on surplus Cost: Free (cap after count). |
| `peak_mode` | enum {none, range_doppler, all} | `none` | static | Raw detections, per-plane RD maxima, or full N-D maxima Cost: NMS launch unless none. |
| `peak_radius` | int64 vector | `[1]` | static | NMS half-width: scalar or RD pair for `range_doppler`; scalar or post-NCI-rank vector for `all`; zero excludes an axis Cost: Grows neighborhood traversal multiplicatively. |

Contract: [`OPERATOR_API.md` § `CFARDetectorOp`](../../../docs/planning/OPERATOR_API.md) ·
group: [`operators/detection/README.md`](../README.md)

## Algorithm

`compute()` is a thin shell: resolve the `[live]` overrides, classify `rd_maps` dtype, build an `NciPlan`, ensure the α LUT, run the stage, compact `detection_idx`, then loan → copy → emit. All windowed math lives in `cfar_stage.hpp`.

1. **Power.** Complex `rd_maps` → `matx::abs2` (square-law `|·|²`). Real `rd_maps` is copied as an already-formed power map. No parameter selects the path.

2. **Non-coherent integration.** When `nci_axes` is non-empty, `nci_sum_kernel` sums power along those leading axes. The integration fold is `K = ∏ sizes of nci_axes` (`K = 1` when `nci_axes = []`). Other leading axes stay batched. The cell statistic is then Gamma(`K`).

3. **Fused CFAR stencil** (one thread per cell). Guard cells around the cell under test (CUT) are excluded; training cells on each side form the noise estimate. Out-of-map cells are skipped (border-correct `N`). The shared score and mask rule is:

   ```text
   threshold = threshold_scale * alpha * noise
   score = power / threshold ; mask = score > 1
   ```

   `alpha` is `alpha_lut[n]` (CA / OS, `n` = in-bounds training-cell count) or `alpha_lut[go_lut_index(...)]` (GO / SO). Slot 0 is unused. `train_cnt == 0` (or GO/SO `lut_idx == 0`) writes zeros and no detection.

   - **CA** — `noise_estimate = train_sum / train_cnt` over the CA window.
   - **GO / SO** (2-D GOGO / SOSO) — split training cells by range sign (`rr`) and Doppler sign (`dd`) relative to the CUT; per axis form `max` (GO) or `min` (SO) of the two half-side means (single-sided at borders); then `noise_estimate` is `max` (GO) or `min` (SO) of the two axis statistics. `go_lut_index`: per axis, `min(neg, pos)` if both half-sides are alive else the alive side’s count; then `min(range_side, doppler_side)` if both axes are alive else the alive axis.
   - **OS** — collect the same CA-window training powers, insertion-sort ascending, take index `os_rank` (0 = smallest). Border cells with `train_cnt <= os_rank` clamp to `train_cnt − 1`. Footprints with more than `kMaxOsTrainingCells` (512) training cells are rejected.

4. **α calibration** when `alpha_lut` is empty. CA inverts `Pfa = I_{1/(1 + alpha/N)}(N·K, K)` (regularized incomplete beta) by host bisection (`cfar_alpha.cpp`). At `K = 1` this matches `α = N·(pfa^(−1/N) − 1)` (`CaAlphaK1MatchesContractClosedForm`). OS at `K = 1` inverts the closed form quoted in `cfar_alpha.cpp`:

   ```text
   Pfa = prod_{i=0}^{rank} (N - i) / (N - i + alpha)
   ```

   GO, SO, and OS with `K > 1` use direct Monte Carlo with a Pfa- and accuracy-dependent trial budget. Each trial draws training powers and a CUT, then forms `CUT / noise_statistic`. Streamed upper-tail selection gives alpha without repeating draws for a threshold search. An external `alpha_lut` skips calibration and ignores `pfa` for α. `threshold_scale` always multiplies the active α.

5. **Start-time pre-warm.** `start()` pre-builds analytic LUTs and queues cancellable multi-look OS calibration for **CA and OS only**, keyed on a canonical window-sized map (`win = 2·(guard+train)+1` per axis). GO / SO are map-dependent (`go_lut_index`) and are **not** pre-warmed. A missing `nci_fold` when `nci_axes` is set defers calibration to the first tick. `preintegrated_looks` contributes to `K` even when `nci_axes` is empty.

6. **Peak-mode NMS** (static, default `none`). `range_doppler` runs `cfar_peak_suppress_kernel` independently in every leading plane. `all` dispatches rank-2/3/4 N-D kernels. `peak_radius` defines a full hyperrectangle in bins; a scalar broadcasts, while `all` also accepts one value per post-NCI output axis. Zero excludes an axis. Both modes compare only detected neighbors using raw `cfar_power`; exact ties retain the lowest full row-major index. Dense score/noise/power remain unchanged; mask/index follow selected peaks.

7. **`detection_idx` compaction.** CUB reduce + exclusive-scan scatter writes rows in batch-major raster order: preserved leading coordinates, then `(doppler_bin, range_bin)`. A scalar device→host count sync sizes the exact output (`count_mask_detections`). If `max_detections > 0` and the scene exceeds it, surplus rows are dropped (raster-first kept) and the operator warns.

## Offline calibration and accuracy

Automatic Monte Carlo calibration defaults to `calibration_relative_error: 0.10`
and `calibration_confidence: 0.95`. The initial budget is approximately
`z^2 * (1-pfa) / (error^2 * pfa)`, with at least 4096 trials. The final budget
and tail rank are checked against binomial order-statistic coverage. Confidence
describes the probability that the selected alpha's model Pfa lies in
`[(1-error)*pfa, min(1,(1+error)*pfa)]` for one geometry. It is not a bound on
alpha error, simultaneous coverage across the table, or a guarantee for
correlated application noise. Taking the maximum alpha across aliased GO/SO
geometries may make a shared LUT entry conservative. Coverage refers to the
unscaled per-bin test, before `threshold_scale` and peak suppression.

At Pfa `1e-9`, the defaults require roughly 400 billion trials per geometry.
This is intentionally no longer approximated using 4096 trials. Calculations
use checked uint64 counts and reject unrepresentable budgets instead of
silently reducing accuracy. Trials use deterministic Philox streams and bounded
65536-trial GPU batches. Only the required upper tail survives between batches;
there is no full-run sample buffer or CPU sampling fallback.

On a cache miss, warnings report the start timestamp, settings, geometries,
trial counts, and whether the caller blocks or serves a previous table. A
bounded GPU probe estimates duration and completion time; progress logs refresh
at most every five seconds. Estimates include generation and tail selection
and can change with GPU load and geometry. Builds continue until completion or
cancellation. Explicit `alpha_lut` bypasses this work. Cancelled/failed jobs do
not enter the cache; a shared build is cancelled only after its last consumer
releases it. Calibration settings and algorithm version are part of cache identity.

Both offline APIs use the same engine and actual input shape as the operator:

```cpp
#include <holoscan/radar/detection/cfar_calibration.hpp>
#include <holoscan/radar/detection/cfar_detector_op.hpp>

holoscan::radar::CFARParams params;
params.cfar_variant = holoscan::radar::CfarVariant::kGO;
params.guard_cells = {3, 3};
params.training_cells = {3, 3};
params.pfa = 1e-9;
const std::vector<std::int64_t> shape{128, 256};
auto plan = holoscan::radar::plan_cfar_calibration(params, shape);  // Bounded timing probe.
// Inspect plan.total_trials and plan.estimated_seconds before a long job.
auto result = holoscan::radar::calibrate_cfar(params, shape);  // Potentially hours.
params.alpha_lut = result.alpha_lut;  // Bypass runtime Monte Carlo.
```

The C++ overload accepts a progress callback and `std::stop_token`. Callback
exceptions cancel the build. Python releases the GIL while computing and
checks interrupts between batches; raising from its callback also cancels:

```python
import json
from holoscan_radar import calibrate_cfar, plan_cfar_calibration

params = {
    "cfar_variant": "GO",
    "guard_cells": [3, 3],
    "training_cells": [3, 3],
    "pfa": 1e-9,
    "calibration_relative_error": 0.10,
    "calibration_confidence": 0.95,
}
plan = plan_cfar_calibration(params, [128, 256])
print(plan["total_trials"], plan["estimated_seconds"])
# Run deliberately offline; this example's requested accuracy is expensive.
result = calibrate_cfar(params, [128, 256])
with open("cfar_calibration.json", "w") as stream:
    json.dump(result, stream)
with open("cfar_calibration.json") as stream:
    saved = json.load(stream)
params["alpha_lut"] = saved["alpha_lut"]
```

Results use `method` to distinguish analytic, external, empty-training and Monte Carlo paths. Python `coverage` is null outside Monte Carlo; no empirical accuracy is claimed for a supplied LUT.

Results contain the LUT plus configuration, shape, effective looks, per-geometry
alphas, trial budget, coverage, seed/algorithm version, timestamps, and elapsed
time. Store the metadata alongside the LUT and regenerate when its noise-model
configuration changes. YAML serialization of the returned ordinary Python
values is also supported. This version has no calibration CLI or persistent
runtime cache. A more efficient rare-event estimator is tracked as P1 in
[CLARAHOLOS-3583](https://nvidia.atlassian.net/browse/CLARAHOLOS-3583).

### Local calibration timing

A Release HSDK 5 GA CUDA 13 check on NVIDIA GB10 (driver 580.173.02) used GO, a 128x256 map,
guard `[3,3]`, training `[3,3]`, and the default 95% / 10% accuracy settings.
Three complete offline builds at `pfa: 1e-3` each processed 64,855,778 trials
across 169 geometries. Observed durations were 1.09, 0.82, and 0.82 seconds;
initial estimates were 1.75, 1.69, and 1.69 seconds. The timing probe warms
CUDA first, then uses the median of three bounded batches at the largest LUT
index. Applying that rate to every geometry is deliberately conservative.
Runtime progress updates use observed complete-build throughput instead.

Planning the same configuration at `pfa: 1e-9` selected 384,145,881,686 trials
per geometry (coverage 0.95022), 64,920,654,004,934 trials in total, and a
conservative initial estimate of approximately 20 days on this machine.
That rare-event job was **not executed**. This is a local timing observation,
not a latency bound or a guarantee under different GPU load, clocks, or looks.

## Shape and dtype rules

- **`detection_mask` / `detection_score` / `noise_estimate` / `cfar_power`.** Trailing axes are the input `(num_doppler, num_range_bins)`. Leading axes listed in `nci_axes` are reduced; remaining leading axes are preserved in the same order. `NciPlan::output_shape` is `preserved_dims + (num_doppler, num_range)`.
- **`detection_idx`.** Rows = detection count after peak selection and `max_detections`. Columns = `2 + n_preserved`: preserved leading indices, then `doppler_bin`, then `range_bin`. Empty scene → shape `(0, 2 [+ n_preserved])`.
- **Named leading axes.** Admitted only up to the rank implied by ctor `MaxTensorExtents`: none, one (`max_rx` or `max_channels`), or two (`max_tx` plus `max_rx` / `max_channels`). Extra leading rank is rejected. The dense RDA path (`num_az`, `num_el` leading) is admitted by standing `max_tx` / `max_rx` in for `max_az` / `max_el` (see Construction); the operator does not read `max_az` / `max_el`.
- **Precision.** Template `T ∈ {float, double}` only. Complex input must be `complex<T>`; real input must be `T`. A mismatched dtype is rejected (message starts `rd_maps: received dtype … does not match this operator's float|double-precision instantiation`).
- **Legal mid-stream shape change.** A `[live]` `nci_axes` change changes the output rank (downstream re-infers); any change that alters the workspace geometry reallocates the ping-pong buffers.
- **Rejected combinations** (operator’s own wording unless noted):

  | Condition | Message |
  | --- | --- |
  | `guard_cells` / `training_cells` not length 2 | `CFARDetectorOp: 'guard_cells' must have exactly 2 entries (doppler, range), got N` (same for `training_cells`) |
  | Negative pair entry | `CFARDetectorOp: 'guard_cells' entries must be >= 0` |
  | Rank `< 2` | `CFARDetectorOp: rd_maps must have at least 2 axes (num_doppler, num_range)` |
  | Extra leading rank | `CFARDetectorOp: unexpected leading axes (opaque batch is not supported); got leading rank R, max named leading M` |
  | Tick exceeds ctor maxima | `CFARDetectorOp: input doppler_bins exceed MaxTensorExtents` (same for `range_bins`) |
  | `nci_axes` index out of range / duplicate | `rd_maps: nci_axes index A is out of range for L leading axis` / `rd_maps: duplicate nci_axes index A` |
  | Non-device or non-contiguous `rd_maps` | `CFARDetectorOp: tensor received on input port 'rd_maps' is not CUDA-accessible …` / `… must be contiguous (compact row-major)` |
  | OS `os_rank` unset (`-1`) | `CFARDetectorOp: os_rank is required for OS variant; set a 0-based rank into the ascending-sorted training cells` |
  | OS `os_rank < -1` | `CFARDetectorOp: os_rank must be non-negative for OS variant` |
  | OS `os_rank >= max training-cell count` | `CFARDetectorOp: os_rank must be less than the maximum training-cell count for the configured window footprint` |
  | OS window `N > 512` | `CFARDetectorOp: OS window footprint exceeds the supported training-cell limit` |
  | External `alpha_lut` too short | `CFARDetectorOp: alpha_lut length must exceed the maximum LUT index for the configured variant, footprint, and map size` |
  | External `alpha_lut` bad entry | Negative/non-finite anywhere: `… entries for used indices must be finite and non-negative`; zero at a reachable index: `… entries for reachable indices must be finite and positive` |
  | Missing ctor extent | `CFARDetectorOp: MaxTensorExtents.max_doppler_bins must be a positive value (required for Tensor pool sizing)` (same for `max_range_bins`) |
  | No CUDA stream | `CFARDetectorOp: no CUDA compute stream` (`ErrorCode::kNotSupported`) |

## Performance

**Cold.** First-seen α LUT (host incomplete-beta for CA; Monte Carlo for GO/SO/OS-`K>1` — potentially hours or longer for rare Pfa); start-time CA/OS pre-warm when `nci_fold` is set; dense workspace allocation (score / noise / mask / peak_mask / power / integrated_power) on geometry change; CUB temps and NCI lookup-table H2D; first exact-size `detection_idx` loan.

**Steady.** Fused stencil + optional NCI/NMS; CUB compact while `detection_idx` is active; one D2D emit per active output. Workspace and CUB state reuse when `(batch_in, out_batch, num_doppler, num_range)` is unchanged. Previously seen LUT keys are served from the process-wide `CfarAlphaLutKey` cache (variant, pfa, fold, os_rank, guard, train, key map size). For 1–65535 post-NCI planes, one detector launch maps planes to `grid.z`; larger counts retain the per-plane fallback. Each cell keeps the same ordered stencil arithmetic; NCI and peak-suppression algorithms are unchanged.

A paired Thor HSDK 5 check (5 warmups, 50 measured frames, three fresh pairs) measured rank-four medium float marginal medians 0.492103→0.462042ms and double 1.142230→1.115161ms. Paired reductions were 30.659µs/6.230% and 29.876µs/2.609%; two of three pairs per precision exceeded the declared useful-gain floor, with all pairs positive. All eight selected protected endpoint rules passed, alongside 182 native tests and exact all-five-output comparisons across eleven cases in both precisions.

These are steady-workload results. Medium double had fresh paired slowdowns within the declared 5% allowance (worst 3.44%). Float rank-four setup marginal median increased about 0.843ms; isolated actual NCI override costs increased up to 0.904ms float and 1.124ms double. One measured 50-compute-plus-event diagnostic sum also increased. Startup, frequent reconfiguration and tail latency are not guaranteed to improve.

**Live rebuilds.** `pfa` / `os_rank` / `guard_cells` / `training_cells` / `nci_axes` / `alpha_lut` can miss the LUT cache: warn once per configuration (`CFARDetectorOp: calibrating the alpha LUT at runtime …`). `threshold_scale` and `max_detections` do not rebuild. `blocking_alpha_lut_rebuild = false` serves the previous LUT while the new table builds (`CFARDetectorOp: alpha LUT for the new configuration is still calibrating; using the previous LUT for this tick`).

`preintegrated_looks` assumes independent Gamma-distributed looks. Spatially
correlated receiver noise, shared-LO phase noise, mutual coupling, or clutter
residue lower the effective degrees of freedom; calibrate an effective value
or provide `alpha_lut` when nominal Pfa must be maintained under those models.

Harness: `performance/operators/detection/cfar_detector/adapter.cpp` (Graph). It wires all five outputs, so steady timings include publication copies plus index compaction. Axes include size, variant, footprint, NCI fold, peak mode/radius, and alpha source. Rank-3/4 `all` rows characterize full N-D NMS.

Run this performance smoke **inside a development container**, after
configuring the named build tree with performance enabled (see the
[developer guide](../../../DEVELOPER.md#performance-characterization)):

```bash
ctest --test-dir build -R perf_cfar_detector_smoke
```

## Usage

```yaml
cfar_detector:
  cfar_variant: CA
  guard_cells: [3, 3]
  training_cells: [3, 3]
  pfa: 1.0e-9
  peak_mode: none
```

```cpp
auto cfar = graph.op<CFARDetectorOp<float>>("cfar_detector", extents);
cfar.set_params(from_config("cfar_detector"));  // or a CFARParams filled by the app

graph.add_flow(doppler->rd_maps, cfar->rd_maps, q);
graph.add_flow(cfar->detection_idx, feat->detection_idx, q);
graph.add_flow(cfar->detection_score, feat->detection_score, q);
```

`extents` must set `max_doppler_bins` and `max_range_bins`. Sparse SIMO/MIMO with channel NCI also sets `max_rx` (and `max_tx` when a TX axis is present). Dense RDA CFAR must present a rank-2 leading admission — today that means `max_tx = max_az`, `max_rx = max_el`.
