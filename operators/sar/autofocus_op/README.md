# `AutofocusOp`

## Purpose

Estimate and remove a residual, scene-common phase error along the azimuth axis of a formed image.

## Responsibilities

- Estimate a per-pulse phase error by phase gradient autofocus (PGA) on strong, isolated scatterers.
- Apply that estimate in-image, and publish it so a different formation can be corrected instead.

## When to use

Use when a formed image is defocused by a residual phase the geometry model did not capture. This is a separate, omittable stage rather than a mode inside image formation, and it is off by default. Correcting in-image is only meaningful when the product is itself slow-time-conjugate (WORKFLOWS Figure 3); on a ground or DEM product the estimate instead reaches formation through `PhaseCorrectionOp` on a replayed aperture (Figure 2). Input and output image names differ because HSDK 5 requires unique port names.

## Ports

The image drives the tick. The four scalar inputs are `[live]` overrides: each joins the image trigger with `Exact{}` and `on_missing SafeDefault{}`, so wiring one is what enables it. `T=float` means complex64/float32; `T=double` means complex128/float64.

| Kind | Port | Payload type / dtype | Shape | Timing / optionality | Responsibility |
| --- | --- | --- | --- | --- | --- |
| In | `image` | complex `T` | `(num_v,num_u)` | Required, drives the tick | Formed SLC image |
| In | `num_iterations` | `int64` | scalar | `[live]`, optional | Override refinement count |
| In | `window_fraction` | `float64` | scalar | `[live]`, optional | Override first window width |
| In | `min_scatterers` | `int64` | scalar | `[live]`, optional | Override scatterer floor |
| In | `max_correction` | `float64`, radians | scalar | `[live]`, optional | Override peak-to-peak ceiling |
| Out | `image_out` | complex `T` | `(num_v,num_u)` | Every tick | Corrected image |
| Out | `phase_correction` | real `T`, radians | `(num_pulses)` | Every tick, optional | Pulse-indexed estimate |
| Out | `sharpness` | real `T` | `(num_iterations + 1)` | Every tick, optional | Focus metric per iteration |

## Construction

| Item | Type | Requirement | Responsibility |
| --- | --- | --- | --- |
| Operator | `AutofocusOp` | Add to an HSDK 5 Graph | Own the named ports and Params |
| Tensor pool | `MaxTensorExtents` | Capture at `graph.op()` | Bound fat output tensors before arm |

Set `max_image_v` and `max_image_u` to cover the largest image. The azimuth axis is whichever of the two the caller names, so the `phase_correction` pool is sized from the larger of the pair rather than from `azimuth_axis`, which this constructor must not read.

## Parameters

| Parameter | Type | Default | Mutability | Responsibility |
| --- | --- | --- | --- | --- |
| `algorithm` | `AutofocusAlgorithm` | `pga` | Static | Estimator selection; `pga` is the sole value |
| `azimuth_axis` | `AutofocusAzimuthAxis` | required | Static | Which trailing axis is slow-time-conjugate |
| `num_iterations` | `int64` | 4 | `[live]` | Refinement iterations |
| `window_fraction` | `float64` | 0.5 | `[live]` | Fraction of azimuth extent in the first window |
| `min_scatterers` | `int64` | 16 | `[live]` | Minimum strong scatterers before estimating |
| `max_correction` | `float64` | uncapped | `[live]` | Peak-to-peak ceiling in radians |

## Operation and synchronization

Each iteration selects range bins whose azimuth peak stands well above the row mean, circularly shifts each selected peak to the azimuth centre, applies a window that shrinks geometrically across iterations, and estimates the phase gradient by the linear unbiased estimator. The integrated gradient is accumulated across iterations and applied to the image spectrum. `sharpness` records the focus metric before each iteration and after the last, so a caller can see whether the estimate helped.

**`window_fraction` narrows the image support, never the transform.** The gradient FFT is always the full azimuth extent, zeros outside the window included. Packing the window into a shorter transform is tempting — it is `window_fraction` times the work — but it decimates the phase history by the inverse of that fraction, so each gradient difference spans that many samples and the estimator's small-angle form saturates. That cost nothing on a gentle error and up to 5.5x the residual on a steep one; see the app [README](../../../applications/sar/autofocus/README.md).

PGA is blind to a constant and a linear phase ramp: both translate the image rather than blur it. Compare an estimate against a truth only after removing an affine fit from each.

A third guard discards an iteration whose estimate lowers sharpness and stops there, keeping the last accepted estimate. It is a faithful proxy: forcing the refused iterations through on Kourou at the full aperture takes the residual against injected truth from 0.332 to 1.295 rad. Because it stops before accumulating, a rejection at iteration 0 emits a zero correction and the refused estimate cannot be seen from outside — set `SAR_AF_IGNORE_DROP` in the environment to keep it anyway and score it. That is a diagnostic, not a tuning knob; it will leave the image worse.

Two guards cause a pass-through, in which the image is copied unchanged and `phase_correction` is all zeros: fewer strong scatterers than `min_scatterers`, and an estimate whose peak-to-peak span exceeds `max_correction`. Naming `azimuth_axis` also asserts that the image is conjugate along that axis; that assertion is not dimensionally checkable, so a wrong axis is not an error — it simply fails to recover, which is why the parameter is required and has no default.

The image must be a contiguous, CUDA-accessible, rank-2 complex `T` tensor within `MaxTensorExtents`. A leading batch axis is not accepted: multi-image throughput is an application concern. The operator rejects a non-`pga` algorithm, an unset `azimuth_axis`, a non-conforming image, a missing CUDA stream, and `[live]` values outside their valid ranges. Because `num_iterations` is `[live]` while pools are sized at graph build, it is capped at `kMaxAutofocusIterations` (255) and an override beyond that is refused.

`phase_correction` and `sharpness` are optional: the single PGA run writes them to operator-owned device scratch, and each wired port takes a device-to-device copy, so leaving either unwired costs nothing and cannot fail the tick. A full `image_out` queue fails `compute()`.

**`phase_correction` is pulse-indexed, which is not the index PGA works in.** The estimate is accumulated per azimuth-FFT bin with `fftfreq` semantics — bin 0 is DC, which is the *centre* of the dwell — while a pulse index runs monotonically across it, so pulse `p` sits at bin `(p - n/2) mod n` and the emitted vector is `fftshift(phase_acc)`. `image_out` is corrected in the spectrum domain, where the bin order is already right, so only the emitted vector is shifted. The sign is the error, not the correction: `PhaseCorrectionOp` with `sense: subtract` removes it as emitted.

A measured **one-sample step at exactly `floor(n/2)`** in the emitted vector, 473x to 1518x the median sample-to-sample step on Kourou, is an open item against this shift or against `PhaseResampleOp` — see [`OPEN_ITEMS.md`](OPEN_ITEMS.md) §1.

Neither property is dimensionally checkable and both were wrong until 2026-10-27, in opposite directions that largely cancelled — see the app [README](../../../applications/sar/autofocus/README.md) for the measurement and `SarAutofocusGraph.Figure2OversampledDiagnostics` for the test that pins it.

## Use and verification

```yaml
autofocus:
  azimuth_axis: u
  num_iterations: 4
  window_fraction: 0.5
  min_scatterers: 16
```

The C++ tests inject a known quadratic phase onto a synthetic range-Doppler
image and assert recovery up to an affine term, on either azimuth axis; they
also cover both pass-through guards, sharpness monotonicity, float/double
agreement, and per-tick (non-latching) `[live]` override behaviour. The
estimate's transfer onto a differently formed product is tested with the
application that owns the Figure 2 path. See the
[SAR operator contract](../../../docs/planning/SAR/OPERATOR_API.md).
