# `AutofocusOp`

## Purpose

Estimate and remove a residual, scene-common phase error along the azimuth axis of a formed image.

## Responsibilities

- Estimate a per-pulse phase error by phase gradient autofocus (PGA) on strong, isolated scatterers.
- Apply that estimate in-image, and publish it so a different formation can be corrected instead.

## When to use

Use when a formed image is defocused by a residual phase the geometry model did not capture. This is a separate, omittable stage rather than a mode inside image formation, and it is off by default. Correcting in-image is only meaningful when the product is itself slow-time-conjugate (a slow-time-conjugate grid); on a ground or DEM product the estimate instead reaches formation through `PhaseCorrectionOp` on a replayed aperture (the estimation/reformation path). Input and output image names differ because HSDK 5 requires unique port names.

Declarations and parameter defaults: [autofocus_op.hpp](autofocus_op.hpp). Behavioral requirements are documented below.

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
| Out | `phase_correction` | real `T`, radians | `(num_az)` | Every tick, optional | Pulse-ordered estimate, one value per azimuth sample |
| Out | `sharpness` | real `T` | `(num_iterations + 1)` | Every tick, optional | Focus metric per iteration |
| Out | `termination` | `int32` | `(1)` | Every tick, optional | Why the loop stopped |
| Out | `iterations` | `int32` | `(1)` | Every tick, optional | Iterations actually accepted |

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
| `num_iterations` | `int64` | 8 | `[live]` | Refinement iterations (an upper bound; the stage stops early) |
| `window_fraction` | `float64` | 0.5 | `[live]` | Fraction of azimuth extent in the first window |
| `min_scatterers` | `int64` | 16 | `[live]` | Strong range bins used: the brightest N are kept, and fewer than N refuses |
| `max_correction` | `float64` | uncapped | `[live]` | Peak-to-peak ceiling in radians |

## Operation and synchronization

Each iteration selects range bins whose azimuth peak stands well above the row mean, circularly shifts each selected peak to the azimuth centre, applies a window that shrinks geometrically across iterations, and estimates the phase gradient by the linear unbiased estimator. The integrated gradient is accumulated across iterations and applied to the image spectrum. `sharpness` records the focus metric before each iteration and after the last, so a caller can see whether the estimate helped.

**`window_fraction` narrows the image support, never the transform.** The gradient FFT is always the full azimuth extent, zeros outside the window included. Packing the window into a shorter transform is tempting — it is `window_fraction` times the work — but it decimates the phase history by the inverse of that fraction, so each gradient difference spans that many samples and the estimator's small-angle form saturates. That cost nothing on a gentle error and up to 5.5x the residual on a steep one; see the app [README](../../../applications/sar/autofocus/README.md).

PGA is blind to a constant and a linear phase ramp: both translate the image rather than blur it. Compare an estimate against a truth only after removing an affine fit from each.

Two guards cause a pass-through, in which the image is copied unchanged and `phase_correction` is all zeros: fewer strong scatterers than `min_scatterers`, and an estimate whose peak-to-peak span exceeds `max_correction`. Naming `azimuth_axis` also asserts that the image is conjugate along that axis; that assertion is not dimensionally checkable, so a wrong axis is not an error — it simply fails to recover, which is why the parameter is required and has no default.

A third guard discards an iteration whose estimate lowers sharpness and stops there, keeping the last accepted estimate. It is a faithful proxy: forcing the refused iterations through on Kourou at the full aperture takes the residual against injected truth from 0.332 to 1.295 rad. Because it stops before accumulating, a rejection at iteration 0 emits a zero correction and the refused estimate cannot be seen from outside — set `SAR_AF_IGNORE_DROP` in the environment to keep it anyway and score it. That is a diagnostic, not a tuning knob; it will leave the image worse.

The image must be a contiguous, CUDA-accessible, rank-2 complex `T` tensor within `MaxTensorExtents`. A leading batch axis is not accepted: multi-image throughput is an application concern. The operator rejects a non-`pga` algorithm, an unset `azimuth_axis`, a non-conforming image, a missing CUDA stream, and `[live]` values outside their valid ranges. Because `num_iterations` is `[live]` while pools are sized at graph build, it is capped at `kMaxAutofocusIterations` (255) and an override beyond that is refused.

`phase_correction`, `sharpness`, `termination` and `iterations` are optional: the single PGA run writes them to operator-owned scratch, and each wired port takes a copy, so leaving any unwired costs nothing and cannot fail the tick. A full `image_out` queue fails `compute()`.

**`termination` and `iterations` are provenance, not quality.** `termination` is int32 and says why the loop stopped — `converged`, `budget_exhausted`, `refused_sharpness`, `refused_scatterers`, `refused_max_correction`, `stopped_scatterers` — and `iterations` is how many iterations' phase is actually in `phase_correction`, which is not how many were attempted, because each one is applied, scored, and discarded if sharpness fell.

They exist because **three independent paths return the image untouched and were indistinguishable from success on every other port**: iteration 0 lowering sharpness, fewer than `min_scatterers` bins passing the gate, and the `max_correction` ceiling. All three emit a zero `phase_correction` and a flat `sharpness` trace — which is also exactly what a correctly-focused image produces — and the only statement of what happened was a `HOLOSCAN_LOG_WARN` no consumer can read. `autofocus_refused()` collapses the three codes for a caller that only needs "was anything applied".

What they deliberately do not do is tell a caller whether the estimate is *good*. That is measured to be impossible from inside the operator, not merely unimplemented: coherence, convergence and gradient-against-limit were all built and scored against injected truth, and all three are blind — see [qualification notes](README.md). A `converged` code is not an accuracy claim.

**`phase_correction` is pulse-ordered, which is not the order PGA works in.** Its length is `num_az`, the extent of `azimuth_axis`: the operator sees only the image, so the pulse count is not available to it. The two lengths agree only on a grid formed at one azimuth sample per pulse; otherwise the caller resamples onto `num_pulses` before `PhaseCorrectionOp` (the autofocus app uses `PhaseResampleOp`).
The estimate is accumulated per azimuth-FFT bin with `fftfreq` semantics — bin 0 is DC, which is the *centre* of the dwell — while a pulse index runs monotonically across it.
The two differ by a half-aperture shift *and by the handedness of the axis*, because azimuth is the Fourier conjugate of slow time and runs the opposite way, so pulse `p` reads bin `(n - 1 - p + (n + 1) / 2) mod n`. `image_out` is corrected in the spectrum domain, where the bin order is already right, so only the emitted vector is remapped. The sign is the error, not the correction: `PhaseCorrectionOp` with `sense: subtract` removes it as emitted.

The reversal half of that was missing until 2026-10-30, along with a seam in the integration that fed it; both were invisible on a quadratic because a quadratic is even about the dwell centre. The **one-sample step at exactly `floor(n/2)`** previously recorded here as an open item was the seam, and is resolved — see [qualification notes](README.md).

Neither property is dimensionally checkable and both were wrong until 2026-10-28, in opposite directions that largely cancelled — see the app [README](../../../applications/sar/autofocus/README.md) for the measurement and `SarAutofocusGraph.Figure2OversampledDiagnostics` for the test that pins it.

## Use and verification

```yaml
autofocus:
  azimuth_axis: u
  num_iterations: 8
  window_fraction: 0.5
  min_scatterers: 16
```

The C++ tests inject a known quadratic phase onto a synthetic range-Doppler
image and assert recovery up to an affine term, on either azimuth axis; they
also cover both pass-through guards, sharpness monotonicity, float/double
agreement, and per-tick (non-latching) `[live]` override behaviour. The
estimate's transfer onto a differently formed product is tested with the
application that owns the Figure 2 path. See the
[shared architecture](../../../docs/ARCHITECTURE.md).

## Accuracy limits and open questions

PGA assumes dominant stationary scatterers persisting across the aperture. It hard-zeros the phase estimate outside occupied support, producing steps at the dwell edges when support is cropped. Zero fill, edge holding, and extrapolation are different algorithm choices; measure them across scenes before selecting one. Long-aperture behavior and window sizing against phase-error magnitude also remain open. A clean Kourou full dwell can refuse correction, while injected-error runs can accept an inaccurate estimate. Asymmetric error profiles and a band-limited random walk are needed to distinguish model bandwidth from estimator limits.

Coherence, convergence, gradient limits and sharpness were investigated without finding a reliable runtime accuracy signal. `termination` reports provenance only. Real-collect raw residuals also include the scene's own phase error; compare a clean-collect baseline before attributing all residual to injected-error recovery. Detailed historical measurements, including corrected phase-order/seam results, are recoverable at Git revision `6c261de7c153daa0c8a6be9d6829b149cfbdf41c`, path `operators/sar/autofocus_op/OPEN_ITEMS.md`; do not mix pre-fix sweep rows with current results.

Remaining test gaps include trajectory-error-to-image recovery (rather than phase injection alone), accuracy on an explicit-points conjugate lattice, and warning capture on refusal. The application resamples the estimate from `num_az` onto `num_pulses`; library ownership of that bridge remains undecided.

## Checkout diagnostic tools

These scripts are optional checkout tools, outside installed packages. Launch from the repository root with `uv run`; their inline declarations supply NumPy/PyYAML without changing application dependencies. They require built C++ examples and local CPHD data. Build `spotlight` for `pga_screen.py`, and `autofocus` for `sweep.py`, using the host CLI with `--language cpp`. Run the scripts in an SDK/CUDA environment where those binaries execute (for example the development container), with `uv` available. Use each script's executable-path override for the actual build tree (ordinary app builds use `build/<app>`, aggregate module builds use `build/holoscan-radar`).

```bash
uv run operators/sar/autofocus_op/tools/pga_screen.py --help
uv run operators/sar/autofocus_op/tools/sweep.py --help
uv run operators/sar/autofocus_op/tools/sweep.py --grid shapes \
  --binary build/autofocus/applications/sar/autofocus/cpp/autofocus
uv run operators/sar/autofocus_op/tools/sweep.py --grid symmetry --scenes all \
  --binary build/autofocus/applications/sar/autofocus/cpp/autofocus
uv run --no-project --with pyyaml python -m unittest discover -s operators/sar/autofocus_op/tools/tests -v
```

Screening measures isolation and persistence over disjoint dwell blocks; it cannot establish stationarity or guarantee PGA success. Sweeps run injected error profiles against the example app and require both a valid score and termination provenance. Failed, timed-out, nonfinite or incomplete campaigns fail rather than supplying accuracy results. Use `--base` for another app preset and inspect help for binary, pulse-count, window and scene options. `SAR_AF_IGNORE_DROP` exposes a refused estimate for diagnosis and may worsen the image; it is not a quality tuning knob.
