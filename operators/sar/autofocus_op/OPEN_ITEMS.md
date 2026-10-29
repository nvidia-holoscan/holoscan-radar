# AutofocusOp — open items

## Implementation follow-ups

1. **The estimate degrades with aperture length, non-monotonically, and
   at 16000 pulses it leaves the acceptance target.** Measured on
   Kourou (UMBRA-05) with a known 2 cm look-axis quadratic injected
   (`position_error`, 5.365 rad peak-to-peak of truth), 2048² product,
   4 iterations, varying only `pulse_range` / `max_aperture_pulses`.
   Residual is `AutofocusScoreOp`'s affine-removed RMS against the
   injected truth; the target is ≤ 0.39 rad (`SAR/TESTING.md`).

   | pulses | residual | swing | sharpness | iterations kept |
   | --- | --- | --- | --- | --- |
   | 8000 | 0.149 rad | 1.09 | 8.66× | all 4 |
   | 12000 | 0.283 rad | — | 8.94× | 3 of 4 |
   | 16000 | **0.692 rad** | 1.31 | 5.28× | all 4 |
   | 20000 | **0.428 rad** | — | 2.73× | 1 of 4 |
   | 23375 (full) | 0.332 rad | 1.25 | 2.84× | 1 of 4 |

   **Sharpness gain is anti-correlated with phase accuracy here, so it
   must not be read as a quality metric.** 16000 pulses accepts all
   four iterations and is the worst configuration by phase truth;
   the full aperture accepts one and is more than twice as accurate.

   Decomposing the residual separates three distinct defects:

   - **A curvature over-estimate that appears at some aperture lengths
     and not others.** At 16000 the estimate over-curves by 31% and a
     best-fit quadratic accounts for **85%** of the residual (0.636 of
     0.692 rad); remove it and 0.272 rad of ripple is left. At 8000 and
     23375 the quadratic term is ~0.01 rad, i.e. the curvature is
     right. Note the sign: PGA *over*-reports swing here, where the
     narrow-window operator tests under-report it (0.80–0.99 of truth),
     so this is not the same effect as `window_fraction` swing loss.
   - **Broadband ripple that grows with aperture length** — 0.145 /
     0.272 / 0.331 rad at 8000 / 16000 / 23375. At the full aperture
     this *is* the entire residual and consumes most of the 0.39 rad
     budget, so the full-aperture pass is not comfortable.
   - **A one-sample step discontinuity at exactly `floor(n_pulses/2)`,
     at every aperture length.** 1.832 rad at 8000, 0.700 at 16000 and
     1.799 at 23375, against a median sample-to-sample step of
     0.0012–0.0018 rad — 473× to 1518×. The two halves are not offset
     from each other (half-means are ±0.06 rad), so this is a seam in
     one sample rather than a constant. It is narrow enough to barely
     move the RMS and is almost certainly a fencepost at the dwell
     centre in the `fftshift(phase_acc)` emit or in `PhaseResampleOp`'s
     central-`num_pulses` extraction. **Cheapest of the three to chase
     and the most clearly a bug.**

   **The sharpness guard is correct and is not the problem.** Forcing
   the refused iterations through with `SAR_AF_IGNORE_DROP` (see below)
   at the full aperture takes the residual from 0.332 to 1.295 rad and
   sharpness from 2.84× to 2.02×, so the iterations it declines really
   do diverge. The operator is behaving as specified.

   **Supersedes the earlier reading of this item, which was measured
   without ground truth and drew the wrong conclusion.** That sweep ran
   the *clean* collect, where no error is injected, and read the
   1.000× sharpness results as the estimator failing. They are correct
   behaviour: the clean full-aperture image is already focused — its
   sharpness (6.159e20) is higher than the best the injected-error run
   recovers (5.697e20) — so the first estimate can only make it worse
   and the guard refuses it. Nothing is wrong at the full aperture on
   clean data; what is wrong is the accuracy trend above.

   Reproduce with a `position_error` block over the desired
   `pulse_range`; `AutofocusScoreOp` reports the residual and, with
   `dump: true` plus `output.path`, writes `af_expected_phase.npy` and
   `af_estimated_phase.npy` beside the image for the comparison above.

2. **`num_iterations` is bounded at 255 by the contract's `uint8`,
   which has not been exercised anywhere near that.** The preset
   uses 4 and the sweeps above use 4. Whether the guard-and-keep
   behaviour is still sensible over tens of iterations is unknown.

## Diagnostics

`SAR_AF_IGNORE_DROP` (environment, unset by default) makes the stage
keep an estimate that lowered sharpness instead of discarding it and
stopping. It exists because the guard otherwise leaves a rejected
estimate unobservable — `phase_acc` is never accumulated, so the
emitted correction is zero and anything scoring it downstream sees the
uncorrected error rather than the estimate that was refused. It is a
diagnostic, not a tuning knob: the contract says the guard holds, and
setting this will leave the image worse.
