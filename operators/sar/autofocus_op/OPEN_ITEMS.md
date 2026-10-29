# AutofocusOp — open items

## Implementation follow-ups

1. **The shipped `num_iterations: 4` fails on the error magnitudes this
   operator exists for.** The contract names airborne residual phase as
   the reason to opt in at all — "spaceborne collections with good
   ephemeris generally stay on Figure 1" (`SAR/OPERATOR_API.md`) — but
   every measurement to date used a single 2 cm quadratic, 5.4 rad, on
   spaceborne collects. Sweeping magnitude at 8000 pulses:

   | injected | shipped (350-sample window, 4 iter) | 1000-sample window, 8 iter |
   | --- | --- | --- |
   | 5.4 rad | 0.149 rad | 0.201 rad |
   | 26.8 rad | 0.275 rad | 0.317 rad |
   | 107.3 rad | **1.365 rad** | **0.274 rad** |

   107 rad is about 17 wavelengths of line-of-sight error, which is
   ordinary for an airborne platform on a modest INS. At the shipped
   settings it misses not just the 0.39 rad target but the 0.785 rad
   hard floor — **while sharpness still reports a 17.1× gain**, so
   nothing in the operator's own output says it failed.

   Widening the first window alone does not fix it (1.365 / 0.433 /
   1.181 / 1.099 / 0.502 rad at 350 / 1000 / 2000 / 4000 / 8000
   samples — non-monotonic, and the best is still outside the target).
   **Iterating does:** at a 1000-sample first window, 8 iterations
   takes it to 0.274 rad, and 16 and 32 are identical to 8, so it
   converges rather than drifting. The cost is a little peak accuracy
   on small errors, 0.149 → 0.201 rad, which is the right trade when
   the magnitude is not known in advance — one setting then covers 5 to
   107 rad entirely inside the target where the shipped one fails above
   roughly 27 rad.

   **What raising the default costs, measured.** On the Kourou-sized
   auxiliary (2478 x 8000) the perf harness gives 51.8 ms median at 4
   iterations against 19.5 ms at 1 — about **10.8 ms per iteration**
   over an 8.7 ms floor (the strong-bin scan and the copy to
   `image_out`). So 4 → 8 is at most ~43 ms. In practice it is less,
   because **the guard makes the cost self-limiting**: `num_iterations`
   is an upper bound, and on the 8000-pulse collect the loop stops at 6
   whether it is set to 8 or to 16, for ~22 ms. The extra cost is
   therefore paid mostly where the extra iterations are actually doing
   something, and is near zero on an image that converges early —
   which is exactly the small-error case where 4 was already enough.
   For scale, product formation on that configuration is 37 ms and a
   video-SAR-rate frame budget would be about 458 ms.

   Open question for the team, since it is a default and therefore a
   contract surface: whether `num_iterations` should default higher
   than 4, or the guidance should simply say that an airborne opt-in
   must raise it. This is also what makes open item 5's 255 ceiling
   worth checking rather than theoretical.

2. **Beyond roughly 16000 pulses PGA produces an estimate that lowers
   sharpness, so the guard refuses it and autofocus contributes
   nothing.** This is the one real defect against how the operator will
   be used: an operator cannot hand-pick `pulse_range` per collect, and
   on Kourou (UMBRA-05) the useful aperture ceiling sits somewhere
   between 16000 and 20000 of the collect's 23375 pulses. **Lower
   priority than it looks:** every shipped workflow forms below that
   ceiling — `video_sar` at 8262 pulses, `spotlight` at 11500,
   `tiled_strip` at 15905 — so only this app's own full-collect preset
   exceeds it.

   Measured on the clean collect — no injected error, so the number is
   the magnitude of what PGA estimates for the collect's own residual
   motion, and `SAR_AF_IGNORE_DROP` (below) exposes the estimate the
   guard refuses:

   | pulses | as shipped | with the guard bypassed | sharpness |
   | --- | --- | --- | --- |
   | 8000 | 0.142 rad, accepted | — | 1.10× |
   | 12000 | 0.000, refused | 0.265 rad | 1.00× |
   | 16000 | 0.507 rad, accepted | — | 1.49× |
   | 20000 | 0.000, refused | 1.166 rad | 1.00× |
   | 23375 (full) | 0.000, refused | 1.335 rad | 1.00× |

   Where the guard accepts, the estimate is real: at 16000 applying it
   raises sharpness 1.49× on an already-formed clean image, so there
   genuinely was that much error. Where it refuses, applying the
   estimate lowers sharpness, and the refused estimate is large
   (1.17–1.34 rad) rather than marginal — PGA is not failing to find a
   small error, it is confidently finding a wrong one.

   **The shape of this is the standard long-aperture PGA problem**: one
   estimate across a very long dwell, with a single geometrically
   shrinking window, rather than a coarse-to-fine or sub-aperture
   schedule whose estimates are composed. Fixing it is an algorithm
   decision for the team, not a parameter change — `min_scatterers`
   (16 / 64 / 256 / 718) and `window_fraction` (512 / 1024 / 2048 /
   4096 absolute samples) each move the result by less than they move
   each other, and `num_iterations` 1 / 2 / 3 / 4 / 6 does not rescue
   it either, so there is no setting of the shipped parameters that
   makes a long aperture work.

3. **`AutofocusScoreOp`'s residual conflates the injected error with the
   collect's own, and that invalidated an earlier reading of this
   file.** The score is taken against the injected truth alone, but PGA
   also estimates whatever real error the collect carries — correctly
   — and is charged for it. Subtracting a clean-collect baseline
   separates the two, and the estimator is far better than the raw
   number suggests:

   | pulses | residual vs injected truth | the collect's own | estimator alone |
   | --- | --- | --- | --- |
   | 8000 | 0.149 rad | 0.142 rad | **0.042 rad** |
   | 16000 | 0.692 rad | 0.507 rad | **0.346 rad** |

   Both are inside the 0.39 rad target of `SAR/TESTING.md` once the
   collect's own error is removed, where the raw 0.692 reads as a clear
   failure. **So the apparent "accuracy degrades with aperture length"
   is mostly the collect's uncompensated motion growing with aperture
   length, which is physically expected, not the estimator decaying.**
   The app's validation apparatus should run the clean baseline and
   subtract it rather than leaving this to be rediscovered; until it
   does, read any single residual as an upper bound on estimator error.

4. **A one-sample step at exactly `floor(n_pulses/2)` in the emitted
   estimate, at every aperture length.** 1.832 rad at 8000, 0.700 at
   16000 and 1.799 at 23375, against a median sample-to-sample step of
   0.0012–0.0018 rad — 473× to 1518×. The two halves are not offset
   from each other (half-means are within ±0.06 rad), so it is a seam
   in one sample rather than a constant. It is narrow enough to barely
   move the RMS. The dwell centre is where the bin-to-pulse
   `fftshift(phase_acc)` wraps, so that emit is the first suspect;
   `PhaseResampleOp` is second but the aux azimuth extent equals the
   aperture on these runs, which makes it an identity copy. One
   explanation is ruled out: it is not a residual linear term left in
   bin order, which would map to exactly this sawtooth — detrending in
   bin order and mapping back leaves the residual and the step
   unchanged to four decimal places.

5. **`num_iterations` is bounded at 255 by the contract's `uint8`,
   which has not been exercised anywhere near that.** The preset uses 4
   and the sweeps above use 4 except where noted. Whether the
   guard-and-keep behaviour is still sensible over tens of iterations
   is unknown.

## What is not wrong

**The sharpness guard.** Forcing the refused iterations through at the
full aperture takes the residual from 0.332 to 1.295 rad, so the
iterations it declines really do diverge. It is doing its job.

**The estimator on a clean synthetic scene.**
`AutofocusOp.AccuracyIsFlatOverAzimuthExtentOnACleanScene` sweeps the
azimuth extent 16× at a fixed ~64-column window and finds it flat and
near-exact, 0.002 to 0.004 rad with swing within 2%. The long-aperture
defect needs a real collect to see: CI has only the generated 5-pulse
fixture, far too small for PGA to recover anything.

**Sharpness as a *safety* signal, though not as a quality metric.**
Across every configuration measured, the guard either accepted an
estimate that was real or refused one that was harmful. What sharpness
cannot do is rank two accepted estimates: at 16000 pulses it reads
1.49× on the clean collect, the best of any length, while that same
configuration carries the largest real error.

## Diagnostics

`SAR_AF_IGNORE_DROP` (environment, unset by default) makes the stage
keep an estimate that lowered sharpness instead of discarding it and
stopping. It exists because the guard otherwise leaves a rejected
estimate unobservable — `phase_acc` is never accumulated, so the
emitted correction is zero and anything scoring it downstream sees the
uncorrected error rather than the estimate that was refused. It is a
diagnostic, not a tuning knob: the contract says the guard holds, and
setting this will leave the image worse.

To reproduce any of the above, uncomment the `position_error` block in
the [autofocus app config](../../../applications/sar/autofocus/config/autofocus.yaml),
set `cphd_reader.pulse_range: [0, N]` and
`aperture_manager.max_aperture_pulses: N`, and run. A `polynomial` of
`[0, 0, 0.0]` injects nothing and scores PGA's estimate of the
collect's own error. `AutofocusScoreOp` logs the affine-removed RMS,
and `dump: true` with `output.path` writes `af_expected_phase.npy` and
`af_estimated_phase.npy` beside the image. Note the key is
`output.path`; `dump` itself is only a boolean.
