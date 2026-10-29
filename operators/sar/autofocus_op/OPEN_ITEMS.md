# AutofocusOp — open items

## Implementation follow-ups

1. **~~The shipped `num_iterations: 4` fails on the error magnitudes
   this operator exists for.~~ Resolved — the default is now 8.** The contract names airborne residual phase as
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

   **Resolved 2026-10-29: the default is now 8**, a contract change
   carried in `SAR/OPERATOR_API.md` and the operator README in the same
   commit. The alternative — leaving it at 4 and requiring an airborne
   opt-in to raise it in guidance — was rejected because it only
   protects a caller who reads the guidance, and the failure mode for
   one who does not is the bad one: a wrong estimate that `sharpness`
   reports as a 17x success. An app that needs a tighter latency bound
   still sets its own ceiling, statically or through the `[live]`
   `num_iterations` port. On the shipped 8000-pulse configuration the
   new default moves the residual 0.149 -> 0.161 rad and sharpness
   8.66x -> 8.78x. This is also what makes item 5's 255 ceiling worth
   checking rather than theoretical.

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
   which has not been exercised anywhere near that.** The default is 8
   and the sweeps above use 4 except where noted. Whether the
   guard-and-keep behaviour is still sensible over tens of iterations
   is unknown.

6. **The first window must be sized to the error magnitude, and past
   roughly 100 rad no width works well. TENTATIVE — one collect, one
   aperture, one error shape.** Residual in radians at 8 iterations on
   Kourou / 8000 pulses, against the 0.39 rad target and 0.785 rad
   hard floor:

   | injected | 350-sample window | 1000 | 2000 | 4000 |
   | --- | --- | --- | --- | --- |
   | 0 (clean collect) | 0.157 | 0.166 | — | — |
   | 5.4 rad | 0.161 | 0.201 | 0.176 | — |
   | 26.8 rad | — | 0.317 | — | — |
   | 107.3 rad | **1.035** | 0.274 | **1.181** | — |
   | 214.6 rad | **50.48** | **5.41** | **2.21** | **2.47** |

   Three readings, in descending confidence.

   **Oversizing is cheap below ~27 rad.** A 1000-sample window costs
   40 mrad against 350 on a 5.4 rad error and 9 mrad on a clean
   collect, with everything well inside target. So for a system whose
   error budget sits in that range — and an integrator *does* know
   their budget from INS grade, dwell time and wavelength — sizing the
   window generously from the budget is sound and needs nothing from
   the operator. An earlier framing of this item claimed the caller
   cannot know the magnitude; that was wrong, and the measurement
   above is why.

   **Above ~100 rad the optimum is sharp rather than a floor.** At
   107.3 rad, 1000 samples passes at 0.274 and 2000 fails at 1.181 —
   one doubling either side of the optimum is the difference. The
   non-monotonicity reproduces at both 4 and 8 iterations (4-iteration
   sweep: 1.365 / 0.433 / 1.181 / 1.099 / 0.502 rad at 350 / 1000 /
   2000 / 4000 / 8000 samples) so it is not run-to-run noise, and it
   is unexplained. A budget known to an order of magnitude does not
   land inside a window that tight.

   **At 214.6 rad (about 34 wavelengths) nothing works.** Widening
   from 350 to 2000 samples improves the residual from 50.48 to 2.21
   rad, an enormous gain, and 2.21 is still nearly 3x the hard floor;
   4000 is worse again. `sharpness` reports 11.1x to 17.6x across that
   whole row and the guard terminates normally, so nothing marks it as
   a failure (see *no runtime quality signal* below). A caller whose
   budget reaches this magnitude is not served by PGA as implemented.

   **Why this is tentative and what would settle it.** Every cell is
   one collect (UMBRA-05 Kourou), one aperture (8000 pulses) and one
   error shape (a pure quadratic) — the smoothest, most symmetric
   error there is, and the one a geometric window suits best. The
   sharp optimum at 107.3 rad in particular could be specific to that
   shape. Needed before any of this becomes contract guidance: the
   other error profiles (higher-order, sinusoidal, random-walk) and
   the other collects on disk.

   **What is open**, pending that: whether the contract should publish
   a window-sizing relationship against error budget and declare a
   magnitude precondition, or whether the first width should be
   *derived* from the measured azimuth extent of the selected
   scatterers — which is directly observable, the blur being the
   defocus — rather than configured at all. Deriving it matters most
   in the above-100 rad regime where the optimum is too sharp to
   configure. Note our own app config hard-codes `window_fraction:
   0.0438` (the 350-sample column) with a comment calling it tuned; on
   this table that is the worst cell in two of five rows.

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

## There is no runtime quality signal, and that looks structural

The operator cannot tell a caller whether its estimate is good. Three
candidate signals were implemented and measured against injected truth
at 8000 Kourou pulses, and all three are blind to the failure:

| injected | iter | first window | residual | verdict | coherence | converged? | max&#124;grad&#124; |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.4 rad | 4 | 350 | 0.149 rad | pass | 1.0000 | yes | 0.017 |
| 26.8 rad | 4 | 350 | 0.275 rad | pass | 1.0000 | yes | — |
| 53.6 rad | 4 | 350 | 0.342 rad | pass | — | **no** | — |
| 53.6 rad | 8 | 350 | 0.194 rad | pass | — | yes | 0.093 |
| 107.3 rad | 4 | 350 | **1.365 rad** | **fail** | 0.9993 | **no** | 0.152 |
| 107.3 rad | 8 | 350 | **1.035 rad** | **fail** | 0.9993 | yes | 0.152 |
| 107.3 rad | 8 | 1000 | 0.274 rad | pass | 0.9990 | yes | — |
| 214.6 rad | 8 | 350 | **50.48 rad** | **fail** | 0.9993 | yes | 0.109 |

1. **Phase-gradient coherence** — `|Σ_r g₁ conj(g₀)| / Σ_r |g₁||g₀|`
   over the selected range bins, the standard measure of whether the
   scatterers agree about the gradient. It saturates: 1.0000 on the
   passing cases and 0.9993 on the 50 rad catastrophe. The scatterers
   agree perfectly *and are perfectly wrong* — the error is a
   systematic bias common to all of them, which is exactly what a
   coherence statistic is constructed to be insensitive to.
2. **Convergence** — whether the loop stopped on the guard or ran out
   of budget. Falsified in both directions by the table: 107.3 rad at 8
   iterations converges and still fails at 1.035 rad, while 53.6 rad at
   4 iterations does not converge and passes at 0.342 rad.
3. **LUE gradient against its unambiguous limit** — the aliasing
   hypothesis, that a large error wraps the small-angle gradient
   estimate. Measured max |grad| is 0.001–0.152 rad/bin against a π
   limit, so nothing is near wrapping, on the failures included. Not
   the mechanism.

**Why this is unlikely to yield to a better metric.** Candidate 2
generalizes: the next iteration's estimate *is* a fresh PGA pass on the
corrected image, so "does a second opinion agree" and "did it
converge" are the same question. On the 107.3 rad / 8-iteration row
the last two iterations estimate essentially nothing left to find
(max |grad| 0.0039 then 0.0010) with 1.035 rad still on the table. PGA
converges to a fixed point of its own sharpness-and-gradient
objective, and that fixed point is not the truth. The residual lives in
the null space of everything the estimator measures, which is why no
internal quantity sees it. Establishing accuracy appears to need
information the operator does not have — an independent phase
reference, a second estimator of a different family, or the known
truth the validation app injects.

None of the three was shipped; the measurement code was reverted. What
a caller has is `sharpness` as a safety signal, `max_correction` as a
blunt magnitude gate, and the guidance that a first window must be
wide enough for the error magnitude expected.

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
