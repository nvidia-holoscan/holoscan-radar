# AutofocusOp — open items

## Two defects the quadratic-only sweeps could not see (fixed 2026-10-30)

Every accuracy measurement before 2026-10-29 injected a **pure quadratic**, and
a quadratic is **even about the dwell centre**. Both of the following are
invisible on an even error, and on an odd one they are indistinguishable from a
global sign flip, which `AutofocusScoreOp`'s sign search absorbs. Only an error
that is neither even nor odd exposes them — a sinusoid at 45° scored **4.92 rad
against a 4.75 rad baseline**, no recovery at all, reported with a normal sign.

1. **The emitted phase had a constant offset between the two halves of the
   aperture.** The gradient was integrated straight up the azimuth-FFT bin
   axis, but that axis is the pulse axis rotated by half the aperture: bins
   `n/2-1` and `n/2` are *not* adjacent pulses, so the gradient between them
   carries nothing, while bins `n-1` and `0` are adjacent and were being
   skipped (`lue_kernel` zeroed that pair). Integrating through the wrap gave
   the second half of the aperture a constant of integration taken across it.
   Measured directly on the dumped estimate: a **17.34 rad step at pulse
   n/2** on an odd error and 2.33 rad on an even one, against a 0.008 rad
   median step elsewhere, with the two half-means equal and opposite. Fixed by
   integrating outward from DC in both directions and computing the wrap pair.
   This is also the "one-sample step at `floor(n_pulses/2)`" recorded below as
   unexplained — not a cosmetic seam but the signature of the offset.

2. **The emitted phase was time-reversed.** Azimuth is the Fourier conjugate of
   slow time and runs the opposite way, so pulse `p` reads bin
   `(n - 1 - p + n/2) mod n`; the mapping had the shift but not the reversal.
   Established by testing four candidate mappings against injected truth over
   a quadratic, a cubic, a sine and the 45° sinusoid — plain reversal wins on
   all four (0.181 / 0.219 / 0.186 / 0.178 rad) and no other candidate wins on
   any. **Only the exported vector was wrong:** the in-image correction
   multiplies bin `k` by `phase_acc[k]` in the spectrum domain and never
   consults the mapping, so WORKFLOWS Figure 3 was always right while Figure 2
   handed `PhaseCorrectionOp` a reversed correction. A prior fix
   (`f371b008`) removed a spurious negate and added the shift, and its own note
   recorded that the two "partially cancel on a symmetric quadratic" — which is
   exactly the blind spot that let the reversal survive.

**Effect, on Kourou at 8000 pulses, 8 iterations, 1000-sample first window:**

| injected shape | baseline | before | after |
| --- | --- | --- | --- |
| quadratic (even) | 2.400 rad | 0.201 rad | 0.181 rad |
| cosine, 1 cycle (even) | 5.691 rad | 0.160 rad | 0.183 rad |
| cubic (odd) | 1.217 rad | **1.602 rad** | 0.219 rad |
| sine, 1 cycle (odd) | 3.564 rad | **3.818 rad** | 0.186 rad |
| sinusoid, 45° | 4.748 rad | **4.922 rad** | 0.178 rad |

Over the broader shape sweep (`--grid shapes --scenes kourou --pulses 8000
--window-samples 1000`), **3 of 11 rows were inside the 0.785 rad hard floor
before and 10 of 11 after**: every polynomial and every sinusoid to 6 cycles
passes, 12 cycles is marginal at 0.689 rad, and of the random walks only the
0.05 m one (a 20 rad baseline) is still harmful. An earlier revision of this
file said 8 of 11 with all three walks harmful; those three rows were measured
between the two fixes and are superseded — see §7.

Two tests changed with the fix, both because they encoded the first defect.
`injected_phi()` built its "quadratic" against the raw bin index, which put a
**7.56 rad jump between adjacent pulses 31 and 32** against a 0.125 rad median
— a phase error that teleports most of a wavelength between consecutive pulses
— and the estimator passed by reproducing it. The accuracy assertions also ran
in bin order, which round-trips through the very mapping under test and lets a
wrong mapping cancel out of its own comparison; they now assert in pulse order,
the domain the port is indexed by. `RecoversOddAndEvenErrorsAlike` is the new
gate: it asserts that the estimator has no symmetry preference, which is what
a pass/fail on one shape cannot pin.

## The scene axis: the fixes hold on three more collects (2026-10-30)

Every number above and below came from Kourou, and Kourou was **screened** for
PGA — `applications/sar/autofocus/config/autofocus.yaml` says so: an isolated
corner reflector (221× over its annulus) lit across the whole dwell, picked
with a tool that was never ported. The operator's precondition is satisfied
there by construction, so a number measured only on it describes the estimator
under ideal conditions rather than the estimator. Three more collects now have
configs beside it, and `sweep.py --scenes all` runs the grid across them.

Symmetry grid, 8000 pulses, 1000-sample first window, 0.02 m injected:

| scene | collect | quadratic | cosine | cubic | sine | 45° |
| --- | --- | --- | --- | --- | --- | --- |
| Kourou (screened reflector) | UMBRA-05 | 0.181 | 0.183 | 0.219 | 0.186 | 0.178 |
| Atlanta (airport) | UMBRA-07 | 0.093 | 0.101 | 0.081 | 0.177 | 0.165 |
| Houston (dense urban) | mode02 | 0.065 | 0.099 | 0.044 | 0.059 | 0.107 |
| Imperial Valley (agriculture) | Capella SM | refused | refused | refused | refused | refused |

Three readings, in descending confidence:

1. **The two fixes are not a Kourou artifact.** Three independent collects, at
   two frequencies and three scene types, recover all five symmetries to
   0.04–0.22 rad. Before the fixes the 45° column was no recovery at all.
2. **Screening bought reliability, not accuracy — Kourou is the *worst* of the
   three that work.** Urban scenes beat it by a factor of two to four. A single
   isolated reflector gives the gradient estimate one range bin's worth of
   phase history to average; a city gives it hundreds. This is worth saying
   because the config comment presents Kourou as the favourable case, and for
   accuracy it is not — which also means the residuals quoted throughout this
   file are not flattered by the scene.
3. **Agriculture is refused outright, not mis-corrected.** All five shapes hit
   the sharpness guard at iteration 0 and the image is returned untouched.
   That is the designed response to a scene with no dominant scatterer, and it
   is the first evidence that the guard does its job on real clutter rather
   than only on synthetic overreach. Note the asymmetry it creates in any
   tally: a refusal leaves the full injected error in the image, so it counts
   as outside the floor, but the operator did not do the damage — it declined.
   `sweep.py` reports those rows as `refused` rather than `HARMFUL` for that
   reason, and the distinction is only visible because the residual comes back
   *equal to the baseline* to several digits.

   At 2000 pulses the same scene partially works (quadratic 0.169, cosine
   0.293) and only the cubic is refused, so the refusal is not a property of
   the scene alone but of scene and aperture together: the longer the dwell,
   the further distributed clutter decorrelates.

**What this does not establish.** All four are spaceborne X-band collects with
good ephemeris — the regime the contract says should stay on Figure 1. The
error is injected, so none of this tests a real uncompensated airborne platform,
which is the case autofocus exists for. And Imperial Valley is the only
distributed-clutter scene here, so "agriculture is refused" rests on one
collect.

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

   **With an error actually present the guard does not refuse — it accepts a
   partly-wrong estimate, and accuracy decays well before the refusal
   ceiling.** The table above is the clean collect, where the only thing to
   find is the collect's own small residual. Injecting a 0.02 m quadratic and
   holding the window at 1/8 of the aux width (`--grid apertures`) separates
   the two effects:

   | pulses | residual | removed | sharpness |
   | --- | --- | --- | --- |
   | 2000 | 0.194 rad | 12.4× | 5.06× |
   | 4000 | 0.184 rad | 13.0× | 4.17× |
   | 8000 | 0.181 rad | 13.2× | 8.90× |
   | 12000 | 0.292 rad | 8.2× | 7.73× |
   | 16000 | 0.442 rad | 5.4× | 4.70× |
   | 23375 (full) | 0.436 rad | 5.5× | 2.89× |

   Flat to 8000, then a knee: the π/8 acceptance target is left somewhere
   between 12000 and 16000 pulses, and every one of these is *accepted* —
   sharpness rises 2.9–8.9×, so the guard sees a genuine improvement and has no
   reason to refuse. The ceiling is therefore two different failures at two
   different lengths. Past ~16000 on a clean collect the guard refuses and
   autofocus contributes nothing, which is safe; past ~12000 with a real error
   it silently returns an estimate that is only half right, which is not. The
   second is the one that matters for a caller, and it is not visible from
   sharpness.

   **This also makes the priority note above wrong about the one workflow it
   excused.** The shipped preset for this app forms the full 23375 and lands at
   0.436 rad, outside the target on the easiest shape there is.

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

4. **~~A one-sample step at exactly `floor(n_pulses/2)`.~~ Resolved —
   that step was defect 1 above.** 1.832 rad at 8000 pulses, 0.700 at
   16000 and 1.799 at 23375, against a median sample-to-sample step of
   0.0012–0.0018 rad. It was recorded here as narrow enough to barely
   move the RMS and therefore low priority, which was the wrong read:
   a one-sample seam at the dwell centre is what a constant offset
   between the two aperture halves looks like, and on an error that is
   not even about that centre the offset *is* the whole error. Noted
   at the time that the `fftshift(phase_acc)` emit was the first
   suspect; it was, along with the integration feeding it.

5. **`num_iterations` is bounded at 255 by the contract's `uint8`,
   which has not been exercised anywhere near that.** The default is 8
   and the sweeps above use 4 except where noted. Whether the
   guard-and-keep behaviour is still sensible over tens of iterations
   is unknown.

6. **The first window must be sized to the error magnitude, and past
   roughly 100 rad no width works well.** Residual in radians at 8
   iterations on Kourou / 8000 pulses (`--grid windows`), against the
   0.39 rad target and 0.785 rad hard floor. Magnitudes are the
   affine-removed RMS the scorer reports; an earlier revision of this
   table labelled the same five errors by their peak (5.4 / 26.8 /
   107.3 / 214.6 rad), which is 2.24× the RMS:

   | injected | 350-sample window | 1000 | 2000 | 4000 |
   | --- | --- | --- | --- | --- |
   | 0 (clean collect) | 0.150 | 0.168 | 0.169 | refused |
   | 2.400 rad | 0.142 | 0.181 | 0.184 | refused |
   | 12.000 rad | 0.274 | 0.203 | 0.239 | 0.366 |
   | 48.001 rad | 0.618 | **0.305** | 0.485 | 0.787 |
   | 96.002 rad | **45.73** | **6.57** | **2.87** | **2.04** |

   Four readings, in descending confidence.

   **The optimum width tracks the error magnitude, monotonically.**
   350 samples is best at 0 and 2.4 rad, 1000 at 12 and 48 rad, and
   4000 at 96 rad, with a shallow penalty either side. That is the
   relationship the item name asserts, and it is the first measurement
   of it that is a complete grid rather than a diagonal.

   **Oversizing is cheap, undersizing is not.** Across the four rows
   that work at all, going one step too wide costs at most 160 mrad
   while going one step too narrow costs up to 39 rad (the 96 rad row,
   350 vs 1000). Since an integrator *does* know their error budget
   from INS grade, dwell time and wavelength, sizing generously from
   the budget is sound and needs nothing from the operator. An earlier
   framing of this item claimed the caller cannot know the magnitude;
   that was wrong, and this is why.

   **The sharp non-monotone optimum is gone — it was the reversal
   bug.** The previous revision found 1000 samples passing at 0.274
   and 2000 failing at 1.181 one doubling away, called it unexplained,
   and made it the main argument for *deriving* the width rather than
   configuring it, on the grounds that no budget lands inside a window
   that tight. On the fixed estimator that row is 0.305 / 0.485 — a
   gentle penalty, not a cliff — and the argument goes with it.

   **At 96 rad (about 15 wavelengths RMS) nothing works.** The best
   cell is 2.04 rad, still 2.6× the hard floor, and `sharpness`
   reports 14–19× across the whole row with the guard terminating
   normally, so nothing marks it as a failure (see *no runtime quality
   signal* below). A caller whose budget reaches this magnitude is not
   served by PGA as implemented. This is the one conclusion the fixes
   did not move.

   **Where 4000 samples "refuses"** on the two smallest rows, the
   guard is catching the oversizing rather than the estimator
   degrading quietly — the right failure of the two, and the reason
   the top-left of this table is safer than it looks.

   **What is still thin.** Every cell is one collect (UMBRA-05
   Kourou), one aperture (8000 pulses) and one error shape (a pure
   quadratic). The scene axis above shows the *symmetry* results hold
   on three more collects, but this width grid has not been re-run
   across them, and the shape × width grid (`--grid shape-windows`)
   has not been re-run since the fixes at all.

   **What is open:** whether the contract should publish a
   window-sizing relationship against error budget and declare a
   magnitude precondition. Deriving the width from the measured
   azimuth extent of the selected scatterers remains possible, but the
   measurement that motivated it no longer stands, so configured
   guidance is now the cheaper answer. Note our app config hard-codes
   `window_fraction: 0.0438` (the 350-sample column) with a comment
   calling it tuned; on this table that is the *best* cell for the
   clean collect it ships against, and the worst for everything above
   12 rad.

7. **Random-walk error is recoverable up to about 4 rad and harmful beyond
   it.** Re-measured on Kourou at 8000 pulses after both fixes:

   | injected | baseline | 350-sample window | 1000-sample window |
   | --- | --- | --- | --- |
   | 0.002 m rms | 0.805 rad | 0.245 rad | 0.242 rad |
   | 0.01 m rms | 4.024 rad | 0.569 rad | 0.587 rad |
   | 0.05 m rms | 20.120 rad | 17.057 rad (FAIL) | 23.547 rad (HARMFUL) |

   **An earlier revision of this file reported 0.964 / 4.961 / 22.674 rad here
   and concluded that walks are harmful at every magnitude. That was wrong.**
   The baselines match these to three decimals, so the injected truth was
   identical and the estimator is what changed: those residuals sat at
   1.13–1.23× their baselines, between leaving the image alone (1.0×) and
   applying a wholly uncorrelated correction of the same size (1.41×), which is
   what a *reversed* estimate does to an error that is neither even nor odd.
   They were measured between the two fixes above. A random walk is the shape
   least protected by symmetry, so it was hit hardest and recovered most.

   What survives is the top row, and the open question narrows to the same two
   candidates as before, now for 20 rad of broadband error rather than all of
   it. The error model may be unphysical: `ErrorProfile` drives the walk with
   white noise, so it carries content down to the pulse-to-pulse scale, whereas
   a real platform is a mass on a spring and its INS error is heavily
   band-limited. Or PGA's geometrically shrinking window genuinely cannot
   represent broadband error of that magnitude. **Next step is a band-limited
   walk**, which distinguishes the two: if accuracy tracks the cutoff, the
   model was the problem.

   The lesson about method is worth keeping separate from the finding: this
   number was published from a working tree mid-fix, and it stood in the file
   as the one known remaining defect. Sweep rows are cheap to re-run and
   expensive to trust; the re-measured ones above all come from the committed
   tree.

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
