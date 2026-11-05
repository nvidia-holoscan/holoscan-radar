# `AutofocusOp` handoff: what it does, why it exists, how it works

Written for a holoscan.radar developer who knows the module's conventions —
ports, `[live]` overrides, `MaxTensorExtents`, the contract docs — and has
worked on the SAR side of it, but has not been inside `AutofocusOp`. The
background section restates the SAR framing the operator depends on, in the
terms the code and contracts use; skip it if it is familiar.

## The one-paragraph version

`AutofocusOp` takes a formed SAR image that is blurred along one axis, works
out the phase error that caused the blur, and removes it. The error is one
phase value per transmitted pulse, common to every pixel in the scene, and it
comes from not knowing where the antenna was to a small fraction of a
wavelength. The operator estimates it with **phase gradient autofocus (PGA)**,
which reads the error off a handful of bright, isolated points in the image.
It emits both a corrected image and the estimated phase vector. In this
module's main use the phase vector is the product: it is applied to the raw
pulses, which are then formed a second time onto a different grid.

## The SAR framing autofocus depends on

**Two time axes.** A SAR collect is a sequence of pulses. *Fast time* runs
within one pulse and maps to range — how far away a reflector is. *Slow time*
is the pulse index and maps to where the platform was along its path. After
range compression, the data is a 2-D array of `(num_pulses, num_range_bins)`;
the module calls those rows *range profiles*.

**Forming an image.** Backprojection forms each pixel by asking every pulse
"what did you see at the distance from your antenna to this pixel?" and
summing the answers with a phase that undoes the round-trip travel. That sum
is coherent: it focuses only if the phase predicted for each pulse is right.
The predicted phase is `4π/λ · R`, with `R` computed from the recorded antenna
position.

**Why tiny navigation errors matter.** A line-of-sight position error `ΔR`
shifts the phase by `4π/λ · ΔR`. At X-band (λ ≈ 3 cm) one centimetre is about
4 radians, so the navigation has to be good to a few millimetres for the sum to
stay coherent. When it isn't, the error enters as an extra phase `φ(p)` on
pulse `p`, and that `φ` is the same for every pixel — it belongs to the pulse,
not the scene. This is the *scene-common, per-pulse phase error* the operator
estimates.

**What the error does to the image.** Take an image formed on a grid whose
rows are constant range from the aperture and whose columns are uniform in
bearing — a *polar* or range-Doppler grid. On such a grid the image's azimuth
axis is the Fourier conjugate of slow time; the code and contracts call this
a *slow-time-conjugate*, or just *conjugate*, grid. Because of that Fourier
relationship, a per-pulse phase that multiplies every scatterer's history by
the same `e^{jφ(p)}` convolves every point in the image with the same blur
kernel, the Fourier transform of `e^{jφ(p)}`.

What that kernel looks like depends on the shape of `φ` across the aperture.
The standard way to describe the shape is to expand `φ(p)` as a polynomial in
pulse position measured from the aperture centre, and to take the terms one
order at a time:

- A **constant** does nothing visible.
- A **linear** term shifts the whole image.
- A **quadratic** smears points into lines — classic defocus.
- **Higher orders** raise sidelobes and ghosts.

The first two produce no blur, so no image-based method can see them. That is
why every PGA comparison in the repo removes an affine fit first. It is also
the reason autofocus costs absolute geolocation: whatever linear error was
there is left there.

**Conjugate grids, and why there are two figures.** PGA works through the
Fourier relationship above, so it needs a conjugate grid. The module's usual
product is a ground or DEM lattice, which is not conjugate, so PGA cannot run
on it directly. Decision D20 in the
SAR contract resolves this by splitting *estimation* from *correction*. The
two resulting chains are Figures 2 and 3 in
[`WORKFLOWS.md` → Port-level wiring](WORKFLOWS.md#port-level-wiring), which
has the full port-level diagrams. The figure numbers are used throughout the
SAR docs and code comments.

- **Figure 2 (the main path, and what the autofocus app builds).** Retain the
  whole aperture. Form a small auxiliary image on a conjugate grid, estimate
  `φ` from it, apply `φ` to the *same* retained pulses with
  `PhaseCorrectionOp`, then form the real product once on whatever grid it
  needs. That is two formations per aperture, and it is why
  `ApertureManagerOp` must run with `max_aperture_pulses > 0` and
  `emit_mode: reform`. In outline:

  ```mermaid
  flowchart LR
    RC["SARRangeCompressionOp"] --> AM["ApertureManagerOp<br>(retain + reform)"]
    AM -->|"aperture replay"| BPA["BackprojectionOp<br>(auxiliary, conjugate grid)"]
    BPA -->|"image"| AF["AutofocusOp"]
    AF -->|"phase_correction (num_az)"| RS["app-owned resample"]
    RS -->|"phase_correction (num_pulses)"| PC["PhaseCorrectionOp"]
    AM -->|"same replay"| PC
    PC -->|"corrected pulses"| BPP["BackprojectionOp<br>(product, ground / DEM)"]
  ```

- **Figure 3 (a footnote).** If the product itself is already on a conjugate
  grid, the operator corrects it in place and `image_out` is the deliverable.

**Why it is off by default.** It costs a second formation and aperture-sized
retention, it gives up geolocation, and it depends on the scene containing
suitable bright points. With good navigation the answer is to leave it out.
Spaceborne collects mostly have good enough navigation; airborne ones often
don't, and that is the case this operator exists for.

## How the operator works

Code: `operators/sar/autofocus_op/`. `autofocus_op.{hpp,cu}` is the Holoscan
shell. `autofocus_stage.hpp` holds the algorithm, and its header comment
lists the steps; `run_autofocus()` is the entry point.

### Shape handling

The input is a single rank-2 complex image, `(num_v, num_u)`. A leading batch
axis is rejected; multi-image throughput is an application concern. The
caller names the azimuth axis with `azimuth_axis` (`u` or `v`). That parameter
is required with no default because a wrong axis is not dimensionally
detectable — it simply fails to recover. Naming it is also how the caller
asserts the image is conjugate; the operator cannot check that either.

Internally everything runs on a working cube with azimuth last: a copy for `u`
and a transpose for `v` (`copy_to_work` / `copy_from_work`). Below, `n_az` is
the azimuth extent and `n_range` the other axis.

### The per-image loop

Before iteration 0 the stage measures the image's **sharpness**, the sum of
`|z|⁴` over all pixels. A focused image concentrates energy into fewer, brighter
pixels, so sharpness rises as focus improves.

1. **Pick scatterers** (`peak_find_kernel`, `cap_strong_scatterers`). For each
   range row, find the brightest azimuth sample. Keep rows whose peak power is
   at least 10× the row mean (`kPgaMinPeakToMean`), then keep only the
   brightest `min_scatterers` of those. That parameter is **both a floor and
   a cap**. Fewer qualifying rows than the floor makes the operator refuse.
   The cap matters as much: on a dense scene nearly every row passes the gate,
   and averaging hundreds of mediocre rows dilutes the estimate toward zero.
2. **Centre and window** (`circshift_window_kernel`). Circularly shift each
   selected row so its peak sits at azimuth index 0, then zero everything
   outside a window of the current width around index 0. The first width is
   `window_fraction × n_az`, and it halves each iteration with a floor of 2.
   The window cuts away clutter so the estimate is driven by the scatterer.
3. **Go to the slow-time domain.** FFT each windowed row along azimuth at
   **full length**. The window narrows the image support, never the
   transform. Packing the window into a shorter FFT was a real bug: it
   decimates the phase history and saturated the estimator, costing up to
   5.5× the residual.
4. **Estimate the gradient** (`lue_kernel`). For each adjacent pair of bins,
   the phase difference averaged across the selected rows is the *linear
   unbiased estimator*, or LUE:
   `Σ_r Im(g[k+1]·conj(g[k])) / Σ_r |g[k]|²`.
   Every selected scatterer carries the same `φ`, so summing across rows
   reinforces it while each row's own content averages out.
5. **Integrate and detrend** (`integrate_detrend_kernel`). Sum the gradient
   back to a phase, **outward from DC in both directions**. The bin axis is
   the pulse axis rotated by half the aperture, so bins `n/2 − 1` and `n/2` are
   not neighbouring pulses, while bins `n − 1` and `0` are. A running sum up
   the bin axis puts a step at the dwell centre — small for an error that is
   even about the centre, and the whole error for an odd one. That was a real
   bug, fixed 2026-10-30. Integration covers only the bins holding 99% of the
   spectral energy (`kPgaOccupiedEnergy`), or all of them once that support
   reaches 85% (`kPgaAllBinsFrac`). The linear and constant terms are then
   removed, since they are unobservable.
6. **Apply and score** (`mul_phasor_kernel`, `sharpness_kernel`). Multiply the
   full, unwindowed image's azimuth spectrum by `e^{−jφ}`, inverse-FFT, and
   re-measure sharpness. If sharpness did not fall, accept: add this
   iteration's `φ` into the accumulator. If it fell, restore the previous
   image from a backup and stop.

The loop runs for at most `num_iterations` iterations; it is an upper bound,
not a count. An image that stops improving stops early, so a large value
costs nothing on an easy image.

### Guards and the `termination` port

Three paths return the image untouched, with a zero `phase_correction`:

| Code | Why |
| --- | --- |
| `refused_scatterers` (3) | Fewer than `min_scatterers` rows passed the gate. |
| `refused_sharpness` (2) | Iteration 0 lowered sharpness, so nothing was accepted. |
| `refused_max_correction` (4) | The estimate's peak-to-peak exceeded `max_correction`, which suggests a failed estimate rather than a badly blurred image. |

The non-refusal codes are `converged` (0), `budget_exhausted` (1) and
`stopped_scatterers` (5). `iterations` reports how many iterations' phase is
actually in the output, which is not how many were attempted.

These two ports exist because a refusal used to be indistinguishable from
success: a zero correction and a flat sharpness trace are also exactly what an
already-focused image produces. `autofocus_refused()` collapses the three
refusal codes for a caller who only needs a yes or no.

They report **what happened, not whether the answer is good.** There is no
runtime quality signal, and the open items explain why that looks structural.
Gradient coherence, convergence, and gradient-versus-aliasing-limit were all
built and scored against injected truth. All three are blind: a 214.6 rad
error converged with 0.9993 coherence and a 50 rad residual. The scatterers
agree perfectly and are perfectly wrong. Never read `converged` as an
accuracy claim.

### Output indexing and sign

This is the subtlest part of the operator and where both historical bugs
lived.

PGA works in azimuth-FFT bins with `fftfreq` order: bin 0 is DC, which is the
*centre* of the dwell. A pulse index runs monotonically from the first pulse
to the last. The two differ by a half-aperture rotation **and a reversal**,
because azimuth is the conjugate of slow time and runs the opposite way.
`fftshift_phase_kernel` remaps the emitted vector so element `p` reads bin
`(n − 1 − p + (n + 1)/2) mod n`. `image_out` is corrected in the spectrum
domain, where bin order is already right, so only the emitted vector is
remapped.

The emitted vector therefore has **pulse ordering and `n_az` elements**. It
equals the per-pulse phase when the auxiliary grid has one azimuth sample per
pulse — the default for the app's polar grid. When the two differ, the app's
`PhaseResampleOp` maps it onto the pulse count before `PhaseCorrectionOp`.

The emitted value is **the error, not the correction**:
`PhaseCorrectionOp` with `sense: subtract` removes it as emitted.

### Holoscan-side mechanics

- **Ports.** `image` drives the tick. `num_iterations`, `window_fraction`,
  `min_scatterers` and `max_correction` are `[live]` overrides, joined with
  `Exact{}` and `SafeDefault{}`, effective for one tick, never latched.
- **Outputs.** `image_out` is emitted every tick, and a full queue fails
  `compute()`, which is why the app drains it with a `NullSinkOp`.
  `phase_correction`, `sharpness`, `termination` and `iterations` are
  optional: one PGA run writes to operator-owned scratch, and each wired port
  copies out. Leaving any unwired costs nothing.
- **Sizing.** Give `max_image_v` and `max_image_u` in `MaxTensorExtents`. The
  constructor must not read `azimuth_axis`, so the `phase_correction` pool is
  sized from the larger of the two. `sharpness` has `num_iterations + 1`
  slots, and since `num_iterations` is `[live]` while pools are fixed at
  build, it is capped at `kMaxAutofocusIterations = 255`.
- **Precision.** One template parameter `T`, float or double, like most
  operators. Sharpness is accumulated in double regardless of `T`, because a
  single bright backprojected pixel overflows float32 in `|z|⁴`. Once two
  iterations both read `inf`, the accept/reject guard stops rejecting. That
  was fixed on this branch.
- **Synchronization.** The accept, reject and refuse decisions are made on the
  host, so `run_autofocus` synchronizes the stream inside the loop. That is
  deliberate, not an oversight.

## What we know about how well it works

The acceptance target from `SAR/TESTING.md` is a residual of **≤ π/8 ≈ 0.39 rad
RMS** after affine removal, with π/4 as the hard floor, and a corrected peak of
≥ 0.9 of the error-free peak.

- **Synthetic unit tests** recover a pure quadratic essentially exactly at full
  window, and odd and even errors equally well.
- **Real collects**, with error injected through the app's `position_error`
  block at 8000 pulses: Kourou, Atlanta and Houston all land between 0.04 and
  0.22 rad across quadratic, cosine, cubic, sine and mixed error shapes.
  Imperial Valley (agricultural, no isolated bright points) refuses every
  case, which is the correct behaviour.
- **Large errors.** The default rose from 4 to 8 iterations because 4 fails a
  107 rad error — about 17 wavelengths of line-of-sight error, ordinary for an
  airborne platform. At 4 iterations the residual is 1.365 rad, outside even
  the π/4 floor, while sharpness still reports a 17× gain. Widening the first
  window alone does not fix it; iterating does. With a 1000-sample first
  window, 8 iterations reaches 0.274 rad, and 16 or 32 give the same answer.
- **Only Kourou was screened, and it is not the favourable case.** Kourou was
  picked with `tools/sar_autofocus/pga_screen.py` for its isolated corner
  reflector; the other three collects were not screened. Screening bought
  reliability, not accuracy: Kourou is the *least* accurate of the three
  scenes that focus, and the two urban scenes beat it by a factor of two to
  four, because a city gives the gradient estimate hundreds of range bins to
  average where one reflector gives it one. The large-error result above, and
  the quality-signal measurements, were taken on Kourou only.

## Reproducing the validation

Validation runs in three tiers. Only the first runs in CI: the CI fixture is a
generated five-pulse CPHD, far too small for PGA to estimate anything from. So
the tests prove the code and the wiring, and real collects prove recovery.
Run everything inside the container (`./holoscan-radar run-container` on the
host) from the repository root, after building with
`-DHOLOSCAN_RADAR_BUILD_TESTING=ON`.

### Tier 1: tests, no data needed

```bash
ctest --test-dir build -R 'AutofocusOp\.|SarAutofocusGraph\.|^Sar_autofocus_' --output-on-failure
```

That selects three groups, each proving something different:

- **`AutofocusOp.*`** (22 cases in `tests/cpp/sar/test_autofocus.cpp`). The
  operator on synthetic images with a known injected phase: recovery of even
  and odd errors (exact at full window for pure quadratics and cubics), the
  narrow-window cost, both azimuth axes and the wrong-axis hazard, each guard
  and termination code, a `[live]` override, and float/double agreement. These are the estimator-accuracy tests.
- **`SarAutofocusGraph.*`** (6 cases in
  `tests/cpp/integration/test_sar_autofocus_graph.cpp`). The Figure 2 chain
  end to end on a synthetic scene: an estimate transferred onto a different
  product geometry, retain-and-reform through `ApertureManagerOp`, the 1:1 and
  oversampled resample paths, and the non-conjugate ground grid shown to be a
  hazard. These are what pin pulse ordering and sign.
- **`Sar_autofocus_*`** (4 app smokes on the CI fixture: `smoke`, `lie`,
  `oracle`, `polar`). Graph wiring, error injection, the oracle apply path and
  the polar auxiliary grid. **They do not test recovery:** no range bin of the
  fixture clears the strong-scatterer gate, so the operator passes the image
  through. A passing smoke says nothing about accuracy.

The performance smokes carry the `performance` label, which the default suite
excludes. Run them explicitly with `ctest --test-dir build -R '^perf_autofocus'
--output-on-failure`; they check that the recipe runs, not how fast.

### Tier 2: one real collect with a known answer

This is the measurement behind the accuracy numbers above. The app's
`position_error` block injects a known line-of-sight error and scores the
estimate against it.

1. **Get the data.** Each scene config carries its download command in its
   header comment, for example the `curl` line at the top of
   `applications/sar/autofocus/config/autofocus.yaml` for Kourou. Collects go
   under `data/sar/`.
2. **Copy the config and edit the copy.** Uncomment `position_error` (the
   shipped block is a 2 cm look-axis quadratic). To keep the run tractable,
   add `cphd_reader.pulse_range: [0, 8000]` and
   `aperture_manager.max_aperture_pulses: 8000`; the full Kourou aperture
   needs far more memory and time.
3. **Run it:**
   `build/applications/sar/autofocus/cpp/autofocus <your copy>.yaml`

Four log lines matter. Read them in this order:

- `PerturbPositionsOp: ... affine-removed X rad RMS` is **what was injected**,
  after the constant and linear terms PGA cannot see are removed. It is the
  baseline the residual is compared against.
- `AutofocusSharpnessReportOp: ... termination <code>, N iteration(s)
  accepted` says **what the operator did**. An `autofocus REFUSED (<code>)`
  warning instead means it returned the image untouched.
- `AutofocusSharpnessReportOp: sharpness ... (Gx over N iteration(s))` is
  **not an accuracy measure.** On the 107 rad case it reports 17× while the
  residual is outside the floor.
- `AutofocusScoreOp: affine-removed RMS R rad (M mm LOS) ... baseline B rad
  RMS ... removed Kx` is **the result.** Judge `R` against the `SAR/TESTING.md`
  targets: at most 0.39 rad (π/8) passes, and 0.785 rad (π/4) is the hard
  floor. `removed` is `B / R`. The scorer tries both signs and reports the
  one that fits, so a sign error in the emit path does not show up here; the
  `SarAutofocusGraph` tests are what pin the sign.

Two refinements:

- **`R` is an upper bound on estimator error, not the estimator's accuracy.**
  PGA also correctly estimates whatever error the collect itself carries, and
  the score charges it for that. To separate them, re-run with
  `polynomial: [0, 0, 0.0]` to measure the collect's own residual. At 8000
  Kourou pulses the 0.149 rad total splits into 0.142 rad from the collect and
  0.042 rad from the estimator.
- **To tell a correction-path fault from an estimator fault,** add
  `oracle: true` to `position_error`. That skips `AutofocusOp` and applies the
  exact injected error. If the oracle product focuses and the estimated one
  does not, the fault is in the estimator.

Adding `dump: true` and `output.path` also writes `af_expected_phase.npy` and
`af_estimated_phase.npy` beside the image, for plotting the estimate against
the truth.

### Tier 3: the grid across scenes

`tools/sar_autofocus/sweep.py` automates Tier 2 over a grid of injected errors
and settings, scraping the same `AutofocusScoreOp` line. It needs the built app
and the collects for whichever scenes it runs.

```bash
python3 tools/sar_autofocus/sweep.py --grid symmetry --scenes all --pulses 8000 --window-samples 1000
python3 tools/sar_autofocus/sweep.py --grid shapes
python3 tools/sar_autofocus/sweep.py --grid windows --pulses 8000
python3 tools/sar_autofocus/sweep.py --grid apertures
```

The first reproduces the four-scene table quoted above, and is also the
regression set for the emit path: all five shapes must land inside the target
on every scene that focuses, or the pulse mapping is wrong again. The grids vary error shape
(`symmetry`, `shapes`), first-window width (`windows`) and aperture length
(`apertures`); `--scenes` takes `kourou`, `atlanta`, `houston`, `imperial` or
`all`, and `--iterations` and `--window-samples` override the operator.

Each row prints `baseline`, `residual`, `removed`, `sharp` and a verdict:

| verdict | meaning |
| --- | --- |
| `pass` | residual at most 0.39 rad |
| `marginal` | inside the 0.785 rad floor but over the target |
| `FAIL` | outside the floor, but better than doing nothing |
| `HARMFUL` | the residual is larger than the injected error: applying the estimate made the image worse |
| `refused` | the guard kept the image untouched (residual equals baseline) |

`refused` is a separate verdict because declining to act is the designed
response on a scene with no dominant scatterer, which is a different finding
from estimating wrongly. Expect Imperial Valley to refuse every row. The
summary counts rows inside the floor overall and per scene. A shape failing on
every scene points at the estimator; a shape failing on one scene points at
that scene.

Before adding a new scene, screen it:

```bash
python3 tools/sar_autofocus/pga_screen.py data/sar/<collect>.cphd --spacing 1.0
```

It scores both halves of PGA's precondition for its best candidates:
*isolation* (peak over the median of a surrounding annulus) and *persistence*
(the candidate's weakest amplitude across eight disjoint dwell blocks,
relative to its mean). Kourou's best is 135× and 0.68. A bright return that
collapses in some blocks is glint and will not support PGA. The screen
predicts whether a scene will work, not how well: Atlanta screens best and
Houston focuses best.

## Known limitations and open items

The operator's own items are in its `OPEN_ITEMS.md`; the module-level ones are in [`OPEN_ITEMS.md`](OPEN_ITEMS.md).

- **Dwell-edge zero block (open, needs a team decision).** When the occupied
  spectral support is under 85%, the excluded bins are left at exactly zero.
  After the bin-to-pulse remap those bins land on the first and last pulses,
  so the emitted correction has a flat zero at both aperture edges with a step
  at the boundary. Zeroing, holding the edge value, and extrapolating are
  three different answers with different real-collect consequences, so it was
  documented rather than guessed at. [CLARAHOLOS-3620](https://nvidia.atlassian.net/browse/CLARAHOLOS-3620).
- **Performance.** About 10.8 ms per iteration, measured on the Kourou-sized
  auxiliary image when the iteration default was raised. Characterized, not
  yet optimized; the obvious targets are in the next section.
- **Test gaps deferred off this branch:** the trajectory-error leg (perturb
  positions rather than injecting phase), estimate accuracy on a points-mode
  conjugate lattice, and an assertion on the refusal warning
  ([CLARAHOLOS-3616](https://nvidia.atlassian.net/browse/CLARAHOLOS-3616), [CLARAHOLOS-3617](https://nvidia.atlassian.net/browse/CLARAHOLOS-3617), [CLARAHOLOS-3618](https://nvidia.atlassian.net/browse/CLARAHOLOS-3618)).
- **Design question:** whether the azimuth-to-pulse resample belongs in the
  module rather than in the app (`docs/planning/SAR/OPEN_ITEMS.md`, [CLARAHOLOS-3619](https://nvidia.atlassian.net/browse/CLARAHOLOS-3619)).

## Review findings left as-is

Three warnings from the branch's pre-push review have no fix in this branch.
They are recorded here so they are not rediscovered as new defects.

- **`BackprojectionOp` input connectivity latches on arrival, not on wiring**
  (`backprojection_op.cu`, `sample_points_connected_` and `weight_connected_`).
  The operator learns that `sample_points` or `weight` is wired the first time
  a tensor arrives on it, so behaviour before that first arrival follows the
  unwired path. The fix is to read wiring in `setup()`, but HSDK 5 exposes no
  input-connectivity query. Recorded in
  [`backprojection_op/OPEN_ITEMS.md`](../../../operators/sar/backprojection_op/OPEN_ITEMS.md)
  and [CLARAHOLOS-3621](https://nvidia.atlassian.net/browse/CLARAHOLOS-3621).
- **Serial kernels in the PGA loop** (`autofocus_stage.hpp`).
  `integrate_detrend_kernel` launches `<<<batch, 1>>>`, one thread per image,
  because the outward-from-DC integration is a sequential scan.
  `sharpness_kernel` and `p2p_kernel` launch one block per image, which is a
  single block for the rank-2 input. All three are small next to the FFTs at
  current sizes, but they are where per-iteration time goes first as images
  grow. A parallel prefix scan and a multi-block reduction are the standard
  replacements. [CLARAHOLOS-3622](https://nvidia.atlassian.net/browse/CLARAHOLOS-3622).
- **The autofocus app's four partitions have no thread binding**
  (`applications/sar/autofocus/cpp/autofocus.cpp`: `estimate`, `sharpness`,
  `score`, `correction`). The shared SAR driver binds a `ThreadPlacement` to
  its own partitions only, and `run_sar_graph_app()` offers no hook for an app
  to add more. This is probably harmless today: HSDK 5 gives every partition
  its own lane with a default placement, and the driver binds that same
  default. Recorded, with the plan-dump check that would confirm it, in the
  SAR [`OPEN_ITEMS.md`](OPEN_ITEMS.md) and [CLARAHOLOS-3623](https://nvidia.atlassian.net/browse/CLARAHOLOS-3623).

All of the above are tracked under the SAR epic [CLARAHOLOS-3615](https://nvidia.atlassian.net/browse/CLARAHOLOS-3615).

## Where to look

| What | Where |
| --- | --- |
| Contract | [`OPERATOR_API.md`](OPERATOR_API.md), `AutofocusOp` section |
| Workflows (Figures 2 and 3, D20) | [`WORKFLOWS.md`](WORKFLOWS.md) |
| Test requirements | [`TESTING.md`](TESTING.md), `AutofocusOp` |
| Decision record | [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md), D20 |
| Operator | [`operators/sar/autofocus_op/`](../../../operators/sar/autofocus_op/README.md) |
| Operator open items | [`autofocus_op/OPEN_ITEMS.md`](../../../operators/sar/autofocus_op/OPEN_ITEMS.md) |
| Termination and axis enums | [`enums.hpp`](../../../operators/common/radar_common/descriptors/enums.hpp) |
| Unit tests | [`test_autofocus.cpp`](../../../tests/cpp/sar/test_autofocus.cpp), [`autofocus_phase_support.hpp`](../../../tests/cpp/sar/autofocus_phase_support.hpp) |
| Graph tests (Figure 2 end to end) | [`test_sar_autofocus_graph.cpp`](../../../tests/cpp/integration/test_sar_autofocus_graph.cpp) |
| Application | [`applications/sar/autofocus/`](../../../applications/sar/autofocus/README.md) (README has the graph and the aux-lattice reasoning) |
| Sweep and screening tools | [`tools/sar_autofocus/`](../../../tools/sar_autofocus/) (`sweep.py`, `pga_screen.py`) |
| Performance recipe | [`performance/operators/sar/autofocus/`](../../../performance/operators/sar/autofocus/recipe.yaml) |
