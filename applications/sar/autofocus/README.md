# SAR autofocus (Figure 2)

Phase gradient autofocus on a ground / DEM product: retain the whole aperture, form a small conjugate-grid image from it, estimate a per-pulse phase, apply that phase to the same retained pulses, and form the product once. This is the WORKFLOWS Figure 2 chain. The five sibling SAR applications all form directly and leave autofocus out.

## Graph

```mermaid
flowchart LR
    R[CPHDReaderOp] --> RC[SARRangeCompressionOp]
    R -- boundary --> A[ApertureManagerOp]
    R -- positions / reference --> E[EcefToSceneEnuOp]
    E -. scene positions .-> L[PerturbPositionsOp]
    E -. scene positions .-> A
    RC -- profiles --> L
    L -. profiles / positions / reference .-> A
    A -- profiles --> P[AuxPointsSourceOp]
    P -- profiles + polar lattice --> AB[aux BackprojectionOp]
    A -- profiles / positions / control --> AB
    AB -- aux image --> AF[AutofocusOp]
    AF -- phase_correction --> PR[PhaseResampleOp]
    AF -- image_out --> Drain[NullSinkOp]
    AF -- sharpness / termination / iterations --> SR[AutofocusSharpnessReportOp]
    PR --> PC[PhaseCorrectionOp]
    PR -. score .-> S[AutofocusScoreOp]
    A -. profiles .-> O[OraclePhaseOp]
    O -. oracle .-> PC
    A -- profiles --> PC
    PC -- corrected profiles --> B[product BackprojectionOp]
    A -- positions / reference / control --> B
    B --> Report[Image report]
    B -. dump .-> Dump[TensorDumpOp]
    B -. visualize .-> Viz[HolovizOp]
```

Solid paths are the required flows. `PerturbPositionsOp`, `OraclePhaseOp` and `AutofocusScoreOp` appear only when the `position_error` block is present, and `OraclePhaseOp` replaces the whole estimation branch rather than adding to it. `AuxPointsSourceOp` appears only for a polar aux lattice, and sits *in series* on the replay edge rather than beside it — profiles in, profiles and lattice out — so both carry the same capture time. `EcefToSceneEnuOp` appears only when `enu_origin_ecef` is set; without it the reader's positions and reference go straight to `PerturbPositionsOp` or the manager.

`ApertureManagerOp` runs with `emit_mode: reform`: the retained aperture is replayed once as a whole, which is what makes a corrected second pass possible at all. Nothing is staged behind that emission, so the manager emits `kResetThenEmit` and the product image is terminal even though no reset follows it. `weight` stays unconnected unless `aperture_window` is set — retention here is replay, not eviction. The aperture is never staged through `ApertureManagerOp.sample_geometry`; that port belongs to the product lattice and Figure 2 does not wire it.

`AutofocusOp.image_out` is emitted every tick whether or not anyone wants it, so it needs a consumer; Figure 2 wants the pulse-indexed `phase_correction` instead.

## The auxiliary lattice

The estimation grid is application geometry (`aux_lattice` in YAML, or derived from CPHD `TxPos` / `SRPPos`), and it comes in two shapes.

`aux_lattice.mode: polar` is the **range-bearing** grid, and it is the one PGA's Fourier relationship actually holds on: rows of constant slant range from the mid-aperture phase centre, columns uniform in bearing within the slant plane. A scatterer's phase history over the aperture is `-(4*pi/lambda) * R * cos(theta_j - theta)`, so over the small angle a spotlight dwell subtends, image bearing is the conjugate of aperture angle — of slow time, on a uniformly sampled pass.

The bearing step is `rho_az / R`, whose `num_u`-column span is exactly the unaliased conjugate swath; `num_u` defaults to one column per pulse, which makes `PhaseResampleOp` an identity copy. The `num_v` range rows are centred on the **product lattice centre**, and omitting `num_v` sizes them from the product corners' own slant-range span, clamped to 4096.

**Why the clamp.** The product's slant-range span is the product's requirement, not the estimator's: PGA keeps `min_scatterers` range bins (16 by default) out of however many pass the peak-to-mean gate, so rows beyond the point where good candidates are plentiful buy only memory — and the aux is the second-largest allocation in this graph. A natively-sized Kourou product (7150 square) asks for 8652 rows and does not fit in 96 GiB; clamped to 4096 it forms and autofocuses, recovering a 1.090x sharpness gain against the 1.096x the unclamped 2048-square run gets.

The clamp is deliberately loose because row count *is* the candidate pool the scatterer selection draws from, and cutting it too far leaves a worse best sixteen: measured on Kourou, 4096 and 2478 rows both recover the full gain, while 512 and 1024 produce an estimate bad enough that iteration 0 regresses sharpness and the loop aborts. That failure is silent — the app still emits an image and exits zero, and only the sharpness report's `1.000x` gives it away — so the clamp warns whenever it fires. An explicit `aux_lattice.num_v` is honoured as given and never clamped.

The clamp is a guard against the aux tracking product size, not a memory strategy: it is worth about 5 GiB, and the graph's ceiling is set by the retained aperture, which reserves seventeen copies of the pulse block and accounts for roughly 71% of the footprint. Products past Kourou's native size still exceed 96 GiB.
Any set of constant-range rows is a valid polar grid, so the anchor is free, and the SRP is the wrong end of it: a product centred away from the IARP would need `num_v` to span the whole IARP-to-scene distance before covering the scene at all — on Kourou that is 638 m at 0.273 m bins, 2335 rows of pure overhead. The derivation uses the product's four corners to pick the row count; it does not report where they land, and the only warning it raises is the row clamp above.

A polar grid is not a `SampleGeometry` — two unit axes and a uniform metre spacing cannot express it — so the app-owned `AuxPointsSourceOp` supplies it on `BackprojectionOp.sample_points` in points mode, with no contract change. `BackprojectionOp` joins that port on capture time with `Exact{}`, so the source sits in series on the replay edge — profiles in, profiles and lattice out — giving both the same capture time. It emits the lattice first on purpose: the profiles are the consumer's trigger, and because `sample_points` is a `[live]` override under `on_missing SafeDefault{}` the activation fires as soon as the trigger set completes rather than waiting for the lattice. `Exact{}` decides which sample matches, not whether the consumer blocks.

`aux_lattice.mode: cartesian` (the default) is the older look-by-slant-plane grid: `uhat` is track with look removed, sampled at least Nyquist along slow time (`aux_lattice.samples_per_psf`, default 2), so `n_u` may exceed the pulse count and `PhaseResampleOp` maps the estimate back onto pulses. **It is not a conjugate grid**, and at spaceborne range that is not a technicality: a straight `uhat` chord of `L_perp` walks tens of compressed range bins across the dwell, which is exactly what a constant-range row exists not to do. Jakowatz and Wahl (SPIE 7337, 2009) make the point for backprojection directly — PGA wants a range-bearing image, not a ground-plane one. Near the SRP a small Cartesian patch is a local stand-in, which is why the generated fixture still uses it.

When YAML omits `autofocus.window_fraction`, the application sizes the first PGA window to about 64 azimuth samples: half of a Nyquist-long `u` axis is mostly clutter.

Figure 3, in-place PGA on a radar-geometry product, is a different graph and is not this binary.

## Validating the estimate

"The image got sharper" is not a measurement. The `position_error` block gives autofocus a known answer to find:

- `PerturbPositionsOp` shifts the aperture along the look axis by a polynomial in `s` over `[-1, 1]` **and** multiplies the equivalent `4*pi*f*delta/c` onto the range profiles. The profile multiply is the part that matters: an APC shift along look is common-mode in SRP-relative `dR` and would not smear on its own, so the phase is what remains after motion compensation and what PGA actually sees.
- `OraclePhaseOp` hands `PhaseCorrectionOp` the exact recorded leftover and skips the estimator entirely. If the product comes back focused, the sign convention, the pulse indexing and the `reform` replay are all correct, and any shortfall in a full run belongs to `AutofocusOp`.
- `AutofocusScoreOp` compares the resampled, pulse-indexed estimate to the recorded truth after removing the affine term PGA cannot observe, and logs the residual in radians and in millimetres of line of sight. It reports which sign fits better rather than assuming one, because which way round the estimate comes out depends on the image's conjugate convention. Exceeding `score_warn_rms_rad` is a warning, not a failure, and with `dump: true` it also writes `af_expected_phase.npy` and `af_estimated_phase.npy` beside the product.

The recorded lie travels from the liar to the scorer as a shared host buffer rather than a Holoscan edge: the liar runs on ingest ticks and the estimate exists only on the single reform tick, so there is no tick on which the two could be joined.

## Build and offline fixture

Open the devcontainer or run `./holoscan-radar run-container` on the **host**, then from the repository root **inside the container**:

```bash
cmake -S . -B build -DHOLOSCAN_RADAR_BUILD_PYTHON=OFF -DHOLOSCAN_RADAR_BUILD_TESTING=ON
cmake --build build --target autofocus generate_cphd_fixture
ctest --test-dir build -R '^Sar_autofocus_' --output-on-failure
```

Four fixture runs: the clean Figure 2 chain, the same chain with the lie injected and scored, the oracle apply path, and the polar auxiliary. They check graph wiring, the injection and the scoring plumbing — not recovery. A five-pulse aperture over a handful of point targets is far too small a scene for PGA to estimate anything from, and the generated fixture is already focused; the estimator passes the image straight through because no range bin clears the strong-scatterer gate. Recovery is measured on real data.

## Kourou (the PGA vehicle)

The development collect is an Umbra SNAPSHOT over the Guiana Space Centre (`2023-06-24-13-22-56_UMBRA-05`, 5.23 N 52.77 W, 23375 by 23999, 4.19 GiB), configured in [`config/autofocus.yaml`](config/autofocus.yaml) with the download command. It was chosen by screening for PGA's scene preconditions rather than for brightness: `python3 tools/sar_autofocus/pga_screen.py data/sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd` finds six persistent candidates, the brightest isolated 135 times over its annulus and holding 0.68 of its mean in its worst of eight equal dwell blocks. Two of the scene's brightest returns collapse to 0.07 and 0.09 across the dwell, which is the glint case the screen exists to reject.

The product is a 2048-square **0.25 m** east-north patch on that scatterer, and the fine spacing is load-bearing rather than cosmetic. The target is diffraction-limited to one or two samples at 0.25 m, so on a 1 m lattice it falls between pixels: the same peak reads 43468 at 1 m and 160525 at 0.25 m (both over the full aperture), and adding a quadratic error *raises* the 1 m number by broadening the PSF onto a pixel. **Do not score focus at 1 m here.** The aux is the polar lattice with rows auto-sized about the product centre, 2460 by 23375 at the full aperture (2478 by 8000 at the `pulse_range` the measurements below use).

On a persistent target the 2 cm quadratic needs no scaling: it drops the peak to 39.2% of focused.

**Where this stands, re-measured on HSDK 5 over an 8000-pulse `pulse_range`**, scoring the product peak against the clean image and the resampled estimate against the injected truth. (The full 23375-pulse aperture needed 128.4 GiB against 96 GiB of GPU until samples went to single precision on 2026-10-29; it now fits at 66.9 GiB and is measured in the sweep below.)

| config | product peak | ratio | sharpness | affine-removed phase RMS |
| --- | --- | --- | --- | --- |
| clean, no lie | 65493.0 | 1.000 | 6.98e19 | n/a |
| 2 cm lie, uncorrected | 25686.3 | 0.392 | 1.12e19 | 2.400 rad |
| 2 cm lie, PGA estimate applied | 66603.6 | 1.017 | 7.18e19 | 0.146 rad |
| 2 cm lie, oracle phase applied | 65492.6 | 1.000 | 6.98e19 | 0 by construction |

Measured against the acceptance target in [`docs/planning/SAR/TESTING.md`](../../../docs/planning/SAR/TESTING.md) (residual <= 0.39 rad RMS, hard floor 0.785 rad, product peak >= 0.9 of error-free), the estimator **passes**: 0.146 rad and 1.017. The corrected peak sits slightly above the error-free one because the clean collect is not perfectly focused either — PGA's estimate on it has a 0.96 rad peak-to-peak swing — so correcting the lied collect also removes residual error the clean reference still carries.

### Accuracy over aperture length

The table above is one aperture length. Sweeping `pulse_range` and `max_aperture_pulses` together, with everything else at the preset's settings, the estimator degrades with length and **not monotonically**:

| pulses | residual | swing | sharpness | iterations kept |
| --- | --- | --- | --- | --- |
| 8000 | 0.149 rad | 1.09 | 8.66× | all 4 |
| 12000 | 0.283 rad | — | 8.94× | 3 of 4 |
| 16000 | **0.692 rad** | 1.31 | 5.28× | all 4 |
| 20000 | **0.428 rad** | — | 2.73× | 1 of 4 |
| 23375 (full) | 0.332 rad | 1.25 | 2.84× | 1 of 4 |

**The table above was measured at the old `num_iterations: 4`.** The default became 8 on 2026-10-29 (see the operator [README](../../../operators/sar/autofocus_op/README.md)); on this configuration that moves the 8000-pulse residual from 0.149 to 0.161 rad and sharpness from 8.66× to 8.78×, and it is what makes a 107 rad error recoverable at all.

**Most of that spread is not the estimator.** The score is taken against the injected truth alone, while PGA also estimates whatever real error the collect carries — correctly — and is charged for it. Re-running each length with `polynomial: [0, 0, 0.0]` measures the collect's own error by itself, and subtracting it separates the two:

| pulses | residual vs injected truth | the collect's own | estimator alone |
| --- | --- | --- | --- |
| 8000 | 0.149 rad | 0.142 rad | **0.042 rad** |
| 16000 | 0.692 rad | 0.507 rad | **0.346 rad** |

Both are inside the 0.39 rad target once the baseline is removed. **Read any single residual in the table above as an upper bound on estimator error, not as the estimator's accuracy.** What is left after that correction is a genuine aperture ceiling: beyond roughly 16000 pulses PGA produces an estimate that lowers sharpness and the guard refuses it, so autofocus contributes nothing. See [`operators/sar/autofocus_op/OPEN_ITEMS.md`](../../../operators/sar/autofocus_op/OPEN_ITEMS.md) for the ruled-out causes and the measured refused estimates.

To reproduce: uncomment the `position_error` block in [`config/autofocus.yaml`](config/autofocus.yaml), add `cphd_reader.pulse_range: [0, N]` and `aperture_manager.max_aperture_pulses: N`, and run. `AutofocusScoreOp` logs the affine-removed RMS; adding `dump: true` with `output.path` writes `af_expected_phase.npy` and `af_estimated_phase.npy` beside the image, which is what the swing and the error decomposition come from. Note the key is `output.path` — `dump` itself is only a boolean.

### The index-order and sign defect, and how it hid

These numbers were 0.523 and 1.246 rad until 2026-10-28. Two independent defects in how `AutofocusOp` emitted its estimate were partially cancelling:

- **Index order.** `phase_acc` is accumulated per azimuth-FFT bin with `fftfreq` semantics — bin 0 is DC, which is the *centre* of the dwell. `phase_correction` is contracted as pulse-indexed, and pulse index runs monotonically across the dwell. The two differ by an fftshift, and the operator emitted the vector unshifted.
- **Sign.** A `neg_n_kernel` negated the vector on the way out, so what reached `PhaseCorrectionOp` with `sense: subtract` was the correction rather than the error.

A symmetric quadratic circularly shifted by half its length is approximately its own negative plus a constant, so the two errors largely undid each other. What was left recovered about half the error and roughly a fifth of the peak deficit — enough to clear the 0.7 peak-recovery gate the transfer tests used, and enough to look like a weak-but-working estimator rather than a wiring fault.

The fix is one kernel: emit `fftshift(phase_acc)`, with no negate. The contract did not change; the code now matches what [`OPERATOR_API.md`](../../../docs/planning/SAR/OPERATOR_API.md) already said.

The shape that was previously described as a scale error of about two between the estimate's axis and the pulse axis is exactly this: a parabola minus its own half-length circular shift is twice the truth at the dwell centre, about half at the edges, and sign-inverted at the quarter points. That reading was right about the symptom and wrong about the cause.

`PhaseResampleOp` changed with it. Its oversampled branch used to take the low-frequency bins from each end of the vector, which is the correct extraction for a bin-ordered estimate; on a pulse-ordered one it takes the central `num_pulses` samples instead, which is where the real pulses sit in an azimuth extent wider than the aperture. Both branches are now index-consistent and the identity branch is unchanged.

`SarAutofocusGraph.Figure2OversampledDiagnostics` pins the result by applying ten alternative maps of the same estimate by hand and requiring the shipped one to win.

**The core estimator is exact.** On a synthetic IFFT-formed image at the full azimuth window it recovers a pure quadratic to 7.7e-05 rad RMS at 2.5 rad of error and 0.002 rad at 5 rad — four orders of magnitude inside the target, pinned by `AutofocusOp.PureQuadraticIsRecoveredExactlyAtFullWindow`. The 0.2 rad that the older `RecoversQuadraticUpToAffine` reports is not estimator error at all: that test injects a linear ramp and an offset alongside the quadratic, and the integer `circshift` cannot remove the sub-bin remainder.

**The narrow-window bias is fixed too, and it was the same class of defect.** The window step exists to isolate the scatterer from clutter in the *image*, and it narrows the image support; the gradient FFT behind it has to stay at the full azimuth extent regardless. It did not — the window was packed into a `width`-long transform, which decimates the phase history by `n_az / width`, so every gradient difference spanned that many phase-history samples instead of one and the estimator's small-angle form saturated on steep errors. Transforming the full windowed line instead, zeros included, recovers it:

| window-to-azimuth ratio | residual, packed | residual, full-length | swing retained |
| --- | --- | --- | --- |
| 1:1 | 0.003 rad | 0.002 rad | 1.00 |
| 1:2 | 0.069 rad | 0.009 rad | 1.00 |
| 1:4 | 0.245 rad | 0.036 rad | 0.96 |
| 1:8 | 0.830 rad | 0.150 rad | 0.86 |

on a 5 rad error, which brings every window ratio inside the 0.39 rad acceptance target. What a narrow window still costs is *swing*: the estimate keeps its shape and under-reports the amplitude, down to 86% at 1:8. That is the honest information cost of estimating from fewer image samples, and it is now the only window effect. `AutofocusOp.NarrowWindowCostsSwingAndStaysInsideTheTarget` pins both columns and doubles as the regression guard on the transform length.

The earlier reading of this fault — that the aperture *edges* were lost while the dwell centre held — was an artifact of the index-order defect above, since the edge and centre labels were read off a vector that was half a dwell out of position. Re-measured in pulse order it is the opposite: the dwell centre degrades first, to a ratio of 0.86 at 1:8, while the edges hold.

**On Kourou this changes almost nothing** — 0.146 to 0.149 rad, and the product peak 1.017 to 1.011 — and that is the expected result rather than a disappointment. The fault only bites when the per-sample phase step is large, and a 5.4 rad error spread over 8000 pulses has a step of order a milliradian even decimated 23-fold. The prior diagnosis said as much when it ruled fault 1 out as the explanation for Kourou. The table above is where it was ever observable.

## Scene preconditions are a real constraint

Two earlier collects failed on the scene rather than on the estimator, and both are worth knowing about before picking a new target.

PGA assumes each selected range bin is dominated by one strong scatterer that persists across the *full* synthetic aperture (Wahl, Eichel, Ghiglia and Jakowatz, *IEEE T-AES* 30(3), 1994), and that one phase error is common to the whole scene.

A Strait of Dover collect looked ideal on amplitude — isolated bright returns on dark water — but its brightest scatterer is a glint, not a persistent one: the middle quarter of the dwell carries 72% of the coherent peak and the outer half of the aperture adds 5%. Ships also violate space-invariance, since one swinging in wind and waves gives every scattering point its own error; there is a maritime sub-literature (SV-PGA, Mosaic PGA, IPGRA) whose premise is precisely that classical PGA cannot focus swing ships. The Dover result is not an estimator defect, and a 2 cm quadratic centred on the dwell is flattest exactly where a glint is bright, so it costs such a target about 1% of peak and cannot be used as a stimulus at all.

An urban Atlanta collect was recorded here as failing the isolation half instead. **That verdict is superseded** — see the measured screen below, where Atlanta passes both halves more cleanly than Kourou does.

### Three more scenes, and what screening actually buys

Three further collects now have configs beside the Kourou default — [`autofocus_atlanta.yaml`](config/autofocus_atlanta.yaml), [`autofocus_houston.yaml`](config/autofocus_houston.yaml) and [`autofocus_imperial.yaml`](config/autofocus_imperial.yaml) — so an estimator result can be separated from a Kourou result. `python3 tools/sar_autofocus/sweep.py --grid symmetry --scenes all --pulses 8000 --window-samples 1000` runs all four. Worst affine-removed residual over the five injected error symmetries, in radians:

| scene | collect | scene type | worst of five |
| --- | --- | --- | --- |
| Kourou | UMBRA-05 | screened isolated reflector | 0.219 |
| Atlanta | UMBRA-07 | airport | 0.177 |
| Houston | mode02 | dense urban | 0.107 |
| Imperial Valley | Capella SM | irrigated agriculture | refused, all five |

**Screening every scene the same way says the Kourou collect is the weakest of the three that work.** `python3 tools/sar_autofocus/pga_screen.py <collect> --spacing 1.0` forms the scene and its eight disjoint dwell blocks and scores both halves of PGA's precondition; run over each config's own pulse range on a matched 2048² lattice at 1.0 m, the best candidate in each scene is:

| scene | isolation (peak / annulus median) | persistence (worst block / mean) | persistent of 8 candidates |
| --- | --- | --- | --- |
| Kourou | 135× | 0.68 | 6 |
| Atlanta | 661× | 0.96 | 8 |
| Houston | 585× | 0.85 | 7 |
| Imperial Valley | no cell cleared 30× | — | 0 |

This **corrects the Atlanta claim above**, which predates the tool being restored: Atlanta does not fail the isolation half. It clears it by a wider margin than Kourou on both halves, and the earlier verdict cannot be reproduced at any setting tried here. Read the two measurements together and the screen is broadly predictive — the three scenes with an isolated, persistent scatterer all autofocus to 0.11–0.22 rad, and the one with none is refused outright — but it does not rank them, since Atlanta screens best and Houston focuses best. Compare isolation across scenes with care: the ratio depends on how concentrated a point target is relative to the lattice, and these collects have different native resolutions, so a common 1.0 m lattice oversamples them differently.

What screening still buys is a scene you can *check*: a single dominant scatterer lets you verify focus afterwards by looking at one impulse response. It does not buy the lowest residual, because the estimator averages a phase gradient over `min_scatterers` range bins and a city supplies hundreds of usable bins where Kourou supplies a handful.

**Agriculture is the floor, and the guard finds it.** All five shapes hit the sharpness guard at iteration 0 on Imperial Valley and the image comes back untouched — distributed clutter has no dominant scatterer in any range bin, which is the precondition stated at the top of this section, and the operator declines rather than applying a wrong correction. It is worth running as the negative control for exactly that reason: without a scene that *should* fail, a per-scene failure elsewhere cannot be attributed.

One caveat on all four: they are spaceborne X-band collects with good ephemeris, which is the regime [`SAR/OPERATOR_API.md`](../../../docs/planning/SAR/OPERATOR_API.md) says should stay on Figure 1, and the error is injected rather than real. None of this tests an uncompensated airborne platform.
