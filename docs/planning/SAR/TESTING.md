# Holoscan Radar Module — SAR Testing & Simulation Requirements

## Purpose

Defines what the SAR operators must be tested against: (1) the **point-target simulator** — an analytic signal model that produces phase history with known ground truth — and (2) the **high-level, I/O-level tests** each operator must pass. Like the FMCW [`TESTING.md`](../TESTING.md) one level up, this document stays at the interface level: it specifies *what property* each test asserts, not the fixtures, tolerances, or framework. Lower-level unit tests (kernel internals, numerical tolerances, boundary indexing) are deferred.

Two things differ structurally from the FMCW pass, and both change how tests are wired:

- **The simulator never feeds an operator in-process.** It always writes CPHD first.
- **Several operators have no reference implementation to diff against**, so assertion effort is deliberately unbalanced toward them. The per-operator weighting is in [`SIGPROC_PLAN.md`](SIGPROC_PLAN.md), since it is a planning concern.

## Test tiers

- **Unit** — one operator, minimal synthetic input with a closed-form answer. Inputs are constructed **inline as tensors** in fixture code, not produced by a simulator.
- **Integration** — a chain fed by a CPHD file the simulator wrote, read through the real `CPHDReaderOp`.
- **Validation** — real collections, asserting convention correctness rather than numerical accuracy.

There is no separate statistical tier as in FMCW, where Monte-Carlo false-alarm rates are the calibration check. Its nearest analogue here is the speckle-statistics assertion on `MultilookOp`.

---

## The point-target simulator (analytic signal model)

The simulator is a **signal model, not a physics or EM solver**: it synthesizes phase history by summing each point target's closed-form contribution over (pulse, fast-time sample), then adds noise. Cheap, deterministic, and its own correctness is easy to reason about.

It exists today in Python, outside this repository, and writes CPHD. This section specifies what it must do, because every quantitative assertion in the module depends on it and none of the repo's own tests can police it.

### Inputs it must accept

- **Scene** — a list of point targets, each with a **Cartesian position** `(x, y, z)` (m) in the declared frame, plus a complex amplitude (or an RCS and a range law). Targets are **stationary**; see *Modeling assumptions*.
- **Platform trajectory** — per-pulse antenna phase-center positions in the same frame, supplied either parametrically (start position, velocity, optional acceleration) or as an explicit per-pulse position list. **An arbitrary, non-straight track must be expressible**, because a straight track cannot distinguish a correct backprojection from several plausible wrong ones.
- **Trajectory error, injectable and separable** — a deliberate per-pulse position perturbation, *and* the true trajectory alongside it. The simulator must be able to write the *erroneous* trajectory into the file while reporting the *true* one as ground truth. This is a first-class requirement, not a test convenience: it is the only way to validate autofocus, and the only way to test that a wrong-convention trajectory is detectable.
- **Injected phase error** — a per-pulse phase (or a polynomial in pulse index) added to the phase history, with the applied values reported as ground truth. The direct autofocus ground truth, independent of the trajectory-error path.
- **Collection geometry** — the reference point the phase history is referenced against, emitted as `reference_position`, from which the per-pulse reference range follows.
- **Waveform** — the `LFMWaveformDescriptor` fields (carrier frequency, bandwidth or slope, chirp duration, sample rate, pulse repetition interval), plus `num_pulses` and `num_samples`.
- **Noise** — additive complex Gaussian, configurable by SNR or noise power, with a seed.
- **Coordinate frame** — the `CoordinateFrame` the scene and trajectory are expressed in, recorded in the output file so a frame mismatch is detectable rather than assumed.

### Outputs it must produce

- **A CPHD file** carrying the phase history, per-pulse positions, reference ranges, pulse times, and the collection metadata a reader needs. This is the only interface to the operators.
- **Ground truth alongside it** (a sidecar, not inside the CPHD): each target's Cartesian position; its slant range at aperture centre; the **predicted image-quality metrics** for the configured collection — range resolution `≈ c / (2·B)`, cross-range resolution `≈ λ / (2·Δθ)` for the subtended aperture angle `Δθ`, the range PSLR and ISLR the configured range window implies, and the cross-range PSLR and ISLR of an unweighted aperture (see R6); the true trajectory where an error was injected; and the injected phase error where one was applied.
  - **Not pixel indices.** Those depend on the formation configuration — grid origin, spacing, axes, `nfft` — which the simulator does not know.
  - Tests therefore assert in the **physical domain**: map an image peak back to a position through the formation grid parameters and compare to ground truth within a fraction of a resolution cell. That also validates the grid parameters themselves.
  - The predicted metrics are what make the assertions non-circular: they come from the collection parameters, not from a previous run of the pipeline.

### The signal model (per target, summed)

Phase history is the deramped, frequency-domain form CPHD stores. With `p_k` the phase-center position at pulse `k`, `t` the target position, `R_k = ‖p_k − t‖` the one-way range, and `R_ref,k` the reference range, fast-time sample `s` corresponds to transmitted frequency `f_s = f_start + s·Δf`. The contribution to sample `(k, s)` is the product of:

- the **differential-range phase** `exp(−j·4π·f_s·(R_k − R_ref,k) / c)` — the term that makes range compression a transform along `s` and that carries the aperture phase along `k`;
- the target's **complex amplitude**, times an optional range law (`R⁻³` or `R⁻⁴`);
- an optional **injected per-pulse phase** `exp(j·φ_k)`;
- an optional scalar **antenna-pattern weighting** as a function of the pulse's look direction to the target.

Sum over targets, add noise. **Ranges are computed in double precision** so the simulator is never the accuracy limit in a `precision_mode` comparison.

### Scene frame and coordinate conventions

The scene and trajectory are specified in one declared Cartesian frame, and the simulator only ever takes norms of position differences — so like the operator kernels it is **convention-free** with respect to angles and takes no `AngleConvention`. It does take a `CoordinateFrame`, because it must record in the output file what its coordinates mean; that is the label the reader validates against.

This is what lets the simulator test the `positions` port conventions without depending on them: it can emit a trajectory that is correct, or one that is offset in time, expressed in a mirrored frame, or referenced to the wrong point, and the test asserts that the resulting image degrades as predicted.

### Modeling assumptions (state these explicitly)

- **Stop-and-hop.** The platform and scene are frozen *within* a pulse and advance *between* pulses. Intra-pulse motion is omitted, matching the operators' assumption — and matching the `positions` port's "transmit instant" specification.
- **Range migration is inherent, not omitted.** Because the model evaluates the exact range per pulse rather than linearizing it, target energy migrates across range bins over the aperture exactly as in real data. This is a deliberate difference from the FMCW simulator, which explicitly omits intra-CPI range migration: for SAR it is the dominant effect image formation exists to handle, so a simulator that omitted it would validate nothing.
- **Static scene.** Targets do not move. Real moving targets smear and displace in SAR imagery; excluding them keeps the impulse-response assertions clean, and moving-target imaging is out of scope for this wave anyway.
- **Far-field per pulse**, **monostatic**, **ideal LFM**, **ideal hardware**, no propagation medium.
- **Flat-Earth is not assumed.** Targets are arbitrary 3-D points, so terrain height is expressible even though terrain *surfaces* are not modeled.

### Explicitly out of scope (first wave)

EM propagation and scattering; multipath; layover and shadow; distributed surfaces and terrain (targets are points); speckle from distributed clutter — which means `MultilookOp`'s speckle assertion needs a separate synthetic clutter field, not this simulator; antenna element patterns beyond a scalar weighting; hardware impairments (phase noise, IQ imbalance, ADC quantization); atmospheric and ionospheric propagation; moving targets; polarimetry; bistatic geometry. These mirror the operator-lockdown exclusions and can be added later as optional layers.

### Cross-cutting simulator requirements

- **Aliasing modeled.** Targets beyond the unambiguous range must fold correctly, and azimuth ambiguities from PRF under-sampling must appear, so both can be tested rather than assumed absent.
- **Determinism.** A fixed seed reproduces the file bit-for-bit.
- **Guard against tautology.** The signal model must be derived **independently of the operator implementations**, and canonical cases cross-checked against an independent reference — a textbook closed form for the point-target response, or an established external SAR simulator. Tests assert against **analytic predictions** (a target at a known position, imaged at the resolution the bandwidth and aperture imply, with the range sidelobe levels the range window implies), never merely that the generator round-trips through the pipeline.
- **A kinematic driver is required.** FMCW's `TESTING.md` defers a time-stepping driver, noting it waits for "a frame-spanning consumer (a tracker, multi-frame integration, or clutter-map CFAR — all out of scope now)". **SAR is that consumer**: an aperture spans many pulse batches by construction, so the trajectory must evolve across ticks. This is the wave that adds it, and it is required rather than optional.
- **Versioned and self-tested.** The simulator is a first-class test dependency living outside this repository, so the repo must **pin a version** and its self-checks must ship with it. The committed CPHD fixture used by integration tests must be regenerable from that pinned version, and regeneration must reproduce it bit-for-bit — otherwise a silent simulator change becomes a silent change in what the module is validated against.

---

## Real collections

The convention leg. Real data is where frame, sign, epoch and reference-point conventions are pinned, because those are exactly what a simulator cannot catch: a simulator sharing an assumption with the operator under test validates the assumption into existence.

**Umbra CPHD collections are in hand.** Matching SICD products are also available and are **authoritative for geometry** — scene centre, grid vectors, spacings — making them the reference for whether the image is in the right place at the right scale.

**SICD is not usable as a pixel-level golden reference.** The provider's image-formation chain includes autofocus corrections not recoverable from the delivered product, so a pixel comparison would measure the difference between two focus solutions rather than the module's correctness. Used that way it would produce a test that fails for the wrong reason and gets tuned until it passes.

---

## Test structure

**Settled: the simulator never feeds an operator directly.** It always writes a CPHD file, and tests read that file through the real `CPHDReaderOp`. Three consequences, all deliberate:

- **There is no in-process simulator source operator** and no C++ port of the signal model. One signal model exists, in Python, and it stays there.
- **Integration tests exercise the same ingress path as production.** The reader is not bypassed, so its field interpretation is under test on every integration run rather than only in its own unit tests.
- **Unit tests construct their inputs inline** — range profiles, positions, and sample points assembled as tensors in fixture code. A single point target's range profile is a few lines of closed-form arithmetic; it does not need a simulator, and using one would couple every unit test to the signal model.

**This deliberately departs from the FMCW pattern**, where `point_target_sim.hpp` is called in-process from GTests. The SAR simulator already exists in Python and already emits CPHD, so porting it to C++ would create a second signal model to keep in sync — the exact duplication the FMCW arrangement avoids by having only one. The FMCW choice was right for its constraints; this one is right for these. The cost is accepted and named: the simulator is outside the repo's own test suite, which is why the pinning and regeneration requirements above exist.

---

## Per-operator high-level tests

Each operator lists the I/O-level properties to validate. Unless noted, all also carry the module-wide checks in *Cross-cutting tests*.

Ten of the fifteen contracted operators appear below. The five omitted — `CRSDReaderOp`, `RVPRemovalOp`, `BandEqualizationOp`, `CRSDWriterOp`, `SIDDWriterOp` — have deferred port lists ([`OPERATOR_API.md`](OPERATOR_API.md) → *Open items*, I2), so there is nothing yet to write assertions against; their test requirements land with their contracts. The deramped pair additionally cannot be satisfied synthetically at all, which is a sequencing constraint rather than a testing one — see [`SIGPROC_PLAN.md`](SIGPROC_PLAN.md).

### Ingress

#### CPHDReaderOp

- **Round trip, qualified by sample type.** Read a collection, write it back with `CPHDWriterOp`, read it again. Positions, reference ranges, pulse times and the declared fast-time domain all match exactly — they are floating-point metadata, independent of the sample representation. Phase history matches **exactly for `cf32`**, and for `ci16` and `ci8` only within that representation's quantization.
- **Quantized comparisons are scale-factor-aware.** The per-vector scale factor must be applied before the residual is judged, or the test measures the scale factor rather than the round trip. `ci8` gets its own case rather than riding on `ci16`: its far coarser step is exactly where a tolerance written for 16-bit passes silently. Same class as the `amp8phs8` qualification already made for SICD.
- Emitted `positions` match the file's Aperture Reference Point record; a `coordinate_frame` disagreeing with the file's declared frame is an **error**, not a silent reinterpretation.
- `pulses_per_tick` batching: the concatenation of all ticks equals the selected pulse range, with nothing dropped or duplicated at batch boundaries.
- **The final batch is short** when the range is not a multiple of the batch size, and downstream still forms a complete image — the regression that catches a stage reading the parameter instead of the tensor.
- `pulse_range` selects the requested subset; an out-of-bounds range errors.
- A truncated or malformed file fails with a clear error rather than emitting partial garbage.
- `cf32`, `ci16` and `ci8` sample types all decode, with the per-vector scale factors applied.
- `sample_axis` matches the file's declared fast-time domain and sample spacing — frequency for an FX-domain collection, time of arrival for a TOA-domain one — and is distinct by construction from the compressed `range_bins` axis it must not be confused with.
- **Multi-file `path`.** A directory and an explicit ordered list both read the whole sequence; the directory form is enumerated lexicographically and the list form in the order given. Pulses from consecutive files are never merged into one batch, so a file boundary always coincides with a tick boundary — the property the downstream flush-and-reset depends on. `pulse_range` applies per file.
- `sample_domain` survives the round trip: an FX-domain collection is not silently rewritten as TOA or the reverse. Written as part of the round-trip case above, since it is exactly what that test would otherwise miss.

### Signal processing

#### SARRangeCompressionOp

- Single point target → the `range_profiles` peak maps, via `range_bins`, to the target's differential range within one bin; `range_bins` itself matches the analytic axis for the configured bandwidth, `nfft` and `propagation_speed`.
- **Transform direction and bin-zero sense.** A target at a positive differential range appears at an increasing bin index, and a target at the derived reference range peaks at the **centre** bin `N/2`. This is the assertion that catches an inverted transform or a **missing** range-axis `fftshift` — both of which pass a magnitude-only resolution test. A target nearer than the reference must appear *below* the centre bin rather than vanishing, which is the specific regression an uncentred axis produces.
- **Half-bin sensitivity.** A target placed at a known sub-bin offset lands where predicted in phase, not merely in magnitude. This is what distinguishes an `N/2` centring from an `(N − 1)/2` one for even `N`; magnitude-only assertions cannot see the difference.
- **Normalization.** Absolute output amplitude matches the analytic prediction for each `fft_norm` value, so the `1/N` lands on the direction the contract says it does (I5).
- Windowing lowers peak sidelobe level against rectangular by the amount the window's theoretical PSLR predicts.
- `nfft` oversampling adds bins and interpolates the peak — no added resolution, no added extent.
- Leading batch axes preserved and processed independently.

#### RangeMatchedFilterOp

- A direct-sampled pulse containing a delayed replica → after the matched filter and the downstream range compression, a peak at the delay's range.
- The output length is `nfft`, and an `nfft` below the wrap-free bound is rejected rather than silently producing circular contamination.
- An explicit `replica` overrides the synthesized one, and a deliberately mismatched replica degrades the peak as predicted.

### Image formation

#### BackprojectionOp

The anchor, and the operator with the most contract surface.

**Image quality, against the simulator:**

- Point target → peak at the correct sample point, within a fraction of a resolution cell.
- Peak **phase** matches the analytic two-way path. This is the assertion that catches range-computation errors a magnitude-only test passes over, and it is also the one that polices **phase-controlled interpolation**: the contract's case for two-tap interpolation rests on the carrier phase being applied from the exact computed range rather than being carried through the interpolator. An implementation that interpolates the complex samples directly still produces a plausible-looking image, and this assertion is what fails on it.
- Range resolution, PSLR and ISLR match the simulator's predicted metrics for the configured bandwidth and `range_window`.
- Cross-range resolution matches the prediction for the subtended aperture angle, and cross-range sidelobes match the **unweighted** aperture — there is no azimuth weighting in the contract (R6). Asserting a weighted figure here would be asserting a parameter that does not exist; when R6 resolves, this assertion changes with it.
- Multiple targets superpose linearly.
- A target beyond the unambiguous range folds where predicted rather than vanishing.

**Geometry and conventions:**

- Grid mode and points mode produce the same values at the same physical locations — the modes differ in output rank, not in the answer.
- A grid on a tilted, non-axis-aligned plane forms correctly; a sheared (non-orthogonal-axis) grid forms correctly.
- The `range_bins` port, when connected, supersedes `range_bin_spacing` / `range_bin_origin`, and the two wirings agree; a non-uniform `range_bins` errors.
- **The default range-bin origin is centred.** With `range_bins` unconnected and `range_bin_origin` unset, bin zero sits at `−⌊N/2⌋ · range_bin_spacing`, so a scene straddling the reference point forms **completely** — the regression being that an uncentred default drops everything nearer than the reference. An explicit `range_bin_origin` overrides it, which is the range-cropped case.
- **`reference_position` carries a point, and the derived range matches.** The per-pulse range derived from it equals the analytic distance from the phase centre to the reference point. The two precision strategies agree where both are valid: the reformulated single-precision path matches the double-precision direct subtraction at short range, which is the assertion that justifies carrying a position rather than a distance.
- **The phase reference is the waveform's centre frequency.** Forming with a band-edge reference instead leaves a predicted residual phase ramp across the aperture, so the test asserts that ramp is absent. Needed because both choices produce a plausible-looking image and differ only in phase.
- `interpolation = nearest` and `interpolation = sinc` are **rejected with an error**, not quietly serviced as `linear`. Asserting the rejection is what keeps the deferral honest — a silent fallback would let a caller believe they had bought impulse-response quality they did not get. Same pattern as the non-uniform TDM schedule rejection (L12).
- **Positions-port convention assertions.** Each pinned property gets a test that *fails when it is violated*, because no library operator encodes them and the specification is the only defense. The simulator's separable trajectory-error input is what makes these constructible:
  - a frame mismatch between `positions` and `sample_points` is **not** detectable by this operator, and the test says so rather than asserting a rejection the contract cannot deliver: both ports are plain tensors carrying no frame identity, and `coordinate_frame` is a single parameter, so there is no second frame to compare against. The assertion is on the *failure signature* — a displaced or defocused image — recorded as the R5 metadata-plane gap in its sharpest form;
  - a length-unit error produces a detectably wrong scale;
  - a reversed or offset pulse ordering defocuses detectably;
  - positions at receive instants rather than transmit produce the predicted along-track shift;
  - a lever-arm offset (phase centre versus platform reference) produces the predicted displacement.

**Accumulation (a new test pattern in this module):**

- Multi-tick: forming one aperture in `N` batches equals forming it in one batch, within tolerance.
- Emit-and-continue: a sliding window emits repeatedly, and each emitted image equals a from-scratch formation over the same pulse set. Because backprojection is linear in pulses, this is exact to numerical tolerance rather than approximate.
- Emit-and-reset: the accumulator clears; the next image carries no contribution from the previous aperture.
- A `[live]` geometry override (`grid_origin`, `grid_axis_*`, `grid_spacing`, `grid_counts`, or `sample_points`) **forces a reset** and warns; partial sums from the old geometry never leak into the new image.
- `initial_image` seeds the accumulator: seeding with a prior result and adding the next block equals forming both blocks together. Arriving mid-aperture **replaces** the partial sums rather than adding to them, and warns; a shape disagreeing with the active sample geometry errors.
- **Termination flush:** a collection ending mid-aperture still emits its partial image rather than dropping it.
- The emitted buffer is independent of the accumulator — mutating the accumulator after emit does not alter the emitted image.
- **Batch semantics:** leading axes produce independent images, each with its own accumulator, and a batched run equals the same batches run separately.

**Precision:**

- The `precision_mode` values agree with one another within expected tolerances at short (airborne) ranges.
- At spaceborne ranges, `fp32` is demonstrably degraded while `mixed` and `float_float` hold. This test exists to make the platform dependence *visible* rather than folklore, so a caller choosing `fp32` for speed at long range sees it fail rather than shipping a quietly defocused product.
- `taylor` matches `mixed` at airborne geometry and is exercised at the aperture extent where its expansion is least accurate.

#### AutofocusOp

- **The canonical test: a simulated collection with a known injected phase error.** Assert the estimated `phase_correction` recovers the injected error, allowing for exactly the linear and constant terms PGA cannot observe — which correspond to an unobservable position shift and an absolute phase, and nothing more.
- The trajectory-error path is the second, independent leg: perturb the trajectory, form with the erroneous positions, and assert autofocus recovers the resulting phase and restores the impulse response to near its error-free metrics.
- A well-focused image passes through essentially unchanged: autofocus on already-good data must not degrade it.
- The `sharpness` output improves monotonically across iterations on a defocused input, and is flat on a focused one.
- Fewer than `min_scatterers` strong scatterers → pass through, zero `phase_correction`, warn. An estimate exceeding `max_correction` → pass through and warn.
- `azimuth_axis` is required: omitting it errors at build. Naming the wrong axis produces a detectably worse image, which is the regression that justifies the parameter having no default.
- `sample_geometry_mode = points` upstream → build-time error, since there is no azimuth axis to name.
- **The grid-conjugacy assumption is tested explicitly, whichever way R4 resolves** — either a non-conjugate geometry is rejected with a clear error, or the chosen alternative is shown to work on one. What must not happen is silently producing a wrong answer on an arbitrary ground grid.

### Products

#### MultilookOp

- Speckle standard deviation over a homogeneous region falls as roughly `1/sqrt(looks_v · looks_u)`. This is the property the operator exists for, so it is the property to assert — and it needs a **synthetic distributed-clutter field**, not the point-target simulator, which models no speckle.
- Mean intensity is preserved (the estimator is unbiased in radiometry).
- Output dimensions follow `looks`; `remainder = truncate` drops the ragged block, `remainder = error` rejects.
- `output_scale` conversions are correct at known values; `db_floor` prevents `-inf` on zero samples.

#### GeocodeOp

- A synthetic scene with known ground-control points lands at the right map coordinates.
- Terrain relief with a DEM is placed correctly; without a DEM, the misplacement matches the predicted reference-surface error rather than being arbitrary.
- Samples outside the input footprint carry `no_data_value`.
- A `scene_local` frame without `origin_llh` is rejected.
- A geographic CRS in `projection` is **rejected at build**. Asserting the rejection matters more than most: degrees read as metres produce a well-formed file that is wrong by orders of magnitude, so nothing downstream would catch it.
- Against real data, geolocation agrees with the matching SICD's geometry fields — the one place SICD is authoritative.

### Standard-format I/O

#### CPHDWriterOp

- Round trip through the reader (above), for each `sample_type`.
- **Accumulation:** a collection written across many ticks produces one file whose pulse count and order match the input stream; `stop()` finalizes the header.
- `max_pulses_per_file` rolls files at the configured boundary with no lost or duplicated pulses across the split.

#### SICDWriterOp

- The product opens in an independent reader (SarPy) and its geometry fields match the configured formation geometry.
- Where autofocus ran, the recorded correction matches `phase_correction`.
- Each `sample_type` reads back through that independent reader at the fidelity the representation allows: `cf32` bit-exact, `ci16` and `amp8phs8` within their quantization, which the test states as a tolerance rather than calling a lossy representation a round trip.
- A geometry mirror disagreeing with the upstream formation configuration is **not** detectable by this operator — which is the concrete demonstration of the R5 metadata-plane gap, and is asserted as a known limitation rather than a passing behaviour.
- A formation geometry SICD cannot describe is **rejected at build**, not written with metadata that disagrees with the pixels. The case to construct is a warped or DEM-conformal sample geometry, which `BackprojectionOp` supports deliberately; uniformity of the grid is the test.

#### TensorDumpOp

- `npy` round-trips shape, dtype and values. `raw` round-trips values only, checked against the shape and dtype the test supplied, since the format carries neither.
- `every_n` and `max_files` throttle as configured. `every_n` of zero or negative **errors** rather than being coerced to 1.

---

## Cross-cutting tests

- **End-to-end, simulator.** A scene of known point targets → CPHD → the full wave-one chain → an image whose targets sit at the right places with the predicted impulse responses.
- **End-to-end, real data.** An Umbra collection → the chain → an image whose geometry matches the delivered SICD. Asserts convention correctness, not pixel equality.
- **`[live]` override behavior.** Overriding a live parameter mid-stream changes the output accordingly and warns only on a genuine rebuild, reusing cached configurations. Geometry overrides additionally reset the accumulator.
- **Precision.** Single and double instantiations agree within their tolerances — a separate axis from `precision_mode`.
- **Memory spaces.** Operators accept device, managed and pinned-host tensors and reject pageable host input with a clear error; outputs are device memory. `CPHDReaderOp` is the sanctioned exception, reading host memory and producing device memory.
- **Determinism.** A fixed seed and a fixed pulse batching reproduce identical images. This is stronger than the FMCW requirement, because accumulation order is a new way to lose it: batching the same aperture differently must not change the result beyond floating-point reassociation.
- **Coordinate-frame rejection is narrower than it first looks**, and is tested only where a second frame genuinely exists. **No operator holds two frames:** `coordinate_frame` is one parameter per operator, and `positions` and `sample_points` are plain tensors carrying no frame identity (invariant 1), so there is no in-operator pairing to check and none is asserted.
- **The two real frame checks**, each tested where it lives. `CPHDReaderOp` — and later `CRSDReaderOp` — rejects a `coordinate_frame` disagreeing with the frame its file declares, the file being the second frame that makes the check possible. `GeocodeOp` and every writer reject a frame lacking the `origin_llh` they need to geolocate, so a `scene_local` frame with no anchor is legal upstream and rejected there. Agreement *between* operators is the application's responsibility and is asserted as a known gap under R5, never as a passing behaviour.

---

## CI notes

- Unit tests are fast, deterministic and per-commit. Integration tests need a CPHD fixture: a small simulator-generated collection, committed with the pinned simulator version recorded beside it and regenerable bit-for-bit.
- **Real-collection validation runs out of band**, not per-commit — the files are large and are not repository content. It runs as a periodic or pre-release check.
- The independent-reader checks (SarPy) are a Python-side test dependency, so the SICD assertions live in `tests/python/` even though the writer is C++.
- The simulator's own self-checks run wherever it lives, not here; what this repo owns is the pinned version and the regeneration check.
