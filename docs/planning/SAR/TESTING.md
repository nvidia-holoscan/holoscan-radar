# Holoscan Radar Module — SAR Testing & Simulation Requirements

## Purpose

Defines what the SAR operators must be tested against: (1) the **point-target simulator** — an analytic signal model that produces phase history with known ground truth — and (2) the **high-level, I/O-level tests** each operator must pass. Like the FMCW [TESTING.md](../TESTING.md) one level up, this document stays at the interface level: it specifies *what property* each test asserts, not the fixtures, tolerances, or framework. Lower-level unit tests (kernel internals, numerical tolerances, boundary indexing) are deferred.

Two things differ structurally from the FMCW pass, and both change how tests are wired:

- **The simulator never feeds an operator in-process.** It always writes CPHD first.
- **Several operators have no reference implementation to diff against**, so assertion effort is deliberately unbalanced toward them. The per-operator weighting is in [SIGPROC_PLAN.md](SIGPROC_PLAN.md), since it is a planning concern.

## Test tiers

- **Unit** — one operator, minimal synthetic input with a closed-form answer. Inputs are constructed **inline as tensors** in fixture code, not produced by a simulator.
- **Integration** — a chain fed by a CPHD file the simulator wrote, read through the real `CPHDReaderOp`.
- **Validation** — real collections, asserting convention correctness rather than numerical accuracy.

There is no separate statistical tier as in FMCW, where Monte-Carlo false-alarm rates are the calibration check. Its nearest analogue here is the speckle-statistics assertion on `MultilookOp`.

---

## The point-target simulator (analytic signal model)

The simulator is a **signal model, not a physics or EM solver**: it synthesizes phase history by summing each point target's closed-form contribution over (pulse, fast-time sample), then adds noise. Cheap, deterministic, and its own correctness is easy to reason about.

It lives in Python **in this repository** and writes CPHD via `sarpy`. This section specifies what it must do, because every quantitative assertion in the module depends on it. Its being in-repo is a policy decision rather than a convenience: users are expected to generate their own input, which an out-of-repo simulator cannot support. See [INPUT_DATA.md](INPUT_DATA.md) → *Policy*.

### Inputs it must accept

- **Scene** — a list of point targets, each with a **Cartesian position** `(x, y, z)` (m) in the declared frame, plus a complex amplitude (or an RCS and a range law). Targets are **stationary**; see *Modeling assumptions*.
- **Platform trajectory** — per-pulse antenna phase-center positions in the same frame, supplied either parametrically (start position, velocity, optional acceleration) or as an explicit per-pulse position list. **An arbitrary, non-straight track must be expressible**, because a straight track cannot distinguish a correct backprojection from several plausible wrong ones.
- **Trajectory error, injectable and separable** — a deliberate per-pulse position perturbation, *and* the true trajectory alongside it. The simulator must be able to write the *erroneous* trajectory into the file while reporting the *true* one as ground truth. This is a first-class requirement, not a test convenience: it is the only way to validate autofocus, and the only way to test that a wrong-convention trajectory is detectable.
- **Injected phase error** — a per-pulse phase (or a polynomial in pulse index) added to the phase history, with the applied values reported as ground truth. The direct autofocus ground truth, independent of the trajectory-error path. The simulator keeps this even though `PhaseCorrectionOp` with `sense = add` can impress the same error in-pipeline: an error written into the *file* is what tests the path a real collection takes, and using the operator to generate the error its counterpart is supposed to remove would validate one sign convention against itself.
- **Collection geometry** — the reference point the phase history is referenced against, emitted as `reference_position`, from which the per-pulse reference range follows.
- **Waveform** — the `LFMWaveformDescriptor` fields (carrier frequency, bandwidth or slope, chirp duration, sample rate, pulse repetition interval), plus `num_pulses` and `num_samples`.
- **Noise** — additive complex Gaussian, configurable by SNR or noise power, with a seed.
- **Coordinate frame** — the `CoordinateFrame` the scene and trajectory are expressed in, recorded in the output file so a frame mismatch is detectable rather than assumed.

### Outputs it must produce

- **A CPHD file** carrying the phase history, per-pulse positions, reference positions, pulse times, and the collection metadata a reader needs. This is the only interface to the operators.
- **Ground truth alongside it** (a sidecar, not inside the CPHD): each target's Cartesian position; its slant range at aperture centre; the **predicted image-quality metrics** for the configured collection — range resolution `≈ c / (2·B)`, cross-range resolution `≈ λ / (2·Δθ)` for the subtended aperture angle `Δθ`, the range PSLR and ISLR the configured range window implies, and the cross-range PSLR and ISLR for both an unweighted aperture and the configured `aperture_window`; the true trajectory where an error was injected; and the injected phase error where one was applied.
  - **Not pixel indices.** Those depend on the formation configuration — grid origin, spacing, axes, `nfft` — which the simulator does not know.
  - Tests therefore assert in the **physical domain**: map an image peak back to a position through the formation sample geometry and compare to ground truth within a fraction of a resolution cell. That also validates the sample geometry itself.
  - The predicted metrics are what make the assertions non-circular: they come from the collection parameters, not from a previous run of the pipeline.

### The signal model (per target, summed)

Phase history is the deramped, frequency-domain form CPHD stores. With `p_k` the phase-center position at pulse `k`, `t` the target position, `R_k = ‖p_k − t‖` the one-way range, and `R_ref,k` the reference range, fast-time sample `s` corresponds to transmitted frequency `f_s = f_start + s·Δf`. The contribution to sample `(k, s)` is the product of:

- the **differential-range phase** `exp(−j·4π·f_s·(R_k − R_ref,k) / c)` — the term that makes range compression a transform along `s` and that carries the aperture phase along `k`;
- the target's **complex amplitude**, times an optional range law (`R⁻³` or `R⁻⁴`);
- an optional **injected per-pulse phase** `exp(j·φ_k)`;
- an optional scalar **antenna-pattern weighting** as a function of the pulse's look direction to the target.

Sum over targets, add noise. **Ranges are computed in double precision** so the simulator is never the accuracy limit in a `backend` comparison.

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
- **Determinism, and the limit of it.** A fixed seed on a fixed toolchain reproduces the file bit-for-bit, which is what a regeneration check on one machine may assert. **Across platforms it is not claimed**: floating-point and library differences make cross-platform reproduction a bar to be established rather than assumed, and what "stable enough" means for a generated fixture is open question 4 in [INPUT_DATA.md](INPUT_DATA.md). Until that settles, a test consuming a generated fixture asserts **analytic properties within tolerances** and never a checksum of the file.
- **Guard against tautology.** The signal model must be derived **independently of the operator implementations**, and canonical cases cross-checked against an independent reference — a textbook closed form for the point-target response, or an established external SAR simulator. Tests assert against **analytic predictions** (a target at a known position, imaged at the resolution the bandwidth and aperture imply, with the range sidelobe levels the range window implies), never merely that the generator round-trips through the pipeline.
- **A kinematic driver is required.** FMCW's `TESTING.md` defers a time-stepping driver, noting it waits for "a frame-spanning consumer (a tracker, multi-frame integration, or clutter-map CFAR — all out of scope now)". **SAR is that consumer**: an aperture spans many pulse batches by construction, so the trajectory must evolve across ticks. This is the wave that adds it, and it is required rather than optional.
- **Self-tested, and versioned with the repository.** Its self-checks ship with it, in `tests/` alongside the operator suites, and a change to the signal model is a change to this repository reviewed like any other. *Amended 2026-08-06:* this requirement previously demanded a **pinned version** of an out-of-repo simulator and a committed CPHD fixture regenerable from it **bit-for-bit**. Both existed to contain an external-dependency hazard that no longer exists — the simulator is in-repo and no CPHD is committed ([INPUT_DATA.md](INPUT_DATA.md) → *Policy*). The version pin is therefore dropped outright. What replaces bit-for-bit is below.

---

## Real collections

The convention leg. Real data is where frame, sign, epoch and reference-point conventions are pinned, because those are exactly what a simulator cannot catch: a simulator sharing an assumption with the operator under test validates the assumption into existence.

**Umbra CPHD collections are in hand.** Matching SICD products are also available and are **authoritative for geometry** — scene centre, grid vectors, spacings — making them the reference for whether the image is in the right place at the right scale.

Which collections, their keys, headers and pull cost are in [INPUT_DATA.md](INPUT_DATA.md) → *Public collections, per mode*; they are credential-free and pinned per first-ship mode rather than chosen per test.

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

Twelve of the seventeen contracted operators appear below. The five omitted — `CRSDReaderOp`, `RVPRemovalOp`, `BandEqualizationOp`, `CRSDWriterOp`, `SIDDWriterOp` — have deferred port lists ([OPERATOR_API.md](OPERATOR_API.md) → *Open items*, I2), so there is nothing yet to write assertions against; their test requirements land with their contracts. The deramped pair additionally cannot be satisfied synthetically at all, which is a sequencing constraint rather than a testing one — see [SIGPROC_PLAN.md](SIGPROC_PLAN.md).

### Ingress

#### CPHDReaderOp

- **Round trip, qualified by sample type.** Read a collection, write it back with `CPHDWriterOp`, read it again. Positions, reference positions, pulse times and the declared fast-time domain all match exactly — they are floating-point metadata, independent of the sample representation. Phase history matches **exactly for `cf32`**, and for `ci16` and `ci8` only within that representation's quantization.
- **Quantized comparisons are scale-factor-aware.** The per-vector scale factor must be applied before the residual is judged, or the test measures the scale factor rather than the round trip. `ci8` gets its own case rather than riding on `ci16`: its far coarser step is exactly where a tolerance written for 16-bit passes silently. Same class as the `amp8phs8` qualification already made for SICD.
- Emitted `positions` match the file's Aperture Reference Point record; a `coordinate_frame` disagreeing with the file's declared frame is an **error**, not a silent reinterpretation.
- `pulses_per_tick` batching: the concatenation of all ticks equals the selected pulse range, with nothing dropped or duplicated at batch boundaries.
- **The final batch is short** when the range is not a multiple of the batch size, and downstream still forms a complete image — the regression that catches a stage reading the parameter instead of the tensor.
- `pulse_range` selects the requested subset; an out-of-bounds range errors.
- A truncated or malformed file fails with a clear error rather than emitting partial garbage.
- `cf32`, `ci16` and `ci8` sample types all decode, with the per-vector scale factors applied.
- `sample_axis` matches the file's declared fast-time domain and sample spacing — frequency for an FX-domain collection, time of arrival for a TOA-domain one — and is distinct by construction from the compressed `range_bins` axis it must not be confused with.
- **Multi-file `path`.** A directory and an explicit ordered list both read the whole sequence; the directory form is enumerated lexicographically and the list form in the order given. Pulses from consecutive files are never merged into one batch, so a file boundary always coincides with a tick boundary — the property the downstream flush-and-reset depends on. `pulse_range` applies per file.
- **The boundary announcement lands on the last batch of a collection, not the first of the next (D22).** `holoscan.radar.stream_boundary` appears exactly once per seam, valued `collection_boundary` at each file seam and `end_of_stream` on the final batch, and appears on no other tick. The single-file case emits only `end_of_stream`. Stamping the *following* batch instead is the plausible off-by-one and the reason the test asserts which batch carries the key rather than only that some batch does — a boundary announced one batch late has already contaminated the closing aperture.
- **`coordinate_frame` is emitted on the descriptor output** and equals the frame validated against the file, so a consumer wiring it receives the same value the reader is using rather than a re-derivation.
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

#### ApertureManagerOp

**The one operator in this modality that is fully testable without a GPU**, which is most of the reason it exists as a separate stage: aperture bookkeeping is where off-by-one errors live, and here they are caught by asserting on emitted pulse identities and weights rather than by looking at an image. These run in the CPU-only tier.

- **Membership under manager-owned eviction.** With `max_aperture_pulses = N` and `evict` unconnected, the first `N` admitted pulses produce no evictions; from pulse `N + 1` onward every tick evicts exactly as many as it admits, and the evicted pulse is the oldest resident one. With `max_aperture_pulses = 0` nothing is ever evicted.
- **No retention exists when the capacity is zero.** With `max_aperture_pulses = 0` the operator holds no pulse handles at all across a long run — asserted on retained-buffer occupancy, not on output, since the symptom of getting this wrong is unbounded memory growth over a collection rather than a wrong image.
- **Membership under application-owned eviction.** With `evict` connected, a count of `k` retires exactly the `k` oldest pulses and no others; a count of `0`, or no message at all, retires nothing; and the manager never retires a pulse on its own no matter how full the buffer is.
- **Eviction precedes admission.** A count arriving on the same tick as new pulses retires from the pre-admission buffer, so "oldest" never includes a pulse admitted on that tick. Asserted on pulse identities, because the two orderings differ only when the count approaches the occupancy — which is exactly when an application is at its aperture limit.
- **Capacity overflow errors rather than dropping.** With `evict` connected, admitting past `max_aperture_pulses` without a retiring command fails loudly. The competing behaviour — dropping the oldest — leaves a running graph whose aperture is quietly not what the application believes, so the test asserts the failure, not a recovery.
- **Weights and ordering.** Admitted pulses carry `+1`, departing pulses `−1`, and both appear in one batch per tick, so a consumer never has to reason about the order of two messages.
- **Departing pulses carry the geometry they were admitted with**, not the current tick's. This is the assertion that makes eviction exact; re-emitting current geometry would subtract a contribution that was never added, and the resulting image error is subtle enough to survive casual inspection.
- **Emit cadence.** A control message appears after exactly `emit_interval_pulses` admissions since the previous one, carries the value `emit_mode` selects, and appears on no other tick.
- **`emit_interval_pulses = 0` emits on no count at all** — no control message appears until an upstream flush request or end of stream, which is the single-image-per-collection configuration and must not require knowing the collection length.
- **The six configurations each produce the documented behaviour**, asserted on emitted weights and control messages: one image at end of stream, disjoint blocks, a growing aperture, a fixed-length sliding window, an application-length sliding window, and the Figure 2 reform row. Five of the six are asserted with **no application-owned ports wired at all**, which is the claim the table makes and the one a regression would quietly break. The sliding and block rows are additionally checked to give identical images when configured for the same pulse sets, since the two paths reach the same sum differently.
- **`emit_interval_pulses ≥ max_aperture_pulses` warns** under manager-owned eviction, because consecutive images then share no pulses and the cheaper `max_aperture_pulses = 0` path would produce the same result.
- **A variable-length aperture stays exact.** Driving `evict` with a varying count — the orbiting-spotlight case, where pulses per degree changes with ground speed — leaves each emitted image containing exactly the pulses the application intended, asserted against a directly formed image over that same pulse set. This is the assertion that the application's accounting and the manager's buffer cannot drift apart.
- **Termination on the announcement.** A collection ending mid-aperture is flushed rather than dropped when the tick's metadata carries `holoscan.radar.stream_boundary = end_of_stream` (D22, algorithm step 7), with no `control` edge wired at all — that being the configuration the announcement exists to make work. An explicit request on the `control` input produces the identical emit, so the two paths are asserted equivalent rather than each specified separately. The manager is still asserted *not* to invent a flush from nothing: absent both an announcement and a request, the partial aperture stays resident and only `BackprojectionOp`'s `stop()` warning fires. Paired with `BackprojectionOp`'s termination-flush test, which is the other half of the same guarantee.
- **A collection boundary flushes and re-arms; a file seam is the case that matters.** With a two-file `path`, the batch stamped `collection_boundary` produces an emit and a reset, the first pulses of the second file land in a fresh aperture, and **no image contains pulses from both files** — asserted on pulse membership rather than on image content, since a contaminated image is plausible-looking. Where a staged geometry is pending, it is forwarded on that same tick, so downstream never forms the second collection against the first's lattice.
- **The announcement survives an intervening operator and a coalescing one.** Inserting a passthrough between reader and manager does not lose the key, and neither does the manager's own retention — the property that motivated metadata over a control port. A fragment with metadata disabled, or with the SDK default `kRaise` policy, is asserted to fail visibly (the announcement dropped, or a merge throwing on a repeated key) rather than to degrade quietly, so the application-level requirement is covered by a test rather than only by documentation.
- **Short final batch.** A batch shorter than the rest is admitted and accounted normally, since a file-backed source runs out mid-batch.
- **Conservation.** Over a whole run, every admitted pulse is eventually either evicted, dropped by a reset, or still resident at termination — the invariant that catches a leak or a double-eviction regardless of which rule produced it.
- **Reset drops retention silently.** An upstream `control` request, or a staged geometry change, makes the manager clear its retained pulses and emit **no** eviction weights for them. The distinction between dropping and evicting is the single most error-prone thing in this operator, so it is asserted directly rather than inferred from an image.
- **Geometry is held and released at a boundary.** A `sample_geometry` or `sample_points` arriving mid-aperture is not forwarded until the aperture ends, and is then forwarded on the same tick as the control message — one producer, one tick, so downstream never sees one without the other.
- **Tapering.** With `aperture_window` set on a block aperture, the emitted weights are the window sampled over the aperture's known length, in slot order. Under `emit_mode = reform` the same weights appear on the replayed batch, computed against the membership as it stood at the boundary rather than as it stood when each pulse was admitted — the distinction that makes the taper exact.
- **Tapering an unformable configuration is refused at initialization.** `aperture_window` set on a sliding or growing aperture without `emit_mode = reform` fails, rather than silently tapering against a membership that was not final when the weights were computed. `emit_mode = reform` with `max_aperture_pulses = 0` fails for the complementary reason: there is no retained set to replay, so a growing or collection-length aperture must be given a capacity before it can be tapered — and the same capacity is what Figure 2 autofocus needs for the corrected product pass.
- **Re-form replaces rather than supplements.** Under `emit_mode = reform` no pulse batch is emitted between boundaries, and the boundary batch is the entire retained set with *reset-then-emit*. A re-formed image is asserted bit-comparable, to tolerance, against forming the same pulse set directly.
- **Control phases.** *reset-without-emit* and *reset-then-emit* clear before the tick's pulses are accumulated, so those pulses survive into the new aperture; *emit-and-reset* clears after the emit, so the tick's pulses appear in the image that comes out. Asserted on which pulses land in which image, because getting the phase backwards produces an image short or long by one tick's worth of pulses — a defect too subtle to see.
- **Geometry lands in its reset's phase.** A geometry change accompanying *emit-and-reset* does not resize the accumulator until after that tick's image is emitted. This is the assertion for the latent bug the phase model fixed.
- **Build-time wiring.** A graph with `BackprojectionOp.control` unconnected fails at initialization with a message naming the port, rather than running and silently emitting nothing. Connecting both `sample_geometry` and `sample_points` fails the same way, as does connecting `evict` with `max_aperture_pulses = 0`.

#### BackprojectionOp

The anchor, and the operator with the most contract surface.

**Image quality, against the simulator:**

- Point target → peak at the correct sample point, within a fraction of a resolution cell.
- Peak **phase** matches the analytic two-way path. This is the assertion that catches range-computation errors a magnitude-only test passes over, and it is also the one that polices **phase-controlled interpolation**: the contract's case for two-tap interpolation rests on the carrier phase being applied from the exact computed range rather than being carried through the interpolator. An implementation that interpolates the complex samples directly still produces a plausible-looking image, and this assertion is what fails on it.
- Range resolution, PSLR and ISLR match the simulator's predicted metrics for the configured bandwidth and `range_window`.
- Cross-range resolution matches the prediction for the subtended aperture angle. Cross-range sidelobes match the **unweighted** aperture where no taper is supplied, and match the figure a configured `aperture_window` predicts where one is — asserted both for a block aperture, where the manager stamps weights at admission, and for a re-formed aperture, where it computes them at the boundary. An incrementally-slid window is asserted to be unweighted, since that is the decided behaviour rather than a gap.
- Multiple targets superpose linearly.
- A target beyond the unambiguous range folds where predicted rather than vanishing.

**Geometry and conventions:**

- The two geometry modes produce the same values at the same physical locations, **and the same output shape**: handing `sample_points` a lattice that reproduces the `sample_geometry` descriptor gives an identical image, since the modes differ only in how positions are specified. Connecting both is a build-time error.
- **The lattice locality check fires and stays silent in the right cases.** A deliberately scattered `(num_v, num_u, 3)` lattice warns when a Taylor variant is selected; a spatially coherent one does not; and no check runs for a uniform lattice or under an exact-range variant. Both directions are asserted so the check cannot rot unnoticed, and the warning is never an error — a caller who does not care about sub-wavelength phase still gets an image.
- A grid on a tilted, non-axis-aligned plane forms correctly; a sheared (non-orthogonal-axis) grid forms correctly.
- The `range_bins` port, when connected, supersedes `range_bin_spacing` / `range_bin_origin`, and the two wirings agree; a non-uniform `range_bins` errors.
- **The default range-bin origin is centred.** With `range_bins` unconnected and `range_bin_origin` unset, bin zero sits at `−⌊N/2⌋ · range_bin_spacing`, so a scene straddling the reference point forms **completely** — the regression being that an uncentred default drops everything nearer than the reference. An explicit `range_bin_origin` overrides it, which is the range-cropped case.
- **`reference_position` carries a point, and the derived range matches.** The per-pulse range derived from it equals the analytic distance from the phase centre to the reference point. The two precision strategies agree where both are valid: the reformulated single-precision path matches the double-precision direct subtraction at short range, which is the assertion that justifies carrying a position rather than a distance.
- **The phase reference is the waveform's centre frequency.** Forming with a band-edge reference instead leaves a predicted residual phase ramp across the aperture, so the test asserts that ramp is absent. Needed because both choices produce a plausible-looking image and differ only in phase.
- `interpolation = nearest` and `interpolation = sinc` are **rejected with an error**, not quietly serviced as `linear`. Asserting the rejection is what keeps the deferral honest — a silent fallback would let a caller believe they had bought impulse-response quality they did not get. Same pattern as the non-uniform TDM schedule rejection (L12).
- **Positions-port convention assertions.** Each pinned property gets a test that *fails when it is violated*, because no library operator encodes them and the specification is the only defense. The simulator's separable trajectory-error input is what makes these constructible:
  - a frame mismatch between `positions` and `sample_points` is **not** detectable by this operator, and the test says so rather than asserting a rejection the contract cannot deliver: both ports are plain tensors carrying no frame identity, and `coordinate_frame` is a single Static parameter here, so there is no second frame to compare against. The assertion is on the *failure signature* — a displaced or defocused image. This is the residue D21 deliberately leaves: the optional descriptor edges reach the operators that *describe* a frame, not this one, so the gap is asserted as a known limitation rather than as something a wiring choice can close;
  - a length-unit error produces a detectably wrong scale;
  - a reversed or offset pulse ordering defocuses detectably;
  - positions at receive instants rather than transmit produce the predicted along-track shift;
  - a lever-arm offset (phase centre versus platform reference) produces the predicted displacement.

**Accumulation (a new test pattern in this module):**

- Multi-tick: forming one aperture in `N` batches equals forming it in one batch, within tolerance.
- **Signed weights are the eviction mechanism.** Accumulating a pulse at `+1` and later at `−1` returns the accumulator to where it was, to numerical tolerance; a weight of `0` contributes nothing; and a fractional weight scales the contribution linearly. This is the assertion the whole sliding-window story rests on.
- Emit-and-continue: a sliding window driven by the manager emits repeatedly, and each emitted image equals a from-scratch formation over the pulse set then resident. Exact to numerical tolerance rather than approximate, because backprojection is linear in pulses.
- **The floating-point drift is measured, not assumed.** A long add-then-subtract run is compared against periodic re-formation, and the divergence is gated against an empirically calibrated tolerance with headroom, with the observed growth recorded. `sqrt(N)·eps` is the random-walk *expectation* rather than a bound: it assumes independent rounding, and a long add-then-subtract run can round with bias instead, so conforming code may exceed it and asserting it as a limit would fail good implementations. What the threshold must catch is drift growing *linearly* in `N`, or divergence large enough to matter against the image's dynamic range. The contract previously claimed sliding windows had no drift to bound; this test is what stops that claim returning.
- Emit-and-reset: a *reset* control message clears the accumulator; the next image carries no contribution from the previous aperture. A *reset-without-emit* message clears it and emits nothing.
- An absent `control` message emits nothing, and an absent `weight` port is equivalent to all-`+1`.
- **A geometry change without a reset errors.** A `sample_geometry` or `sample_points` override arriving on a tick with no resetting control message is rejected, naming the missing message — it is not deferred, not applied silently, and not allowed to produce an image of the wrong scene. This is the assertion that keeps the co-emission protocol honest.
- **A geometry change with a reset applies within the tick**, and partial sums from the old lattice never leak into the new image. Sending the two in either order produces bit-identical results, which is what keeps the design independent of message ordering.
- **The operator holds no pending configuration.** After a rejected change, the next accumulation still uses the original geometry — asserted so a future implementation cannot quietly reintroduce buffering.
- **`BackprojectionOp` never resets on its own.** Overriding geometry with no reset following produces no reset — asserted directly, because the regression it guards against silently corrupts a sliding window rather than failing.
- **A reset drops the manager's retention without evicting.** Across a reset the manager emits no `−1` weights for the pulses it was holding, and the next image contains no phantom subtraction. Asserted on emitted weights rather than on the image, since the image symptom is a mild defocus that a loose tolerance would pass.
- `initial_image` seeds the accumulator: seeding with a prior result and adding the next block equals forming both blocks together. Arriving mid-aperture **replaces** the partial sums rather than adding to them, and warns; a shape disagreeing with the active sample geometry errors.
- **Termination flush:** a collection ending mid-aperture still emits its partial image rather than dropping it, driven by the boundary announcement reaching whoever owns `control` (D22).
- **`stop()` warns on an unemitted accumulator, and only then.** Wiring a control owner positioned where it cannot receive the announcement — a policy operator fed by nothing, the case Figure 1 previously drew — leaves pulses accumulated at termination, and the operator warns naming the pulse count. The complementary assertion matters as much: a run whose final aperture *was* emitted produces no warning, so the check cannot be satisfied by warning unconditionally. Together these are the only coverage a lost tail can have, since `stop()` cannot emit and no build-time check can see whether a source is finite.
- The emitted buffer is independent of the accumulator — mutating the accumulator after emit does not alter the emitted image.
- **Batch semantics:** leading axes produce independent images, each with its own accumulator, and a batched run equals the same batches run separately.

**Scrolling accumulation (`geometry_change = scroll`):**

- **A translated strip equals a from-scratch formation.** A scrolled run over a simulated pass is compared pixel-by-pixel against forming each row's full illumination interval directly, over the region where the scroll had the whole interval in view. This is the mode's correctness claim — partial sums survive the translation — and everything else here guards a way of breaking it.
- **A point target at the ring's wrap position forms correctly.** The buffer is a ring, so a scroll puts the wrap at a known row; a target placed there must form to the same tolerance as one anywhere else. This is the assertion that fails if the two kernel launches are ever merged into one, which is otherwise an invisible optimization that distorts whichever block straddles the wrap — see [OPERATOR_API.md](OPERATOR_API.md) → `BackprojectionOp` → *Scrolling accumulation*. The test is worth more than the code it covers, because nothing in the implementation explains why there are two launches.
- **Rows entering the leading edge are zeroed**, asserted by scrolling a fully-populated buffer and checking that the entering rows carry nothing from the pixels that previously occupied that memory.
- **The emitted image is in logical order**, not the ring's rotated order — a target's row index in the emitted tensor matches its position in the sample geometry, whatever the ring's rotation happens to be at that moment.
- **Every illegal change is rejected, naming what was wrong**: in grid mode a translation with a `u` component, a fractional row, or a change to spacing, counts or axes; in points mode a `num_u` disagreement or an advance of `k ≥ num_v`; and in both, a connected `weight` port. The port check is at initialization and the rest on arrival, because the alternative to rejecting is a silently mis-registered or smeared strip.
- **Both geometry modes scroll, and reach the same strip.** A DEM-conformal lattice fed as entering-row slabs produces the same image as the equivalent grid descriptor where the lattice reproduces that grid, which is the points-mode counterpart of the two-modes-agree assertion above. The incremental form is additionally checked to leave retained rows' positions untouched — asserted on the lattice ring rather than on the image, since a wrong retained position shows up as a defocus a loose tolerance would pass.
- **The leading margin is emitted, not cropped.** Rows that have not yet seen their full illumination interval come out under-integrated, and the operator neither masks nor removes them. Where consecutive products overlap, the assertion is that a row complete in the later product matches the from-scratch formation, and that the earlier product's copy of the same row is the partial sum it should be. This pins the documented consumer rule (*prefer the later product*) rather than leaving it as prose.
- **So is the trailing margin at end of strip.** The emit triggered by the boundary announcement carries the strip's final rows including those still owing pulses that will never arrive — asserted symmetrically with the leading margin, since the operator applies one rule to both and cropping either is the consumer's decision. The corresponding negative assertion is that the trailing rows are *not* silently dropped, which is what a well-meaning implementation would do.

**Precision:**

- The `backend` values agree with one another within expected tolerances at short (airborne) ranges.
- At spaceborne ranges, `fp32` is demonstrably degraded while `mixed` and `float_float` hold. This test exists to make the platform dependence *visible* rather than folklore, so a caller choosing `fp32` for speed at long range sees it fail rather than shipping a quietly defocused product.
- `taylor` matches `mixed` at airborne geometry and is exercised at the aperture extent where its expansion is least accurate.

#### AutofocusOp

- **The canonical test: a simulated collection with a known injected phase error.** Assert the estimated `phase_correction` recovers the injected error, allowing for exactly the linear and constant terms PGA cannot observe — which correspond to an unobservable position shift and an absolute phase, and nothing more.
- The trajectory-error path is the second, independent leg: perturb the trajectory, form with the erroneous positions, and assert autofocus recovers the resulting phase and restores the impulse response to near its error-free metrics.
- A well-focused image passes through essentially unchanged: autofocus on already-good data must not degrade it.
- The `sharpness` output improves monotonically across iterations on a defocused input, and is flat on a focused one.
- Fewer than `min_scatterers` strong scatterers → pass through, zero `phase_correction`, warn. An estimate exceeding `max_correction` → pass through and warn.
- `azimuth_axis` is required: omitting it errors at build. Naming the wrong axis produces a detectably worse image, which is the regression that justifies the parameter having no default.
- An explicit `sample_points` lattice upstream is **not** rejected on rank — it has nameable axes. What governs is whether the named axis is the Fourier conjugate of slow-time, so a points-mode lattice that *is* conjugate is a legal thing to estimate from, and the test covers both geometry modes.
- **`phase_correction` is indexed by pulse, and the test that pins it is a transfer test**, not a shape assertion: estimate from a conjugate-grid image and apply the result to a formation on a *different* geometry, then assert the second image is focused. Indexing by azimuth bin would pass a shape check and fail this one wherever the two axes have different lengths, which is the failure D20's port change exists to prevent. Run it with the estimation image deliberately range-decimated against the product, since that is the recommended pattern and it is exactly where a bin-indexed correction would appear to work.
- **Figure 2 end-to-end uses retain + reform.** The transfer test above is wired through `ApertureManagerOp` with `max_aperture_pulses > 0` and `emit_mode = reform`: one aperture replay feeds the auxiliary formation and, after `PhaseCorrectionOp`, the product formation on the same tick. Assert the product image is focused and that no pulse batch leaves the manager between boundaries. The AF-off spelling `max_aperture_pulses = 0` is **not** a Figure 2 configuration — there is nothing to replay after AF — and is covered by the AF-off path instead.
- **The declared-not-checked precondition is tested as a hazard, not as a rejection.** The operator cannot see the geometry it was handed, so a wrong `azimuth_axis` or a non-conjugate lattice produces a plausible wrong answer rather than an error — asserted by estimating from a deliberately non-conjugate ground grid and showing the recovered phase does *not* match the injected one. The test exists to document the hazard D20 accepts and to catch anyone later adding a check that would make it silently pass.

#### PhaseCorrectionOp

- **Injected and removed round-trips to identity.** Impress a known per-pulse phase with `sense = add`, remove it with `sense = subtract`, and assert the profiles return to the original within floating-point tolerance.
- **The sign is asserted against a formed image, not against the multiply.** `sense` is the operator's one unverifiable parameter, so the test that matters forms an image from pulses carrying an injected error, corrects with `subtract`, and asserts the impulse response is restored — where the wrong sign doubles the error and is unmissable. A test that only checks the arithmetic would pass with either sign.
- The correction broadcasts across the range axis: every range bin of a pulse receives the same phase, asserted on a profile with distinct per-bin values.
- Pulse-axis length mismatch between `range_profiles` and `phase_correction` errors rather than broadcasting or truncating.

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
- **An `input_sample_geometry` disagreeing with the formation that produced the input misplaces the whole raster**, and the test asserts the predicted displacement rather than a rejection, since the operator has no way to detect it. Wired instead from `BackprojectionOp.sample_geometry` (D21), the same scene geocodes correctly with no parameter set — the pair of assertions that shows what the optional edge buys.

### Standard-format I/O

#### CPHDWriterOp

- Round trip through the reader (above), for each `sample_type`.
- **Accumulation:** a collection written across many ticks produces one file whose pulse count and order match the input stream; `stop()` finalizes the header.
- `max_pulses_per_file` rolls files at the configured boundary with no lost or duplicated pulses across the split.

#### SICDWriterOp

- The product opens in an independent reader (SarPy) and its geometry fields match the configured formation geometry.
- Where autofocus ran, the recorded correction matches `phase_correction`.
- Each `sample_type` reads back through that independent reader at the fidelity the representation allows: `cf32` bit-exact, `ci16` and `amp8phs8` within their quantization, which the test states as a tolerance rather than calling a lossy representation a round trip.
- **The two dispositions of a mirrored parameter, tested as a pair.** With the mirror left as a parameter, a value disagreeing with the upstream formation configuration is **not** detectable, and that is asserted as a known limitation rather than a passing behaviour — the hazard D21 accepts under invariant 5 rather than designs out. With `BackprojectionOp.sample_geometry` wired into the override port instead, the writer records the formation's actual lattice and the disagreement cannot arise, which is the escape hatch D21 offers. Testing both is what keeps the decision honest: only the second is a check, and the contract claims no more.
- **Override precedence, asserted on all three edges.** Where an override port is wired and the mirrored parameter is also set, the **port's** value is what the product records, and a disagreement warns once rather than per message. Where the port is wired and the parameter is unset, initialization succeeds — the parameter being required only when its port is unwired, which is the D3 wiring check doing double duty. Same three assertions for `GeocodeOp.input_sample_geometry` and `coordinate_frame`, since a precedence bug there misplaces the raster rather than mislabelling it.
- A formation geometry SICD cannot describe is **rejected at build**, not written with metadata that disagrees with the pixels. The case to construct is a warped or DEM-conformal sample geometry, which `BackprojectionOp` supports deliberately; uniformity of the grid is the test.

#### TensorDumpOp

- `npy` round-trips shape, dtype and values. `raw` round-trips values only, checked against the shape and dtype the test supplied, since the format carries neither.
- `every_n` and `max_files` throttle as configured. `every_n` of zero or negative **errors** rather than being coerced to 1.

---

## Cross-cutting tests

- **End-to-end, simulator.** A scene of known point targets → CPHD → the full wave-one chain → an image whose targets sit at the right places with the predicted impulse responses.
- **End-to-end, real data.** An Umbra collection → the chain → an image whose geometry matches the delivered SICD. Asserts convention correctness, not pixel equality.
- **`[live]` override behavior.** Overriding a live parameter mid-stream changes the output accordingly and warns only on a genuine rebuild, reusing cached configurations. Geometry overrides additionally reset the accumulator.
- **Precision.** Single and double instantiations agree within their tolerances — a separate axis from `backend`.
- **Memory spaces.** Operators accept device, managed and pinned-host tensors and reject pageable host input with a clear error; outputs are device memory. `CPHDReaderOp` is the sanctioned exception, reading host memory and producing device memory.
- **Determinism.** A fixed seed and a fixed pulse batching reproduce identical images. This is stronger than the FMCW requirement, because accumulation order is a new way to lose it: batching the same aperture differently must not change the result beyond floating-point reassociation.
- **Coordinate-frame rejection is narrower than it first looks**, and is tested only where a second frame genuinely exists. **An operator holds two frames only where one arrives on a port:** `coordinate_frame` is otherwise one parameter, and `positions` and `sample_points` are plain tensors carrying no frame identity (invariant 1), so there is no in-operator pairing to check and none is asserted.
- **The three real frame checks**, each tested where it lives. `CPHDReaderOp` — and later `CRSDReaderOp` — rejects a `coordinate_frame` disagreeing with the frame its file declares, the file being the second frame that makes the check possible. `GeocodeOp` and every writer reject a frame lacking the `origin_llh` they need to geolocate, so a `scene_local` frame with no anchor is legal upstream and rejected there. And where D21's frame edge is wired, the consumer has a second frame at last and warns when its own parameter disagrees. Agreement between operators whose frames are *both* parameters stays the application's responsibility and is asserted as a known gap, never as a passing behaviour — including `BackprojectionOp`, which D21's edges deliberately do not reach.
- **The metadata plane's two failure modes**, asserted once here rather than per operator. Neither is an application precondition — metadata is enabled by default and `kUpdate` is set per-operator by the merging consumer (D22) — so what is under test is that the module configures itself correctly and degrades visibly when something external interferes. First, an operator merging several ports from one producer must not throw, which is the assertion that fails if `BackprojectionOp` ever loses its own `kUpdate`. Second, metadata *explicitly* disabled on an intervening operator loses the boundary announcement, and the resulting dropped final aperture must surface as `BackprojectionOp`'s `stop()` warning rather than as silently missing output.

---

## CI notes

- Unit tests are fast, deterministic and per-commit. Integration tests need a CPHD fixture, which CI **generates from a committed recipe** as a build step rather than checking out — no CPHD binary is in git ([INPUT_DATA.md](INPUT_DATA.md) → *Policy*). The point-target recipes generate in seconds, which is what makes generating rather than committing affordable per-commit.
- **Real-collection validation runs out of band**, not per-commit — the files are large and are not repository content. It runs as a periodic or pre-release check.
- The independent-reader checks (SarPy) are a Python-side test dependency, so the SICD assertions live in `tests/python/` even though the writer is C++.
- The simulator's own self-checks run **here**, with the rest of the suite, since it is now repository content. What this repo owns is therefore the signal model itself, the recipes, and the fixture-generation step ahead of the integration tests — not, as previously written, a pinned external version and a regeneration check against it.
