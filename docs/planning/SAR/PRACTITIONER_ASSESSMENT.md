# Holoscan Radar Module — SAR Practitioner Assessment

## Purpose

An independent read of the proposed SAR modality as of the current
[WORKFLOWS.md](WORKFLOWS.md) / [OPERATOR_API.md](OPERATOR_API.md) /
[TESTING.md](TESTING.md) / [SIGPROC_PLAN.md](SIGPROC_PLAN.md) set, plus the
audience derivatives [PRACTITIONER_OVERVIEW.md](PRACTITIONER_OVERVIEW.md) and
[DEVELOPER_OVERVIEW.md](DEVELOPER_OVERVIEW.md). Written from the seat of a SAR
practitioner deciding whether to build a processing chain on this design.

Nothing here is implemented yet. Judgments are about the *proposal*.

**Audience ranking used below.** First customer is **satellite / spaceborne**
(CPHD → formed SLC → SICD, typically ground or DEM geometry, autofocus usually
off). Secondary goal is to remain attractive to airborne, spotlight, video SAR,
and live front ends without letting those cases rewrite the primary product.

Open contract items map to [TEAM_BLOCKED.md](TEAM_BLOCKED.md): **R1, R2, R5,
R7**. Settled design calls that a reviewer should still pressure-test are
indexed there as **D1–D20**.

**Editorial note — added 2026-08-05 by the contract authors, not by the
assessor.** The assessment is **left exactly as written**, including the
open-item numbering just above, which was current when it was written. Three of
the four R-items it names have since closed — **R7 → D22**, **R5 → D21**,
**R2 → D23** — leaving R1 open and adding R10. Where a closure bears on a
finding, a reply is inserted immediately after it as an indented blockquote
labelled *Response*. **Nothing outside those blockquotes has been altered**, so
the findings, their severity ranking and their wording remain the assessor's.

One distinction is worth carrying into the reading, because a decision is not
the same as a fix. **D22 is a fix**: the boundary announcement now has a
producer, a carrier and a consumer. **D21 is a declination**: there is no
metadata plane and there will not be one, so the mirroring finding is answered
by accepting the hazard with an optional escape hatch rather than by removing
it. **D23** was a narrow question and closed narrowly.

---

## Verdict

For a spaceborne practitioner who already has CPHD (or will), will accept
**time-domain backprojection**, and will ship **map / DEM SLCs without
autofocus by default**, this is a coherent and unusually careful formation
design. Precision at orbital range, explicit frames, scrolling continuous
strip, and a resolved autofocus story for the cases that need it are real
strengths.

Three holes still block trusting it as a production spine on the satellite
path: **nobody can announce end-of-stream (R7)**, **SICD writers still mirror
geometry by hand (R5)**, and **operational strip still cannot be both
seam-free and cross-range tapered** (deferred, not an open R-item, but the
gap the design itself names). Frequency-domain formation remains the main
reason a stripmap throughput shop walks away even if those three are fixed.

> **Response — added 2026-08-05; not part of the original assessment.** Of these
> three, one is closed, one is answered without being closed, and one is
> unchanged. End-of-stream has a producer: a source stamps the boundary as
> message metadata and whoever drives `control` translates it (**D22**), which
> removes the silently-dropped final aperture from the satellite path and adds a
> `stop()` warning so the residual failure is visible. Writer mirroring is
> **accepted rather than fixed** (**D21**) — the response under § 2 is where that
> judgment actually sits. Operational tapered strip is unchanged, and with the
> other two disposed of it is now the only one of the three still standing.
> Frequency-domain formation is likewise unchanged.

Autofocus is no longer the structural contradiction it once was: it is off by
default, and when opted in on a ground product it estimates on a small
conjugate auxiliary image and corrects pulses before the product formation.
That is the right shape for this module’s usual product. Residual hazards on
that path are real but secondary for satellite-first.

---

## What earns trust

**Mode taxonomy that matches how products are named.** Nine supported modes
(spotlight, video SAR, tiled strip, continuous strip, independent images,
wide swath, sub-aperture looks, circular, tomographic), two deferred, one
foreclosed (bistatic). Collapsing almost everything to “independent images
with different geometry scheduling,” plus overlapping and scrolling as the
two special accumulators, is the right software story and a clear product
story.

**One chain, modes as configuration.** Retention / emit live in an optional
`ApertureManagerOp`; formation only accumulates. Stripmap is not a second
pipeline: tiled strip resets per tile, continuous strip scrolls the
accumulator. Video SAR is signed-weight eviction. That composability is what
a library should look like.

**Continuous strip is first-class (D18).** Scrolling along the outer image
axis so each pixel integrates from beam entry to exit — no tile seam, no
re-forming shared pulses — is what operational strip wants on the seam axis.
Fixing the scroll axis to `v` for CUDA-block locality is an honest
implementation constraint, not marketing.

**Autofocus matches the product (D20).** Default path is ground / DEM,
autofocus omitted — correct for good spaceborne ephemeris. Opt-in PGA
estimates only on a conjugate lattice, emits **per-pulse** phase, and
`PhaseCorrectionOp` applies it ahead of product formation. Map geometry stays
the deliverable; conjugacy binds the estimator, not the product. Cost is a
small auxiliary formation plus one product pass on a retained / reformed
aperture, not “PGA after DEM BP” as a silent wrong stage.

**Silent-failure discipline.** Declared `CoordinateFrame`, transmit-instant
phase centres, centred range axis, band-centre phase reference,
`reference_position` as a 3-vector, named-and-rejected unimplemented
interpolators, build-time wiring checks for ports that must be connected —
these are the failure modes that produce *plausible* bad images. Spaceborne
`precision_mode` is treated as physics, not a speed knob.

**Standard formats in wave one.** CPHD read/write and SICD write, with Umbra
(or similar) as the convention leg and provider SICD as geometry truth rather
than pixel gold, matches how satellite validation actually works.

**Honesty about absences.** No fake motion-compensation stage for BP; FMCW
CFAR does not carry over; ScanSAR/TOPS and sub-aperture looks stop short of
mosaic / look combination; bistatic is foreclosed rather than half-supported;
the tapered seam-free strip is deferred with a clear reason (per-pixel
illumination interval has no home in the interface yet).

---

## Deal-breakers on the satellite-primary path

### 1. End-of-stream has no producer (R7)

Satellite work is file-backed. The contract requires a flush on `control` so
the final partial aperture is not dropped; `Operator::stop()` cannot emit.
**No operator declares a `control` output.** An unconnected `control` silently
discards the tail. That is a product bug on the default ingress, not a
footgun. Until `CPHDReaderOp` (or an SDK metadata pattern — see
[SDK_QUESTIONS.md](../SDK_QUESTIONS.md)) can announce collection / file
boundaries, a CPHD → SLC chain is not finished.

> **Response — added 2026-08-05; not part of the original assessment.** Closed as
> **D22**, along the second of the two routes this finding names, and it is a fix
> rather than an acceptance. `CPHDReaderOp` stamps
> `holoscan.radar.stream_boundary` on the batch that ends a collection, and
> whoever drives `control` turns it into a flush; `ApertureManagerOp` reads it
> directly, so the default CPHD → SLC chain needs no extra wiring at all.
> Metadata rather than a `control` output because a port cannot stay aligned with
> the data it describes across a stage that coalesces ticks: non-blocking, the
> flag races a batch still in flight, and blocking, FIFO alignment holds only
> through strictly 1:1 stages — which the manager is not.
>
> One residual a spaceborne reader should note — corrected 2026-08-06, this
> response originally listed two, the other being fragment-level metadata setup.
> That was wrong: metadata is enabled by default and `kUpdate` is set
> per-operator by the one consumer that merges several ports from a producer, so
> the application configures nothing. The real residual is that an application
> owning its own control operator must feed that operator from the source or it
> never hears the boundary. `BackprojectionOp` warns at `stop()` if it is still holding
> an unemitted image, which converts a silent tail loss into a noisy one — the
> best available, since no build-time check can see whether a source is finite.

### 2. Writer metadata is still mirrored configuration (R5)

The deliverable for many spaceborne users **is** SICD (and later SIDD).
Writers restate formation geometry and windows as parameters. That is how you
ship NITF whose XML disagrees with the pixels — the same silent class the
rest of the design works to close. Either a metadata plane belongs in the
contract, or the mirroring story is hardened and accepted as a known hazard
with tests that treat disagreement as failure. Leaving it open while selling
SICD out is unfinished.

> **Response — added 2026-08-05; not part of the original assessment.** Resolved
> as **D21**, by declining the plane — which is the outcome to read carefully.
> There is no metadata plane and there will not be one: the mirrors stay the
> ordinary path, and the hazard is explicitly **accepted** under invariant 5.
>
> This finding offers two acceptable outcomes, and the answer is neither exactly.
> It is the second — a hardened mirroring story — weakened in one place and
> strengthened in another. Weakened: the tests assert the *undetectability* of a
> disagreement as a known limitation rather than as a failure, because the
> operator genuinely cannot detect it. Strengthened: a caller who wants agreement
> by construction can wire `BackprojectionOp.sample_geometry` into
> `SICDWriterOp.sample_geometry` (or `GeocodeOp.input_sample_geometry`), and the
> frame likewise from the reader, at which point the disagreement cannot arise
> for that fact. `range_window` gets the same treatment; `azimuth_window` cannot,
> since SICD's `WgtType` wants a window *name* that no vector of weights carries.
>
> So the disposition is conditional on the caller's wiring rather than on the
> contract. Wire the two geometry edges and the frame edge and the class of bug
> named here is gone from that graph; leave them unwired — which is legal, and
> what a YAML-configured pipeline does by default — and you are exactly where
> this finding found you. That is a weaker guarantee than the rest of the design
> offers, and the contract authors flag it as the place to push back if anywhere.

### 3. Operational strip still chooses seams or sidelobes

Mode 3 (tiled): taper allowed, under-integrated joins. Mode 4 (continuous
scroll): seam-free, **no** cross-range taper through `weight` (~−13 dB). The
product most operational strip processors ship — continuous, unjoined,
azimuth-weighted — is **deferred**, not missing by accident. The design
states this clearly; a strip-primary satellite buyer should treat it as the
most likely “not ready for us” verdict even after R7/R5 close. Closing it is
an interface addition (where along-track aperture length / pointing live),
not a tweak to scroll.

> **Response — added 2026-08-05; not part of the original assessment.** Unchanged
> and still deferred. With §§ 1 and 2 disposed of by D22 and D21, this is the only
> one of the three original deal-breakers still standing, and the assessment's
> reading of it is accepted as written. One clarification D22 settles: the
> trailing margin at the end of a scrolled strip is emitted **under-integrated**
> rather than truncated, so cropping it is the consumer's decision, not a silent
> loss.

---

## Serious friction (not always deal-breakers)

**Backprojection-only formation.** Defensible for spotlight, irregular
tracks, DEM-direct products, and GPU research stacks. A high-volume stripmap
throughput shop still often lives in range-Doppler / CSA / omega-k; that is a
weaker walk-away for tasked spotlight / tile BP workflows that already accept
time-domain formation. Deferral is honest; closing it is a parallel formation
contract, not a switch — see *What an FD stripmap wave would cost the API*
below.

**SICD cannot describe the nicest BP product.** DEM-conformal / warped
lattices are first-class in formation and rejected by `SICDWriterOp` (SICD
grid metadata is uniform planar). Direct map formation without a second
resample cannot be the interchange complex product without SIDD or a custom
container. Painful if the selling point is “form on the DEM.”

**Radiometry and calibration are thin.** `fft_norm` default still open (I5);
no σ⁰ / antenna-pattern / absolute RCS path. Fine for geometric SLC QA; weak
for calibrated detected products.

**Direct-sampled ingress is later-wave.** Many spaceborne front ends digitize
full bandwidth; wave one assumes phase history / CPHD. Fine if CPHD is the
ingest contract; weaker if Level-0-like raw is the ask.
`RangeMatchedFilterOp` is named but not first wave.

**Opt-in autofocus still asks a lot.** You must supply a correct conjugate
estimation geometry; conjugacy is declared via `azimuth_axis`, not checked;
wrong axis → plausible wrong correction. PGA discards linear/constant phase
(image shift), which cuts against DEM geolocation claims — correctly why AF
is off by default. Figure 2 needs retention + `emit_mode = reform` and
same-tick DAG order through aux BP → AF → `PhaseCorrectionOp` → product BP;
if Holoscan does not honour that within a tick, the path is not implementable
as drawn. `PhaseCorrectionOp.sense` is required with no default — right, but
easy to get fatally wrong.

**ScanSAR / TOPS and tomography stop early.** Mode 6: N tiled strips, no
mosaic, no pattern compensation. Mode 9: N formation instances (geometry has
no batch axis — D19 deferred); fine for small N, awkward for dense height
stacks. Circular SAR is supported but explicitly untested on PSF claims.

**Integration cost for “form this file.”** CPHD carries positions, so offline
satellite is lighter than live airborne. Sample geometry, emit/`control`, and
optional manager wiring still demand Holoscan literacy. The docs should lead
with a thin file-backed recipe (manager optional; reader-driven `control`
once R7 exists).

> **Response — added 2026-08-05; not part of the original assessment.** D22 makes
> the recipe thin, though not in the shape assumed here: `CPHDReaderOp` has no
> `control` output by design, so the reader-driven `control` this paragraph waits
> on never arrives. Instead a manager in the graph handles the flush from the
> metadata announcement with no application-owned ports wired at all. The cost is
> nothing — corrected 2026-08-06, this response originally claimed one line of
> fragment setup (metadata enabled, policy `kUpdate`) whose omission loses the
> final aperture. Metadata is on by default and `kUpdate` is per-operator, so the
> recipe carries no setup step at all.

**External Python simulator.** Correct that it writes CPHD and is never wired
in-process; correctness of the module still hangs on an out-of-repo, pinned,
regenerable artifact.

**R2 (control payload).** Narrower than R7/R5: four emit/reset values are
settled; whether messages carry aperture id / sequence / reason affects
whether writers can label products without a metadata plane. Couples to R5.

> **Response — added 2026-08-05; not part of the original assessment.** Closed as
> **D23**, by subtraction: the four values are all the message carries. The
> coupling identified here is exactly what resolved it — with D21 declining the
> plane, `SICDWriterOp` labels its product from the emitted-image counter it
> already keeps (`{index}` in `path`), so the question dissolved rather than being
> answered. Nothing left for a practitioner to weigh.
>
> One item has been *added* since this assessment, and it is not a SAR question.
> D22 needs a metadata key name, so the module now has a key namespace,
> `holoscan.radar.<key>`. Whether that key sits module-wide or under a `sar` tier
> is **R10**, open. It is irrelevant to whether a SAR chain works, and relevant to
> anyone composing this module with another library, since a key prefix is a
> public surface.

---

## What an FD stripmap wave would cost the API

Frequency-domain formation (range-Doppler, chirp scaling, omega-k, PFA) is
deferred on merit: the docs already say it needs a **second kind of image
geometry** and classical correct-to-a-nominal-track motion compensation, and
that inventing those interfaces before a kernel exists would freeze the wrong
spelling. From a practitioner seat, the useful question is how much of *this*
API would have to move to serve a stripmap-throughput buyer.

**It is not a flag on `BackprojectionOp`.** Cartesian `SampleGeometry` /
`sample_points`, `precision_mode`, signed-weight eviction, and
`geometry_change = scroll` are BP semantics. Overloading them for radar-
geometry FFT formation would corrupt both contracts. The clean shape is a
**parallel formation modality** beside BP, reusing the spine.

### What largely stays

| Piece | Why it survives |
| :--- | :--- |
| CPHD ingress → phase history → `SARRangeCompressionOp` | Same front half; classical RDA still wants range profiles (CSA / ω-k differ more inside formation). |
| `ApertureManagerOp` + `control` / block emit | Already called algorithm-neutral; “which pulses, when emit” still applies to stripmap blocks / CPIs. |
| `PhaseCorrectionOp` | Still a per-pulse phase multiply. |
| `AutofocusOp` Figure 3 (in-place) | FD SLCs *are* conjugate / radar-geometry — in-place PGA is the natural path. Figure 2 (aux conjugate + DEM product) is the BP-specific story. |
| `MultilookOp`, `SICDWriterOp`, `CoordinateFrame`, waveform descriptor | SICD is happier with uniform radar-geometry grids than with DEM lattices. |
| Cross-tick accumulation | Still need block emit / flush (and still need R7). |

> **Response — added 2026-08-05; not part of the original assessment.** D22's
> boundary announcement is algorithm-neutral, so the flush this row is waiting on
> carries over to a frequency-domain wave unchanged. The row otherwise stands.

### What has to be added or rewritten

1. **Radar-geometry (the deferred interface).** Today every pixel is a
   Cartesian metre position. FD forms on range × azimuth/Doppler (or polar for
   PFA), then resamples. That needs a new descriptor / geometry language — not
   fake ECEF stuffed into `SampleGeometry`. This is the invariant-2-sized piece
   the contract refuses to guess early.

2. **Classical motion compensation as a stage.** Sense (1) of MoCo — phase /
   data to a **nominal track** — is absent for BP and returns with FD. New
   operator (or fused front of the FD former): trajectory in, corrected
   profiles out, plus nominal-track parameters. App-owned navigation stays;
   the *correction* does not.

3. **A formation operator family.** e.g. `RangeDopplerFormationOp` first, with
   CSA / ω-k / PFA as later siblings or variants. Different knobs (Doppler
   centroid, effective velocity / FM rate, RCMC, block size, azimuth window).
   Manager outputs feed it; it emits `(..., Naz, Nrg)` in radar geometry.

4. **Strip / scroll semantics do not transfer.** Mode 4 continuous strip is
   “translate a Cartesian lattice + ring accumulator” — meaningless for RDA.
   FD stripmap looks like contiguous pulse blocks → radar-geometry strip/tile
   → **mandatory** `GeocodeOp` (or mosaic) for map products. Azimuth taper is
   a Doppler-domain window: ironically, the *tapered seam-free strip* that BP
   deferred is closer to FD’s native strip than to BP scroll. WORKFLOWS
   mode→config mapping needs an FD column; `geometry_change = scroll` stays
   BP-only.

5. **Product-path shift.** Direct DEM formation stops being the selling point
   on the FD path; geocode becomes required for map deliverables.
   Radar-geometry → SICD gets easier; DEM-conformal SLC without resample goes
   away there.

6. **Precision and tests.** `precision_mode` (long-range norm arithmetic) is
   BP-specific. FD needs its own numerical contract and MoCo convention tests.

### Blast radius (rough)

| Bucket | Share of current design | FD stripmap wave |
| :--- | :--- | :--- |
| Ingress + range compress + writers | ~⅓ | Reuse |
| Aperture control (`control` / `weight` / manager) | ~⅙ | Reuse for block stripmap; less need for sliding −1 eviction |
| `BackprojectionOp` + scroll + Cartesian geometry | ~⅓ | Parallel replacement — do not lightly extend |
| Autofocus Figure 2 machinery | small | Optional on FD; Figure 3 becomes primary |
| New: MoCo + radar-geometry + FD former + geocode-first recipes | — | Additive bulk of the wave |

**Minimum credible stripmap-throughput slice:** `MotionCompensationOp`, a
radar-geometry descriptor, `RangeDopplerFormationOp` (P0), manager wired for
**block** apertures (mode-3-like, not BP scroll), `GeocodeOp` documented as
the map path, and an azimuth window on the FD former (weighted strip without
BP’s deferred per-pixel taper story). PFA is another geometry again; CSA /
ω-k are mostly kernel variants once radar-geometry + MoCo exist.

**Bottom line for R1 honesty.** Supporting FD stripmap is “add a formation
modality beside BP,” not “flip a switch.” The expensive API invention is the
**radar-geometry + MoCo** pair already named in *Deferred*; aperture I/O and
CPHD → range → SICD are the reusable half. Do not accept pitch language that
implies RDA-class strip throughput is a small follow-on to wave one.

---

## Breadth: who else is served

| Audience | Fit today | Notes |
| :--- | :--- | :--- |
| Spaceborne spotlight / DEM SLC, AF off, CPHD → SICD | Strong, if R7 + R5 close | Primary path |
| Spaceborne continuous strip, sidelobes OK | Strong after R7 | Mode 4 scroll |
| Spaceborne continuous strip, need azimuth taper | Weak | Deferred tapered seam-free strip |
| Spaceborne tiled strip catalogue | Usable | Accept seams or shorten tiles |
| Spaceborne strip at RDA-class throughput | Poor | Needs FD wave |
| Residual POD / occasional AF on map product | Spec’d (Figure 2) | Pay conjugate geometry + reform; test DAG order |
| Airborne live deramped | Later wave | Evidence-gated RVP / band-eq |
| Video SAR | Spec’d | Manager + weights; taper ⇒ reform; FP drift on long slides |
| Wide swath / tomography deliverables | Partial | Constituent images only / N instances |
| Bistatic | Foreclosed | Do not expect a path |

> **Response — added 2026-08-05; not part of the original assessment.** The first
> two rows' conditions are partly met. D22 closed the flush gap, so "after R7"
> reads as satisfied; the R5 half of row one is satisfied only where the caller
> wires D21's geometry and frame edges, per the response under § 2. One detail
> worth adding to row two rather than changing it: the trailing margin of a
> scrolled strip arrives under-integrated, so cropping it is the consumer's call.
> No other row changes.

---

## What to pressure at contract acceptance (R1)

**Must close or explicitly demote before calling the satellite path done:**

1. **R7** — a producer for end-of-stream / file-boundary flush (port on the
   reader, or an SDK-backed metadata pattern that actually works).
2. **R5** — metadata plane in scope, or a hardened “mirrored params are the
   product contract” stance with tests that fail on disagreement.
3. **Tapered seam-free strip** — either schedule the interface addition or
   market strip as “pick seams or −13 dB,” not as full operational strip
   parity.

> **Response — added 2026-08-05; not part of the original assessment.** Items 1
> and 2 have both been acted on, but not in the same way. Item 1 is closed as
> **D22** — the source announces on the metadata plane, the control owner
> translates, and `BackprojectionOp` warns at `stop()` on an unemitted image. What
> remains of it is narrower: that the application-level metadata requirements
> appear wherever a recipe does, and that the announcement is verified to cross
> fragment boundaries before anyone partitions a graph, which is so far unverified.
>
> Item 2 is answered as **D21** by *declining* the plane, so the pressure this
> item asks for is still worth applying, and **this is the one closure to push
> back on if a reviewer is going to push back on anything**: the guarantee is
> conditional on the caller's wiring rather than on the contract. The question to
> put is whether "you may wire this, and by default you will not" is enough for a
> product whose deliverable is the SICD. Item 3 is untouched.

**Should not block satellite no-AF CPHD → SICD, but should be honest in the
pitch:**

1. Frequency-domain formation as a future wave with its own geometry + MoCo
   contract (see *What an FD stripmap wave would cost the API*) — not a
   parameter on backprojection.
2. Radiometric / SIDD path timing relative to “calibrated product” claims.
3. Figure 2 autofocus: prove same-tick ordering; keep conjugacy and geolocation
   costs in the practitioner docs (already largely there).

**Ratify deliberately (D-list highlights worth a focused pass):** single-owner
reset (D1); scroll vs tile (D18); AF estimate-then-correct-pulses (D20);
optional manager (D4/D17); `reference_position` as a point (D9); capacity
overflow as error (D8). These are the places a satellite reviewer can still
reasonably disagree.

> **Response — added 2026-08-05; not part of the original assessment.** Three more
> have joined that list since: declining the metadata plane (**D21**), announcing
> boundaries on it anyway for one key (**D22**), and the empty control payload
> (**D23**). D21 and D22 sit oddly together on purpose — the plane is refused for
> description and used for announcement, on the argument that a transient signal
> and a persistent fact are different problems. That premise is the thing to test.

---

## Bottom line

The proposal is ready to be judged as a **streaming backprojection formation
box aimed at spaceborne CPHD → map/DEM SLC**, with a credible opt-in autofocus
path and a real continuous-strip accumulator. It is not yet ready to be judged
as a finished production spine: flush announcement and product metadata are
still open, and operational tapered strip is still out of reach by design.
Accept the contract if that scope is the product; do not accept language that
implies RDA-class strip throughput or seam-free weighted strip without naming
the deferrals.

> **Response — added 2026-08-05; not part of the original assessment.** Two of the
> three things held to block a production-spine verdict have moved. The flush
> announcement is solved (**D22**). Product metadata is now a wiring choice a
> caller can get right (**D21**) rather than a gap nobody can close — a weaker
> outcome than a fix, and the one place to spend review attention. Operational
> tapered strip is unchanged. The closing caution about pitch language stands as
> written.
