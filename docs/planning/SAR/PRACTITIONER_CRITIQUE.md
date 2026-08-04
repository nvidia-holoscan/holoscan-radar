# Holoscan Radar Module — SAR Practitioner Critique

## Purpose

A critique of the proposed SAR modality contract
([OPERATOR_API.md](OPERATOR_API.md), [WORKFLOWS.md](WORKFLOWS.md),
[TESTING.md](TESTING.md), [SIGPROC_PLAN.md](SIGPROC_PLAN.md)) from the
perspective of a SAR practitioner evaluating whether to implement a processing
chain on this software. Written after reading the locked FMCW planning docs one
level up and the SAR proposal under review.

This is review input for contract acceptance (R1), not a substitute for
[TEAM_BLOCKED.md](TEAM_BLOCKED.md). Open items named here map to that index.

**The text below is left as written.** One item it raises has since been
answered: **R9**, the stripmap tile overlap of § 2, is closed as **D18** — a
strip now scrolls rather than being tiled, so the seams the critique objects to
do not arise. Every other item named here is still open, and the critique's own
severity ranking still stands for them.

**Audience assumed here.** First target is **satellite / spaceborne**; the goal
is still to appeal to as many practitioners as is feasible (airborne,
spotlight, video SAR, live front ends) without letting secondary cases dictate
the primary product story. Severity below is ranked for that split: what blocks
a spaceborne CPHD → SLC / SICD chain first, then what limits breadth of appeal.

**Verdict.** For a spaceborne practitioner who already has CPHD and will accept
time-domain backprojection, this is a credible formation box — especially on
precision, frames, and standard-format I/O. Several contract holes (end-of-
stream, stripmap seams, writer metadata) still block trusting it as a
production spine. Autofocus/grid conjugacy (R4) is a real design wound, but it
is a *breadth* problem more than a satellite-primary one: good ephemeris often
means skipping autofocus entirely. Frequency-domain formation remains the
largest “why not our existing stack?” objection for operational stripmap
throughput.

---

## What would make a practitioner lean in

The design clearly knows the failure modes that produce *plausible* bad images:

- Explicit `CoordinateFrame` (including ECEF), transmit-instant phase-centre
  positions, a centred range axis, phase from band centre, and
  `reference_position` as a point (not a scalar) are the right kinds of pins —
  ECEF and long-range arithmetic matter first for satellite.
- `precision_mode` that treats spaceborne FP32 collapse as a first-class
  concern, not folklore, is exactly what a satellite buyer looks for.
- Splitting aperture bookkeeping (`ApertureManagerOp`) from accumulation
  (`BackprojectionOp`) is sound: membership and emit bugs belong on CPU
  assertions, not “looks a bit soft.”
- Forming onto DEM-derived lattices so geocode is optional is a real
  backprojection advantage for map-ready products.
- CPHD in / SICD out, with honesty that provider SICD is geometry-truth rather
  than pixel-gold (and Umbra collections as the convention leg), matches how
  spaceborne validation actually works.

Wave-one scope (the SLC formation box plus writers) is coherent if the target
is “GPU streaming backprojection from CPHD,” including satellite spotlight and
tile-wise stripmap.

Other calls a practitioner notices as written by people who have been burned
before: naming motion compensation’s three meanings and *not* inventing a fake
MoCo stage for backprojection; refusing to reuse FMCW CFAR on SAR clutter;
testing impulse response in the physical domain; measuring floating-point drift
on sliding windows instead of claiming exactness; rejecting unimplemented
`nearest` / `sinc` instead of silent fallback.

---

## Deal-breakers for the satellite-primary path

These block a file-backed spaceborne CPHD → formed SLC → SICD chain even when
autofocus is omitted.

### 1. Nothing can announce end-of-stream (R7)

Satellite work is overwhelmingly **file-backed**. Termination flush is
required; no operator emits `control`. An unconnected `control` silently drops
the final partial aperture. That is not a footgun — it is a product bug on the
default ingress. This should be closed before acceptance, not listed as open.

### 2. Stripmap tile overlap (R9)

Operational satellite stripmap is the volume product. Tiles are disjoint in
pixels but need pulse sets that overlap by roughly one synthetic aperture.
As written, reset plus once-emitted pulses yield soft seams at tile edges.
Accepting that as a mode property is fine only if documented as a known
limitation and sold as spotlight-first / seam-tolerant stripmap; calling
stripmap supported without a replay or overlap story oversells the primary
collection mode.

### 3. Metadata plane deferred (R5)

Spaceborne deliverables are often **SICD / SIDD**, not a Holoviz window.
Writers mirror formation geometry and windows by hand. That is how you ship
products whose XML disagrees with the pixels — the exact class of silent
failure the rest of the contract works hard to avoid. For a satellite
pipeline whose handoff *is* the NITF, this is first-class product risk.

---

## Breadth: autofocus vs the geometry you will actually form on (R4)

Not ranked above as a satellite-primary deal-breaker: the contract already
says spaceborne collections with good ephemeris generally skip autofocus, and
that matches the first audience. R4 still matters if the module should appeal
to airborne users, to satellite cases with ephemeris residual (maneuvers,
poor POD, atmospheric path delay treated as phase), or to anyone who wants
“append PGA after ground-plane BP” as a documented chain.

This is the sharpest product–algorithm mismatch in the proposal, and it is
worse here than in a classical range-Doppler chain because backprojection is
what makes the mismatch *the common case* whenever autofocus *is* required.

**What PGA needs.** Phase gradient autofocus estimates phase error in the
domain conjugate to the image’s azimuth axis: select strong scatterers, shift
them to azimuth centre, window, FFT along azimuth, estimate the phase
gradient, integrate, apply. That transform relationship only holds when the
image’s azimuth axis *is* the Fourier conjugate of slow-time — a polar or
range-Doppler / slant-plane grid sampled so that an azimuth bin corresponds to
a pulse (or a uniform sample of the aperture). On that geometry PGA is the
right default: cheap, high-order, well understood, and what most autofocus-
heavy (often airborne) pipelines ship.

**What backprojection usually produces.** The selling point of this module’s
formation operator is that sample geometry is arbitrary: a ground-plane
uniform lattice, a tilted facet, a sheared grid, or a DEM-conformal
`(num_v, num_u, 3)` point set. Those are exactly the geometries a practitioner
wants for orthorectified or map-ready SLCs without a second geocode
interpolation — including satellite map products. They are also exactly the
geometries for which “FFT along `azimuth_axis`” is *not* conjugacy to
slow-time. The kernel still produces a focused-looking image when navigation
is good; the autofocus *assumption* does not travel with it.

**Satellite-primary reading.** The happy path is:

```text
CPHD → range compress → aperture manage → backproject(ground or DEM grid) → SICD
```

with `AutofocusOp` omitted. That path does **not** require R4. Precision
modes, ECEF, and writer metadata do. Selling “optional autofocus” without
resolving R4 is honest for satellite-first; claiming a universal airborne-
capable AF slot without resolving it is not.

**Breadth reading — why it still cannot stay fuzzy.** Whenever autofocus *is*
in the graph, wave one’s only algorithm is PGA, and formation’s distinctive
value is non-conjugate geometry. These cannot all be true of one default AF
chain:

```text
… → backproject(ground grid) → PGA → SICD
```

Either formation is restricted to conjugate grids (giving up the DEM / map
lattice story AF users often still want), or PGA is not a legal downstream
stage on the geometries people configure, or autofocus is a different
algorithm / loop than the one contracted. Leaving that as R4 means a
secondary-audience practitioner cannot draw the graph they intend — and a
satellite user who *does* need residual-phase correction inherits the same
hole.

**Failure mode if someone wires it anyway.** PGA on a ground grid does not
throw. It produces a phase vector and a “corrected” image that can look
mildly sharper or mildly worse depending on scene content, residual motion,
and how close the ground axes happen to be to a polar sampling. That is the
same silent-plausible class the rest of the contract works hard to close for
frames, range-axis centring, and band-edge phase references — and here the
operator that exists to *fix* residual phase is the one that can invent it.
`min_scatterers` and `max_correction` do not catch conjugacy failure; they
catch empty scenes and runaway estimates. [TESTING.md](TESTING.md) correctly
requires that whichever way R4 resolves, a non-conjugate geometry is either
rejected or shown to work — until that lands, the test suite cannot police
the hazard.

**The three options, from a practitioner’s seat:**

| Option | What it means in a chain | What you give up | When it is acceptable |
| :--- | :--- | :--- | :--- |
| **(a) Restrict to conjugate grids** | Form on polar / range-Doppler-like lattices; reject ground / DEM lattices at `AutofocusOp` (or at formation when AF is in the graph) | Direct map / DEM formation on the AF path — the BP advantage that makes `GeocodeOp` optional; SICD-friendly uniform planar grids survive, warped ones were already rejected by the writer | Satellite-first with AF rare: document “PGA requires conjugate grid; skip AF or form twice.” Airborne QA / bring-up on polar grids. Two-pass (focus on conjugate grid, then resample) reintroduces the interpolation BP was meant to avoid |
| **(b) Estimate then re-form** | `AutofocusOp` emits `phase_correction` (and needs a defined azimuth↔pulse map); correction is applied on the pulse / phase-history side and `BackprojectionOp` runs again | A single downstream stage; latency and FLOPs roughly double per focused product; the control / aperture story must support a feedback tick | Physically the right model for residual phase that is a function of slow-time; best quality when you can afford the second pass — plausible for satellite spotlight scenes where AF is occasional |
| **(c) Backprojection-domain / geometry-agnostic AF** | e.g. minimum-entropy or sharpness optimization over the actual sample geometry, possibly with re-formation inside the loop | PGA’s speed and maturity in wave one; `algorithm = pga` is no longer the whole story; iterative cost and weaker guarantees on distributed clutter | The escape hatch already named in [WORKFLOWS.md](WORKFLOWS.md); best match if broad appeal means “append AF after any lattice” |

**Satellite-first recommendation bias.** Prefer documenting the no-AF spaceborne
default as first-class, resolve R4 as **(a) with a hard reject** for wave-one
PGA (cheap, matches `azimuth_axis`, avoids silent wrong corrections), and
treat (b) or (c) as the breadth follow-on when airborne / residual-POD demand
shows up. Shipping PGA that quietly runs on ground grids is worse than
shipping no autofocus on those grids.

**Interaction with other open items.** R5’s writer metadata wants a recorded
phase correction when AF ran; that correction’s *domain* (image azimuth vs
pulse index) depends on R4. R2’s control payload and aperture replay matter if
(b) needs a second formation pass. SICD’s uniform-grid limit already rejects
DEM-conformal products — so (a) plus SICD-describable grids is a coherent
narrow AF story alongside a wider no-AF map-lattice story.

**Bottom line.** For satellite-primary, R4 does not block the default product;
it blocks advertising a general autofocus stage. Until it is decided,
specified products are “unautofocused image on an arbitrary lattice”
(spaceborne default) and “PGA on a conjugate grid” — not “airborne- or
residual-phase-ready SLC on a ground grid.” Resolve it for breadth and for
honest optional-AF docs; do not hold satellite CPHD → SICD behind it the way
R7 / R9 / R5 should.

---

## Friction that would slow or divert adoption

Ranked with satellite-first in mind; airborne / live / video-SAR notes called
out as breadth.

**Formation algorithm monopoly (high for satellite stripmap).** Wave one is
backprojection only. Defensible for spotlight, irregular tracks, and
GPU-friendly research formation. Operational spaceborne stripmap usually
lives in range-Doppler, chirp scaling, or omega-k for throughput. Deferral is
honest; “SAR module” still reads as “backprojection module” until
frequency-domain formation is on a roadmap with a concrete contract. This is
the main “why not our existing stack?” objection from a satellite stripmap
shop — larger than R4 for that audience.

**SICD cannot carry the nicest backprojection product.** DEM-conformal or
warped lattices are first-class in formation and rejected by `SICDWriterOp`.
The path that avoids a second resample cannot be the interchange product
without SIDD or a custom container. Painful for satellite map-ready ambitions
that still need a standard complex deliverable.

**Radiometry / calibration is thin.** The `fft_norm` default is still open
(I5); there is no clear σ⁰ / absolute RCS path. Spaceborne users often care
about calibrated detected products and SIDD; “pretty SLC / geometry check” is
not enough for that segment.

**Integration surface for “form this file.”** CPHD already carries positions,
so the offline satellite path is lighter than live airborne. Still, sample
geometry, emit policy, and manager wiring are a lot of Holoscan literacy for
a batch spotlight job. A thin “one image per file” recipe should be the
documented satellite default (manager optional; reader-driven `control` once
R7 exists).

**Direct-sampled ingress deferred.** Spaceborne front ends often digitize full
bandwidth rather than deramp; `RangeMatchedFilterOp` is contracted but later-
wave. Wave one assumes phase history / CPHD — fine if Umbra-like CPHD is the
ingest, weaker if Level-0-like raw is the ask. (Breadth: live deramped
airborne is also deferred and evidence-gated.)

**External Python simulator as the only signal model.** Pinning and
bit-identical regeneration are the right discipline, but correctness hangs on
an out-of-repo artifact. Satellite CI will want that pinned and owned.

**Cognitive load vs FMCW.** Accumulating operators and a control-plane stage
earn their keep for streaming video SAR (breadth). For batch satellite
spotlight they may feel like paying for a capability you do not use — unless
the docs lead with the simple file-backed recipe.

**Video SAR taper trade (breadth).** Incremental sliding ⇒ unweighted
≈ −13 dB; taper ⇒ full re-form. Honest; secondary for satellite-primary.

---

## Adoption by use case

| If the job is… | Likelihood of adopting as-is |
| :--- | :--- |
| Spaceborne spotlight / tile BP from CPHD → SICD, AF omitted | High interest — close R7 and R5 first; clarify stripmap seams (R9) |
| Operational satellite stripmap at RDA-class throughput | Poor fit until a frequency-domain wave; BP may still serve specialty scenes |
| Operational stripmap with clean tile seams + SICD | Wait — R9 + R5 (BP path) or FD wave |
| Spaceborne with residual phase / poor POD, ground-grid AF | Wait — R4 (breadth hole on the primary geometry) |
| Airborne live deramped + PGA on ground grids | Wait — ingress + R4 |
| Streaming video SAR | Interested in the manager/weight design; taper trade and R7 less central |
| “Drop in operators, minimal app code” | Document a satellite file-backed default; expect more app ownership than FMCW demos for anything beyond that |

---

## What to close before accepting the contract

**Satellite-primary (block acceptance of the default path):**

1. **R7** — someone must emit `control` at collection / file boundaries.
2. **R5** — metadata plane in scope, or an explicit hardened mirroring story for
   SICD/SIDD writers.
3. **R9** — solve stripmap pulse overlap, or demote stripmap seams to an
   explicit known limitation and lead with spotlight / seam-tolerant tiles.

**Breadth / honest optional autofocus (do not advertise AF-on-any-lattice
until done):**

1. **R4** — prefer wave-one **(a) hard reject** on non-conjugate grids for PGA,
   document no-AF spaceborne + conjugate-grid PGA as the two supported AF
   shapes, and schedule (b) or (c) when airborne / residual-POD demand
   requires AF on ground or DEM lattices.

Those primary three are the difference between a careful research formation
box and something a satellite practitioner would trust as a CPHD → SICD spine.
R4 is what keeps the “appeal to as many as feasible” claim from overselling
autofocus.
