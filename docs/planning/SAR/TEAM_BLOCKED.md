# SAR Team-Blocked Items

Contract / design questions raised by the **proposed SAR modality** that need a
**team decision**. Individuals must not silently resolve these in code.

This file is the SAR counterpart to the module-wide roll-up,
[../TEAM_BLOCKED.md](../TEAM_BLOCKED.md), and uses the same two classes and
the same resolution procedure. When a decision lands, record it in
[STATUS.md](../STATUS.md) → Settled decisions, then remove the entry here *and*
the matching entry in [OPERATOR_API.md](OPERATOR_API.md) → *Open items*.

The entries below in *Needs team look* are open questions. A second list,
*Decided — focused review requested*, carries calls that already have an answer
and are flagged so review attention reaches them; it uses `D` numbering to keep
the count of items gating acceptance unambiguous.

**Before an operator directory exists**, the "detail lives in `OPEN_ITEMS.md`"
convention has nowhere to point, so detail lives in the planning doc that
raised the question — for the entries below, [OPERATOR_API.md](OPERATOR_API.md)
→ *Open items*, which is also where this modality's implementation-time items
(the non-team-blocked class, I1–I7) are enumerated. When the operator
directories land, migrate each open entry into the matching `OPEN_ITEMS.md` and
repoint the Detail line; an entry that is still open at that point must not be
dropped in the move.

These were tables until 2026-08-06. They are prose sections because the
rationale outgrew what a table cell can hold: cells cannot wrap, so every
addition pushed rows further past markdownlint's 800-character line cap.

---

## Stops development

**None currently.** R1 closed on 2026-08-07 (accepted), which was the entry that
gated SAR operator code, so implementation is under way and nothing open pauses
a checklist.

**R10 is the one survivor**, and it needs an answer sooner than its "needs team
look" class suggests: `D22`'s `stream_boundary` key cannot be written without
choosing the namespace it sits in, so `CPHDReaderOp` and `ApertureManagerOp`
reach it early. It is not reclassified as development-stopping, because the
SAR-tier option resolves it inside SAR — see the entry for why that is the
cheaper default.

---

## Needs team look

### R1 — SAR contract acceptance — **CLOSED 2026-08-07: accepted**

The team accepted the contract as a whole, with implementation to begin against
it. The amendments below are therefore **in force**, not proposed. Two conditions
came with acceptance, and they are the live constraint now that the question
itself is settled. Revising the contract during implementation is expected, but
**every revision is approved by the developer in the chat session before it is
made** — SAR-scoped ones included. A revision reaching **beyond SAR** clears the
same gate and additionally is marked for reviewers and rowed in the module-wide
[TEAM_BLOCKED.md](../TEAM_BLOCKED.md). Both are specified in
[OPERATOR_API.md](OPERATOR_API.md) → *Contract changes since approval*.

Kept here rather than moved to the decided list because what was accepted is the
whole contract, and this is where a reviewer looks for the scope of that.

What acceptance covered — **five** amendments to the locked invariants:

- descriptor set 6→**9**, adding `CoordinateFrame`, `SampleGeometry` and
  `LFMWaveformDescriptor` — purely additive, touching no landed FMCW code,
  since D24 withdrew the rename that would have;
- descriptor-carrying *output* ports 1→**4**, per D21;
- an orthogonal internal-arithmetic strategy on `BackprojectionOp`;
- output shape from declared sample geometry rather than an input tensor;
- a `[live]` override that changes an accumulating operator's geometry forcing
  a reset.

Plus **two** new conventions — cross-tick accumulation, and a one-key metadata
plane (D22) — and one new operator whose job is control rather than signal
processing.

Review input that fed the decision: a practitioner critique ranking these items
against a spaceborne CPHD → SICD chain, in
[PRACTITIONER_CRITIQUE.md](PRACTITIONER_CRITIQUE.md).

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Contract changes

### R10 — Metadata key namespace scope

Raised 2026-08-05 by D22, which needs a key name and so a scheme. Proposed:
`holoscan.radar.<key>` for module-wide keys, `holoscan.radar.sar.<key>` /
`.fmcw.<key>` reserved for modality-specific ones, with D22's `stream_boundary`
at the **module** level — the accumulation convention it serves is module-wide
and a finite FMCW source has the same tail-loss problem.

The question is whether that placement is right, or whether it should sit under
`sar` and be promoted when a second modality needs it. It commits a modality
whose owners have not weighed in, and a key prefix becomes a public composition
surface the moment anything outside this module reads it. Narrow, but not
resolvable in code.

**Sharpened by contract acceptance, 2026-08-07.** The two options are no longer
symmetric. Module-level placement is a **beyond-SAR** change under the rules in
[OPERATOR_API.md](OPERATOR_API.md) → *Contract changes since approval*, since it
commits a namespace FMCW inherits; the `sar` tier is SAR-scoped and needs no one
else's agreement. The standing preference is therefore the `sar` tier by default
— `holoscan.radar.sar.stream_boundary` — with promotion to the module level when
a finite FMCW source actually wants it, which costs one constant. That is a
recommendation, not a resolution: the team may still prefer module-level, and
until it says so the implementation should keep the key behind a single named
constant so either answer stays a one-line change.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Stream boundary announcement →
Key namespace

---

Agents: [AGENTS.md](../../../AGENTS.md) ("Do not silently resolve documented
open items") and [OPERATOR_DEVELOPMENT.md](../OPERATOR_DEVELOPMENT.md)
§ Open items.

---

## Decided — focused review requested

Each already had an answer, so none of them gated R1 — and R1's acceptance on
2026-08-07 **ratified all of them**. They stay enumerated because that is still
what they are for: a reviewer should not have to read nine hundred lines to find
the two dozen places where the call could reasonably have gone the other way,
and revisiting one later is far cheaper with the rejected alternative written
down. Each entry states the decision, the alternative it beat, and where the
reasoning lives.

Revising any one of them is now a **contract change** under
[OPERATOR_API.md](OPERATOR_API.md) → *Contract changes since approval*: proposed
to the developer and approved before the edit, then tagged and logged — and
where it reaches outside SAR, marked for reviewers and rowed one level up.

### D1 — Single-owner reset

Exactly one operator may invalidate accumulated state; the convention forbids
the shape rather than patching instances. Rejected: letting each stateful
operator reset itself on an invalidating change, which corrupted silently —
formation cleared while the manager kept emitting `−1` evictions for
contributions that were already gone.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation

### D2 — Termination is announced, not emitted from `stop()`

An end-of-stream *announcement* driving the flush along the `control` path.
Forced rather than chosen — `Operator::stop()` has no `OutputContext`.
**Amended by D22**, which supplies the announcement's producer and carrier and
so removes the consequence originally recorded here: a file-backed collection
no longer needs a `control` edge wired purely to avoid dropping its final
partial aperture. What survives is that no build-time check can catch a lost
tail, because finiteness is not visible in the graph — hence the `stop()`-time
warning.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation

### D3 — Build-time wiring validation

An operator whose correctness depends on an optional port being wired refuses
to start, naming the connection. Rejected: failing on the first geometry change
hours into a collection, or emitting nothing at all and leaving the caller to
work out why.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Build-time wiring validation

### D4 — Retention and emission split out of formation

A separate manager owns the aperture; `BackprojectionOp` holds an accumulator
and obeys. Rejected: `emit_policy` / `reset_policy` / retention parameters on
the formation operator, which put the bookkeeping where it needs a GPU to test.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp

### D5 — One parameterized manager, not a family of types

Structural choices stay static parameters on one operator, following the
`cfar_variant` precedent. Rejected: separate operator types per aperture
behaviour, which would make invalid parameter regions unrepresentable — a real
benefit, traded for YAML-switchable configuration.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes

### D6 — Admissions and evictions in one batch

One message per tick carrying both, signed by `weight`. Rejected: separate port
groups, which are genuinely zero-copy where the departing group is an
unmodified tensor; measured cost of the choice is one extra pass over
`2 × pulses_per_tick` profiles.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes

### D7 — Applications evict by count, not by aperture length

The `evict` port carries how many pulses retire. Rejected: a `[live]` aperture
length, which leaves the manager owning the rule and the application shadowing
its buffer to predict what it decided.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes

### D8 — Capacity overflow is an error, not a drop

Under application-owned eviction, exceeding `max_aperture_pulses` fails. **In
tension with invariant 5** ("allow reasonable footguns and warn where
relevant"); the argument for erroring is that a silent drop is not a visible
footgun — it surfaces as a mild defocus, not a message.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes

### D9 — `reference_position` is a 3-vector, not a scalar range

A position, because one of the two viable precision strategies cannot be
reconstructed from a distance. Rejected: the scalar reference range, which is
what **both** reference implementations carry and is the smaller port.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp

### D10 — Range bin zero is centred

Default `range_bin_origin` resolves to `−⌊N/2⌋ · spacing` per tick, matching
both reference implementations. Rejected: bin zero at the reference range,
which was our own earlier default and made range-cropped input the natural case
rather than the special one.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp

### D11 — Unimplemented enum values are named and rejected

`interpolation` declares `nearest` and `sinc` but errors on them, so the enum
need not grow later. Rejected: omitting them (a later API change) and silently
falling back to `linear` (a plausible wrong image).

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp

### D12 — One CPHD channel per reader instance

Per-channel pulse and sample counts may differ, so channels do not stack into
one rectangular tensor. Rejected: a leading batch axis over channels, which
works until it meets a heterogeneous collection.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → CPHDReaderOp → Notes

### D15 — A tapered aperture re-forms; an incrementally-slid one is unweighted

Was R6; closed 2026-07-30 by declining to approximate. A fixed taper over a
moving window changes every retained pulse's weight each tick, costing `N`
pulse-formations per tick — the price of re-forming — so incremental
accumulation and an exact taper are mutually pointless. Rejected: a separable
azimuth weighting applied to the emitted image (exact only under R4's
conjugacy assumption), and any per-tick approximation. Consequence:
`emit_mode = reform` is the sanctioned tapered path, and unweighted ~−13 dB
sidelobes are an honest description rather than a placeholder.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp → Notes

### D17 — `ApertureManagerOp` + `BackprojectionOp`

Was R8; decided 2026-07-30. Renamed from `ApertureControllerOp`, which predated
the split and named nothing in radar. "Manager" was chosen over the
house-pattern `ApertureAssemblyOp` because the operator's four jobs —
membership, emission, reset authority, geometry staging — all derive from one
concept, the aperture's lifecycle, and a transformation-style name describes
only the first. Accepted cost: a role name, from which the ports cannot be
guessed. Rejected: `CPIAssemblyOp` (standard vocabulary, but pulse-Doppler
dialect), anything on "Scheduler" (collides with Holoscan), and
`BackprojectionAccumulatorOp` (buries the algorithm). Invariant 7 does not
govern, there being no literature term for this stage.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes

### D16 — Four control values, and the tick has phases

Emission is always evaluated *after* the tick's pulses; the reset's phase
(before accumulation, or after the emit) is what distinguishes the values,
which makes the vocabulary a 2×2 and adds *reset-then-emit* as the completing
cell. A geometry change lands in its reset's phase, which fixes a latent bug:
the previous "control-plane before pulses" rule would have resized the
accumulator out from under a closing aperture's image. Rejected: chunking a
re-form across ticks using only the three original values, which leaves
formation observably half-re-formed.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation

### D14 — `BackprojectionOp` publishes no aperture state

Was R3; closed 2026-07-30 once the manager split and the `evict` port left
nothing needing it — a manager decides from externally observable state, and an
application owning eviction knows its occupancy by construction. Rejected: an
output port carrying pulses accumulated and angle subtended. Reopens only for a
rule needing state internal to formation, such as partial-image sharpness.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation

### D18 — A strip scrolls; it is not a sequence of tiles

Was R9; closed 2026-08-03. `BackprojectionOp.geometry_change = scroll`
translates the accumulator with the lattice, so each pixel accumulates
continuously from entering the beam to leaving it and no tile boundary exists
to degrade. Rejected: replaying a retained tail across the step, having the
reader re-emit overlapping pulse ranges, and accepting the seams — the first
two pay to re-present pulses a previous aperture already formed, and the third
is the defect.

Two consequences ride with it and are the review surface: the scrolled axis is
**fixed to `v`** so the ring's wrap can only ever fall between two launches and
never inside a CUDA block, and the mode **cannot taper through `weight`**,
because a strip taper is a function of *(pulse, pixel)* and no per-pulse vector
carries it — **not** D15's argument, which is about final membership being
unknown until an aperture ends; a strip pixel's illumination interval is fixed
by geometry, so the weighting is exactly computable as pulses arrive, and the
tapered seam-free strip is deferred on where the along-track aperture length
belongs rather than on feasibility.

Both geometry modes scroll; in points mode a translation is expressed as the
entering rows alone, `(k, num_u, 3)`, so the rows that stay are unchanged by
construction. A grid-mode diff was briefly required on the grounds that it
*proves* a change is a translation, which conflated protecting the
accumulator's invariant with verifying the advance against the platform's
motion — the latter is uncheckable in either mode and belongs to the
application. The never-completing trailing margin at end of strip was R7's
question rather than a separate one, and takes D22's answer: the strip is
emitted including its under-integrated trailing rows, and cropping them is the
consumer's call.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp → Scrolling
accumulation

### D19 — Sample geometry carries no batch axis

Deferred 2026-08-04. `sample_points` is `(num_v, num_u, 3)`, shared across the
data's leading axes, so a batch is a batch of collections rather than of
scenes. Consequence: a tomographic volume (mode 9) is `N` operator instances
over one pulse stream, one per height plane — even though the planes share
pulses, weights and aperture boundaries and differ only in geometry, which
makes one operator forming all `N` per tick the shape to want.

Deferred rather than rejected: letting `sample_points` carry a leading axis the
data does not, so the output batch is the outer product of the two, at
identical compute and memory. Additive, and not needed until tomography is in
scope; grid mode has no equivalent spelling, `SampleGeometry` being one atomic
descriptor. Wide swath (mode 6) is unaffected — its sub-swaths close apertures
at different pulses, which one `control` port cannot express, so separate
instances are structural there.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp → Notes

### D20 — Autofocus estimates on a conjugate grid and corrects the pulses

Was R4; closed 2026-08-04. `phase_correction` is **per-pulse**;
`PhaseCorrectionOp` applies it ahead of product formation so PGA conjugacy
binds the aux estimate image, never the DEM product. Recommended: small
conjugate aux + one product formation (few % cost, not double). Unlocked by:
error is per-pulse; azimuth was only where the estimator sees it. Rejected:
restrict formation to conjugate grids (a); geometry-agnostic AF as only
wave-one answer (c). Costs: app builds conjugate estimation geometry;
conjugacy declared via `azimuth_axis`, not checked; AF discards
linear/constant phase. Figure 2 needs `emit_mode = reform` + retention.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → AutofocusOp → Notes

### D13 — No trajectory operator, and no scene-geometry operator

Both of `BackprojectionOp`'s non-pulse inputs are the application's to produce.
Rejected: library operators for either, which could only guess at
deployment-specific conventions while looking authoritative.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Two deliberate non-operators

### D21 — No descriptive metadata plane; three optional descriptor edges instead

Was R5; closed 2026-08-05 by **declining** it. A per-message plane carrying the
facts the writers mirror is not in scope: it is a second, invisible transport
added for the benefit of the chain's optional taps, and the disagreement it
prevents is one visible in the application's own configuration. So the mirrors
stay the ordinary path and the hazard is **accepted** under invariant 5 ("allow
reasonable footguns"), not designed out.

Offered instead, using machinery already in the contract:
`CPHDReaderOp.coordinate_frame`, `SARRangeCompressionOp.range_window` and
`BackprojectionOp.sample_geometry` become optional descriptor outputs, wirable
into the consumer's same-named `[live]` override, where the port wins over the
parameter and a disagreement warns once. The corresponding mirrors on
`GeocodeOp` and the writers are retagged `[live]`.

Sorted by consequence rather than symmetry: a wrong grid or frame misplaces the
product while the file stays well-formed; `waveform` earns no port (collection
config, contradicts no processing decision) and `azimuth_window` cannot have
one (no upstream stage holds a `WindowSpec` where the app drives `weight`, and
SICD's `WgtType` wants the window's *name*).

Two limits: it is opt-in, so a hand-mirroring app is where it was; and
`BackprojectionOp.coordinate_frame` stays Static, so its agreement with the
reader is still unchecked. Rejected: the thick plane (every descriptive fact
ambient), and the thin plane (transient signals only) as a *descriptive*
mechanism — the latter survives at one key, as D22.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Standard-Format I/O

### D22 — A source announces stream boundaries as message metadata

Was R7; closed 2026-08-05. A finite source stamps
`holoscan.radar.stream_boundary` — `{end_of_stream, collection_boundary}` — on
the batch that ends a collection, and whoever drives `control` translates it
into the control message the convention already defines. Two kinds because the
response differs: a collection boundary's reset must be paired with new
geometry on one tick. The **control path stays the mechanism**; this only
settles how the news arrives.

Rejected: a `control` output on `CPHDReaderOp`, which couples a source to
formation-control vocabulary and, worse, cannot stay aligned with the data —
non-blocking, the flag races ahead of a batch still in flight; blocking,
alignment holds only while every intervening stage is 1:1, and
`ApertureManagerOp` is not. Metadata rides *on* the message, so it survives any
number of hops and any coalescing stage.

Consequences. `ApertureManagerOp` gains algorithm step 7 and reads the key
itself, so the rows that once needed a `control` edge for end-of-stream no
longer do. An operator hears the announcement only if fed transitively by the
source, which is the one requirement falling on an app-owned control producer.
And `BackprojectionOp` warns at `stop()` if it still holds unemitted pulses,
which is what makes a missed boundary visible.

The plane imposes **nothing on the application** — corrected 2026-08-06. This
entry previously claimed `is_metadata_enabled(true)` and a fragment-wide
`metadata_policy = kUpdate` were conditions of composing with this module, and
both were wrong: metadata is on by default, and `Operator::metadata_policy`
takes precedence over the fragment default, so the one consumer merging several
ports from a single producer sets `kUpdate` on itself. Still mechanically
forced there, since `kRaise` fires on presence rather than disagreement, but
scoped to one operator.

The key's namespace scope is **R10**.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → Stream boundary announcement

### D23 — The control message carries the four values and nothing else

Was R2; closed 2026-08-05. No aperture identifier, no sequence number, no
reason code. R2's stake was whether a writer could label its product without a
metadata plane; with D21 declining the plane, `SICDWriterOp` labels from the
emitted-image counter it already keeps. Formation's behaviour is fully
determined by the four values (D16); the geometry a reset admits is paired with
it structurally rather than named in the payload; and the *reason* an aperture
ends is needed by whoever translates a boundary — where D22 puts it — not by
the operator obeying it. Dropped as hypothetical rather than designed for:
attributing images to their source file in a multi-file collection.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → How the application supplies
the emit decision

### D24 — A new `LFMWaveformDescriptor`, and no rename of the FMCW one

Decided 2026-08-06, withdrawing a proposal rather than closing an R-item. SAR
gets a **new** descriptor carrying the same fields; `FMCWWaveformDescriptor` is
untouched, so the descriptor set goes 6→**9** instead of 6→8.

Rejected: the rename, which was the cheaper answer on type count and the far
more expensive one on blast radius — `FMCWWaveformDescriptor` is exported from
`holoscan.radar`'s `__all__` and named in 57 non-doc files, making it a
**breaking change to the shipped FMCW public API proposed from within a SAR
contract**, on a modality whose owners are not the ones proposing it. Also
rejected: reusing the FMCW type under its FMCW name from SAR (free, but
misleading at every SAR call site).

The accepted cost is a duplicated field set that can drift silently, since a
field added to one type will not fail to compile in the other. Two things keep
the intended merge cheap and should be honoured when the type lands: the new
type **mirrors FMCW's field names exactly** rather than adopting SAR
vocabulary, and it already carries the name a merged type should have — so
merging later is one sweep (migrate FMCW's call sites, delete the old type),
not two. This makes the SAR contract reach nothing outside SAR except **R10**.

**Detail:** [OPERATOR_API.md](OPERATOR_API.md) → LFMWaveformDescriptor
