# SAR Team-Blocked Items

Contract / design questions raised by the **proposed SAR modality** that need a
**team decision**. Individuals must not silently resolve these in code.

This file is the SAR counterpart to the module-wide roll-up,
[../TEAM_BLOCKED.md](../TEAM_BLOCKED.md), and uses the same two classes and
the same resolution procedure. When a decision lands, record it in
[STATUS.md](../STATUS.md) → Settled decisions, then remove the row here *and*
the matching entry in [OPERATOR_API.md](OPERATOR_API.md) → *Open items*.

The rows below in *Needs team look* are open questions. A second list, *Decided
— focused review requested*, carries calls that already have an answer and are
flagged so review attention reaches them; it uses `D` numbering to keep the
count of items gating acceptance unambiguous.

**Before an operator directory exists**, the "detail lives in `OPEN_ITEMS.md`"
convention has nowhere to point, so detail lives in the planning doc that
raised the question — for the rows below, [OPERATOR_API.md](OPERATOR_API.md)
→ *Open items*, which is also where this modality's implementation-time items
(the non-team-blocked class, I1–I7) are enumerated. When the operator
directories land, migrate each open row into the matching `OPEN_ITEMS.md` and
repoint the Detail column; a row that is still open at that point must not be
dropped in the move.

---

## Stops development

**None currently.** No SAR operator code exists, so nothing here can pause a
checklist or block a dependent operator from starting.

That classification carries one qualification that the two-class scheme does
not express: every row below **must be answered before the SAR contract is
accepted** (R1 *is* that acceptance). They do not stop today's work only
because there is no SAR work in flight to stop.

---

## Needs team look

| Item | Question | Detail |
| --- | --- | --- |
| **R1 — SAR contract acceptance** | Accept, revise, or reject the proposed SAR modality contract as a whole, including its **four** amendments to the locked invariants (descriptor set 6→**8**, adding `CoordinateFrame` and `SampleGeometry`, plus an `FMCWWaveformDescriptor` → `LFMWaveformDescriptor` rename; an orthogonal internal-arithmetic strategy on `BackprojectionOp`; output shape from declared sample geometry rather than an input tensor; and a `[live]` override that changes an accumulating operator's geometry forcing a reset), the new cross-tick accumulation convention, and one new operator whose job is control rather than signal processing | [OPERATOR_API.md](OPERATOR_API.md) → Contract changes |
| **R2 — Control message payload** | Narrowed 2026-07-30. The *mechanism* is settled: `ApertureManagerOp` owns the decision and drives a non-blocking `control` port into `BackprojectionOp`, with the closed-enum and `std::function` alternatives recorded as rejected. What remains is what the message carries beyond the four settled values (emit-and-continue, emit-and-reset, reset-without-emit, reset-then-emit; see D16) — an aperture identifier, a sequence number, a reason code — which decides whether a writer can label its product without R5 | [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation |
| **R7 — Who announces a collection boundary** | Raised 2026-07-30, split out of R2. The contract requires an end-of-stream or file-boundary announcement on the `control` path — the termination flush, the manager's flush behaviour, and D2 all depend on it — but **no operator declares a `control` output**, so nothing can produce one. `CPHDReaderOp` is the only stage that knows. Whether it gains a `control` output, whether the boundary is surfaced some other way, and whether a source operator emitting formation-control vocabulary is the right coupling | [OPERATOR_API.md](OPERATOR_API.md) → CPHDReaderOp → Notes |
| **R9 — Stripmap tile overlap** | Raised 2026-07-30. Stripmap tiles are disjoint in pixels but overlap in *pulses* by one synthetic aperture length, because a tile's edge pixels are illuminated from beyond the tile. `CPHDReaderOp` emits each pulse once, and a reset drops the manager's retention, so a tile boundary discards exactly the pulses the next tile's leading edge needs and those pixels form at degraded azimuth resolution. Whether the manager replays a retained tail across a geometry step (`reform`'s mechanism, applied between geometries rather than within one), the reader gains a way to re-emit overlapping ranges, or seam degradation is accepted and documented as a property of the mode | [OPERATOR_API.md](OPERATOR_API.md) → Open items |
| **R4 — Autofocus grid conjugacy** | PGA assumes the image azimuth axis is the Fourier conjugate of slow-time, which does not hold for an arbitrary backprojection sample geometry. Restrict `AutofocusOp` to conjugate grids, make it a re-form feedback loop through `BackprojectionOp`, or implement a backprojection-domain autofocus | [OPERATOR_API.md](OPERATOR_API.md) → AutofocusOp |
| **R5 — Metadata plane** | Whether a per-message metadata plane is in scope for the SAR lockdown. Forced by the standard-format writers, since metadata is *transformed* by each stage rather than merely carried. Candidate mechanism is Holoscan's metadata dictionary, used nowhere in this repo. Cross-cutting: it belongs in the contract or is explicitly deferred | [OPERATOR_API.md](OPERATOR_API.md) → Standard-Format I/O |

---

Agents: [AGENTS.md](../../../AGENTS.md) ("Do not silently resolve documented
open items") and [OPERATOR_DEVELOPMENT.md](../OPERATOR_DEVELOPMENT.md)
§ Open items.

---

## Decided — focused review requested

Same **Needs team look** class as the rows above, listed separately because
they differ in disposition rather than in kind: each already has an answer, so
none of them gates R1. Accepting the contract ratifies them by default, which
is exactly why they are enumerated — a reviewer should not have to read seven
hundred lines to find the dozen places where the call could reasonably have
gone the other way. Each row states the decision, the alternative that was
rejected, and where the reasoning lives. Revising any one of them is a contract
change, not a code change.

| Item | Decision, and the alternative it beat | Detail |
| --- | --- | --- |
| **D1 — Single-owner reset** | Exactly one operator may invalidate accumulated state; the convention forbids the shape rather than patching instances. Rejected: letting each stateful operator reset itself on an invalidating change, which corrupted silently — formation cleared while the manager kept emitting `−1` evictions for contributions that were already gone | [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation |
| **D2 — Termination is announced on `control`, not `stop()`** | An end-of-stream *announcement* on the `control` path. Forced rather than chosen — `Operator::stop()` has no `OutputContext` — but the consequence is a judgment call: a finite source whose reader is not wired to `control` silently drops its final partial aperture, and no build-time check can catch it, because finiteness is not visible in the graph | [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation |
| **D3 — Build-time wiring validation** | An operator whose correctness depends on an optional port being wired refuses to start, naming the connection. Rejected: failing on the first geometry change hours into a collection, or emitting nothing at all and leaving the caller to work out why | [OPERATOR_API.md](OPERATOR_API.md) → Build-time wiring validation |
| **D4 — Retention and emission split out of formation** | A separate manager owns the aperture; `BackprojectionOp` holds an accumulator and obeys. Rejected: `emit_policy` / `reset_policy` / retention parameters on the formation operator, which put the bookkeeping where it needs a GPU to test | [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp |
| **D5 — One parameterized manager, not a family of types** | Structural choices stay static parameters on one operator, following the `cfar_variant` precedent. Rejected: separate operator types per aperture behaviour, which would make invalid parameter regions unrepresentable — a real benefit, traded for YAML-switchable configuration | [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes |
| **D6 — Admissions and evictions in one batch** | One message per tick carrying both, signed by `weight`. Rejected: separate port groups, which are genuinely zero-copy where the departing group is an unmodified tensor; measured cost of the choice is one extra pass over `2 × pulses_per_tick` profiles | [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes |
| **D7 — Applications evict by count, not by aperture length** | The `evict` port carries how many pulses retire. Rejected: a `[live]` aperture length, which leaves the manager owning the rule and the application shadowing its buffer to predict what it decided | [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes |
| **D8 — Capacity overflow is an error, not a drop** | Under application-owned eviction, exceeding `max_aperture_pulses` fails. **In tension with invariant 5** ("allow reasonable footguns and warn where relevant"); the argument for erroring is that a silent drop is not a visible footgun — it surfaces as a mild defocus, not a message | [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes |
| **D9 — `reference_position` is a 3-vector, not a scalar range** | A position, because one of the two viable precision strategies cannot be reconstructed from a distance. Rejected: the scalar reference range, which is what **both** reference implementations carry and is the smaller port | [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp |
| **D10 — Range bin zero is centred** | Default `range_bin_origin` resolves to `−⌊N/2⌋ · spacing` per tick, matching both reference implementations. Rejected: bin zero at the reference range, which was our own earlier default and made range-cropped input the natural case rather than the special one | [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp |
| **D11 — Unimplemented enum values are named and rejected** | `interpolation` declares `nearest` and `sinc` but errors on them, so the enum need not grow later. Rejected: omitting them (a later API change) and silently falling back to `linear` (a plausible wrong image) | [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp |
| **D12 — One CPHD channel per reader instance** | Per-channel pulse and sample counts may differ, so channels do not stack into one rectangular tensor. Rejected: a leading batch axis over channels, which works until it meets a heterogeneous collection | [OPERATOR_API.md](OPERATOR_API.md) → CPHDReaderOp → Notes |
| **D15 — A tapered aperture re-forms; an incrementally-slid one is unweighted** | Was R6; closed 2026-07-30 by declining to approximate. A fixed taper over a moving window changes every retained pulse's weight each tick, costing `N` pulse-formations per tick — the price of re-forming — so incremental accumulation and an exact taper are mutually pointless. Rejected: a separable azimuth weighting applied to the emitted image (exact only under R4's conjugacy assumption), and any per-tick approximation. Consequence: `emit_mode = reform` is the sanctioned tapered path, and unweighted ~−13 dB sidelobes are an honest description rather than a placeholder | [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp → Notes |
| **D17 — `ApertureManagerOp` + `BackprojectionOp`** | Was R8; decided 2026-07-30. Renamed from `ApertureControllerOp`, which predated the split and named nothing in radar. "Manager" was chosen over the house-pattern `ApertureAssemblyOp` because the operator's four jobs — membership, emission, reset authority, geometry staging — all derive from one concept, the aperture's lifecycle, and a transformation-style name describes only the first. Accepted cost: a role name, from which the ports cannot be guessed. Rejected: `CPIAssemblyOp` (standard vocabulary, but pulse-Doppler dialect), anything on "Scheduler" (collides with Holoscan), and `BackprojectionAccumulatorOp` (buries the algorithm). Invariant 7 does not govern, there being no literature term for this stage | [OPERATOR_API.md](OPERATOR_API.md) → ApertureManagerOp → Notes |
| **D16 — Four control values, and the tick has phases** | Emission is always evaluated *after* the tick's pulses; the reset's phase (before accumulation, or after the emit) is what distinguishes the values, which makes the vocabulary a 2×2 and adds *reset-then-emit* as the completing cell. A geometry change lands in its reset's phase, which fixes a latent bug: the previous "control-plane before pulses" rule would have resized the accumulator out from under a closing aperture's image. Rejected: chunking a re-form across ticks using only the three original values, which leaves formation observably half-re-formed | [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation |
| **D14 — `BackprojectionOp` publishes no aperture state** | Was R3; closed 2026-07-30 once the manager split and the `evict` port left nothing needing it — a manager decides from externally observable state, and an application owning eviction knows its occupancy by construction. Rejected: an output port carrying pulses accumulated and angle subtended. Reopens only for a rule needing state internal to formation, such as partial-image sharpness | [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation |
| **D13 — No trajectory operator, and no scene-geometry operator** | Both of `BackprojectionOp`'s non-pulse inputs are the application's to produce. Rejected: library operators for either, which could only guess at deployment-specific conventions while looking authoritative | [OPERATOR_API.md](OPERATOR_API.md) → Two deliberate non-operators |
