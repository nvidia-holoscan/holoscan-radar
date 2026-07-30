# SAR Team-Blocked Items

Contract / design questions raised by the **proposed SAR modality** that need a
**team decision**. Individuals must not silently resolve these in code.

This file is the SAR counterpart to the module-wide roll-up,
[`../TEAM_BLOCKED.md`](../TEAM_BLOCKED.md), and uses the same two classes and
the same resolution procedure. When a decision lands, record it in
[`STATUS.md`](../STATUS.md) → Settled decisions, then remove the row here *and*
the matching entry in [`OPERATOR_API.md`](OPERATOR_API.md) → *Open items*.

**Before an operator directory exists**, the "detail lives in `OPEN_ITEMS.md`"
convention has nowhere to point, so detail lives in the planning doc that
raised the question — for the rows below, [`OPERATOR_API.md`](OPERATOR_API.md)
→ *Open items*, which is also where this modality's implementation-time items
(the non-team-blocked class, I1–I5) are enumerated. When the operator
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
| **R1 — SAR contract acceptance** | Accept, revise, or reject the proposed SAR modality contract as a whole, including its three amendments to the locked invariants (descriptor set 6→7 plus an `FMCWWaveformDescriptor` → `LFMWaveformDescriptor` rename; an orthogonal internal-arithmetic `precision_mode` on `BackprojectionOp`; output shape from declared sample geometry rather than an input tensor) and the new cross-tick accumulation convention | [OPERATOR_API.md](OPERATOR_API.md) → Contract changes |
| **R2 — Aperture emit decision** | How an application supplies the "enough aperture traversed" decision to `BackprojectionOp`. Three mechanisms verified viable against HSDK 4.3 source (a closed `emit_policy` enum; a non-blocking `emit_now` control port; a `std::function` parameter); `set_dynamic_flows` and condition-gating are ruled out with reasons. Pick one — do not implement a default | [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation |
| **R3 — Aperture state on a port** | Whether `BackprojectionOp` publishes its aperture state (pulses accumulated, integration angle subtended). Sub-question of R2: it is what would let an external controller decide on operator-internal state | [OPERATOR_API.md](OPERATOR_API.md) → Cross-tick accumulation |
| **R4 — Autofocus grid conjugacy** | PGA assumes the image azimuth axis is the Fourier conjugate of slow-time, which does not hold for an arbitrary backprojection sample geometry. Restrict `AutofocusOp` to conjugate grids, make it a re-form feedback loop through `BackprojectionOp`, or implement a backprojection-domain autofocus | [OPERATOR_API.md](OPERATOR_API.md) → AutofocusOp |
| **R5 — Metadata plane** | Whether a per-message metadata plane is in scope for the SAR lockdown. Forced by the standard-format writers, since metadata is *transformed* by each stage rather than merely carried. Candidate mechanism is Holoscan's metadata dictionary, used nowhere in this repo. Cross-cutting: it belongs in the contract or is explicitly deferred | [OPERATOR_API.md](OPERATOR_API.md) → Standard-Format I/O |
| **R6 — Aperture (azimuth) weighting** | Nothing in the SAR chain applies one, leaving cross-range sidelobes at the unweighted ~−13 dB and `SICDWriterOp` unable to populate SICD's azimuth impulse-response fields. Not a one-line parameter addition: an aperture taper needs the finished aperture extent, which a streaming accumulator does not have when a pulse arrives. Three resolutions — weight only in emit-and-reset mode, weight the emitted image (which reintroduces R4's conjugacy assumption), or buffer and form on emit | [OPERATOR_API.md](OPERATOR_API.md) → BackprojectionOp → Notes |

---

Agents: [`AGENTS.md`](../../../AGENTS.md) ("Do not silently resolve documented
open items") and [`OPERATOR_DEVELOPMENT.md`](../OPERATOR_DEVELOPMENT.md)
§ Open items.
