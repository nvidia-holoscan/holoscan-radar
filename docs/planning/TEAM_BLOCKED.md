# Team-Blocked Items

Contract / design questions that need a **team decision**. Individuals must
not silently resolve these in code.

Per-operator detail lives in each operator’s
[`OPEN_ITEMS.md`](../../operators/). This file is the roll-up. When a decision
lands, record it in [`STATUS.md`](STATUS.md) → Settled decisions, then remove
or archive the operator section *and* the matching row here.

Items fall into **two classes**. Keep that distinction sharp when adding rows:

| Class | Meaning | Development impact |
| --- | --- | --- |
| **Stops development** | Work cannot finish (or cannot start the next operator) until the team picks an answer | Checklist incomplete, param intentionally undeclared, or next-wave op blocked from landing |
| **Needs team look** | An interim answer is already in code (or the feature is out of current scope); the team should ratify, revise, or reject | Does **not** block landing or day-to-day implementation; do not treat as a stop-ship |

**Not in this file:** perf follow-ups, encoding notes pending bindings, deferred
algorithm slices that are simply out of scope, and SDK caveats that are only
documented (unless they need a product policy — then they belong in one of the
two classes below).

**Before an operator directory exists**, the "detail lives in `OPEN_ITEMS.md`"
convention has nowhere to point, so detail lives in the planning doc that raised
the question — for the SAR rows below, [`SAR/OPERATOR_API.md`](SAR/OPERATOR_API.md)
→ *Open items*, which is also where that document's implementation-time items
(the non-team-blocked class) are enumerated. When the operator directories land,
migrate each open row into the matching `OPEN_ITEMS.md` and repoint the Detail
column; a row that is still open at that point must not be dropped in the move.

---

## How to use

| Action | Do this |
| --- | --- |
| Discover a new team question | Write it in `operators/<group>/<op>/OPEN_ITEMS.md`, then add a row in the **correct class** below |
| Land code ahead of the contract | Record the conflict in `OPEN_ITEMS.md`; put the row under **Needs team look** (unless it truly stops checklist completion) |
| Resolve | Team decision → `STATUS.md` Settled decisions → delete/archive both places |

Agents: [`AGENTS.md`](../../AGENTS.md) (“Do not silently resolve documented open
items”) and [`OPERATOR_DEVELOPMENT.md`](OPERATOR_DEVELOPMENT.md) § Open items.

---

## Stops development

These are the only items that should pause or refuse checklist completion /
starting a dependent operator.

**None currently.** (S1 spatial-smoothing, S2 source-count, S3 subarray-selection,
and S4 dense-path angle features are all settled — see STATUS.md → Settled
decisions; residual ratification forks are under **Needs team look** below.)

---

## Needs team look

Interim stance already chosen (or the capability is explicitly out of current
scope). Development and checklist completion proceed; these need ratification
or a deliberate contract change when the team has bandwidth.

All prior FMCW items (L1–L15, plus S1–S4) are settled — see STATUS.md →
Settled decisions. Capabilities deliberately deferred out of the current scope
(e.g. staggered TDM, sum-set holey subarray smoothing) are tracked under
STATUS.md → Roadmap, not here.

The SAR rows below are open. They do **not** stop development, because no SAR
operator code exists yet; they must be answered before the SAR contract is
accepted, and none of them may be resolved in code.

| Item | Question | Detail |
| --- | --- | --- |
| **R1 — SAR contract acceptance** | Accept, revise, or reject the proposed SAR modality contract as a whole, including its three amendments to the locked invariants (descriptor set 6→7 plus an `FMCWWaveformDescriptor` → `LFMWaveformDescriptor` rename; an orthogonal internal-arithmetic `precision_mode` on `BackprojectionOp`; output shape from declared sample geometry rather than an input tensor) and the new cross-tick accumulation convention | [SAR/OPERATOR_API.md](SAR/OPERATOR_API.md) → Contract changes |
| **R2 — Aperture emit decision** | How an application supplies the "enough aperture traversed" decision to `BackprojectionOp`. Three mechanisms verified viable against HSDK 4.3 source (a closed `emit_policy` enum; a non-blocking `emit_now` control port; a `std::function` parameter); `set_dynamic_flows` and condition-gating are ruled out with reasons. Pick one — do not implement a default | [SAR/OPERATOR_API.md](SAR/OPERATOR_API.md) → Cross-tick accumulation |
| **R3 — Aperture state on a port** | Whether `BackprojectionOp` publishes its aperture state (pulses accumulated, integration angle subtended). Sub-question of R2: it is what would let an external controller decide on operator-internal state | [SAR/OPERATOR_API.md](SAR/OPERATOR_API.md) → Cross-tick accumulation |
| **R4 — Autofocus grid conjugacy** | PGA assumes the image azimuth axis is the Fourier conjugate of slow-time, which does not hold for an arbitrary backprojection sample geometry. Restrict `AutofocusOp` to conjugate grids, make it a re-form feedback loop through `BackprojectionOp`, or implement a backprojection-domain autofocus | [SAR/OPERATOR_API.md](SAR/OPERATOR_API.md) → AutofocusOp |
| **R5 — Metadata plane** | Whether a per-message metadata plane is in scope for the SAR lockdown. Forced by the standard-format writers, since metadata is *transformed* by each stage rather than merely carried. Candidate mechanism is Holoscan's metadata dictionary, used nowhere in this repo. Cross-cutting: it belongs in the contract or is explicitly deferred | [SAR/OPERATOR_API.md](SAR/OPERATOR_API.md) → Standard-Format I/O |
| **R6 — Aperture (azimuth) weighting** | Nothing in the SAR chain applies one, leaving cross-range sidelobes at the unweighted ~−13 dB and `SICDWriterOp` unable to populate SICD's azimuth impulse-response fields. Not a one-line parameter addition: an aperture taper needs the finished aperture extent, which a streaming accumulator does not have when a pulse arrives. Three resolutions — weight only in emit-and-reset mode, weight the emitted image (which reintroduces R4's conjugacy assumption), or buffer and form on emit | [SAR/OPERATOR_API.md](SAR/OPERATOR_API.md) → BackprojectionOp → Notes |
