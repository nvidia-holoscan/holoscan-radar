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
two classes below). HSDK 5 / EA migration friction lives in
[`HSDK5_PORT.md`](HSDK5_PORT.md).

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

Items L1–L18 and S1–S4 are settled — see STATUS.md → Settled decisions.
Capabilities deliberately deferred out of the current scope (e.g. staggered TDM,
sum-set holey subarray smoothing) are tracked under STATUS.md → Roadmap, not
here.

| ID | Question | Current state | Detail |
| --- | --- | --- | --- |
| A1 | How should required `AngleConvention` Params represent omission? | The contract says no default, but the shared value type default-constructs to valid `radar_boresight`, so operators cannot distinguish omission from an explicit preset. Choose optional Params, an unset enum value, or schema-level presence module-wide. | [`PhaseShiftBeamformerOp` OPEN_ITEMS §3](../../operators/angle_estimation/phase_shift_beamformer_op/OPEN_ITEMS.md) |

L16/L17 settled temporal semantics and L18 settled the mixed HSDK 5 physical
`[live]` payload policy. Transmit-stamp ownership (sequencer versus device
report) remains an application architecture choice; the consumer contract is
unchanged ([`TEMPORAL_CONTRACTS.md`](TEMPORAL_CONTRACTS.md)).
