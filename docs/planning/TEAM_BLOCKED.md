# Team-Blocked Items

Contract / design questions that need a **team decision**. Individuals must
not silently resolve these in code.

Per-operator detail lives in each operator’s
[OPEN_ITEMS.md](../../operators/). This file is the roll-up. Before a modality
has operator directories, its planning docs hold that detail instead — see
*Per-modality files* below. When a decision
lands, record it in [STATUS.md](STATUS.md) → Settled decisions, then remove
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

**Per-modality files.** A modality whose contract is still under review keeps
its own roll-up alongside its planning docs, and this file indexes it rather
than restating its rows. Today that is
[SAR/TEAM_BLOCKED.md](SAR/TEAM_BLOCKED.md), which also carries the convention
for where detail lives before any operator directory exists.

---

## How to use

| Action | Do this |
| --- | --- |
| Discover a new team question | Write it in `operators/<group>/<op>/OPEN_ITEMS.md` — or, before that directory exists, in the planning doc that raised it (see *Per-modality files*) — then add a row in the **correct class** below |
| Land code ahead of the contract | Record the conflict in `OPEN_ITEMS.md`; put the row under **Needs team look** (unless it truly stops checklist completion) |
| Resolve | Team decision → `STATUS.md` Settled decisions → delete/archive both places |

Agents: [AGENTS.md](../../AGENTS.md) (“Do not silently resolve documented open
items”) and [OPERATOR_DEVELOPMENT.md](OPERATOR_DEVELOPMENT.md) § Open items.

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

**None currently for FMCW.** The six open SAR items (R1, R2, R4, R5, R7, R9) are indexed in
[SAR/TEAM_BLOCKED.md](SAR/TEAM_BLOCKED.md). They do **not** stop development,
because no SAR operator code exists yet; they must be answered before the SAR
contract is accepted, and none of them may be resolved in code. That file also
carries D1–D17: contract calls that already have an answer and do not gate
acceptance, enumerated so focused review reaches them rather than relying on a
reviewer finding them in seven hundred lines of contract.
