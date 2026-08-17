# Max Tensor Extents (in progress)

**Status:** C++ operator wave landed on `dpc/hs5.0-fixes` (all fat ops +
Graph unit tests at `ctest -j4`). Remaining: apps Graph port (phase 4),
Python bindings polish, xl perf ladders vs honest spans. **Pinned baseline
(2026-08-15):**
[`MAX_TENSOR_EXTENTS_2026-08-15.md`](MAX_TENSOR_EXTENTS_2026-08-15.md) —
keep that file unchanged; edit this living doc for later revisions.

Module convention for sizing HSDK 5 Tensor output pools, **plus** dropping
opaque leading-batch as a module invariant. Pool sizing is **not** an HSDK 5
SDK type — Holoscan only exposes per-output
`tensor_allocation({.max_rank, .max_byte_span})`. See also
[`HSDK5_PORT.md`](HSDK5_PORT.md).

---

## For Dylan — decision brief

**HSDK 5 fact:** each Tensor output with `tensor_allocation` reserves
approximately `slot_count × max_byte_span` at startup (before arm). Graph may
only **tighten** bounds after `setup()`, never widen. `set_params()` cannot
change `tensor_allocation` (`GRAPH_PARAMETER_PORT_SHAPE_CHANGED`). Stock
examples hardcode a `constexpr` span in `setup()` — fine for demos, not for a
radar library with large cubes.

**Problem for us:** today’s hardcoded `256 MiB` per fat port either OOMs
multi-op graphs (`N × slots × ceiling`) or caps apps below real cube sizes. A
bigger library-wide ceiling makes OOM worse. Separately, opaque leading `...`
batch cannot be pooled correctly unless extents include it; leaving batch in
ops without sizing leaves **unusable** capability.

**Proposal (two coupled parts):**

### A — `MaxTensorExtents` (pool sizing)

1. Add one reusable extents type, e.g. `holoscan::radar::MaxTensorExtents`,
   whose **only contract-fixed names** are the dimension fields that size
   tensors, each prefixed `max_` (`max_channels`, `max_chirps`,
   `max_range_bins`, `max_doppler_bins`, `max_detections`, `max_az`, `max_el`,
   … — exact list TBD). Optional/absent fields are allowed; each operator
   documents which fields it requires for its outs. **No** `batch_leading`
   field.
2. **Do not** fix product bucket names (`adc_cube`, `rd_maps`, …) in the
   module type. Apps may group extents however they like in YAML or locals;
   shapes grow/shrink along the pipeline by passing **different filled
   instances** into different ops.
3. Fat-output ops take `MaxTensorExtents` (ctor capture):
   `graph.op<Op>(name, extents)`.
4. Each op’s `setup()` derives `max_byte_span` from the fields it needs (+
   dtype). Helpers live in `radar_common`.
5. Tiny fixed outs keep a small constexpr.
6. App path: YAML DOM → helper `MaxTensorExtents::from_yaml(node)` → ctor
   capture (operators stay yaml-free).

### B — Remove opaque leading batch (invariant change) — **in scope**

**In this plan (not a follow-up):** drop opaque leading `...` batch as a
**module/operator** obligation. Multi-item throughput, if needed, is an
**application** concern (loop / fan-out). Module-level leading batch is not
assumed to be required; if needed later, likely only on **targeted** ops
(explicit contract + extents), not a global restore of rank-agnostic `...`.

**Why coupled to A:** Graph pools are sized from extents at `setup()`. Batch
without an extents field cannot be admitted correctly; batch code without a
legal app path is dead/unusable. Prefer one honest contract: **fixed trailing
radar ranks only**.

**Work included after approval:**

- Rewrite design invariant in `AGENTS.md` (replace “Rank-agnostic batching”).
- Update `OPERATOR_API.md` port shapes: remove leading `...` from contracts
  (keep named axes: NCI dims, `num_snapshots`, etc.).
- Checklist / `OPERATOR_DEVELOPMENT.md` / `STATUS.md` settled decision.
- Operators **fail closed** on unexpected extra leading dims.
- Strip or simplify code and tests that exist only for opaque `...` batch.
- No stub `batch_leading` on `MaxTensorExtents`.

**Tradeoffs (accept deliberately):**

- No in-module MatX over an opaque leading batch axis; apps compose instead.
- Re-adding batch later on selected ops is a real contract change (extents +
  those ops + chain neighbors), not a one-line flip — acceptable.
- Named contract axes stay; only opaque leading `...` goes away.

**Ask:** approve A + B together (or amend).

---

## Terminology

“Known before pools exist” = Holoscan cold declaration at `graph.op` /
`setup()` — **not** C++ compile-time of the `.so`. Graph-build runtime config
is fine. Params/`set_params` arrive too late to **raise** `max_byte_span`.

---

## Mechanism (tentative)

```cpp
// Contract-fixed field names only — not product names; no batch_leading.
// Every dimension field is max_-prefixed.
struct MaxTensorExtents {
  std::optional<std::int64_t> max_channels;
  std::optional<std::int64_t> max_chirps;
  std::optional<std::int64_t> max_range_bins;
  std::optional<std::int64_t> max_doppler_bins;
  std::optional<std::int64_t> max_detections;
  std::optional<std::int64_t> max_az;
  std::optional<std::int64_t> max_el;
  // ...
};

// Parent / grouping keys (after_range, for_beamscan) are app-local.
// Keys *inside* each node match MaxTensorExtents fields (max_channels, …).
const auto after_range = MaxTensorExtentsFromYaml(root["after_range"]);
const auto for_beamscan = MaxTensorExtentsFromYaml(root["for_beamscan"]);

auto range = graph.op<FMCWRangeProcessorOp<float>>("range", after_range);
auto beam = graph.op<BeamscanEstimatorOp<float>>("beamscan", for_beamscan);
beam.set_params(...);  // grids ≤ for_beamscan.max_az/max_el/max_detections
```

```text
App fills MaxTensorExtents (from YAML or cfg)
  -> graph.op<Op>(name, extents)   // must be known here
  -> setup() tensor_allocation from required fields
  -> cold materialize (slot_count × span)
```

### What is fixed vs flexible

| Fixed by module contract | App-flexible |
| --- | --- |
| Extent **field names** on `MaxTensorExtents` (and the same keys under any node passed to `MaxTensorExtentsFromYaml`) | Which YAML **parent/group** nodes exist (`after_range`, `for_beamscan`, …) and which node is passed into which `op()` |
| Which fields each **operator** requires (per-op docs) | Filling extents without YAML (e.g. from `cfg` / cube / grids into the struct) |
| Ctor-capture injection (not Params) | Deriving the numeric values from scene config |
| No opaque leading batch on data ports | App-level multi-item composition |

### Config YAML timing

`run_fmcw_app` loads app YAML into `cfg` **before** `compose()` / `graph.op`.
So apps can build `MaxTensorExtents` from config in time for construction.
`from_config` / `set_params` remain for algorithm Params only.

### Library pieces (after approval)

1. `MaxTensorExtents` + YAML parse helper + byte-span helpers per op/out kind.
2. Fat ops: ctor `(MaxTensorExtents)`; fail closed if required fields missing
   or if input rank implies opaque leading batch.
3. Remove opaque-batch paths/tests; update planning invariants.
4. No library-wide MiB ceiling as the primary policy.
5. Graph tighten remains optional secondary shrink.

### Rejected / deferred

- Fixed product members (`adc_cube`, `rd_maps`, …) on a module bounds object.
- Params-driven pool resize — too late / cannot raise span.
- Per-op positional arrays — brittle.
- Keeping batch-capable op code without extents support — unusable.
- Global “rank-agnostic `...` batch” as a standing promise — out; targeted
  per-op batch later only if a real app need appears.

---

## Blast radius analysis

This is a **module-scale** change (contract + infrastructure), not a small
HSDK 5 fix. Blast radius is **large but uneven** across the two parts.

### Part A — `MaxTensorExtents` (wide, mostly mechanical)

| Surface | Impact |
| --- | --- |
| Fat operators (~19) | New ctor capture; `setup()` derives `max_byte_span`; fail if required fields missing |
| Graph unit tests | Every `graph.op<...>()` must pass extents |
| Perf drivers (`perf_graph` + all `perf_*.cpp`) | Same; oversize ladders must match honest spans |
| `radar_common` | New type + `from_yaml` + byte-span helpers |
| Docs / checklist | Field list, ctor convention, sizing recipe |
| Python bindings (phase 4) | Ctor / bind surface must expose extents when apps return |

Pain is **volume and repetition**, not algorithmic redesign. Same pattern per
op.

### Part B — drop opaque leading `...` (wide + sharper)

| Surface | Impact |
| --- | --- |
| `AGENTS.md` invariant | Replace “Rank-agnostic batching” |
| `OPERATOR_API.md` / `WORKFLOWS.md` | Strip leading `...` from port shapes (many rows) |
| Checklist / `OPERATOR_DEVELOPMENT.md` / `STATUS.md` | Settled decision + done-gate wording |
| Operator implementations | Fail closed on unexpected extra leading dims; audit `collapse_batch` / `max_rank` / `rank = nullopt` usage |
| Tests | Cases with an extra leading dim (e.g. some MIMO / VAA shapes) become reject tests or drop the leading axis |

**Important distinction:** MatX “batch” as a *collapsed product of named axes*
(e.g. RX×chirps) is **not** opaque leading `...` and is not removed. Part B
targets contract-level opaque leading dims only. Underestimating this audit
is the main risk.

### Demo apps

Current demos are still `Application` + `make_operator` / `from_config` and
are already **phase-4 Graph ports** (Holoviz / EA). They are not a second
full DSP rewrite solely for this plan, but the Graph port **must** include
extents wiring (`from_yaml` / from `cfg` → `graph.op(..., extents)`).

**Ordering:** landing **required** extents ctors before apps are Graph-ported
breaks today’s `make_operator` demos unless a temporary default/transition
ctor exists. Prefer shipping extents with the app Graph wave, or updating
apps in the same change set.

Unbatched-only is unlikely to force demo logic changes by itself (demos are
already unbatched).

### What stays small

- No radar algorithm redesign
- Named axes (TX/RX, NCI, `num_snapshots`, …) remain
- Smoke/unbatched Graph tests already match the end-state shape regime

### Suggested landing shape (after approval)

Split to keep reviews reviewable:

1. Docs / invariant (Part B contract text) + `MaxTensorExtents` type
2. Operator waves (ctor + fail-closed rank) + tests/perf
3. Apps when phase 4 Graph port lands (extents at each `op()`)

Expect **multi-PR / multi-day** work, not a single drive-by commit.

---

## Out of scope (unless redirected)

- Library-wide `MAX_BYTE_SPAN_MIB` as the primary fix
- Algorithm changes unrelated to ranks / pool sizing
- Implementing app-level batch helpers (apps own that)

---

## After approval — implement

1. **Invariant / docs:** `AGENTS.md`, `OPERATOR_API.md` (drop leading `...`),
   checklist, `OPERATOR_DEVELOPMENT.md`, `HSDK5_PORT.md`, `STATUS.md` settled
   decision.
2. **`MaxTensorExtents`** + `from_yaml` + byte-span helpers in `radar_common`.
3. **Fat ops:** ctor capture; `setup()` spans from extents; reject unexpected
   leading dims; strip opaque-batch-only code/tests.
4. **Apps / `perf_graph`:** pass extents; fix oversize ladders vs honest spans.
5. Field list locked from current op inventory (see discussion notes /
   OPERATOR_API shapes).
