# Operator Development Guidance

Detailed guidance for agents (and humans) implementing operators in
**holoscan-radar**. This is the experience layer on top of the locked contracts
and the landing checklist: how to build, diagnose, and harden operators without
repeating expensive mistakes.

## Relationship to other docs

| Doc | Role |
| --- | --- |
| [`OPERATOR_API.md`](OPERATOR_API.md) | Locked contract — ports, shapes, dtypes, params. Source of truth for *what*. |
| [`WORKFLOWS.md`](WORKFLOWS.md) | How operators chain; dtype/port wiring across modalities. |
| [`TESTING.md`](TESTING.md) | What properties tests must assert. |
| [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md) | Landing gate — verify before claiming done. |
| [`STATUS.md`](STATUS.md) | Claim work; record decisions. |
| [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md) | Index of items blocked on a team decision (detail in operator `OPEN_ITEMS.md`). |
| [`operators/common/README.md`](../../operators/common/README.md) | Shared conventions and helpers. |
| This doc | *How* to implement well: hot-path discipline, diagnosis, perf, open items. |

Follow the checklist to land. Read this when designing `compute()`, choosing
MatX vs CUDA, adding `[live]` paths, writing tests, or chasing "why is this
slow?"

---

## Non-negotiables

1. **Docs first.** Code matches `OPERATOR_API.md`. Contract changes need an
   explicit doc update in the same PR (or a dedicated PR), called out as a
   contract change — never silent drift. Prefer proposing a contract change
   over inventing an interface in code; treat contract edits as a *hesitant*
   option after diagnosing whether the problem is implementation, wording, or
   a real gap.
2. **Do not silently resolve open items.** Ambiguities go in
   `operators/<group>/<op>/OPEN_ITEMS.md` and stay there until the team
   decides. Index every *team* block in
   [`TEAM_BLOCKED.md`](TEAM_BLOCKED.md) as well. If code must land ahead of
   the contract (rare), record that fact in `OPEN_ITEMS.md` with the conflict
   and the question for the team.
3. **Diagnose before guessing.** When something is wrong or slow, run the
   cheapest experiment that distinguishes plausible causes *before* proposing a
   retrain-scale fix (kernel rewrite, dataset regen, architecture change). A
   five-minute reversible change is fine as a diagnostic; a multi-hour rebuild
   of calibration tables is not a first guess.
4. **Design invariants** in [`AGENTS.md`](../../AGENTS.md) — plain-tensor ports,
   bounded descriptors, template precision, named axes (no opaque leading
   batch), `[live]` convention, convention-free kernels. Do not invent new
   shared message types
   or descriptor ports without a contract change.
5. **Checklist means the whole operator.** Vertical slices are fine for
   progress, but
   [`NEW_OPERATOR_CHECKLIST.md`](NEW_OPERATOR_CHECKLIST.md) clears only when
   the full `OPERATOR_API.md` contract is implemented. Do not call an
   operator ready / checklist-complete while contract features remain staged.

---

## Workflow for a new operator

```text
STATUS.md claim
    → read OPERATOR_API + WORKFLOWS + any OPEN_ITEMS
    → scaffold (group CMake, metadata, three-file split, README row)
    → implement vertical slices toward the full contract (PRs OK mid-way)
    → unit (+ statistical) tests; wire every output port
    → performance baseline when the hot path exists
    → clear NEW_OPERATOR_CHECKLIST only when the full contract is done
    → record team open items; update STATUS; PR
```

### Slice by contract surface, not by "get it compiling"

Land *faithful intermediate slices* that are useful and testable (one variant,
one path), with not-yet-landed modes throwing clearly. Slices are how you
make progress — they are **not** checklist completion. Prefer:

- CA before GO/SO/OS (or whichever mode has a closed-form reference)
- Complex + real dtype paths early (they share most of the shell)
- Exact `[live]` plumbing (wiring + retention) with at least one
  mid-stream match and a following miss tick that reverts to Params
- Sparse / variable-size outputs (compaction) as a staged follow-up if a host
  path unblocks correctness first — but schedule the GPU path before calling
  the operator "performance-ready"

### File layout (GPU operators)

```text
operators/<group>/<op_snake>/
  <op>.hpp           # contract reference + class shell
  <op>_stage.hpp     # kernels / MatX (CUDA-TU only)
  <op>.cu            # setup / compute / explicit instantiations
  metadata.json
  OPEN_ITEMS.md      # when team questions arise
```

Keep `compute()` thin: resolve overrides → obtain buffers → call stage → emit.
Put math in the stage. Put host calibration / LUT builders in dedicated TUs
(`.cpp` / `.cu`) so the operator shell stays readable.

### Shared helpers (prefer these over reinventing)

| Need | Header |
| --- | --- |
| Single-entry `TensorMap` receive/emit | `radar_common/sdk/tensor_io.hpp` |
| Exact `[live]` resolve | `radar_common/sdk/override_param.hpp` |
| Warn-once artifact cache | `radar_common/sdk/config_cache.hpp` |
| Arg setters for custom types | `radar_common/sdk/arg_setters.hpp` |
| Batching / dtype / device / contiguity checks | `radar_common/tensor/tensor_utils.hpp` |
| CUDA last-error guard | `radar_common/tensor/cuda_check.hpp` |
| Device tensor alloc | `radar_common/tensor/dlpack_buffer.hpp` |
| MatX views | `radar_common/tensor/matx_utils.hpp` (`.cu` / stage only) |
| Tests | `tests/cpp/utils/sdk_harness.hpp`, `tensor_fixtures.hpp` |

**Do not re-derive per-operator I/O or validation boilerplate.** The single-entry
`TensorMap` receive/emit (`receive_single` / `receive_optional` / `emit_single`),
the input checks (`require_cuda_accessible` / `require_contiguous` /
`require_dtype`), and the CUDA last-error guard (`check_cuda`) all live in
`radar_common` and take the operator name as a `who` argument so error messages
stay specific. Call them; do not paste a local `namespace { ... }` copy into the
`.cu`. If you genuinely need a helper that is not there yet and more than one
operator would use it, add it to `radar_common` (who-parameterized) rather than
inlining a private version — silent per-operator copies drift apart (subtly
different messages, empty-map handling, null checks) and are exactly what the
detection and FMCW operators were consolidated *out of* once each had grown its
own near-identical set.

---

## Architecture patterns that work

### Thin Holoscan shell, thick stage

Holoscan calls `initialize` / `setup` / `compute`. The operator is a graph node,
not a program. Data plane = single-entry `TensorMap`s; config plane = parameters
plus optional same-named override ports.

### Precision is a template, not a runtime flag

`template <typename T>` with `T ∈ {float, double}`; explicit instantiations in
the `.cu`. Inputs and outputs of an instance match that precision.

### Named trailing axes (no opaque leading batch)

Infer `D`, `R`, FFT lengths, etc. from the tensor each tick. Contract axes are
**named** (including named leading axes such as `N_rx` / `M_tx`); do not admit
an open-ended opaque leading `...` batch at the module level — multi-item
throughput is an app concern. Shape-changing `[live]` overrides remain legal
because downstream re-infers named extents. Fat Tensor outs take
`MaxTensorExtents` at `graph.op()` so HSDK 5 pools can be sized before arm
(see [`MAX_TENSOR_EXTENTS.md`](MAX_TENSOR_EXTENTS.md)).

### Streams and lifetimes

- `receive_cuda_stream()` on the primary data input; enqueue on that stream.
- Retain inputs (and any borrowed device buffers) until the next tick if
  kernels may still be reading them after `compute()` returns
  (`retained_input_` pattern).
- Prefer stream-ordered work; avoid `cudaStreamSynchronize` on the hot path
  unless a scalar result is truly required (and even then, minimize it).

### Optional outputs: never `HOLOSCAN_TRY`, and pick a disposition

An operator cannot see its own wiring. `Output` has no connection query and
`contract()` has no output side, so it cannot tell a port the application
declined from one it wants, and publishing to an output with no consumer
fails. Wrapping that emit in `HOLOSCAN_TRY` — the right reflex everywhere else
— propagates the failure out of `compute()` and fails the entire activation,
so declining an optional output would silently kill the tick's other outputs
too. Keep `HOLOSCAN_TRY` for required outputs only.

Every optional output takes one of the two dispositions locked in
[`OPERATOR_API.md`](OPERATOR_API.md), "Optional output convention". Choose by
asking what work exists **solely** to produce the port:

**`always`** — nothing to save, because the value falls out of work another
output needs anyway (`angle_spectrum` is the same scan-grid pass that yields
`aoa_estimates`). Discard the result and move on; this is silent by design.

```cpp
(void)emit_prepared_tensor(angle_spectrum, context, std::move(spec_loan));
```

**`latched`** — the port has dedicated work worth skipping. Hold a
`LatchedOutput` member, gate that work on `active()`, and emit through it. The
first failed emit warns once and disables the port for the rest of the run:

```cpp
if (waveform_present && range_bins_out_.active()) { /* allocate + fill */ }
...
if (bins_loan.has_value()) {
  range_bins_out_.emit(range_bins, context, std::move(*bins_loan),
                       "FMCWRangeProcessorOp.range_bins");
}
```

Gate the whole cost, not just the emit — that is the point. Where one kernel
feeds two optional ports, gate it on either still being active and let each
port gate its own emit. `LatchedOutput` latches only on `kFailure`, so a
transient fault costs one frame rather than the run.

`emit_prepared_tensor` and `emit_device_tensor` carry `[[nodiscard]]` like the
`Output::emit` they wrap, so forgetting to handle the result is a build error
and the `(void)` reads as the deliberate choice it is.

### Sample time propagation and aligned joins

`capture_time` is the frame's logical time, and HSDK 5 propagates it for you.
For a **data-triggered** operator, `context.activation_time()` equals the
triggering sample's `capture_time` exactly — not the operator's wall-clock
dispatch. So `emit_prepared_tensor` / `emit_device_tensor` stamping
`activation_time()` *forwards* the incoming frame time rather than overwriting
it, and a chain of pass-through operators keeps every branch of a diamond on
one shared time. Do not "fix" those helpers to re-read a clock, and do not
hand-roll a stamp in a pass-through operator.

`OnAligned` enforces its window strictly: `.window = 0ns` means exact
capture-time equality of the queue heads, so `SnapshotGatherOp`'s zero window is
load-bearing and must not be loosened. Alignment is also *consuming* — when the
heads cannot be matched, the unmatched head is discarded rather than paired with
a neighbor, so a branch that skips a frame costs that frame and nothing more.
Never add manual frame-matching on top of an aligned trigger.

Two places the automatic propagation does not apply:

- **`OnClock` sources.** Activation time is the tick, so an operator that owns a
  genuine acquisition time (a real receiver, a simulator modeling one) must
  supply it explicitly in `EmitOptions{.capture_time = ...}`. Everything
  downstream then inherits that time, which is what makes a coherent join
  against a separately-produced descriptor branch possible.
- **Aligned fusion operators.** When an aligned operator's heads differ within a
  non-zero window, its activation time is the *newest* of them, so re-stamping
  advances the frame's time by up to the window on each fusion hop. Where the
  original frame time must survive, read the chosen input with
  `Input<T>::receive()` and forward `sample.metadata.capture_time()` —
  `receive_data()` (and therefore `receive_tensor`) discards the metadata.

**Only `OnAligned` waits for a companion, so `OnEach` + `Exact{}` is a bet on
causal order.** `OnEach` fires the moment the driver lands and never waits, and
`Exact{}` on a side input resolves against that driver's `capture_time`. Whether
that pairing works is therefore decided entirely by whether the companion was
already published:

- **A sibling branch of the same frame loses every time.** A side value computed
  by a *different* operator from the same source has not run yet when the driver
  arrives, so selection freezes with `kNeverSeen` and the activation takes the
  degraded path — on every frame, under the deterministic serial scheduler, so
  the companion is never applied at all. Use `OnAligned` here; do not build it
  out of `OnEach`.
- **A companion that causally precedes the frame wins every time.** Config
  published *before* the frame it describes is already retained when the cube
  lands, so `Exact{}` pairs them exactly, with no window and no wait — verified
  across a full frame of acquisition lag in `tmp-app/control_plane_app.cpp`,
  including a negative control (1 ms stamp skew → `kNeverSeen`, no silent
  pairing). This is the module's chosen shape for coherence-critical `[live]`
  descriptors; the topology it requires, and what it replaces, are in
  [`TEMPORAL_CONTRACTS.md`](TEMPORAL_CONTRACTS.md).

Reach for `OnEach` + `LatestWithin` when stale is acceptable.

`OnWindow` is the only trigger that combines a data driver with a bounded wait,
but it is shaped for sensor-fusion neighborhoods, not for pairing: it rejects a
scalar `Exact` companion (`OnWindow requires at least one BracketedWindow
input`), so the operator scans `selected_batch()` and compares
`capture_timestamp_ns` itself. **A bracket always waits**, because
`lookahead = 0` is not expressible — a lookbehind-only `BracketedWindow` is
rejected as `TEMPORAL_CONTRACT_INVALID`, so at least one strict *successor*
companion is always required to close a window. What that costs depends on the
companion's cadence, not the driver's: the window closes once the next
companion sample after the anchor arrives, bounded above by `max_wait`. A
companion published once per frame therefore costs a frame (and `max_wait` must
exceed the frame period), while a companion published at 40 ms against 100 ms
cubes cost ~20 ms in `tmp-app/lookbehind_probe.cpp`.

**An incomplete bracket is no bracket, so `OnWindow` is the wrong shape for an
irregular companion.** The selection is all-or-nothing: when the window closes
on timeout the input is *absent* with `kWindowTimeout` — the operator does not
receive the lookbehind samples it already has. Measured with a control plane
that publishes three overrides and then falls silent: the 100 ms cube ran at
host 351 ms (a full `max_wait` late) and saw **no** override, though three sat
retained at 0/40/80 ms. Shrinking `max_wait` only trades the latency for
emptiness — at `max_wait = 5 ms` against a 40 ms companion every window times
out, 5 ms late, delivering nothing every frame. And a same-time companion does
not help, because a successor must be *strictly* after the anchor. `OnWindow`
therefore pays off only when the companion is at least as frequent as the
anchor and reliably so; for "use the newest config at or before this frame",
`DrainAll` plus an operator-side `capture_time` comparison is both cheaper and
correct (no wait, and it delivers whatever is retained).

Budget also for degraded activations at stream edges (priming
consumes the first anchor; then `kInsufficientHistory`, and `kWindowTimeout` on
the last frame), and retention for `max_pending_windows * max_samples` on that
edge or the plan is rejected with `TEMPORAL_RETENTION_INSUFFICIENT` — that
bound is the *queue depth*, and it is a genuine constraint at speed: a 5 ms
companion against 100 ms cubes selects 22 per bracket (the inclusive middle
plus the outer brackets, with boundary samples appearing in both adjacent
windows), so `max_samples = 32` with four pending windows needs a 128-deep
queue. Sized correctly it is steady state with ~7 ms latency; under-sized it is
a compile rejection rather than a runtime death. Sizing
`max_pending_windows = 1` at frame rate starves with `kWindowPendingLimit` and
delivers activations out of order. When an input is absent, read
`Input<T>::selection().reason` before concluding anything — it distinguishes
"the producer had not run" from "the window gave up waiting".

Verified experimentally against HSDK 5 EA `v5.0.0.1`; re-check if a later SDK
changes activation-time, alignment, or window semantics.

### MatX where it fits; custom CUDA where it does not

Use MatX for element-wise maps, FFTs, and other expression-friendly ops. For
windowed order/sum statistics, fused multi-output stencils, or algorithms with
no MatX expression, write a fused kernel. **Benchmark before committing to a
MatX formulation that adds passes or scratch** — a convenient `conv2d` can be
several times slower than a fused stencil and still not cover all variants.

---

## Hot-path discipline

These rules generalize from CFAR and apply to any operator whose `compute()`
runs every frame.

### 1. Separate cold work from steady work

| Phase | Examples | Budget expectation |
| --- | --- | --- |
| Cold / first tick / config change | FFT plan, window coeffs, alpha LUT MC, first CUB temp alloc | May be ms…s; warn once; cache |
| Steady tick (unchanged geometry + config) | Stencil/FFT, emit tensors | Dominated by kernel + necessary memops |

Perf reports must show **cold** and **median/steady** separately. A "slow
operator" that is only slow on cold LUT rebuild is a different bug class from
one that allocates 20 MB every frame.

### 2. Never pay for unused paths

If a helper exists only to validate an optional external table, **do not call
it** on the path that does not use that table. CFAR's GO/SO steady path spent
~570 ms/tick scanning reachability for `max_alpha_lut_index()` even when only
`pfa` was active and the result was unused. Fix pattern:

- Gate expensive validation behind the feature that needs it.
- Prefer **closed-form** bounds (`max_training_cell_count`, interior-when-fits)
  over O(map) host scans.

### 3. Do not allocate dense device buffers every tick

`make_device_tensor` → `cudaMalloc` + `cudaFree` on last `shared_ptr` drop.
Allocating score/noise/mask/power (~tens of MB on large RD maps) every tick is
a steady-state tax, not "RAII hygiene."

**Pattern:** ping-pong (or N-buffer) workspace keyed by geometry
`(batch, …, trailing dims)`. Flip the slot each tick so emitted tensors are not
overwritten while a downstream consumer may still hold them. Reallocate only on
geometry / layout change.

Scratch that is *not* emitted can live in the same slot for that tick; outputs
that leave the operator need the ping-pong (or equivalent lifetime) guarantee.

Variable-length outputs (`detection_idx`) may still allocate per tick after an
exact count is known — that is usually fine if the buffer is small. Dense maps
are not.

### 4. Prefer device-side reduction/compaction over host walks

Host path anti-pattern: D2H full mask → CPU scan → H2D indices. That turns a
sub-millisecond kernel into multi-millisecond steady time and adds a hard sync.

Prefer: device mask → count (CUB reduce) → scalar D2H → exact alloc →
exclusive scan + scatter. Cache CUB temp storage across ticks
(`shared_ptr` to an incomplete type in the header is fine if the deleter is
type-erased in the `.cu`).

Preserve contract-visible ordering (e.g. raster-stable) in tests.

### 5. Cache costly artifacts; warn once on new configs

Use `ConfigCache` (or an equivalent keyed cache) for plans, windows, LUTs,
CUB state. Rules from `operators/common/README.md`:

- Initial parameter-driven build: silent.
- Override to an unseen key: warn once, build, cache.
- Alternating values must not thrash.

If a contract parameter implies caching but the implementation has nothing to
cache, do not pretend eviction works: record the no-op in `OPEN_ITEMS.md` as a
team question (keep-as-placeholder vs drop). Default to dropping a hollow knob
from both the contract and the code once the team confirms — an inert parameter
that implies behavior is worse than its absence (see the resolved
`max_cached_kernels` item, TEAM_BLOCKED L3). Re-introduce it as a real, wired
parameter if a variant ever needs it.

### 6. Keep host calibration off the fused kernel

Tiny host LUTs uploaded once per key beat porting special functions into the
stencil. Rebuild LUTs only when the key changes. If calibration is
Monte-Carlo-expensive, that is a **cold** cost — expose a production fast path
(precomputed table parameter) rather than hoping MC becomes free.

### 7. Instrument when diagnosis is stuck

When cold vs steady, or kernel vs shell, is unclear, add *temporary*
scoped timers around: alloc, H2D LUT, NCI, stencil, compact, and estimate
Holoscan graph/sink overhead outside `compute()`. Remove instrumentation before
land unless it is behind a debug flag. One breakdown run beats three speculative
kernel rewrites.

---

## Costly configuration and `[live]` overrides

### Single-tick override semantics

**Module `[live]` ports** join with `Exact{}` + `SafeDefault{}`, declared
unconditionally — an app enables an override by wiring the port and disables it
by leaving it unwired, which reports `kNotConnected` and reads as "no value this
tick". An override applies to **one tick**: `resolve_live_override` returns the
matched sample when there is one and the Params seed otherwise, and the operator
keeps no state between ticks. An app that wants an override to hold publishes it
every tick. Never treat a default-constructed SafeDefault payload as a value.

This holds under every trigger shape. `OnAligned`-triggered operators
(`SnapshotGatherOp`, `DetectionReductionOp`) were the one exception while the
SDK rejected an unwired non-trigger `Exact{}` entry under an aligned trigger;
that is fixed, so no operator carries an `expect_*` gate.

The notes below are the measured `LatestWithin` / `DrainAll` behavior (still
true of the SDK, and still used by unmigrated operators that omit live Inputs
from `contract()`). They are **not** the `[live]` contract.

**Override selection is causal on event time, one-sidedly.** A `LatestWithin`
port takes the newest value stamped at or before the frame being processed and
never one stamped after it, so a producer that stamps a value for a future
frame cannot have it applied early — verified against EA `v5.0.0.1`. `max_age`
is measured on the sample's `capture_time`, not on when it arrived: an override
published *now* but stamped 500 ms in the past is not selected under a 100 ms
window, and is selected under a 1000 ms one
(`tmp-app/selector_probe.cpp --backdate`). So the window is a statement about
sample time, which is what makes it meaningful under replay and simulated
clocks — and it means a producer that neglects to stamp `capture_time` gets
`kCaptureTimeAbsent`, not a fresh sample. The
converse has no protection: a value stamped at or before the frame *will* be
applied to it, whatever the producer meant, because the sample carries no
statement of which frames it belongs to. The producer's `capture_time` is that
whole statement, which is why an `OnClock` control operator must stamp
deliberately rather than by reflex.

**A `LatestWithin` override port drains, it does not collapse.** Each activation
consumes exactly one sample, and the next one selects the next-newest *retained*
survivor — so two overrides published between two ticks arrive on two
consecutive ticks, newest first, and the second one is *older* than the value
already applied. The applied value therefore does not move monotonically
forward in time, and an operator that logs or gates on the override's
`capture_time` will see it go backwards. Do not paper over that in the operator: a control plane
that publishes faster than the consumer ticks is the caller's problem to
notice, and swallowing it hides the real behavior. Both behaviors are
observable in `tmp-app/control_plane_app.cpp`.

**Sustained over-publication on a `LatestWithin` port kills the run.**
Measured on EA `v5.0.0.1` (`tmp-app/selector_probe.cpp`, a 25 ms control plane
against a 100 ms consumer): the activation consumes one sample, the other three
stay retained, and once the queue is full the *producer's* `emit` fails with
`kBackpressured` — which fails its `compute()`, so the session ends with
`RUNTIME_OPERATOR_CALLBACK_FAILED` / `RUNTIME_EXECUTION_LANE_PLAN_INVALID`.
With a 16-deep queue that is under 600 ms. There is no eviction of superseded
values and no silent drop: retention is a resource, and an over-published
override port exhausts it and takes the graph down. Queue depth only buys time.
This is one reason the module's `[live]` ports use `Exact{}` rather than a
wide `LatestWithin`. One unprobed escape hatch: the edge itself can be authored
`QueuePolicy::kDropOldest` (`ConnectionOptions` / `FlowConfigurator`), which the
SDK documents as retaining the newest depth and evicting rather than blocking —
a freshest-only mailbox at depth 1. If that holds it removes the failure without
changing the selector, so probe it before designing around the crash
([`TEMPORAL_CONTRACTS.md`](TEMPORAL_CONTRACTS.md)).

**`DrainAll` is the fix for a fast side stream.** It selects every retained
survivor as one batch, so the backlog is consumed in the activation that sees
it: no leak, no backwards steps, and the operator chooses which sample matters
(`selected_batch()` is random-access, in FIFO order oldest → newest, so
`back()` is the newest and a scan can pick the newest at or before the frame's
`capture_time`). Two constraints: an empty batch is a *successful* selection, so
`on_missing` is not applicable and pairing one with `DrainAll` is rejected as
`TEMPORAL_CONTRACT_INVALID`; and `receive()` is the wrong accessor — use
`selected_batch()` or `for_each_selected()`. Both selectors, same graph, in
`tmp-app/selector_probe.cpp`.

### Static vs live

Structural enums that select a kernel (`cfar_variant`, multiplexing scheme,
`angle_convention`) stay **static** — callers instantiate another operator.
Footguns that are "reasonable" (live `pfa`, live window size) are allowed with
warn-once + cache.

### When the contract says "cheap" but reality is not

If implementing a `[live]` param reveals that the faithful algorithm is
seconds of Monte Carlo (or similar):

1. Measure cold rebuild cost vs steady tick (do not conflate them).
2. Warn once on the slow path.
3. Propose a **fast path** that matches deployed practice (e.g. optional
   precomputed `alpha_lut` with precedence over online derivation).
4. Put the contract tension in `OPEN_ITEMS.md`; do not silently change
   `OPERATOR_API.md` without an explicit decision.

Precedence pattern that worked: external table if provided → else derive from
user-facing statistical param (`pfa`). Clearing the table returns to the derive
path (warn if that re-enters the slow path).

### Async / stale serving (optional)

If a rebuild is huge, serving the previous artifact for a tick while a
background rebuild runs can be correct for some operators — but only if the
contract allows stale thresholds and tests cover the stale/warning behavior.
Default to blocking correctness unless the contract says otherwise.

---

## Testing that catches real bugs

### Wire every output port

Holoscan will not tick an operator with an unconnected required output. Harness
tests must connect **all** outputs even if a test only asserts one. Leaving
`detection_mask` unwired can hide transport bugs.

### Assert the ports you care about

A test that only checks `detection_idx` will not notice a broken mask dtype
path. If the contract emits multiple related tensors, assert consistency
(`mask == score > 1`, idx lists exactly the true cells, etc.).

### Cover `[live]` Exact joins, not only build-time `Arg`

At least one mid-stream Exact match plus a following miss tick, which must
revert to Params. Stamp cube and override with the same producer-chosen
`capture_time`. Build-time args alone do not exercise `resolve_live_override`.
Pick the Params value so the revert is observable — a Params value that only a
miss tick reaches is otherwise never exercised at all.

### Float and double

Where math is precision-sensitive, assert agreement within a documented
tolerance — catches accidental `float` hardcoding in a double instantiation.

### Statistical tests vs operator MC

Monte Carlo **tests** validate operator Pfa/error rates. Monte Carlo inside the
**operator** for calibration is a separate, expensive design choice. Do not
conflate them in docs or reviews. If the operator uses MC calibration, tests
should still validate end-to-end rates; host unit tests should validate the
inverter in isolation.

### Tolerances are evidence, not decoration

If one variant needs a looser envelope (spatial correlation, etc.), document it
inline *and* in `OPEN_ITEMS.md`. Do not silently widen tolerances to green the
CI.

### Tautology guard

Simulators and closed-form references must stay independent of operator/MatX
code. Tests assert against analytic ground truth, not "pipeline round-trips
itself."

### SDK / dtype footguns in tests

`bool` (`kDLBool`) on HSDK 4.3 logs `Unsupported DLPack data type` and can
break Python consumers while C++ raw `cudaMemcpy` capture still works. If you
hit framework dtype gaps: record in `OPEN_ITEMS.md`, keep C++ tests honest
(raw device capture if needed), and do not paper over by changing the contract
dtype without a team decision.

---

## Performance characterization

The `performance/` tree is a **measurement harness**, not a pass/fail
benchmark suite. Goals:

1. Spot obvious pitfalls (per-tick alloc, host walks, accidental O(N) scans).
2. Judge distance to architectural speed-of-light (kernel-only vs full graph).
3. Catch regressions from incremental changes.

### Required coverage

Every `perf_<operator>` consumer **must** exercise `[live]` overrides when the
operator has any — not only build-time `Arg` variants. Match the CFAR pattern:

- Inline a mid-session override in the **same** graph run as steady-state
  compute (do not require a separate scenario loop just to touch the override
  port).
- Report `override_ms` / `override_delta_ms` on the compute row (peak latency
  over a short post-override window vs pre-override median).
- Cover each functionally distinct live param at least once in the default or
  `--full` matrix (cheap value flips and any costly rebuild paths). Claiming
  a fast path (`alpha_lut`, cached plans, …) while only timing the slow path
  is a harness bug.

**Process warmup (required).** Before the measured scenario loop, run one
discarded graph session (`perf_process_warmup.hpp`) so CUDA context init and
Holoscan's first Application tick are not billed to scenario #1's `cold_ms`.
Optionally prime each distinct kernel path the matrix will hit (e.g. each
policy / variant) so first-touch JIT does not land on a measured row either.
`--warmup` frames inside a scenario are separate: they exclude early frames
*within* a run from the steady median, and do not replace process warmup.

Operators with no `[live]` parameters are exempt from the override rule; note
that in the consumer header comment. Process warmup still applies.

### What to report

- Platform annotation (GPU, driver, build type).
- **Setup / instantiate** time vs **compute** time.
- **Cold** (first timed frame after config) vs **median / p90** steady.
- Live-override cost (`override_ms` / `override_delta_ms`) as above.
- Size ladder up to moderate real application shapes, not only micro sizes.

### How to diagnose a slow scenario

```text
Is cold slow or steady slow?
  cold → calibration / first alloc / plan build / first-touch
  steady → per-tick alloc, D2H, host scan, sync, or the kernel itself

Isolate the kernel (raw launch, no Holoscan) vs full graph tick.
  kernel fast, graph slow → shell / alloc / compact / scheduler
  kernel slow → algorithm / launch config / memory pattern

Check whether expensive helpers run on ticks that do not need them.
```

Do not start by rewriting the stencil when steady time is dominated by
`cudaMalloc` or a host reachability scan.

### Regression pairs

Where a slow convenience path and a fast production path both exist (e.g.
`pfa`-derived LUT vs precomputed `alpha_lut`), report them as a named pair
(`slow_*` / `fast_*`) so cold and override columns stay comparable.

---

## Open items and contract hygiene

### Operator-local `OPEN_ITEMS.md`

Create when the first team question appears. Keep items concrete:

- Context (what we observed)
- Impact (who breaks)
- Questions for the team
- Status (`open` / `landed with caveat` / resolved summary)

Resolve into `STATUS.md` settled decisions when the answer affects the module;
then archive or remove the section.

### Code ahead of contract

If implementation must proceed with a correct formula the contract has not
pinned yet, say so explicitly in `OPEN_ITEMS.md` ("code is ahead of the
contract") and either update the contract in a dedicated change or revert.
Never leave reviewers to discover divergence from diffs alone.

### Never change `OPERATOR_API.md` without explicit instruction

Agents: propose the text and wait. Humans: dedicated PR, called out as contract
change.

---

## Holoscan / module footguns (checklist)

- **Single-entry `TensorMap`** — port name is the contract; reject multi-entry
  maps with a clear error.
- **All outputs connected** in applications and tests.
- **`ensure_radar_arg_setters()`** in `initialize()` before
  `Operator::initialize()`.
- **Compact row-major** inputs this wave; reject non-contiguous if unsupported.
- **Incomplete types in headers** — prefer `shared_ptr` with type-erased
  deleter over `unique_ptr` to CUDA state defined only in `.cu`.
- **Emitted tensor lifetime** — ping-pong or equivalent; do not overwrite a
  buffer a consumer still holds.
- **Python bindings deferred** module-wide unless `STATUS.md` says otherwise —
  still avoid choosing dtypes that cannot survive a future Python consumer
  without recording the risk.
- **SPDX headers** on every new source/CMake/YAML/Docker file; markdown exempt.
- **`pre-commit run --all-files`** before push when the tool is available; CI
  enforces the same. If pre-commit is missing locally, say so in the PR / open
  items rather than claiming the checklist item is done.

---

## Appendix — CFAR lessons (concrete → general)

These are the incidents that drove the rules above. Use them as pattern
matching, not as CFAR-only lore.

| Symptom | Root cause | General rule |
| --- | --- | --- |
| GO/SO xlw "steady ~500 ms" with a ~0.07 ms kernel | Host `go_lut_reachability()` full-map scan every tick; also run on paths that did not need it | Gate unused work; closed-form bounds |
| Steady ~7 ms after LUT fix | Per-tick dense `cudaMalloc` + host mask compaction | Pool dense buffers; GPU compact |
| Steady ~0.8 ms after GPU compact | Still ~0.6 ms alloc of ~20 MB maps every tick | Ping-pong workspace on geometry key |
| OS `fast_cold` looks like LUT cost | LUT was precomputed; first-touch + Holoscan overhead | Break down cold into alloc / H2D / kernel / graph |
| Contract "cheap live `pfa`" vs multi-second GO/SO/OS | MC calibration is an implementation choice | Warn + fast-path table param; record contract tension |
| `detection_mask` errors in logs, tests still pass | `kDLBool` transport gap; tests only checked idx | Assert all related ports; OPEN_ITEMS for SDK gaps |
| MatX `conv2d` CFAR prototype | Extra passes + scratch; OS has no expression | Benchmark; fuse when MatX is a poor fit |

Exemplar code to read (patterns, not copy-paste requirements):

- Ping-pong dense workspace — `CFARDetectorOp` members + `compute()` in
  `operators/detection/cfar_detector_op/`
- Device compaction — `cfar_compact.cuh`
- Slow vs fast calibration — `alpha_lut` live override + warn-once on `pfa` MC
- Perf harness consumer — `performance/cpp/perf_cfar_detector.cpp`
- Open-item style — `operators/detection/cfar_detector_op/OPEN_ITEMS.md`

---

## Quick self-check before calling performance "done"

- [ ] Cold and steady reported separately on representative sizes
- [ ] Steady path does not `cudaMalloc` large dense outputs every tick
- [ ] No host O(map) scans or full-mask D2H on the steady path
- [ ] Costly config rebuilds are keyed, cached, and warn-once
- [ ] Expensive optional validation is not on the default path
- [ ] Kernel-only time measured once so graph overhead is visible
- [ ] Live fast paths are what the perf scenario actually exercises
