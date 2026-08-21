# Temporal contracts: findings and decisions

How this module joins a **data plane** (ADC cubes and their derivatives) to a
**config plane** (`[live]` descriptor overrides) on HSDK 5. Everything below was
measured against EA `v5.0.0.1` with the probes in `tmp-app/` (indexed at the
bottom); nothing here is inferred from HSDK 4.x or from the preview docs alone.

Precedence: [`OPERATOR_API.md`](OPERATOR_API.md) remains the locked contract.
This document is the reasoning and the measured facts *behind* the temporal
clauses of that contract — read it before changing any operator's `contract()`,
and treat the "Still open" section as a hard stop rather than a
suggestion.

## Decisions at a glance

| # | Decision | Rejected instead |
| --- | --- | --- |
| 1 | **Coherence rides one shared event-time coordinate: the frame's transmit time.** The sequencer stamps `capture_time` on both the config sample and the frame trigger; the ADC source *copies* that stamp onto the cube rather than reading a clock; consumers join with `Exact{}`. | An `effective_from` / epoch field inside the descriptor; matching by generation ID inside `compute()`. |
| 2 | **The config plane publishes once per frame, not once per change.** Content changes stay irregular; publication is paced. | Publish-on-change plus sticky operator state as the *coherent* path (it remains the *degraded* path). |
| 3 | **`Exact{}` + `on_missing(SafeDefault{})` for `[live]` overrides**; unmatched ticks still run and take Params or the last latch when `sticky_*` is true. `Skip{}` remains available to an application that wants fail-closed, but operators in this module do not skip the cube. `LatestWithin` is not the `[live]` selector. | `LatestWithin{years{100}}` as the default sticky-override shape; `Skip{}` as the operator default (would drop cubes). |
| 4 | **`max_age` is a staleness bound, never a coherence mechanism.** No window value makes "the waveform that produced this cube" expressible. | Tuning `max_age` to approximate frame alignment. |
| 5 | **Optional override ports**: register unconditionally in `setup()`, gate only the `contract()` entry on an `expect_*` parameter (default false). `sticky_*` (default true) is a second static Param and is compute-side only. Interim only — there is no way today to make wiring the port the application's *only* obligation ([`HSDK5_PORT.md`](HSDK5_PORT.md) §11). | A parameter that changes the port set (`GRAPH_PARAMETER_PORT_SHAPE_CHANGED`); operator-side inference of its own wiring (no such API); making the port mandatory and wiring an idle producer (legal, parameter-free, but an idiom no Holoscan author would recognize). |
| 6 | **Queue depth is sized by frames in flight**, because acquisition and readout latency mean a cube arrives long after its config was published. | Treating retention as a time bound. |
| 7 | **`OnWindow` / `BracketedWindow` are not for config joins.** | Using a bracketed window to mean "newest config at or before this frame". |
| 8 | **`DrainAll` + operator-side `capture_time` comparison is the fallback**, for control planes that are fast or that cannot be paced — not the default. | `DrainAll` everywhere (it forfeits the fail-stop that `Exact` gives). |
| 9 | **Structural change requires a recompile.** | Mid-run `set_params` (a silent no-op on a running plan). |
| 10 | **A cube's `capture_time` stays a frame time.** | Stamping the cube with the config generation's own publication time. |

## The adopted shape

Topology (`control-plane-diagram.mmd`, realized in
`tmp-app/control_plane_app.cpp`): the control plane is the sequencer. It ticks
once per frame, and on each tick it stamps one `capture_time` — the instant the
frame begins transmitting — onto *both* the waveform it publishes and the flush
request that starts acquisition. The receiver copies that stamp onto the cube it
returns. Transmit time is therefore the only event-time coordinate in the graph,
and the DSP join needs no window at all:

```cpp
[[nodiscard]] holoscan::Contract contract() const override {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{adc})          // the cube sets the pace
      .input(waveform, holoscan::Exact{})        // the waveform that produced it
      .on_missing(waveform, holoscan::SafeDefault{});  // run; Params/latch
  return result;
}
```

Three properties follow, and they are the reason this shape was chosen over
every alternative below. The pairing is **coherent by construction** — not by a
tolerance that happens to be wide enough. A missing config is **visible to
`compute()`** via `is_missing()`; operators apply Params or the last latch
(`SafeDefault{}`) rather than dropping the cube. An application that wants
fail-closed can still author `Skip{}`. And it is **free**: no wait, no batch
scan, no `max_age` to tune, one scalar receive.

**The load-bearing precondition is causal order.** `Exact{}` under `OnEach`
resolves against the anchor sample's `capture_time`, so the companion must
already have been published when the driver arrives. That holds here because the
config for frame *k* is published at the *start* of frame *k*, while the cube for
frame *k* only lands after acquisition and readout complete — a full frame of
margin. It does **not** hold for a companion produced by a sibling branch of the
same frame, which is the case an earlier probe measured as failing with
`kNeverSeen` on every activation. Both facts are true; the distinction is
whether the companion causally precedes the driver. Any contract using `Exact{}`
must be able to state why it does.

## What we rejected, and what killed it

**Stamping the cube with the config's publication time.** Mechanically legal —
`capture_time` is whatever the producer says — but it overloads the coordinate.
Every frame of a generation carries an identical, non-advancing timestamp, and
because `max_age` is measured on `capture_time`, every downstream freshness
bound and latency figure silently becomes a statement about config age instead
of frame age. `Exact{}` is also consuming, so only the first cube of a
generation would match and every later one would fall to `on_missing`.

**`LatestWithin` with a practically-infinite window as the default.** It cannot
express coherence at all (it takes the newest value at or before the frame,
whatever that value meant), and it is actively unsafe when the control plane
outruns the consumer: each activation retires one sample, the rest stay
retained, and once the queue fills the *producer's* `emit` fails with
`kBackpressured`, ending the run with `RUNTIME_OPERATOR_CALLBACK_FAILED`. Queue
depth only buys time. It remains correct where stale is acceptable and the
publication rate is bounded below the tick rate.

**`OnWindow` + `BracketedWindow`.** All-or-nothing in a way that defeats config
use: `lookahead = 0` is not expressible (rejected as
`TEMPORAL_CONTRACT_INVALID`), so a bracket always waits for a strict successor,
and a same-time companion does not close it. When the wait expires the input is
*absent* with `kWindowTimeout` — the operator does not receive the lookbehind
samples it already holds — so an irregular config plane produces frames that are
both late by `max_wait` and unconfigured. Retention must also cover
`max_pending_windows × max_samples` or the plan is rejected.

**`DrainAll` as the default.** It is the right tool for a fast side stream and
it has no backpressure failure, but for a paced config plane it trades a
runtime-enforced pairing for a hand-written scan, and an empty batch is a
*successful* selection — so there is no fail-stop, and `on_missing` cannot even
be authored (`TEMPORAL_CONTRACT_INVALID`). Keep it for control planes that
cannot be paced.

**An epoch / `effective_from` field in the descriptor** (the L16 candidate).
Unnecessary once transmit time is shared: `capture_time` already *is* the
effective-from binding, and the runtime enforces it. Adding a redundant field
invites the two to disagree.

**Mid-run `set_params`.** Neither rejected nor applied on a running plan, and it
does not replay `contract()`. Anything structural is a recompile.

## SDK behavior this depends on

| Behavior | Status |
| --- | --- |
| `Exact{}` on a non-driver input resolves against the `OnEach` anchor's `capture_time` (docs only illustrate `OnClock`). | **Verified** — `control_plane_app.cpp`; negative control skewing one cube 1 ms reported `SelectionReason::kNeverSeen`. |
| `context.activation_time()` for a data-triggered operator is the anchor's `capture_time`, not wall-clock dispatch. Read the wall clock with `context.clock().now()`. | **Verified** — both columns printed side by side in `control_plane_app.cpp`. |
| `capture_time` is producer-supplied and freely chosen; an `OnClock` producer must stamp deliberately or event-time selection finds nothing (`kCaptureTimeAbsent`). | **Verified**. |
| `LatestWithin::max_age` is measured on `capture_time`, not arrival time. | **Verified** — `selector_probe.cpp --backdate`. |
| `LatestWithin` retires one sample per activation and backpressures the producer when full; `DrainAll` retires the whole backlog. | **Verified** — `selector_probe.cpp`. |
| `QueuePolicy::kDropOldest` on an edge (`ConnectionOptions` / `FlowConfigurator`) retains the newest depth and evicts instead of blocking; at depth 1 it is a freshest-only mailbox. Rejected with `BracketedWindow` (`TEMPORAL_WINDOW_QUEUE_POLICY_DROP_OLDEST`). | **Documented, not yet probed.** If it holds it is a second answer to the `LatestWithin` backpressure failure and worth revisiting decision 3's fallback. |
| An operator cannot discover its own connectivity: `contract()` sees only `params()` and its port handles, no operator-facing header exposes a connection query, and `SelectionReason::kNotConnected` is declared but never raised in the SDK sources. `on_missing` does not exempt an unwired declared input — it is still `TEMPORAL_INPUT_UNCONNECTED` at compile. | **Verified** — `wiring_inference.cpp --mismatch`. |
| The *application* can infer it: `Graph::view()` exposes frozen vertex/port/edge records by name, so "is this input the target of any edge?" is a lookup after wiring, and the `expect_*` parameter can be derived rather than restated. | **Verified** — `wiring_inference.cpp --wired` / `--unwired`. |
| An output port with no consumer fails its `emit` at runtime (`RUNTIME_OPERATOR_CALLBACK_FAILED`), so an application without a control plane must not instantiate one. | **Verified** — `wiring_inference.cpp`. |
| Whether a `capture_time` that no `Exact{}` activation ever matches is retired or retained forever. | **Unknown.** A single skewed frame recovered cleanly; a *sustained* skew may walk the queue to its depth. Probe before relying on `Skip{}` in a system that can drift. |

## Implications for operator contracts

For the agent applying this to `contract()` bodies and `OPERATOR_API.md`:

1. **Every `[live]` override** (POD now; descriptors when HSDK 5 admits them)
   uses `Exact{}` against the data trigger **only where `expect_*` is true**.
   The application must stamp the override with the frame's transmit time
   (or a copy of it) and publish it before the cube arrives. Document that
   next to the `[live]` tag. SafeDefault + retention is what happens on a
   miss — a violated pairing is silent unless the app publishes every frame
   (ephemeral) or inspects elsewhere.
2. **Do not put `LatestWithin` on `[live]` ports.** Stickiness is
   `sticky_*` in `compute()`, not a century-scale window.
   `DrainAll` remains the tool for an unpaced fast side stream that is *not*
   a `[live]` Exact join.
3. **Do not add descriptor-carrying output ports** to make a device echo work.
   Design invariant 2 sanctions exactly one (`VirtualArrayAssemblyOp.array_geometry`),
   and the paced-sequencer topology does not need a second one. If a hardware
   path later demands the device report what it radiated, that is a contract
   change with a team decision, not an implementation detail.
4. **Size queue depth from frames in flight** (acquisition + readout latency
   over the frame period), and say so where a depth is specified. It is not a
   staleness bound and does not scale with `max_age`.
5. **L16 and L17 are adopted** (Exact + `expect_*` / retention). Remaining:
   who owns the transmit stamp — see below.

## Adopted (L16 / L17)

- **L16.** Shared transmit-time stamp plus `Exact{}` on enabled `[live]`
  ports. No epoch field, no descriptor echo. `sticky_*` (default true) is
  compute-side; `on_missing` is `SafeDefault{}` so a miss still processes
  the cube (Params or last latch). Fail-closed `Skip{}` is not the
  operator default.
- **L17.** Per-port `expect_*` (default false) gates the `contract()` entry.
  Ports stay registered in `setup()`.

## Still open

- **Who owns the transmit stamp.** The paced sequencer is faithful for
  simulation and for a host-sequenced front end. Real hardware timestamps its
  own frames, which argues for the device reporting the config it applied. The
  consumer's contract is identical either way, so this can be decided later —
  but it should be decided, not defaulted into.

## Evidence index

| Probe | What it establishes |
| --- | --- |
| `tmp-app/control_plane_app.cpp` | The adopted topology end to end: paced sequencer, copied stamp, `Exact{}` join surviving a one-frame acquisition lag, generation-gated re-planning, and the 1 ms-skew negative control. |
| `tmp-app/selector_probe.cpp` | `LatestWithin` vs `DrainAll` under a fast control plane: backpressure death, batch drain, and `max_age` measured on `capture_time`. |
| `tmp-app/lookbehind_probe.cpp` | `OnWindow` latency, `lookahead = 0` rejection, timeout delivering nothing, and retention sizing. |
| `tmp-app/optional_live_port.cpp` | The optional-override pattern: unconditional `setup()`, parameterized `contract()` entry and selector bound, and mid-run `set_params` as a no-op. |
| `tmp-app/wiring_inference.cpp` | The same operator in two hand-written compositions, override wired and absent, as the reference for how an optional `[live]` port is declared. Earlier revisions of this probe also measured the results recorded above: `expect_*` derivable from `Graph::view()`, a declared-but-unwired entry rejected even with `on_missing`, a never-emitting producer accepted, input fan-in rejected (`GRAPH_INPUT_FANIN_UNSUPPORTED`), and an unconsumed output failing its emit. |

Re-verify all of the above if the SDK changes activation-time, selector, or
retention semantics; the EA is under active development.
