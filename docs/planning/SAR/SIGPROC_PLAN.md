# Holoscan Radar Module — SAR Build Plan

## Purpose

What lands in the first SAR wave, in what order, where the code goes, and what it runs on. The contract lives in [OPERATOR_API.md](OPERATOR_API.md) and the chains in [WORKFLOWS.md](WORKFLOWS.md); the data is in [INPUT_DATA.md](INPUT_DATA.md); the assertions are in [TESTING.md](TESTING.md). This document is the sequencing and the application.

There is no FMCW analogue of this file — the FMCW wave tracked its build order in [STATUS.md](../STATUS.md) → *Up next*. SAR gets a dedicated one because the wave has a genuine dependency structure and a scope boundary worth writing down once, rather than re-deriving from a claims table.

## Status: accepted, implementing

**The contract was accepted on 2026-08-07 and the gate that held back operator code is lifted.** R1 is closed; build against the phases below.

Two conditions came with acceptance and apply to everything here. Revising the contract as implementation teaches us things is expected, but **every revision is approved by the developer in the chat session before it is made** — SAR-scoped ones included — then tagged in place and logged in [OPERATOR_API.md](OPERATOR_API.md) → *Contract changes since approval*. A revision reaching **beyond SAR** clears the same gate, is declared as beyond-SAR when asked, and gets a row in the module-wide [TEAM_BLOCKED.md](../TEAM_BLOCKED.md). If a phase below turns out to need a contract change, that is a stop-and-ask, not a thing to reconcile afterwards.

**One open item, and it lands early.** R10 (the metadata key namespace) survived acceptance because it is a decision about FMCW taken in a SAR document, and it may not be resolved in code. It is on the critical path: `CPHDReaderOp` writes the `stream_boundary` key in phase 1 and the app-owned control operator reads it in phase 2, which is how mode 1 knows the collection has ended. Until the team answers, use the SAR tier — `holoscan.radar.sar.stream_boundary` — behind a single named constant, which keeps either answer a one-line change and commits FMCW to nothing. Note that [OPERATOR_API.md](OPERATOR_API.md) still spells the key at the module level, since that is what D22 proposed and R10 has not ruled; the constant carries the interim value rather than the document being edited to match.

## Wave one

**Wave one is everything inside the single-look complex image formation box, plus three writers.** Nine operators:

| Operator | Group | Reference in hand | Role |
| :--- | :--- | :--- | :--- |
| `CPHDReaderOp` | `io` | yes | the only ingress; every test and demo runs through it |
| `SARRangeCompressionOp` | `sar` | yes | phase history → range profiles |
| `PhaseCorrectionOp` | `sar` | n/a | applies a per-pulse phase; the second half of D20 |
| `ApertureManagerOp` | `sar` | no | aperture membership and emit decisions; no kernel, no GPU |
| `BackprojectionOp` | `sar` | yes, offline only | the anchor |
| `AutofocusOp` | `sar` | **no** | PGA; per-pulse phase-correction vector + corrected image |
| `CPHDWriterOp` | `io` | partial | pairs with the reader for a round-trip test |
| `SICDWriterOp` | `io` | partial | the SLC product |
| `TensorDumpOp` | `io` | n/a | intermediate inspection while the rest is built |

Per-pulse phase-center positions are an **application responsibility** — no library operator computes them. On the CPHD path they come out of the reader; on a live platform the app supplies them. See [OPERATOR_API.md](OPERATOR_API.md) → *Two deliberate non-operators*.

### Two things this scope buys beyond the obvious

**Reading and writing CPHD in the same wave gives a round-trip test.** Read a collection, write it back, compare. That validates the reader's field interpretation without needing an independent implementation to diff against, which is otherwise a hard thing to get confidence in. `sarpy` on the Python side ([INPUT_DATA.md](INPUT_DATA.md) → *Policy*) supplies a second opinion the round trip alone cannot. **No first-ship demo forces it, so it is scheduled into the Products slice** — see *Work no demo forces* — meaning the wave still gets the round trip, just not while the reader is at its newest, which is when it was worth the most.

**`TensorDumpOp` earns its place early.** It is what makes intermediate stages inspectable while the rest is being built, so it is infrastructure for the wave rather than a nicety at the end of it. On the critical path it is also the mode 1 demo's first sink, since the formed image is reviewed as an offline PNG before any Holoviz work begins.

## First-ship scope

**Applications, recipes and public demos ship for modes 1–5 only** — spotlight, video SAR, tiled strip, continuous strip, independent images. That is [WORKFLOWS.md](WORKFLOWS.md)'s common products plus the general case. Modes **6–9** (wide swath, sub-aperture looks, circular, tomography) stay expressible in the API and lose nothing by waiting; what they need is app packaging and recipes, not contract or operators.

**Autofocus is in wave one but is not a sixth mode.** `AutofocusOp` and `PhaseCorrectionOp` ship as operators, exercised as an opt-in configuration on mode 1 per [WORKFLOWS.md](WORKFLOWS.md) Figures 2 and 3. The default spine is autofocus off.

The scope split is about *packaging*, not capability: nothing in modes 6–9 needs an operator or a parameter that modes 1–5 do not already force us to build. Mode 6 and mode 9 need `N` formation instances fanned out in `compose()`, mode 7 needs sub-aperture windows cut from a mode-1 collect, and mode 8 needs a circular trajectory in the recipe. All three are application and data work on a finished operator set.

## Reference availability and where assertion effort goes

Availability of a reference implementation informs **sequencing and risk, not scope** — every operator in [OPERATOR_API.md](OPERATOR_API.md) is contracted whether or not code exists for it. What it does determine is how much of [TESTING.md](TESTING.md)'s effort each operator deserves, because an operator with nothing to diff against carries its correctness entirely in its tests.

| Operator | Reference | Dominant risk | Assertion weight |
| :--- | :--- | :--- | :--- |
| `CPHDReaderOp` | yes | field interpretation | high — round trip plus batching edges |
| `SARRangeCompressionOp` | yes | little; closed-form checkable | low |
| `PhaseCorrectionOp` | n/a | the `sense` sign, which is unverifiable from data | low — but assert it against an injected error of known sign |
| `ApertureManagerOp` | no | bookkeeping off-by-ones | medium — but CPU-only, so cheap to cover |
| `RangeReferenceMultiplyOp` | no | replica and output-length conventions | medium |
| `RVPRemovalOp`, `BandEqualizationOp` | no | **sign and reference conventions** | high, and **not satisfiable synthetically** |
| `BackprojectionOp` | yes, offline only | **contract, not math** | high — see below |
| `AutofocusOp` | **no** | the algorithm itself | **highest in wave one** |
| `MultilookOp` | no | low; statistical assertion needs a clutter field | medium |
| `GeocodeOp` | no | datum and projection conventions | medium |
| `CPHDWriterOp`, `SICDWriterOp` | partial | metadata field population | medium — independent-reader checks |
| `CRSDWriterOp`, `SIDDWriterOp` | partial | as above | deferred with their seams |
| `TensorDumpOp` | n/a | none | low |

Three consequences worth stating rather than re-deriving:

**Wrapping a reference implementation moves the risk rather than removing it, but it does not remove the math either.** `BackprojectionOp` carries contract risk — the coordinate-frame boundary, the `positions`-port conventions, the arithmetic-strategy selection — none of which a reference implementation validates, since a correct kernel fed a wrong-convention trajectory produces a plausible image. It also carries real numerical risk: the tuned kernels exist as portable CUDA C, so the work is *porting a selected two or three* and reproducing their accuracy, not calling a library and trusting it. Splitting retention into `ApertureManagerOp` deliberately moves the cross-tick accumulation risk out of this operator and onto one that can be tested without a GPU.

**`AutofocusOp` holds the most correctness risk in wave one.** It is written from scratch with nothing to diff against, and its failure mode is a subtly worse image rather than an error. `MultilookOp` and `GeocodeOp` are in the same position but fall in a later wave.

**The deramped-branch operators are evidence-gated, not merely reference-free.** Real raw data is what pins their sign and reference conventions, and a synthetic test generated under the same assumption as the implementation validates the assumption into existence rather than the code. That is a sequencing constraint on the branch, not a reason to leave it out of the contract.

## Build order and delivery phases

Two orderings are in play and they are not the same thing. **Operator build order** is dependency and risk. **Slices** are deliverables that pair operators with the application and the data they need to be exercised. No slice blocks on a later one, and each is a state the repository can sit in with a working demo.

**The wave grows by doing the minimum for the next working application** (developer direction, 2026-08-07). Each slice below delivers one demo and stops; work is scheduled against the demo that first *forces* it, not convenience or de-risking. Phases 0–2 apply that rule to mode 1; later milestones apply it to modes **5, 2, 3 and 4** in turn. **Mode 5 moved ahead of mode 2 on 2026-08-25** (developer direction): same per-aperture graph as mode 1 with more than one image, so the cheapest next demo. **2026-08-27:** modes 1 and 5 wire `ApertureManagerOp` with the spotlight YAML row (`max_aperture_pulses: 0`, `emit_interval_pulses: 0`, `emit_mode: reset`). Mode 2 wires the retention row (`mode02.yaml`).

Two consequences follow, and both are deliberate. Cheap, de-risking work gets deferred when no demo needs it yet — the `CPHDWriterOp` round trip is the case where that costs something real. And a few items are forced by **no** first-ship demo at all; those are named in *Work no demo forces* below and scheduled explicitly, rather than being left to drift to whenever someone notices them.

The numbered phases are the mode 1 build-up, where there is no application to name until phase 2. After that the slices are named for the demo they deliver, because a phase number and a mode number sitting side by side invited exactly the confusion the skipped "phase 5" was guarding against.

### The critical path

**Mode 1 needs three library operators, not nine**, and historically `ApertureManagerOp` was not among them. A spotlight has one aperture and evicts nothing, so what the manager uniquely buys — retention for a sliding window — mode 1 never uses. [OPERATOR_API.md](OPERATOR_API.md) → *Running without a manager* still covers that case. **The demo now wires the manager anyway** (2026-08-27) with `max_aperture_pulses = 0` / `emit_mode = reset`, which is the table's "one image over the collection" row: pass-through pulses, D22 → *emit-and-reset*. Mode 5 is the same row plus a `CollectionBoundaryStagerOp` that feeds the next lattice into the manager and dr / f0 into `BackprojectionOp` on `collection_boundary`.

The wiring is correspondingly thin, which is what makes the path short. `BackprojectionOp` requires `range_profiles`, `positions`, `reference_position` and a *connected* `control`. `weight` stays unconnected and defaults to `+1` for every pulse; `sample_points` stays unconnected, so the lattice comes from the `sample_geometry` parameter; `initial_image` is unused. `interpolation = linear` and `geometry_change = reset` are defaults, as is the NVRadar LUT-vec `backend`, so mode 1 sets none of them.

| Phase | Operators | Application | Data ([INPUT_DATA.md](INPUT_DATA.md)) |
| :--- | :--- | :--- | :--- |
| **0** | `CoordinateFrame` / `SampleGeometry` / `LFMWaveformDescriptor` descriptors (all three new — D24 renames nothing), the `dsp/` grid-expansion helper, `operators/sar` + `operators/io` CMake scaffolding | — | simulator: point-target `mode01` recipe only |
| **1** | `CPHDReaderOp` (**landed contract-complete, not at the Atlanta subset** — see below), `TensorDumpOp` (**landed contract-complete**), `SARRangeCompressionOp` (**landed contract-complete**) | — | generated `mode01` + `mode01_ci4` / `mode01_ci2` |
| **2** | `BackprojectionOp` — grid lattice, single batch, default NVRadar LUT-vec | `sar_formation` + `mode01`, plus the control operator; **PNG checkpoint signed off 2026-08-24**; **Holoviz landed 2026-08-25** | Atlanta pulled whole, 1° frame via `pulse_range`; generated `mode01` / `mode01_smoke` for CI; MatX golden still a later parity check |
| **Mode 5** | same per-aperture graph as mode 1 with `ApertureManagerOp` (`max_aperture_pulses = 0`, `emit_mode = reset`); `[live]` geometry at each collection boundary via the manager + app stager for dr / f0 | mode **5** YAML; `pull_cphd.py`; `CPHDReaderOp.loop` / optional `playback_rate` / `duty_cycle`; `pulses_per_tick: 1024` (Ohio ~22 ticks); optional `dump: true` npy (off by default; Holoviz holds last emit); `EventBasedScheduler` (shared pool, one worker per operator) | six unique-IARP Umbra CPHDs **settled 2026-08-25** (Panama, Melbourne, Bingham, Kourou, Buenos Aires, Intel Ohio One). CI stays generated. Does **not** wait on Capella, retention, or the Houston subsetter |
| **Mode 2** | `ApertureManagerOp` retention, `emit_mode = continue`, `evict`; `BackprojectionOp` signed `weight` | `mode02.yaml` (Houston 1.0 m, 6.8 km, `lattice.max_side: 6144`, ~2.18 Hz emit, collect-vs-wall lag, `loop: true` with reset at wrap); signed-weight accumulation tests | Houston 1.0 m live / 0.5 m on disk (subsetter landed) |
| **Mode 3** | per-tile geometry staging (the reader's `CI4` + `AmpSF` + second-minor-version work landed in phase 1) | mode **3** YAML | Capella tiled-strip collect; **the two keys re-verified first** |
| **Mode 4** | `BackprojectionOp` scroll accumulation | mode **4** YAML; ring-wrap test — **first-ship modes 1–5 complete** | Capella continuous-strip subset |
| **Autofocus** | `PhaseCorrectionOp`, `AutofocusOp`; `emit_mode = reform` | AF opt-in YAMLs on mode 1 | injected-phase recipes |
| **Products** | `SICDWriterOp`, plus the scheduled items from *Work no demo forces* | demo polish, performance harness | pull tooling hardened: key re-probe self-check, useful failure on a moved object |
| **Later** | — | modes **6–9** packaging: `compose()` N-instance fan-out, circular recipes, sub-aperture looks | recipes for 6–9 |

**The demo is signed off in two stages, and the first one gates the second.** Phase 2 dumps the formed image through `TensorDumpOp` and renders it to a PNG offline, and *that image is reviewed and agreed correct before any Holoviz work starts*. Rasterization, layout and live display are effort spent making a correct image presentable; spending it before knowing the image is correct risks debugging formation through a display pipeline. **The PNG checkpoint closed 2026-08-24** (developer agreed the Atlanta image is correct). **Holoviz landed 2026-08-25**: `visualize: true` on `mode01.yaml` opens a greyscale log-magnitude window (same 99.8-percentile / 30 dB window as the PNG) and holds it after the single emit. Smoke YAML stays `visualize: false`.

### What mode 1 defers, and where each rejoins

Everything pulled out of phases 0–2 attaches to the demo that first forces it. Two of these are worth checking against intuition, because the obvious guess is wrong.

- **The Capella reader paths were forced by mode 3, not mode 2 — and were built in phase 1 anyway.** Houston is Umbra `CF8` with no `AmpSF`, the *same* decode path Atlanta already forced, so the video SAR demo needs no reader work; `CI4`, `AmpSF`, the second CPHD minor version and the travelling-`SRPPos` case all ride on the two Capella collects. The rule therefore scheduled them two demos out, and the [checklist](../NEW_OPERATOR_CHECKLIST.md) overrode it: a slice never clears that gate, so the reader went to full contract in phase 1. See *the minimum-per-demo rule has one standing exception* below. The unverified-key work is unaffected — open item 3 in [INPUT_DATA.md](INPUT_DATA.md) still gates **mode 3** and nothing before it, being about the data rather than the reader.
- **Signed `weight` is forced by mode 2**, because retention with eviction is what gives `BackprojectionOp` a `−1` to apply. It arrives with `ApertureManagerOp` rather than as part of the formation operator's own build-out.
- **`ApertureManagerOp` is the whole of the mode 2 slice.** It stays independent of `BackprojectionOp` and CPU-only, so it remains parallelizable earlier if someone has time; the rule schedules it, it does not forbid it.
- **Mode 5 is forced by nothing mode 1 did not already build, plus a pull script.** N independent spotlights on the same shortened spine; `collection_boundary` is *emit-and-reset* with geometry staged on that tick. **Pulled forward 2026-08-25** so it is next, not waiting on Capella. The old pin (reuse mode 1 and 3 collections) was packaging, not a dependency. **Demo data is settled:** six unique-IARP Umbra CPHDs (Panama, Melbourne, Bingham, Kourou, Buenos Aires, Intel Ohio One), downloaded by `sar_formation/pull_cphd.py`; CI stays generated. Atlanta stays on mode 1. The reader gained `loop` / `playback_rate` / `duty_cycle` (2026-08-26) so the demo can rotate, pace to collect time, or model a platform cooldown. See [INPUT_DATA.md](INPUT_DATA.md) → *Mode 5 — unique-IARP Umbra collects*.
- **Scroll accumulation is forced by mode 4** and nothing else.
- **The subsetter is forced by mode 2, not by phase 2.** It has two jobs — a pulse window and a contiguous central frequency subband — and mode 1 needs neither. `CPHDReaderOp.pulse_range` already selects an aperture out of a long collection, and a spotlight has no reason to narrow bandwidth, since full bandwidth *is* the finest range resolution the demo wants. So Atlanta can be pulled whole at 28.59 GiB and read with `pulse_range`, the reader seeking rather than loading it. Houston is where that stops working: 110 GiB, and the subband is a strided slice inside every vector rather than a contiguous byte range, which is why its pull is 208,753 separate range requests.

**Mode 2 landed 2026-08-29.** `mode02.yaml` streams the Houston 1.0 m subset with retention, signed `weight`, a ~2.18 Hz emit cadence, collect-vs-wall lag, and the APC look overlay. Formed side is `lattice.max_side: 6144` on a 6.8 km scene (0.5 m ran ~4× behind; 6800² was ~7%). Catch-up pacing held collect time. Capella stays a mode 3 pin. Mode 4 still follows mode 3, because scroll is the first thing that changes the accumulator.

### Work no demo forces

Three items are reached by no first-ship demo. Left to the rule alone they would drift indefinitely, so they are scheduled deliberately into the **Products** slice, and named here so the schedule is a decision rather than an oversight.

- **`CPHDWriterOp` and the round-trip test — the one deferral with a real cost.** The round trip validates the reader's field interpretation without needing an independent implementation, and it is now absent from exactly the stretch where the reader is newest. The substitutes are `sarpy` reading back what the simulator wrote, and the MatX golden at the end of phase 2 — which catches a reader error but localizes it poorly. If reader bugs actually bite during modes 1–3, pull this forward; that is the trigger to watch for.
- **The three non-default `backend` values.** `mixed` is the default and the one the golden comparison runs against, so no demo distinguishes them. They belong with the performance harness, which is the thing that can actually tell them apart.
- **Points-mode `sample_points` geometry.** Modes 1–5 all use the uniform lattice. It is first genuinely needed by mode 9, in the deferred packaging wave.

Also deferred and uncontroversial, because a named milestone forces each: `SICDWriterOp`, `AutofocusOp`, `PhaseCorrectionOp` and the performance harness.

### Dependency reasoning

These are constraints; the slices above are a schedule that satisfies them *and* the minimum-per-demo rule. Where the two disagree the rule wins, which is why the writers sit later than dependency alone would put them.

- **Group scaffolding and `CPHDReaderOp` come first.** Stand up `operators/sar/` and `operators/io/`, then the reader. It gates everything else, because without it there is no data to test with.
- **The simulator's point-target recipe is on the path, ahead of real data.** Not for CI, which is a later benefit, but because bringing up a backprojector against a known answer is much cheaper than against a 3 GiB collect whose conventions are themselves unverified. A wrong-convention trajectory produces a *plausible* image, so a fixture where the target's pixel is predicted analytically is the difference between a bug you can see and one you have to reason about.
- **`TensorDumpOp` earns its place here** rather than at the end: it is how intermediates are inspected while the rest is built, and in phase 2 it is also the demo's first sink.
- **`SARRangeCompressionOp`** is a windowed transform with a closed-form check. Low risk, and it makes the first half of the chain real. *Landed contract-complete 2026-08-10, and the closed-form check was worth more than "low risk" suggested:* predicting the whole Dirichlet profile in phase — rather than checking where the peak is — is what makes an inverted transform, a missing range `fftshift`, a missing leading `ifftshift`, an `(N−1)/2` centring and a wrong `fft_norm` visible, and every one of those passes a peak-position test. The risk column was right that the math is easy and silent about the fact that four distinct ways of getting it wrong all look correct. The leading `ifftshift` (IFFT DC = stored sample `N/2`) landed 2026-08-24.
- **`BackprojectionOp`** is the anchor, and the bulk of the work. Sub-order within it: a uniform `SampleGeometry` lattice over a single batch of pulses first, then signed weights and the accumulator's reset semantics, then explicit lattices, then the remaining arithmetic strategies. Getting a correct static image before adding statefulness means an accumulation bug cannot hide behind a formation bug. Only the first of those four is on the critical path.
- **The control operator is trivial, and the contract is what makes it trivial.** `CPHDReaderOp` stamps the boundary on the messages of the batch that *is* the collection's last — it looks ahead using the header's pulse count — and *emit-and-reset* resets **after** the emit, so the closing batch lands in the image. The operator therefore watches for the key and issues one *emit-and-reset* on that same tick. Its one wiring requirement is reachability: it must be fed from the reader to see the metadata at all, so give it an edge from the reader rather than assuming forwarding.
- **The subsetter lands with the mode 2 slice**, the first demo whose collection cannot be read whole. It is still not late polish — every demo streams a local file and never reaches the network, so it blocks a demo rather than trailing one. See *Layering*.
- **`ApertureManagerOp` is independent of `BackprojectionOp`** and cheap — no kernel, no GPU — so it can be written and fully tested in parallel, then wired in to turn the static formation into a streaming one.
- **`SICDWriterOp` needs a formed image**, so it cannot precede backprojection. It brings the NITF container dependency with it.
- **`AutofocusOp` and `PhaseCorrectionOp` depend only on an image existing**, so they are *independent of `SICDWriterOp`* — the ordering between the two is resourcing, not dependency, and the schedule spends it on autofocus first because it carries the wave's highest correctness risk. Autofocus additionally wants `emit_mode = reform`, which needs the retention the mode 2 slice builds, so it cannot precede that either.
- **Autofocus is no longer blocked** on the grid-conjugacy question, which closed as **D20**, which makes `AutofocusOp` an estimator restricted to conjugate geometries and `PhaseCorrectionOp` the stage that carries its per-pulse estimate into a formation on any geometry. Write `PhaseCorrectionOp` first — it is a broadcast multiply, and having it in hand means the autofocus tests can inject a known phase error through the real operator rather than through test scaffolding.

**The mode 3 milestone is where the real collections earn their keep.** The four pinned collections in [INPUT_DATA.md](INPUT_DATA.md) — two Umbra, two Capella — were chosen so that between them they force every reader path: `CF8` and `CI4` sample decoding, `AmpSF` present and absent, two CPHD minor versions, per-file PVP field discovery (the field set and `NumBytesPVP` both vary), frozen versus travelling `SRPPos`, `SC0`/`SCSS` constancy validation, the Tx/Rcv phase-centre convention, and `pulse_range` subsetting. A from-scratch fixture set would have let every one of those be skipped silently.

Phase 1 was written to implement only the subset Atlanta needs, with mode 2 adding none, which made the rest of that coverage a mode 3 obligation rather than something the reader had — a reader that reads one collection is not a reader that reads CPHD. It was the plan's sharpest instance of the minimum-per-demo rule biting: the reader would look finished two demos before it was. **It is also where the rule was overridden**, for the reason the next section gives.

### The minimum-per-demo rule has one standing exception

*Added 2026-08-09, from the reader.* The rule schedules by demand and the [new-operator checklist](../NEW_OPERATOR_CHECKLIST.md) gates by completeness, and on the reader they pointed opposite ways: the rule said `CI4`/`AmpSF`/TOA/multi-file wait for mode 3, the checklist said none of that can be called done while they are missing. **The checklist wins on the operator's own completion, the rule keeps everything else.** Concretely:

- An operator may **land** as a slice, and should, whenever a demo needs part of it. That is the rule working as intended and nothing here changes it.
- An operator is **not checklist-complete, and is not claimed as done in [STATUS.md](../STATUS.md)**, until its full `OPERATOR_API.md` surface is implemented. Closing that gap is scheduled work in its own right, not a demo's dependency.
- So the question "what do we build next" is still answered by the next demo. The question "is this operator finished" is answered by the contract, and the answer is no until every port, parameter and documented mode is there.

*`TensorDumpOp` landed under the same reading on 2026-08-10, and cost nothing to do so:* mode 1 needs `npy` at `every_n: 1`, and `raw`, `max_files` and the `[live]` override port are a few dozen lines each. Where the reader's completion was an argument, this one was not — which is the ordinary case the exception should produce.

Deciding it on the reader was cheap and worth recording: the deferred paths were four bounded decode and enumeration cases, and closing them took all 229 surveyed public collections from 115 readable to every one. Every Capella file needed `CI4` **and** `AmpSF` together, so the slice boundary happened to fall exactly where half the credential-free CPHD world stops being readable — a bad place to leave an ingress operator sitting for two demos while everything downstream is built on it.

The Capella half of that coverage — every `CI4` and every `AmpSF` path — rides on two keys that are **identified but not yet re-verified**, open item 3 in [INPUT_DATA.md](INPUT_DATA.md). Re-probe and transcribe them before the mode 3 slice starts, not during it.

## Demo application

**One executable**, `applications/sar_formation/`, with a YAML per shipped mode — the [`fmcw_mimo`](../../../applications/fmcw_mimo/) pattern rather than an executable per mode. Modes 1–5 are three chain configurations of one spine, so separate executables would be the same `compose()` copied five times with different parameters.

*This supersedes the earlier `applications/sar_backprojection/` naming.* The app forms images in several modes; only one of them is a plain spotlight backprojection.

| Choice | Rationale |
| :--- | :--- |
| Single executable | Modes 1–5 are three chain configurations of one spine |
| A smoke and a demo YAML per shipped mode | Same graph; the difference is `cphd_path` and scale |
| Autofocus as extra YAMLs | Orthogonal to mode, off by default |
| Defer modes 6–9 packaging | N-instance, circular and looks demos are not in the first ship |

The spine is `CPHDReaderOp` → `SARRangeCompressionOp` → `ApertureManagerOp` → `BackprojectionOp` → sinks, with positions bypassing range compression. The app does no metadata setup for D22 — metadata is on by default and `BackprojectionOp` sets its own `kUpdate`. `compose()` has three shapes in the first ship: the single-formation path (modes 1, 2, 3, 5), scroll wiring (mode 4), and the optional Figure 2 / 3 autofocus splice on mode 1. The N-instance fan-out that modes 6 and 9 need is the fourth, deferred.

**Mode 1 runs `CPHDReaderOp` → `SARRangeCompressionOp` → `ApertureManagerOp` → `BackprojectionOp`.** The manager is configured as the spotlight row (`max_aperture_pulses = 0`, `emit_mode = reset`): pass-through, D22 → *emit-and-reset*. Running without a manager remains a sanctioned configuration; this demo does not use it. **Mode 5 is the same spine** plus `CollectionBoundaryStagerOp` on `range_profiles` (next lattice into the manager, dr / f0 into BP). Modes 2 and 3 turn on retention.

### Mode to YAML

Autofocus off. First-ship modes in bold.

| Mode | Manager | Formation | What the app supplies |
| :--- | :--- | :--- | :--- |
| **1 Spotlight** | `max_aperture_pulses = 0`, `emit_mode = reset` (or an app control operator) | reset accumulation | one geometry; end of stream via reader metadata |
| **2 Video SAR** | retention `N`, `emit_mode = continue` | overlapping apertures | geometry; `evict` where the app owns the window |
| **3 Tiled strip** | per-tile reset | reset on geometry step | geometry per tile, staged through the manager |
| **4 Continuous strip** | optional, cadence only | `geometry_change = scroll` | lattice on `v`; no `weight` |
| **5 Independent images** | same spotlight row as mode 1; stager for live lattice / dr / f0 | reset | geometry per aperture |
| 6 Wide swath | per sub-swath instance | **N** formation instances | zero-weight foreign bursts |
| 7 Sub-aperture looks | as mode 5 or mode 2 | same | fixed geometry |
| 8 Circular | as spotlight | same | circular trajectory, in the collection |
| 9 Tomography | shared control and weights | **N** formation instances | one lattice per height plane |

Figure 2 autofocus, opt-in on mode 1: `emit_mode = reform` with retention, and the conjugate lattice on the auxiliary instance only.

### Layering

The three-tier layering the FMCW apps use, per [AGENTS.md](../../../AGENTS.md) → *Building example applications*:

- **`cpp/sar_formation.cpp`** — `compose()` as the graph, plus a one-line `main` calling `radar_app`'s runner. Nothing else.
- **`cpp/config.hpp`** — `AppConfig : app::CommonRunConfig` plus `load_config`, parsing only the app-owned YAML sections. Operator parameters flow natively through `from_config()`; the app never parses operator blocks.
- **`cpp/custom_ops.hpp`** — app-specific operators: the image display bridge and **mode 5's `CollectionBoundaryStagerOp`**, which feeds the next lattice into `ApertureManagerOp` and dr / f0 into `BackprojectionOp` on `collection_boundary`. When the live-radar path lands in a later wave, the **phase-center trajectory computation goes here**, which is the concrete form of the "application responsibility" decision.

A `run_sar_app` runner joins `run_fmcw_app` under [`applications/common/radar_app/`](../../../applications/common/radar_app/). There is no `SimAdcSourceOp` equivalent: SAR ingress is always a file, because the simulator writes CPHD rather than feeding the graph in process.

**Every demo reads a local CPHD.** The app streams through `CPHDReaderOp` and never reaches for the network, so a public collection has to be on disk before the demo runs. What that costs depends on whether the whole file can be pulled. Atlanta can: 28.59 GiB is a large download but an ordinary one, and `pulse_range` selects the frame out of it, so **phase 2 needs no pull tooling at all**. Houston cannot: at 110 GiB the demo needs the ~7 GB subset the subsetter writes, one strided range request per vector. So the subsetter is a **mode 2** prerequisite — still a blocker on a demo rather than late polish, which is where the natural "tooling last" instinct would put it, but one demo later than this plan first placed it.

Phase 2's *demo* is the Atlanta collect: pull the 28.59 GiB file once, form the 1° closest-approach frame with `pulse_range`, dump the image, review the PNG. The generated `mode01` / `mode01_smoke` fixtures stay the CI / bring-up path (sidecar lattice, seconds to run). The MatX golden remains a later parity check, not the thing that gates the PNG. Bundling Atlanta-as-golden with the subsetter is what used to make the subsetter look like it blocked phase 2; it does not.

Holoviz display reuses the `radar_app` layout builder (`make_image_holoviz_op`, a single COLOR_LUT layer — the RD-map + cluster factory is the wrong shape). A complex image needs a magnitude-and-log-scale rasterization step the FMCW apps do not have; that lives in `custom_ops.hpp` as `SarImageToLutOp` (GPU `|z|`, box-average onto the Holoviz grid, then GPU percentile + uint8 LUT on that grid). The PNG path still windows the full dump. Mode 1 emits once, so `HoldLastImageOp` re-emits for `visualize_hold_s` or the window closes with the graph. Mode 2 uses the same formed-`|z|` window plus the timeline.

## File layout

Two new operator groups, following the existing per-group pattern (`add_holohub_operator()` per operator, one `add_subdirectory()` line in `operators/CMakeLists.txt`).

```text
operators/sar/
  CMakeLists.txt              # add_holohub_operator() per operator
  README.md                   # group overview, per the fmcw/ precedent
  OPEN_ITEMS.md               # group-level team-blocked detail
  sar_range_compression_op/
    CMakeLists.txt
    metadata.json
    sar_range_compression_op.hpp    # the contract: ports, params, class
    range_compression_stage.hpp     # the kernel
    sar_range_compression_op.cu     # setup/compute + explicit instantiation
    python/
  backprojection_op/
    ...                             # same five-part split
    backprojection_stage.hpp        # wraps the matx::experimental::sar_bp seam
    OPEN_ITEMS.md                   # control payload, aperture state, arithmetic strategy
  phase_correction_op/
    ...                             # broadcast multiply; no OPEN_ITEMS.md
  autofocus_op/
    ...

operators/io/
  CMakeLists.txt
  README.md
  cphd_reader_op/
  cphd_writer_op/
  sicd_writer_op/
  tensor_dump_op/
```

**Why `io` is a separate group.** The standard-format operators carry XML and NITF dependencies that the signal-processing operators do not. A separate group puts each behind its own `OP_<name>` gate, so a build that wants only the formation chain pays nothing for the format libraries. `TensorDumpOp` sits there too because it is a sink, not because it shares the dependency.

Shared, non-Holoscan helpers go in `operators/common/radar_common/` as usual: the `CoordinateFrame` descriptor beside the other descriptors, and the grid-to-point-list expansion helper under `dsp/`.

The application, with YAMLs flat in `cpp/` per the FMCW precedent:

```text
applications/sar_formation/
  CMakeLists.txt
  README.md
  cpp/
    CMakeLists.txt
    metadata.json
    sar_formation.cpp
    config.hpp
    custom_ops.hpp
    sar_formation_mode01.yaml          # demo
    sar_formation_mode01_smoke.yaml    # smoke
    ...                                # modes 02-05, same pairing
    sar_formation_mode01_af_fig2.yaml  # autofocus, opt-in
  python/
```

That is twelve YAML files for the first ship, more than any FMCW app carries flat. Keep them flat anyway to start; if it becomes unwieldy, moving to a `yaml/` subdirectory is a deviation from the precedent and should be proposed as one rather than done quietly.

## Testing tiers

Mirrors [TESTING.md](TESTING.md), with the data each tier draws on named.

| Tier | Input | Asserts |
| :--- | :--- | :--- |
| Unit | Inline tensors | Closed-form; no simulator involved |
| Integration | Generated smoke CPHD | Metrics against the analytic sidecar |
| Demo | Capped synthetic scene, or a public pull | Visual, and SICD geometry sanity |
| Validation | Real collections — the Atlanta 1° frame against the offline MatX golden | Conventions, and `backend` accuracy |

Smoke stays fully synthetic and generates in seconds; no CI job pulls public data by default. The validation tier is where the parity golden lives, and it is the only tier that answers "is this right" rather than "is this consistent with itself".

## Landing gate

Each operator follows [NEW_OPERATOR_CHECKLIST.md](../NEW_OPERATOR_CHECKLIST.md). Wave-specific additions:

- **Accumulation tests are part of the gate** for `BackprojectionOp`, not an afterthought: multi-tick equivalence, reset on geometry change, and emitted-buffer independence. This is a new test pattern in the module. An independent-re-formation golden now exists to diff incremental accumulation against — see [INPUT_DATA.md](INPUT_DATA.md) → *The offline reference path*.
- **Do not resolve the team-blocked items in code.** Two remain, indexed in [TEAM_BLOCKED.md](TEAM_BLOCKED.md): contract acceptance itself (R1) and the scope of the metadata key namespace (R10). An operator that quietly picks an answer is not checklist-complete. Closed since this list was written: the autofocus grid-conjugacy question (D20), the metadata plane (D21 — declined, with optional descriptor override ports offered instead), who announces a collection boundary (D22 — a source, as message metadata), and the control message's payload (D23 — the four values only).
- **The boundary announcement lands early and is easy to get wrong.** `CPHDReaderOp` must stamp the key on the batch that *is* a collection's last rather than on the first of the next. `ApertureManagerOp` reads it (algorithm step 7) and treats it like an explicit `control` request. The reader half belonged to the first slice, because mode 1's only emit depends on that flush — cheap to implement and cheap to get subtly wrong.
- **`CPHDReaderOp` validates per-vector PVP rather than trusting it.** Not a hypothetical: the pinned Houston collect carries two vectors out of 208,753 whose `TOA1`/`TOA2` are an order of magnitude wrong while their `SIGNAL` flag reads valid. Anything deriving a global quantity from an unguarded min or max over a PVP field gets a nonsense answer from a clean, public, otherwise unremarkable file.
- **The performance harness needs two of its required fields redefined for an accumulating operator.** The checklist and [OPERATOR_DEVELOPMENT.md](../OPERATOR_DEVELOPMENT.md) → *Performance characterization* were written against operators where one tick produces one complete result, which `BackprojectionOp` is not. Nothing about the gate is a numeric threshold — it is a measurement harness — but two prescribed report fields are ill-defined here and must be stated rather than improvised:
  - **`cold_ms`** ("first timed frame after config") has two distinct meanings once an accumulator exists: the first tick of the process, and the first tick after a reset. Report both, named separately, and report **time-to-emit** for a full aperture alongside per-tick steady state, since that is the figure a caller sizes a system against.
  - **`override_delta_ms`** (post-override peak against pre-override median) assumes an override perturbs and resumes. Every `BackprojectionOp` grid parameter instead *resets*, discarding work, so the two windows are not comparable states. Measure and label the reset cost as its own quantity rather than reporting it as an override delta. That leaves no non-resetting live parameter to measure under the ordinary rule: `interpolation` is the only candidate, and with `linear` its sole implemented value this wave, an override of it cannot change state. Report it as not applicable rather than fabricating a measurement.
  - The size ladder is the pulses × pixels product rather than one tensor's dimensions, so pick a small matrix over both axes instead of a single sweep.
  - **`CPHDReaderOp` has a harness after all**, against this plan's original judgement that the readers and writers were out of scope because their cost is disk bandwidth rather than GPU time. The vocabulary turned out to exist: report bytes decoded and uploaded per second, sweep `pulses_per_tick` against sample format, and a source measures fine. It earned its place immediately — the ladder shows throughput collapsing at small batches, and attributing that found a flat per-tick device-allocation cost rather than the disk (see the reader's `OPEN_ITEMS.md` item 6). The writers stay out until one exists.
  - **So does `TensorDumpOp`**, and the table above rating its risk `low` with no math to get wrong was right about correctness and silent about cost. The question a harness answers here is not "is it fast" but "may I leave this wired in", and it needs one trick to be answerable: run the same scenario to a real file and to `/dev/null`, which separates the device-to-host copy from the filesystem. At 8 MiB the split is **1.1 ms of copy under 8.7 ms of `write()`** — the reverse of the expectation — and a tick that `every_n` skips costs ~10 µs, so a throttled dump may stay and an unthrottled one may not. Neither number was guessable from the code.
- **Do not depend on `matx::experimental::sar_bp`, and do not bump MatX to get it.** Confirmed rather than assumed: it is in the experimental namespace with an API subject to change, and the MatX bundled with the Holoscan SDK this module builds against lacks it entirely. It stays an offline golden-image path (item I1 in [OPERATOR_API.md](OPERATOR_API.md) → *Open items*), which is enough to give `BackprojectionOp` a cross-implementation reference without putting it in the build.
  - **Still the instruction as of 2026-08-12, but now by team decision rather than by assumed cost.** Moving the pin *alone* — without an SDK upgrade — was measured, and mechanically it is one `-DMATX_DIR=` with no CMake edit, because Holoscan turns out not to depend on MatX at all. What it actually forces is the whole module from C++17 to C++20, which builds and passes but reaches well beyond SAR. The team paused it until the Holoscan 5.0 conversion lands on `main`, since that conversion moves the pin anyway. Evidence is under I1 in [OPERATOR_API.md](OPERATOR_API.md) → *Open items*; do not re-derive it.
- **Contract changes get called out explicitly in the PR description** — the descriptor-set change, the precision-mode extension, and the new accumulation convention are all amendments to locked invariants.

## Later waves

Still in the contract, deferred on sequencing.

- **Modes 6–9 packaging.** Application and recipe work on a finished operator set: the `compose()` N-instance fan-out for wide swath and tomography, a circular trajectory recipe for mode 8, and sub-aperture windows cut from the mode-1 collect for mode 7.
- **Products.** `MultilookOp`, then `GeocodeOp`, which together unblock `SIDDWriterOp`. Geocode trails on datum and DEM decisions, and is optional besides — backprojection can form directly onto DEM-derived map points.
- **Alternate ingress.** `CRSDReaderOp`, `RVPRemovalOp` and `BandEqualizationOp` for the deramped uncompensated form; `RangeReferenceMultiplyOp` for direct-sampled; `CRSDWriterOp` at that seam. Evidence-gated — though less so than it looks, since Sentinel-1 Level-0 is freely available and is exactly the direct-sampled form.
- **SAR detection.** Its own wave, with its own clutter model. Not a reuse of `CFARDetectorOp`.

## Claim and coordination

Update [STATUS.md](../STATUS.md) when claiming a phase or a data slice, and again in the PR that lands it. Contract-change PRs call out their amendments as such. Leave R1 and R10 unresolved in code.
