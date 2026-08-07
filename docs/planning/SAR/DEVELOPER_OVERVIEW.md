# Holoscan Radar Module — SAR Image Formation, for Software Developers

## About this document

Onboarding for someone who will **implement or extend** this SAR stack: expert at software and Holoscan-style graphs, comfortable with basic SAR ideas (range, Doppler, synthetic aperture, complex imagery), **not** expected to know collection-mode jargon or which product names map to which pipeline knobs.

**Nothing described here is built yet.** This is a proposed design under review. Claims about what the API “supports” are design claims, not reports on working code.

It is a derivative of [WORKFLOWS.md](WORKFLOWS.md) (modes + wiring) with contracts from [OPERATOR_API.md](OPERATOR_API.md). Where this disagrees with those, **they win**. Regenerate from the prompt in WORKFLOWS.md → *Appendix*, do not hand-patch. The sibling [PRACTITIONER_OVERVIEW.md](PRACTITIONER_OVERVIEW.md) is the same scope stripped of software detail for external SAR reviewers.

**Read next after this:** [OPERATOR_API.md](OPERATOR_API.md) (the contracts in full — a proposal pending **R1**, not locked text, unlike the FMCW contract one level up), [SIGPROC_PLAN.md](SIGPROC_PLAN.md) (build order), [TESTING.md](TESTING.md), [../NEW_OPERATOR_CHECKLIST.md](../NEW_OPERATOR_CHECKLIST.md), [../OPERATOR_DEVELOPMENT.md](../OPERATOR_DEVELOPMENT.md).

---

## What we are building

A Holoscan **operator library** for monostatic, single-pol, **time-domain backprojection** SAR. Pulses stream in; complex images stream out. Wave one reads **CPHD** phase history and writes **SICD** (plus dumps). Later waves add live-radar ingress branches and display products (multilook / geocode).

| In scope (wave one+) | Out of scope / deferred |
| --- | --- |
| Backprojection onto caller-chosen sample points (including DEM / map grids) | Frequency-domain formation (PFA, range-Doppler, …) |
| Streaming apertures and sliding windows | Bistatic geometry (foreclosed) |
| Optional PGA autofocus with hard constraints | Polarimetry, InSAR, SAR detection as first-class products |
| CPHD → range profiles → image → SICD | Fused “one operator does everything” pipelines |

**Design invariants that matter when coding** (full list in repo `AGENTS.md` / OPERATOR_API): plain tensors on data ports; compile-time single/double precision; leading batch axes opaque; `[live]` overrides only where marked; no inventing new shared descriptor types without a contract change.

---

## Mental model: one graph, modes are configuration

There is **one** operator chain. Collection “modes” (spotlight, video SAR, strip, …) are **not** separate pipelines. They differ by:

1. Whether consecutive images **share pulses** (needs retention + signed-weight eviction).
2. Whether the **scene lattice** stays fixed, **steps** per tile, or **scrolls** with the platform.
3. How many **formation instances** run over one pulse stream (wide swath, tomography).

Almost everything that looks like a special mode is “independent images” with different geometry scheduling. The software consequence: implement `ApertureManagerOp` + `BackprojectionOp` correctly once; modes are YAML and wiring, not forks.

**The split has one asymmetry worth learning before you read the diagram.** Formation never decides when an image comes out, so *something* must tell it — that is the emit decision, and it is mandatory in every graph. `ApertureManagerOp` is the operator we ship to make it, and everything else it does is convenience. So the operator is optional and its central job is not.

```mermaid
flowchart TD
  classDef opt stroke-dasharray:4 3
  READ["CPHDReaderOp"] --> RC["SARRangeCompressionOp"]
  RC --> AC["ApertureManagerOp"]
  READ -->|"positions, reference_position"| AC
  AC -->|"range_profiles, weight, control"| BP["BackprojectionOp"]
  GEOM(("sample geometry")) -.-> AC
  BP --> SLC(["SLC → SICDWriterOp"])
  class AC,GEOM opt
```

Default product path: **autofocus off**, form onto ground/DEM geometry the app supplies. That is WORKFLOWS Figure 1. `ApertureManagerOp` is dashed as an **operator**, not as an edge — `control` is required whether or not this box exists, and in a graph without it a small app-owned operator occupies the same slot.

---

## SAR vocabulary → software meaning

Use this when reading mode names in WORKFLOWS or talking to radar folks.

| Term you will hear | Plain meaning | What changes in *our* graph |
| --- | --- | --- |
| **Aperture** | The set of pulses coherently combined into **one** image (not the whole collect) | Membership (`weight`) plus when to emit (`control`, required); both usually from `ApertureManagerOp` |
| **Spotlight** | Radar stares at one patch; usually one image at end | Per-aperture formation; one geometry; flush at end of stream |
| **Video SAR** | Same patch, images emitted repeatedly while collecting | Overlapping apertures: retain pulses, `weight = −1` to evict |
| **Stripmap / strip** | Platform flies past; ground scrolls under the beam | Tiled products (mode 3) or one scrolling product (mode 4) |
| **Tiled strip** | Strip delivered as separate scene files | Step `sample_geometry` each tile; reset accumulator; **seams** at joins |
| **Continuous strip** | One long image that grows with flight | `geometry_change = scroll`; no taper via `weight` |
| **ScanSAR / TOPS** | Beam hops between elevation sub-swaths (TOPS also steers in azimuth) | **N** formation instances; zero-weight foreign bursts; mosaicking absent |
| **Sub-aperture looks** | Split one dwell into several images of the same scene | Disjoint = independent images; overlapping = video-SAR mechanics |
| **Circular SAR** | Orbit around the scene (~360°) | Same as spotlight on a circular trajectory; PSF checks differ |
| **Tomographic / 3D** | Stack of height planes | **N** instances, one lattice per height; same pulses |
| **Range compression** | Fast-time matched filter → range profile | `SARRangeCompressionOp` |
| **Phase history** | Compensated deramped signal (CPHD’s native form) | Convergence point of all ingress branches |
| **SLC** | Single-look complex image | Primary output; `SICDWriterOp` |
| **DEM / ground grid** | Map or terrain sample locations | Usual `sample_points` / `sample_geometry` for the product |
| **Radar-geometry / conjugate grid** | Lattice whose along-track axis is Fourier-related to pulse time | Required for PGA estimation; not the default product |
| **Autofocus / PGA** | Estimate leftover per-pulse phase from a formed image | Optional; see *Autofocus* below — **off by default** |
| **Motion compensation** (classical) | Correct to a nominal straight track for FFT processors | **Not a stage here** — backprojection uses measured positions |

**Supported / deferred / foreclosed** in WORKFLOWS means “can the API express it,” not “is it scheduled for wave one.” Wave-one build order lives in [SIGPROC_PLAN.md](SIGPROC_PLAN.md).

---

## The operator chain (wave one)

Orientation only — contracts are in OPERATOR_API.

**Ingress.** `CPHDReaderOp` → `phase_history`, `positions`, `reference_position` (+ optional `sample_axis` for writers). Live radar and CRSD are later waves.

**To range profiles.** `SARRangeCompressionOp`. Optional later: deramp branch (`RVPRemovalOp`, `BandEqualizationOp`) or direct-sample (`RangeMatchedFilterOp`) into the same phase-history node.

**Aperture vs image (the important split).**

- `ApertureManagerOp` — optional operator, **one mandatory job**: drive `control`, which tells `BackprojectionOp` when an image comes out. Formation never decides that itself, so some operator does it in every graph. Everything else it offers is convenience — staging a geometry change to the aperture boundary, buffering pulses, deciding membership and its `aperture_window` taper, and turning the D22 boundary announcement into a flush. Of those, **retention is the only one an app cannot assemble from operators it already has**; the rest replace a counter or a window function. Emits `weight` (`+1` admit, `−1` evict) alongside `control`. Holds **no** image.
- `BackprojectionOp` — accumulates into an image. Holds **no** aperture. Only sums what it is handed, and emits only when `control` says to.

**The two edges into formation are not equally optional**, which is what decides how much you take on by omitting the manager. `weight` defaults to `+1` per pulse when unconnected, so an untapered aperture needs nothing. `control` is a **required connection**: an unconnected `control` describes a graph that can never emit an image and is rejected at initialization ([OPERATOR_API.md](OPERATOR_API.md) → *Build-time wiring validation*).

**Running without a manager** is legal in every mode, and is how mode 4 is normally built, since scroll leaves a manager nothing to do but the emit cadence. It transfers four things to the application. Three fail loudly; the second does not.

1. The `control` connection itself, caught at initialization.
2. **Reachability.** Your control operator needs at least one input fed, directly or transitively, by the source, because metadata arrives only through `receive()`. Wired as a pure source it never hears `holoscan.radar.stream_boundary` and the final aperture is silently lost. `ApertureManagerOp` cannot have this bug — its inputs are the pulses.
3. Emitting a geometry change **and** the resetting `control` that admits it on one tick; a geometry arriving without one errors rather than buffering. Scroll excepted.
4. Your own `weight` values if you want a cross-range taper, computed over the aperture's final membership.

Forgetting to wire `positions` / `reference_position` around range compression is the classic bug: shapes still look fine.

**After the image.** Wave-one writers: `SICDWriterOp`, `CPHDWriterOp`, `TensorDumpOp`. Later: `MultilookOp`, `GeocodeOp`, `SIDDWriterOp`.

Port-level diagrams: WORKFLOWS Figures 1–5.

---

## Cross-cutting mechanisms you must get right

### Cross-tick accumulation

Some operators integrate across Holoscan ticks and emit less often than they consume (`ApertureManagerOp`, `BackprojectionOp`, writers). Rules are in OPERATOR_API → *Cross-tick accumulation*:

- Emit and reset are separate; reset has a **phase** (before vs after this tick’s data).
- **One owner of reset** — never the accumulator guessing. Geometry changes travel **with** the reset that admits them.
- End-of-stream flush cannot use `Operator::stop()` (no `OutputContext`), so a finite source **announces** it as message metadata — `holoscan.radar.stream_boundary`, valued `end_of_stream` or `collection_boundary` (**D22**) — and whoever drives `control` translates that into a flush. `ApertureManagerOp` reads it itself. Metadata, not a port, because a port cannot stay aligned with its data across a stage that coalesces ticks, which the manager does.
- **One** thing this puts on the application, not two: if you own the control operator, feed it from the source — metadata arrives only through `receive()`, so a policy operator wired as a pure source never hears the boundary. There is no metadata setup to do: it is enabled by default, and `kUpdate` (needed because `kRaise` throws where a consumer takes several ports from one producer) is set per-operator by that consumer, `Operator::metadata_policy` taking precedence over the fragment default. `BackprojectionOp` warns at `stop()` if it is still holding an unemitted image, which is how you find out either way.
- The control message carries the four values and nothing else (**D23**): no aperture id, no sequence number, no reason code.

### `[live]` overrides

Parameters are build-time by default. Marked `[live]` may be overridden via a same-named non-blocking input. Costly overrides warn once and cache. Structural choices (`emit_mode`, `algorithm`, `geometry_change`, …) stay static — instantiate another operator to vary them. Where a port is wired **and** its parameter set, the port wins and a disagreement warns once.

That mechanism is also the answer to the writers' mirrored configuration (**D21**), so there is no descriptive metadata plane to look for. Three optional descriptor *outputs* — `CPHDReaderOp.coordinate_frame`, `SARRangeCompressionOp.range_window`, `BackprojectionOp.sample_geometry` — wire into the same-named `[live]` override on `GeocodeOp` and the writers, whose mirrors are `[live]` for that purpose. Wiring them is optional and mirroring by hand stays legal; the edge worth wiring is `sample_geometry` into `GeocodeOp`, where a disagreement misplaces the whole raster while producing a well-formed file.

### Application-owned inputs (no library operator)

1. **Per-pulse antenna phase-centre positions** (and `reference_position`) — CPHD reader supplies them on file path; live apps derive from nav. Wrong frame/units/time → plausible bad image, not a throw. Formation requires `coordinate_frame` (`CoordinateFrame`) with **no default** — same frame as `sample_points` / positions.
2. **Image sample geometry** — `SampleGeometry` descriptor or `(num_v, num_u, 3)` `sample_points`. Usually staged through the manager so a lattice change co-arrives with reset. Scroll mode and manager-free graphs send geometry straight to formation, which is where obligation 3 above lands on you.
3. **A `control` producer, wherever there is no `ApertureManagerOp`** — a small operator in `custom_ops.hpp` or in your live-sensor source. Not `CPHDReaderOp`, which declares no `control` output by design (**D22**): it announces boundaries on the metadata plane and leaves the emit vocabulary to whoever owns the aperture. Unlike items 1 and 2 this one is conditional, but it is a required *connection* — see the four obligations under *The operator chain*.

### `PhaseCorrectionOp.sense`

Static, **required, no default** (`subtract` vs `add`). Wrong sign doubles the phase error. Set explicitly; tests use `add` to inject known errors.

### Precision

I/O dtype is a compile-time template (single or double). `BackprojectionOp.precision_mode` is a **separate** internal-arithmetic strategy (spaceborne ranges need more than naive FP32). See OPERATOR_API notes and TESTING.

---

## Modes → what you configure

Nine named modes collapse to **three** software configurations:

| Configuration | Modes | Software essence |
| --- | --- | --- |
| **Per-aperture** | 1, 3, 5, 6†, 7 (disjoint), 8, 9† | Each pulse in one image; reset at boundary; manager optional |
| **Overlapping** | 2, 7 (overlapping looks) | Retain pulses; `weight −1` eviction; the one place manager retention **earns its keep** |
| **Scrolling** | 4 | `geometry_change = scroll`; translate lattice along `v`; no `weight`; a manager here is reduced to the emit cadence |

† Modes 6 and 9 are **N instances** of the per-aperture pattern (sub-swaths or height planes).

"Manager optional" throughout means the *operator*, never the emit decision: `control` is driven in every column, and only the identity of the producer changes.

Typical manager knobs (see OPERATOR_API table for full matrix):

| Goal | `max_aperture_pulses` | `emit_mode` | Notes |
| --- | --- | --- | --- |
| One image, AF off | `0` | `reset` | Pass-through; `control` at the announced boundary |
| Video / sliding window | `N` | `continue` | Incremental add/subtract |
| Tapered block aperture | `> 0` | `reform` | Replay full aperture at boundary |
| **DEM product + PGA (Fig. 2)** | `≥ aperture length` | **`reform`** | Required — see below |

Five of the six configurations in OPERATOR_API's table are a pure YAML edit with no application-owned ports wired at all, which is what D22's boundary announcement bought: only the application-chosen sliding window still needs an operator of your own, because only you know the count you put on `evict`.

Mode 3 seams and mode 4 “no taper / buffer depth” are **product physics**, not missing operators. Do not “fix” them in formation.

---

## Autofocus (implementers)

**Default: omit it.** Usual product is ground/DEM backprojection without AF.

**PGA** is wave one’s only algorithm. It needs an image whose along-track axis is Fourier-conjugate to slow time (radar-geometry lattice). A DEM/map product generally is **not** that.

**Figure 2 (DEM + PGA)** — supported only as:

1. `ApertureManagerOp`: `max_aperture_pulses > 0`, `emit_mode = reform`.
2. On the reform tick: aux `BackprojectionOp` (conjugate lattice) → `AutofocusOp` → `PhaseCorrectionOp` → product `BackprojectionOp`.
3. Same-tick DAG order so PC sees replayed profiles **and** `phase_correction`.
4. Product geometry through the manager; conjugate geometry on the **aux** instance only.

AF-off spotlight (`max_aperture_pulses = 0`) **cannot** do Figure 2 — nothing left to replay after estimation.

**Figure 3** — product already on a conjugate lattice: `AutofocusOp` after the single `BackprojectionOp`; no `PhaseCorrectionOp`.

PGA discards constant/linear phase (image shift). Conjugacy is **declared** (`azimuth_axis`), not checked.

---

## Pitfalls that look like “radar is hard” but are wiring/state bugs

- Positions not wired past range compression.
- `CPHDReaderOp.sample_axis` used as `range_bins` (shape-legal, image wrong).
- Geometry change without a resetting `control` (must error, not soft-buffer). Easy to hit the moment you drop the manager, since staging the change was its job.
- Dropping the manager and expecting its edges to drop with it. `control` is a required connection, so the graph fails at initialization — the loud one of the four obligations, and the reason to read them before assuming a manager-free chain is a subtraction.
- Manager resets while still emitting `−1` evictions for dropped retention (phantom subtractions).
- Incremental sliding window + azimuth taper (refused; use `reform`).
- Figure 2 autofocus on `emit_mode = continue` or scroll.
- Assuming `stop()` flushes the last aperture (it warns instead — see D22).
- Disabling metadata on an operator between the source and the control owner: the boundary announcement is dropped and the final aperture is lost. (You do not have to *enable* anything — it is on by default.)
- An app-owned control operator wired as a pure source: it can never hear the boundary, because metadata arrives only through `receive()`.
- Mirroring `sample_geometry` into `GeocodeOp` or `SICDWriterOp` by hand and letting it drift from formation. Legal, and the ordinary path, but the D21 override port removes the failure mode entirely.

---

## What is absent (so you do not invent it)

No `MotionCompensationOp` for straight-track phase (backprojection uses measured trajectory). No bistatic ports. No first-wave frequency-domain formation. No mosaicking / look-combination / SAR CFAR in this modality’s wave one. Detection lessons from FMCW do **not** carry over to SAR clutter statistics.

Team-blocked items that affect implementation: contract acceptance itself (**R1**) and the scope of the metadata key namespace (**R10** — whether `holoscan.radar.stream_boundary` sits module-wide or under a `sar` tier). Everything else you would have found here is decided: the control-message payload as **D23**, the end-of-stream announcement as **D22**, and the metadata plane as **D21**, which declined it. See [TEAM_BLOCKED.md](TEAM_BLOCKED.md); [../SDK_QUESTIONS.md](../SDK_QUESTIONS.md) now holds one upstream design question rather than anything blocking.

---

## Build slice

Wave-one operators and risk order: [SIGPROC_PLAN.md](SIGPROC_PLAN.md). Land against OPERATOR_API contracts; use [../NEW_OPERATOR_CHECKLIST.md](../NEW_OPERATOR_CHECKLIST.md) as the done gate.
