# Holoscan Radar Module — SAR Build Plan

## Purpose

What lands in the first SAR wave, in what order, and where the code goes. The contract lives in [`OPERATOR_API.md`](OPERATOR_API.md) and the chains in [`WORKFLOWS.md`](WORKFLOWS.md); this document is the sequencing.

There is no FMCW analogue of this file — the FMCW wave tracked its build order in [`STATUS.md`](../STATUS.md) → *Up next*. SAR gets a dedicated one because the wave has a genuine dependency structure and a scope boundary worth writing down once, rather than re-deriving from a claims table.

## Wave one

**Wave one is everything inside the single-look complex image formation box, plus three writers.** Seven operators:

| Operator | Group | Reference in hand | Role |
| :--- | :--- | :--- | :--- |
| `CPHDReaderOp` | `io` | yes | the only ingress; every test and demo runs through it |
| `SARRangeCompressionOp` | `sar` | yes | phase history → range profiles |
| `BackprojectionOp` | `sar` | yes | the anchor |
| `AutofocusOp` | `sar` | **no** | PGA; phase-correction vector + corrected image |
| `CPHDWriterOp` | `io` | partial | pairs with the reader for a round-trip test |
| `SICDWriterOp` | `io` | partial | the SLC product |
| `TensorDumpOp` | `io` | n/a | intermediate inspection while the rest is built |

Per-pulse phase-center positions are an **application responsibility** — no library operator computes them. On the CPHD path they come out of the reader; on a live platform the app supplies them. See [`OPERATOR_API.md`](OPERATOR_API.md) → *Two deliberate non-operators*.

### Two things this scope buys beyond the obvious

**Reading and writing CPHD in the same wave gives a round-trip test.** Read a collection, write it back, compare. That validates the reader's field interpretation without needing an independent implementation to diff against, which is otherwise a hard thing to get confidence in.

**`TensorDumpOp` earns its place early.** It is what makes intermediate stages inspectable while the rest is being built, so it is infrastructure for the wave rather than a nicety at the end of it.

## Reference availability and where assertion effort goes

Availability of a reference implementation informs **sequencing and risk, not scope** — every operator in [`OPERATOR_API.md`](OPERATOR_API.md) is contracted whether or not code exists for it. What it does determine is how much of [`TESTING.md`](TESTING.md)'s effort each operator deserves, because an operator with nothing to diff against carries its correctness entirely in its tests.

| Operator | Reference | Dominant risk | Assertion weight |
| :--- | :--- | :--- | :--- |
| `CPHDReaderOp` | yes | field interpretation | high — round trip plus batching edges |
| `SARRangeCompressionOp` | yes | little; closed-form checkable | low |
| `RangeMatchedFilterOp` | no | replica and output-length conventions | medium |
| `RVPRemovalOp`, `BandEqualizationOp` | no | **sign and reference conventions** | high, and **not satisfiable synthetically** |
| `BackprojectionOp` | yes | **contract, not math** | high — see below |
| `AutofocusOp` | **no** | the algorithm itself | **highest in wave one** |
| `MultilookOp` | no | low; statistical assertion needs a clutter field | medium |
| `GeocodeOp` | no | datum and projection conventions | medium |
| `CPHDWriterOp`, `SICDWriterOp` | partial | metadata field population | medium — independent-reader checks |
| `CRSDWriterOp`, `SIDDWriterOp` | partial | as above | deferred with their seams |
| `TensorDumpOp` | n/a | none | low |

Three consequences worth stating rather than re-deriving:

**Wrapping a reference implementation moves the risk rather than removing it.** `BackprojectionOp` sits on a reference implementation, so its risk is *contract* rather than *math*: cross-tick accumulation, the coordinate-frame boundary, the `positions`-port conventions, and the precision-mode selection. None of those is something a reference implementation validates — a correct kernel fed a wrong-convention trajectory produces a plausible image. That is where the test effort goes.

**`AutofocusOp` holds the most correctness risk in wave one.** It is written from scratch with nothing to diff against, and its failure mode is a subtly worse image rather than an error. `MultilookOp` and `GeocodeOp` are in the same position but fall in a later wave.

**The deramped-branch operators are evidence-gated, not merely reference-free.** Real raw data is what pins their sign and reference conventions, and a synthetic test generated under the same assumption as the implementation validates the assumption into existence rather than the code. That is a sequencing constraint on the branch, not a reason to leave it out of the contract.

## Build order within wave one

Dependency and risk ordering. Nothing here is conditional on the step before succeeding, except where a genuine dependency is stated.

1. **Group scaffolding and `CPHDReaderOp`.** Stand up `operators/sar/` and `operators/io/`, then the reader. It gates everything else, because without it there is no data to test with.
2. **`CPHDWriterOp` and `TensorDumpOp`.** Immediately after the reader, so the round-trip test exists before anything depends on the reader being right, and so intermediates are inspectable for the rest of the wave. This is deliberately early: it is cheap and it de-risks everything downstream.
3. **`SARRangeCompressionOp`.** A windowed transform with a closed-form check. Low risk, and it makes the first half of the chain real.
4. **`BackprojectionOp`.** The anchor, and the bulk of the work. Sub-order within it: grid mode with a fixed aperture first (no accumulation), then the accumulator and its reset semantics, then points mode, then the precision modes. Getting a correct static image before adding statefulness means an accumulation bug cannot hide behind a formation bug.
5. **`SICDWriterOp`.** Needs a formed image, so it follows. Brings the NITF container dependency with it.
6. **`AutofocusOp`.** Depends only on an image existing, so it is *independent of step 5* — the ordering between them is resourcing, not dependency. **Blocked on the grid-conjugacy open item** resolving, since that decides whether it is a downstream stage, a feedback loop through `BackprojectionOp`, or a geometry-restricted operator.

The demo application is built alongside step 4 and finished after step 6.

## Later waves

Still in the contract, deferred on sequencing.

- **Products.** `MultilookOp`, then `GeocodeOp`, which together unblock `SIDDWriterOp`. Geocode trails on datum and DEM decisions, and is optional besides — backprojection can form directly onto DEM-derived map points.
- **Alternate ingress.** `CRSDReaderOp`, `RVPRemovalOp` and `BandEqualizationOp` for the deramped uncompensated form; `RangeMatchedFilterOp` for direct-sampled; `CRSDWriterOp` at that seam. Evidence-gated — though less so than it looks, since Sentinel-1 Level-0 is freely available and is exactly the direct-sampled form.
- **SAR detection.** Its own wave, with its own clutter model. Not a reuse of `CFARDetectorOp`.

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
    OPEN_ITEMS.md                   # emit policy, aperture state, precision modes
  autofocus_op/
    ...
    OPEN_ITEMS.md                   # grid conjugacy

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

## Demo application

`applications/sar_backprojection/`, following the three-tier layering the FMCW apps use:

- **`sar_backprojection.cpp`** — `compose()` as the graph, plus a one-line `main` calling `radar_app`'s runner. Nothing else.
- **`cpp/config.hpp`** — `AppConfig : app::CommonRunConfig` plus `load_config`, parsing only the app-owned YAML sections. Operator parameters flow natively through `from_config()`; the app never parses operator blocks.
- **`cpp/custom_ops.hpp`** — app-specific operators. For wave one that is the image display bridge; when the live-radar path lands in a later wave, the **phase-center trajectory computation goes here**, which is the concrete form of the "application responsibility" decision.

The CPHD-driven app is the default demo, since the simulator and the real collections both arrive that way. Holoviz display reuses the `radar_app` layout builder; a complex image needs a magnitude-and-log-scale rasterization step that the FMCW apps do not have, which is the one genuinely new piece of viz.

## Landing gate

Each operator follows [`NEW_OPERATOR_CHECKLIST.md`](../NEW_OPERATOR_CHECKLIST.md). Wave-specific additions:

- **Accumulation tests are part of the gate** for `BackprojectionOp`, not an afterthought: multi-tick equivalence, reset on geometry change, and emitted-buffer independence. This is a new test pattern in the module.
- **Do not resolve the team-blocked items in code.** The emit-decision mechanism, the autofocus grid-conjugacy question, and the aperture-weighting gap are indexed in [`TEAM_BLOCKED.md`](../TEAM_BLOCKED.md); an operator that quietly picks an answer is not checklist-complete.
- **The performance harness needs two of its required fields redefined for an accumulating operator.** The checklist and [`OPERATOR_DEVELOPMENT.md`](../OPERATOR_DEVELOPMENT.md) → *Performance characterization* were written against operators where one tick produces one complete result, which `BackprojectionOp` is not. Nothing about the gate is a numeric threshold — it is a measurement harness — but two prescribed report fields are ill-defined here and must be stated rather than improvised:
  - **`cold_ms`** ("first timed frame after config") has two distinct meanings once an accumulator exists: the first tick of the process, and the first tick after a reset. Report both, named separately, and report **time-to-emit** for a full aperture alongside per-tick steady state, since that is the figure a caller sizes a system against.
  - **`override_delta_ms`** (post-override peak against pre-override median) assumes an override perturbs and resumes. Every `BackprojectionOp` grid parameter instead *resets*, discarding work, so the two windows are not comparable states. Measure and label the reset cost as its own quantity rather than reporting it as an override delta, and cover the non-resetting live parameters (`interpolation`, `sinc_taps`) under the ordinary rule.
  - The size ladder is the pulses × pixels product rather than one tensor's dimensions, so pick a small matrix over both axes instead of a single sweep. `CPHDReaderOp` and the writers are **out of harness scope** for this wave: their cost is disk bandwidth, not GPU time, and the harness has no vocabulary for it.
- **Confirm the `matx::experimental::sar_bp` release status** before any code depends on it. It is in the experimental namespace with an API subject to change, and the MatX version bundled with the Holoscan SDK here lacks it entirely. Do not bump MatX as part of this wave.
- **Contract changes get called out explicitly in the PR description** — the descriptor-set change, the precision-mode extension, and the new accumulation convention are all amendments to locked invariants.
