# Pipeline Tensor Bounds (tentative)

**Status:** design proposal only. **Do not implement** until Dylan approves
(or amends) this approach.

Module convention for sizing HSDK 5 Tensor output pools. This is **not** an
HSDK 5 SDK type — Holoscan only exposes per-output
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
bigger library-wide ceiling makes OOM worse.

**Proposal (module convention, not an HSDK type):**

1. Add `holoscan::radar::PipelineTensorBounds` — **named** max extents
   (`max_channels`, `max_chirps`, `max_range_bins`, `max_doppler_bins`,
   `max_detections`, `max_az`, `max_el`, …) plus helpers that derive byte spans.
2. Fat-output ops take it as a **constructor capture**:
   `graph.op<Op>(name, bounds)`.
3. Each op’s `setup()` maps the fields it needs onto its outs’
   `max_byte_span` (tight to that app’s maxima).
4. Apps build **one** bounds object from config at graph-build time (runtime
   values OK). Params / live grids must stay ≤ those maxima.
5. Tiny fixed outs keep a small constexpr; no positional per-op arrays.

**Ask:** approve, amend (e.g. per-op named structs instead of shared bounds),
or reject in favor of another approach.

---

## Terminology

“Known before pools exist” means Holoscan cold declaration at operator
registration — **not** C++ compile-time of the `.so`. Graph-build / `main`
runtime config is fine.

---

## Mechanism (tentative)

```cpp
const auto bounds = PipelineTensorBounds::from_shapes(/* from YAML */);
auto beam = graph.op<BeamscanEstimatorOp<float>>("beamscan", bounds);
beam.set_params(BeamscanParams{...});  // grid must fit within bounds

// Op ctor stores bounds; setup():
spec.output(angle_spectrum, "angle_spectrum")
    .tensor_allocation({
        .max_rank = 3,
        .max_byte_span = /* detections×az×el×sizeof(T) from bounds */,
    });
```

```text
App max shapes from config
  -> byte_span helpers
  -> graph.op<Op>(name, bounds)
  -> setup() tensor_allocation (tight)
  -> cold materialize (slot_count × span per pooled out)
```

### Library pieces (after approval)

1. `operators/common/radar_common/sdk/` — `PipelineTensorBounds` holding named
   max extents, plus helpers such as `byte_span_rd_maps`,
   `byte_span_rda_cube`, `byte_span_angle_spectrum`.
2. Fat-output operators (RD maps, RDA cube, angle_spectrum, ADC cubes, …) take
   bounds as a constructor argument captured by `graph.op<Op>(name, bounds)`.
3. **Defaults policy:** do not ship a “fits every radar” MiB default. Example
   apps and perf ladders pass bounds derived from their known sizes. Prefer
   fail-closed if a fat op is constructed without bounds.
4. Optional Graph tighten remains available to shrink further; not the primary
   raise path.

### App UX

Apps set **one** bounds object from pipeline max shapes and pass it into each
fat op. Operators pick only the fields they need. Complex apps do not hand-tune
every Graph port name.

---

## Out of scope (unless redirected)

- Library-wide `MAX_BYTE_SPAN_MIB` as the primary fix
- Params-driven pool resize (SDK forbids changing `tensor_allocation` via
  `set_params`)
- Algorithm changes

---

## After approval — implement

- `radar_common` bounds type and helpers
- Constructor capture on fat ops; example apps and `perf_graph` wire-up;
  beamscan / phase_shift `xl` ladders
- Contract docs: `HSDK5_PORT.md`, checklist / `OPERATOR_API.md` note, `STATUS.md`
