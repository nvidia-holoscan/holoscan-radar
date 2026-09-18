# Applications — Open Items

Deferred, agreed-upon work that spans the demo applications. Each item
records the intent and the reason it was deferred, so it can be picked up
without re-litigating the design. Remove items when they land (note the
commit in the PR).

## 1. GUI visual refinement

**Status:** open — low-priority polish; agreed to defer until core signal
chain is stable.

The HSDK 5 dashboards across the standalone SISO, SIMO, and MIMO applications are
functional but unpolished. Candidate improvements:

- Layout and panel sizing (consistent margins, panel proportions).
- Color-map choices for RD maps and cluster overlays.
- Label / legend clarity (axis tick density, units).
- Window title and headless vs. GUI mode indication.

No specific design has been chosen; track against the `radar_app` viz helpers
in `applications/common/radar_app/` so improvements land once and propagate to
all apps.

## 2. GUI dynamic range should be more static

**Status:** open — affects all Holoviz RD-map panels.

The RD-map color scale currently auto-ranges per frame (or per a short rolling
window), causing the display to "breathe" as targets and noise floor fluctuate.
A fixed or slowly-adapting dynamic range would give a more stable visual
reference.

Candidate approaches (not chosen):

- Fixed `[vmin, vmax]` parameters in the YAML config, passed through to the
  Holoviz image widget.
- Percentile-clipped running statistics (e.g. 1st–99th percentile over a
  configurable history window) so the scale adapts slowly rather than
  frame-by-frame.
- Operator-level `DynamicRangeOp` that normalizes the RD map before
  visualization, decoupling the scale policy from the Holoviz widget.

Whichever approach, the dynamic-range policy should be configurable via YAML
and shared across apps via `radar_app` so it does not need to be re-implemented
per demo.

## 3. HDF5 ADC replay source

**Status:** open — shared application input option; simulator sources remain
the default.

Each application should be able to select recorded HDF5 ADC data instead of
the analytic simulator without changing its downstream processing graph. Add
a reusable source under `applications/common/radar_app/` that reads framed ADC
tensors, validates the application-specific shape and complex dtype, uploads
each frame into a device Tensor pool, and emits the same `adc_cube` contract
and cadence/timestamps as the corresponding simulator source.

The application-owned YAML boundary should select `simulator` or `hdf5` and
carry only replay concerns such as file path, dataset path, end-of-file policy,
and optional frame range. Operator parameter blocks remain unchanged. SISO, SIMO, MIMO-TDM, and MIMO-DDM must each validate the recorded tensor against
their existing named ADC axes before the graph starts; file metadata must not
silently override waveform, array, schedule, or DDM-code configuration.

Before implementation, define and document:

- the HDF5 dataset layout and complex-number encoding;
- fixed-shape versus per-frame datasets and supported precisions;
- stop, loop, and bounded-frame behavior at end of file;
- timestamp/cadence behavior needed by Exact side inputs;
- HDF5 build, container, and package dependencies;
- focused reader tests with small deterministic fixtures plus one smoke per
  application family.

This remains a GPU application path: host file I/O is expected, but emitted
ADC tensors are CUDA-accessible and there is no CPU signal-processing fallback.
