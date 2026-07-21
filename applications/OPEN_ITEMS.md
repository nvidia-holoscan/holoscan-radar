# Applications — Open Items

Deferred, agreed-upon work that spans the demo applications. Each item
records the intent and the reason it was deferred, so it can be picked up
without re-litigating the design. Remove items when they land (note the
commit in the PR).

## 1. GUI visual refinement

**Status:** open — low-priority polish; agreed to defer until core signal
chain is stable.

The Holoviz GUIs across SISO, SIMO, MIMO, and dense-SIMO demos are functional
but unpolished. Candidate improvements:

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
