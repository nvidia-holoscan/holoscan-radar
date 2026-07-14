# FMCW SISO

A SISO FMCW example application demonstrating the signal-processing + detection
+ clustering chain: a synthetic ADC source (one static clutter return + one
moving target, analytic dechirped point-target model) feeding

```text
FMCWRangeProcessorOp → ClutterFilterOp → FMCWDopplerProcessorOp
  → CFARDetectorOp → DetectionReportOp
                   → FeatureAssemblyOp → ClusterDBSCANOp → ClusterReportOp
```

The detection sink reports each CFAR detection in physical units via the
operators' `range_bins` / `velocity_bins` axis outputs. The source emits one
cube per tick and advances the targets between ticks, so the reported range
sweeps over the run. In parallel, `FeatureAssemblyOp` gathers a
`[range, velocity]` feature matrix from the CFAR cells (same axis outputs,
physical units) and `ClusterDBSCANOp` density-clusters it, so the handful of
adjacent cells one target lights collapse into a single cluster;
`ClusterReportOp` logs each cluster's size and centroid. An optional Holoviz
branch still renders the RD map from Doppler.

Launch via `./holohub run fmcw_siso`, or build directly:

```bash
cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
./build/applications/fmcw_siso/cpp/fmcw_siso            # default config
./build/applications/fmcw_siso/cpp/fmcw_siso my.yaml    # override config
```

The clutter filter suppresses the (stronger) static return, so CFAR's
detections are dominated by the moving target: ~42 m at ~+5.5 m/s. Additive
noise (`noise_snr_db`) is required — without it the CFAR noise estimate is
zero and no cells cross threshold.

## Configuration

The waveform, cube dimensions, run cadence, clutter method, CFAR parameters,
clustering parameters, targets, and noise are read from
[`cpp/fmcw_siso.yaml`](cpp/fmcw_siso.yaml) at startup. A copy is staged next to
the built executable and used by default; pass a different path as the first
CLI argument to override it. Each target is placed along the +x look direction
at a given `range` (m) and radial `velocity` (m/s, positive = receding) with a
linear `amplitude`.

The `clustering` block selects the FeatureAssemblyOp feature columns
(`features`, fixed to `[range, velocity]` for this app) and the DBSCAN
parameters (`epsilon` per-feature radius, `min_points`, `max_clusters`). It is
optional — the app defaults (a 3 m × 2 m/s neighborhood, `min_points: 2`) hold
when the block or any key is omitted.

## Visualization

Set `visualize: true` in the config to open a live Holoviz window. On the
left is the range-Doppler map (normalized log magnitude, turbo colormap;
range along x, Doppler along y), with the CFAR detections overlaid as a
crosses geometry layer **colored by DBSCAN cluster** (a categorical palette,
one color per cluster; detections that stayed noise are drawn in gray), so
the grouping is visible on the map alongside the textual cluster reports.

On the right (when `plots.enable` is set) two rolling time-series plots
stack: **range (m) vs time** and **velocity (m/s) vs time**. Each scrolling
plot shows every detection as a dot colored by its cluster and each cluster's
centroid as a distinct `+` marker in the same color, over a fixed Y axis (the
range / velocity bin extents) and a window of the last `plots.window` frames.
Over a run the moving target traces a rising range and a roughly constant
velocity. Set `plots.enable: false` to let the RD map fill the window.

This requires a GUI-capable host — a reachable display server and Vulkan. At
startup the app probes Holoviz directly; if no usable display is found
(headless host, or a stale `DISPLAY` with no server behind it), it logs a
warning and runs exactly as if `visualize` were `false`.

Set `headless: true` (with `visualize: true`) to render Holoviz offscreen
with no window — the startup display probe is skipped, letting the visual
pipeline run on a display-less host (e.g. CI) provided a Vulkan driver is
present.

A Python variant will be added when the Python bindings land (deferred to
a later phase).
