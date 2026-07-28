# FMCW SISO

A SISO FMCW example application demonstrating the signal-processing,
detection, and clustering chain: a synthetic ADC source (one static clutter
return + five moving targets, analytic dechirped point-target model) feeding

```text
FMCWRangeProcessorOp → ClutterFilterOp → FMCWDopplerProcessorOp
  → CFARDetectorOp → DetectionReportOp
                   → FeatureAssemblyOp → ClusterDBSCANOp
                        → DetectionReductionOp → ClusterReportOp
```

The detection sink reports each CFAR detection in physical units via the
operators' `range_bins` / `velocity_bins` axis outputs. The source emits one
cube per tick and advances the targets between ticks, so the reported range
sweeps over the run. In parallel, `FeatureAssemblyOp` gathers a
`[range, velocity]` feature matrix from the CFAR cells (same axis outputs,
physical units) and `ClusterDBSCANOp` density-clusters it, so the handful of
adjacent cells one target lights collapse into a single cluster.
`DetectionReductionOp` then reduces each cluster to one representative cell
(default `medoid` — the most-central real detection), and `ClusterReportOp`
logs each cluster's size, its mean centroid, and that representative. An
optional Holoviz branch renders the RD map from Doppler.

Launch via `./holohub run fmcw_siso`, or build directly:

```bash
cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
./build/applications/fmcw_siso/cpp/fmcw_siso            # default config
./build/applications/fmcw_siso/cpp/fmcw_siso my.yaml    # override config
```

The clutter filter suppresses the (stronger) static return, so CFAR's
detections come from the five moving targets. Their trajectories stay inside
the unambiguous range (~256 m) and velocity (~±39 m/s) envelope for the full
30 s run. Three are well separated in speed (−6.7 / +1.2 / +6.7 m/s); the other
two each shadow one of them through a close pass, so you can watch two objects
converge in range and Doppler and then separate again. Every radial sits on a
Doppler bin center (a multiple of the ~0.6083 m/s bin) so the velocity chart
reads as flicker-free lines.

Each close-pass partner runs 1.83 m/s (3 bins) faster than the target it
shadows — resolvable in Doppler, but inside the 2.0 m/s
`cluster_dbscan.epsilon`, so the pair can merge. One closes an 18 m gap and
coincides near t ≈ 10 s, the other a 27 m gap near t ≈ 15 s. SISO has no angle
axis, so at closest approach nothing distinguishes the pair and the two
clusters collapse into one — that merge, and the recovery afterwards, is what
the scene is there to show. Additive noise (`noise_snr_db`) is required —
without it the CFAR noise estimate is zero and no cells cross threshold.

## Configuration

The waveform, cube dimensions, run cadence, clutter method, CFAR parameters,
clustering parameters, targets, and noise are read from
[`cpp/fmcw_siso.yaml`](cpp/fmcw_siso.yaml) at startup. A copy is staged next to
the built executable and used by default; pass a different path as the first
CLI argument to override it. Each target is placed along the +x look direction
at a given `range` (m) and radial `velocity` (m/s, positive = receding) with a
linear `amplitude`.

With `run.meter_cadence: true` (default), the app warns when the source
inter-emit gap or DetectionReport end-to-end latency exceeds `frame_period`
(+ max(10 ms, 10% of period) slack) — i.e. when the pipeline is not keeping up with the
`PeriodicCondition` schedule — and prints a cadence summary at exit. Set
`run.meter_log_every: N` to also log e2e every N frames.

The per-operator blocks (`range_processor`, `clutter_filter`,
`doppler_processor`, `cfar_detector`, `cluster_dbscan`, `reduction`) are all
consumed natively through Holoscan's `from_config()` — every key is an operator
parameter. `FeatureAssemblyOp`'s `features` is the one exception: it is fixed in
the app code to `[range, velocity]` (the parameter is native-only by operator
contract, and the report / overlays assume that 2-column layout), so there is
no `feature_assembly` YAML block.

The `cluster_dbscan` block holds the DBSCAN parameters (`epsilon` per-feature
radius, `min_points`, `max_clusters`). The `reduction` block configures
`DetectionReductionOp` (`representative`: `medoid` | `peak` | `first`;
`score_method`; `feature_weights`). `medoid` is the centroid-like choice for
the `[range, velocity]` feature space (`peak` would rank cells by
range/velocity magnitude, meaningless without a score feature); set
`feature_weights` ~ `1/epsilon` so the medoid metric matches the DBSCAN
neighborhood. The representative is what the GUI marks as each cluster's
centroid. Each operator validates its own parameters at startup.

## Visualization

Set `visualization.enabled: true` in the config to open a live Holoviz window. On the
left is the range-Doppler map (normalized log magnitude, turbo colormap;
range along x, Doppler along y), with **every CFAR detection** overlaid as a
cross **colored by DBSCAN cluster** (a categorical palette, one color per
cluster; detections that stayed noise are drawn in gray), so the grouping is
visible on the map alongside the textual cluster reports.

On the right (when `plots.enable` is set) two rolling time-series plots
stack: **range (m) vs time** and **velocity (m/s) vs time**. These show one
point per cluster — each cluster's centroid, i.e. the `DetectionReductionOp`
representative — as a `+` marker colored by cluster, over a fixed Y axis (the
range / velocity bin extents, with numeric tick labels) and an X axis of the
last `plots.window` frames in seconds (0 = now). So the map carries the raw
color-coded detections and the plots carry the distilled per-cluster tracks.
Over a run each moving target traces its own range/velocity track on the
rolling plots (one colored cluster per target when DBSCAN separates them).
Set `plots.enable: false` to let the RD map fill the window.

This requires a GUI-capable host — a reachable display server and Vulkan. At
startup the app probes Holoviz directly; if no usable display is found
(headless host, or a stale `DISPLAY` with no server behind it), it logs a
warning and runs exactly as if `visualization.enabled` were `false`.

Set `headless: true` (with `visualization.enabled: true`) to render Holoviz
offscreen
with no window — the startup display probe is skipped, letting the visual
pipeline run on a display-less host (e.g. CI) provided a Vulkan driver is
present.

A Python variant with the same graph lives at
[`python/fmcw_siso.py`](python/fmcw_siso.py) (`config.py` colocated, same
layering as the C++ side — see [`python/README.md`](python/README.md)).
