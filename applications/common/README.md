# Application-support library (`radar_app` / `python/`)

Shared, app-agnostic building blocks for the FMCW example applications
(`fmcw_siso`, `fmcw_simo`, `fmcw_dense_simo`, `fmcw_mimo`), for both the C++
side (`radar_app/`, this directory's original content) and the Python side
(`python/`, documented in its own section below — same layering, same
three-tier breakdown). It exists so those apps stay small and readable: each
app is mostly a graph-wiring `compose()` plus its own `AppConfig`, not a pile
of copy-pasted operators and scaffolding.

The applications are organized around three kinds of component:

1. **Graph / flow wiring** — building the operator graph in `compose()`. This is
   app-specific and lives in each application's own translation unit (e.g.
   `fmcw_siso/cpp/fmcw_siso.cpp`).
2. **App-specific operators** — custom to one application, not generalized. For
   the FMCW examples these are the MIMO synthetic sources and multiplexing
   simulators (`MimoTdmAdcSourceOp` / `MimoDdmAdcSourceOp`) and the dense-path
   bridges (`AnglePeakPickOp`, `CubeAoaLookupOp`, `TrailingIdxColumnsOp`), so
   they stay with the app.
3. **App-generalized operators / helpers** — reused across applications. These
   live here, in `radar_app`. The single rank-agnostic synthetic source
   (`SimAdcSourceOp`, rank-2 for SISO and rank-3 for SIMO) is shared here too;
   only MIMO, whose multiplexing needs a bespoke simulator, keeps its own.

## The config boundary (important)

Operator *parameters* are not a shared config layer: the signal-processing,
detection, clustering, reduction, and angle-estimation operators take their
configuration natively through Holoscan's `from_config()` (backed by the
`YAML::convert<>` specializations in `radar_common`). Their YAML blocks
(`range_processor`, `cfar_detector`, `cluster_dbscan`, `beamscan_estimator`, …)
are **never** parsed by the app.

The app-owned YAML sections — `waveform`, `run`, `cube`, `targets`,
`noise_snr_db`, `plots`, the `visualize`/`headless`/`skip_display_check`
toggles, and the array geometry (`rx_array`/`tx_array`) — *are* shared, via the
`radar_app/app/config.hpp` helpers. Each app derives its `AppConfig` from
`CommonRunConfig` and composes the `parse_*` helpers, adding only its own extra
sections.

An app may separately read an *operator* block purely to **mirror** a value into
a visualization or geometry grid — e.g. `beamscan_estimator.az_angles` /
`phase_shift_beamformer.az_angles` are read to size the boresight-view axes and
the dense az/el lookup. That read stays app-side and is commented as a
**read-only mirror**: the operator still owns the parameter via `from_config()`.

## Layout

```text
radar_app/
  app/config.hpp          CommonRunConfig + the app-owned YAML parse helpers
                          (parse_run_and_cube, parse_targets_*, parse_plots,
                          parse_viz_toggles, extract_grid, grid_extent, …)
  app/source.hpp          SimAdcSourceOp — rank-agnostic synthetic ADC source
  app/driver.hpp          resolve_config_path + run_fmcw_app (the shared main:
                          load config, display-probe downgrade, config(), run())
  app/config_path.hpp     resolve_config_path, shared by both drivers
  app/graph_driver.hpp    [EA HSDK5 ONLY] run_fmcw_graph_app + AppFrameBank —
                          the Graph-API counterpart of run_fmcw_app
  app/holoviz_layout.hpp  HolovizLayout / RightPanel + build_holoviz_input_specs
                          + make_rd_holoviz_op factory + panel presets
  report/cadence_meter.hpp  CadenceMeter — wall-clock source / e2e probe
  report/report_ops.hpp   DetectionReportOp (raw/clustered modes), NullSinkOp,
                          TensorMapDiscardOp
  viz/viz.hpp             RdChannelSliceOp (SIMO per-RX RD slice),
                          RdMapToImageOp, ClusterOverlayOp, RollingPlotOp,
                          BoresightViewOp (az/el scatter), the turbo LUT, the
                          cluster palette, and the overlay / plot layer-name
                          helpers
  viz/display.hpp         gui_available() — the Holoviz display probe
  viz/colormap.hpp        the Holoviz-free turbo LUT (kLutSize indices)
  viz/rd_map_to_image_op.hpp  [EA HSDK5 ONLY] Graph RdMapToImageOp (LUT image)
  viz/x11_rd_viewer_op.hpp    [EA HSDK5 ONLY] X11RdViewerOp — sized RD window
                          with axes, cluster-colored detections + rep
                          crosshairs, legend, clock, and the rolling plots
  viz/overlay_style.hpp   [EA HSDK5 ONLY] Holoviz-free cluster palette /
                          rank coloring / tick formatting for that viewer
  viz/x11_display.hpp     [EA HSDK5 ONLY] x11_display_available() probe
```

Two shared geometry builders (`make_ura_geometry` / `make_ura_rx_positions`
and `make_virtual_geometry`) and the `advance_scene` frame-stepper live one
layer deeper, in `radar_common/sim/` — they are pure model helpers with no
Holoscan/CUDA/Holoviz dependency.

Everything in `radar_app` is under `namespace holoscan::radar::app` and included
as `"radar_app/..."`.

`CadenceMeter` is an optional shared probe the demos wire into the ADC source
(`on_source_emit` after each cube emit) and the terminal report op
(`on_sink_done` / `DetectionReportOp.cadence`). It warns when source inter-emit
or end-to-end latency exceeds `frame_period` (+ max(10 ms, 10% of period) slack) and prints a
summary after `run()`. Toggle with YAML `run.meter_cadence` /
`run.meter_log_every`.

## Using it

`radar_app` is a header-only `INTERFACE` CMake target (operator methods are
in-class inline; free helpers and templates are marked `inline`), so it compiles
no objects of its own and carries only include paths + link dependencies
(`radar_common`, Holoviz, CUDA). An application links it with:

```cmake
target_link_libraries(my_app PRIVATE holoscan::radar_app ...)
```

A typical app is then: derive `AppConfig` from `CommonRunConfig`, compose the
`parse_*` helpers in a short `load_config`, wire the graph in `compose()` (using
`SimAdcSourceOp` and the shared report/viz operators), build a `HolovizLayout`
from a preset and pass it to `make_rd_holoviz_op`, and return
`run_fmcw_app<MyApp>(...)` from `main`. The app keeps ownership of the graph
edges — `radar_app` provides the operators, the config vocabulary, and the
layout data; the app decides how they connect and where they render.

---

## Python (`applications/common/python/`)

The Python demos (`fmcw_siso.py`, `fmcw_simo.py`, `fmcw_dense_simo.py`,
`fmcw_mimo.py`) mirror the same three-tier breakdown as the C++ side:

1. **Graph / flow wiring** — each app's `FmcwXApp.compose()`, in the app's own
   `python/fmcw_X.py`. Thin: construct operators, `add_flow`, done.
2. **App-specific operators / config** — colocated with the app in its own
   `python/config.py` (`AppConfig` + `load_config`) and, where the app needs
   truly app-specific operators, `python/custom_ops.py`. SISO and SIMO have no
   `custom_ops.py` — they use the shared `SimAdcSourceOp` unchanged; dense and
   MIMO do (dense: the angle bridges + `DenseClusterReportOp`; MIMO: the
   TDM/DDM ADC sources + `DdmReportVelocityBinsOp`).
3. **App-generalized operators / helpers** — reused across apps, here in
   `applications/common/python/`. Most of the viz/report operators are
   already bound from C++ (`radar_app`'s `ClusterReportOp`, `RollingPlotOp`,
   `BoresightViewOp`, etc. — imported straight from `holoscan.radar`); this
   directory covers what is Python-only.

Both `config.py` and `custom_ops.py` are imported bare (`import config`, `from
custom_ops import ...`) from the app's `fmcw_X.py` — this works because each
app runs as a script, which puts that script's own directory at
`sys.path[0]`, the Python analogue of C++'s colocated bare
`#include "config.hpp"  // NOLINT(build/include_subdir)`. Don't import these
modules by name from another process: `config` / `custom_ops` are not
qualified package names, so importing two apps' `config.py` in the same
interpreter would collide in `sys.modules`.

### Python layout

| Path | Role |
| --- | --- |
| `demo_config.py` | `CommonRunConfig` (mirrors `radar_app/app/config.hpp::CommonRunConfig`; every app's `AppConfig` inherits from it via Python dataclass inheritance) + the app-owned YAML parse helpers (`parse_common_run_config`, `parse_target(s)`, `parse_cube_dims`, `parse_noise`, `parse_waveform`, `parse_window_spec`, `parse_angle_convention`, `parse_tx_schedule`, `parse_ddm_code`, geometry builders, `grid_extent`) + `cluster_layer_name` / `make_view` + `port_tensor` / `tensor_to_numpy` |
| `sim.py` | The point-target simulator: `SimWaveform`, `PointTarget`, `SimScene`, `SimResult`, `simulate()`, `advance_scene()` — mirrors `radar_common/sim/point_target_sim.hpp` |
| `mimo_sim.py` | MIMO TDM/DDM scene + simulators (`MimoScene`, `derive_truth`, `simulate_mimo_*`, the DDM code helpers) — mirrors `radar_common/sim/mimo_sim.hpp`; reuses `sim.py`'s `SimWaveform` / `PointTarget` / `SimResult` rather than redefining them, same as the C++ header |
| `ops.py` | `SimAdcSourceOp` — rank-agnostic synthetic ADC source, mirrors `radar_app/app/source.hpp`; `TensorMapDiscardOp` — a Python-only addition (see caveat below) |
| `driver.py` | `resolve_config_path()` + `run_fmcw_app()` — mirrors `radar_app/app/driver.hpp` |

### The config boundary, Python-specific caveat

The config-boundary rule is the same as the C++ side (see above): operator
YAML blocks flow through Holoscan's `from_config()`, never manual per-app
parsing. All four Python demos follow this for every operator block **except
one**: `fmcw_mimo`'s selected dense TDM/DDM block, which stays a manual kwargs
dict (the split dense parsers).

The reason is a Python-only pybind11 quirk, not a design choice: Holoscan's
Python `Operator.__init__` applies `from_config()`'s `ArgList` (arrives as a
positional arg) **after** all keyword arguments, regardless of the order they
appear at the call site. In C++, `make_operator<Op>(name, from_config(...),
Arg("x") = resolved_value)` applies arguments strictly in the order given, so
a trailing explicit `Arg` overrides `from_config()`. In Python, that ordering
guarantee does not hold — `from_config()` always wins over a same-named kwarg.
This only matters when an app must inject a resolved value that shares a YAML
key with the block (MIMO's `ddm_code`, which the app may expand from
`generate_phase_codes` into explicit `phase_codes`); it does not matter when
the injected kwarg has no corresponding YAML key (e.g. `array_geometry` on
`BeamscanEstimatorOp`, `tx_positions` / `rx_positions` / `wavelength` on
`VirtualArrayAssemblyOp` — both combine `from_config()` with extra kwargs
safely).

### Known binding gaps (not fixed here)

- `radar_app::TensorMapDiscardOp` (`report_ops.hpp`) is not exposed to
  Python, so `ops.TensorMapDiscardOp` is a small Python-only reimplementation.
This is a candidate for a future pybind11-binding change (touching
`_radar_app_bindings.cpp` + a rebuild), out of scope for the Python
application-structure work that landed this section.
