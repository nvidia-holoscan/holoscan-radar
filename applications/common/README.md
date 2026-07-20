# Application-support library (`radar_app`)

Shared, app-agnostic building blocks for the FMCW example applications
(`fmcw_siso`, `fmcw_simo`, `fmcw_dense_simo`, `fmcw_mimo`). It exists so those
apps stay small and readable: each app is mostly a graph-wiring `compose()` plus
its own `AppConfig`, not a pile of copy-pasted operators and scaffolding.

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
  app/holoviz_layout.hpp  HolovizLayout / RightPanel + build_holoviz_input_specs
                          + make_rd_holoviz_op factory + panel presets
  report/report_ops.hpp   DetectionReportOp, ClusterReportOp (2- or 4-column
                          rep_idx), NullSinkOp, TensorMapDiscardOp
  viz/viz.hpp             RdChannelSliceOp (SIMO per-RX RD slice),
                          RdMapToImageOp, ClusterOverlayOp, RollingPlotOp,
                          BoresightViewOp (az/el scatter), the turbo LUT, the
                          cluster palette, and the overlay / plot layer-name
                          helpers
  viz/display.hpp         gui_available() — the Holoviz display probe
```

Two shared geometry builders (`make_ura_geometry` / `make_ura_rx_positions`
and `make_virtual_geometry`) and the `advance_scene` frame-stepper live one
layer deeper, in `radar_common/sim/` — they are pure model helpers with no
Holoscan/CUDA/Holoviz dependency.

Everything in `radar_app` is under `namespace holoscan::radar::app` and included
as `"radar_app/..."`.

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
