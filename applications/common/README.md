# Application-support library (`radar_app`)

Shared, app-agnostic building blocks for the FMCW example applications
(`fmcw_siso`, `fmcw_simo`, and the future `fmcw_mimo`). It exists so those
apps stay small and readable: each app is mostly a graph-wiring `compose()`,
not a pile of copy-pasted operators.

The applications are organized around three kinds of component:

1. **Graph / flow wiring** — building the operator graph, parsing the app's own
   config, and the synthetic source. This is app-specific and lives in each
   application's own translation unit (e.g. `fmcw_siso/cpp/fmcw_siso.cpp`).
2. **App-specific operators** — custom to one application, not generalized.
   For the FMCW examples this is essentially just the synthetic ADC source
   (`AdcSourceOp` in SISO emits rank-2; `SimoAdcSourceOp` in SIMO emits
   rank-3), so it stays with the app.
3. **App-generalized operators / helpers** — reused across applications. These
   live here, in `radar_app`.

Operator *parameters* are not a fourth category: the signal-processing,
detection, clustering, and reduction operators take their configuration
natively through Holoscan's `from_config()` (backed by the `YAML::convert<>`
specializations in `radar_common`), so there is no per-app config-parsing
layer to share.

## Layout

```text
radar_app/
  report/cadence_meter.hpp  CadenceMeter — wall-clock source / e2e probe
  report/report_ops.hpp   DetectionReportOp, ClusterReportOp (2- or 4-column
                          points), NullSinkOp
  viz/viz.hpp             RdChannelSliceOp (SIMO per-RX RD slice),
                          RdMapToImageOp, ClusterOverlayOp, RollingPlotOp,
                          BoresightViewOp (az/el scatter), the turbo LUT, the
                          cluster palette, and the overlay / plot layer-name
                          helpers
  viz/display.hpp         gui_available() — the Holoviz display probe
```

Everything is under `namespace holoscan::radar::app` and included as
`"radar_app/..."`.

`CadenceMeter` is an optional shared probe the demos wire into the ADC source
(`on_source_emit` after each cube emit) and the terminal report op
(`on_sink_done` / `DetectionReportOp.cadence`). It warns when source inter-emit
or end-to-end latency exceeds `frame_period` (+ max(10 ms, 10% of period) slack) and prints a
summary after `run()`. Toggle with YAML `run.meter_cadence` /
`run.meter_log_every`.

## Using it

`radar_app` is a header-only `INTERFACE` CMake target (operator methods are
in-class inline; free helpers are marked `inline`), so it compiles no objects
of its own and carries only include paths + link dependencies (`radar_common`,
Holoviz, CUDA). An application links it with:

```cmake
target_link_libraries(my_app PRIVATE holoscan::radar_app ...)
```

and then wires the operators into its own graph. The app keeps ownership of the
HolovizOp layout (views / `InputSpec`s) — `radar_app` provides the operators
and the shared names/colors, the app decides where they render.
