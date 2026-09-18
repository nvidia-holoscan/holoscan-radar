# Application-support library (`radar_app`)

Shared C++ support for the four standalone FMCW applications:

- `fmcw/simo_detect_then_beamform`
- `fmcw/simo_beamform_then_detect`
- `fmcw/mimo_tdm`
- `fmcw/mimo_ddm`

Each application entrypoint contains only Graph construction and port wiring.
Reusable configuration decoding, analytic sources, reporting, cadence
measurement, and the optional HSDK 5 dashboard live here.

## Layout

```text
radar_app/
  app/
    angle_presentation.hpp   Physical-angle presentation helpers
    config_path.hpp          CLI/default YAML path resolution
    graph_driver.hpp         Shared Graph compile/run lifecycle
    run_config.hpp           Common frame-count and visualization flags
    source.hpp               Rank-agnostic SIMO analytic ADC source
    visualization_config.hpp Dashboard configuration schema
  report/
    cadence_meter.hpp        Source and end-to-end cadence accounting
    report_ops.hpp           DetectionReportOp and drain helpers
  simple/
    simo/                    Typed SIMO schemas, loaders, and workflow facades
    mimo/                    Typed MIMO schemas, loaders, sources, and facades
  viz/
    fmcw_dashboard_graph.hpp Optional dashboard graph wiring
    fmcw_dashboard_op.*      HSDK 5 dashboard compositor
    display_probe.hpp        Runtime display availability check
    x11_display.hpp          X11 display probe
```

The base `holoscan::radar_app` target is header-only. When the pinned SDK
exports the Graph-native Holoviz operator, `holoscan::fmcw_dashboard` adds
the compiled CUDA dashboard compositor.

## Configuration boundary

Operator parameters remain owned by their operators. Application loaders decode
only application-owned sections such as the waveform, run controls, cube
dimensions, targets, arrays, and visualization settings. Derived operator
parameters and maximum tensor extents are produced by the private SIMO/MIMO
loaders under `simple/`.

Application entrypoints include one workflow facade rather than another
application's files. Shared helpers therefore flow inward through
`radar_app`; applications never include one another.

## Python bindings

The `radar_app/python` module retains bindings for the reusable report sinks
used by the Python operator test harness: `DetectionReportOp`,
`DetectionReportMode`, `NullSinkOp`, and `CadenceMeter`. The removed
legacy Python demo framework and Application/Holoviz visualization bindings are
not part of the current application surface.
