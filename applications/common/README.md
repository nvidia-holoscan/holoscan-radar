# Application-support library (`radar_app`)

Shared C++ support for the seven standalone FMCW applications and five
standalone [SAR applications](../sar/README.md):

- `fmcw/siso`
- `fmcw/adaptive_aoa`
- `fmcw/simo`
- `fmcw/dense_beamforming`
- `fmcw/mimo_tdm`
- `fmcw/multi_waveform`
- `fmcw/mimo_ddm`

The SAR apps are `spotlight`, `video_sar`, `tiled_strip`,
`continuous_strip`, and `independent_images`.

Each application entrypoint contains only Graph construction and port wiring.
Reusable configuration decoding, analytic sources, reporting, cadence
measurement, and the optional HSDK 5 dashboard live here.

## Layout

```text
radar_app/
  app/                         Generic Graph, source, config, and run helpers
  fmcw/                        Shared SIMO/MIMO schemas, derivation, and sources
  sar/                         CPHD graph driver, geometry, mosaic, reports, CUDA raster
  report/                      Detection reporting and cadence measurement
  viz/                         Optional dashboard wiring and compositor
  python/                      Optional report/view support bindings
```

The base `holoscan::radar_app` target is header-only. When the SDK exports the Graph-native Holoviz operator,
`holoscan::fmcw_dashboard` adds the compiled CUDA dashboard compositor.
The SAR support builds a CUDA image raster, an on-image Holoviz HUD, a 20 s
Gantt timeline window, and optional Graph-native Holoviz branch; the five app
entrypoints keep their own mode-specific wiring. The Graph driver puts the paced CPHD reader, range/manager, and backprojection in
separate partitions and binds a dedicated `ThreadPlacement` lane on each so
collect-vs-wall lag is the live playback clock rather than formation GPU time
on the controller thread.

## Configuration boundary

Operator parameters remain owned by their operators. Application loaders decode
only application-owned sections such as the waveform, run controls, cube
dimensions, targets, arrays, and visualization settings. Shared SIMO/MIMO schemas and derivation live under `fmcw/`; SAR geometry,
CPHD-derived sizing, reporting, and visualization helpers live under `sar/`. each executable
keeps its application-specific `config.hpp` in its own `cpp/` directory.
Generic document decoding and validation live under `app/`.

Application entrypoints include one workflow facade rather than another
application's files. Shared helpers therefore flow inward through
`radar_app`; applications never include one another.

## SAR configuration and lifecycle boundaries

All six SAR graphs have C++ and Python entry points with shared YAML, presentation, dumping and playback semantics. Samples default to single with double geometry; `sample_precision: double` opts into double samples. Native graphs keep source/ingest, application and formation partitions distinct; do not attribute scheduling independence solely to default thread-placement bindings. `SAR_DUMP_PLAN` can inspect compiled placement.

C++ owns SIGINT/SIGTERM handlers and polls a lock-free stop flag from caller-managed asynchronous sessions. Python retains Python signal ownership, releases the GIL for native work, polls interrupts, then stops and joins before propagating interruption. Window closure and Escape request graph shutdown. GPU work already in flight must finish before cleanup returns. End-of-stream display retention is driven by `app.visualization.final_image_hold_s`: positive holds for that duration, zero closes with the graph, negative holds until closure/interruption. This is implemented; historical claims that negative hold is unsupported are stale.

The app still stores `sidecar_path` and `collection_physics` after configuration derivation, and parses `nvradar_vph_path` without applying an external VPH file. Removing those fields is separate cleanup. Strip crop fields are derived but not applied by the current visualization path. Full-resolution focus, radiometry and seam qualification across real collections remains open; generated fixture smokes and visual preset reviews establish narrower evidence.
