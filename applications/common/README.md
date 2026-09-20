# Application-support library (`radar_app`)

Shared C++ support for the seven standalone FMCW applications:

- `fmcw/siso`
- `fmcw/adaptive_aoa`
- `fmcw/simo`
- `fmcw/dense_beamforming`
- `fmcw/mimo_tdm`
- `fmcw/multi_waveform`
- `fmcw/mimo_ddm`

Each application entrypoint contains only Graph construction and port wiring.
Reusable configuration decoding, analytic sources, reporting, cadence
measurement, and the optional HSDK 5 dashboard live here.

## Layout

```text
radar_app/
  app/                         Generic Graph, source, config, and run helpers
  fmcw/                        Shared SIMO/MIMO schemas, derivation, and sources
  report/                      Detection reporting and cadence measurement
  viz/                         Optional dashboard wiring and compositor
  python/                      Optional report/view support bindings
```

The base `holoscan::radar_app` target is header-only. When the SDK exports the Graph-native Holoviz operator,
`holoscan::fmcw_dashboard` adds the compiled CUDA dashboard compositor.

## Configuration boundary

Operator parameters remain owned by their operators. Application loaders decode
only application-owned sections such as the waveform, run controls, cube
dimensions, targets, arrays, and visualization settings. Shared SIMO/MIMO schemas and derivation live under `fmcw/`; each executable
keeps its application-specific `config.hpp` in its own `cpp/` directory.
Generic document decoding and validation live under `app/`.

Application entrypoints include one workflow facade rather than another
application's files. Shared helpers therefore flow inward through
`radar_app`; applications never include one another.
