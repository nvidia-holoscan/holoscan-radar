# Radar Pipeline

A SISO FMCW example application demonstrating the signal-processing chain:
a synthetic ADC source (one static clutter return + one moving target,
analytic dechirped point-target model) feeding
`FMCWRangeProcessorOp → ClutterFilterOp → FMCWDopplerProcessorOp`, with a
sink that locates the range-Doppler peak and reports it in physical units
via the operators' `range_bins` / `velocity_bins` axis outputs.

Launch via `./holohub run radar_pipeline`, or build directly:

```bash
cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
./build/applications/radar_pipeline/cpp/radar_pipeline
```

The clutter filter suppresses the (stronger) static return, so the reported
peak is the moving target: ~42 m at ~+5.5 m/s.

A Python variant will be added when the Python bindings land (deferred to
a later phase).
