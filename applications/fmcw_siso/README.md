# FMCW SISO

A SISO FMCW example application demonstrating the signal-processing chain:
a synthetic ADC source (one static clutter return + one moving target,
analytic dechirped point-target model) feeding
`FMCWRangeProcessorOp → ClutterFilterOp → FMCWDopplerProcessorOp`, with a
sink that locates the range-Doppler peak and reports it in physical units
via the operators' `range_bins` / `velocity_bins` axis outputs. The source
emits one cube per tick and advances the targets between ticks, so the
reported range sweeps over the run.

Launch via `./holohub run fmcw_siso`, or build directly:

```bash
cmake -S . -B build -DBUILD_ALL=ON && cmake --build build
./build/applications/fmcw_siso/cpp/fmcw_siso            # default config
./build/applications/fmcw_siso/cpp/fmcw_siso my.yaml    # override config
```

The clutter filter suppresses the (stronger) static return, so the reported
peak is the moving target: ~42 m at ~+5.5 m/s.

## Configuration

The waveform, cube dimensions, run cadence, clutter method, targets, and
noise are read from [`cpp/fmcw_siso.yaml`](cpp/fmcw_siso.yaml) at startup.
A copy is staged next to the built executable and used by default; pass a
different path as the first CLI argument to override it. Each target is
placed along the +x look direction at a given `range` (m) and radial
`velocity` (m/s, positive = receding) with a linear `amplitude`.

## Visualization

Set `visualize: true` in the config to open a live Holoviz window showing
the range-Doppler map (normalized log magnitude, gray scale; range along x,
Doppler along y) alongside the textual peak reports. This requires a
GUI-capable host — a reachable display server and Vulkan. At startup the app
probes Holoviz directly; if no usable display is found (headless host, or a
stale `DISPLAY` with no server behind it), it logs a warning and runs
exactly as if `visualize` were `false`.

A Python variant will be added when the Python bindings land (deferred to
a later phase).
