# MIMO TDM visualization resource and UTF-8 assessment

Measured 2026-10-21 at `433a49d6` on NVIDIA GB10, driver 580.173.02,
ARM64, Release build, staging HSDK 5.0.0.7 / CUDA 13.0. Container image ID:
`a6f4b11a393d7fa09ab339e8ae7f5dc6113fbe2f350558b4434f473fef93bcef`.
Results characterize this workload, rather than establishing timing thresholds.

## Workload and method

The production MIMO TDM YAML, Python graph, 2 TX × 16 RX, 64 chirps/TX,
256 samples/chirp, and five simulated targets. Source cadence is 10 Hz;
`display_rate_hz` is 20, so the GUI receives at most ten new radar frames/s.
Canvas: 1280×720. Xvfb runs a local 1920×1080 X11 screen; no SSH forwarding,
remote desktop, physical scanout, or input-to-photon latency is measured.
Production uses the SDK default ASCII monospace renderer.

Resource measurements use three independent 150-frame runs per configuration;
production GUI on/off order alternates between repeats. Sample process CPU and
RSS and device-wide NVML GPU utilization/power every 200 ms. Discard the first
three seconds and last second; exclude zero-RSS/zombie samples. Report arithmetic
means across independent runs. CPU 100% means one logical core. Application CPU
and RSS exclude the sampler, profiler, and X server. All twelve resource runs
completed all 150 report frames with clean Graph sessions.

Latency and GPU activity use separate 100-frame Nsight Systems captures with
`cuda-sw,nvtx,vulkan` tracing, no CPU sampling/context-switch tracing, and CUDA
memory tracking. Analyze source publications 21–99, yielding 79 steady frames.
These are instrumented timings. A C++ production-renderer capture independently
measured 4.59 ms mean visualization-chain latency versus 5.16 ms in Python.

## Current visualization baseline

| Whole application metric | GUI off | GUI on | Increment |
| --- | --- | --- | --- |
| CPU, % of one logical core | 16.64 | 21.34 | 4.70 percentage points |
| Resident process memory, MiB | 529.67 | 719.12 | 189.45 |
| Device-wide GPU utilization, % | 9.30 | 9.95 | 0.65 percentage points |
| Device-wide board power, W | 13.50 | 13.91 | 0.41 |

GPU utilization is coarse, and the small power difference is sensitive to
sampling, clocks, temperature, and run order. These device-wide numbers do not
isolate text rendering. NVML returns unavailable GPU memory on GB10's shared-memory
architecture. Nsight reports peak live CUDA-visible allocations of 1218 MiB off
and 1456 MiB on, an approximately 238 MiB increment. These are logical allocation
observations, include imported Vulkan resources, and are **not** independent
physical GPU residency or memory to add to RSS. A single RGBA canvas is 3.52 MiB;
SDK image/output pools and renderer resources hold multiple canvases.

| Visualization operator, inclusive wall time | Mean, ms | p95, ms |
| --- | --- | --- |
| Dashboard raster/annotation compositor | 0.678 | 0.729 |
| CUDA image bridge | 0.038 | 0.045 |
| SDK offscreen image + native text composition | 0.272 | 0.336 |
| SDK window presentation | 3.981 | 6.118 |
| Complete chain, including scheduling gaps | 5.162 | 7.551 |

Chain latency starts at dashboard compositor entry and ends when window
presentation returns. CPU operator wall time includes synchronization and
waiting; it is not CPU execution time or GPU execution time. Most elapsed time
is presentation: `vkQueuePresentKHR` alone averaged approximately 1.94 ms in the
C++ diagnostic capture. Scanout completion is outside this measurement.

The traced CUDA kernel/copy and Vulkan queue durations sum to approximately
0.27 ms per displayed frame: 0.085 ms CUDA kernels, 0.142 ms CUDA copies, and
0.045 ms Vulkan work. Activities can overlap; this sum is not an end-to-end
latency. CUDA copies move approximately 17.58 MiB/frame, or 176 MiB/s at 10 Hz.

ADC-cube publication to report completion averaged 7.35 ms off and 7.75 ms on;
publication to presentation return averaged 12.93 ms on (p95 15.02 ms). These
exclude simulated ADC generation and physical acquisition. One profiled pair
cannot establish a statistically significant 0.40 ms radar-processing penalty.

## UTF-8 renderer comparison

The prototype replaces only the SDK text-renderer factory, retaining the same
TDM graph, labels, image composition, synchronization tokens, and presentation.
Default-font UTF-8 isolates the renderer change; the TrueType variant adds
DejaVu Sans Mono with a 32 px atlas. Neither changes the application default.
The additional upload stream is retained until process exit in this measurement
shim; production integration must explicitly own and retire it.

| Metric, GUI enabled | Current ASCII | UTF-8, default font | UTF-8, TrueType |
| --- | --- | --- | --- |
| Whole-process CPU, % of one core | 21.34 | 21.72 | 22.43 |
| Whole-process RSS, MiB | 719.12 | 721.28 | 721.57 |
| Offscreen image/text operator mean, ms | 0.272 | 0.304 | 0.327 |
| Offscreen image/text operator p95, ms | 0.336 | 0.328 | 0.413 |
| Offscreen Vulkan GPU work/frame, ms | 0.0236 | 0.0239 | 0.0252 |
| Complete visualization chain mean, ms | 5.162 | 4.388 | 4.298 |
| Complete visualization chain p95, ms | 7.551 | 5.107 | 4.787 |

The isolated composition-stage increase is approximately 32 µs with the same
font, or 56 µs with TrueType. The UTF-8 path adds approximately 35 KiB of CUDA
geometry uploads per frame, about 0.34 MiB/s at 10 Hz, and 2.2–2.5 MiB RSS.
Whole-process CPU varies across repeats: current 20.99–21.97%, UTF-8 default
21.33–22.27%, TrueType 21.74–23.49%. The small CPU differences are diagnostic,
not confidence-bounded overhead estimates. Presentation variation dominates
complete-chain timing; the lower UTF-8 totals do not establish a speedup.

The real `°` glyph was visually verified in the TrueType UTF-8 path using a
separate preview-only label substitution. Benchmark runs retain identical ASCII
labels and omit that substitution. The proof capture is
`build-gui/gui-checks/mimo_tdm_style_utf8_degrees.png`; Unicode-safe label widths,
centering, and truncation still need production integration.

`holoviz::text::create_text_renderer()` and `TextRenderer::add_text()` are SDK
implementations; the former uses Holoviz's private Dear ImGui renderer. No new
external UI framework is required. However, these are internal HoloViz APIs.
[NVIDIA's HSDK 5 Visualization documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/modules/visualization)
identifies `HolovizOp` as the supported public interface and gives the lower
renderer no source/ABI compatibility guarantee. The installed operator currently
selects the ASCII renderer and exposes no renderer-selection setting.

For this preset, UTF-8/TrueType cost is small. Adoption is mainly an SDK
integration/maintenance decision: owned upload stream and buffer ring,
Unicode-aware label layout, resize/recovery checks, and verification on SDK
updates. This assessment does not establish performance at 60 Hz, other
resolutions, forwarded displays, or arbitrary Unicode/font workloads.

## Reproduction and artifacts

Run inside the same GPU container from the repository root, with the Python
provider enabled in `build-gui` and the SDK/provider Python paths configured.
Start Xvfb separately and keep it running for all comparisons:

```bash
Xvfb :94 -screen 0 1920x1080x24 -ac -noreset &
export DISPLAY=:94
python3 performance/applications/measure_fmcw_viz.py
```

Build the measurement-only factory interposer and run its default-font and
TrueType variants. Do not preload it into production sessions:

```bash
g++ -std=c++20 -O2 -fPIC -shared performance/applications/utf8_text_probe.cpp \
  -o build-gui/viz-performance/utf8_preview.so \
  -I/usr/include/cccl -I/usr/local/cuda/include \
  -I/usr/include/holoscan/third_party -lholoviz -lcuda -ldl
LD_PRELOAD="$PWD/build-gui/viz-performance/utf8_preview.so" \
  python3 performance/applications/measure_fmcw_viz.py --modes on \
  --output build-gui/viz-performance/utf8_default_resources
UTF8_TTF=1 LD_PRELOAD="$PWD/build-gui/viz-performance/utf8_preview.so" \
  python3 performance/applications/measure_fmcw_viz.py --modes on \
  --output build-gui/viz-performance/utf8_ttf_resources
```

For each configuration, profile a separate 100-frame run. `on.yaml` and `off.yaml`
in the original artifact directory differ only in visualization enablement and
retain production settings with `num_frames: 100`:

```bash
nsys profile --trace=cuda-sw,nvtx,vulkan --sample=none --cpuctxsw=none \
  --cuda-memory-usage=true --force-overwrite=true \
  -o build-gui/viz-performance/python_on python3 -c \
  'from holoscan_radar.applications import mimo_tdm; mimo_tdm.build_graph("build-gui/viz-performance/on.yaml").run()'
nsys export --type sqlite --force-overwrite=true \
  --output build-gui/viz-performance/python_on.sqlite \
  build-gui/viz-performance/python_on.nsys-rep
```

Use `nsys profile ... env LD_PRELOAD=... [UTF8_TTF=1] python3 ...` for the UTF-8
captures, so the interposer loads into the application rather than the profiler.
Use CUDA software tracing: the default hardware trace on this host dropped
incomplete kernel timestamps. Nsight warns that some NVTX events may be missing;
all four expected visualization stages have 79 complete matched steady frames.
The GUI-off capture has no Vulkan events, as expected.

[Versioned measurements](2026-10-21-mimo-tdm-viz.json) contain repeat-level results
and profile statistics. Raw configs, logs, NVML CSVs, Nsight reports/SQLite, and
the SQL-analysis scripts remain under `build-gui/viz-performance/`. The first
resource batch predates shutdown filtering in the runner; its raw samples were
filtered identically by `analyze_resources.py` before producing these results.
No production code or operator contracts were changed.
