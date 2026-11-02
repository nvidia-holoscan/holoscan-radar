# Shared FMCW dashboard

All seven FMCW applications use `FmcwDashboardOp` and the same C++/Python
visualization wiring. The dashboard presents the CFAR power surface, current
range-Doppler detections, spatial detection trails where angles are available,
and rolling feature histories. Lines show analytic simulated truth; dots show
detections. The status line reads `Frame <N> (<t> s) | <M> detections`, using scenario time
and the number of displayed markers (cluster representatives in clustered workflows).

The power map uses a sequential viridis palette and noise-anchored limits by
default. A bounded sample estimates the mean and standard deviation of background
power in dB. The endpoints are:

```text
lower = mean_db - noise_sigma * stddev_db
upper = mean_db + noise_sigma * stddev_db + noise_headroom_db
```

The default is mean plus/minus two standard deviations, with 6 additional
dB above the upper bound. Both limits are smoothed together, and the image
and colorbar use the same limits. This narrow window reveals background
texture; strong targets saturate at the upper endpoint. Increase
`noise_headroom_db` to show more of their power range. These are dB relative
to one input power unit, not calibrated dBm or a CFAR threshold.
Dense angle cubes use the same maximum
projection across leading axes for sampling and display.

```yaml
app:
  visualization:
    noise_scaling: true
    noise_sample_limit: 2048
    noise_range_half_width: 3
    noise_doppler_half_width: 3
    noise_sigma: 2.0
    noise_headroom_db: 6.0
    dynamic_range_db: 60.0  # Peak-relative fallback span only
```

Each raw CFAR detection excludes a full range/Doppler cross: a horizontal strip
within the Doppler half-width and a vertical strip within the range half-width.
Doppler wraps periodically; range clips at its endpoints. Two compact GPU bitsets
represent the strips. Up to `noise_sample_limit` distinct, stratified cells are
jittered each rendered frame; masked, nonfinite, and nonpositive samples are
rejected without a search for replacements. The host computes dB moments with
one two-sided three-standard-deviation clipping pass. No sorting or full-map
readback is required. This describes the displayed background distribution;
projection and detection-dependent censoring can bias it as a physical noise estimate.

At least `max(64, candidate_count / 16)` samples must survive both passes.
Limits are smoothed with a 0.25 s time constant. A missing estimate holds the
last valid pair of limits for at most 1 s, then falls back to the current peak minus
`dynamic_range_db`. More than 4096 raw detections skips estimation and takes
the same fallback path. With `noise_scaling: false`, peak-relative scaling is
immediate. All-zero/invalid frames have finite fallback limits of `[-60, 0]`
for the default span. Half-widths accept 0..64 bins, the sample cap 64..65536,
and the sigma multiplier finite values in 0..10. Headroom must be finite,
nonnegative, and representable as float32. A degenerate noise band (flat
background or zero sigma) is widened symmetrically to at least 0.1 dB.
`noise_sigma` replaces the earlier `noise_floor_sigma` setting and controls
both endpoints; `dynamic_range_db` now applies only to peak-relative fallback.

Sampling and the peak readback share the existing annotation input barrier,
with no additional stream synchronization. Work is bounded by the candidate
cap times the number of projected planes, plus at most 4096 detection crosses
and a reset of the two axis bitsets.
Default sampling adds an 8 KiB device sample payload, at most 8 KiB of host readback,
and two bitsets (40 bytes for the TDM application's 64 by 256 map). Frames
skipped by the display-rate cap run neither reduction nor sampling. The
processing graph, CFAR outputs, and detection reporting are unchanged.

Range increases upward; velocity increases to the right. Physical angle axes
use the existing radar-boresight presentation conversion.

## Measured scaling cost

A matched MIMO TDM check at `d9d92ac5` on 2026-10-21 used the Release staging CUDA 13 image
on NVIDIA GB10 (driver 580.173.02), a 1280 by 720 Xvfb window, the production
64 by 256 power map, 10 Hz input, and the default sampling settings above.
The comparison toggled only `noise_scaling` on the same binary. Nsight Systems
2026.4.1 used software CUDA, NVTX, and Vulkan tracing. Three runs per setting
processed 100 frames each, excluding the first 20 frames and the final frame
from timing (79 samples per run). Timing values below are medians of run medians.
Operator wall times include synchronization and waiting; they are not CPU execution times.
The later visibility and symmetric-band changes affect scalar limit placement;
these timings precede those changes.

| Metric | Peak scaling | Noise scaling |
| --- | --- | --- |
| Dashboard compositor wall time | 0.677 ms | 0.699 ms |
| Compositor CUDA kernels, summed per frame | 82.5 us | 85.6 us |
| Added cross-mask kernel | — | 2.30 us |
| Added sampling kernel | — | 2.05 us |
| Whole-application CPU, percent of one core | 21.75% | 21.33% |
| Whole-application mean RSS | 718.50 MiB | 718.46 MiB |

CPU/RSS used three separate unprofiled runs per setting, sampled every 0.2 s,
excluding the first 3 s and final 1 s; the table reports the median of the
three run averages. The small CPU/RSS differences do not
establish a reduction. Default payload costs are 8232 device bytes and up to
8192 host sample bytes for this map, plus tensor/allocator bookkeeping.

Complete visualization time, from dashboard compute entry through the matching
window operator return, had per-run medians of 3.73/3.28/3.27 ms with peak
scaling and 3.80/3.37/4.29 ms with noise scaling. Paired differences were
+0.07/+0.09/+1.03 ms; the slow third pair spent most of the difference in
window presentation (3.05 versus 2.21 ms). This is a small local Xvfb study
with default clocks, not a bound on GUI, scanout, or SSH display latency.
All profiled runs retained 100 source/report frames. The matched reports' 460
cluster rows were identical after excluding floating-point-sensitive representative fields.

The 58 focused dashboard/config/provider tests, nine Python GUI workflow and
compact-layout checks, a native C++ TDM GUI run, and the full lint suite passed.
One-off capture/profiling artifacts live under ignored
`build-gui/noise-scaling-checks/`; no assessment scripts are part of the library.

## Panel widths

Set `app.visualization.range_doppler_width_percent` to the percentage of usable
horizontal panel space allocated to the range-Doppler section, including its
axis gutter. The default is 45 (previously approximately 58 with angles and 68
without). The remaining width goes to spatial plots and histories. Both C++
and Python read the same setting:

```yaml
app:
  visualization:
    range_doppler_width_percent: 45
```

Values must be finite and between 25 and 70. Outer margins and the panel gap
are excluded. Compact canvases clamp the split to leave at least 240 px for
the range-Doppler section and 280 px for angle panels (240 px without angles).

## Azimuth / range sector

For fields of view up to 180 degrees, the sector fills the panel width and
height independently. This expands narrow scans instead of leaving a tiny
wedge inside a mostly empty panel. Range rings mark quarters of the displayed
maximum range (halves in compact windows), with whole-meter labels outside
the sector edge. Azimuth ticks above the wedge show the actual scan limits and
center in degrees (only limits in compact windows). Units appear once in the
`Azimuth / range (deg)` title; ring labels include meters. The `X scale` caption reports the horizontal distance scale relative
to the vertical distance scale. This is a fitted view with unequal axis scales,
so screen angles and distances should not be measured as a plan view.

Grid lines, simulated truth, and detection trails use the same projection.
Samples outside the displayed range or azimuth bounds are omitted, and truth
trails break when leaving the field of view. Wider scans retain the circular
plan view. The projection is shared by the C++ and Python applications.

The boresight and azimuth/elevation history plots display angle ticks in degrees,
as indicated in their titles. This is a label conversion only: angle bounds,
projections, and stored detection/truth coordinates retain radians.

## Native text and presentation

FMCW applications select X11 finished-image presentation when X11/XRender support
is built, `DISPLAY` is nonempty, and `app.visualization.x11_presentation` is
`true` (the default). Set it to `false` to use the native HoloViz window path:

```yaml
app:
  visualization:
    x11_presentation: false
```

This setting applies to both C++ and Python. It controls window presentation;
HoloViz still composes the image and native text offscreen in either case.
The native window path requires a compatible Vulkan presentation surface.
Startup logs `Radar presentation: X11 finished-image transport` for X11,
or `Radar presentation: CUDA device <N>` for the native window path.

HSDK 5.0.0.7 provides `TextLayerDesc` and `TextRunHeader`. The compositor emits
an owned, host-resident uint8 tensor of SDK text records separately from the
CUDA RGBA image. Holoviz renders labels, ticks, titles, counters, and legends;
Radar no longer maintains bitmap glyphs or rotates text into the image.
Text uses the staging SDK's verified pixel-coordinate mode; layout and glyph sizes
target the configured window dimensions, with compact spacing below 540 px height.
Subplot titles are centered on their panels, with compact top insets and tighter
spacing above the data; axis captions are centered on the plot area.
The main title uses `#76B900` text at 20 px (17 px in compact layouts), with a
one-pixel horizontal overdraw for a bold appearance; the native font has no weight setting.
The subtitle shows the selected mode's metadata description in muted blue-gray,
followed by a brighter, separate frame-status line. Descriptions are compiled from the C++ application manifests
and shared by both language dashboards; metadata edits trigger CMake regeneration.
AoA algorithm and DDM coding settings select the matching description even when
a custom YAML is launched directly. Compact layouts reserve one extra header row. Subplot titles use brighter text
than ticks and labels; supporting captions are muted.
Axis gutters reserve compact, separate rows for ticks and captions, with small
outer insets so the data fills more of each subplot. Endpoint X labels
stay within the plot width; vertical gutters keep centered Y labels clear of
titles and bottom tick labels to avoid corner collisions. Cartesian axes use short inward ticks, with four unlabeled
minor ticks between each pair of major ticks (one in compact layouts). Sector
edges have matching minor range/azimuth marks. Minor ticks add no grid lines;
label widths and tick density follow the selected canvas size.
Dark panel backgrounds and subdued grids keep target accents prominent.
`FmcwFrameComposerOp` uses the SDK's `CudaSinkDesc` to render text and the canvas
into one CUDA-backed image before presentation. Resizing the window scales the
complete frame, keeping labels and plots aligned. This scales the configured
layout; it does not reflow panels or change the render resolution. Increase
`visualization.window_width` / `window_height` for a higher-resolution canvas.
The operator's built-in monospace font supports ASCII; longer workflow/profile titles are shortened to fit.

The app-private `FmcwFrameComposerOp` reuses the SDK renderer and requires image
and text with one shared completion stamp (`OnAligned` with zero tolerance and
`Exact` selectors on both inputs). Each completed pair drives one offscreen SDK render.
On X11, `ImageDownloadOp` copies that completed image into a bounded host pool and
`X11ImagePresenterOp` sends it to a resizable X window. XRender scales the whole
canvas on the display server. This is finished-image transport: all radar
processing, plot composition and native text rendering remain on the GPU. It
works over SSH and with all GPUs visible, without using a Vulkan window swapchain
on a potentially different GPU. No device index is hardcoded or hidden. Other
window systems retain the image-only SDK `FmcwHolovizOp` presentation path.
Joining both inputs lets the viewer recover after resize or remote-display stalls
have dropped queued frames. An image-driven `OnEach` with an `Exact` text side
selector can strand a stale image after its matching text is evicted. An independent
render clock can also redraw the image without its labels. The compositor caps
rasterization at `visualization.display_rate_hz`, while updating histories for
every admitted radar frame. Display products share a completion timestamp so
presentation alignment excludes upstream processing latency. Radar data and
report timestamps retain their existing behavior.

Startup discovers the CUDA ordinal selected by offscreen rendering. On X11, no
window presentation GPU is needed. For other window systems, startup additionally
queries the SDK's window-surface adapter.
When an image producer and its renderer differ, `ImageDownloadOp` and
`ImageUploadOp` move the image through a bounded host pool into the receiving GPU.
This covers both the input canvas and completed frame, preserving image metadata
and timestamps. This path does not require CUDA peer access and adds two copies
and synchronization per device boundary. Same-device rendering retains
the direct connection. All seven C++ and Python workflows share this policy.

Composition and window presentation run in separate partitions. Native CUDA
capture adds a GPU render/resolve stage and one device copy, with bounded SDK
pools; text remains SDK-rendered rather than drawn by Radar. Depth-one drop-oldest display queues
bound pending images; processing queues retain their existing aligned-input
policy. The image bridge retains its dedicated device pool for consumption on
the renderer's CUDA stream. Canvas initialization, heatmap coloring, and raster
annotation composition share one CUDA kernel. These remove redundant work;
no frame-rate improvement is claimed without a matched performance run.

## Remaining raster work

Colored detections and truth trails remain raster annotations. The installed
Holoviz operator's geometry-input path uses a white batch color and has no
per-primitive color input, so moving these to geometry layers would lose the
existing target-color distinction. Dear ImGui and configurable TTF rendering
are available through the lower-level renderer, rather than callbacks on
`HolovizOp`; they would require a custom presentation operator.

Window events are serviced on rendered frames. A stalled or stopped producer
leaves the last image visible but does not provide an independent UI event
pump; this is an SDK operator limitation.

Resize recovery can be checked with the delayed-X11 regression probe in
[`tests/gui/README.md`](../../../../tests/gui/README.md). It verifies update
recovery across resize bursts and delayed X11 traffic. The GPU dashboard test
also checks that native text is embedded in the emitted frame and unmatched
image/text inputs do not render.
