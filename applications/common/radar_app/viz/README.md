# Shared FMCW dashboard

All seven FMCW applications use `FmcwDashboardOp` and the same C++/Python
visualization wiring. The dashboard presents the CFAR power surface, current
range-Doppler detections, spatial detection trails where angles are available,
and rolling feature histories. Lines show analytic simulated truth; dots show
detections. The subtitle reads `Frame <N> (<t> s) | <M> detections`, using scenario time
and the number of displayed markers (cluster representatives in clustered workflows).

The power map uses a sequential viridis palette and a moving power window.
Normalization remains relative to the current frame's peak; the colorbar labels
show `10 log10(peak) - dynamic_range_db` through `10 log10(peak)` for that same
frame. The default window stays 60 dB wide while both endpoints follow signal
power. These are dB relative to one input power unit, not calibrated dBm.
The peak readback shares the existing annotation input synchronization; frames
skipped by the display-rate cap do not run the reduction. Dense angle cubes
are projected with the maximum power across the leading angle axes. Range
increases upward; velocity increases to the right. Physical angle axes use the
existing radar-boresight presentation conversion.

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

HSDK 5.0.0.7 provides `TextLayerDesc` and `TextRunHeader`. The compositor emits
an owned, host-resident uint8 tensor of SDK text records separately from the
CUDA RGBA image. Holoviz renders labels, ticks, titles, counters, and legends;
Radar no longer maintains bitmap glyphs or rotates text into the image.
Text uses the staging SDK's verified pixel-coordinate mode; layout and glyph sizes
target the configured window dimensions, with compact spacing below 540 px height.
Subplot titles are centered on their panels, with compact top insets and tighter
spacing above the data; axis captions are centered on the plot area.
Titles use brighter text than ticks and labels; supporting captions are muted.
Axis gutters reserve separate rows for ticks and captions. Endpoint X labels
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
The image-only `FmcwHolovizOp` presents the completed frame in a resizable window.
Joining both inputs lets the viewer recover after resize or remote-display stalls
have dropped queued frames. An image-driven `OnEach` with an `Exact` text side
selector can strand a stale image after its matching text is evicted. An independent
render clock can also redraw the image without its labels. The compositor caps
rasterization at `visualization.display_rate_hz`, while updating histories for
every admitted radar frame. Display products share a completion timestamp so
presentation alignment excludes upstream processing latency. Radar data and
report timestamps retain their existing behavior.

Composition and window presentation run in their own partition. Native CUDA
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
