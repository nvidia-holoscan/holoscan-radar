# Shared FMCW dashboard

All seven FMCW applications use `FmcwDashboardOp` and the same C++/Python
visualization wiring. The dashboard presents the CFAR power surface, current
range-Doppler detections, spatial detection trails where angles are available,
and rolling feature histories. Lines show analytic simulated truth; dots show
detections. The header distinguishes scenario time from wall-clock performance.

The power map uses a sequential viridis palette with a relative dB legend.
Normalization remains relative to the current frame's peak. Dense angle cubes
are projected with the maximum power across the leading angle axes. Range
increases upward; velocity increases to the right. Physical angle axes use the
existing radar-boresight presentation conversion.

## Azimuth / range sector

For fields of view up to 180 degrees, the sector fills the panel width and
height independently. This expands narrow scans instead of leaving a tiny
wedge inside a mostly empty panel. Range rings mark quarters of the displayed
maximum range (halves in compact windows), in meters; the azimuth legend gives the actual scan limits and
center. The `X scale` caption reports the horizontal distance scale relative
to the vertical distance scale. This is a fitted view with unequal axis scales,
so screen angles and distances should not be measured as a plan view.

Grid lines, simulated truth, and detection trails use the same projection.
Samples outside the displayed range or azimuth bounds are omitted, and truth
trails break when leaving the field of view. Wider scans retain the circular
plan view. The projection is shared by the C++ and Python applications.

## Native text and presentation

HSDK 5.0.0.7 provides `TextLayerDesc` and `TextRunHeader`. The compositor emits
an owned, host-resident uint8 tensor of SDK text records separately from the
CUDA RGBA image. Holoviz renders labels, ticks, titles, counters, and legends;
Radar no longer maintains bitmap glyphs or rotates text into the image.
Text uses the staging SDK's verified pixel-coordinate mode; layout and glyph sizes
target the configured window dimensions, with compact spacing below 540 px height.
Subplot titles are centered on their panels; axis captions are centered on the plot area.
Titles use brighter text than ticks and labels; supporting captions are muted.
Axis gutters are compact, and label widths follow the selected text size.
Dark panel backgrounds and subdued grids keep target accents prominent.
Resizing the window can stretch the raster independently of those text positions.
The operator's built-in monospace font supports ASCII; longer workflow/profile titles are shortened to fit.

The app-private `FmcwHolovizOp` reuses the SDK renderer and requires image and
text with one shared completion stamp (`OnEach` + required `Exact` text selection,
`Skip` on a missing match). Each completed pair drives one Holoviz render. This matters because the SDK
operator skips a missing text sample instead of retaining it: an independent
render clock can redraw the image without its labels. The compositor caps
rasterization at `visualization.display_rate_hz`, while updating histories for
every admitted radar frame. Display products share a completion timestamp so
presentation alignment excludes upstream processing latency. Radar data and
report timestamps retain their existing behavior.

The renderer runs in its own partition. Depth-one drop-oldest display queues
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
