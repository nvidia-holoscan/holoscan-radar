# Shared radar visualization

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

FMCW applications offer a presenter choice when X11/XRender support is built,
`DISPLAY` is nonempty, visualization is enabled, and
`app.visualization.x11_presentation` is omitted. At startup, an interactive terminal
asks whether the setup needs X11 image transport (for example SSH/X11 forwarding)
or a native HoloViz window on a local desktop. Enter selects X11.
An explicit boolean bypasses both the question and any remembered answer:

```yaml
app:
  visualization:
    x11_presentation: false
```

Set `true` to select X11 without asking, or `false` to select native HoloViz.
Both C++ and Python then ask how long to remember the choice:

- **This run only** (default): no preference is saved.
- **Remember for 7 days**: expires exactly seven days after selection.
- **Remember until changed**: no expiration.

Preferences are keyed by `DISPLAY`, `WAYLAND_DISPLAY`, and local/SSH session type,
under `$XDG_CACHE_HOME/holoscan-radar/presentation-v2`, or
`$HOME/.cache/holoscan-radar/presentation-v2` when `XDG_CACHE_HOME` is unset.
Version-2 JSON records store the presenter, setup key, save time, and expiration
as UTC Unix seconds (`expires_at: null` means until changed). Each load checks
version, field types, presenter, setup identity, timestamps, and expiration.
Expired, malformed, wrong-setup, or future-dated records are ignored. Old
`presentation-v1` files are not reused. Display variables identify a setup;
they do not prove that SSH forwarding is in use.

The host CLI mounts the project as the container home, so its default cache is
`.cache/holoscan-radar/presentation-v2` in the host checkout and survives container
restarts. Direct runs use their own environment's cache location. Every reused
answer prints its expiration (or "until changed") and how to choose again.
Use `--choose-presenter` to clear the current setup's saved answer and ask again;
selecting "this run only" then leaves no saved answer. Explicit YAML still wins:
omit `x11_presentation` to use the interactive choice. For example, on the host:

```bash
./holoscan-radar run mimo_tdm --language cpp --run-args=--choose-presenter
./holoscan-radar run mimo_tdm --language python --run-args=--choose-presenter
```

Direct executables accept `[config.yaml] --choose-presenter`; Python's
`build_graph(path, choose_presenter=True)` exposes the same behavior.
The reset option requires an interactive terminal when a choice is needed.
Without a terminal, a valid saved answer is used; otherwise X11 remains the
default and no answer is cached. EOF at the presenter question defaults to X11;
EOF at the remember question keeps the selected presenter for this run only.
Nonvisual runs and configurations without X11 never ask. Cache write failures
are reported but do not prevent the selected presenter from running.

This setting controls window presentation; HoloViz still composes the image and
native text offscreen in either case. The native window path requires a
compatible Vulkan presentation surface. Startup logs
`Radar presentation: X11 finished-image transport` for X11,
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
composition clock can also redraw the image without its labels. The terminal
event clock therefore replays only already-composed complete frames. The compositor caps
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

X11 events are polled independently of scientific frame arrivals. Native
windows receive bounded periodic replay of the last completed frame, including
an initial blank frame, so the SDK can service close and resize during producer
gaps and final-image hold. Closing a window or pressing Escape stops the whole
application cooperatively; SAR's docked view and either separate window have this behavior.

Resize recovery can be checked with the delayed-X11 regression probe in
[`tests/gui/README.md`](../../../../tests/gui/README.md). It verifies update
recovery across resize bursts and delayed X11 traffic. The GPU dashboard test
also checks that native text is embedded in the emitted frame and unmatched
image/text inputs do not render.

## Shared presentation controls

All seven FMCW and six SAR applications expose the same C++ and Python options:

```bash
./holoscan-radar run spotlight --language cpp --run-args='--presenter x11'
./holoscan-radar run spotlight --language python --run-args='--presenter x11'
./holoscan-radar run simo --language python --run-args='--presenter headless'
```

Executables accept `--presenter x11|native|headless|off` before or after the
YAML path, plus `--choose-presenter`. SAR retains positional CPHD paths and
`--cphd`. Python builders accept `presenter=None` and
`choose_presenter=False`; SAR also retains `cphd_override`.

An explicit launch mode overrides YAML. Explicit YAML overrides preferences
and prompts. With no explicit selection, both modalities reuse the existing
FMCW version-2 cache, keyed by display and local/SSH setup. Its expiration,
reset, interactive, and unattended behavior is described above. Headless and
off modes never prompt. An explicit X11 selection cannot silently fall back to
native Vulkan presentation. A remote desktop is unnecessary for X11.

```yaml
app:
  visualization:
    enabled: true
    x11_presentation: true
    headless: false
    display_rate_hz: 60
    resize_mode: stretch  # fit preserves the whole canvas aspect ratio
    final_image_hold_s: 30  # 0: finish; positive: seconds; negative: until closed
    max_side: 1024  # SAR display downsampling, independent of product resolution
    timeline:  # SAR diagnostics; ignored by FMCW scientific rendering
      enabled: true
      display_rate_hz: 10
      window_width: 1280
      window_height: 112
```

FMCW retains its zero-second hold default. SAR retains its 30-second default
and preset overrides. Headless/off runs do not wait for an interactive hold.
SAR's omitted width/height use existing product-aware sizing and desktop-fitted
strip sizing; explicit `window_width`/`window_height` override the window canvas.
Resizing scales a bounded complete canvas rather than changing formation
resolution, reallocating pools, or reflowing scientific plots.

SAR's old `visualize`, `headless`, `skip_display_check`, `visualize_size`, and
`visualize_hold_s` remain aliases for `enabled`, `headless`,
`skip_display_check`, `max_side`, and `final_image_hold_s` in this block.
Aliases normalize within each file before inheritance. A child alias overrides
a parent setting; common spelling wins over a conflicting alias in the same
file, with a warning. Relative `extends` paths resolve against their containing
file; existing data paths retain leaf-config-relative resolution. Arrays and
scalars replace; mappings merge recursively; explicit nulls remain explicit.

SAR keeps grayscale percentile scaling, product orientation, crops, mosaics,
HUD, playback diagnostics, and autofocus displays. FMCW keeps its dashboard
plots, physical axes, truth/detection annotations, and noise scaling. Timeline
pixels render on the GPU from bounded host commands; its labels use the same
SDK GPU text renderer as formation and dashboards.

X11 transfers only finished display frames through bounded host pools. Native
presentation requires a CUDA-backed Vulkan adapter that can present to the
selected display. X11 and headless require an offscreen CUDA/Vulkan renderer,
but never create a native presentation surface during startup. GPU indices and
visibility are not changed. Cross-device bridges transfer bounded display RGBA
images, never raw apertures or complex mosaics.

Normal completion drains reporting and enabled dump sinks and gives the final
display frame a bounded completion opportunity. SIGINT/SIGTERM and window close
stop cooperatively and join execution. Display queues discard stale frames;
window resize and display delay must not corrupt saved scientific products.

Completed frames are consumed from bounded arrival queues without a short
capture-age cutoff: a SAR image may finish long after acquisition. Image,
geometry, and text join exactly before this boundary; retained/replayed frames
keep their original content timestamp. Native replay polls at least 20 Hz for
window events even when the scientific display-rate cap is lower.

## Docked SAR timeline

SAR now defaults to one viewing window with the timeline docked below the
formation image. A compact left status panel shares the footer with the timeline;
scene, formed dimensions, and backprojection processing cost
no longer cover the image. The APC direction marker stays over the image.
The image canvas keeps its configured resolution; initial
window height adds the timeline height. On resize, the footer stays 112 physical
pixels high by default and the image uses the remaining height. Very short
windows clamp the footer to half the available height. The image honors
`resize_mode`; the timeline fills its footer region independently and polls for batch events
at 10 Hz by default. Status text updates only with its associated SAR image.

While forming, the independent progress panel shows the forming image's title
under a stable checkpoint: `Receiving pulses`, `Forming image`, or `Image ready`.
Accepted/declined autofocus outcomes remain visible; a pending autofocus result
uses `Forming image` because the diagnostics do not distinguish auxiliary BP from
estimation. The panel labels green GPU-completed progress `Last completed` and
keeps the cyan Received marker and percentage on the same track. The marker
remains visible when the frontiers meet, including at zero and completion.
`Updates on completion` replaces the running collection clock; there is no
next-image countdown. These readouts use existing snapshots at batch events and otherwise at 1 Hz;
the timeline polls for completions at up to 10 Hz. GPU work can delay either presentation;
the checkpoint is not a heartbeat or a promise of immediate phase updates.
The right-hand time history advances only at batch events, as described below.
The current-image `BP work` line uses its known processing grid and expected
pulse count; replayed apertures count their retained pulses, while incremental
formation counts the current update. Unknown grids show `--`, never another
image's work. This diagnostic needs no new GPU transfer.
BP work uses 14px text in both the completed-image and progress panels, including
the compact layout. Compact layouts put the update hint beside the checkpoint
and BP work beside the title above the percentage labels and shared track;
long titles use ellipses.
This strip uses 56 pixels within the
existing footer; the main SAR image keeps its height. The left panel describes the displayed image in
five rows: title, `<px>x<px> pixels  <count> pulses`,
`BP work: <work> Gpx*pulses`, `Latency: <time> / <budget> @<rate>x`, and
`Throughput: <rate> Gpx*pulses/sec`. Here Latency means accumulated GPU
backprojection work time, excluding waits for incoming batches, playback pacing,
and display hold/transport. Throughput divides the measured BP pixel-pulse work
by that same time and always uses giga-units, including rates below one.
BP work uses the same measured processing lattice and pulse count as throughput,
including incremental processing for periodic outputs. The pulse count is
grouped with commas; unavailable diagnostics show `--`. The measured BP lattice
may differ from the displayed product or mosaic size. These five rows fit the
existing footer and remain frozen with the completed image.
The second Latency value is the time budget to keep up at the configured
`playback_rate`. Its radar-time basis is the selected collection's first-to-last
pulse `TxTime` span for a whole image, or the interval between image-closing
pulse times for periodic/overlapping updates (first pulse to closing pulse for
the initial image). This interval excludes duty-cycle quiet periods and is
divided by the positive playback rate frozen
with that image. For example, `Latency: 8.2s / 12.0s @0.2x` means 8.2 seconds of
GPU work against a 12-second budget at 0.2x playback. With `playback_rate: 0`,
there is no fixed pacing deadline, so the row shows `-- unpaced` for the budget.
A missing or invalid rate shows `-- @--`. A GPU time above the paced
budget means backprojection alone cannot sustain that playback update cadence;
a time below it does not guarantee that the complete pipeline keeps up.
The reader reuses its existing host pulse timestamps, with no additional file
reads, GPU transfers, or synchronization. Diagnostic history holds at most
4096 timestamps per packet across 128 packets (4 MiB of pulse-time samples);
larger packets retain endpoints only. Unknown/nonpositive intervals and missing
internal seams show `--`; exact interval tracking recovers at subsequent known
image boundaries without interpolating a pulse rate.
These rows freeze with the completed image; they do not show acquisition elapsed
or final-batch latency. Diagnostics match the image's capture timestamp in both
event-driven and clocked startup paths; unavailable measurements show `--`.
Periodic images share collection start in the middle panel; a replay loop begins
a new collection clock. Batch receipt/completion history and live playback lag remain on the timeline.
Missing diagnostic history leaves percentages/timing unknown rather than
substituting a later collection's measurement.

```yaml
app:
  visualization:
    timeline:
      enabled: true
      placement: bottom  # window restores the previous separate timeline
      display_rate_hz: 10
      window_height: 112
```

The status panel takes up to 320 canvas pixels, capped at half the canvas width.
Its text uses a fixed pixel size; narrow windows crop its right edge rather than
stretching the font. Long labels use an ellipsis. The timeline raster takes the
remaining width. `timeline.window_width` applies to separate-window mode; `window_height` is the dock height in either
layout. Disabling the timeline removes the footer. Separate-window and
timeline-disabled layouts retain the on-image HUD. C++ and Python share this
layout through the same native runtime. Saved image extents and values remain
independent of the viewing window.

For X11, one managed parent owns two child drawing surfaces. Each panel retains
its own server pixmap, bounded finished-frame download, presentation worker,
and event clock. A timeline tick uploads and paints only the footer; it cannot
activate formation, grayscale conversion, image HUD composition, or image
transport. Resizes and expose events repaint affected cached panels. Closing
the parent or pressing Escape in either panel stops the complete application.
The child windows are implementation details, with no separate taskbar entries
or decorations. Host and GPU work remain bounded by the configured canvases.

Native HoloViz presents a complete window on each refresh. `DockedFrameOp`
retains at most one completed GPU image and one completed GPU timeline and
composites them into a bounded GPU output pool. It does not rerun SAR rendering
or download the image when the timeline changes. Native composition/window
presentation still reads the cached image each refresh; this differs from X11's
partial repaint. Native refresh retains the existing display-rate setting and
minimum 20 Hz event pump; new timeline content still defaults to 10 Hz. The native footer uses the existing X11 window-extent integration to compensate
for window size so its physical height stays fixed. Without accessible window
extents, native sizing falls back to the configured canvas. This path requires positive visual qualification on a
presentation-capable GPU/display; GPU tests cover pixel placement and timestamps.

The image and its HUD still join exactly before caching. For the docked layout,
`FrameComposerOp` renders the SAR image into its top viewport and the status
into additional footer rows of the same completed GPU frame. Their acquisition
time, queue admission, and transport remain inseparable under stalls or drops.
X11 caches this combined frame and paints its image and status regions separately;
the independently updated timeline child occupies the footer's right side.
The extra footer rows add a fixed bounded amount of completed-frame transport,
only when a new image arrives. No raw aperture or complex product is downloaded.
The live timeline retains its independent clock; it is not forced to match an acquisition time.
Native combined display frames preserve the main image's capture time and
provenance. Headless execution renders and consumes both completed panels
independently, without creating a window or adding an interactive hold.

## SAR startup and readable diagnostics

Interactive SAR viewers show a charcoal startup canvas with a cyan PREPARING
label, a larger centered first-image message, and the first configured collection
name when available. The existing GPU compositor renders the background and
native SDK text once. Timeline ticks never regenerate or download this canvas.
The first completed image replaces it permanently. Headless execution emits no
placeholder. Closing the window works while the first aperture is accumulating.

The small timeline panel shows observed startup stage and elapsed time at 1 Hz;
its stage comes from existing reader/BP diagnostics and presentation progress.
It distinguishes starting up, receiving a collection, forming the first image
after reader EOF, and preparing the display. It does not invent a percentage
or claim a stage for which no diagnostic exists. The main placeholder stays
static while that text changes independently.

The left status panel uses 16-pixel titles and 14-pixel details at native canvas
resolution. Collection names scroll above their corresponding thinner bars;
image latency labels sit above the image-emission ticks. The footer remains
112 pixels high by default. The sidebar title describes the displayed image;
timeline labels describe collection history, which may run ahead of display.

GUI throughput uses readable SI prefixes, for example `12.4 Gpx*pulse/s`.
The displayed number converts the same held report measurement, preserving
its pairing with the image. Console measurements retain their original
scientific notation and units. Neither signal processing nor saved products
change. All six SAR applications and both languages share this implementation.

Interactive timelines show Received and Last completed pulse progress toward
an image. Received counts unique ingested pulses; Last completed advances only
after GPU batches complete. Replay and signed eviction work never inflate
received counts. Long batches may leave the completed bar stationary. The panel
uses a stable status and update-on-completion hint rather than an ETA or ticking
elapsed time. The readout refreshes at 1 Hz within the cached timeline panel,
beside history at normal widths and above compact history in narrower windows.
Headless runs and runs without a timeline do not enable extra completion tracking.

### SAR input/output event history

The right timeline shows `Pulse batch received` in cyan and
`Pulse batch processed` in green. Ticks mark original reader batches,
rather than running counts or newly formed images; nearby events can coalesce. Receipt means all
required reader outputs were published. Processed means nonblocking CUDA event
queries (or the existing image-boundary synchronization) proved completion of
backprojection covering that entire batch. Pulses discarded before GPU work
never produce a processed event. Split batches produce one processed
event after their final slice; replay can complete several source batches at
once. File and loop generations remain separate. Tick times are the host times
when completion was observed, not precise GPU finish timestamps or measured
end-to-end latency. Several ticks can share a pixel.

The common 20-second time axis ends at the latest observed batch receipt or
completion, labeled `Latest batch`, with `-20s` at the left. Tick spacing remains
actual elapsed time between events. Normal-height history adds `Time relative to
last batch update`; compact history uses the bookends alone. The window stays
fixed between events and after EOF, so delayed presentations never age events
out merely because wall time passed. Both received and processed events can
advance the endpoint. Completion observation is still polled at the configured
timeline cadence (default 10 Hz), without changing processing or event timestamps.
Identical bounded drawing/text snapshots skip GPU rasterization and display
transport. Changed status, optional playback lag, or resize feedback can redraw
the footer while preserving its event-time axis. Independent cached presentation
continues servicing window events and closure. Collection-colored dwell bars and image-emission ticks are
replaced by these two event lanes; the image title remains in the image and
progress cards. Optional playback lag stays available. Checkpoint text refreshes at
1 Hz. The main SAR image and HUD caches are unchanged. Timeline ticks do not
rerun backprojection or scientific image rendering. X11 keeps the main image
cached; native presentation composites its cached panels.

Diagnostics retain at most 1024 event-bin records and 128 source-packet records.
Events of the same kind within 40 ms coalesce, preserving the full 20-second
activity window at high batch rates. Overlapping tick rectangles merge before
GPU rasterization, keeping dense activity inexpensive to draw. History remains
best effort: expired packet metadata suppresses completion ticks rather than
inventing correspondences. No queue waits, new GPU synchronization, pulse-data copies,
public operator parameters, or preset changes are introduced. Automated GPU
pixel/text and real X11 window checks supplement manual visual review.

The history uses thin tracks and short cyan/green markers, with larger labels
above and below the tracks. Thin neutral diagonals connect exact receipt and
observed GPU-completion timestamps for the same source batch. Connecting areas
are never filled. Pending work keeps its receipt tick without a diagonal to an
invented completion at Now. Horizontal displacement describes observed delay,
including waiting, rather than GPU service time. A connection whose receipt
predates the window is clipped geometrically without moving its endpoints.

The bounded diagnostic history retains one actual matched pair per 40 ms
completion-time bin in 512 slots; both endpoints come from that batch and retain
their exact timestamps. This supplements the existing 1024 event bins and 128
source-packet records. Expired or unproven work never gains a completion link.
At dense rates, identical projected connectors merge and at most 64 real pairs
are selected across the retained history; the separate receipt/completion ticks
remain. Each connector is one bounded GPU drawing command. No full-frame host
raster or scientific image refresh is added. Both lanes retain the 20-second
window and default 10 Hz cadence. Optional playback lag remains in the header.

Both lane titles and the `20s` / `Now` bookends use identical, brighter glyph
metrics within each layout: 16px in the full history and 14px in the compact
history, slightly above the adjacent panel's body text and using the native monospace font. Titles stay above
and below the tracks; time captions sit beside the tracks at their vertical
midpoint. Reserved side gutters keep time captions outside connection geometry.
The docked timeline lays out its middle and history panels at the actual footer
width, retaining native glyph proportions through fullscreen and restore.
X11 paints completed footer pixels one-to-one rather than stretching them;
scientific image interpolation is unchanged. Rendering remains native HoloViz
text composed into the GPU frame.

The next-image title uses 14px text below its processing checkpoint in both
footer layouts. Completion and receipt labels use matching 12px text; BP work
remains 14px. The panel omits countdown and elapsed-clock text so a held frame
reads as a checkpoint rather than a stalled timer.

### Autofocus outcome and strip sizing

The autofocus footer reports capture-stamped receiving, auxiliary formation,
estimation, final formation and image-ready checkpoints from application stage
boundaries. Correction accepted/declined is a separate detail, using the existing
aligned termination and accepted-iteration reports. Zero-copy boundary operators
forward original tensor samples without copying their storage or waiting on CUDA.
Intermediate checkpoints can be skipped during a GPU presentation stall; there is
no additional ETA or stage percentage. The oracle workflow skips estimation. Declined results use amber and retain the uncorrected image.
Zero accepted iterations without a refusal are reported as no correction
applied; the oracle diagnostic is labeled separately. Missing status never
implies acceptance. A new collection clears the prior outcome, and late reports
from an older collection cannot overwrite current status. No image transfer or
additional GPU synchronization is introduced by this status mailbox.

Automatic strip windows fit within both the available desktop work area and
`app.visualization.max_side` (1400px in the strip presets). The X11 virtual
desktop can span several physical screens; its combined width is no longer the
initial window width. Explicit window dimensions retain precedence, and users
can resize normally. Formation geometry and dump products are unchanged.

The completed-image HUD selects BP measurements from a fixed 64-entry host
history using the displayed image's capture stamp. A newer BP completion cannot
blank the matching older mosaic's metrics. Evicted or unknown stamps remain
unavailable rather than borrowing a different image's work or timing. Continuous
strip labels its full product dimensions as a mosaic and adds `Last strip update:
<N> pulses`; BP work, latency and throughput describe that update's BP window,
not the full mosaic. These rows remain cached with the image between updates;
no extra image readback, GPU synchronization or timer-driven redraw is added.

Continuous-strip diagnostic history explicitly maps the stager's distinct slice
stamps to their source packet. Every proven slice contributes to completion and
exact pulse-time budgets; a processed history tick still means completion of
one original reader batch, rather than one slice. Mapping uses one bounded
interval per retained packet and changes no processing timestamps or joins.

The three footer panels share a black background and an 8px text inset. The
completed image title uses an 18px cyan heading aligned with the middle panel's
headline. Dimensions and metrics use brighter, consistent secondary/primary
colors. This theme changes only cached text and timeline drawing commands;
scientific images and their refresh/upload behavior are unchanged.

### Docked footer resize bounds

The pinned SDK's CUDA sink extent is immutable. Interactive docked SAR viewers
reserve a rectangular footer canvas of `max(initial_footer_width, 8192)` by
the configured footer height. The raster evaluates drawing commands only in
the visible region and clears the rest; image-layer textures use rectangular
bounds rather than allocating a max-width square. Headless and separate-window
layouts retain their configured canvas sizes. Windows wider than the bound
clip the footer at that bound rather than stretching text.

Window resize feedback uses an atomic width/height snapshot. Each GPU-drawn
footer records its exact capture stamp and layout in a 64-entry bounded
history. The X11 download copies only that frame's visible rectangle with a
pitched CUDA copy, preserving its capture time. A stalled frame that outlives
the history is dropped; the next current frame recovers without guessing at
another frame's layout. The presenter holds old pixels without stretching
until the resized GPU frame arrives. Native composition samples the footer in
physical window coordinates using the same resize feedback. Neither route
reruns formation, converts the scientific image, nor uploads its cached pixels
because of a timeline update. No user config migration is required.

### Sensor look annotation

SAR images label the projected direction of the radar platform with **Sensor look
angle**. A short cyan arrow points inward from the sensor-facing image edge;
subtle schematic rays suggest viewing direction, not a measured beamwidth or an
incidence-angle value. The native HoloViz text sits on a dark backing. Pixel-based
layout keeps the cue compact on wide strips and clamps the label inside the image.
Invalid directions and very small image canvases omit the annotation.

The backing and colored lines touch only a small display-raster patch on the GPU
when a new image is rendered. Text and patch remain cached with that image and
share its capture stamp; timeline updates do not repaint them. Scientific image
and dump tensors are unchanged.
