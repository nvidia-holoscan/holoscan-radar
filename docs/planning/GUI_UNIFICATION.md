# Radar application GUI unification

Implementation branch: `deustice/sar-gui`, based on `deustice/sar-python`
at `3e01e150`. The behavioral baseline is the merged implementation of
[PR #69](https://github.com/nvidia-holoscan/holoscan-radar/pull/69), merge
`9cebac3c5fda84131e620b0438bcaca834911971`. Earlier font and presenter experiments
in its discussion are superseded by GPU composition with SDK text and bounded
finished-image transport. Public signal-processing operator contracts are unchanged.

## Diagnosis and parity

SAR previously selected a native presentable Vulkan device before creating its
formation and timeline windows. A valid SSH/X11 connection does not imply that
the Thor Vulkan adapter can present to that display. Both SAR windows now use
GPU offscreen composition before selecting their transport. X11 mode never
requests a native Vulkan surface during discovery or window construction.
A separate remote desktop is unnecessary for that mode.

| Capability | Previous FMCW | Previous SAR | Unified behavior / classification |
| --- | --- | --- | --- |
| Finished-image X11 | GPU composition, bounded host download, XRender | Missing | Shared across all 13 applications and both languages |
| Native window | GPU composer plus native image window | Direct native formation and timeline windows | Shared terminal image presenter; requires a CUDA/Vulkan presentation-capable display |
| Headless / off | Off worked; headless selection was inconsistent | Offscreen / disabled options | Shared explicit `headless` and `off`; no display probe or preference prompt |
| Presenter policy | Prompt, V2 remembered choices, expiration/reset | Missing | Shared existing policy and preference keys; no cache migration |
| Launch / programmatic options | Reset flag, no common mode override | Positional YAML/CPHD | Shared `--presenter`, `--choose-presenter`; matching Python keyword arguments |
| GPU discovery / transport | Offscreen discovery plus native capability probe, device bridge | Native probe required | Shared discovery; X11 bypasses native surface probe; existing cross-device bridge retained |
| Text / overlay alignment | GPU image and SDK text composed together | Native layered scientific view | Shared exact-timestamp composition, native text; complete image transported as a unit |
| Resize / aspect | Stretch | Stretch | Shared stretch default, optional fit; configured bounded canvas remains fixed |
| Resolution / display cap | Dashboard dimensions and cap | Decimation, no common cap | Shared dimensions and cap; SAR max-side decimation remains specialized |
| Color scaling | Noise-anchored viridis and peak fallback | Percentile grayscale | Intentionally different; original scientific scaling retained |
| Windows / placement | Dashboard | Formation plus bottom-left timeline | SAR defaults to one window with a docked timeline; explicit separate-window mode remains; both panels use the selected transport |
| Close / Escape / signals | Window/session controls | Multiple windows with producer-coupled servicing | Shared cooperative stop from the docked view or either separate SAR window; events continue during idle hold |
| Loop / EOF / products | Finite simulated frame count | CPHD loop, dwell flush, report/dump drain | Specialized completion rules retained, followed by shared bounded display drain |
| Final hold | Zero | 30 seconds or preset override | Intentionally different defaults; negative holds until closed; headless ignores GUI hold |
| Inheritance / config | Recursive SDK merge | Direct file loads in several paths | Shared inherited loader, including autofocus and Python sizing reads |
| Diagnostics / tests | Presenter diagnostics and FMCW window probes | No X11 terminal coverage | Shared mode diagnostics; all-app window/headless matrix plus specialized SAR parity tests |

## Boundaries and decisions

- `app/config_loader.hpp` owns inherited configuration and per-layer SAR alias
  normalization. `app/visualization_config.hpp` owns common settings.
  `app/presentation_config.hpp` resolves launch options, policy, preferences,
  display availability, and GPU discovery once. Python calls this same resolver.
- `viz/frame_composer_op.hpp` owns exact image/text/optional-geometry composition
  into GPU RGBA frames. FMCW dashboards and SAR formation/HUD retain separate
  producers, scaling, layout, and scientific interpretation.
- `viz/presentation_graph.hpp` owns headless consumption, native completed-frame
  replay, best-effort queue wiring, and cross-device presentation. X11 uses the
  existing bounded download and `x11_image_presenter_op.hpp`. `image_fit.cu`
  applies native fit on the GPU; X11 fits the completed image through XRender.
- `sar/timeline_raster.cu` rasterizes bounded timeline commands on the GPU and
  supplies SDK text separately to the shared composer. It never downloads
  apertures or full-resolution complex products to draw a window.
- `app/gui_lifecycle.hpp` holds application-private progress tracking. FMCW's
  driver/BuiltGraph and SAR's runtime preserve their own completion conditions,
  then allow up to ten seconds for the final completed frame before stopping.
  SAR explicitly waits for report and dump completion before GUI hold.
- `app/shutdown_signals.hpp` owns the shared C++ SIGINT/SIGTERM dispositions.
  A lock-free signal flag is polled by each driver's control loop, which calls
  `request_stop()` on a caller-managed HSDK session. No HSDK/CUDA work runs in
  the signal callback and no thread-mask adoption is required. This permits
  profiler-created startup threads. Window-close/Escape stop requests stay in
  the presenters; Python retains its own signal handlers. Cleanup waits for
  in-flight GPU work. Actual CUDA/NVTX captures and X11 signal tests passed on
  2026-11-19; native interactive presentation remains unverified on SSH X11.

Canonical settings live under `app.visualization`. SAR's `visualize`, `headless`,
`skip_display_check`, `visualize_size`, and `visualize_hold_s` remain aliases.
Normalize each inheritance layer before merging: a child alias overrides a
parent's common spelling; the common spelling wins a conflict in the same file
with a warning. Shipped presets use canonical spelling; legacy fixture presets
continue testing compatibility. Unspecified SAR canvas dimensions follow the
existing max-side-derived aspect ratio.

Precedence is explicit launch mode, then YAML, then a valid remembered choice,
then the existing interactive/unattended default. `--presenter` bypasses reset
and preferences for that run. Headless/off never prompt or persist choices.
Both modalities share the existing V2 setup key, one-run, seven-day, and
until-changed choices, expiry validation, and `--choose-presenter` reset.
SAR visualization stays opt-in when absent; FMCW enabled defaults are retained.

All display queues are bounded and drop old completed frames. Native replay
retains one GPU frame; X11 retains one packed finished image. Independent clock
activations service events before the first frame, during stalls, and at EOF.
Completed-frame inputs drain bounded arrival queues rather than rejecting old
acquisition timestamps. Replay preserves capture time and provenance; exact
image/text/look joins occur upstream. This also avoids falsifying SAR timing
for a long aperture formation. HSDK 5 `DrainAll` requires an unspecified missing
policy: an empty batch is valid and keeps the event clock running.

Retaining SAR's direct native visualization while adding a second X11 renderer
was rejected because it duplicates composition and lifecycle semantics.
Rewriting SAR as an FMCW dashboard was rejected because it erases scientific
views. A universal data-plane descriptor and CPU plotting path were rejected:
application-private helpers suffice and preserve the locked tensor contracts.

## Validation and limits

Use the pinned HSDK 5 GA image and host CLI. Automated Xvfb tests are real X11
windows with pixel, geometry, resize, and shutdown observations; they do not
establish laptop forwarding quality or scientific visual correctness.

| Validation | Evidence / remaining requirement |
| --- | --- |
| Numerical and operator regressions | All 1,455 non-performance CTests passed in 377.72 seconds on Thor, excluding native-window cases requiring a presentation-capable display |
| Final affected coverage | All 149 focused CTests passed, including provider pytest, all six SAR X11 windows in both languages, all eight FMCW X11 presets in both languages, all 14 C++ headless cases, configuration, scientific/dump and shutdown cases |
| Timestamp / GPU fit | GPU tests cover a frame carrying a 30-second-old acquisition stamp, provenance retention, and fit borders |
| Delayed display / resources | C++ and Python SAR video soaks passed with 10 ms per-chunk delay for ten seconds. RSS: C++ 327.2→327.0 MiB, Python 564.4→564.2 MiB; FDs: 148–150 and 145. Processing advanced from image 1 to 10,646 / 9,803. C++ and Python SIMO delayed resize passed at five sizes |
| Native presentation | Explicit native probes fail on both Thor/Xvfb and the working laptop forwarding connection because no CUDA-backed adapter can present; positive native window/fit checks remain required on a compatible display |
| Actual SSH/X11 | All 28 real SSH/X11 cases passed on `localhost:10.0`: all 13 applications in both languages, including both DDM variants; application pixels, resize, cooperative closure, SAR two-window/fit/Escape coverage |
| Manual visual review | Captured synthetic formation/HUD and timeline images inspected; real-collection scientific layouts, mosaic orientation/crops, autofocus quality, and native presentation still require human visual review |

Positive multi-GPU native transport and long-duration memory/performance
qualification remain hardware-specific checks. Short diagnostic soaks do not
establish a production latency bound. Test commands and the complete registered
matrix are in [the GUI test guide](../../tests/gui/README.md).

Final automated execution used `holoscan-radar:ngc-v5.0.0-cuda13` on Thor.
The selected full run excluded only performance-labeled and native-window
cases. All 149 focused CTests also passed independently before that run.
Repository-wide pinned pre-commit hooks pass. Captured synthetic formation/HUD
and timeline images were inspected; this does not replace real-collection and
laptop visual review. Raw logs and captures remain local ignored build artifacts.

The implementation is split into configuration/policy, shared GPU presentation
and SAR lifecycle, Python facade parity, and regression/documentation commits.
The existing native presenter and composer class names remain compatibility
aliases for app-private callers; no public operator or shared descriptor was added.

### Forwarded display follow-up

On 2026-11-17 the user identified `localhost:10.0`. Host and project-container
connections succeeded. The 28 window probes ran directly on that laptop display,
without Xvfb or a separate remote desktop, and passed in 137.87 seconds of summed
case time. The SAR probe exercised formation closure in C++ and timeline Escape
with aspect fit in Python. The same display rejected explicit native SAR startup
with the expected Vulkan capability error and X11 guidance.

The first forwarded probe exposed a test limitation: a real window manager can
reparent the timeline beneath decoration windows. The shared probe now searches
bounded child trees under an atomic X11 snapshot, with a 512-window budget,
bounded depth, and 0.5-second traversal deadline. Pixel checks wait for
mapping and inspect application pixels, so laptop wallpaper cannot satisfy them.
All 28 Xvfb window cases plus the fixture (29 CTests) passed again after these
test-only changes. No production GUI code changed during this follow-up.

## Docked timeline follow-up

SAR now defaults to a single window with a dedicated bottom timeline strip.
The agreed strip height is 112 physical pixels, independently updating at the
existing 10 Hz default. The image takes the remaining window height and retains
its own fit/stretch policy. Very short windows clamp the strip to half the
available height. `app.visualization.timeline.placement: window` preserves the
previous separate-window layout; disabling the timeline removes the strip.
The docked raster uses the image canvas width; the old timeline width setting
continues to control the separate-window layout. Image saving is unchanged.

`viz/x11_docked_window.hpp` manages one X11 parent and two child surfaces. Each
child reuses the existing finished-image presenter and independent download
queue/worker. A timeline-only update uploads and repaints only that child.
The shell only handles layout and events; it never composites a whole display
frame. Parent resize/expose can repaint cached panels, without reallocating
render pools or rerunning scientific rendering. The entire view stops on close
or Escape in either panel.

`viz/docked_frame_op.hpp` handles native presentation. It retains one finished
GPU image/HUD and one finished GPU timeline, admits them by independent arrival,
and composites a bounded combined GPU display frame for the native SDK window.
The user explicitly accepted that native HoloViz must redraw its whole window
on refresh. This does not rerun SAR image formation, grayscale conversion, HUD
creation, or a main-image host download. The kernel compensates for actual window
size to keep the footer height fixed. Native window refresh retains the existing
display-rate/minimum-event-pump cadence, independently of 10 Hz timeline content.
Size compensation uses the existing X11 extent helper; without accessible
window extents it falls back to configured dimensions. Native display frames preserve the image's
capture time and provenance; the diagnostic timeline remains independently live.
Headless rendering still consumes both completed panels independently.

GPU tests cover footer placement, fit, blank image startup, and different old
panel timestamps. X11 window tests assert shared parentage, fixed footer height,
image and timeline pixel recovery after resize, and Escape/close. They also read
per-panel upload/repaint counters: after EOF, the timeline must advance while
both image counters stay unchanged. Explicit separate-window tests cover C++
and Python compatibility. The 28 real SSH/X11 workflow/language cases passed
again with the docked layout. A larger delayed-display soak retained stable RSS
and file descriptors while processing continued. Native positive window review
still requires a compatible display; no native visual success is claimed here.

The final affected suite passed all 154 CTests serially in 166.17 seconds,
including provider tests, numerical/dump parity, both precision paths, all SAR
X11 panels, and explicit separate-window compatibility. The first parallel run
hit a cold-start unpaced tiled-strip source backpressure stop; the isolated
rerun and complete serial run passed without changing processing queues,
playback behavior, or tolerances. The refreshed SSH/X11 matrix passed all 28
workflow/language cases on the laptop connection. New GPU docking tests also
passed without any native presentation surface. A 1024-by-768 image plus footer
delayed-display soak (10 ms/chunk, ten seconds) advanced from image 1,459 to
13,862 while RSS stayed about 339.6 MiB and file descriptors stayed at 149.
A preview capture must wait for both panels to recover after resize: a visible
footer alone cannot establish that the image finished repainting.

The corresponding Python fit/Escape soak passed with RSS fixed at 576.9 MiB,
146–148 file descriptors, and processing advancing from image 2,529 to 14,131.
The 1024-by-768-plus-footer laptop capture was inspected after explicit recovery
of both panels. Pinned repository-wide lint passes. These short fixture soaks
establish recovery and bounded growth for the exercised path, not long-duration
production latency or real-collection image-quality bounds.

The actual native presentation helper also compiles successfully through HSDK's
Graph compiler without starting a window: this checks its partitions, typed
viewer ports, and device placement. All seven render/configuration rechecks
passed, including that additional native wiring test. This is distinct from
positive native visual validation, which remains unavailable on this display.

## Docked status panel follow-up

The latest middle-panel review separates the displayed image title on the
left from the forming image's title in the middle. A subsequent label review
removes title prefixes; completion remains indicated by the ETA's `Image ready`.
Next-image ETA remains the largest text. A shared track shows GPU-completed
Formed pulses as green fill and Received pulses as a thin cyan marker, with
both percentages above and elapsed time below. The marker is drawn last and
extends beyond the track height to stay visible when the frontiers coincide,
including at zero and completion. Narrow layouts put title/ETA on one row
and use a 56-pixel strip for the four compact rows within the existing footer;
this does not change the scientific image canvas. Unknown progress stays `--`. This replaces the earlier
two-bar layout described below without changing progress semantics, the
1 Hz diagnostic refresh, 10 Hz timeline cadence, caches, or image transport.

This panel passes all 81 selected GUI regressions, including GPU-rendered
pixels/native text, distinct received/formed percentages, title selection,
wide/compact layouts, all six C++/Python X11 workflows under Xvfb, headless,
startup, resize, shutdown, and separate-window compatibility. The initial run
exposed a startup probe reading a named child before it was viewable; the probe
now checks map state and bounds atomically around its pixel read. All four
startup cases then passed three consecutive runs and the full 81-check suite
passed again. Pinned all-file lint passes. Logs/scripts are under
`build/holoscan-radar/gui-sar-next-panel/`; this adds no laptop SSH or positive
native-window visual claim. Processing kernels and scientific products were
not changed.

The subsequent shared-track change retains those caches and update rates.
Its GPU regression checks unequal frontiers, coincident frontiers, zero, and
100 percent at both widths, including marker position and visible overlap at
completion. Startup handoff probes sample the formed fill and the received
marker at their completed edges.
All 81 selected GUI checks and pinned all-file lint pass after this change.
Evidence/scripts are in `build/holoscan-radar/gui-sar-shared-track/`; actual
window tests use Xvfb, with laptop/native visual review still pending.

The docked SAR default moves image-specific text into a compact status panel to
the left of the timeline. Its source width is at most 320 pixels and half the
image canvas; the timeline uses the remaining width. Scene, product dimensions,
backprojection processing time, and throughput use a
restrained text palette. Long labels use ellipses; narrow windows crop the
panel rather than shrinking or stretching its text. The APC direction marker
remains over the SAR image. No new config setting or migration is required.
Separate-window and timeline-disabled layouts retain their previous HUD.

Fullscreen/resize reflows the middle and right panels at physical pixel sizes.
Interactive docked viewers use a bounded rectangular offscreen footer canvas
because the pinned SDK CUDA sink cannot resize. A 64-entry capture-stamp/layout
history makes X11 crop each completed frame using its own extent, including
after a delayed display; expired frames drop safely. Only visible footer pixels
are sent to X11; existing cross-device bridges retain bounded completed-frame
transport. Cached SAR images remain independent of footer ticks. Native
composition samples the same footer in window pixel coordinates.

The subsequent manual review settled the left panel on five rows:
title, product dimensions with `pixels` and grouped measured pulse count,
`BP work: <work> Gpx*pulses`, `Latency: <time> / <budget> @<rate>x`, and
`Throughput: <rate> Gpx*pulses/sec`. Latency is explicitly the accumulated GPU
backprojection work time, excluding acquisition/pacing waits. Throughput uses
the same time denominator and measured BP pixel-pulse counts, with a fixed giga
unit; the scientific product dimensions may differ from the BP lattice. BP work
uses that same measured pixel-pulse numerator, including incremental work for
periodic outputs. Five rows fit the existing footer without shrinking the image.
This replaces the completed-image elapsed-time hierarchy described below.
The middle progress/ETA panel also shows expected current-image BP work from
its matching processing grid, with unknown work shown as `--`. Collection wall
time is labeled Total time at the bottom (wide) or beside the ETA (compact);
compact BP work shares the title row with safe title truncation. Image metrics use the
selected image's capture timestamp rather than the interactive GUI clock;
missing or newer measurements show `--` instead of a different image's cost.

The Latency budget uses the image's original radar interval divided by its
configured positive playback rate, replacing the initial fixed-1x comparison.
Whole-collection images use the selected first-to-last pulse span. Periodic
images use the interval between successive image-closing pulse `TxTime` values,
including an irregular or partial final cadence, rather than the full retained
aperture; the first image starts at the collection's first pulse. Duty-cycle
quiet time is excluded. The rate is captured alongside that image's budget,
not sampled from the live reader after it advances to another collection.
For example, a 2.4-second radar interval gives a 12-second budget at 0.2x and
shows `Latency: 8.2s / 12.0s @0.2x`. Unpaced playback (`playback_rate: 0`) shows
`-- unpaced`, with no fixed deadline; missing or invalid rates show `-- @--`.
Measured GPU time and throughput stay unchanged. This compares backprojection
cost with its required update interval; it does not certify end-to-end real-time
performance.
Existing host timestamps supply these intervals without extra I/O, device
transfers, or GPU waits. The optional progress mailbox retains at most 4096
timestamps per packet across 128 packets (4 MiB of pulse-time samples), with
endpoints only for larger packets. Missing history, unavailable internal seams,
and invalid/nonpositive intervals remain unknown; known seams restore tracking.
The budget follows the image's capture stamp through the existing host HUD
diagnostics. Public operator ports, parameters, processing, and the middle
panel are unchanged. No configuration setting or migration is needed.

The playback-rate refinement passes 157 selected CTests and 63 SAR Python
checks, including slower/faster pacing, unpaced playback, unavailable and
out-of-range timing, frozen per-image rate propagation, image-matched GPU text
in both clocked and event-driven paths, and the six-workflow X11/headless and
numerical/dump checks below. Pinned all-file lint passes. Logs and the script
are in `build/holoscan-radar/gui-sar-playback-budget/`. Window checks in this
run use Xvfb; laptop and positive native visual review remain pending.

The initial 1x budget follow-up passed 156 selected CTests and 63 SAR Python checks.
Coverage includes exact irregular pulse-time intervals at different playback
rates, multiple image seams within a source packet, partial cadence, collection
loop reset, oversized-packet timing limits, lost-history recovery, real reader
and backprojection diagnostic propagation, image-matched native GPU text, and
unchanged scientific pixels. All six workflows pass automated C++/Python X11
window checks under Xvfb, headless runs, and numerical/dump parity in both
precisions, with resize/shutdown and separate-window coverage. Pinned all-file
lint passes. Evidence and the reproduction script are in
`build/holoscan-radar/gui-sar-budget/`. This does not add a new laptop SSH visual
review or a positive native-window presentation result; those remain manual
validation items.

The four-row card passes 80 selected checks, including exact row contents and
unchanged scientific pixels through the GPU composer, matching and newer
diagnostics under both event-driven and clocked startup, all six C++/Python
X11 workflows, separate-window compatibility, startup handoff, headless
rendering, and resize/shutdown. Pinned repository-wide lint passes. Middle-panel
rendering and numerical operator code are unchanged in this follow-up.
Logs and the reproduction script are under
`build/holoscan-radar/gui-sar-card/`; manual visual review is still pending.

The timing hierarchy now puts next-image ETA and live collection elapsed in
the independent progress panel, and frozen collection-start-to-formation wall
time first in the completed-image status. GPU time/rate and final-batch latency
are secondary. Paced playback lag stays in the live timeline; image completion
ticks retain their positions but omit numeric latency labels. The larger
primary text and smaller diagnostics fit the existing 112-pixel footer using
the actual SDK native text renderer, not the generated design mockup's fonts.
The original compact progress preserved both bars and combined ETA/elapsed;
the latest middle-panel review above replaces that arrangement with one bar.
Elapsed uses the matching bounded collection record and the existing GPU
completion observation; it never adds a GPU wait or includes display hold.
Periodic images share collection start, and a loop resets that start. Missing
history omits timing. Separate-window/on-image latency is renamed consistently.

The timing follow-up passes 150 selected CTests and 63 SAR Python checks in the
pinned HSDK 5 GA GPU image. Coverage includes frozen elapsed during hold,
collection-loop reset, periodic-image clock semantics, declining a newer
image's elapsed time for a delayed older image, native GPU text composition,
cached native docking, all six C++/Python X11 workflows, separate-window
compatibility, startup/first-image handoff, resize/shutdown, headless rendering,
and complex-product/dump parity in both precisions. Repository-wide pinned
lint passes. These are automated checks; the user will review the real viewer's
appearance manually. No new SSH/laptop visual capture or positive native-window
review is claimed for this follow-up, and the generated mockup is not rendering
evidence or an application asset. Logs are under
`build/holoscan-radar/gui-sar-timing/` in this checkout.

Status is rendered with the exact-timestamp image/geometry join into extra
footer rows of the same completed GPU frame. This deliberately avoids a
latest-status side channel, which could advance labels beyond the displayed
image under display drops. The X11 image/status child spans the parent and
paints the two source regions independently; the narrower timeline child sits
above its unused footer-right region. Timeline updates do not repaint or
transport the image/status frame. Resize/expose repaints use cached pixels.
The fixed extra footer rows add bounded completed-frame transport only on
new image arrivals. Native composition reads the paired image/status frame
and independent timeline from GPU caches. Its existing full-window refresh
remains unchanged. Scientific rasters and saved products are unchanged.

The affected selection covers 195 CTests: SAR numerical/dump checks, provider
parity, both precision paths, headless rendering, shared FMCW presentation,
all twelve SAR X11 workflow/language cases, separate-window compatibility,
GPU image/status composition, cached native panel placement, and lifecycle.
194 passed initially; the sole failure was a sparse pixel grid missing a thin
APC marker on the flat spotlight fixture after aspect-fit resize. A dense
fallback resolves that assertion, and the fixture plus failed case pass on
rerun. The new GPU test uses a patterned image and checks every pixel/channel
above the footer, long-label containment, and visible native footer text.

Repeated C++/Python X11 soaks at a 1024-by-768 image canvas, with 10 ms/chunk
delay for ten seconds, pass. Final RSS medians were 340.5→340.5 MiB (C++) and
574.4→574.4 MiB (Python); descriptor ranges were 149–151 and 146–148. Processing
advanced from image 5,351 to 17,593 and 5,354 to 17,483. Settled synthetic
captures were visually inspected. Capture now waits for status pixels at the
new footer origin: window geometry alone can advance before a delayed cached
repaint finishes. These checks establish bounded recovery, not a production
latency guarantee or real-collection image-quality qualification.

Direct C++ and Python runs on the forwarded `localhost:10.0` display initialized
and held the docked view, then stopped cleanly. The five-second Python hold
logged one image upload and 52 timeline uploads. A 640-by-480 run logged one image upload
(1,515,520 bytes including the footer) and 430 timeline uploads over its hold,
confirming independent transport at the default 10 Hz content cadence. Remote
window-query and pixel-readback probes repeatedly stalled, including checks of
the existing separate-window layout; complete forwarded interaction and pixel
qualification is therefore not claimed. Positive native-window visual review
still requires a compatible display. Repository-wide pinned lint passes.

## Startup and typography follow-up

The approved UI retains image titles in the left status panel and moves each
collection name above its corresponding scrolling timeline bar. Collection and
image-emission bars are thinner; both label rows fit the existing 112-pixel
footer. Sidebar titles increase to 16 pixels and details to 14. Display-only
throughput conversion replaces scientific notation with k/M/G/T prefixes,
while the console retains its existing measurement string. The GUI converts
that same held string rather than sampling a later BP measurement.

`SarImageVizOp` now optionally uses a clock/DrainAll selector for interactive
startup. With no image yet, it emits one 1-by-1 opaque charcoal GPU background
and bounded native SDK text records. The existing exact image/text/geometry
composer expands those into the configured display canvas. Subsequent idle
clock activations return immediately; they never rerender or redownload the
placeholder. After input arrives, the newest queued display image is rendered
with its original capture time and provenance, and startup never reappears.
Display admission remains best-effort and bounded. Headless graphs retain the
original OnEach selector and emit no startup frame. No public operator contract
or scientific port changes.

Startup display chrome uses capture time -1, the existing unset-presentation
sentinel. It therefore cannot satisfy the EOF display-drain check or masquerade
as an acquired image. All three composer inputs carry that same stamp. The
clocked GPU tests exercise this sentinel through the real SDK composer and
verify the handoff preserves the real image's stamp. Numerical reporting and
dumping branch before this private display operator.

The timeline's existing small-frame updates show elapsed time and a truthful
startup stage at 1 Hz from reader/BP diagnostics and presentation progress.
Stage changes are not inferred as numerical progress percentages. Names above
bars stay tied to their collection spans; the sidebar stays tied to its image.

Validation in the pinned HSDK 5 GA image:

- All 202 selected non-performance CTests passed, covering the Python provider,
  SAR numerical/dump/headless checks, all six C++/Python X11 workflows, existing
  FMCW presentation checks, and startup composition/handoff/closure checks.
  The subsequently added GPU timeline test also passed, bringing the total to
  203. It locates the collection label and colored bar and verifies that the
  glyphs sit above the bar rather than relying on hardcoded layout coordinates.
  After the host reboot, all nine focused fixture/startup/rate/timeline checks
  passed again in the same pinned image.
- Startup GPU checks verify one placeholder emission, original real-image
  timestamps, charcoal/native-text composition, and display-only SI formatting.
  Four Xvfb window checks cover handoff and closing before the first image in
  both languages. Existing resize/EOF checks now wait for an actual formed
  grayscale image so the placeholder cannot satisfy their assertions.
- Ten-second C++/Python X11 runs with 10-ms delayed transport at 1024-by-768
  continue processing through resize/recovery and shutdown. RSS remains
  341.0 MiB for C++ and 574.2 MiB for Python; FD counts remain 149 and 146–148,
  respectively. Startup and settled fixture screenshots were visually inspected.
- Direct laptop-forwarded `localhost:10.0` runs in both languages complete
  startup, first-image handoff, and final hold. Each logs two completed-image
  uploads and 61 independent timeline uploads: timeline ticks do not rerender
  or redownload the main image. This is execution/transport evidence; complete
  SSH pixel/interaction review is still unqualified because cross-client remote
  pixel/geometry probes previously stalled. Positive native-window review still
  requires a display with Vulkan presentation support. Real-collection scientific
  image quality is outside the synthetic-fixture visual checks.

## Live pulse progress and next-image estimate

The approved follow-up shows separate Received and Formed pulse counts/bars,
plus an approximate wall time to the next image. The readout lives in the
independent timeline frame, beside history at normal widths and above a compact
history at narrow widths. It refreshes at 1 Hz. Neither readout changes nor
ETA ticks render/download the main image or its image-specific sidebar.

`FormationHud` optionally owns a pure host `sar_progress::Monitor`. The reader
announces bounded packet identity/ranges before publication and acknowledges
receipt after every required output succeeds. It reads only the selected first
and last TxTime PVP records for a collection-duration estimate; speculative
prefetch never advances received counts. The aperture manager records unique
admissions before its corresponding output, preserving the order of slices
that share a capture stamp. Signed evictions and reform replay never inflate
acquisition counts. Loop generations distinguish repeated files. Packet,
collection, and pending-work histories each have a fixed 128-record bound;
overflow discards ambiguous diagnostics instead of changing the data graph.

`radar_common/sdk/sar_gpu_progress.hpp` owns a fixed 16-pair CUDA event ring on
the actual compute device. Formation records each admitted output batch around
its existing GPU work. The timeline queries completion without waiting; the
producer also reclaims completed event slots when beginning a batch. A full
ring drops diagnostic tracking rather than backpressuring processing. Query
and lifetime ownership are independent of operator clones; cross-device polls
restore the caller's selected CUDA device. No stream callbacks, extra GPU
synchronization, or raw aperture/image downloads are introduced.

Only completed event batches advance Formed. A long single batch or a whole
reform replay can leave this bar stationary until that work finishes. The ETA
uses observed effective unique-pulse throughput for the image lattice and the
remaining acquisition deadline (including playback rate and first-batch-ASAP
release), taking the larger duration because acquisition and processing
overlap. Collection TxTime interpolation is approximate; irregular pulse times,
GPU contention, changed grids, and output/display overhead limit accuracy.
Unpaced acquisition uses its observed wall rate. Reform timing includes the
wait through auxiliary formation/autofocus to product formation, where present.
Without a matching measurement, the readout says Estimating rather than using
hardware-specific guesses. Image ready means formation completed, not that a
remote display or a saved-product write has finished. Numerical reports,
scientific color scaling, EOF drain, and saved products retain their contracts.

The shared runtime enables this only for interactive, timeline-enabled viewers
in both languages. No YAML defaults or user config files change. The optional
manager HUD constructor argument is source-compatible and documented in the
SAR contract; data ports, selectors, and parameters do not change.

Validation in the pinned GA image passes all 221 selected non-performance
checks, including the Python provider, SAR numerical/dump and headless cases,
C++/Python X11 windows for all six workflows, completed-frame transport, FMCW
regressions, aperture admission diagnostics, and the new progress/ETA tests.
The 19 focused GPU/model/layout/startup checks pass again after exact count
formatting. Window probes wait for both received and formed bars to fill on
the actual reader/manager/BP path before closing.

C++/Python 10-second 1024-by-768 soaks with 10-ms delayed X11 transport pass
resize, recovery, EOF, and closure. Processing advances from 6,716 to 19,769
images for C++ and 5,632 to 18,674 for Python during the delayed interval.
C++ RSS remains 345.93 MiB with 149–151 FDs; Python RSS is 581.51–581.64 MiB
with 146–148 FDs. These bounded soaks are functional/performance diagnostics,
not a controlled estimate of instrumentation overhead.

Startup, compact and wide footer, and formed-fixture captures were visually
inspected. A real Panama Canal CPHD preview uses a temporary one-eighth linear
pixel scale and the inherited 0.02 playback rate: it shows Received and Formed
1,024 / 9,693 with an approximate 1m 32s remaining. Closing before the first
image completes logs one main upload and 46 independent timeline uploads.
This qualifies live diagnostics and early closure on real input, not completed
real-product numerical accuracy or full-resolution performance. Original user
configuration files are preserved and excluded from the commit.

After forwarding was restored on `localhost:11.0`, direct laptop SSH/X11
checks passed in both C++ and Python: startup, first-image handoff, native-text
captures, and actual pixel reads of both completed progress bars. Additional
1024-by-768 forwarded-window checks passed resizing, recovery, EOF idle
behavior, window closure, and Python Escape with fit resizing. These checks
use the real forwarded X server without a remote desktop. Isolated screenshots
and real-input diagnostic captures received visual review; broader human
scientific-image review is separate. Positive native Vulkan window review
still requires a compatible display. Repository-wide pinned lint passes.

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

The common 20-second wall-clock axis scrolls at the existing timeline cadence
(default 10 Hz). Collection-colored dwell bars and image-emission ticks are
replaced by these two event lanes; the image title remains in the image and
progress cards. Optional playback lag stays available. ETA text remains at
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

The batch-history follow-up passes 164 selected CTests and 63 SAR Python tests
in the pinned HSDK 5 GA GPU image. Coverage includes complete source-batch
proof across slices, replay and evicted prefixes, collection generations,
receipt retries/publication races, bounded/coalesced history under bursts,
nonblocking queries with a blocked GPU stream, native GPU text composition,
all six C++/Python X11 workflows, headless rendering, shutdown, and unchanged
complex products/dumps in both precisions. Additional 12-second C++ independent
images and Python video SAR runs use a 10 ms X11 delay proxy, resize and capture
wide/compact footers, and verify continued formation and RSS/FD bounds. Median
RSS stays at 343.07 MiB for C++ and 572.72 MiB for Python; FDs span 149–151 and
146 respectively. These short fixture runs qualify bounded recovery, rather
than general application throughput or long-run memory behavior.

The stress probe now reads application logs through an independent descriptor,
avoiding changes to the child's write offset, and avoids querying already-closed
windows in failure capture. Evidence, reproduction scripts, resource samples,
and actual viewer footer captures are under
`build/holoscan-radar/gui-sar-batch-history/`. The captures confirm the final
coalesced history retains older activity under dense input. They use real GPU
rendering with Xvfb; no new laptop forwarding or native-window visual approval
is claimed. Manual appearance review remains pending.

The pulse-history follow-up uses thin tracks, short cyan/green markers, and
labels beside the tracks. A faint span connects receipt to observed GPU
completion for the same source batch. Retained unfinished batches extend their
shading to Now; they do not gain a processed marker until completion is proven.
Green is reserved for short completion markers on the lower track; connecting
spans never fill that track with green. Matching
happens before coalescing: independent received/processed ticks are never paired
by proximity or order. Overlapping matched intervals merge at 40 ms resolution
in a fixed 512-bin occupancy ring; pending intervals use the existing bounded
128-packet history. Discarded or unproven prefixes already passed by ordered GPU
work are omitted rather than shown pending forever. These spans describe
observed input-to-processing delay, including queueing, rather than GPU service
time. Both lanes keep the existing 20-second window and 10 Hz default cadence.
Only the timeline raster and its bounded drawing records change each tick;
completed SAR image/HUD caches and scientific processing remain unchanged.

Validation for paired shading passes all 169 selected CTests and 63 SAR Python
tests in the pinned GA GPU image. This includes controlled-timestamp host tests,
wide/compact GPU pixel and label alignment checks, all six SAR workflows in
both languages, headless rendering, GPU composition, shutdown, and numerical/dump
regressions. A real forwarded-window check on `localhost:14.0` passes resize,
continued image formation, capture, and closure; its sampled RSS stays at
342.98 MiB, with 149–150 FDs. Two further 12-second runs use a 10 ms X11 delay
proxy and Xvfb: C++ independent images holds RSS at 340.78 MiB with 149–151 FDs;
Python video SAR moves from 572.55 to 572.36 MiB with 146–148 FDs. Formation
continues in both runs. These are bounded fixture/recovery checks, not general
throughput claims. Actual forwarded and wide/compact viewer captures and
reproduction scripts are under `build/holoscan-radar/gui-sar-pulse-shading/`.
Native-window and real-collection visual qualification remain separate from
these automated and captured-viewer checks. No global Keeping pace/Falling
behind heuristic is introduced by this appearance follow-up.

The subsequent completion-marker correction removes the green horizontal fill
across completed receipt-to-processing spans. Overlapping spans had made the
processed track solid green even between distinct completion events. Connecting
shading remains neutral, and green now marks only observed completion events.
Wide/compact GPU regression checks assert that one batch completed several
seconds after receipt produces only one three-pixel green marker. All 10
targeted startup/timeline and independent-images/video-SAR C++/Python X11 checks
pass after the correction. A paced fixture on actual forwarded
`localhost:14.0` passes resize, continued formation, and closure; the viewer
capture confirms distinct ticks. Evidence is under
`build/holoscan-radar/gui-sar-marker-fix/`. User presets are unchanged.

### SAR matched diagonal follow-up

Replace receipt-to-completion area fills with thin neutral diagonal connectors.
Place larger received/processed labels above/below the tracks, reclaiming the
side-label column for history; compact history keeps time labels in small side
gutters. The previous occupancy union cannot preserve batch identity, so it is
replaced by a fixed 512-slot history retaining one exact actual pair per 40 ms
completion-time bin. Endpoints are never combined across batches. Clip old
receipts geometrically, preserving their slope. Pending batches retain their
receipt ticks without an invented completed endpoint at Now. At most 64 actual
representative pairs are drawn after identical projected pairs merge; all
independently coalesced event ticks remain. Each connection is one bounded GPU
command, evaluated directly by the existing timeline kernel. The default 10 Hz
cadence, 20-second window, scientific processing, image/HUD caches, public
operator contracts, and user presets remain unchanged. No global pacing-health
heuristic is added.

Diagonal-history validation passes all 170 selected CTests and 63 SAR Python
tests. Host checks verify exact overlapping pairs, pending/partial/replayed
batches, publication races, and bounded histories. GPU checks verify sloping
thin lines without interval fills, geometric clipping, pending work without a
fabricated completion, and larger labels outside both tracks in wide/compact
layouts. Broader coverage includes all six C++/Python workflows, headless
rendering, native GPU composition, shutdown, and unchanged numerical/dump
products. Actual forwarded `localhost:14.0` passes a paced fixture with resize,
continued formation, capture, and closure; sampled RSS holds at 335.23 MiB with
149 FDs. Two further 12-second Xvfb runs with a 10 ms X11 delay proxy exercise
wide C++ independent images (342.47 MiB RSS, 149–151 FDs) and compact Python video
SAR (572.24 to 572.36 MiB RSS, 146–148 FDs), with continued formation and bounded
resource checks passing. Actual viewer captures, samples, logs, and reproduction
scripts are under `build/holoscan-radar/gui-sar-diagonals/`. These fixture checks
qualify rendering/recovery; native-window and real-collection visual review
remain separate, and no general throughput claim is made.

### SAR typography and time bookends

Both titles were already authored with the same size, but fractional native-font
scaling and X11 resampling could soften their appearance. Use consistent integer
font scales, with 26px full-history and 13px compact-history titles and time
captions. Increase text contrast. Replace the small bottom time captions with
`20s` and `Now` in reserved side gutters, vertically centered beside the tracks.
Lane titles remain above/below the tracks. Use nearest sampling only for the
docked X11 timeline's completed GPU frame, avoiding another smoothing pass at
resize; scientific image interpolation and cache independence remain unchanged.
The native GPU text regression compares actual pixels of the common title
prefix, alongside exact size and time-bookend position checks.

Typography validation passes 170 selected CTests and 63 SAR Python tests. The
native compositor renders the common `Pulse batch` prefix consistently for
both titles in wide/compact layouts; maximum per-channel pixel difference is
bounded at two levels. Header checks require all four captions to share size,
with time bookends outside the plot and vertically centered beside it. Actual
forwarded `localhost:14.0` passes paced resize/capture/closure checks with RSS at
334.04 MiB and 149 FDs. Two 12-second Xvfb runs with a 10 ms delay proxy pass
wide C++ independent images (341.75 to 341.88 MiB, 149–151 FDs) and compact Python
video SAR (572.68 to 572.62 MiB, 146–148 FDs), with continued formation. Evidence
is under `build/holoscan-radar/gui-sar-text/`. HoloViz's existing font remains;
this change removes additional X11 resize smoothing without introducing a font
replacement or claiming native-window visual approval.

### Footer spacing refinement and workflow review (2026-11-18)

Reduce all four history captions together to 24px in the full layout and 12px
in compact history. Increase the next-image title from 12px to 14px and place
it below the ETA. The compact title occupies its own full-width row; elapsed
moves beside the ETA, with pulse counts and their shared bar below the title.
Preserve the existing ETA calculation, cached images, and presenter behavior.
Update both GPU layout assertions and actual-window handoff pixel coordinates.

Ten targeted native-text/layout and C++/Python X11 startup checks pass, alongside
63 Python checks. Workflow review uses actual forwarded X11 captures; it does
not establish native-window appearance approval. The shared footer omits
modality details: autofocus stage, iterations, sharpness and refusal status;
strip mosaic coverage and active tile/window context; and video aperture/hop
context. In particular, BP compute cost and ETA do not measure the entire
autofocus pipeline. Those semantic additions need a separate UI decision.

A second requested reduction sets full-history captions to 22px and compact
captions to 11px, keeping lane titles and both time bookends matched.

### Autofocus outcome and continuous-strip corrections (2026-11-18)

Expose the existing autofocus termination/accepted-iteration reports in the
middle footer, with amber declined status and explicit no-correction, unknown,
and oracle cases. Before a result, show preparing/running status rather than an
ETA that excludes auxiliary formation and estimation. Reset on collection
change and reject late prior-collection reports. The app-owned report already
reads the tiny diagnostics; the shared mailbox adds no device transfers.

Continuous-strip slices use explicit base+slice-index stamps. Register their
bounded interval with the diagnostic monitor before emission, preserving exact
source-packet completion and pulse-time budgets without changing data-plane
stamps or temporal contracts. The reported missing history was also explained
by Now being offscreen: cap automatic strip window size at the configured
visualization maximum while fitting the desktop work area. Explicit YAML window
sizes and manual resize behavior retain precedence. Collection numbering and
other proposed footer additions remain out of scope.
