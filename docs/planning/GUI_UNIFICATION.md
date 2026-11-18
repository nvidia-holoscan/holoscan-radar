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

The docked SAR default moves image-specific text into a compact status panel to
the left of the timeline. Its source width is at most 320 pixels and half the
image canvas; the timeline uses the remaining width. Scene, product dimensions,
backprojection timing/rate, last-pulse latency, and paced playback lag use a
restrained text palette. Long labels use ellipses; narrow windows crop the
panel rather than shrinking or stretching its text. The APC direction marker
remains over the SAR image. No new config setting or migration is required.
Separate-window and timeline-disabled layouts retain their previous HUD.

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
