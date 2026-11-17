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
