# Radar GUI checks

## Registered CI smoke coverage

The default GPU CTest dashboard includes these checks for enabled applications:

| Path | Coverage |
| --- | --- |
| C++ without visualization | All eight smoke presets complete two frames, including linear and nonlinear DDM. |
| Python without visualization | The same eight workflows run through `pytest_provider`. |
| X11 presentation | All eight presets in C++ and Python produce visible pixels and close successfully. |
| Native HoloViz window | All eight FMCW presets in C++ and Python produce visible pixels and close successfully. |
| Headless GPU rendering | Eight FMCW presets and six SAR workflows in C++; eight FMCW presets and SAR execution in provider tests. No display environment or interactive hold. |
| SAR windows | Six workflows in both languages and both window transports; formation and timeline pixels, resize, idle closure, and Escape. |

SISO retains detailed X11 resize/close checks, including Python over TCP X11.
Other X11 cases perform shorter pixel/close checks. GUI probes explicitly enable
visualization and select the backend in a temporary YAML, use a private preference
cache, and disconnect terminal input. A saved developer preference cannot change
the backend or cause CI to wait for input. The log must confirm the selected
backend. Window lookup takes an atomic X11 tree/name snapshot to avoid racing
short-lived native renderer discovery windows.

Each GUI case starts its own Xvfb server. Window tests share the `fmcw_gui` CTest
resource lock, so parallel CTest runs do not overlap these GPU window checks.
They have a 30-second observation deadline and a 60-second CTest timeout. The
existing SDK/CUDA/Vulkan and X11 dependencies remain required; there are no
GPU-optional or renderer-failure skips.

Run the window tests inside the project container after building the applications
and Python provider:

```bash
ctest --test-dir build --output-on-failure -L visualization
```

The full Python-enabled window matrix has 58 cases: 32 FMCW (eight presets,
two languages, two presenters) and 24 SAR (six workflows, two languages, two
presenters), plus two explicit separate-window compatibility cases. Fourteen C++ headless cases additionally exercise the real launch
override without a display. Native cases require a presentation-capable display.

## Historical PR #69 test-stage cost

A local check on 2026-11-03 used an NVIDIA GB10 (Linux ARM64, driver
580.173.02), the HSDK 5.0.0 development image, and serial CTest (`-j1`). Three
runs of the original four checks at `a3b048fd` used a preserved original GUI
probe and an empty preference cache. Three runs of the final expanded matrix
used the same application binaries and host. Builds were completed before
measurement; these are test-stage wall times including Xvfb startup and shutdown.

| Affected test subset | Cases | Median wall time | Range across three runs |
| --- | --- | --- | --- |
| Original checks | 4 | 3.78 s | 3.75–3.95 s |
| Expanded checks | 26 | 32.54 s | 32.50–32.84 s |
| Added cost (difference of medians) | 22 | 28.77 s | — |

The added cases account for approximately 3.96 s in six C++ smokes, 21.71 s in
fourteen X11 workflow/language checks, and 3.10 s in two native window checks
(medians of each category's summed case times). All 26 cases passed in each
final run. Both native window cases additionally passed ten consecutive
executions after fixing the transient-window enumeration race.

These numbers describe the changed subset, not total CI duration. Roughly
30 extra seconds is a local serial test-stage estimate; CI hardware, driver,
scheduling, and contention can change it. No cold-build comparison was made:
there are no new compiled targets or sources, and the default full build
already includes all seven applications and the provider. A subsequent build
of those targets reported no work. The initial local build enabled targets
that had been disabled in the developer build tree, so its time is not added
CI build cost.

Raw CTest JUnit reports, timing samples, the original probe snapshot, and the
summary are under ignored `build/mimo_tdm/ci-gui-timing/` in the measurement
checkout. These historical measurements predate SAR GUI unification; they do not estimate
the current matrix cost.

## Delayed-display resize diagnostic

This diagnostic runs the actual GPU dashboard on Xvfb and forwards its X11
connection through a local TCP proxy. Each direction adds 10 ms per chunk by
default, making display stalls long enough to expose lost image/text pairing.
It sends bursts of real window resize requests, from 1280x720 through 1024x640,
1440x900, and 640x360, then restores 1280x720. The frame-counter region must keep
changing at every size. A stopped display gets ten seconds to recover before
failure; the radar process must remain alive throughout the check.

Build the applications and Python provider first. Run these commands inside
the project container, from the repository root (Xvfb, Pillow, PyYAML, and the
CUDA-capable NVIDIA GPU are required):

```bash
export PYTHONPATH="$PWD/build-gui/python/lib:$PYTHONPATH"
python3 tests/gui/check_fmcw_resize.py --build-dir build-gui
python3 tests/gui/check_fmcw_resize.py --build-dir build-gui --language cpp
```

The build tree must provide the native Python package on `PYTHONPATH` for the
Python case, as with direct provider tests. `--app` selects another FMCW workflow;
`--latency-ms` adjusts the display delay. Displays :96 and :97 and TCP ports
6096 and 6097 must be free. Run one probe at a time. Logs and screenshots go to
`<build-dir>/gui-resize-checks/`.

The status-header capture follows the scaled canvas as the window changes size.
Text and plots are composed into one GPU frame, so their alignment is preserved.
This checks update recovery; it does not require the panel layout to reflow.

## SAR panel and window checks

`SarWindow_<application>_<language>_<presenter>` covers all six SAR apps in
C++ and Python with explicit X11/native modes and isolated preferences. The
fixture uses inherited configuration, verifies formation and timeline pixels,
resizes the viewing window, and closes during final-image hold. The default
timeline is a child panel in the same window; its height stays 112 pixels as
the window resizes. A left status panel belongs to the cached SAR surface;
the probe verifies its pixels after each resize and checks the narrower timeline
child extent. While the source is at EOF, the timeline must advance
without either uploading or repainting the held image. C++ closes
formation; Python exercises aspect fit and closes timeline. Native tests
require a presentation-capable CUDA/Vulkan display and do not skip renderer
failures. The fixtures share the existing GUI resource lock.

Actual laptop-to-Thor SSH/X11 windows require a separate visual walkthrough of
all 13 applications in both languages. Inspect text/overlay alignment, SAR
mosaic orientation and crops, autofocus products, timeline annotations, both
resize policies, placement, and close/Escape behavior. Also review native
presentation on capable hardware. Automated pixel checks and headless success
alone cannot establish interactive visual correctness.

For bounded delay and resource diagnostics, the SAR probe accepts
`--latency-ms 10 --soak-seconds 10`. It forwards X11 through a local proxy,
checks image recovery at each size, and records RSS and file-descriptor samples
under `<build-dir>/gui-sar/`. Add `--capture` to save both completed window
images for visual review. Processing must continue during the soak. RSS must
plateau within 64 MiB or 10%, and file descriptors within four. These short
checks detect regressions; long-duration deployment qualification remains a
separate measurement.

## Real forwarded-display probes

The same scripts run directly on the `DISPLAY` forwarded into the project
container; omit `xvfb-run`. For example, inside that container:

```bash
python3 tests/gui/check_sar_windows.py build-test/applications/sar/spotlight/cpp/spotlight build-test/applications/sar/spotlight/cpp/spotlight_generated_smoke.yaml --application spotlight --presenter x11
python3 tests/cpp/check_fmcw_x11.py build-test/applications/fmcw/simo/cpp/simo "$PWD/applications/fmcw/simo/config/simo.yaml" --application simo --presenter x11
```

Add `--python` for Python. For SAR, `--close timeline --escape --resize-mode fit`
checks the alternate window and resize policy. The provider must be on
`PYTHONPATH`, as with registered tests. These commands open, resize, and close
actual laptop windows. The probe traverses reparented windows with bounded work,
waits for mapping, and reads application pixels instead of desktop pixels.

The 2026-11-17 Thor-to-laptop run passed all 28 X11 cases (13 applications in
both languages plus the second DDM variant). This establishes transport and
basic interaction behavior; scientific image-quality review remains separate.

For the docked SAR default, the X11 probe checks the panel's parent window and
reads `RADAR_PRESENTATION_COUNTERS` on each child: upload count and repaint
count (two format-32 CARDINAL values). After EOF and resize settling, the image
counts must stay unchanged while the footer upload count increases. This
assertion detects full-image redraw or transport triggered by timeline ticks.
`--timeline-placement window` covers the compatibility layout instead.
`--capture --canvas-width 1024 --canvas-height 768` saves a larger preview of the
final layout after the normal resize checks. Capture waits for status pixels
at the new footer origin as well as child geometry, so a delayed cached repaint
cannot leave stale pre-resize content in the preview. These are synthetic fixture images,
not scientific image-quality qualification. Native probes verify footer pixels;
GPU tests check physical-height compensation, fit, blank startup, and independent
old panel timestamps. Positive native-window review requires capable hardware.

The `SarFooterStatus` GPU test composes an exact-timestamp image/status pair and
checks every image pixel stays unobscured, SDK text lands below the image, and
status pixels do not enter the timeline area. `CompletedFrameDock` checks native
status placement, physical sizing after resize, and independent timeline startup.

## SAR startup placeholder

`SarStartupWindow_<language>_<behavior>` uses a paced CPHD fixture so startup is
observable before the first image. Both C++ and Python must show the distinct
charcoal placeholder. The `close` case closes before formation and asserts
exactly one main-panel upload; the `handoff` case waits for a formed image to
replace the placeholder, then closes normally. The existing window tests still
cover resizing, EOF idle updates, separate windows, and Escape.

`SarStartup` GPU tests hold back the first input, prove repeated clock ticks
emit no extra placeholder or held-image frames, verify the first real image's
capture timestamp, and render the actual native text through the shared SDK
composer. The placeholder's -1 stamp cannot advance completed-image drain.
`SarDisplayRate` checks SI conversions and unchanged console measurements.
`SarTimeline` verifies collection-title placement above its colored bar using
the GPU raster and native text records.
It also verifies pulse counts, ETA labels, and colored progress bars at compact
and normal panel widths. The C++/Python handoff probes wait for the received and
formed bars to reach completion on the actual CPHD/formation path.

`SarPulseProgress` checks pacing-aware and unpaced ETA overlap, distinct receipt
and completion, slices sharing a stamp, reform replay, loop generations, partial
cadences, and expired diagnostic history. `SarGpuProgress` holds real CUDA work
behind a gate, fills the fixed event ring, and proves that queries return pending
without claiming completion or waiting for a slot. Numerical operator and dump
regressions remain separate from these UI/diagnostic assertions.

The startup probe can also save a visual-review artifact inside the container:

```bash
xvfb-run -a python3 tests/gui/check_sar_startup.py build-test/applications/sar/spotlight/cpp/spotlight build-test/applications/sar/spotlight/cpp/spotlight_generated_smoke.yaml --capture build-test/gui-sar/startup.png
```

Omit `xvfb-run` to use a real forwarded display. These pixel-readback probes
require a responsive X11 server; a timeout does not establish visual success.
Native-window and real-collection visual qualification remain separate from
GPU composition, isolated X11, and synthetic-fixture checks.
