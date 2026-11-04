# FMCW GUI checks

## Registered CI smoke coverage

The default GPU CTest dashboard includes these checks for enabled applications:

| Path | Coverage |
| --- | --- |
| C++ without visualization | All eight smoke presets complete two frames, including linear and nonlinear DDM. |
| Python without visualization | The same eight workflows run through `pytest_provider`. |
| X11 presentation | All eight presets in C++ and Python produce visible pixels and close successfully. |
| Native HoloViz window | SISO in C++ and Python produces visible pixels, resizes, and closes successfully. |

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

This adds 22 CTest entries to the default all-application Python-enabled build:
six missing C++ smokes, fourteen X11 workflow/language cases, and two native
window cases. Existing SISO X11 and adaptive/multi-waveform smoke names remain
unchanged. No compiled source, executable target, or dependency is added; the
default CI build already builds these applications.

## Measured test-stage cost

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
checkout. The broader delayed-display diagnostic below remains manual.

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
