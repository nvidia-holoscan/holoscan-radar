# FMCW window resize regression

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
