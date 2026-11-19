# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check a real SAR startup window, one cached placeholder, and first-image handoff."""

import argparse
import ctypes as c
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp"))
from check_fmcw_x11 import Event, WindowAttributes, find_window, geometry, xlib  # noqa: E402


def check(args):
    lib = xlib()
    display = lib.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Cannot open startup test display")
    root = lib.XDefaultRootWindow(display)
    with tempfile.TemporaryDirectory(prefix="sar-startup-") as directory:
        preset = Path(directory) / "startup.yaml"
        preset.write_text(
            f"extends: {json.dumps(str(Path(args.config).resolve()))}\n"
            "cphd:\n  names: [Panama Canal]\n"
            "cphd_reader:\n  playback_rate: 0.01\n  pulses_per_tick: 1\n  loop: false\n"
            "app:\n  visualization:\n    enabled: true\n    headless: false\n"
            "    final_image_hold_s: -1\n    max_side: 256\n"
            "    window_width: 640\n    window_height: 480\n"
            "    window_title: Radar CI SAR startup\n"
        )
        command = (
            [sys.executable, "-m", "holoscan_radar.applications.cli", "spotlight"]
            if args.python
            else [args.executable]
        )
        with (Path(directory) / "run.log").open("w+") as log:
            process = subprocess.Popen(
                command + [str(preset), "--presenter", "x11"],
                stdout=log,
                stderr=log,
                stdin=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 25

                def wait(predicate):
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise AssertionError("SAR stopped before startup check completed")
                        value = predicate()
                        if value:
                            return value
                        time.sleep(0.02)
                    raise TimeoutError("Startup window deadline")

                window = wait(lambda: find_window(lib, display, root, b"Radar CI SAR startup"))
                panel = wait(lambda: find_window(lib, display, window, b"SAR image"))

                def pixel(x=2, y=2, surface=None):
                    drawable = panel if surface is None else surface
                    # A named child can exist before it is mapped. Keep the
                    # map/size check and read atomic with respect to X clients.
                    lib.XGrabServer(display)
                    try:
                        attributes = WindowAttributes()
                        if (
                            not lib.XGetWindowAttributes(display, drawable, c.byref(attributes))
                            or attributes.map_state != 2  # IsViewable
                            or not 0 <= x < attributes.width
                            or not 0 <= y < attributes.height
                        ):
                            return None
                        image = lib.XGetImage(display, drawable, x, y, 1, 1, c.c_ulong(-1).value, 2)
                    finally:
                        lib.XUngrabServer(display)
                        lib.XFlush(display)
                    if not image:
                        return None
                    try:
                        return lib.XGetPixel(image, 0, 0) & 0xFFFFFF
                    finally:
                        lib.XDestroyImage(image)

                wait(lambda: pixel() == 0x13181F)  # opaque charcoal, never a formed grayscale image
                time.sleep(0.2)
                assert pixel() == 0x13181F, "Placeholder did not remain cached"
                if args.capture:
                    from PIL import Image

                    width, height = geometry(lib, display, window)
                    data = lib.XGetImage(
                        display, window, 0, 0, width, height, c.c_ulong(-1).value, 2
                    )
                    try:
                        values = [
                            lib.XGetPixel(data, x, y) for y in range(height) for x in range(width)
                        ]
                    finally:
                        lib.XDestroyImage(data)
                    picture = Image.new("RGB", (width, height))
                    picture.putdata([((v >> 16) & 255, (v >> 8) & 255, v & 255) for v in values])
                    picture.save(args.capture)
                if not args.close_before_image:
                    wait(lambda: pixel() not in (None, 0x13181F))
                    timeline = wait(lambda: find_window(lib, display, window, b"SAR timeline"))
                    width, _ = geometry(lib, display, timeline)
                    # The shared track must reach GPU completion, with its
                    # Received marker still visible at the completed edge.
                    wait(lambda: pixel(width - 20, 47, timeline) == 0x73C896)
                    wait(lambda: pixel(width - 9, 47, timeline) == 0x4BCDDC)
                    log.seek(0)
                    assert "SAR: image 1 formed" in log.read()
                event = Event()
                event.client.type = 33
                event.client.display = display
                event.client.window = window
                event.client.message_type = lib.XInternAtom(display, b"WM_PROTOCOLS", 0)
                event.client.format = 32
                event.client.data.longs[0] = lib.XInternAtom(display, b"WM_DELETE_WINDOW", 0)
                assert lib.XSendEvent(display, window, 0, 0, c.byref(event))
                lib.XFlush(display)
                assert process.wait(timeout=10) == 0
                log.seek(0)
                output = log.read()
                assert "X11 finished-image transport" in output
                if args.close_before_image:
                    assert "SAR: image 1 formed" not in output
                    assert "image_uploads=1 " in output, "Startup clock regenerated the main panel"
                print("PASS: SAR startup " + ("close" if args.close_before_image else "handoff"))
            except BaseException:
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                lib.XCloseDisplay(display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable")
    parser.add_argument("config")
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--close-before-image", action="store_true")
    parser.add_argument("--capture")
    check(parser.parse_args())
