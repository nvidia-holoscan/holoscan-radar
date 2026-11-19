# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise SAR panels, independent updates, resizing, and idle shutdown on real X11."""

import argparse
import ctypes as c
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cpp"))
from check_fmcw_x11 import Event, colors, find_window, geometry, xlib  # noqa: E402
from x11_delay import DelayedX11  # noqa: E402


def check(args):
    lib = xlib()
    display = lib.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Cannot open test display")
    root = lib.XDefaultRootWindow(display)
    with tempfile.TemporaryDirectory(prefix="sar-gui-") as directory:
        preset = Path(directory) / "gui.yaml"
        preset.write_text(
            f"extends: {json.dumps(str(Path(args.config).resolve()))}\n"
            "app:\n  visualization:\n    enabled: true\n    headless: false\n"
            "    final_image_hold_s: -1\n    max_side: 256\n"
            f"    window_width: {args.canvas_width}\n    window_height: {args.canvas_height}\n"
            "    window_title: Radar CI SAR formation\n"
            f"    resize_mode: {args.resize_mode}\n"
            f"    timeline: {{placement: {args.timeline_placement}}}\n"
        )
        if args.soak_seconds:
            with preset.open("a") as config:
                config.write("cphd_reader:\n  loop: true\n")
        command = (
            [sys.executable, "-m", "holoscan_radar.applications.cli", args.application]
            if args.python
            else [args.executable]
        )
        command += [str(preset), "--presenter", args.presenter]
        proxy = (
            DelayedX11(os.environ["DISPLAY"], args.latency_ms, directory)
            if args.latency_ms
            else nullcontext(dict(os.environ))
        )
        with proxy as environment, (Path(directory) / "run.log").open("w+") as log:
            environment["XDG_CACHE_HOME"] = str(Path(directory) / "cache")
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                env=environment,
            )
            deadline = time.monotonic() + 30 + args.soak_seconds
            try:

                def wait(predicate):
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise AssertionError("SAR exited before the GUI check")
                        result = predicate()
                        if result:
                            return result
                        time.sleep(0.02)
                    raise TimeoutError("SAR window deadline")

                def save_windows():
                    from PIL import Image

                    artifact_dir = Path(args.executable).resolve().parents[4] / "gui-sar"
                    artifact_dir.mkdir(exist_ok=True)
                    for name, window in (("formation", formation), ("timeline", timeline)):
                        w, h = geometry(lib, display, window)
                        data = lib.XGetImage(display, window, 0, 0, w, h, c.c_ulong(-1).value, 2)
                        if data:
                            pixels = [lib.XGetPixel(data, x, y) for y in range(h) for x in range(w)]
                            lib.XDestroyImage(data)
                            picture = Image.new("RGB", (w, h))
                            picture.putdata(
                                [((v >> 16) & 255, (v >> 8) & 255, v & 255) for v in pixels]
                            )
                            picture.save(
                                artifact_dir / f"{args.application}_{name}_{args.presenter}.png"
                            )

                formation = wait(lambda: find_window(lib, display, root, b"Radar CI SAR formation"))
                docked = args.timeline_placement == "bottom"
                x11_dock = docked and args.presenter == "x11"
                timeline = (
                    formation
                    if docked and not x11_dock
                    else wait(
                        lambda: find_window(
                            lib, display, formation if x11_dock else root, b"SAR timeline"
                        )
                    )
                )
                image_panel = (
                    wait(lambda: find_window(lib, display, formation, b"SAR image"))
                    if x11_dock
                    else formation
                )
                if not docked:
                    lib.XMoveWindow(display, timeline, 0, 600)
                    lib.XFlush(display)
                wait(
                    lambda: colors(lib, display, image_panel, geometry(lib, display, image_panel))
                    > 8
                )
                if not docked or x11_dock:
                    wait(
                        lambda: colors(lib, display, timeline, geometry(lib, display, timeline)) > 8
                    )

                def first_image_is_displayed():
                    if "SAR: image 1 formed" not in (Path(directory) / "run.log").read_text():
                        return False
                    width, height = geometry(lib, display, image_panel)
                    image_height = height - (112 if docked else 0)
                    # SAR pixels are grayscale; the startup background is tinted
                    # charcoal. Sample a region so a sensor arrow or its label
                    # cannot hide the formed image from a single-pixel probe.
                    data = lib.XGetImage(
                        display,
                        image_panel,
                        0,
                        0,
                        width,
                        image_height,
                        c.c_ulong(-1).value,
                        2,
                    )
                    if not data:
                        return False
                    try:
                        gray = 0
                        for row in range(3):
                            for column in range(5):
                                x = min(width - 1, width * (2 + column) // 8)
                                y = min(image_height - 1, image_height * (7 + row) // 10)
                                pixel = lib.XGetPixel(data, x, y)
                                gray += (
                                    (pixel & 255) == ((pixel >> 8) & 255) == ((pixel >> 16) & 255)
                                )
                        return gray >= 12
                    finally:
                        lib.XDestroyImage(data)

                # The visible startup canvas must never satisfy formed-image,
                # resize-recovery, or EOF idle-counter assertions.
                wait(first_image_is_displayed)

                def status_width(width):
                    return min(320, args.canvas_width // 2, max(1, width // 2))

                def image_has_content(width, height):
                    if colors(lib, display, image_panel, (width, height)) > 1:
                        return True
                    # A flat synthetic raster can have only a thin APC marker.
                    # The coarse 48-by-32 grid may miss it after aspect-fit resize.
                    data = lib.XGetImage(
                        display, image_panel, 0, 0, width, height, c.c_ulong(-1).value, 2
                    )
                    if not data:
                        return False
                    try:
                        background = lib.XGetPixel(data, 0, 0)
                        return any(
                            lib.XGetPixel(data, x, y) != background
                            for y in range(height)
                            for x in range(width)
                        )
                    finally:
                        lib.XDestroyImage(data)

                lib.XGetWindowProperty.argtypes = [
                    c.c_void_p,
                    c.c_ulong,
                    c.c_ulong,
                    c.c_long,
                    c.c_long,
                    c.c_int,
                    c.c_ulong,
                    c.POINTER(c.c_ulong),
                    c.POINTER(c.c_int),
                    c.POINTER(c.c_ulong),
                    c.POINTER(c.c_ulong),
                    c.POINTER(c.c_void_p),
                ]

                def property_pair(window, name):
                    atom = lib.XInternAtom(display, name, 0)
                    actual, count, remaining = c.c_ulong(), c.c_ulong(), c.c_ulong()
                    form, data = c.c_int(), c.c_void_p()
                    lib.XGetWindowProperty(
                        display,
                        window,
                        atom,
                        0,
                        2,
                        0,
                        6,
                        c.byref(actual),
                        c.byref(form),
                        c.byref(count),
                        c.byref(remaining),
                        c.byref(data),
                    )
                    try:
                        if form.value == 32 and count.value == 2:
                            return tuple(c.cast(data, c.POINTER(c.c_ulong))[i] for i in range(2))
                        return None
                    finally:
                        if data:
                            lib.XFree(data)

                if x11_dock:
                    # Both widths select the full footer layout. Rendering at
                    # physical pixels must preserve the entire static lane label
                    # while its timeline expands, then shrinks back.
                    reference = None
                    for width in (960, 1200, 960):
                        lib.XResizeWindow(display, formation, width, 640)
                        lib.XFlush(display)
                        panel_width = width - status_width(width)
                        wait(lambda: geometry(lib, display, timeline) == (panel_width, 112))
                        # Wait for a frame drawn for this exact extent, not
                        # merely a resized child still holding an older frame.
                        wait(
                            lambda: property_pair(timeline, b"RADAR_PRESENTED_EXTENT")
                            == (panel_width, 112)
                        )
                        data = lib.XGetImage(
                            display, timeline, 264 + 8, 0, 200, 24, c.c_ulong(-1).value, 2
                        )
                        assert data, "Cannot capture native footer glyphs"
                        try:
                            glyphs = tuple(
                                lib.XGetPixel(data, x, y) for y in range(24) for x in range(200)
                            )
                        finally:
                            lib.XDestroyImage(data)
                        assert any(glyphs), "Receipt label is missing"
                        if reference is not None:
                            assert glyphs == reference, "Resize stretched native footer glyphs"
                        reference = glyphs

                for width, height in ((640, 360), (320, 480), (256, 256)):
                    lib.XResizeWindow(display, formation, width, height)
                    lib.XFlush(display)
                    wait(lambda: geometry(lib, display, formation) == (width, height))
                    if x11_dock:
                        wait(
                            lambda: geometry(lib, display, timeline)
                            == (width - status_width(width), 112)
                        )
                        wait(lambda: geometry(lib, display, image_panel) == (width, height))
                        # This tiny numerical fixture can have a flat SAR raster.
                        # Its direction marker must survive; status text is
                        # checked separately below, outside the image viewport.
                        wait(lambda: image_has_content(width, height - 112))
                        wait(
                            lambda: colors(
                                lib, display, timeline, (width - status_width(width), 112)
                            )
                            > 8
                        )
                        # The status belongs to the cached SAR surface, below its
                        # image viewport, and must remain legible after resizing.
                        wait(
                            lambda: colors(
                                lib,
                                display,
                                image_panel,
                                (status_width(width), 112),
                                (0, height - 112),
                            )
                            > 8
                        )
                    wait(lambda: colors(lib, display, formation, (width, height)) > 8)
                    if docked and not x11_dock:
                        wait(
                            lambda: colors(lib, display, formation, (width, 112), (0, height - 112))
                            > 8
                        )
                if x11_dock and not args.soak_seconds:
                    # With the source at EOF and the resize settled, neither
                    # cached surface needs a new upload. Window events/closure
                    # remain serviced independently of the held event history.
                    def counters(window):
                        return property_pair(window, b"RADAR_PRESENTATION_COUNTERS")

                    wait(lambda: counters(image_panel))
                    wait(lambda: counters(timeline))
                    time.sleep(0.3)
                    before_image, before_timeline = counters(image_panel), counters(timeline)
                    time.sleep(0.5)
                    assert (
                        counters(image_panel) == before_image
                    ), "Timeline repainted/uploaded SAR image"
                    assert (
                        counters(timeline)[0] == before_timeline[0]
                    ), "Idle timeline uploaded a duplicate GPU footer at EOF"
                # Read through a separate descriptor. Seeking the writer's
                # descriptor also moves the child's stdout offset and can
                # overwrite startup diagnostics during a busy looping run.
                initial_counts = [
                    int(x)
                    for x in re.findall(
                        r"SAR: image (\d+) formed", (Path(directory) / "run.log").read_text()
                    )
                ]
                initial_count = max(initial_counts, default=0)
                samples = []
                soak_end = time.monotonic() + args.soak_seconds
                while time.monotonic() < soak_end:
                    assert process.poll() is None, "SAR stopped during the display soak"
                    status = Path(f"/proc/{process.pid}/status").read_text()
                    rss = int(re.search(r"VmRSS:\s+(\d+)", status)[1]) * 1024
                    fds = len(list(Path(f"/proc/{process.pid}/fd").iterdir()))
                    samples.append((rss, fds))
                    time.sleep(0.2)
                if samples:
                    first = statistics.median(x[0] for x in samples[: max(1, len(samples) // 3)])
                    last = statistics.median(x[0] for x in samples[-max(1, len(samples) // 3) :])
                    assert last - first < max(64 * 1024**2, first * 0.1), "RSS did not plateau"
                    assert max(x[1] for x in samples) - min(x[1] for x in samples) <= 4, "FD growth"
                    artifacts = Path(args.executable).resolve().parents[4] / "gui-sar"
                    artifacts.mkdir(exist_ok=True)
                    (
                        artifacts
                        / f"{args.application}_{'python' if args.python else 'cpp'}_resources.json"
                    ).write_text(
                        json.dumps(
                            {
                                "latency_ms": args.latency_ms,
                                "samples": samples,
                                "rss_first": first,
                                "rss_last": last,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                if args.capture:
                    capture_height = args.canvas_height + (112 if docked else 0)
                    lib.XResizeWindow(display, formation, args.canvas_width, capture_height)
                    lib.XFlush(display)
                    wait(
                        lambda: geometry(lib, display, formation)
                        == (args.canvas_width, capture_height)
                    )
                    if x11_dock:
                        wait(
                            lambda: geometry(lib, display, timeline)
                            == (args.canvas_width - status_width(args.canvas_width), 112)
                        )
                        wait(
                            lambda: geometry(lib, display, image_panel)
                            == (args.canvas_width, capture_height)
                        )
                        wait(
                            lambda: colors(
                                lib, display, image_panel, (args.canvas_width, args.canvas_height)
                            )
                            > 1
                        )
                        wait(
                            lambda: colors(
                                lib,
                                display,
                                timeline,
                                (args.canvas_width - status_width(args.canvas_width), 112),
                            )
                            > 8
                        )
                    if x11_dock:
                        # Geometry can arrive before the delayed presenter has
                        # repainted its cached status at the new footer origin.
                        wait(
                            lambda: colors(
                                lib,
                                display,
                                image_panel,
                                (status_width(args.canvas_width), 112),
                                (0, capture_height - 112),
                            )
                            > 8
                        )
                    time.sleep(0.2)
                    save_windows()
                # The fixture reaches EOF quickly. Exercise closure while the
                # formation producer is idle and the last image is retained.
                time.sleep(0.5)
                window = timeline if args.close == "timeline" else formation
                event = Event()
                event.client.type = 33
                event.client.display = display
                event.client.window = window
                event.client.message_type = lib.XInternAtom(display, b"WM_PROTOCOLS", 0)
                event.client.format = 32
                event.client.data.longs[0] = lib.XInternAtom(display, b"WM_DELETE_WINDOW", 0)
                if args.escape:
                    event = Event()
                    event.key.type = 2
                    event.key.display = display
                    event.key.window = window
                    event.key.root = root
                    event.key.same_screen = 1
                    event.key.keycode = lib.XKeysymToKeycode(display, 0xFF1B)
                if args.signal:
                    process.send_signal(getattr(signal, args.signal))
                else:
                    assert lib.XSendEvent(
                        display, window, 0, 1 if args.escape else 0, c.byref(event)
                    )
                    lib.XFlush(display)
                assert process.wait(timeout=10) == 0
                output = (Path(directory) / "run.log").read_text()
                marker = (
                    "X11 finished-image transport"
                    if args.presenter == "x11"
                    else "Radar presentation: CUDA device"
                )
                assert marker in output
                assert "Choose how to display" not in output
                if samples:
                    counts = [int(x) for x in re.findall(r"SAR: image (\d+) formed", output)]
                    assert counts and max(counts) > max(
                        2, initial_count
                    ), "Processing did not progress during display delay"
                    print(
                        f"Processing advanced from {initial_count} to {max(counts)} images during delay"
                    )
                assert "VK_ERROR" not in output
                assert "no CUDA-backed Vulkan adapter can present" not in output
                print(
                    f"PASS: {args.application} {args.presenter} {args.timeline_placement} timeline, "
                    f"resize, idle {args.close} close"
                )
            except BaseException:
                print((Path(directory) / "run.log").read_text()[-10000:], file=sys.stderr)
                if process.poll() is None and "formation" in locals() and "timeline" in locals():
                    save_windows()
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
    parser.add_argument("--application", required=True)
    parser.add_argument("--python", action="store_true")
    parser.add_argument("--presenter", choices=("x11", "native"), default="x11")
    parser.add_argument("--signal", choices=("SIGINT", "SIGTERM"))
    parser.add_argument("--close", choices=("formation", "timeline"), default="formation")
    parser.add_argument("--resize-mode", choices=("stretch", "fit"), default="stretch")
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--soak-seconds", type=float, default=0.0)
    parser.add_argument("--escape", action="store_true")
    parser.add_argument("--timeline-placement", choices=("bottom", "window"), default="bottom")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--canvas-width", type=int, default=256)
    parser.add_argument("--canvas-height", type=int, default=256)
    check(parser.parse_args())
