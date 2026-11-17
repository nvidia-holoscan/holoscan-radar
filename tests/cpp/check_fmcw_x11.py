# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check FMCW window pixels and shutdown with forwarded X11, Xvfb, or native HoloViz."""

from __future__ import annotations

import argparse
import ctypes as c
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class ClientData(c.Union):
    _fields_ = [("bytes", c.c_char * 20), ("shorts", c.c_short * 10), ("longs", c.c_long * 5)]


class ClientMessage(c.Structure):
    _fields_ = [
        ("type", c.c_int),
        ("serial", c.c_ulong),
        ("send_event", c.c_int),
        ("display", c.c_void_p),
        ("window", c.c_ulong),
        ("message_type", c.c_ulong),
        ("format", c.c_int),
        ("data", ClientData),
    ]


class KeyEvent(c.Structure):
    _fields_ = [
        ("type", c.c_int),
        ("serial", c.c_ulong),
        ("send_event", c.c_int),
        ("display", c.c_void_p),
        ("window", c.c_ulong),
        ("root", c.c_ulong),
        ("subwindow", c.c_ulong),
        ("time", c.c_ulong),
        ("x", c.c_int),
        ("y", c.c_int),
        ("x_root", c.c_int),
        ("y_root", c.c_int),
        ("state", c.c_uint),
        ("keycode", c.c_uint),
        ("same_screen", c.c_int),
    ]


class Event(c.Union):
    _fields_ = [("client", ClientMessage), ("key", KeyEvent), ("padding", c.c_long * 24)]


class WindowAttributes(c.Structure):
    _fields_ = [
        ("x", c.c_int),
        ("y", c.c_int),
        ("width", c.c_int),
        ("height", c.c_int),
        ("border_width", c.c_int),
        ("depth", c.c_int),
        ("visual", c.c_void_p),
        ("root", c.c_ulong),
        ("window_class", c.c_int),
        ("bit_gravity", c.c_int),
        ("win_gravity", c.c_int),
        ("backing_store", c.c_int),
        ("backing_planes", c.c_ulong),
        ("backing_pixel", c.c_ulong),
        ("save_under", c.c_int),
        ("colormap", c.c_ulong),
        ("map_installed", c.c_int),
        ("map_state", c.c_int),
        ("all_event_masks", c.c_long),
        ("your_event_mask", c.c_long),
        ("do_not_propagate_mask", c.c_long),
        ("override_redirect", c.c_int),
        ("screen", c.c_void_p),
    ]


def xlib():
    lib = c.CDLL("libX11.so.6")
    signatures = {
        "XGetWindowAttributes": (c.c_int, [c.c_void_p, c.c_ulong, c.POINTER(WindowAttributes)]),
        "XOpenDisplay": (c.c_void_p, [c.c_char_p]),
        "XDefaultRootWindow": (c.c_ulong, [c.c_void_p]),
        "XGrabServer": (c.c_int, [c.c_void_p]),
        "XUngrabServer": (c.c_int, [c.c_void_p]),
        "XQueryTree": (
            c.c_int,
            [
                c.c_void_p,
                c.c_ulong,
                c.POINTER(c.c_ulong),
                c.POINTER(c.c_ulong),
                c.POINTER(c.POINTER(c.c_ulong)),
                c.POINTER(c.c_uint),
            ],
        ),
        "XFetchName": (c.c_int, [c.c_void_p, c.c_ulong, c.POINTER(c.c_void_p)]),
        "XGetGeometry": (
            c.c_int,
            [
                c.c_void_p,
                c.c_ulong,
                c.POINTER(c.c_ulong),
                c.POINTER(c.c_int),
                c.POINTER(c.c_int),
                c.POINTER(c.c_uint),
                c.POINTER(c.c_uint),
                c.POINTER(c.c_uint),
                c.POINTER(c.c_uint),
            ],
        ),
        "XGetImage": (
            c.c_void_p,
            [c.c_void_p, c.c_ulong, c.c_int, c.c_int, c.c_uint, c.c_uint, c.c_ulong, c.c_int],
        ),
        "XGetPixel": (c.c_ulong, [c.c_void_p, c.c_int, c.c_int]),
        "XDestroyImage": (c.c_int, [c.c_void_p]),
        "XMoveWindow": (c.c_int, [c.c_void_p, c.c_ulong, c.c_int, c.c_int]),
        "XResizeWindow": (c.c_int, [c.c_void_p, c.c_ulong, c.c_uint, c.c_uint]),
        "XKeysymToKeycode": (c.c_ubyte, [c.c_void_p, c.c_ulong]),
        "XInternAtom": (c.c_ulong, [c.c_void_p, c.c_char_p, c.c_int]),
        "XSendEvent": (c.c_int, [c.c_void_p, c.c_ulong, c.c_int, c.c_long, c.POINTER(Event)]),
        "XFlush": (c.c_int, [c.c_void_p]),
        "XCloseDisplay": (c.c_int, [c.c_void_p]),
        "XFree": (c.c_int, [c.c_void_p]),
    }
    for name, (result, args) in signatures.items():
        function = getattr(lib, name)
        function.restype = result
        function.argtypes = args
    return lib


def find_window(lib, display, root, prefix=b"Radar CI dashboard"):
    # Native renderer discovery creates a short-lived probe window. Keep the
    # tree/name snapshot atomic so it cannot disappear during XFetchName.
    lib.XGrabServer(display)
    try:
        # A real window manager can reparent clients below decoration windows.
        # Bound both the traversal and the server-lock interval on forwarded X11.
        pending = [(root, 0)]
        budget = 512
        deadline = time.monotonic() + 0.5
        while pending and budget and time.monotonic() < deadline:
            node, depth = pending.pop()
            parent, returned_root, count = c.c_ulong(), c.c_ulong(), c.c_uint()
            children = c.POINTER(c.c_ulong)()
            lib.XQueryTree(
                display,
                node,
                c.byref(returned_root),
                c.byref(parent),
                c.byref(children),
                c.byref(count),
            )
            try:
                for index in reversed(range(count.value)):
                    if not budget or time.monotonic() >= deadline:
                        break
                    budget -= 1
                    child = children[index]
                    name = c.c_void_p()
                    lib.XFetchName(display, child, c.byref(name))
                    if name.value:
                        try:
                            if c.string_at(name).startswith(prefix):
                                return child
                        finally:
                            lib.XFree(name)
                    if depth < 4:
                        pending.append((child, depth + 1))
            finally:
                if children:
                    lib.XFree(children)
        return None
    finally:
        lib.XUngrabServer(display)
        lib.XFlush(display)


def geometry(lib, display, window):
    root, x, y = c.c_ulong(), c.c_int(), c.c_int()
    width, height, border, depth = (c.c_uint() for _ in range(4))
    lib.XGetGeometry(
        display,
        window,
        c.byref(root),
        c.byref(x),
        c.byref(y),
        c.byref(width),
        c.byref(height),
        c.byref(border),
        c.byref(depth),
    )
    return width.value, height.value


def colors(lib, display, root, size):
    # Wait for mapping before reading client pixels; desktop wallpaper must
    # never satisfy the image assertion on a real forwarded display.
    attributes = WindowAttributes()
    if (
        not lib.XGetWindowAttributes(display, root, c.byref(attributes))
        or attributes.map_state != 2
    ):
        return 0
    width, height = size
    image = lib.XGetImage(display, root, 0, 0, width, height, c.c_ulong(-1).value, 2)
    if not image:
        return 0
    try:
        return len(
            {
                lib.XGetPixel(image, x * (width - 1) // 47, y * (height - 1) // 31)
                for y in range(32)
                for x in range(48)
            }
        )
    finally:
        lib.XDestroyImage(image)


def check(
    executable: str,
    config: str,
    python: bool,
    tcp: bool,
    application: str,
    presenter: str,
    smoke: bool,
):
    if tcp:
        os.environ["DISPLAY"] = "localhost:" + os.environ["DISPLAY"].lstrip(":")
    lib = xlib()
    display = lib.XOpenDisplay(None)
    if not display:
        raise RuntimeError("the test display could not be opened")
    root = lib.XDefaultRootWindow(display)
    deadline = time.monotonic() + int(os.environ.get("HOLOSCAN_RADAR_TEST_TIMEOUT_SECONDS", "30"))
    with tempfile.TemporaryDirectory(prefix="radar-x11-test-") as directory:
        preset = Path(directory) / "window.yaml"
        # The bounded multi-profile simulated link is deliberately fail-stop
        # under overload. Give this window test cold-start margin while other
        # GPU unit tests run; numerical smoke presets keep their normal cadence.
        cadence = "    frame_period: 0.5\n" if application == "multi_waveform" else ""
        preset.write_text(
            f"extends: {json.dumps(config)}\napp:\n  run:\n    num_frames: 1000\n"
            f"{cadence}  visualization:\n    enabled: true\n"
            f"    x11_presentation: {str(presenter == 'x11').lower()}\n"
            "    window_title: Radar CI dashboard\n"
        )
        environment = dict(os.environ, XDG_CACHE_HOME=str(Path(directory) / "cache"))
        command = (
            [sys.executable, "-m", "holoscan_radar.applications.cli", application]
            if python
            else [executable]
        ) + [str(preset)]
        with (Path(directory) / "run.log").open("w+") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=environment
            )
            try:

                def wait(predicate):
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise AssertionError("application exited before the window check")
                        value = predicate()
                        if value:
                            return value
                        time.sleep(0.01)
                    raise TimeoutError("window observation deadline expired")

                window = wait(lambda: find_window(lib, display, root))
                size = geometry(lib, display, window)
                root_size = geometry(lib, display, root)
                size = tuple(min(a, b) for a, b in zip(size, root_size, strict=True))
                wait(lambda: colors(lib, display, window, size) > 32)
                if not smoke:
                    lib.XResizeWindow(display, window, 640, 360)
                    lib.XFlush(display)
                    wait(lambda: geometry(lib, display, window) == (640, 360))
                    wait(lambda: colors(lib, display, window, (640, 360)) > 32)

                event = Event()
                event.client.type = 33  # ClientMessage
                event.client.display = display
                event.client.window = window
                event.client.message_type = lib.XInternAtom(display, b"WM_PROTOCOLS", 0)
                event.client.format = 32
                event.client.data.longs[0] = lib.XInternAtom(display, b"WM_DELETE_WINDOW", 0)
                assert lib.XSendEvent(display, window, 0, 0, c.byref(event))
                lib.XFlush(display)
                result = process.wait(timeout=max(0.1, deadline - time.monotonic()))
                assert result == 0, f"window close returned {result}"
                log.seek(0)
                output = log.read()
                if presenter == "x11":
                    assert "X11 finished-image transport" in output
                else:
                    assert "Radar presentation: CUDA device" in output
                    assert "X11 finished-image transport" not in output
                assert "Choose how to display" not in output
                assert "Using remembered" not in output
                assert "VK_ERROR" not in output
                print(
                    f"PASS: {application} {presenter} rendered pixels, "
                    + ("" if smoke else "resize and ")
                    + "cooperative close",
                    flush=True,
                )
            except BaseException:
                log.seek(0)
                print(log.read()[-8000:], file=sys.stderr)
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
    parser.add_argument("--tcp", action="store_true")
    parser.add_argument("--application", default="siso")
    parser.add_argument("--presenter", choices=("x11", "native"), default="x11")
    parser.add_argument(
        "--smoke", action="store_true", help="Check pixels and close without resize"
    )
    arguments = parser.parse_args()
    check(
        arguments.executable,
        arguments.config,
        arguments.python,
        arguments.tcp,
        arguments.application,
        arguments.presenter,
        arguments.smoke,
    )
