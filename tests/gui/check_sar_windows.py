# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise both SAR windows and idle final-image shutdown on a real X11 server."""

import argparse
import ctypes as c
import json
import os
import re
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
            "    window_width: 256\n    window_height: 256\n"
            "    window_title: Radar CI SAR formation\n"
            f"    resize_mode: {args.resize_mode}\n"
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
                timeline = wait(lambda: find_window(lib, display, root, b"SAR timeline"))
                # Xvfb has no window manager; keep the diagnostic window from
                # obscuring the formation canvas during pixel observations.
                lib.XMoveWindow(display, timeline, 0, 600)
                lib.XFlush(display)
                wait(lambda: colors(lib, display, formation, geometry(lib, display, formation)) > 8)
                wait(lambda: colors(lib, display, timeline, geometry(lib, display, timeline)) > 8)
                for width, height in ((640, 360), (320, 480), (256, 256)):
                    lib.XResizeWindow(display, formation, width, height)
                    lib.XFlush(display)
                    wait(lambda: geometry(lib, display, formation) == (width, height))
                    wait(lambda: colors(lib, display, formation, (width, height)) > 8)
                log.seek(0)
                initial_counts = [
                    int(x) for x in re.findall(r"SAR: image (\d+) formed", log.read())
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
                assert lib.XSendEvent(display, window, 0, 1 if args.escape else 0, c.byref(event))
                lib.XFlush(display)
                assert process.wait(timeout=10) == 0
                log.seek(0)
                output = log.read()
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
                    f"PASS: {args.application} {args.presenter} both windows, resize, idle {args.close} close"
                )
            except BaseException:
                log.seek(0)
                print(log.read()[-10000:], file=sys.stderr)
                if "formation" in locals() and "timeline" in locals():
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
    parser.add_argument("--close", choices=("formation", "timeline"), default="formation")
    parser.add_argument("--resize-mode", choices=("stretch", "fit"), default="stretch")
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--soak-seconds", type=float, default=0.0)
    parser.add_argument("--escape", action="store_true")
    parser.add_argument("--capture", action="store_true")
    check(parser.parse_args())
