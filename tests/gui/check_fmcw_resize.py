# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real dashboard resizes over delayed TCP X11; see tests/gui/README.md."""

import argparse
import ctypes as C
import hashlib
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

import yaml
from PIL import ImageGrab

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--app",
    default="mimo_tdm",
    choices=(
        "siso",
        "simo",
        "dense_beamforming",
        "mimo_tdm",
        "mimo_ddm",
        "adaptive_aoa",
        "multi_waveform",
    ),
)
parser.add_argument("--language", choices=("python", "cpp"), default="python")
parser.add_argument("--build-dir", type=Path, default=Path("build-gui"))
parser.add_argument("--latency-ms", type=float, default=10.0)
args = parser.parse_args()
if args.latency_ms < 0:
    parser.error("--latency-ms must be nonnegative")
root = Path(__file__).resolve().parents[2]
build = args.build_dir.resolve()
out = build / "gui-resize-checks"
out.mkdir(parents=True, exist_ok=True)
name = args.app
language = args.language
preset = "mimo_ddm_linear" if name == "mimo_ddm" else name
config = {
    "extends": str(root / f"applications/fmcw/{name}/config/{preset}.yaml"),
    "app": {
        "run": {"num_frames": 1000},
        "visualization": {"enabled": True, "x11_presentation": True},
    },
}
path = out / f"resize_{name}_{language}.yaml"
path.write_text(yaml.safe_dump(config))
x = C.CDLL("libX11.so.6")
D = C.c_void_p
W = C.c_ulong
U = C.c_uint
x.XOpenDisplay.argtypes = [C.c_char_p]
x.XOpenDisplay.restype = D
x.XDefaultRootWindow.argtypes = [D]
x.XDefaultRootWindow.restype = W
x.XQueryTree.argtypes = [D, W, C.POINTER(W), C.POINTER(W), C.POINTER(C.POINTER(W)), C.POINTER(U)]
x.XFetchName.argtypes = [D, W, C.POINTER(C.c_char_p)]
x.XFree.argtypes = [C.c_void_p]
x.XResizeWindow.argtypes = [D, W, U, U]
x.XSync.argtypes = [D, C.c_int]
x.XCloseDisplay.argtypes = [D]


def find(display, window):
    text = C.c_char_p()
    if x.XFetchName(display, window, C.byref(text)) and text.value:
        title = text.value.decode()
        x.XFree(text)
        if title.startswith("FMCW"):
            return window
    parent = W()
    rootwin = W()
    children = C.POINTER(W)()
    count = U()
    x.XQueryTree(
        display, window, C.byref(rootwin), C.byref(parent), C.byref(children), C.byref(count)
    )
    try:
        for i in range(count.value):
            result = find(display, children[i])
            if result:
                return result
    finally:
        if children:
            x.XFree(children)
    return None


def proxy():
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 6097))
    listener.listen()

    def copy(a, b):
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                time.sleep(args.latency_ms / 1000.0)
                b.sendall(data)
        except OSError:
            pass
        finally:
            try:
                b.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    while True:
        client, _ = listener.accept()
        upstream = socket.create_connection(("127.0.0.1", 6096))
        threading.Thread(target=copy, args=(client, upstream), daemon=True).start()
        threading.Thread(target=copy, args=(upstream, client), daemon=True).start()


existing = x.XOpenDisplay(b":96")
if existing:
    x.XCloseDisplay(existing)
    raise RuntimeError("X11 display :96 is in use; stop that diagnostic before running this one")
server = subprocess.Popen(
    ["Xvfb", ":96", "-screen", "0", "1600x1000x24", "-ac", "-noreset", "-listen", "tcp"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
try:
    time.sleep(1)
    threading.Thread(target=proxy, daemon=True).start()
    display = x.XOpenDisplay(b":96")
    assert display
    code = "\n".join(
        [
            "from pathlib import Path",
            f"from holoscan_radar.applications import {name} as app",
            f"workflow = app.build_graph(Path({str(path)!r}))",
            "session = workflow.run()",
            "assert not session.diagnostics, session.diagnostics",
        ]
    )
    command = (
        ["python3", "-c", code]
        if language == "python"
        else [str(build / f"applications/fmcw/{name}/cpp/{name}"), str(path)]
    )
    with (out / f"resize_{name}_{language}.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env={**os.environ, "DISPLAY": "localhost:97", "XDG_CACHE_HOME": str(out / "cache")},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
        )
        try:
            window = None
            for i in range(100):
                window = find(display, x.XDefaultRootWindow(display))
                if window:
                    break
                time.sleep(0.1)
            assert window, "No GUI window"
            for stage, (width, height) in enumerate(
                [(1280, 720), (1024, 640), (1440, 900), (640, 360), (1280, 720)]
            ):
                for step in range(24):
                    x.XResizeWindow(display, window, width + step % 4 * 8, height + step % 3 * 8)
                    x.XSync(display, 0)
                    time.sleep(0.025)
                x.XResizeWindow(display, window, width, height)
                x.XSync(display, 0)
                time.sleep(1)
                # The complete canvas scales with the window; follow its status
                # header rather than sampling the old fixed framebuffer pixels.
                header = (0, 30 * height // 720, 600 * width // 1280, 65 * height // 720)
                signatures = []
                for sample in range(5):
                    time.sleep(0.3)
                    picture = ImageGrab.grab(xdisplay=":96")
                    signature = hashlib.sha256(picture.crop(header).tobytes()).hexdigest()[:12]
                    signatures.append(signature)
                    picture.save(out / f"resize_{name}_{language}_{stage}_{sample}.png")
                unique = len(set(signatures))
                print(
                    f"{width}x{height}: {unique} distinct headers; alive={process.poll() is None}",
                    flush=True,
                )
                assert process.poll() is None, f"Radar process exited: {process.returncode}"
                if unique < 2:
                    for retry in range(10):
                        time.sleep(1)
                        picture = ImageGrab.grab(xdisplay=":96")
                        new = hashlib.sha256(picture.crop(header).tobytes()).hexdigest()[:12]
                        print("recovery", retry, new, "changed", new != signatures[-1], flush=True)
                        if new != signatures[-1]:
                            break
                    else:
                        raise AssertionError(f"Display frozen after resize to {width}x{height}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    x.XCloseDisplay(display)
finally:
    server.terminate()
    server.wait(timeout=5)
