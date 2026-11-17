# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded-lifetime TCP proxy for delayed X11 GUI diagnostics."""

import os
import socket
import subprocess
import threading
from pathlib import Path


class DelayedX11:
    def __init__(self, display, milliseconds, directory):
        self.display = display
        self.delay = milliseconds / 1000
        self.directory = Path(directory)
        self.sockets = []
        self.threads = []
        self.stopped = threading.Event()

    def __enter__(self):
        host, number = self.display.rsplit(":", 1)
        self.number = int(number.split(".")[0])
        self.upstream_host = host or None
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.listener.settimeout(0.2)
        self.sockets.append(self.listener)
        proxy_display = f"localhost:{self.listener.getsockname()[1] - 6000}"
        self.environment = dict(os.environ, DISPLAY=proxy_display)
        # Copy only the current test display's cookie into a private authority
        # file. Keep cookie bytes out of command arguments and diagnostic output.
        auth = subprocess.run(["xauth", "list", self.display], capture_output=True, check=True)
        if auth.stdout.strip():
            cookie = auth.stdout.splitlines()[0].split()[-1].decode("ascii")
            authority = self.directory / "proxy.Xauthority"
            authority.touch(mode=0o600)
            subprocess.run(
                ["xauth", "-f", str(authority)],
                input=f"add {proxy_display} MIT-MAGIC-COOKIE-1 {cookie}\n".encode(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.environment["XAUTHORITY"] = str(authority)
        thread = threading.Thread(target=self.accept, daemon=True)
        self.threads.append(thread)
        thread.start()
        return self.environment

    def accept(self):
        while not self.stopped.is_set():
            try:
                client, _ = self.listener.accept()
                if self.upstream_host:
                    upstream = socket.create_connection(
                        (self.upstream_host, 6000 + self.number), timeout=2
                    )
                    upstream.settimeout(None)
                else:
                    upstream = socket.socket(socket.AF_UNIX)
                    upstream.connect(f"/tmp/.X11-unix/X{self.number}")
                self.sockets.extend((client, upstream))
                for source, target in ((client, upstream), (upstream, client)):
                    thread = threading.Thread(target=self.copy, args=(source, target), daemon=True)
                    self.threads.append(thread)
                    thread.start()
            except TimeoutError:
                continue
            except OSError:
                break

    def copy(self, source, target):
        try:
            while not self.stopped.is_set():
                data = source.recv(65536)
                if not data:
                    break
                if self.stopped.wait(self.delay):
                    break
                target.sendall(data)
        except OSError:
            pass
        finally:
            try:
                target.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def __exit__(self, *_):
        self.stopped.set()
        for connection in self.sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        for thread in self.threads:
            thread.join(timeout=1)
