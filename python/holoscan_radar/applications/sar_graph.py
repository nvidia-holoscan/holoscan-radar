# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared native lifetime and driver support for Python-authored SAR graphs."""

import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holoscan_radar import _radar

import holoscan as hs


@dataclass(frozen=True)
class BuiltSarGraph:
    """Inspectable graph and native support, retained until execution ends."""

    _context: Any

    @property
    def graph(self) -> Any:
        return self._context.graph

    @property
    def options(self) -> Any:
        return self._context.options

    @property
    def pools(self) -> dict[str, int]:
        return self._context.pools

    def run(self) -> int:
        """Run once through final image drain, or until cooperative shutdown.

        Return the number of reported images. Native failures raise RuntimeError.
        """
        if threading.current_thread() is not threading.main_thread():
            return self._context.run()
        # Python owns signal dispositions; the native driver polls Python and
        # cooperatively stops and joins the session before propagating interrupts.
        previous = signal.getsignal(signal.SIGTERM)

        def stop(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        try:
            return self._context.run()
        finally:
            signal.signal(signal.SIGTERM, previous)


@dataclass
class _SarBuilder:
    native: Any
    graph: hs.Graph

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native, name)

    @property
    def options(self) -> hs.CompileOptions:
        """Export equivalent SDK options for graph inspection and compilation."""
        options = hs.CompileOptions()
        deployment = self.native.deployment
        for operator, port, device in deployment["tensors"]:
            options.deployment.bind_tensor_output_device(
                operator_path=operator, output_port=port, device=hs.execution.DeviceId(device)
            )
        for name in deployment["threads"]:
            options.deployment.bind_thread_placement(name, hs.execution.ThreadPlacement())
        return options


def prepare(application: str, config: str | Path, cphd_override: str | Path | None) -> Any:
    """Reuse C++ config validation, pool sizing and per-graph support owners."""
    native = _radar._prepare_sar(
        application,
        str(Path(config).expanduser().resolve()),
        "" if cphd_override is None else str(Path(cphd_override).expanduser().resolve()),
    )
    graph = hs.Graph(application)
    graph.op(native.registry_name, "cphd_reader", context=native)
    return _SarBuilder(native, graph)
