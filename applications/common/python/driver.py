# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared application entry-point scaffolding for the FMCW example apps.

Mirrors ``radar_app/app/driver.hpp``: config-path resolution, the
display-probe downgrade for ``visualize`` on a GUI-less host, feeding the
same file to Holoscan's native ``config()`` path, and running the app inside
one try/except — so each app's ``main()`` collapses to one call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from holoscan.radar import gui_available

logger = logging.getLogger(__name__)


def resolve_config_path(argv: list[str], cpp_dir: Path, default_yaml: str) -> Path:
    """Resolve the config path: the first CLI arg if given, else `default_yaml`
    staged next to the app's ``cpp/`` YAMLs. A bare filename (e.g.
    ``fmcw_mimo_ddm.yaml``) also resolves beside `cpp_dir`; any other path
    (including a smoke YAML under ``python/``) is used as given.
    """
    if len(argv) > 1:
        given = Path(argv[1])
        if given.is_file():
            return given
        beside = cpp_dir / given.name
        if beside.is_file():
            return beside
        return given  # load_config reports the original path
    return cpp_dir / default_yaml


def run_fmcw_app(
    app_cls: type,
    argv: list[str],
    log_prefix: str,
    cpp_dir: Path,
    default_yaml: str,
    load_config: Callable[[Path], object],
) -> int:
    """Load config, downgrade `visualize` on a GUI-less host, then run.

    `load_config` is the app's own loader (``Path -> AppConfig``); `AppConfig`
    must expose ``.visualize`` / ``.headless`` / ``.skip_display_check``
    (satisfied by deriving it from ``demo_config.CommonRunConfig``).
    `app_cls` is constructed as ``app_cls(cfg)`` and, once ``run()`` returns,
    may expose an optional ``.cadence`` (a `CadenceMeter`) to summarize.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = load_config(resolve_config_path(argv, cpp_dir, default_yaml))
        if (
            cfg.visualize
            and not cfg.headless
            and not cfg.skip_display_check
            and not gui_available()
        ):
            logger.warning(
                "%s: visualize is enabled but no usable display was detected "
                "(Holoviz could not initialize a window); running without "
                "the visualization window",
                log_prefix,
            )
            cfg.visualize = False

        app = app_cls(cfg)
        app.run()
        cadence = getattr(app, "cadence", None)
        if cadence is not None:
            cadence.log_summary()
    except Exception as exc:  # noqa: BLE001 - top-level app entry point
        logger.error("%s: %s", log_prefix, exc)
        return 1
    return 0
