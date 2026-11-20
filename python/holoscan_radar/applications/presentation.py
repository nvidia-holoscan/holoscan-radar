# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared native presenter policy for Python-authored application graphs."""

from holoscan_radar import _radar


def prepare_presentation(config, path, presenter=None, choose_presenter=False):
    """Resolve launch overrides and preferences before any renderer is selected."""
    viz = config["visualization"]
    selected = _radar._prepare_presentation(
        config["visualize"],
        viz["headless"],
        viz["skip_display_check"],
        viz["x11_presentation"],
        path,
        presenter or "",
        choose_presenter,
    )
    config["visualize"] = selected["enabled"]
    viz["headless"] = selected["headless"]
    viz["x11_presentation"] = selected["x11_presentation"]
    return selected["device"]
