# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-light validation shared by recipe consumers."""

from pathlib import PureWindowsPath


def validate_recipe_name(name: object) -> str:
    """Require one filename component; never reinterpret a name as a path."""
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in {".", ".."}
        or any(character in name for character in ("/", "\\", "\0"))
        or PureWindowsPath(name).drive
    ):
        raise ValueError("recipe name must be a non-empty single filename component")
    return name
