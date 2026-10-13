#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate metadata.json files against holoscan-cli JSON schemas.

Walks the repository, loads every metadata.json that isn't under an
excluded directory, and validates it using holoscan_cli's schema
registry (operator, application, module, package, …).  The correct
schema is chosen automatically from the top-level envelope key.
"""

import errno
import json
import os
import stat
import sys
from pathlib import Path

_EXCLUDE_DIRS = {".git", ".local", ".cache", ".venv", "venv", "_CPack_Packages"}
_MAX_METADATA_BYTES = 512 * 1024


def _is_excluded_dir(name: str) -> bool:
    # Skip build trees: the literal `build/` plus any `build-<variant>/`
    # (mirrors the `build-*/` pattern in .gitignore, e.g. build-docker).
    return name in _EXCLUDE_DIRS or name == "build" or name.startswith("build-")


def iter_metadata_files(repo_root: Path):
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(d)]
        if "metadata.json" in filenames:
            yield Path(dirpath) / "metadata.json"


def read_metadata_file(path: Path, repo_root: Path) -> str:
    """Read one bounded, regular metadata file within ``repo_root``."""
    try:
        relative_path = path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("metadata.json resolves outside the repository") from exc

    if not relative_path.parts or any(
        part in (os.curdir, os.pardir) for part in relative_path.parts
    ):
        raise ValueError("metadata.json resolves outside the repository")

    try:
        nofollow_flag = os.O_NOFOLLOW
        directory_flag = os.O_DIRECTORY
        nonblock_flag = os.O_NONBLOCK
    except AttributeError as exc:
        raise ValueError(
            "secure metadata validation requires no-follow directory traversal"
        ) from exc

    directory_flags = os.O_RDONLY | directory_flag | nofollow_flag
    try:
        root_descriptor = os.open(repo_root, directory_flags)
    except OSError as exc:
        raise ValueError(f"unable to open repository root safely: {exc}") from exc

    directory_descriptors = [root_descriptor]
    try:
        for component in relative_path.parts[:-1]:
            try:
                descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptors[-1],
                )
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise ValueError(
                        "metadata path contains a symlink or non-directory component"
                    ) from exc
                raise ValueError(f"unable to open metadata path safely: {exc}") from exc
            directory_descriptors.append(descriptor)

        try:
            file_descriptor = os.open(
                relative_path.parts[-1],
                os.O_RDONLY | nofollow_flag | nonblock_flag,
                dir_fd=directory_descriptors[-1],
            )
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise ValueError("metadata.json must be a regular file, not a symlink") from exc
            raise ValueError(f"unable to open metadata.json safely: {exc}") from exc

        with os.fdopen(file_descriptor, "rb") as metadata_file:
            file_info = os.fstat(metadata_file.fileno())
            if not stat.S_ISREG(file_info.st_mode):
                raise ValueError("metadata.json must be a regular file")
            if file_info.st_size > _MAX_METADATA_BYTES:
                raise ValueError(f"metadata.json exceeds {_MAX_METADATA_BYTES} byte limit")
            content = metadata_file.read(_MAX_METADATA_BYTES + 1)
    finally:
        for descriptor in reversed(directory_descriptors):
            os.close(descriptor)

    if len(content) > _MAX_METADATA_BYTES:
        raise ValueError(f"metadata.json exceeds {_MAX_METADATA_BYTES} byte limit")

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"metadata.json is not UTF-8: {exc}") from exc


def main():
    try:
        from holoscan_cli.metadata.metadata_validator import validate_json
    except ImportError:
        print(
            "error: holoscan-cli is not installed in the active Python environment.\n"
            "       Activate the environment that has holoscan-cli before running "
            "pre-commit.",
            file=sys.stderr,
        )
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[3]
    failed = False

    for path in sorted(iter_metadata_files(repo_root)):
        rel = path.relative_to(repo_root)
        try:
            data = json.loads(read_metadata_file(path, repo_root))
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"{rel}: invalid metadata — {exc}", file=sys.stderr)
            failed = True
            continue

        ok, message = validate_json(data, path.parent)
        if not ok:
            print(f"{rel}: {message}", file=sys.stderr)
            failed = True
        else:
            print(f"{rel}: ok")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
