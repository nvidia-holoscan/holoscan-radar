# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict artifact I/O and candidate-only checkpoints; no git resets or commits."""

import base64
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path


def read_json(path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=invalid, object_pairs_hook=unique)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(root):
    """Include tracked and untracked source, excluding git-ignored build artifacts."""
    root = Path(root).resolve()
    paths = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    ).split(b"\0")
    hasher = hashlib.sha256()
    for raw in sorted(set(paths) - {b""}):
        path = root / os.fsdecode(raw)
        hasher.update(raw + b"\0")
        if path.is_symlink():
            hasher.update(b"link:" + os.fsencode(os.readlink(path)))
        elif path.exists():
            hasher.update(str(path.stat().st_mode & 0o777).encode())
            hasher.update(path.read_bytes())
        else:
            hasher.update(b"deleted")
    return hasher.hexdigest()


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected number")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name}: invalid finite value {value}")
    return value


def contained(root, name):
    root = Path(root).resolve()
    path = root / name
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"unsafe artifact path: {name}")
    if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root):
        raise ValueError(f"symlink not supported: {name}")
    if not path.resolve().is_relative_to(root) or path == root:
        raise ValueError(f"path outside root: {name}")
    return path


def file_state(root, names):
    states = {}
    for name in names:
        path = contained(root, name)
        if path.exists() and not path.is_file():
            raise ValueError(f"not a file: {name}")
        states[name] = (
            {"sha256": digest(path), "mode": path.stat().st_mode & 0o777} if path.exists() else None
        )
    return states


def checkpoint(root, names):
    if not names or len(set(names)) != len(names):
        raise ValueError("checkpoint requires unique explicit file paths")
    root = str(Path(root).resolve())
    states = file_state(root, names)
    return {
        "root": root,
        "states": states,
        "contents": {
            name: base64.b64encode(contained(root, name).read_bytes()).decode()
            for name, state in states.items()
            if state is not None
        },
    }


def restore(saved, candidate):
    if candidate["root"] != saved["root"] or set(candidate["states"]) != set(saved["states"]):
        raise ValueError("candidate state does not match checkpoint paths/root")
    root = saved["root"]
    if file_state(root, saved["states"]) != candidate["states"]:
        raise ValueError("intervening changes detected; refusing candidate rollback")
    # Validate every saved byte sequence before changing any file.
    contents = {}
    for name, state in saved["states"].items():
        contained(root, name)
        if state is not None:
            data = base64.b64decode(saved["contents"][name], validate=True)
            if hashlib.sha256(data).hexdigest() != state["sha256"]:
                raise ValueError(f"corrupt checkpoint: {name}")
            contents[name] = data
    for name, state in saved["states"].items():
        path = contained(root, name)
        if state is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[name])
            path.chmod(state["mode"])
