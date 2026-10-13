# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run separate precision executables; retain their reports before merging."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

operator, build, *arguments = sys.argv[1:]
archive = Path(build) / "runtime"
if archive.is_dir():
    os.environ["LD_LIBRARY_PATH"] = str(archive) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
executables = [
    Path(build) / "performance/cpp" / f"perf_{operator}{suffix}" for suffix in ("", "_double")
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if "--list-cases" in arguments:
    outputs = [
        subprocess.check_output([str(executable), *arguments], text=True)
        for executable in executables
    ]
    if outputs[0] != outputs[1]:
        raise RuntimeError("precision case inventories differ")
    print(outputs[0], end="")
    sys.exit(0)
if "--validate" in arguments or "--help" in arguments:
    for executable in executables:
        subprocess.run([str(executable), *arguments], check=True)
    sys.exit(0)
flag = "--verify-output" if "--verify-output" in arguments else "--output"
position = arguments.index(flag) + 1
output = Path(arguments[position]).resolve()
if output.exists():
    raise RuntimeError(f"refusing to overwrite {output}")
reports, children = [], []
for precision, executable in zip(("float", "double"), executables):
    child = output.with_name(output.stem + f".{precision}" + output.suffix)
    if child.exists():
        raise RuntimeError(f"refusing to overwrite {child}")
    argv = list(arguments)
    argv[position] = str(child)
    subprocess.run([str(executable), *argv], check=True)
    report = json.loads(child.read_text())
    if flag == "--output":
        if any(row["precision"] != precision for row in report["results"]):
            raise RuntimeError("incorrect timing precision label")
    else:
        if any(case["precision"] != precision for case in report["cases"]):
            raise RuntimeError("incorrect snapshot precision label")
    reports.append(report)
    children.append({"path": str(child), "sha256": digest(child)})
merged = reports[0]
if flag == "--output":
    for field in ("schema_version", "platform", "run_settings"):
        if merged[field] != reports[1][field]:
            raise RuntimeError(f"precision report {field} differs")
    if merged["provenance"]["git_commit"] != reports[1]["provenance"]["git_commit"]:
        raise RuntimeError("precision commit differs")
    merged["results"] += reports[1]["results"]
else:
    for field in ("schema_version", "byte_order", "fixture_id", "operator"):
        if merged[field] != reports[1][field]:
            raise RuntimeError(f"precision snapshot {field} differs")
    merged["cases"] += reports[1]["cases"]
merged["precision_children"] = children
output.write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n")
