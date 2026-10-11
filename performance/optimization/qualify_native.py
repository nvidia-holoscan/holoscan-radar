# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-required qualification of native snapshots, separate from acceptance timing."""

import argparse
import base64
import copy
import subprocess
import time
from pathlib import Path

from artifacts import digest, fingerprint, read_json, write_json
from equivalence import compare_snapshots, tensor_bytes, values


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_inventory(export, inventory):
    op = export["operator"]
    expected = {(case, precision) for case in inventory[op] for precision in ("float", "double")}
    actual = [(case["id"], case["precision"]) for case in export["cases"]]
    require(len(actual) == len(set(actual)), "duplicate native case")
    require(set(actual) == expected, "native case inventory changed")
    for case in export["cases"]:
        require(not case["coverage_gaps"], "incomplete native capture")
        require(case["termination"] == "stopped", "fixture termination changed")
        for port, output in case["outputs"].items():
            suppressed = port == "range_bins" and case["id"] == "optional-suppressed"
            emissions = output["emissions"]
            count = 3 if output["wired"] and not suppressed else 0
            require(len(emissions) == count, f"unexpected emission count: {case['id']}/{port}")
            for index, emission in enumerate(emissions):
                require(emission["sequence"] == index + 1, "output sequence changed")
                require(emission["capture_time_ns"] == (index + 1) * 1_000_000, "stamp changed")
                require(emission["clock_id"] > 0, "invalid observed output clock")
                require(emission["memory_kind"] == "device", "output placement changed")
                bits = "64" if case["precision"] == "float" else "128"
                dtype = "complex" + bits
                if port == "range_bins":
                    dtype = "float32" if case["precision"] == "float" else "float64"
                require(emission["dtype"] == dtype, "output precision changed")
                tensor_bytes(emission)
            if not emissions:
                continue
            shapes = [emission["shape"] for emission in emissions]
            if op == "fmcw_range_processor" and "live-match-miss" in case["id"]:
                require(shapes[0] == shapes[2], "live override latched on miss")
                require(shapes[1][-1] == shapes[0][-1] * 2, "live FFT shape not applied")
            if op == "clutter_filter":
                # Passthrough on the first frame and after the one-tick live match.
                for frame in (0, 2):
                    actual = values(tensor_bytes(emissions[frame]), dtype)
                    require(
                        actual == [complex(i % 17 + frame, i % 7) for i in range(len(actual))],
                        "clutter default passthrough changed after live miss",
                    )
                if "live-match-miss" in case["id"]:
                    actual = values(tensor_bytes(emissions[1]), dtype)
                    chirps, bins = shapes[1]
                    original = [complex(i % 17 + 1, i % 7) for i in range(len(actual))]
                    means = [
                        sum(original[row * bins + col] for row in range(chirps)) / chirps
                        for col in range(bins)
                    ]
                    require(
                        all(
                            abs(value - (original[i] - means[i % bins])) < 1e-5
                            for i, value in enumerate(actual)
                        ),
                        "live mean subtraction not applied",
                    )


def qualify(executable, root, output, inventory):
    output = output / f"run-{time.time_ns()}"
    output.mkdir(parents=True)
    source_id = fingerprint(root)
    results = []
    for op in inventory:
        recipe = root / "performance" / "operators" / "fmcw" / op / "recipe.yaml"
        exports = []
        for run in range(2):
            path = output / f"{op}-{run}.json"
            with path.with_suffix(".log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [str(executable), op, str(recipe), str(path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                    cwd=root,
                )
            export = read_json(path)
            export["source_id"] = source_id
            write_json(path, export)
            check_inventory(export, inventory)
            exports.append(export)
        policies = {
            port: {precision: {"mode": "exact"} for precision in ("float", "double")}
            for port in exports[0]["cases"][0]["outputs"]
        }
        result = compare_snapshots(*exports, policies)
        require(result["passed"], f"repeated baseline mismatch: {result}")
        for fault in ("value", "temporal"):
            altered = copy.deepcopy(exports[1])
            emission = next(iter(altered["cases"][0]["outputs"].values()))["emissions"][0]
            if fault == "value":
                data = bytearray(tensor_bytes(emission))
                data[0] ^= 1
                emission["data_base64"] = base64.b64encode(data).decode()
            else:
                emission["capture_time_ns"] += 1
            rejected = compare_snapshots(exports[0], altered, policies)
            require(not rejected["passed"], f"{fault} mismatch incorrectly accepted")
            write_json(output / f"{op}-{fault}-rejection.json", rejected)
        results.append(result)
    report = {
        "passed": True,
        "source_id": source_id,
        "executable_sha256": digest(executable),
        "results": results,
        "scope": "native fixture qualification, not candidate acceptance",
    }
    write_json(output / "qualification.json", report)
    print(
        f"GPU qualification passed: {sum(item['cases'] for item in results)} cases, "
        f"repeat equivalence and {2 * len(results)} injected mismatch rejections"
    )
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--inventory", type=Path, default=Path(__file__).with_name("fixture_inventory.json")
    )
    args = parser.parse_args()
    qualify(
        args.executable.resolve(),
        args.root.resolve(),
        args.output.resolve(),
        read_json(args.inventory),
    )
