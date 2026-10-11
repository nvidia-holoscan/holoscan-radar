# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare bounded native fixture exports with per-port, per-precision policies."""

import base64
import math
import struct

from artifacts import finite

DTYPES = {
    "float32": ("f", 4),
    "float64": ("d", 8),
    "complex64": ("f", 8),
    "complex128": ("d", 16),
    "int32": ("i", 4),
    "int64": ("q", 8),
    "uint8": ("B", 1),
    "bool": ("B", 1),
}


def validate_policy(policy):
    if policy.get("mode") not in {"exact", "tolerance"}:
        raise ValueError("policy mode must be exact or tolerance")
    if policy["mode"] == "tolerance":
        for field in ("atol", "rtol"):
            if finite(policy.get(field), field) < 0:
                raise ValueError(f"{field} cannot be negative")
        if not policy.get("rationale"):
            raise ValueError("tolerance requires a rationale")


def tensor_bytes(emission):
    shape, dtype = emission["shape"], emission["dtype"]
    if dtype not in DTYPES or any(type(x) is not int or x < 0 for x in shape):
        raise ValueError("unsupported dtype or invalid shape")
    data = base64.b64decode(emission["data_base64"], validate=True)
    if len(data) != math.prod(shape) * DTYPES[dtype][1]:
        raise ValueError("snapshot byte count does not match shape/dtype")
    return data


def values(data, dtype):
    fmt = DTYPES[dtype][0]
    scalars = [item[0] for item in struct.iter_unpack("<" + fmt, data)]
    if dtype.startswith("complex"):
        return [complex(*scalars[i : i + 2]) for i in range(0, len(scalars), 2)]
    return scalars


def compare_emission(before, after, policy):
    metadata = ("kind", "sequence", "capture_time_ns", "clock_id")
    if any(before.get(key) != after.get(key) for key in metadata):
        return "emission metadata changed"
    if before.get("capture_time_ns") is None or before.get("clock_id") is None:
        return "missing output capture coordinate coverage"
    if before["kind"] == "descriptor":
        return None if before["value"] == after["value"] else "descriptor changed"
    if before["kind"] != "tensor":
        raise ValueError("unsupported emission kind")
    if any(before[key] != after[key] for key in ("shape", "dtype", "memory_kind")):
        return "tensor representation changed"
    a, b = tensor_bytes(before), tensor_bytes(after)
    if policy["mode"] == "exact":
        return None if a == b else "tensor bytes changed"
    if not before["dtype"].startswith(("float", "complex")):
        raise ValueError("discrete outputs require exact comparison")
    for left, right in zip(values(a, before["dtype"]), values(b, before["dtype"])):
        if not (math.isfinite(abs(left)) and math.isfinite(abs(right))):
            return "nonfinite values require exact comparison"
        if abs(left - right) > policy["atol"] + policy["rtol"] * abs(left):
            return "numerical tolerance exceeded"
    return None


def compare_snapshots(before, after, policies):
    for export in (before, after):
        if export.get("schema_version") != 1 or export.get("byte_order") != "little":
            raise ValueError("expected little-endian snapshot schema 1")
        if not export.get("cases"):
            raise ValueError("empty verification export")
    if before.get("fixture_id") != after.get("fixture_id"):
        raise ValueError("verification fixture identity changed")
    if not before.get("operator") or before.get("operator") != after.get("operator"):
        raise ValueError("verification operator identity missing or changed")
    if not before.get("source_id") or not after.get("source_id"):
        raise ValueError("verification source fingerprint missing")

    def index(export):
        result = {}
        for case in export["cases"]:
            key = (case["id"], case["precision"])
            if key in result:
                raise ValueError("duplicate verification case")
            result[key] = case
        return result

    left, right = index(before), index(after)
    if left.keys() != right.keys():
        raise ValueError("verification case set changed")
    if {key[1] for key in left} != {"float", "double"}:
        raise ValueError("verification requires both float and double precision")
    expected_pairs = {
        (case_id, precision) for case_id, _ in left for precision in ("float", "double")
    }
    if set(left) != expected_pairs:
        raise ValueError("each verification case requires both precisions")
    failures = []
    for key, case in left.items():
        other = right[key]
        if case.get("coverage_gaps") or other.get("coverage_gaps"):
            failures.append({"case": key, "reason": "incomplete verification coverage"})
        if case.get("diagnostics") != other.get("diagnostics"):
            failures.append({"case": key, "reason": "diagnostic behavior changed"})
        if case.get("termination") != other.get("termination"):
            failures.append({"case": key, "reason": "termination changed"})
        if case["outputs"].keys() != other["outputs"].keys():
            raise ValueError("verification output set changed")
        if set(case["outputs"]) != set(policies):
            raise ValueError(
                "verification must declare every policy output, including unwired ports"
            )
        for port, output in case["outputs"].items():
            policy = policies.get(port, {}).get(key[1])
            if policy is None:
                raise ValueError(f"missing confirmed policy for {port}/{key[1]}")
            validate_policy(policy)
            candidate = other["outputs"][port]
            if output["wired"] != candidate["wired"]:
                reason = "output wiring changed"
            elif len(output["emissions"]) != len(candidate["emissions"]):
                reason = "emission count changed"
            elif not output["wired"] and output["emissions"]:
                reason = "unwired output unexpectedly emitted"
            else:
                reason = next(
                    (
                        issue
                        for a, b in zip(output["emissions"], candidate["emissions"])
                        if (issue := compare_emission(a, b, policy))
                    ),
                    None,
                )
            if reason:
                failures.append({"case": key, "port": port, "reason": reason})
    return {
        "passed": not failures,
        "cases": len(left),
        "failures": failures,
        "operators": [before["operator"]],
        "case_ids": sorted({key[0] for key in left}),
    }
