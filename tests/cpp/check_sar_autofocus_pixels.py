#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare saved Figure 2 products, including complex phase, through the real app.

The tiny CPHD cannot support PGA recovery. Force the scatterer refusal for
clean/injected products, then require oracle correction to restore the clean
complex pixels. The injected case must differ substantially, so passing is not
vacuous. Uses only the standard library, including when bindings are disabled.
"""

import argparse
import ast
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def read_product(path: Path, precision: str) -> list[complex]:
    data = path.read_bytes()
    if data[:8] != b"\x93NUMPY\x01\x00":
        raise AssertionError(f"{path}: expected a NumPy v1 array")
    header_size = struct.unpack_from("<H", data, 8)[0]
    header = ast.literal_eval(data[10 : 10 + header_size].decode("ascii"))
    dtype, layout = ("<c8", "<ff") if precision == "single" else ("<c16", "<dd")
    if header != {"descr": dtype, "fortran_order": False, "shape": (5, 5)}:
        raise AssertionError(f"{path}: unexpected image metadata {header}")
    payload = data[10 + header_size :]
    if len(payload) != 25 * struct.calcsize(layout):
        raise AssertionError(f"{path}: incomplete or oversized pixel payload")
    pixels = [complex(real, imag) for real, imag in struct.iter_unpack(layout, payload)]
    if not all(math.isfinite(z.real) and math.isfinite(z.imag) for z in pixels):
        raise AssertionError(f"{path}: non-finite complex pixels")
    return pixels


def relative_error(actual: list[complex], reference: list[complex]) -> float:
    energy = sum(abs(z) ** 2 for z in reference)
    if energy <= 0:
        raise AssertionError("clean reference has no energy")
    return math.sqrt(sum(abs(a - b) ** 2 for a, b in zip(actual, reference, strict=True)) / energy)


def run_product(binary: Path, base: str, work: Path, precision: str, mode: str) -> list[complex]:
    output = work / f"{mode}.npy"
    config = work / f"{mode}.yaml"
    text = base + (
        f"\nsample_precision: {precision}\ndump: true\n"
        f"output:\n  path: {json.dumps(str(output))}\n"
    )
    if mode != "clean":
        text += "position_error:\n  axis: look\n  polynomial: [0, 0, 0.02]\n"
        if mode == "oracle":
            text += "  oracle: true\n"
    config.write_text(text)
    result = subprocess.run([str(binary), str(config)], capture_output=True, text=True, timeout=60)
    log = result.stdout + result.stderr
    if result.returncode or not output.is_file():
        raise AssertionError(
            f"{mode}: exit {result.returncode}, image exists={output.is_file()}\n{log}"
        )
    if f"TensorDumpOp: saved image {output}" not in log:
        raise AssertionError(f"{mode}: saved image path was not reported\n{log}")
    if mode == "oracle":
        if "OraclePhaseOp: published 5 pulses" not in log:
            raise AssertionError(f"oracle correction was not exercised\n{log}")
    elif "termination refused_scatterers, 0 iteration(s) accepted (budget 8)" not in log:
        raise AssertionError(f"{mode}: expected explicit identity correction\n{log}")
    return read_product(output, precision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--precision", choices=("single", "double"), required=True)
    args = parser.parse_args()
    base = args.config.read_text()
    # This fixture's 32-row aux cannot provide 1024 scatterers. Force an
    # identity estimate independently of future estimator improvements.
    old = "  min_scatterers: 1\n"
    if base.count(old) != 1:
        raise AssertionError("expected the generated five-pulse smoke config")
    base = base.replace(old, "  min_scatterers: 1024\n")
    with tempfile.TemporaryDirectory(prefix="autofocus_pixels_") as temp:
        work = Path(temp)
        products = {
            mode: run_product(args.binary.resolve(), base, work, args.precision, mode)
            for mode in ("clean", "injected", "oracle")
        }
        clean, injected, oracle = (products[mode] for mode in ("clean", "injected", "oracle"))
        injected_error = relative_error(injected, clean)
        oracle_error = relative_error(oracle, clean)
        imaginary_energy = sum(z.imag**2 for z in clean) / sum(abs(z) ** 2 for z in clean)
        print(
            f"{args.precision}: injected relative complex error={injected_error:.6g}; "
            f"oracle={oracle_error:.6g}; clean imaginary energy fraction={imaginary_energy:.6g}"
        )
        if imaginary_energy <= 0.01:
            raise AssertionError(
                "reference must have nontrivial phase to detect magnitude-only dumps"
            )
        if injected_error <= 0.05:
            raise AssertionError("injection did not materially change the saved product")
        if oracle_error >= 1e-3 or oracle_error >= 0.01 * injected_error:
            raise AssertionError("oracle did not restore the clean complex product")


if __name__ == "__main__":
    main()
