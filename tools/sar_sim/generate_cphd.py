#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a CPHD fixture and its ground-truth sidecar from a recipe.

python3 generate_cphd.py recipes/mode01_smoke.yaml -o /tmp/sar_fixtures
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sar_sim import generate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", type=Path, help="recipe YAML to generate from")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="directory to write <name>.cphd and <name>_truth.json into",
    )
    parser.add_argument(
        "--no-self-check",
        action="store_true",
        help="skip the range-compression convention check (not recommended)",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    result = generate(args.recipe, args.output_dir, self_check=not args.no_self_check)
    elapsed = time.perf_counter() - start

    size_mib = result.cphd_path.stat().st_size / (1024 * 1024)
    print(f"{result.cphd_path}  ({size_mib:.1f} MiB, {elapsed:.1f} s)")
    print(f"{result.sidecar_path}")
    if result.self_check:
        print(
            "self-check: worst target range error "
            f"{result.self_check['worst_range_error_m']:.4f} m "
            f"against {result.self_check['range_resolution_m']:.4f} m resolution"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
