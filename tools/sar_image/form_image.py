#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Form a CPHD with the reference backprojector and render it to PNG.

    python3 form_image.py /tmp/sar_fixtures/mode01.cphd -o /tmp/mode01.png

Given a simulator sidecar, the predicted target pixels are drawn over the
image, which is what turns a picture into a check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sarkit.cphd as skcphd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sar_image import backproject, render  # noqa: E402


def read_cphd(path: Path):
    with path.open("rb") as handle:
        reader = skcphd.Reader(handle)
        channel = reader.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
        return reader.read_signal(channel), reader.read_pvps(channel)


def default_sidecar(cphd_path: Path) -> Path:
    return cphd_path.with_name(f"{cphd_path.stem}_truth.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cphd", type=Path, help="CPHD file to form")
    parser.add_argument("-o", "--output", type=Path, help="PNG to write (default: <cphd stem>.png)")
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="Ground-truth sidecar; defaults to <stem>_truth.json beside the CPHD",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=8,
        help="Range-compression zero-pad factor (default: 8)",
    )
    parser.add_argument(
        "--floor-db", type=float, default=-45.0, help="Display floor (default: -45)"
    )
    parser.add_argument("--save-npy", type=Path, help="Also write the complex image as .npy")
    args = parser.parse_args(argv)

    sidecar_path = args.sidecar or default_sidecar(args.cphd)
    if not sidecar_path.exists():
        parser.error(
            f"no sidecar at {sidecar_path}; the lattice to form on comes from it. "
            "Pass --sidecar explicitly."
        )
    sidecar = json.loads(sidecar_path.read_text())

    signal, pvps = read_cphd(args.cphd)
    lattice = sidecar["sample_geometry"]
    print(
        f"{args.cphd.name}: {signal.shape[0]} pulses x {signal.shape[1]} samples "
        f"-> lattice {lattice['counts']}"
    )

    start = time.perf_counter()
    image = backproject(signal, pvps, lattice, oversample=args.oversample)
    elapsed = time.perf_counter() - start
    print(f"formed in {elapsed:.2f} s")

    markers = [
        {"index": target["image_index_vu_rounded"], "label": target["name"]}
        for target in sidecar.get("targets", [])
    ]

    output = args.output or args.cphd.with_suffix(".png")
    render(
        image,
        output,
        title=(
            f"{sidecar.get('recipe_name', args.cphd.stem)} — reference backprojector\n"
            "circles = pixels the sidecar predicts"
        ),
        floor_db=args.floor_db,
        markers=markers,
    )
    print(f"wrote {output}")

    if args.save_npy:
        np.save(args.save_npy, image)
        print(f"wrote {args.save_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
