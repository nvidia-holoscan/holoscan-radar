#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Form a CPHD with the reference backprojector, or render an operator dump.

    python3 form_image.py /tmp/sar_fixtures/mode01.cphd -o /tmp/mode01.png
    python3 form_image.py --image dump.npy --sidecar scene_truth.json -o out.png
    python3 form_image.py --image atlanta.npy -o atlanta.png

The PNG is a 1:1 greyscale raster of the formed image (one file pixel per
sample, no axes). A sidecar is still required to *form* from a CPHD — that
is the lattice. Predicted target pixels live in the sidecar; they are not
drawn on the PNG. Orientation is the formation lattice — this tool does
not resample.
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


def load_mapping(path: Path) -> dict:
    """JSON sidecar or YAML (formation config / truth sidecar)."""
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            f"{path} is YAML; install pyyaml to read sample_geometry from it"
        ) from exc
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise SystemExit(f"{path} did not contain a mapping")
    return loaded


def lattice_from_mapping(mapping: dict) -> dict | None:
    lattice = mapping.get("sample_geometry")
    if lattice is None and {"origin", "axis_v", "axis_u", "spacing", "counts"} <= (
        mapping.keys()
    ):
        lattice = mapping
    return lattice


def read_cphd(path: Path):
    with path.open("rb") as handle:
        reader = skcphd.Reader(handle)
        channel = reader.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
        return reader.read_signal(channel), reader.read_pvps(channel)


def default_sidecar(cphd_path: Path) -> Path:
    return cphd_path.with_name(f"{cphd_path.stem}_truth.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cphd",
        nargs="?",
        type=Path,
        help="CPHD file to form (omit when --image is given)",
    )
    parser.add_argument("-o", "--output", type=Path, help="PNG to write (default: <cphd stem>.png)")
    parser.add_argument(
        "--sidecar",
        type=Path,
        help="Ground-truth sidecar; defaults to <stem>_truth.json beside the CPHD",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        help=(
            "YAML/JSON with sample_geometry, used to check an operator dump "
            "against the lattice (e.g. mode01.yaml)"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Render this complex .npy instead of forming from a CPHD",
    )
    parser.add_argument(
        "--oversample",
        type=int,
        default=8,
        help="Range-compression zero-pad factor (default: 8)",
    )
    parser.add_argument(
        "--floor-db",
        type=float,
        default=-30.0,
        help="Widest display window in dB below the 99.8th-percentile magnitude "
        "(default: -30). Each image may use a tighter floor from its 25th percentile.",
    )
    parser.add_argument("--save-npy", type=Path, help="Also write the complex image as .npy")
    args = parser.parse_args(argv)

    if args.image is None and args.cphd is None:
        parser.error("give a CPHD to form, or --image to render a dump")

    sidecar_path = args.sidecar
    if sidecar_path is None and args.cphd is not None:
        sidecar_path = default_sidecar(args.cphd)
    lattice = None
    if sidecar_path is not None and sidecar_path.exists():
        lattice = lattice_from_mapping(load_mapping(sidecar_path))
    elif args.image is None:
        parser.error(
            f"no sidecar at {sidecar_path}; the lattice to form on comes from it. "
            "Pass --sidecar explicitly."
        )
    if lattice is None and args.geometry is not None:
        if not args.geometry.exists():
            parser.error(f"geometry file not found: {args.geometry}")
        lattice = lattice_from_mapping(load_mapping(args.geometry))
        if lattice is None:
            parser.error(f"{args.geometry} has no sample_geometry")

    if args.image is not None:
        image = np.load(args.image)
        if lattice is not None:
            expected = tuple(lattice["counts"])
            if image.shape[-2:] != expected:
                print(
                    f"warning: dump shape {image.shape} does not match "
                    f"lattice {expected}",
                    file=sys.stderr,
                )
            print(f"{args.image.name}: {image.shape}  lattice {expected}")
        else:
            print(f"{args.image.name}: {image.shape}")
        default_output = args.image.with_suffix(".png")
    else:
        if lattice is None:
            parser.error(
                "forming from a CPHD needs a lattice; pass --sidecar or --geometry"
            )
        signal, pvps = read_cphd(args.cphd)
        print(
            f"{args.cphd.name}: {signal.shape[0]} pulses x {signal.shape[1]} samples "
            f"-> lattice {lattice['counts']}"
        )
        start = time.perf_counter()
        image = backproject(signal, pvps, lattice, oversample=args.oversample)
        elapsed = time.perf_counter() - start
        print(f"formed in {elapsed:.2f} s")
        default_output = args.cphd.with_suffix(".png")

    output = args.output or default_output
    render(image, output, floor_db=args.floor_db)
    print(f"wrote {output}")

    if args.save_npy:
        np.save(args.save_npy, image)
        print(f"wrote {args.save_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
