#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned public CPHDs for the standalone SAR applications (INPUT_DATA.md).

The app reads only local files. This script pulls over HTTPS (or
``aws s3 --no-sign-request`` when available) and skips files already
present.

Independent images is the six unique-IARP Umbra SNAPSHOTs and
refuses the set if two IARPs collide — the check reads each
file's CPHD XML, not the folder name.

Tiled and continuous strip share the Imperial Valley Capella collection.

python3 tools/sar_data/pull_cphd.py --app tiled_strip
python3 tools/sar_data/pull_cphd.py --app independent_images
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UMBRA_BUCKET = "umbra-open-data-catalog"
UMBRA_HTTPS = f"https://{UMBRA_BUCKET}.s3.us-west-2.amazonaws.com/"
CAPELLA_BUCKET = "capella-open-data"
# STAC asset hrefs use this host; us-west-2 also serves the bucket.
CAPELLA_HTTPS = f"https://{CAPELLA_BUCKET}.s3.amazonaws.com/"

# Settled 2026-08-25. A swap needs a broken key or an explicit reopen.
MODE5: list[tuple[str, str]] = [
    (
        "Panama Canal",
        "sar-data/tasks/Panama Canal, Panama/"
        "2d482820-501d-4a21-9c8a-b0213637a4f4/"
        "2023-09-12-02-36-33_UMBRA-04/"
        "2023-09-12-02-36-33_UMBRA-04_CPHD.cphd",
    ),
    (
        "Melbourne",
        "sar-data/tasks/Melbourne, Australia/"
        "0a23b3bc-96f2-48cc-8214-c10cef0d48f3/"
        "2023-07-14-23-20-06_UMBRA-05/"
        "2023-07-14-23-20-06_UMBRA-05_CPHD.cphd",
    ),
    (
        "Bingham Copper Mine",
        "sar-data/tasks/Bingham Copper Mine/"
        "b276284e-c8c6-4fb5-8aac-0916bc6ef4d6/"
        "2023-10-20-04-22-56_UMBRA-04/"
        "2023-10-20-04-22-56_UMBRA-04_CPHD.cphd",
    ),
    (
        "Kourou",
        "sar-data/tasks/Kourou, French Guiana/"
        "b40dc22c-c15f-4295-b71f-3a1db2007bbd/"
        "2023-02-22-12-54-56_UMBRA-04/"
        "2023-02-22-12-54-56_UMBRA-04_CPHD.cphd",
    ),
    (
        "Buenos Aires",
        "sar-data/tasks/Buenos Aires, ARG/"
        "a7a03ef5-6bb6-4f85-aceb-638685a9b3e7/"
        "2023-02-08-02-11-21_UMBRA-05/"
        "2023-02-08-02-11-21_UMBRA-05_CPHD.cphd",
    ),
    (
        "Intel Ohio One",
        "sar-data/tasks/Intel Ohio One/"
        "95d6685e-a4af-4a87-ba0b-b4a43e636e81/"
        "2023-10-11-02-20-47_UMBRA-05/"
        "2023-10-11-02-20-47_UMBRA-05_CPHD.cphd",
    ),
]

# Re-verified 2026-08-30. INPUT_DATA.md open item 3.
# Modes 3 and 4 share Imperial Valley. Palmdale stays in
# INPUT_DATA as an optional CPHD 1.0.1 spare, not this pin.
IMPERIAL_VALLEY: list[tuple[str, str]] = [
    (
        "Imperial Valley",
        "data/2024/11/29/"
        "CAPELLA_C14_SM_CPHD_VV_20241129025546_20241129025602/"
        "CAPELLA_C14_SM_CPHD_VV_20241129025546_20241129025602.cphd",
    ),
]

# app -> (bucket, https prefix, collects, default dest, iarp check)
APPS: dict[str, tuple[str, str, list[tuple[str, str]], str, bool]] = {
    "tiled_strip": (CAPELLA_BUCKET, CAPELLA_HTTPS, IMPERIAL_VALLEY, "/tmp/sar/imperial", False),
    "continuous_strip": (
        CAPELLA_BUCKET,
        CAPELLA_HTTPS,
        IMPERIAL_VALLEY,
        "/tmp/sar/imperial",
        False,
    ),
    "independent_images": (UMBRA_BUCKET, UMBRA_HTTPS, MODE5, "/tmp/sar/mode5", True),
}

# Two IARPs closer than this are the same scene (Ohio's second plant
# cluster is ~10 km west; 1 km catches a duplicate of one collect).
IARP_COLLISION_M = 1000.0
HTTPS_TIMEOUT_S = 60.0


def local_name(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def child(parent: ET.Element, name: str) -> ET.Element | None:
    for node in parent:
        if local_name(node.tag) == name:
            return node
    return None


def find_path(root: ET.Element, path: str) -> ET.Element | None:
    node: ET.Element | None = root
    for part in path.split("/"):
        if node is None:
            return None
        node = child(node, part)
    return node


def read_cphd_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with path.open("rb") as stream:
        first = stream.readline()
        if not first.startswith(b"CPHD/"):
            raise RuntimeError(f"{path}: not a CPHD file")
        for _ in range(512):
            line = stream.readline()
            if not line:
                break
            if line.startswith(b"\x0c"):
                return header
            text = line.decode("ascii", errors="replace").strip()
            separator = text.find(":=")
            if separator < 0:
                continue
            key = text[:separator].strip()
            value = text[separator + 2 :].strip()
            header[key] = value
    raise RuntimeError(f"{path}: ASCII header has no form-feed terminator")


def read_cphd_xml(path: Path) -> ET.Element:
    header = read_cphd_header(path)
    try:
        offset = int(header["XML_BLOCK_BYTE_OFFSET"])
        size = int(header["XML_BLOCK_SIZE"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"{path}: header missing XML block keys") from exc
    with path.open("rb") as stream:
        stream.seek(offset)
        xml = stream.read(size)
    if len(xml) != size:
        raise RuntimeError(f"{path}: truncated XML block")
    return ET.fromstring(xml)


def iarp_ecef(path: Path) -> tuple[float, float, float]:
    root = read_cphd_xml(path)
    values = []
    for axis in ("X", "Y", "Z"):
        node = find_path(root, f"SceneCoordinates/IARP/ECF/{axis}")
        if node is None or node.text is None:
            raise RuntimeError(f"{path}: missing IARP/ECF/{axis}")
        values.append(float(node.text.strip()))
    return values[0], values[1], values[2]


def iarp_llh(path: Path) -> tuple[float, float, float] | None:
    root = read_cphd_xml(path)
    lat = find_path(root, "SceneCoordinates/IARP/LLH/Lat")
    lon = find_path(root, "SceneCoordinates/IARP/LLH/Lon")
    hae = find_path(root, "SceneCoordinates/IARP/LLH/HAE")
    if lat is None or lon is None or hae is None:
        return None
    if lat.text is None or lon.text is None or hae.text is None:
        return None
    return float(lat.text), float(lon.text), float(hae.text)


def distance_m(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def https_url(https_prefix: str, key: str) -> str:
    return https_prefix + urllib.parse.quote(key, safe="/")


def cphd_readable(path: Path) -> bool:
    try:
        read_cphd_xml(path)
        return True
    except (OSError, ET.ParseError, RuntimeError, ValueError):
        return False


def download_https(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"GET {url}")
    try:
        with (
            urllib.request.urlopen(url, timeout=HTTPS_TIMEOUT_S) as response,
            partial.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        partial.replace(dest)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def download_aws(bucket: str, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    if partial.exists():
        partial.unlink()
    uri = f"s3://{bucket}/{key}"
    print(f"aws s3 cp {uri}")
    try:
        subprocess.run(
            ["aws", "s3", "cp", uri, str(partial), "--no-sign-request"],
            check=True,
        )
        partial.replace(dest)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def pull_one(
    scene: str,
    key: str,
    dest_dir: Path,
    force: bool,
    bucket: str,
    https_prefix: str,
) -> Path:
    dest = dest_dir / Path(key).name
    if dest.is_file() and not force:
        if cphd_readable(dest):
            print(f"skip {scene}: {dest.name} already present")
            return dest
        print(f"re-pull {scene}: existing file is unreadable")
    print(f"pull {scene} -> {dest}")
    aws = shutil.which("aws")
    if aws is not None:
        try:
            download_aws(bucket, key, dest)
            return dest
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"aws s3 failed ({exc}); falling back to HTTPS")
    download_https(https_url(https_prefix, key), dest)
    return dest


def refuse_colliding_iarps(files: list[tuple[str, Path]]) -> None:
    parsed: list[tuple[str, tuple[float, float, float]]] = []
    for scene, path in files:
        ecef = iarp_ecef(path)
        llh = iarp_llh(path)
        if llh is None:
            print(f"  {scene}: IARP ECEF {ecef}")
        else:
            print(f"  {scene}: IARP {llh[0]:.4f}°, {llh[1]:.4f}°  " f"HAE {llh[2]:.1f} m")
        parsed.append((scene, ecef))
    for i, (scene_a, ecef_a) in enumerate(parsed):
        for scene_b, ecef_b in parsed[i + 1 :]:
            gap = distance_m(ecef_a, ecef_b)
            if gap < IARP_COLLISION_M:
                raise SystemExit(
                    f"IARPs collide ({gap:.1f} m < {IARP_COLLISION_M:.0f} m): "
                    f"{scene_a} and {scene_b}. Refusing the set."
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--app",
        choices=sorted(APPS),
        help="SAR application whose public collections should be downloaded",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the selected manifest without downloading",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="directory to write CPHD files into (default: app preset path)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download files that already exist",
    )
    args = parser.parse_args()
    if args.app is None:
        parser.error("--app is required")
    bucket, https_prefix, collects, dest_default, check_iarp = APPS[args.app]
    dest_dir: Path = args.dest if args.dest is not None else Path(dest_default)
    if args.list:
        for scene, key in collects:
            print(f"{scene}: {https_url(https_prefix, key)} -> {dest_dir / Path(key).name}")
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)

    pulled: list[tuple[str, Path]] = []
    for scene, key in collects:
        pulled.append(
            (
                scene,
                pull_one(scene, key, dest_dir, args.force, bucket, https_prefix),
            )
        )

    if check_iarp:
        print("checking IARPs (must be unique)…")
        refuse_colliding_iarps(pulled)
    else:
        for scene, path in pulled:
            llh = iarp_llh(path)
            if llh is None:
                print(f"  {scene}: {path.name}")
            else:
                print(f"  {scene}: IARP {llh[0]:.4f}°, {llh[1]:.4f}°  " f"HAE {llh[2]:.1f} m")
    print(f"{args.app} collection set ready in {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
