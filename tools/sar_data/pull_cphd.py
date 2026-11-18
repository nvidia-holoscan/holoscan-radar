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

uv run tools/sar_data/pull_cphd.py --config applications/sar/autofocus/config/autofocus.yaml
uv run tools/sar_data/pull_cphd.py --app tiled_strip
uv run tools/sar_data/pull_cphd.py --app independent_images
uv run tools/sar_data/pull_cphd.py --app all --dest data/sar
"""

# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0"]
# ///

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

UMBRA_BUCKET = "umbra-open-data-catalog"
UMBRA_HTTPS = f"https://{UMBRA_BUCKET}.s3.us-west-2.amazonaws.com/"
CAPELLA_BUCKET = "capella-open-data"
# STAC asset hrefs use this host; us-west-2 also serves the bucket.
CAPELLA_HTTPS = f"https://{CAPELLA_BUCKET}.s3.amazonaws.com/"

# Settled 2026-08-25. A swap needs a broken key or an explicit reopen.
INDEPENDENT_IMAGES: list[tuple[str, str]] = [
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
# Tiled strip and continuous strip share Imperial Valley. Palmdale stays in
# INPUT_DATA as an optional CPHD 1.0.1 spare, not this pin.
IMPERIAL_VALLEY: list[tuple[str, str]] = [
    (
        "Imperial Valley",
        "data/2024/11/29/"
        "CAPELLA_C14_SM_CPHD_VV_20241129025546_20241129025602/"
        "CAPELLA_C14_SM_CPHD_VV_20241129025546_20241129025602.cphd",
    ),
]

# The default autofocus preset uses this collect, distinct from INDEPENDENT_IMAGES's Kourou.
AUTOFOCUS_KOUROU: list[tuple[str, str]] = [
    (
        "Kourou autofocus",
        "sar-data/tasks/Kourou, French Guiana/1378d9a4-8eb3-4d5c-96ce-15797a5a8d27/"
        "2023-06-24-13-22-56_UMBRA-05/"
        "2023-06-24-13-22-56_UMBRA-05_CPHD.cphd",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
APPS = {
    name: REPO_ROOT / f"applications/sar/{name}/config/{name}.yaml"
    for name in (
        "autofocus",
        "tiled_strip",
        "continuous_strip",
        "independent_images",
        "spotlight",
        "video_sar",
    )
}
# Preserve the historical eight-object --app all selection.
ALL_APPS = ("autofocus", "tiled_strip", "continuous_strip", "independent_images")


@dataclass(frozen=True)
class Dataset:
    """Pinned source and exact path relative to the application's data root."""

    scene: str
    key: str
    bucket: str
    https_prefix: str
    path: Path
    check_iarp: bool = False
    recipe: Path | None = None


def dataset_catalog() -> dict[Path, Dataset]:
    datasets = []
    for collects, bucket, prefix, directory, check in (
        (AUTOFOCUS_KOUROU, UMBRA_BUCKET, UMBRA_HTTPS, "sar", False),
        (IMPERIAL_VALLEY, CAPELLA_BUCKET, CAPELLA_HTTPS, "sar", False),
        (INDEPENDENT_IMAGES, UMBRA_BUCKET, UMBRA_HTTPS, "sar", True),
    ):
        for scene, key in collects:
            datasets.append(
                Dataset(scene, key, bucket, prefix, Path(directory) / Path(key).name, check)
            )
    atlanta = "2026-02-20-03-03-38_UMBRA-07"
    datasets.append(
        Dataset(
            "Atlanta",
            f"sar-data/task-data/0f1231be-26d3-4b98-96ab-f675a5375c14/"
            f"{atlanta}/{atlanta}_CPHD.cphd",
            UMBRA_BUCKET,
            UMBRA_HTTPS,
            Path(f"sar/{atlanta}_CPHD.cphd"),
        )
    )
    recipe = REPO_ROOT / "tools/sar_data/recipes/houston_1m.yaml"
    spec = yaml.safe_load(recipe.read_text())
    # The recipe owns Houston's source and window; never substitute the full raw object.
    datasets.append(
        Dataset(
            "Houston 1 m subset",
            spec["source"]["key"],
            spec["source"]["bucket"],
            UMBRA_HTTPS,
            Path("sar") / f"{spec['name']}.cphd",
            recipe=recipe,
        )
    )
    return {dataset.path: dataset for dataset in datasets}


# Two IARPs closer than this are the same scene (Ohio's second plant
# cluster is ~10 km west; 1 km catches a duplicate of one collect).
IARP_COLLISION_M = 1000.0
HTTPS_TIMEOUT_S = 60.0
CHUNK_BYTES = 1024 * 1024
PROGRESS_INTERVAL_S = 0.25


@dataclass(frozen=True)
class PullJob:
    """One CPHD object and the directory it should land in."""

    scene: str
    key: str
    bucket: str
    https_prefix: str
    dest_dir: Path
    check_iarp: bool
    recipe: Path | None = None
    output_name: str | None = None

    @property
    def destination(self) -> Path:
        return self.dest_dir / (self.output_name or Path(self.key).name)


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


def format_bytes(num_bytes: float) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def format_duration(seconds: float) -> str:
    whole = int(max(0.0, seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def jobs_for_config(config: Path, data_root: Path | None = None) -> list[PullJob]:
    """Validate the entire config against the catalog before doing any work."""
    root = data_root if data_root is not None else Path(os.environ.get("HOLOHUB_DATA_DIR", "data"))
    spec = yaml.safe_load(config.read_text())
    cphd = spec.get("cphd") if isinstance(spec, dict) else None
    paths = cphd.get("path") if isinstance(cphd, dict) else None
    paths = [paths] if isinstance(paths, str) else paths
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path.strip() for path in paths)
    ):
        raise ValueError(f"{config}: cphd.path must be a nonempty string or list of strings")
    catalog = dataset_catalog()
    jobs: dict[Path, PullJob] = {}
    for text in paths:
        given = Path(text)
        # Match the application's existing-file precedence before its data-root fallback.
        candidates = (given, config.parent / given) if not given.is_absolute() else (given,)
        existing = next((path for path in candidates if path.is_file()), None)
        destination = (
            existing
            if existing is not None
            else (root / given if not given.is_absolute() else given)
        )
        dataset = next(
            (
                item
                for rel, item in catalog.items()
                if (root / rel).resolve() == destination.resolve()
            ),
            None,
        )
        if dataset is None:
            raise ValueError(
                f"{config}: no manifest entry matches cphd.path {text!r} "
                f"(resolved destination: {destination}); register its dataset or recipe"
            )
        if dataset.recipe is None and Path(dataset.key).name != destination.name:
            raise ValueError(
                f"{config}: manifest source {dataset.key!r} does not produce {destination}"
            )
        jobs[destination.resolve()] = PullJob(
            dataset.scene,
            dataset.key,
            dataset.bucket,
            dataset.https_prefix,
            destination.parent,
            dataset.check_iarp,
            dataset.recipe,
            destination.name,
        )
    return list(jobs.values())


def jobs_for(app: str, dest: Path | None) -> list[PullJob]:
    """Resolve default configs; retain the legacy --dest directory override."""
    jobs: dict[Path, PullJob] = {}
    for name in ALL_APPS if app == "all" else (app,):
        for job in jobs_for_config(APPS[name]):
            if dest is not None:
                job = replace(job, dest_dir=dest)
            jobs[job.destination.resolve()] = job
    return list(jobs.values())


def prepare_one(job: PullJob, force: bool, index: int, total: int) -> Path:
    if job.recipe is None:
        return pull_one(job, force, index, total)
    command = [
        "uv",
        "run",
        str(REPO_ROOT / "tools/sar_data/subset_cphd.py"),
        str(job.recipe),
        "--output",
        str(job.destination),
    ]
    if force:
        command.append("--force")
    subprocess.run(command, check=True)
    return job.destination


class DownloadProgress:
    """Single-line byte progress with rate and, when the size is known, ETA."""

    def __init__(self, label: str, total: int | None) -> None:
        self.label = label
        self.total = total if total and total > 0 else None
        self.done = 0
        self.started = time.monotonic()
        self.last_draw = 0.0
        self.tty = sys.stderr.isatty()

    def add(self, num_bytes: int) -> None:
        self.done += num_bytes
        self._draw(final=False)

    def set(self, num_bytes: int) -> None:
        self.done = num_bytes
        self._draw(final=False)

    def finish(self) -> None:
        self._draw(final=True)

    def abort(self) -> None:
        if self.tty:
            print(file=sys.stderr)

    def _draw(self, final: bool) -> None:
        now = time.monotonic()
        interval = PROGRESS_INTERVAL_S if self.tty else 2.0
        if not final and now - self.last_draw < interval:
            return
        self.last_draw = now
        elapsed = max(now - self.started, 1e-6)
        rate = self.done / elapsed
        line = f"{self.label}  {format_bytes(self.done)}"
        if self.total is not None:
            pct = 100.0 * self.done / self.total
            line += f" / {format_bytes(self.total)}  {pct:5.1f}%"
        line += f"  {format_bytes(rate)}/s"
        if final:
            line += f"  in {format_duration(elapsed)}"
        elif self.total is not None and rate > 0.0 and self.done < self.total:
            remaining = (self.total - self.done) / rate
            line += f"  ETA {format_duration(remaining)}"
        if self.tty:
            end = "\n" if final else ""
            print(f"\r{line}\033[K", end=end, file=sys.stderr, flush=True)
        elif final or now - self.started >= interval:
            print(line, file=sys.stderr, flush=True)


def cphd_readable(path: Path) -> bool:
    try:
        read_cphd_xml(path)
        return True
    except (OSError, ET.ParseError, RuntimeError, ValueError):
        return False


def content_length(headers: object) -> int | None:
    raw = getattr(headers, "get", lambda _key: None)("Content-Length")
    if raw is None:
        return None
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return None
    return size if size >= 0 else None


def https_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=HTTPS_TIMEOUT_S) as response:
            return content_length(response.headers)
    except (OSError, urllib.error.URLError, ValueError):
        return None


def aws_size(bucket: str, key: str) -> int | None:
    try:
        completed = subprocess.run(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                bucket,
                "--key",
                key,
                "--no-sign-request",
                "--query",
                "ContentLength",
                "--output",
                "text",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        size = int(completed.stdout.strip())
    except (subprocess.CalledProcessError, OSError, ValueError):
        return None
    return size if size > 0 else None


def partial_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".part")


def download_https(url: str, dest: Path, progress: DownloadProgress) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(dest)
    if partial.exists():
        partial.unlink()
    try:
        with (
            urllib.request.urlopen(url, timeout=HTTPS_TIMEOUT_S) as response,
            partial.open("wb") as out,
        ):
            expected_bytes = content_length(response.headers)
            received_bytes = 0
            if progress.total is None:
                progress.total = expected_bytes or None
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                out.write(block)
                received_bytes += len(block)
                progress.add(len(block))
            if expected_bytes is not None and received_bytes != expected_bytes:
                raise RuntimeError(
                    f"HTTPS download size mismatch: received {received_bytes} bytes, "
                    f"expected {expected_bytes} bytes"
                )
        partial.replace(dest)
        progress.finish()
    except BaseException:
        progress.abort()
        if partial.exists():
            partial.unlink()
        raise


def download_aws(bucket: str, key: str, dest: Path, progress: DownloadProgress) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = partial_path(dest)
    if partial.exists():
        partial.unlink()
    uri = f"s3://{bucket}/{key}"
    cmd = ["aws", "s3", "cp", uri, str(partial), "--no-sign-request"]
    if not sys.stderr.isatty():
        cmd.append("--no-progress")
    try:
        # AWS CLI writes a transfer-manager temp and only renames to `partial`
        # on success, so polling `.part` stays at 0. Native CLI progress is
        # the live indicator (stdout); our meter records the size after rename.
        output = sys.stderr if sys.stderr.isatty() else subprocess.DEVNULL
        completed = subprocess.run(cmd, check=False, stdout=output)
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, cmd)
        if not partial.exists():
            raise RuntimeError(f"aws s3 cp produced no file at {partial}")
        progress.set(partial.stat().st_size)
        partial.replace(dest)
        progress.finish()
    except BaseException:
        progress.abort()
        if partial.exists():
            partial.unlink()
        raise


def pull_one(job: PullJob, force: bool, index: int, count: int) -> Path:
    dest = job.dest_dir / Path(job.key).name
    label = f"[{index}/{count}] {job.scene}"
    if dest.is_file() and not force:
        if cphd_readable(dest):
            print(f"skip {label}: {dest.name} already present")
            return dest
        print(f"re-pull {label}: existing file is unreadable")
    print(f"pull {label} -> {dest}")
    aws = shutil.which("aws")
    if aws is not None:
        try:
            progress = DownloadProgress(label, aws_size(job.bucket, job.key))
            download_aws(job.bucket, job.key, dest, progress)
            return dest
        except (subprocess.CalledProcessError, OSError) as exc:
            print(f"aws s3 failed ({exc}); falling back to HTTPS", file=sys.stderr)
    url = https_url(job.https_prefix, job.key)
    progress = DownloadProgress(label, https_size(url))
    download_https(url, dest, progress)
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


def report_positions(files: list[tuple[str, Path]]) -> None:
    for scene, path in files:
        llh = iarp_llh(path)
        if llh is None:
            print(f"  {scene}: {path.name}")
        else:
            print(f"  {scene}: IARP {llh[0]:.4f}°, {llh[1]:.4f}°  HAE {llh[2]:.1f} m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", type=Path, help="read required CPHD inputs from this YAML")
    parser.add_argument(
        "--data-root", type=Path, help="data root for --config (default: HOLOHUB_DATA_DIR or data)"
    )
    selection.add_argument(
        "--app",
        choices=[*sorted(APPS), "all"],
        help="select a default config, or 'all' for the historical eight-object collection set",
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
        help=(
            "directory to write CPHD files into (default: app preset path). "
            "All selected files go directly into this directory."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download files that already exist",
    )
    args = parser.parse_args()
    if args.config is not None and args.dest is not None:
        parser.error("--config uses exact configured paths; use --data-root instead of --dest")
    if args.data_root is not None and args.config is None:
        parser.error("--data-root requires --config")
    try:
        jobs = (
            jobs_for_config(args.config, args.data_root)
            if args.config is not None
            else jobs_for(args.app, args.dest)
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.list:
        for job in jobs:
            url = https_url(job.https_prefix, job.key)
            print(f"{job.scene}: {url} -> {job.destination}")
            if job.recipe is not None:
                print(
                    f"  Prepare with recipe: {job.recipe.relative_to(REPO_ROOT)} (HTTP ranges; not the full object)"
                )
        return 0

    for job in jobs:
        job.dest_dir.mkdir(parents=True, exist_ok=True)

    pulled: list[tuple[PullJob, Path]] = []
    count = len(jobs)
    for index, job in enumerate(jobs, start=1):
        try:
            pulled.append((job, prepare_one(job, args.force, index, count)))
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(
                f"error: {job.scene}: download failed: {exc}\n"
                f"Source: {https_url(job.https_prefix, job.key)}\n"
                f"Destination: {job.destination}\n"
                "See tools/sar_data/README.md for download instructions.",
                file=sys.stderr,
            )
            return 1

    iarp_files = [(job.scene, path) for job, path in pulled if job.check_iarp]
    if iarp_files:
        print("checking IARPs (must be unique)…")
        refuse_colliding_iarps(iarp_files)
    report_positions([(job.scene, path) for job, path in pulled if not job.check_iarp])

    dests = ", ".join(str(path) for path in dict.fromkeys(job.dest_dir for job in jobs))
    print(f"{args.config or args.app} collection set ready in {dests}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
