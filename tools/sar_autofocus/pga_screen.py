#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Screen a CPHD collect for PGA suitability, persistence included.

`sweep.py` measures how well autofocus recovers an injected error on a
scene. This answers the question that comes before it: is this scene a
PGA vehicle at all?

Brightness is the obvious screen and it is the wrong one. PGA assumes
each selected range bin is dominated by one strong scatterer that
**persists across the full synthetic aperture** (Wahl, Eichel, Ghiglia
& Jakowatz, IEEE T-AES 30(3), 1994). Ships pass an isolated-bright-
return test and fail that assumption: a hull glints rather than
scattering steadily, and a ship swinging in wind and waves gives every
point on it a different phase error. So this screens two measurable
things, and names the third:

  isolation    peak over the median of a surrounding annulus
  persistence  the candidate's amplitude across K equal, disjoint
               blocks of the dwell, as a profile
  stationarity not measurable here -- it is why the candidate list
               should come from things bolted to the ground

Persistence is measured **between blocks, never against the full
aperture**. A sub-aperture image has K times coarser cross-range
resolution, which can merge neighbouring scatterers into one cell and
inflate its peak on its own; comparing a half-aperture peak to the
full one cannot tell that artefact apart from genuine concentration.
Across K blocks of equal width the inflation is common-mode, so the
profile's *shape* is trustworthy even though its scale is not.

The reported number is the worst block over the profile mean. A
persistent scatterer sits near 1.00 everywhere. A glint, and plenty of
man-made structure with aspect-dependent RCS, collapses to near zero
over part of the dwell -- which is where PGA has no signal to estimate
a phase from, and the outer dwell is exactly where a quadratic error
is largest.

Formation runs through the `spotlight` app on an IARP-centred `lattice`
rather than a transcribed ECEF grid, so a verdict here is expressed in
the same terms the scene configs beside it already use: a cell on a
square lattice of a given side and spacing.

    python3 tools/sar_autofocus/pga_screen.py \\
        data/sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd

Note that a scene can fail this screen and still autofocus well --
Atlanta does. PGA's precondition is sufficient, not necessary: the
estimator averages a phase gradient over many range bins, and a city
supplies hundreds of mediocre ones where a screened scene supplies one
good one. Read a failure as "this scene cannot be *trusted* to
autofocus", not as "this scene will not".
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree

import numpy as np
import yaml

# The data root the SAR apps resolve `cphd.path` against.
DATA_ROOT = pathlib.Path("data")
SPOTLIGHT = pathlib.Path("build/applications/sar/spotlight/cpp/spotlight")

# CPHD's ASCII header is terminated by a form feed, and its fields are
# "KEY := VALUE" lines. Reading NumVectors needs only that header and the
# XML block it points at, so this stays on the standard library rather
# than taking the sarkit dependency tools/sar_data needs for rewriting.
_SECTION_TERMINATOR = b"\f\n"
_HEADER_PROBE = 8192


def collect_extent(path: pathlib.Path) -> tuple[int, int]:
    """Vector and sample counts, from Data/Channel specifically.

    A CPHD header carries other NumSamples elements (support arrays);
    taking the wrong one silently mis-derives anything downstream that
    depends on it.
    """
    with open(path, "rb") as handle:
        probe = handle.read(_HEADER_PROBE)
        if not probe.startswith(b"CPHD/"):
            raise SystemExit(f"{path}: not a CPHD file (missing CPHD/ magic)")
        term = probe.find(_SECTION_TERMINATOR)
        if term < 0:
            raise SystemExit(f"{path}: ASCII header has no form-feed terminator")
        fields = {}
        for line in probe[:term].decode("ascii").split("\n")[1:]:
            if " := " in line:
                key, value = line.split(" := ", 1)
                fields[key] = value
        try:
            offset = int(fields["XML_BLOCK_BYTE_OFFSET"])
            size = int(fields["XML_BLOCK_SIZE"])
        except KeyError as exc:
            raise SystemExit(f"{path}: CPHD header missing {exc}") from exc
        handle.seek(offset)
        tree = ElementTree.fromstring(handle.read(size).decode("utf-8"))
    channel = tree.find(".//{*}Data/{*}Channel")
    if channel is None:
        raise SystemExit(f"{path}: no Data/Channel in the header")
    return (
        int(channel.find("{*}NumVectors").text),
        int(channel.find("{*}NumSamples").text),
    )


def form(
    binary: pathlib.Path,
    cphd: str,
    counts: int,
    spacing: float,
    tag: str,
    workdir: pathlib.Path,
    pulse_range: tuple[int, int] | None,
) -> np.ndarray:
    """Form one image, over the whole dwell or over a pulse range.

    One run per sub-aperture, rather than one run emitting all K on an
    `emit_interval_pulses` cadence: the block cadence is faster than
    TensorDumpOp can write a large image, so the single-pass version
    silently drops most of its emits. Re-reading the CPHD per block
    costs a second or two and cannot lose a frame.
    """
    for stale in workdir.glob(f"screen_{tag}_*.npy"):
        stale.unlink()
    cfg = {
        "cphd": {"path": cphd, "sidecar": ""},
        "lattice": {"counts": counts, "spacing": spacing},
        "dump": True,
        "output": {"path": str(workdir / f"screen_{tag}_{{index}}.npy"), "max_files": 1},
        "cphd_reader": {"coordinate_frame": "ecef", "pulses_per_tick": 1024},
        # Spotlight spelling: no retention, one emit-and-reset over the
        # pulses the reader hands it, so the dump is exactly this block.
        "aperture_manager": {
            "max_aperture_pulses": 0,
            "emit_interval_pulses": 0,
            "emit_mode": "reset",
        },
        "range_compression": {"propagation_speed": 299792458.0},
        "backprojection": {"coordinate_frame": "ecef"},
        "visualize": False,
    }
    if pulse_range is not None:
        cfg["cphd_reader"]["pulse_range"] = list(pulse_range)
    path = workdir / f"screen_{tag}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    result = subprocess.run(
        [str(binary), str(path)], capture_output=True, text=True, check=False, timeout=3600
    )
    formed = workdir / f"screen_{tag}_0.npy"
    if not formed.is_file():
        sys.stderr.write(result.stdout[-4000:] + result.stderr[-4000:])
        raise SystemExit(f"formation produced no dump for {tag}")
    mag = np.abs(np.load(formed))
    while mag.ndim > 2:
        mag = mag[0]
    return mag


def isolated_candidates(
    mag: np.ndarray, count: int, inner: int, outer: int, min_ratio: float
) -> list[tuple[int, int, float, float]]:
    """Brightest cells whose surrounding annulus is quiet.

    Isolation is peak over the annulus median rather than over the
    global median: a bright cell inside a bright block (Atlanta) is not
    an isolated scatterer even though it clears a global gate.
    """
    n_v, n_u = mag.shape
    # One partition over the brightest cells, then a walk down them.
    # Repeatedly argmax-ing a large image and zeroing a neighbourhood
    # costs thousands of full scans on a scene with no isolated
    # scatterers, which is the case where the answer is "none" and the
    # run should be quick.
    pool = min(mag.size, max(200000, count * 20000))
    flat = mag.ravel()
    idx = np.argpartition(flat, -pool)[-pool:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    out: list[tuple[int, int, float, float]] = []
    taken: list[tuple[int, int]] = []
    for position in idx:
        if len(out) >= count:
            break
        v, u = divmod(int(position), n_u)
        peak = float(mag[v, u])
        if peak <= 0.0:
            break
        # Keep candidates apart, so each pick is a different scatterer
        # rather than another cell of the same mainlobe.
        if any(abs(v - pv) <= outer and abs(u - pu) <= outer for pv, pu in taken):
            continue
        taken.append((v, u))
        v0, v1 = max(0, v - outer), min(n_v, v + outer + 1)
        u0, u1 = max(0, u - outer), min(n_u, u + outer + 1)
        patch = mag[v0:v1, u0:u1].astype(np.float64)
        vv, uu = np.ogrid[v0:v1, u0:u1]
        ring = (np.abs(vv - v) > inner) | (np.abs(uu - u) > inner)
        if ring.sum() < 16:
            continue
        floor = float(np.median(patch[ring]))
        ratio = peak / floor if floor > 0.0 else np.inf
        if ratio >= min_ratio:
            out.append((v, u, peak, ratio))
    return out


def persistence_profile(subs: list[np.ndarray], v: int, u: int) -> np.ndarray:
    """The candidate's amplitude across the blocks, over its own mean.

    A 2-cell search window around the cell, because a block's mainlobe
    is K times wider in cross-range and can sit a sample off.
    """
    prof = []
    for sub in subs:
        v0, v1 = max(0, v - 2), min(sub.shape[0], v + 3)
        u0, u1 = max(0, u - 2), min(sub.shape[1], u + 3)
        prof.append(float(sub[v0:v1, u0:u1].max()))
    arr = np.asarray(prof)
    mean = float(arr.mean())
    return arr / mean if mean > 0.0 else arr


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cphd", type=pathlib.Path, help="path to the collect, under data/")
    parser.add_argument("--spotlight", type=pathlib.Path, default=SPOTLIGHT)
    parser.add_argument("--spacing", type=float, default=1.0, help="lattice spacing, metres")
    parser.add_argument("--counts", type=int, default=2048, help="square lattice side")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--min-isolation", type=float, default=30.0, help="peak / annulus-median gate"
    )
    parser.add_argument("--inner", type=int, default=4)
    parser.add_argument("--outer", type=int, default=24)
    parser.add_argument(
        "--subapertures", type=int, default=8, help="equal dwell blocks for the profile"
    )
    parser.add_argument(
        "--min-persistence",
        type=float,
        default=0.5,
        help="worst block over profile mean to call it persistent",
    )
    parser.add_argument(
        "--pulse-range",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        help="screen this window rather than the whole collect",
    )
    parser.add_argument(
        "--workdir",
        type=pathlib.Path,
        default=None,
        help="keep intermediates here instead of a temp dir",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.cphd.is_file():
        raise SystemExit(f"no such CPHD: {args.cphd}")
    if not args.spotlight.is_file():
        raise SystemExit(f"no spotlight binary at {args.spotlight}; build it first")
    try:
        relative = args.cphd.resolve().relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(f"{args.cphd} is not under {DATA_ROOT}/; the apps resolve there") from exc

    num_vectors, num_samples = collect_extent(args.cphd)
    first, last = args.pulse_range if args.pulse_range else (0, num_vectors)
    if not 0 <= first < last <= num_vectors:
        raise SystemExit(f"pulse range [{first}, {last}) is outside [0, {num_vectors})")
    span = last - first
    if args.subapertures < 2:
        raise SystemExit("persistence needs at least 2 blocks")
    if span // args.subapertures < 1:
        raise SystemExit(f"{span} pulses cannot be split into {args.subapertures} blocks")

    print(f"{args.cphd.name}: {num_vectors} vectors x {num_samples} samples")
    print(
        f"screening pulses [{first}, {last}) on a {args.counts}^2 IARP lattice at "
        f"{args.spacing} m ({args.counts * args.spacing / 1000.0:.2f} km square)"
    )

    context = tempfile.TemporaryDirectory() if args.workdir is None else None
    workdir = pathlib.Path(context.name) if context else args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        full = form(
            args.spotlight, str(relative), args.counts, args.spacing, "full", workdir, (first, last)
        )
        cands = isolated_candidates(
            full, args.candidates, args.inner, args.outer, args.min_isolation
        )
        if not cands:
            print(
                f"\nno cell cleared isolation {args.min_isolation}x: this scene has no "
                "isolated scatterers, so PGA cannot be trusted on it"
            )
            return 1

        block = span // args.subapertures
        subs = [
            form(
                args.spotlight,
                str(relative),
                args.counts,
                args.spacing,
                f"block{k}",
                workdir,
                (first + k * block, first + (k + 1) * block),
            )
            for k in range(args.subapertures)
        ]
        print(
            f"\n{args.subapertures} disjoint blocks of {block} pulses "
            f"({block / span * 100:.1f}% of the dwell each); "
            "profile is each block over the profile mean"
        )
        head = "  ".join(f"{i:4d}" for i in range(args.subapertures))
        print(f"\n{'cell':>13} {'peak':>9} {'isol':>6}  {head}  {'min/mean':>8}")
        rows = []
        for v, u, peak, ratio in cands:
            norm = persistence_profile(subs, v, u)
            worst = float(norm.min())
            rows.append((v, u, peak, ratio, worst))
            body = "  ".join(f"{x:4.2f}" for x in norm)
            print(f"{f'({v},{u})':>13} {peak:9.0f} {ratio:6.0f}  {body}  {worst:8.2f}")

        print()
        persistent = []
        for v, u, peak, ratio, worst in rows:
            if worst >= args.min_persistence:
                verdict = "PERSISTENT"
                persistent.append((peak, v, u, ratio, worst))
            elif worst >= 0.5 * args.min_persistence:
                verdict = "partial"
            else:
                verdict = "GLINT"
            print(f"({v},{u}): worst block is {worst:.2f} of its mean -> {verdict}")
        if not persistent:
            print("\nno persistent scatterer here; not a PGA vehicle")
            return 1
        # Brightest of the persistent ones, not the brightest overall and
        # not the most persistent: PGA wants signal-to-clutter AND
        # persistence, and on Kourou the scene's brightest return is a
        # glint -- which is the whole reason this screen exists.
        peak, v, u, ratio, worst = max(persistent)
        print(
            f"\npick ({v},{u}): peak {peak:.0f}, isolation {ratio:.0f}x, worst block "
            f"{worst:.2f} -- brightest of {len(persistent)} persistent candidate(s)"
        )
        print(
            f"a scene config over pulses [{first}, {last}) with "
            f"lattice.counts {args.counts} / lattice.spacing {args.spacing} sees it at "
            f"row {v}, column {u}"
        )
        return 0
    finally:
        if context is not None:
            context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
