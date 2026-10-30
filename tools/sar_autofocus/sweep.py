#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweep AutofocusOp accuracy against injected truth and tabulate the result.

Characterization, not a test: it runs the `autofocus` application over a grid
of injected error profiles and operator settings on a real collect, scrapes
`AutofocusScoreOp`'s affine-removed residual, and prints a table. CI cannot do
this -- the fixtures are 5-pulse generated CPHDs, far too small for PGA -- so
the numbers in `operators/sar/autofocus_op/OPEN_ITEMS.md` come from here and
this script is how they are reproduced or refuted.

The grid axes are the ones that have been found to matter, in order: the error
*shape* (a pure quadratic is the easiest case and flatters the estimator), the
magnitude, the first window width, and the aperture length.

  python3 tools/sar_autofocus/sweep.py --grid shapes
  python3 tools/sar_autofocus/sweep.py --grid windows --pulses 8000
  python3 tools/sar_autofocus/sweep.py --grid shapes --base config/other.yaml

Residuals are read against the targets in `docs/planning/SAR/TESTING.md`: pi/8
(0.39 rad) to pass, pi/4 (0.785 rad) as the hard floor.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import yaml

PASS_RAD = 0.3927  # pi/8, the acceptance target
FLOOR_RAD = 0.7854  # pi/4, the hard floor

SCORE_RE = re.compile(r"affine-removed RMS ([-\d.eE+]+) rad .*?baseline ([-\d.eE+]+) rad RMS")
SHARP_RE = re.compile(r"\(([\d.eE+]+)x over (\d+) iteration")


@dataclasses.dataclass
class Case:
    """One grid point: a label plus the YAML overrides that produce it."""

    label: str
    error: dict
    window_fraction: float | None = None
    num_iterations: int | None = None
    pulses: int | None = None


@dataclasses.dataclass
class Result:
    label: str
    baseline_rad: float
    residual_rad: float
    sharpness: float
    iterations: int

    @property
    def verdict(self) -> str:
        if self.residual_rad <= PASS_RAD:
            return "pass"
        if self.residual_rad <= FLOOR_RAD:
            return "marginal"
        if self.residual_rad >= self.baseline_rad:
            return "HARMFUL"
        return "FAIL"


def quadratic(metres: float) -> dict:
    return {"polynomial": [0, 0, metres]}


def sinusoid(metres: float, cycles: float) -> dict:
    return {"sinusoids": [{"amplitude": metres, "cycles": cycles, "phase_deg": 0}]}


def walk(metres: float, seed: int = 7) -> dict:
    return {"random_walk": {"rms": metres, "seed": seed}}


def shape_grid() -> list[Case]:
    """Error shape at comparable magnitude -- the axis the quadratic sweeps missed.

    A quadratic is smooth, symmetric and monotone in curvature, which is the
    best case for a geometrically shrinking window. A sub-Hz oscillation (wing
    flex, autopilot limit cycle) and a drift are what an airborne platform
    actually produces.
    """
    cases: list[Case] = []
    for metres in (0.02, 0.1, 0.4):
        cases.append(Case(f"quadratic {metres} m", quadratic(metres)))
    for cycles in (1.0, 2.0, 3.0, 6.0, 12.0):
        cases.append(Case(f"sinusoid {cycles:g} cyc 0.02 m", sinusoid(0.02, cycles)))
    for metres in (0.002, 0.01, 0.05):
        cases.append(Case(f"random walk {metres} m rms", walk(metres)))
    return cases


def window_grid() -> list[Case]:
    """First window width against error magnitude, the 2026-10-29 tentative table."""
    cases: list[Case] = []
    for metres in (0.0, 0.02, 0.1, 0.4, 0.8):
        for width in (350, 1000, 2000, 4000):
            cases.append(
                Case(
                    f"quadratic {metres} m / {width} samples",
                    quadratic(metres) if metres else {"polynomial": [0, 0, 0.0]},
                    window_fraction=width / 8000.0,
                )
            )
    return cases


def shape_window_grid() -> list[Case]:
    """Window width against error *shape*, to separate sizing from mechanism.

    If a non-quadratic shape is recoverable at some width, it is a sizing
    problem like the quadratic magnitude ceiling. If no width works, the
    windowing itself is the wrong tool for that shape -- a sinusoidal phase
    error puts its energy in paired sidelobes rather than a compact smear, and
    a window centred on the peak clips exactly the part that carries it.
    """
    cases: list[Case] = []
    shapes = [
        ("quadratic 0.02 m", quadratic(0.02)),
        ("sinusoid 1 cyc", sinusoid(0.02, 1.0)),
        ("sinusoid 3 cyc", sinusoid(0.02, 3.0)),
        ("sinusoid 12 cyc", sinusoid(0.02, 12.0)),
        ("random walk 0.01 m", walk(0.01)),
    ]
    for name, error in shapes:
        for width in (350, 1000, 2000, 4000, 8000):
            cases.append(Case(f"{name} / {width} samples", error, window_fraction=width / 8000.0))
    return cases


GRIDS = {"shapes": shape_grid, "windows": window_grid, "shape-windows": shape_window_grid}


def render(base: dict, case: Case) -> dict:
    cfg = copy.deepcopy(base)
    pe = {"axis": "look", "score_warn_rms_rad": 1.0e9}
    pe.update(case.error)
    cfg["position_error"] = pe
    af = cfg.setdefault("autofocus", {})
    if case.window_fraction is not None:
        af["window_fraction"] = case.window_fraction
    if case.num_iterations is not None:
        af["num_iterations"] = case.num_iterations
    if case.pulses is not None:
        cfg.setdefault("cphd_reader", {})["pulse_range"] = [0, case.pulses]
        cfg.setdefault("aperture_manager", {})["max_aperture_pulses"] = case.pulses
    cfg["visualize"] = False
    return cfg


def run_one(binary: pathlib.Path, cfg: dict, label: str) -> Result | None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        path = handle.name
    try:
        proc = subprocess.run([str(binary), path], capture_output=True, text=True, timeout=1800)
    finally:
        pathlib.Path(path).unlink(missing_ok=True)
    text = proc.stdout + proc.stderr
    score = SCORE_RE.search(text)
    sharp = SHARP_RE.search(text)
    if not score:
        tail = "\n".join(text.strip().splitlines()[-6:])
        print(f"  !! {label}: no score line (exit {proc.returncode})\n{tail}", file=sys.stderr)
        return None
    return Result(
        label=label,
        residual_rad=float(score.group(1)),
        baseline_rad=float(score.group(2)),
        sharpness=float(sharp.group(1)) if sharp else float("nan"),
        iterations=int(sharp.group(2)) if sharp else 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="shapes")
    parser.add_argument(
        "--base",
        type=pathlib.Path,
        default=pathlib.Path("applications/sar/autofocus/config/autofocus.yaml"),
    )
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=pathlib.Path("build/applications/sar/autofocus/cpp/autofocus"),
    )
    parser.add_argument("--pulses", type=int, default=None, help="override the aperture")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--window-fraction", type=float, default=None)
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"no autofocus binary at {args.binary}", file=sys.stderr)
        return 2
    base = yaml.safe_load(args.base.read_text())

    cases = GRIDS[args.grid]()
    for case in cases:
        if args.pulses is not None:
            case.pulses = args.pulses
        if args.iterations is not None and case.num_iterations is None:
            case.num_iterations = args.iterations
        if args.window_fraction is not None and case.window_fraction is None:
            case.window_fraction = args.window_fraction

    width = max(len(c.label) for c in cases)
    print(
        f"{'case'.ljust(width)}  {'baseline':>9}  {'residual':>9}  {'removed':>8}  "
        f"{'sharp':>8}  verdict"
    )
    print("-" * (width + 50))
    results: list[Result] = []
    for case in cases:
        result = run_one(args.binary, render(base, case), case.label)
        if result is None:
            continue
        results.append(result)
        removed = result.baseline_rad / result.residual_rad if result.residual_rad else 0.0
        print(
            f"{result.label.ljust(width)}  {result.baseline_rad:9.3f}  "
            f"{result.residual_rad:9.3f}  {removed:7.1f}x  {result.sharpness:7.2f}x  "
            f"{result.verdict}"
        )
    bad = [r for r in results if r.verdict in ("FAIL", "HARMFUL")]
    print(f"\n{len(results) - len(bad)} of {len(results)} inside the {FLOOR_RAD:.3f} rad floor")
    return 0


if __name__ == "__main__":
    if shutil.which("nvidia-smi") is None:
        print("warning: no nvidia-smi; the app needs a CUDA GPU", file=sys.stderr)
    raise SystemExit(main())
