#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///

"""Sweep AutofocusOp accuracy against injected truth and tabulate the result.

Characterization, not a test: it runs the `autofocus` application over a grid
of injected error profiles and operator settings on a real collect, scrapes
`AutofocusScoreOp`'s affine-removed residual, and prints a table. CI cannot do
this -- the fixtures are 5-pulse generated CPHDs, far too small for PGA -- so
the numbers in `operators/sar/autofocus_op/README.md` come from here and
this script is how they are reproduced or refuted.

The grid axes are the ones that have been found to matter, in order: the error
*shape* (a pure quadratic is the easiest case and flatters the estimator), the
magnitude, the first window width, the aperture length, and the scene.

  uv run operators/sar/autofocus_op/tools/sweep.py --grid shapes
  uv run operators/sar/autofocus_op/tools/sweep.py --grid windows --pulses 8000
  uv run operators/sar/autofocus_op/tools/sweep.py --grid symmetry --scenes all
  uv run operators/sar/autofocus_op/tools/sweep.py --grid shapes --base config/other.yaml

The scene axis matters for a reason worth stating plainly: Kourou, which
produced every accuracy number through 2026-10-30, was *screened* for PGA --
an isolated corner reflector lit across the whole dwell. The operator's
precondition is therefore satisfied there by construction. That buys
reliability, not accuracy: measured across scenes, Kourou is the least accurate
of the three that focus, because one reflector gives the gradient estimate one
range bin to average where a city gives it hundreds. `--scenes all` spans from
that screened case to agriculture,
which has no dominant scatterer anywhere in it and is expected to fail; the
negative control is what separates "this shape breaks the estimator" from
"this scene does not meet the precondition".

Residuals are read against the targets in `docs/TESTING.md`: pi/8
(0.39 rad) to pass, pi/4 (0.785 rad) as the hard floor.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
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
SHARP_RE = re.compile(r"\(([\d.eE+]+)x")
STATUS_RE = re.compile(r"termination (\w+), (\d+) iteration\(s\) accepted")
REFUSED_RE = re.compile(r"autofocus REFUSED \((refused_\w+)\)")
REFUSALS = {"refused_sharpness", "refused_scatterers", "refused_max_correction"}
TERMINATIONS = REFUSALS | {"converged", "budget_exhausted", "stopped_scatterers"}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
CONFIG_DIR = REPO_ROOT / "applications/sar/autofocus/config"

# One config per scene, ordered easiest-to-hardest for PGA's precondition. The
# configs carry the reasoning for each choice of scene; what matters here is
# that they describe the same experiment on different ground, so the only thing
# a row-to-row difference can be attributed to is the scene.
SCENES = {
    "kourou": CONFIG_DIR / "autofocus.yaml",
    "atlanta": CONFIG_DIR / "autofocus_atlanta.yaml",
    "houston": CONFIG_DIR / "autofocus_houston.yaml",
    "imperial": CONFIG_DIR / "autofocus_imperial.yaml",
}


@dataclasses.dataclass
class Case:
    """One grid point: a label plus the YAML overrides that produce it."""

    label: str
    error: dict
    # Window width in aux samples rather than as a fraction: the fraction is of
    # the aux width, which tracks the aperture, so a fixed fraction means
    # different sample counts at different --pulses and the two grids that
    # sweep width would silently stop sweeping what they say they do.
    window_samples: int | None = None
    num_iterations: int | None = None
    pulses: int | None = None


@dataclasses.dataclass
class Result:
    scene: str
    label: str
    baseline_rad: float
    residual_rad: float
    sharpness: float
    iterations: int
    termination: str

    @property
    def refused(self) -> bool:
        """Refusal is operator provenance, independent of the measured residual."""
        return self.termination in REFUSALS

    @property
    def verdict(self) -> str:
        # Refusal first. A refusal leaves the residual at the baseline, so on
        # any case whose injected error is already under PASS_RAD the old
        # ordering reported the operator declining to act as a "pass".
        if self.refused:
            return "refused"
        if self.residual_rad <= PASS_RAD:
            return "pass"
        if self.residual_rad <= FLOOR_RAD:
            return "marginal"
        if self.residual_rad > self.baseline_rad:
            return "HARMFUL"
        return "FAIL"


def quadratic(metres: float) -> dict:
    return {"polynomial": [0, 0, metres]}


def sinusoid(metres: float, cycles: float) -> dict:
    return {"sinusoids": [{"amplitude": metres, "cycles": cycles, "phase_deg": 0}]}


def sinusoid_at(metres: float, cycles: float, phase_deg: float) -> dict:
    return {"sinusoids": [{"amplitude": metres, "cycles": cycles, "phase_deg": phase_deg}]}


def walk(metres: float, seed: int = 7) -> dict:
    return {"random_walk": {"rms": metres, "seed": seed}}


def symmetry_grid() -> list[Case]:
    """The five shapes that tell an estimator bug from a shape limit.

    Symmetry about the dwell centre is the axis the quadratic-only sweeps held
    fixed, and it is the one that found both 2026-10-30 defects. A quadratic
    (s^2) and a 90-degree sinusoid are even; a cubic (s^3) and a 0-degree
    sinusoid are odd; 45 degrees is neither. Reversing an even error is a
    no-op, and on an odd error it is indistinguishable from a global negation,
    which AutofocusScoreOp's sign search absorbs -- so only the last row can
    fail an axis-handedness bug outright.

    This is the regression set for the emit path: all five inside the target,
    on a scene that meets the precondition, or the mapping is wrong again.
    """
    return [
        Case("quadratic (even)", {"polynomial": [0, 0, 0.02]}),
        Case("cosine 1 cyc (even)", sinusoid_at(0.02, 1.0, 90.0)),
        Case("cubic (odd)", {"polynomial": [0, 0, 0, 0.02]}),
        Case("sine 1 cyc (odd)", sinusoid_at(0.02, 1.0, 0.0)),
        Case("sinusoid 45 deg", sinusoid_at(0.02, 1.0, 45.0)),
    ]


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


def aperture_grid() -> list[Case]:
    """Aperture length at a fixed error and a fixed window *fraction*.

    The fraction is held rather than the sample count because PGA's window
    schedule is scale-relative: it shrinks geometrically from a fraction of the
    aux width, so holding samples would change the schedule's starting point as
    well as the aperture. Only Kourou's 23375-vector collect spans the top of
    this range; the other scene configs have shorter usable windows.
    """
    return [
        Case(
            f"quadratic 0.02 m / {pulses} pulses",
            quadratic(0.02),
            pulses=pulses,
            window_samples=pulses // 8,
        )
        for pulses in (2000, 4000, 8000, 12000, 16000, 23375)
    ]


def window_grid() -> list[Case]:
    """First window width against error magnitude, the 2026-10-29 tentative table."""
    cases: list[Case] = []
    for metres in (0.0, 0.02, 0.1, 0.4, 0.8):
        for width in (350, 1000, 2000, 4000):
            cases.append(
                Case(
                    f"quadratic {metres} m / {width} samples",
                    quadratic(metres) if metres else {"polynomial": [0, 0, 0.0]},
                    window_samples=width,
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
            cases.append(Case(f"{name} / {width} samples", error, window_samples=width))
    return cases


GRIDS = {
    "apertures": aperture_grid,
    "shapes": shape_grid,
    "symmetry": symmetry_grid,
    "windows": window_grid,
    "shape-windows": shape_window_grid,
}


def render(base: dict, case: Case) -> dict:
    cfg = copy.deepcopy(base)
    pe = {"axis": "look", "score_warn_rms_rad": 1.0e9}
    pe.update(case.error)
    cfg["position_error"] = pe
    reader = cfg.setdefault("cphd_reader", {})
    first, last = reader.get("pulse_range", [0, 0])
    if case.pulses is not None:
        # Trim from the config's own first pulse, not from 0: a scene config
        # picks its window because that is where the scene is lit (Atlanta's
        # closest-approach frame starts at pulse 91273), and restarting at 0
        # would silently image a different part of every pass.
        last = first + case.pulses
        reader["pulse_range"] = [first, last]
        cfg.setdefault("aperture_manager", {})["max_aperture_pulses"] = case.pulses
    af = cfg.setdefault("autofocus", {})
    if case.window_samples is not None:
        aperture = last - first
        if aperture <= 0:
            raise ValueError("a window-width grid needs a known aperture: pass --pulses")
        af["window_fraction"] = case.window_samples / aperture
    if case.num_iterations is not None:
        af["num_iterations"] = case.num_iterations
    cfg["visualize"] = False
    return cfg


def run_one(binary: pathlib.Path, cfg: dict, scene: str, label: str) -> Result | None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
        path = handle.name
    try:
        proc = subprocess.run([str(binary), path], capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        # One stalled case must not end a multi-hour multi-scene sweep.
        print(f"  !! {scene}/{label}: timed out after 1800 s", file=sys.stderr)
        return None
    finally:
        pathlib.Path(path).unlink(missing_ok=True)
    text = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = "\n".join(text.strip().splitlines()[-6:])
        print(
            f"  !! {scene}/{label}: application failed (exit {proc.returncode})\n{tail}",
            file=sys.stderr,
        )
        return None
    # Last match, not first. A configuration that forms more than one image
    # emits a score per tick, and taking the first silently reported tick 0
    # for the whole run with no indication the others existed.
    scores = list(SCORE_RE.finditer(text))
    sharps = list(SHARP_RE.finditer(text))
    statuses = list(STATUS_RE.finditer(text))
    refusals = list(REFUSED_RE.finditer(text))
    if len(scores) > 1:
        print(
            f"  .. {scene}/{label}: {len(scores)} scored images, reporting the last",
            file=sys.stderr,
        )
    score = scores[-1] if scores else None
    sharp = sharps[-1] if sharps else None
    if not score:
        tail = "\n".join(text.strip().splitlines()[-6:])
        print(
            f"  !! {scene}/{label}: no score line (exit {proc.returncode})\n{tail}",
            file=sys.stderr,
        )
        return None
    # Older reports emit only a refusal warning on refused images. Accept
    # that explicit provenance too; never infer refusal from a residual.
    if statuses and (not refusals or statuses[-1].start() > refusals[-1].start()):
        termination = statuses[-1].group(1)
        iterations = int(statuses[-1].group(2))
    elif refusals:
        termination = refusals[-1].group(1)
        iterations = 0
    else:
        print(f"  !! {scene}/{label}: no termination provenance", file=sys.stderr)
        return None
    residual, baseline = float(score.group(1)), float(score.group(2))
    if termination not in TERMINATIONS or not all(
        math.isfinite(value) and value >= 0 for value in (residual, baseline)
    ):
        print(f"  !! {scene}/{label}: invalid score or termination", file=sys.stderr)
        return None
    return Result(
        scene=scene,
        label=label,
        residual_rad=residual,
        baseline_rad=baseline,
        sharpness=float(sharp.group(1)) if sharp else float("nan"),
        iterations=iterations,
        termination=termination,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=sorted(GRIDS), default="shapes")
    parser.add_argument(
        "--scenes",
        default="kourou",
        help="comma-separated scene names, or 'all' (" + ", ".join(SCENES) + ")",
    )
    parser.add_argument(
        "--base",
        type=pathlib.Path,
        default=None,
        help="a config path, bypassing --scenes (one scene, labelled by file stem)",
    )
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=(REPO_ROOT / "build/applications/sar/autofocus/cpp/autofocus"),
    )
    parser.add_argument(
        "--pulses", type=int, default=None, help="aperture for grids that do not set one"
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--window-samples", type=int, default=None)
    args = parser.parse_args()

    if not args.binary.exists():
        print(f"no autofocus binary at {args.binary}", file=sys.stderr)
        return 2

    if args.base is not None:
        scenes = {args.base.stem: args.base}
    elif args.scenes == "all":
        scenes = dict(SCENES)
    else:
        scenes = {}
        for name in args.scenes.split(","):
            if name not in SCENES:
                print(f"unknown scene {name!r}; have {', '.join(SCENES)}", file=sys.stderr)
                return 2
            scenes[name] = SCENES[name]
    missing = [str(p) for p in scenes.values() if not p.exists()]
    if missing:
        print("no config at " + ", ".join(missing), file=sys.stderr)
        return 2

    cases = GRIDS[args.grid]()
    for case in cases:
        # Fill, never overwrite -- same rule as the two below. The apertures
        # grid varies `pulses` per case, so overwriting it collapsed six rows
        # onto one configuration while still labelling them 2000..23375, and
        # left each case's `window_samples = pulses // 8` describing an
        # aperture that was no longer being run.
        if args.pulses is not None and case.pulses is None:
            case.pulses = args.pulses
        if args.iterations is not None and case.num_iterations is None:
            case.num_iterations = args.iterations
        if args.window_samples is not None and case.window_samples is None:
            case.window_samples = args.window_samples

    scene_w = max(len(s) for s in scenes)
    width = max(len(c.label) for c in cases)
    print(
        f"{'scene'.ljust(scene_w)}  {'case'.ljust(width)}  {'baseline':>9}  {'residual':>9}  "
        f"{'removed':>8}  {'sharp':>8}  verdict"
    )
    print("-" * (scene_w + width + 52))
    results: list[Result] = []
    failed = 0
    for scene, config in scenes.items():
        base = yaml.safe_load(config.read_text())
        for case in cases:
            result = run_one(args.binary, render(base, case), scene, case.label)
            if result is None:
                failed += 1
                continue
            results.append(result)
            removed = result.baseline_rad / result.residual_rad if result.residual_rad else 0.0
            print(
                f"{scene.ljust(scene_w)}  {result.label.ljust(width)}  "
                f"{result.baseline_rad:9.3f}  {result.residual_rad:9.3f}  {removed:7.1f}x  "
                f"{result.sharpness:7.2f}x  {result.verdict}"
            )

    inside = [r for r in results if r.residual_rad <= FLOOR_RAD]
    refused = [r for r in results if r.refused]
    print(f"\n{len(inside)} of {len(results)} inside the {FLOOR_RAD:.3f} rad floor")
    if refused:
        print(f"{len(refused)} of {len(results)} refused by autofocus guards (image untouched)")
    if len(scenes) > 1:
        # Per scene as well as overall: the whole point of the scene axis is
        # that a shape failing everywhere and a shape failing on one scene are
        # different findings, and a pooled count hides which one happened.
        for scene in scenes:
            rows = [r for r in results if r.scene == scene]
            if rows:
                ok = sum(1 for r in rows if r.residual_rad <= FLOOR_RAD)
                gave_up = sum(1 for r in rows if r.refused)
                worst = max(rows, key=lambda r: r.residual_rad)
                note = f", {gave_up} refused" if gave_up else ""
                print(
                    f"  {scene.ljust(scene_w)}  {ok}/{len(rows)} inside the floor{note}, "
                    f"worst {worst.residual_rad:.3f} rad ({worst.label})"
                )
    if failed:
        print(
            f"{failed} of {len(cases) * len(scenes)} cases failed to produce valid evidence",
            file=sys.stderr,
        )
    # Numerical verdicts characterize the estimator; execution failures make
    # the campaign incomplete and must propagate to its caller.
    return 1 if failed else 0


if __name__ == "__main__":
    if shutil.which("nvidia-smi") is None:
        print("warning: no nvidia-smi; the app needs a CUDA GPU", file=sys.stderr)
    raise SystemExit(main())
