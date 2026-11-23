# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that a package-linked FMCW graph reports the injected target."""

import math
import re
import subprocess
import sys


REPORT = re.compile(
    r"cluster \d+: \d+ pts, centroid range ([-+\d.]+) m, velocity "
    r"([-+\d.]+) m/s(?:, az ([-+\d.]+) rad, el ([-+\d.]+) rad)?"
)


def main() -> int:
    executable, config, graph = sys.argv[1:4]
    result = subprocess.run(
        [executable, config], capture_output=True, text=True, timeout=120, check=False
    )
    log = result.stdout + result.stderr
    print(log)
    if result.returncode:
        raise RuntimeError(f"{graph} exited {result.returncode}")
    frames = re.split(r"DBSCAN clusters:", log)[1:]
    if len(frames) != 2:
        raise RuntimeError(f"{graph} did not report both frames and a cluster")

    expected_angle = {"siso": None, "simo": (0.15, 0.09), "mimo_ddm": (0.15, 0.10)}[graph]
    # The smoke presets inject the same 245 m, approximately -6.5 m/s target.
    # Bounds cover two range bins, two Doppler bins and one coarse angle cell.
    for frame in frames:
        found = False
        for report in REPORT.finditer(frame):
            distance, speed = map(float, report.group(1, 2))
            if not math.isfinite(distance) or not math.isfinite(speed):
                continue
            if abs(distance - 245.0) > 2.1 or abs(speed + 6.5) > 1.3:
                continue
            if expected_angle is not None:
                if report.group(3) is None:
                    continue
                azimuth, elevation = map(float, report.group(3, 4))
                if not all(
                    math.isfinite(value) and abs(value - target) <= 0.2
                    for value, target in zip((azimuth, elevation), expected_angle)
                ):
                    continue
            found = True
            break
        if not found:
            raise RuntimeError(f"{graph} reported no injected target in a frame")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
