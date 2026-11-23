# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check an installed Radar wheel using the active Python environment."""

import argparse
import importlib.metadata
import importlib.util
import sys
from pathlib import Path

COMPANION_LIBRARIES = ("libholoscan_core", "libcufft", "libcublas", "libcusolver")


def loaded_libraries():
    """Return the resolved paths of shared objects mapped into this process."""
    with open("/proc/self/maps", encoding="utf-8") as maps:
        fields = (line.split(maxsplit=5) for line in maps)
        return {Path(f[5].strip()).resolve() for f in fields if len(f) == 6 and ".so" in f[5]}


def check_library_resolution(package_lib, root, companion_wheels):
    loaded = loaded_libraries()
    for library in sorted(package_lib.glob("*.so*")):
        matches = {path for path in loaded if path.name == library.name}
        if matches != {library.resolve()}:
            raise RuntimeError(f"{library.name} resolved to {sorted(matches)}, not {library}")
    if not companion_wheels:
        return
    for prefix in COMPANION_LIBRARIES:
        matches = [path for path in loaded if path.name.startswith(prefix + ".so")]
        if not matches or not all(path.is_relative_to(root) for path in matches):
            raise RuntimeError(f"{prefix} resolved to {matches}, not a wheel under {root}")


def check_installed_wheel(expected_prefix):

    import holoscan_radar as radar

    import holoscan as hs

    root = expected_prefix.resolve()
    for module in (radar, radar._radar):
        location = Path(module.__file__).resolve()
        if not location.is_relative_to(root):
            raise RuntimeError(f"Radar import {location} is outside expected prefix {root}")
    package_lib = Path(radar.__file__).resolve().parent / "lib"
    # A Core wheel in this environment must supply Core and CUDA math libraries too.
    companion_wheels = Path(hs.__file__).resolve().is_relative_to(root)
    check_library_resolution(package_lib, root, companion_wheels)
    entries = importlib.metadata.distribution("holoscan-radar").entry_points
    if not any(entry.group == "holoscan.modules" and entry.name == "radar" for entry in entries):
        raise RuntimeError("Installed wheel is missing the Radar provider entry point")
    if not radar.__all__ or not all(callable(getattr(radar, name)) for name in radar.__all__):
        raise RuntimeError("Installed wheel is missing public operator factories")
    graph = hs.Graph("installed_radar")
    extents = dict(max_chirps=4, max_samples_per_chirp=16, max_range_bins=16)
    radar.FMCWRangeProcessorOpF32(graph, "single", extents=extents)
    radar.FMCWRangeProcessorOpF64(graph, "dual", extents=extents)
    extents = dict(
        max_pulses=2, max_samples_per_pulse=8, max_range_bins=8, max_image_v=2, max_image_u=2
    )
    for name in radar.__all__:
        if "Geom" in name or name.startswith(("SARRange", "Autofocus", "PhaseCorrection")):
            getattr(radar, name)(graph, name, extents=extents)
    # The wheel distributes operators only: no TensorDumpOp and no example-app support.
    if hasattr(radar, "TensorDumpOp"):
        raise RuntimeError("Installed wheel unexpectedly registers TensorDumpOp")
    if importlib.util.find_spec("holoscan_radar.applications") is not None:
        raise RuntimeError("Installed wheel unexpectedly contains holoscan_radar.applications")
    source = "companion wheels" if companion_wheels else "the SDK installation"
    print(
        f"Installed wheel import, provider discovery, and Graph authoring passed; "
        f"Radar libraries resolved from the wheel and Core/CUDA from {source}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sdk-site", type=Path, help="SDK Python site-packages, if not on sys.path"
    )
    parser.add_argument(
        "--expected-prefix",
        type=Path,
        default=Path(sys.prefix),
        help="Required Radar import prefix (default: active Python environment)",
    )
    args = parser.parse_args()
    if args.sdk_site:
        sys.path.append(str(args.sdk_site.resolve()))
    check_installed_wheel(args.expected_prefix)


if __name__ == "__main__":
    main()
