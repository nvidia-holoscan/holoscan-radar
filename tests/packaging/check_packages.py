# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and consume actual artifacts in the GPU/SDK test container."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run(*command, cwd, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def registered_operators(source):
    """Use operator registrations, independently of the package's selection list."""
    names = set()
    for path in (source / "operators").rglob("CMakeLists.txt"):
        names.update(
            re.findall(r"^holoscan_radar_add_operator\(\s*(\w+)", path.read_text(), re.MULTILINE)
        )
    if not names:
        raise RuntimeError("No Radar operator registrations found")
    return names


def cache_values(build):
    return dict(
        re.findall(
            r"^([A-Za-z_][A-Za-z_0-9]*):[^=\n]+=([^\n]*)$",
            (build / "CMakeCache.txt").read_text(),
            re.M,
        )
    )


def configure(args, name, *options, error=None):
    build = args.work / name
    command = [
        "cmake",
        "-S",
        str(args.source),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-Dholoscan_DIR={args.sdk_dir}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_ALL=OFF",
        "-DPKG_holoscan_radar=ON",
        "-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF",
        "-DHOLOSCAN_RADAR_TEST_PACKAGING=OFF",
        *options,
    ]
    if shutil.which("ccache"):
        command += [
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        ]
    print(f"Configuring {name}", flush=True)
    result = subprocess.run(command, cwd=args.work, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    if error:
        if result.returncode == 0 or f"Unknown Radar package operator: {error}" not in output:
            raise RuntimeError(f"Expected rejection of {error}:\n{output}")
    elif result.returncode:
        raise RuntimeError(output)
    return build


def check_configuration(args):
    full = configure(args, "full")
    values = cache_values(full)
    expected = registered_operators(args.source)
    enabled = {
        name.removeprefix("OP_")
        for name, value in values.items()
        if name.startswith("OP_") and value == "ON"
    }
    if enabled != expected:
        raise RuntimeError(f"Package inventory mismatch: expected {expected}, enabled {enabled}")
    if values["HOLOSCAN_RADAR_BUILD_TESTING"] != "OFF":
        raise RuntimeError("Fresh package-only configuration must default tests OFF")
    if any(value == "ON" for name, value in values.items() if name.startswith("APP_")):
        raise RuntimeError("Package-only configuration enabled example applications")
    subset = configure(
        args, "subset", "-DHOLOSCAN_RADAR_PKG_OPERATORS=clutter_filter_op;cphd_reader_op"
    )
    selected = {
        name.removeprefix("OP_")
        for name, value in cache_values(subset).items()
        if name.startswith("OP_") and value == "ON"
    }
    if selected != {"clutter_filter_op", "cphd_reader_op"}:
        raise RuntimeError(f"Unexpected subset selection: {selected}")
    for name in ("cphd_reder_op", "radar_common"):
        configure(args, f"invalid-{name}", f"-DHOLOSCAN_RADAR_PKG_OPERATORS={name}", error=name)
    configure(
        args,
        "package-disabled",
        "-DPKG_holoscan_radar=OFF",
        "-DHOLOSCAN_RADAR_PKG_OPERATORS=not_a_radar_operator",
        "-DHOLOSCAN_RADAR_BUILD_TESTING=OFF",
    )
    explicit = configure(args, "explicit-tests", "-DHOLOSCAN_RADAR_BUILD_TESTING=ON")
    if not (explicit / "tests/CTestTestfile.cmake").exists():
        raise RuntimeError("Explicit tests ON was overridden by packaging")
    # On a reused tree the explicit/cached setting must still win.
    configure(args, "explicit-tests")
    if cache_values(explicit)["HOLOSCAN_RADAR_BUILD_TESTING"] != "ON":
        raise RuntimeError("Package reconfiguration lost the cached test setting")
    return full, expected


def check_debian(args):
    build, expected = check_configuration(args)
    run("cmake", "--build", build, "--parallel", args.parallel, cwd=args.work)
    artifacts = args.work / "artifacts"
    artifacts.mkdir()
    run(
        "cpack",
        "--config",
        build / "pkg/CPackConfig-holoscan-radar.cmake",
        "-B",
        artifacts,
        cwd=build,
    )
    packages = list(artifacts.glob("*.deb"))
    if len(packages) != 1:
        raise RuntimeError(f"Expected one Debian artifact, found {packages}")
    unpacked = args.work / "unpacked"
    run("dpkg-deb", "--extract", packages[0], unpacked, cwd=args.work)
    prefix = unpacked / "opt/nvidia/holoscan"
    for name in expected:
        if not any(prefix.rglob(f"lib{name}.*")):
            raise RuntimeError(f"Debian artifact missing operator library: {name}")
    if (prefix / "bin").exists() and any((prefix / "bin").iterdir()):
        raise RuntimeError("Operator-only Debian artifact contains executables")
    if (prefix / "holoscan_radar").exists():
        raise RuntimeError("C++-only Debian artifact unexpectedly includes Python bindings")
    # Build against the extracted artifact, with no source/build include paths.
    consumer = args.work / "consumer"
    run(
        "cmake",
        "-S",
        args.source / "tests/packaging/consumer",
        "-B",
        consumer,
        f"-DRADAR_PREFIX={prefix}",
        f"-Dholoscan_DIR={args.sdk_dir}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}",
        cwd=args.work,
    )
    run("cmake", "--build", consumer, "--parallel", args.parallel, cwd=args.work)
    run(consumer / "radar_package_consumer", cwd=args.work)


def check_wheel(args):
    artifacts = args.work / "artifacts"
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = args.parallel
    # Backend options preserve pyproject.toml's production wheel selection.
    command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        artifacts,
        f"-Cbuild-dir={args.work / 'build'}",
        f"-Ccmake.define.holoscan_DIR={args.sdk_dir}",
        f"-Ccmake.define.CMAKE_CUDA_ARCHITECTURES={args.architectures}",
    ]
    if shutil.which("ccache"):
        command += [
            "-Ccmake.define.CMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-Ccmake.define.CMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        ]
    run(*command, args.source, cwd=args.work, env=env)
    wheels = list(artifacts.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel artifact, found {wheels}")
    # Inherit SDK dependencies, but require Radar itself to come from this venv.
    environment = args.work / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / "bin/python"
    run(
        python,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        wheels[0],
        cwd=args.work,
    )
    sdk_site = args.sdk_dir.parents[2] / "lib/python3.12/site-packages"
    run(
        python,
        "-I",
        "-c",
        """
import importlib.metadata
import pathlib
import sys
sys.path.append(sys.argv[1])
import holoscan as hs
import holoscan_radar as radar
root = pathlib.Path(sys.prefix)
assert pathlib.Path(radar.__file__).is_relative_to(root), radar.__file__
assert pathlib.Path(radar._radar.__file__).is_relative_to(root), radar._radar.__file__
entry = importlib.metadata.distribution('holoscan-radar').entry_points
assert any(e.group == 'holoscan.modules' and e.name == 'radar' for e in entry)
assert radar._REGISTRY_NAMES
assert all(callable(getattr(radar, name)) for name in radar._REGISTRY_NAMES)
graph = hs.Graph('installed_radar')
extents = dict(max_chirps=4, max_samples_per_chirp=16, max_range_bins=16)
radar.FMCWRangeProcessorOpF32(graph, 'single', extents=extents)
radar.FMCWRangeProcessorOpF64(graph, 'dual', extents=extents)
from holoscan_radar.applications.siso import build_graph
assert callable(build_graph)
print('Installed wheel import, provider discovery, and Graph authoring passed')
""",
        sdk_site,
        cwd=args.work,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("debian", "wheel"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--sdk-dir", required=True, type=Path)
    parser.add_argument("--architectures", default="native")
    parser.add_argument("--parallel", default=os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2"))
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.sdk_dir = args.sdk_dir.resolve()
    args.work = args.work.resolve()
    # A fresh tree is essential: cached OP_* selections would hide omissions.
    args.work.mkdir(parents=True, exist_ok=True)
    args.work = Path(tempfile.mkdtemp(prefix="run-", dir=args.work))
    if args.mode == "debian":
        check_debian(args)
    else:
        check_wheel(args)
    print(f"PASS Packaging_{args.mode}; artifacts: {args.work / 'artifacts'}", flush=True)


if __name__ == "__main__":
    main()
