# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and consume actual artifacts in the GPU/SDK test container."""

import argparse
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
from pathlib import Path


def run(*command, cwd, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def cache_values(build):
    return dict(
        re.findall(
            r"^([A-Za-z_][A-Za-z_0-9]*):[^=\n]+=([^\n]*)$",
            (build / "CMakeCache.txt").read_text(),
            re.M,
        )
    )


def configure(args, name, *options):
    """Probe CMake configuration semantics; artifacts use the public package CLI."""
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
    if result.returncode:
        raise RuntimeError(output)
    return build


def check_configuration(args, full):
    values = cache_values(full)
    # CMake evaluates the curated package list; registrations do not set policy.
    expected = set(filter(None, values["HOLOSCAN_RADAR_PKG_OPERATORS"].split(";")))
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
    explicit = configure(args, "explicit-tests", "-DHOLOSCAN_RADAR_BUILD_TESTING=ON")
    if not (explicit / "tests/CTestTestfile.cmake").exists():
        raise RuntimeError("Explicit tests ON was overridden by packaging")
    # On a reused tree the explicit/cached setting must still win.
    configure(args, "explicit-tests")
    if cache_values(explicit)["HOLOSCAN_RADAR_BUILD_TESTING"] != "ON":
        raise RuntimeError("Package reconfiguration lost the cached test setting")
    return full, expected


def check_operator_libraries(prefix, expected):
    """Require lib<selection-key>.* per the convention documented in pkg/CMakeLists.txt.

    Selection keys name operator subdirectories, not targets. This checks library
    presence only; the external consumer separately checks representative usability.
    """
    for name in expected:
        if not any(path.is_file() for path in prefix.rglob(f"lib{name}.*")):
            raise RuntimeError(f"Package artifact missing operator library: {name}")


def package_with_cli(args, generators):
    artifacts = args.work / "artifacts"
    artifacts.mkdir()
    build_parent = args.work / "cli-build"
    env = os.environ.copy()
    env["HOLOSCAN_CLI_BUILD_PARENT_DIR"] = str(build_parent)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = args.parallel
    # DEB/TGZ builds take no CMAKE_ARGS; keep their architectures matched to the wheel.
    env["CUDAARCHS"] = args.architectures
    # The CLI delegates wheel builds to the production backend, with isolation.
    env["SKBUILD_BUILD_DIR"] = str(build_parent / "wheel")
    env["CMAKE_ARGS"] = shlex.join(
        [
            *shlex.split(env.get("CMAKE_ARGS", "")),
            f"-Dholoscan_DIR={args.sdk_dir}",
            f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}",
        ]
    )
    if shutil.which("ccache"):
        env["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
        env["CMAKE_CUDA_COMPILER_LAUNCHER"] = "ccache"
    run(
        sys.executable,
        "-m",
        "holoscan_cli",
        "--project-root",
        args.source,
        "package",
        "holoscan-radar",
        f"--pkg-generator={generators}",
        "--language",
        "python" if generators == "WHEEL" else "cpp",
        # Already inside the SDK/GPU container; do not start nested containers.
        "--local",
        cwd=artifacts,
        env=env,
    )
    if generators == "WHEEL":
        for wheel in (build_parent / "dist").glob("*.whl"):
            shutil.copy2(wheel, artifacts)
    return build_parent, artifacts


def check_debian(args):
    build_parent, artifacts = package_with_cli(args, "DEB,TGZ")
    build = build_parent / "holoscan_radar/package"
    _, expected = check_configuration(args, build)
    packages = list(artifacts.glob("*.deb"))
    if len(packages) != 1:
        raise RuntimeError(f"Expected one Debian artifact, found {packages}")
    depends = subprocess.check_output(
        ["dpkg-deb", "--field", str(packages[0]), "Depends"], text=True
    )
    has_tinyxml2 = any(dep.strip().split()[0] == "libtinyxml2-dev" for dep in depends.split(","))
    if has_tinyxml2 != ("cphd_reader_op" in expected):
        raise RuntimeError(f"Debian tinyxml2 dependency does not match CPHD selection: {depends}")
    unpacked = args.work / "unpacked"
    run("dpkg-deb", "--extract", packages[0], unpacked, cwd=args.work)
    prefix = unpacked / "opt/nvidia/holoscan"
    check_operator_libraries(prefix, expected)
    if (prefix / "bin").exists() and any((prefix / "bin").iterdir()):
        raise RuntimeError("Operator-only Debian artifact contains executables")
    if (prefix / "holoscan_radar").exists():
        raise RuntimeError("C++-only Debian artifact unexpectedly includes Python bindings")
    archives = list(artifacts.glob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one TGZ artifact, found {archives}")
    archive_root = args.work / "archive"
    with tarfile.open(archives[0]) as archive:
        archive.extractall(archive_root, filter="data")
    exports = list(archive_root.rglob("holoscan_radar_targets.cmake"))
    if len(exports) != 1:
        raise RuntimeError(f"Expected one Radar export tree in TGZ, found {exports}")
    check_operator_libraries(exports[0].parents[3], expected)
    check_cmake_package(prefix, expected)
    check_native_consumer(args, prefix, "debian", expected=expected)
    check_installed_headers(args, prefix)
    check_cmake_package(exports[0].parents[3], expected)
    check_native_consumer(args, exports[0].parents[3], "tgz", expected=expected)
    # Ordinary installs must support discovery with the Debian option disabled.
    subset = configure(
        args,
        "install-subset",
        "-DPKG_holoscan_radar=OFF",
        "-DOP_fmcw_range_processor_op=ON",
        "-DHOLOSCAN_RADAR_BUILD_TESTING=OFF",
    )
    run("cmake", "--build", subset, "--parallel", args.parallel, cwd=args.work)
    installed = args.work / "installed-subset"
    run("cmake", "--install", subset, "--prefix", installed, cwd=args.work)
    relocated = args.work / "relocated-subset"
    installed.rename(relocated)
    check_cmake_package(relocated, {"fmcw_range_processor_op"})
    check_native_consumer(
        args,
        relocated,
        "subset",
        expected={"fmcw_range_processor_op"},
        absent={"cphd_reader_op", "fmcw_doppler_processor_op"},
    )


def check_cmake_package(prefix, expected):
    configs = list(prefix.rglob("holoscan-radar-config.cmake"))
    if len(configs) != 1:
        raise RuntimeError(f"Expected one Radar config, found {configs}")
    directory = configs[0].parent
    if not (directory / "holoscan-radar-config-version.cmake").is_file():
        raise RuntimeError("Missing Radar version config")
    exports = list(prefix.rglob("holoscan_*_targets.cmake"))
    if exports != [directory / "holoscan_radar_targets.cmake"]:
        raise RuntimeError(f"Unexpected Radar exports: {exports}")
    content = exports[0].read_text()
    for target in expected:
        if f"holoscan::{target}" not in content:
            raise RuntimeError(f"Missing exported target: {target}")


def check_installed_headers(args, prefix):
    # Compile each installed header alone so includes of uninstalled files fail here.
    build = args.work / "installed-headers"
    run(
        "cmake",
        "-S",
        args.source / "tests/packaging/headers",
        "-B",
        build,
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-Dholoscan_DIR={args.sdk_dir}",
        f"-DRADAR_INCLUDE_ROOT={prefix / 'include/holoscan/radar'}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}",
        cwd=args.work,
    )
    run("cmake", "--build", build, "--parallel", args.parallel, cwd=args.work)


def check_native_consumer(args, prefix, name, expected=(), absent=()):
    # Build against the extracted artifact, with no source/build include paths.
    consumer = args.work / f"consumer-{name}"
    command = [
        "cmake",
        "-S",
        args.source / "tests/packaging/consumer",
        "-B",
        consumer,
        f"-DCMAKE_PREFIX_PATH={prefix}",
        f"-Dholoscan_DIR={args.sdk_dir}",
        f"-DRADAR_EXPECT_TARGETS={';'.join(sorted(expected))}",
        f"-DRADAR_ABSENT_TARGETS={';'.join(sorted(absent))}",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}",
    ]
    run(*command, cwd=args.work)
    run("cmake", "--build", consumer, "--parallel", args.parallel, cwd=args.work)
    run(consumer / "radar_package_consumer", cwd=args.work)

    version = tomllib.loads((args.source / "pyproject.toml").read_text())["project"]["version"]
    # Probe discovery failures in separate build trees, never stale caches.
    for label, options, success in (
        ("exact", [f"-DRADAR_VERSION={version};EXACT"], True),
        ("future", ["-DRADAR_VERSION=999.0"], False),
        ("component", ["-DRADAR_COMPONENT_ARGS=COMPONENTS;unsupported"], False),
    ):
        probe = command.copy()
        probe[probe.index("-B") + 1] = args.work / f"consumer-{name}-{label}"
        if name == "subset":
            probe.append("-DCMAKE_DISABLE_FIND_PACKAGE_tinyxml2=TRUE")
        result = subprocess.run(
            [*map(str, probe), *options], cwd=args.work, text=True, capture_output=True, check=False
        )
        if (result.returncode == 0) != success:
            raise RuntimeError(
                f"Unexpected {label} discovery result: {result.stdout}{result.stderr}"
            )


def check_wheel(args):
    _, artifacts = package_with_cli(args, "WHEEL")
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
    smoke = [python, "-I", Path(__file__).resolve().with_name("wheel_smoke.py")]
    # Isolated mode drops PYTHONPATH; forward the SDK bindings this interpreter sees.
    sdk = importlib.util.find_spec("holoscan")
    if sdk and sdk.submodule_search_locations:
        smoke += ["--sdk-site", Path(list(sdk.submodule_search_locations)[0]).parent]
    run(*smoke, cwd=args.work)


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
