# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed CMake discovery metadata; no operator compute is mocked."""

import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def legal_file_fixture(source):
    """Copy the root legal-file install block and its inputs into a fixture project."""
    block = re.search(
        r"^if\(NOT SKBUILD\)\n.*?^endif\(\)\n", (ROOT / "CMakeLists.txt").read_text(), re.M | re.S
    )
    for name in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / name, source / name)
    return "include(GNUInstallDirs)\n" + block.group(0)


class InstalledConfig(unittest.TestCase):
    def test_debian_dependencies_follow_final_cphd_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 4.0)\n"
                "project(holoscan_radar VERSION 0.2.0 LANGUAGES NONE)\n"
                'set(PROJECT_DESCRIPTION "Package dependency fixture")\n'
                f'list(PREPEND CMAKE_MODULE_PATH "{ROOT}/cmake")\n'
                # Match production order: package registration precedes operators.
                f'add_subdirectory("{ROOT}/pkg/holoscan-radar" pkg)\n'
                "if(BUILD_CPHD)\nadd_library(cphd_reader_op INTERFACE)\nendif()\n"
                f'install(FILES "{ROOT}/pkg/holoscan-radar/README.md" DESTINATION share)\n'
            )
            build = root / "build"
            # Reuse the tree to verify both adding and removing the dependency.
            for enabled, shared in ((False, True), (True, True), (True, False), (False, False)):
                with self.subTest(cphd=enabled, shared=shared):
                    subprocess.run(
                        [
                            "cmake",
                            "-S",
                            str(source),
                            "-B",
                            str(build),
                            f"-DBUILD_CPHD={'ON' if enabled else 'OFF'}",
                            f"-DBUILD_SHARED_LIBS={'ON' if shared else 'OFF'}",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        [
                            "cpack",
                            "--config",
                            str(build / "pkg/CPackConfig-holoscan-radar.cmake"),
                            "-G",
                            "DEB",
                            "-B",
                            str(root / "artifacts"),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    package = next((root / "artifacts").glob("*.deb"))
                    depends = subprocess.check_output(
                        ["dpkg-deb", "--field", str(package), "Depends"],
                        text=True,
                    ).strip()
                    expected = "holoscan (>= 5.0.0)"
                    if enabled:
                        expected += ", libtinyxml2-10" if shared else ", libtinyxml2-dev"
                    self.assertEqual(depends, expected)

    def write_package_fixture(self, root):
        source = root / "source"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 4.0)\n"
            "project(holoscan_radar VERSION 0.2.0 LANGUAGES NONE)\n"
            'set(PROJECT_DESCRIPTION "Package layout fixture")\n'
            f'list(PREPEND CMAKE_MODULE_PATH "{ROOT}/cmake")\n'
            f'add_subdirectory("{ROOT}/pkg/holoscan-radar" pkg)\n'
            f'install(FILES "{ROOT}/pkg/holoscan-radar/README.md" DESTINATION share)\n'
        )
        return source

    def test_packaging_rejects_unsupported_processor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.write_package_fixture(root)
            result = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(root / "build"),
                    "-DCMAKE_SYSTEM_NAME=Linux",
                    "-DCMAKE_SYSTEM_PROCESSOR=ppc64le",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("supports only x86_64 and SBSA; got ppc64le", result.stderr)

    def test_archive_is_raw_install_tree_and_debian_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.write_package_fixture(root)
            build = root / "build"
            artifacts = root / "artifacts"
            for command in (
                ["cmake", "-S", str(source), "-B", str(build)],
                [
                    "cpack",
                    "--config",
                    str(build / "pkg/CPackConfig-holoscan-radar.cmake"),
                    "-G",
                    "DEB;TGZ",
                    "-B",
                    str(artifacts),
                ],
            ):
                subprocess.run(command, check=True, capture_output=True, text=True)
            arch = {"aarch64": "sbsa"}.get(platform.machine(), platform.machine())
            archive = artifacts / f"holoscan-radar-linux-{arch}-0.2.0.tar.gz"
            with tarfile.open(archive) as contents:
                names = [name.removeprefix("./") for name in contents.getnames()]
            self.assertIn("opt/nvidia/holoscan/share/README.md", names)
            self.assertTrue(all(name.split("/")[0] == "opt" for name in names), names)
            deb_arch = subprocess.check_output(["dpkg", "--print-architecture"], text=True)
            package = artifacts / f"holoscan-radar_0.2.0_{deb_arch.strip()}.deb"
            debian = subprocess.check_output(["dpkg-deb", "--contents", str(package)], text=True)
            self.assertIn("./opt/nvidia/holoscan/share/README.md", debian)

    def test_package_artifacts_carry_license_and_notice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 4.0)\n"
                "project(holoscan_radar VERSION 0.2.0 LANGUAGES NONE)\n"
                'set(PROJECT_DESCRIPTION "Package legal-file fixture")\n'
                f'list(PREPEND CMAKE_MODULE_PATH "{ROOT}/cmake")\n'
                f'add_subdirectory("{ROOT}/pkg/holoscan-radar" pkg)\n' + legal_file_fixture(source)
            )
            build = root / "build"
            artifacts = root / "artifacts"
            for command in (
                ["cmake", "-S", str(source), "-B", str(build)],
                [
                    "cpack",
                    "--config",
                    str(build / "pkg/CPackConfig-holoscan-radar.cmake"),
                    "-G",
                    "DEB;TGZ",
                    "-B",
                    str(artifacts),
                ],
            ):
                subprocess.run(command, check=True, capture_output=True, text=True)
            doc = "opt/nvidia/holoscan/share/doc/holoscan-radar"
            legal = {f"{doc}/LICENSE.txt": "LICENSE", f"{doc}/NOTICE": "NOTICE"}
            debian = {**legal, "usr/share/doc/holoscan-radar/copyright": "LICENSE"}
            unpacked = root / "deb"
            subprocess.run(
                ["dpkg-deb", "--extract", str(next(artifacts.glob("*.deb"))), str(unpacked)],
                check=True,
            )
            for path, origin in debian.items():
                with self.subTest(deb=path):
                    self.assertEqual((unpacked / path).read_bytes(), (ROOT / origin).read_bytes())
                    self.assertEqual((unpacked / path).stat().st_mode & 0o777, 0o644)
            with tarfile.open(next(artifacts.glob("*.tar.gz"))) as archive:
                names = {name.removeprefix("./") for name in archive.getnames()}
            for path in legal:
                with self.subTest(tgz=path):
                    self.assertIn(path, names)
            self.assertFalse(any(name.startswith("usr/") for name in names), names)

    def test_plain_install_carries_license_and_notice_except_for_wheels(self):
        for skbuild in (False, True):
            with self.subTest(skbuild=skbuild), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source"
                source.mkdir()
                (source / "CMakeLists.txt").write_text(
                    "cmake_minimum_required(VERSION 4.0)\n"
                    "project(holoscan_radar LANGUAGES NONE)\n" + legal_file_fixture(source)
                )
                build = root / "build"
                prefix = root / "prefix"
                configure = ["cmake", "-S", str(source), "-B", str(build)]
                if skbuild:
                    configure.append("-DSKBUILD=2")
                for command in (
                    configure,
                    ["cmake", "--install", str(build), "--prefix", str(prefix)],
                ):
                    subprocess.run(command, check=True, capture_output=True, text=True)
                doc = prefix / "share/doc/holoscan-radar"
                if skbuild:
                    self.assertFalse(prefix.exists() and any(prefix.rglob("*")))
                    continue
                for name, origin in (("LICENSE.txt", "LICENSE"), ("NOTICE", "NOTICE")):
                    self.assertEqual((doc / name).read_bytes(), (ROOT / origin).read_bytes())

    def test_relocated_config_dependencies_versions_and_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / "sdk"
            for name in ("holoscan", "matx", "tinyxml2"):
                directory = sdk / "lib/cmake" / name
                directory.mkdir(parents=True)
                (directory / f"{name}-config.cmake").write_text(f"set({name}_FOUND TRUE)\n")
                (directory / f"{name}-config-version.cmake").write_text(
                    "set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
                )
            modules = root / "modules"
            modules.mkdir()
            (modules / "FindCUDAToolkit.cmake").write_text("set(CUDAToolkit_FOUND TRUE)\n")
            (modules / "FindThreads.cmake").write_text("set(Threads_FOUND TRUE)\n")
            source = root / "source"
            source.mkdir()
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 4.0)\nproject(Fixture VERSION 0.2.0 LANGUAGES NONE)\n"
                "include(CMakePackageConfigHelpers)\n"
                "set(HOLOSCAN_RADAR_CONFIG_HAS_CPHD ON)\n"
                "set(BUILD_SHARED_LIBS OFF)\n"
                f'configure_package_config_file("{ROOT}/cmake/holoscan-radar-config.cmake.in" '
                '"${CMAKE_CURRENT_BINARY_DIR}/holoscan-radar-config.cmake" '
                "INSTALL_DESTINATION lib/cmake/holoscan-radar)\n"
                "write_basic_package_version_file("
                '"${CMAKE_CURRENT_BINARY_DIR}/holoscan-radar-config-version.cmake" '
                "VERSION 0.2.0 COMPATIBILITY AnyNewerVersion)\n"
                "add_library(sample INTERFACE)\n"
                "install(TARGETS sample EXPORT holoscan_radar_targets)\n"
                "install(EXPORT holoscan_radar_targets NAMESPACE holoscan:: "
                "DESTINATION lib/cmake/holoscan-radar)\n"
                'install(FILES "${CMAKE_CURRENT_BINARY_DIR}/holoscan-radar-config.cmake" '
                '"${CMAKE_CURRENT_BINARY_DIR}/holoscan-radar-config-version.cmake" '
                "DESTINATION lib/cmake/holoscan-radar)\n"
            )

            def run(*command, success=True):
                result = subprocess.run(command, text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)

            run("cmake", "-S", str(source), "-B", str(root / "build"))
            prefix = root / "prefix"
            run("cmake", "--install", str(root / "build"), "--prefix", str(prefix))
            relocated = root / "relocated"
            prefix.rename(relocated)
            consumer = root / "consumer"
            consumer.mkdir()
            (consumer / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 4.0)\nproject(Consumer LANGUAGES NONE)\n"
                'set(saved "${CMAKE_PREFIX_PATH}")\n'
                "find_package(holoscan-radar ${version} CONFIG REQUIRED ${components})\n"
                "find_package(holoscan-radar CONFIG REQUIRED)\n"
                "if(NOT TARGET holoscan::sample OR NOT CMAKE_PREFIX_PATH STREQUAL saved)\n"
                'message(FATAL_ERROR "Broken discovery or leaked search path")\nendif()\n'
            )
            for name, options, success in (
                ("normal", [], True),
                ("exact", ["-Dversion=0.2.0;EXACT"], True),
                ("future", ["-Dversion=999.0"], False),
                ("component", ["-Dcomponents=COMPONENTS;missing"], False),
            ):
                run(
                    "cmake",
                    "-S",
                    str(consumer),
                    "-B",
                    str(root / name),
                    f"-DCMAKE_PREFIX_PATH={relocated};{sdk}",
                    f"-DCMAKE_MODULE_PATH={modules}",
                    *options,
                    success=success,
                )
