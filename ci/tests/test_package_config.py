# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed CMake discovery metadata; no operator compute is mocked."""

import platform
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
            for enabled in (False, True, False):
                with self.subTest(cphd=enabled):
                    subprocess.run(
                        [
                            "cmake",
                            "-S",
                            str(source),
                            "-B",
                            str(build),
                            f"-DBUILD_CPHD={'ON' if enabled else 'OFF'}",
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
                        expected += ", libtinyxml2-dev"
                    self.assertEqual(depends, expected)

    def test_archive_is_relocatable_prefix_and_debian_keeps_opt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
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
            archive = artifacts / f"holoscan-radar_0.2.0_{platform.machine()}.tar.gz"
            with tarfile.open(archive) as contents:
                names = contents.getnames()
            prefix = archive.name.removesuffix(".tar.gz")
            self.assertIn(f"{prefix}/share/README.md", names)
            self.assertTrue(all(name.split("/")[0] == prefix for name in names), names)
            debian = subprocess.check_output(
                ["dpkg-deb", "--contents", str(next(artifacts.glob("*.deb")))], text=True
            )
            self.assertIn("./opt/nvidia/holoscan/share/README.md", debian)

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
            competing = root / "competing/lib/cmake/matx"
            competing.mkdir(parents=True)
            (competing / "matx-config.cmake").write_text(
                'message(FATAL_ERROR "MatX selected from competing SDK")\n'
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
                    f"-DCMAKE_PREFIX_PATH={relocated};{root / 'competing'};{sdk}",
                    f"-DCMAKE_MODULE_PATH={modules}",
                    *options,
                    success=success,
                )
