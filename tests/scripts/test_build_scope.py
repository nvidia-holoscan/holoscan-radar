# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure-only regressions for application defaults and reused build trees."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "cmake/HoloscanRadarApplicationBuild.cmake"


class ApplicationBuildScopeTest(unittest.TestCase):
    def test_every_language_and_mode_selects_the_same_narrow_defaults(self):
        manifests = list((ROOT / "applications").glob("*/*/*/metadata.json"))
        self.assertEqual(len(manifests), 26)
        for path in manifests:
            app = json.loads(path.read_text())["application"]
            for name, mode in app["modes"].items():
                with self.subTest(application=app["name"], language=path.parent.name, mode=name):
                    options = mode["build"]["cmake_options"]
                    for option in (
                        "-DBUILD_ALL=OFF",
                        "-DHOLOSCAN_RADAR_BUILD_TESTING=OFF",
                        "-DPKG_holoscan_radar=OFF",
                        f"-DHOLOSCAN_RADAR_APPLICATION={app['name']}",
                    ):
                        self.assertIn(option, options)

    def configure(self, directory, *options):
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(directory),
                "-B",
                str(directory / "build"),
                "-G",
                "Ninja",
                *options,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def commands(self, directory, target="all"):
        return subprocess.check_output(
            ["ninja", "-C", str(directory / "build"), "-t", "commands", target], text=True
        )

    def fixture(self, directory):
        (directory / "apps").mkdir()
        (directory / "apps/CMakeLists.txt").write_text(
            "add_custom_target(required_op COMMAND ${CMAKE_COMMAND} -E echo REQUIRED)\n"
            "add_custom_target(spotlight ALL COMMAND ${CMAKE_COMMAND} -E echo SPOTLIGHT)\n"
            "add_dependencies(spotlight required_op)\n"
            "add_custom_target(test_mvdr ALL COMMAND ${CMAKE_COMMAND} -E echo UNRELATED_TEST)\n"
            "add_custom_target(other_app ALL COMMAND ${CMAKE_COMMAND} -E echo UNRELATED_APP)\n"
            "add_custom_target(provider_op COMMAND ${CMAKE_COMMAND} -E echo PROVIDER_OPERATOR)\n"
            "add_custom_target(holoscan_radar_python ALL COMMAND ${CMAKE_COMMAND} -E echo PYTHON)\n"
            "add_dependencies(holoscan_radar_python required_op provider_op)\n"
        )
        (directory / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\nproject(scope NONE)\n"
            'option(APP_other_app "Cached explicit selection" ON)\n'
            'option(OP_unused "Cached explicit operator" ON)\n'
            "add_subdirectory(apps)\n"
            f'include("{HELPER}")\nholoscan_radar_select_application_build()\n'
        )

    def test_fresh_application_build_selects_only_language_dependencies(self):
        for python_enabled in (False, True):
            with self.subTest(python=python_enabled), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.fixture(directory)
                self.configure(
                    directory,
                    "-DHOLOSCAN_RADAR_APPLICATION=spotlight",
                    "-DBUILD_ALL=OFF",
                    f"-DHOLOSCAN_RADAR_BUILD_PYTHON={'ON' if python_enabled else 'OFF'}",
                )
                commands = self.commands(directory)
                self.assertIn("REQUIRED", commands)
                self.assertNotIn("UNRELATED", commands)
                self.assertEqual("PYTHON" in commands, python_enabled)
                self.assertEqual("SPOTLIGHT" in commands, not python_enabled)
                self.assertEqual("PROVIDER_OPERATOR" in commands, python_enabled)

    def test_reused_broad_cache_switches_languages_without_building_unrelated_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            self.configure(directory)
            self.assertIn("UNRELATED_TEST", self.commands(directory))
            self.configure(directory, "-DHOLOSCAN_RADAR_APPLICATION=spotlight", "-DBUILD_ALL=OFF")
            cpp = self.commands(directory)
            self.assertIn("SPOTLIGHT", cpp)
            self.assertIn("REQUIRED", cpp)
            self.assertNotIn("UNRELATED", cpp)
            self.assertNotIn("PYTHON", cpp)
            cache = (directory / "build/CMakeCache.txt").read_text()
            self.assertIn("APP_other_app:BOOL=ON", cache)
            self.assertIn("OP_unused:BOOL=ON", cache)
            self.assertIn("UNRELATED_APP", self.commands(directory, "other_app"))
            self.configure(directory, "-DHOLOSCAN_RADAR_BUILD_PYTHON=ON")
            python = self.commands(directory)
            self.assertIn("PYTHON", python)
            self.assertIn("PROVIDER_OPERATOR", python)
            self.assertNotIn("SPOTLIGHT", python)
            self.assertNotIn("UNRELATED", python)
            self.configure(directory, "-DHOLOSCAN_RADAR_BUILD_PYTHON=OFF")
            self.assertEqual(self.commands(directory), cpp)

    def test_explicit_broad_workflows_keep_their_default_targets(self):
        for variable in (
            "BUILD_ALL",
            "HOLOSCAN_RADAR_BUILD_TESTING",
            "PKG_holoscan_radar",
            "MODULE_holoscan_radar",
            "SKBUILD",
        ):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.fixture(directory)
                self.configure(
                    directory, "-DHOLOSCAN_RADAR_APPLICATION=spotlight", f"-D{variable}=ON"
                )
                self.assertIn("UNRELATED_TEST", self.commands(directory))
                self.assertIn("UNRELATED_APP", self.commands(directory))


if __name__ == "__main__":
    unittest.main()
