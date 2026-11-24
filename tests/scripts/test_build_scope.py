# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configure-only regressions for application defaults and reused build trees."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "cmake/HoloHubConfigHelpers.cmake"


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
                    ):
                        self.assertIn(option, options)
                    self.assertFalse(any("HOLOSCAN_RADAR_APPLICATION" in x for x in options))

    def test_root_does_not_apply_a_second_component_selection_policy(self):
        root_cmake = (ROOT / "CMakeLists.txt").read_text()
        self.assertNotIn("HoloscanRadarApplicationBuild", root_cmake)
        self.assertNotIn("holoscan_radar_select_application_build", root_cmake)
        self.assertNotIn("HOLOSCAN_RADAR_APPLICATION", root_cmake)

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
        for name in ("siso", "simo", "required_op", "extra_op"):
            (directory / name).mkdir()
            if name in ("siso", "simo"):
                body = (
                    "if(NOT HOLOHUB_BUILD_PYTHON)\n"
                    f"  add_custom_target({name} ALL COMMAND ${{CMAKE_COMMAND}} -E echo {name})\n"
                    f"  add_dependencies({name} required_op)\nendif()\n"
                )
            else:
                body = f"add_custom_target({name} ALL COMMAND ${{CMAKE_COMMAND}} -E echo {name})\n"
            (directory / name / "CMakeLists.txt").write_text(body)
        (directory / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\nproject(selection NONE)\n"
            f'include("{HELPER}")\n'
            "add_holohub_application(siso DEPENDS OPERATORS required_op)\n"
            "add_holohub_application(simo DEPENDS OPERATORS required_op)\n"
            "add_holohub_operator(required_op)\nadd_holohub_operator(extra_op)\n"
            "if(HOLOHUB_BUILD_PYTHON)\n"
            "  add_custom_target(holoscan_radar_python ALL COMMAND ${CMAKE_COMMAND} -E echo PROVIDER)\n"
            "  add_dependencies(holoscan_radar_python required_op)\nendif()\n"
        )

    def test_multiple_applications_and_independent_operator_build_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            self.configure(
                directory, "-DBUILD_ALL=OFF", "-DAPP_siso=ON", "-DAPP_simo=ON", "-DOP_extra_op=ON"
            )
            commands = self.commands(directory)
            for target in ("siso", "simo", "required_op", "extra_op"):
                self.assertIn(f"echo {target}", commands)

    def test_cached_selections_remain_enabled_until_explicitly_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            self.configure(directory, "-DBUILD_ALL=ON")
            self.configure(directory, "-DBUILD_ALL=OFF", "-DAPP_siso=ON")
            self.assertIn("echo simo", self.commands(directory))
            self.assertIn("echo extra_op", self.commands(directory))
            self.configure(directory, "-DAPP_simo=OFF", "-DOP_extra_op=OFF", "-DOP_required_op=OFF")
            commands = self.commands(directory)
            self.assertNotIn("echo simo", commands)
            self.assertNotIn("echo extra_op", commands)
            self.assertIn("echo required_op", commands)

    def test_python_without_cpp_target_and_language_switches(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            self.configure(
                directory, "-DBUILD_ALL=OFF", "-DAPP_siso=ON", "-DHOLOSCAN_RADAR_APPLICATION=siso"
            )
            cpp = self.commands(directory)
            self.assertIn("echo siso", cpp)
            self.configure(directory, "-DHOLOHUB_BUILD_PYTHON=ON")
            commands = self.commands(directory)
            self.assertNotIn("echo siso", commands)
            self.assertIn("echo PROVIDER", commands)
            self.assertIn("echo required_op", commands)
            self.configure(directory, "-DHOLOHUB_BUILD_PYTHON=OFF")
            self.assertEqual(cpp, self.commands(directory))


if __name__ == "__main__":
    unittest.main()
