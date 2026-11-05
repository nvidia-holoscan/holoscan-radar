# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the CTest driver on disposable metadata fixtures, not radar operators.

Run with `python -m unittest discover -s ci/tests -v`. Requires cmake, ctest,
and Ninja on PATH; no GPU suite is replaced or skipped by these checks.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[1] / "container.ctest"
CTEST = os.environ.get("RADAR_CTEST_BIN", "ctest")


class ContainerCTestContract(unittest.TestCase):
    def run_dashboard(self, extra_args=(), *, empty=False, fail_configure=False, fail_test=False):
        with tempfile.TemporaryDirectory(prefix="radar-ctest-contract-") as directory:
            root = Path(directory)
            (root / "ci").mkdir()
            shutil.copy2(DRIVER, root / "ci/container.ctest")
            cmake = [
                "cmake_minimum_required(VERSION 3.24)",
                "project(RadarCIFixture LANGUAGES NONE)",
                "enable_testing()",
                'option(HOLOSCAN_RADAR_TEST_PACKAGING "Artifact coverage fixture" OFF)',
            ]
            if fail_configure:
                cmake.append('message(FATAL_ERROR "fixture configure failure")')
            elif not empty:
                for name in ("unit", "statistical", "integration", "performance"):
                    # A failing performance test proves exclusion works while
                    # the statistical and integration tiers remain covered.
                    command = "false" if name == "performance" or fail_test else "true"
                    cmake.extend(
                        [
                            f'add_test(NAME {name} COMMAND "${{CMAKE_COMMAND}}" -E {command})',
                            f"set_tests_properties({name} PROPERTIES LABELS {name})",
                        ]
                    )
                cmake.extend(
                    [
                        "if(HOLOSCAN_RADAR_TEST_PACKAGING)",
                        'add_test(NAME packaging COMMAND "${CMAKE_COMMAND}" -E true)',
                        "set_tests_properties(packaging PROPERTIES LABELS packaging)",
                        "endif()",
                    ]
                )
            (root / "CMakeLists.txt").write_text("\n".join(cmake) + "\n")
            result = subprocess.run(
                [CTEST, "-S", "ci/container.ctest", "-VV", "-DCTEST_SITE=CI-Fixture", *extra_args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            reports = {
                path.name: path.read_text()
                for path in (root / "build-test/Testing").glob("*/*.xml")
            }
            junit = root / "build-test/test-results.xml"
            names = []
            if junit.exists():
                names = [test.attrib["name"] for test in ET.parse(junit).iter("testcase")]
            return result, reports, names

    def test_explicit_dashboard_name_and_complete_nonperformance_selection(self):
        result, reports, names = self.run_dashboard(
            ("-DRADAR_CDASH_BUILD_NAME:STRING=radar-linux-main-5x", "-DCTEST_BUILD_FLAGS=-j2")
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(set(names), {"unit", "statistical", "integration", "packaging"})
        for report in reports.values():
            self.assertEqual(ET.fromstring(report).attrib["BuildName"], "radar-linux-main-5x")
        self.assertIn("CTest coverage: 4 tests", result.stdout)
        self.assertIn("-j2", result.stdout)

    def test_packaging_coverage_can_be_disabled_explicitly(self):
        result, _, names = self.run_dashboard(
            ("-DCONFIGURE_OPTIONS=-DHOLOSCAN_RADAR_TEST_PACKAGING:BOOL=OFF",)
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(set(names), {"unit", "statistical", "integration"})

    def test_local_default_name(self):
        result, reports, _ = self.run_dashboard()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(reports)
        for report in reports.values():
            self.assertEqual(
                ET.fromstring(report).attrib["BuildName"], "holoscan-radar-linux--latest"
            )

    def test_empty_suite_fails(self):
        result, _, names = self.run_dashboard(empty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(names, [])
        self.assertIn("zero tests", result.stdout + result.stderr)

    def test_configure_failure_is_preserved(self):
        result, _, names = self.run_dashboard(fail_configure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(names, [])
        self.assertIn("Configure failed with return value", result.stdout + result.stderr)
        self.assertNotIn("CTest selected zero tests", result.stdout + result.stderr)

    def test_test_failure_fails_dashboard(self):
        result, _, names = self.run_dashboard(fail_test=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(set(names), {"unit", "statistical", "integration", "packaging"})
        self.assertIn("Tests failed with return value", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
