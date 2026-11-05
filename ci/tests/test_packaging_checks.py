# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only metadata tests for the artifact checker; artifact tests require the GPU SDK."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[2] / "tests/packaging/check_packages.py"
SPEC = importlib.util.spec_from_file_location("check_packages", CHECKER)
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


class PackagingMetadata(unittest.TestCase):
    def test_cache_entries_survive_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "# CMake cache\n\n//Operator selected by package\nOP_sample_op:BOOL=ON\n"
                "\n//Explicit preference\nHOLOSCAN_RADAR_BUILD_TESTING:UNINITIALIZED=OFF\n"
                "CMAKE_CUDA_ARCHITECTURES:STRING=80;90\n"
            )
            self.assertEqual(
                packaging.cache_values(build),
                {
                    "OP_sample_op": "ON",
                    "HOLOSCAN_RADAR_BUILD_TESTING": "OFF",
                    "CMAKE_CUDA_ARCHITECTURES": "80;90",
                },
            )

    def test_artifact_check_allows_contents_outside_curated_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "lib").mkdir()
            (prefix / "lib/libselected_op.a").touch()
            (prefix / "lib/libother_op.a").touch()
            packaging.check_operator_libraries(prefix, {"selected_op"})

    def test_artifact_check_rejects_missing_curated_library(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "lib").mkdir()
            (prefix / "lib/libselected_op.a").touch()
            with self.assertRaisesRegex(RuntimeError, "missing operator library: typo_op"):
                packaging.check_operator_libraries(prefix, {"selected_op", "typo_op"})

    def package_tree(self, prefix):
        directory = prefix / "lib/cmake/holoscan-radar"
        directory.mkdir(parents=True)
        (directory / "holoscan-radar-config.cmake").touch()
        (directory / "holoscan-radar-config-version.cmake").touch()
        (directory / "holoscan_radar_targets.cmake").write_text("holoscan::selected_op")
        return directory

    def test_combined_package_requires_version_and_selected_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            directory = self.package_tree(prefix)
            packaging.check_cmake_package(prefix, {"selected_op"})
            with self.assertRaisesRegex(RuntimeError, "Missing exported target"):
                packaging.check_cmake_package(prefix, {"missing_op"})
            (directory / "holoscan-radar-config-version.cmake").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing Radar version"):
                packaging.check_cmake_package(prefix, {"selected_op"})

    def test_combined_package_rejects_legacy_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            directory = self.package_tree(prefix)
            (directory / "holoscan_selected_op_targets.cmake").touch()
            with self.assertRaisesRegex(RuntimeError, "Unexpected Radar exports"):
                packaging.check_cmake_package(prefix, {"selected_op"})

    def test_sdk_prefix_supports_hsdk_ga_and_apt_layouts(self):
        for sdk_dir, prefix in (
            ("/opt/nvidia/holoscan/lib/cmake/holoscan", "/opt/nvidia/holoscan"),
            ("/usr/lib/cmake/holoscan", "/usr"),
            ("/usr/lib64/cmake/holoscan", "/usr"),
            ("/usr/lib/x86_64-linux-gnu/cmake/holoscan", "/usr"),
            ("/usr/lib/aarch64-linux-gnu/cmake/holoscan", "/usr"),
        ):
            with self.subTest(sdk_dir=sdk_dir):
                self.assertEqual(packaging.sdk_prefix(Path(sdk_dir)), Path(prefix))

    def test_sdk_prefix_rejects_unknown_layouts(self):
        for sdk_dir in ("/usr/share/holoscan", "/opt/holoscan/cmake/holoscan"):
            with self.subTest(sdk_dir=sdk_dir), self.assertRaises(RuntimeError):
                packaging.sdk_prefix(Path(sdk_dir))

    def test_sdk_python_site_finds_layout_or_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            self.assertIsNone(packaging.sdk_python_site(prefix))
            site = prefix / "lib/python3/dist-packages"
            (site / "holoscan").mkdir(parents=True)
            self.assertEqual(packaging.sdk_python_site(prefix), site)

    def test_consumer_matx_requires_sdk_copy_when_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            build = Path(temporary)
            sdk_matx = build / "sdk/lib/cmake/matx"
            sdk_matx.mkdir(parents=True)
            cache = build / "CMakeCache.txt"
            cache.write_text(f"matx_DIR:PATH={sdk_matx}\n")
            packaging.check_consumer_matx(build, sdk_matx)
            packaging.check_consumer_matx(build, None)
            cache.write_text("matx_DIR:PATH=/other/lib/cmake/matx\n")
            with self.assertRaisesRegex(RuntimeError, "not the SDK"):
                packaging.check_consumer_matx(build, sdk_matx)
            cache.write_text("matx_DIR:PATH=matx_DIR-NOTFOUND\n")
            with self.assertRaisesRegex(RuntimeError, "did not resolve MatX"):
                packaging.check_consumer_matx(build, None)


if __name__ == "__main__":
    unittest.main()
