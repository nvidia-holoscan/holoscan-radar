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

    def test_inventory_uses_registrations_instead_of_package_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            operators = source / "operators/example"
            operators.mkdir(parents=True)
            (operators / "CMakeLists.txt").write_text(
                "# holoscan_radar_add_operator(commented_op)\n"
                "holoscan_radar_add_operator(first_op)\n"
                "holoscan_radar_add_operator(second_op\n DIRECTORY elsewhere)\n"
            )
            self.assertEqual(packaging.registered_operators(source), {"first_op", "second_op"})


if __name__ == "__main__":
    unittest.main()
