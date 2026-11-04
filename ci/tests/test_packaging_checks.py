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


if __name__ == "__main__":
    unittest.main()
