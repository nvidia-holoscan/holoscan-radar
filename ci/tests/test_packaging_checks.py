# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only metadata tests for the artifact checker; artifact tests require the GPU SDK."""

import importlib.util
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
            (prefix / "lib/libholoscan_radar_selected_op.a").touch()
            (prefix / "lib/libholoscan_radar_other_op.a").touch()
            packaging.check_operator_libraries(prefix, {"selected_op"})

    def test_artifact_check_rejects_missing_curated_library(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "lib").mkdir()
            (prefix / "lib/libholoscan_radar_selected_op.a").touch()
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

    def archive_members(self, *entries):
        members = []
        for name, kind, link in entries:
            member = tarfile.TarInfo(name)
            member.type, member.linkname = kind, link
            members.append(member)
        return members

    def test_archive_allows_packaged_payload_under_debian_prefix(self):
        prefix = "opt/nvidia/holoscan"
        packaging.check_archive_members(
            self.archive_members(
                ("opt", tarfile.DIRTYPE, ""),
                (f"{prefix}/include/holoscan/radar/op.hpp", tarfile.REGTYPE, ""),
                (f"{prefix}/lib/libselected_op.a", tarfile.REGTYPE, ""),
                (f"{prefix}/lib/libshared_op.so.1", tarfile.REGTYPE, ""),
                (f"{prefix}/lib/libshared_op.so", tarfile.SYMTYPE, "libshared_op.so.1"),
                (
                    f"{prefix}/lib/cmake/holoscan-radar/holoscan-radar-config.cmake",
                    tarfile.REGTYPE,
                    "",
                ),
                (f"{prefix}/share/doc/holoscan-radar/LICENSE.txt", tarfile.REGTYPE, ""),
                (f"{prefix}/share/doc/holoscan-radar/NOTICE", tarfile.REGTYPE, ""),
            )
        )

    def test_archive_rejects_unpackaged_or_escaping_members(self):
        prefix = "opt/nvidia/holoscan"
        for name, kind, link, message in (
            (f"{prefix}/bin/siso", tarfile.REGTYPE, "", "Unexpected"),
            (f"{prefix}/bin", tarfile.DIRTYPE, "", "Unexpected"),
            (f"{prefix}/examples/siso.yaml", tarfile.REGTYPE, "", "Unexpected"),
            (f"{prefix}/holoscan_radar/__init__.py", tarfile.REGTYPE, "", "Unexpected"),
            ("holoscan-radar_0.2.0/include/op.hpp", tarfile.REGTYPE, "", "Unexpected"),
            (f"/{prefix}/include/op.hpp", tarfile.REGTYPE, "", "Unsafe"),
            (f"{prefix}/../include/op.hpp", tarfile.REGTYPE, "", "Unsafe"),
            (f"{prefix}/lib/libop.so", tarfile.SYMTYPE, "/usr/lib/libop.so", "Unsafe"),
            (f"{prefix}/lib/libop.so", tarfile.SYMTYPE, "../../libop.so", "Unsafe"),
        ):
            with self.subTest(name=name, link=link):
                with self.assertRaisesRegex(RuntimeError, message):
                    packaging.check_archive_members(self.archive_members((name, kind, link)))

    def test_library_rpaths_must_be_origin_relative(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "lib").mkdir()
            (prefix / "lib/libop.so.1").touch()
            dynamic = " 0x1d (RUNPATH)  Library runpath: [{}]\n"
            for runpath in ("$ORIGIN", "$ORIGIN/../lib", "${ORIGIN}/lib:$ORIGIN"):
                with (
                    self.subTest(runpath=runpath),
                    mock.patch.object(
                        packaging.subprocess, "check_output", return_value=dynamic.format(runpath)
                    ),
                ):
                    packaging.check_library_rpaths(prefix)
            for runpath in ("$ORIGIN:/opt/nvidia/holoscan/lib", "$ORIGIN_SUFFIX", "lib"):
                with (
                    self.subTest(runpath=runpath),
                    mock.patch.object(
                        packaging.subprocess, "check_output", return_value=dynamic.format(runpath)
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "non-relative RPATH"):
                        packaging.check_library_rpaths(prefix)


if __name__ == "__main__":
    unittest.main()
