# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for bounded copyright lint inputs."""

import datetime
import importlib.util
import pathlib
import sys
import tempfile
import unittest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2] / ".github/workflows/scripts/check_copyright.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("check_copyright", SCRIPT)
check_copyright = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_copyright)


class CopyrightInputTest(unittest.TestCase):
    def test_excludes_are_globs_not_regular_expressions(self):
        self.assertTrue(check_copyright.matches_exclude("/repo/build/generated.py", "build"))
        self.assertFalse(check_copyright.matches_exclude("/repo/" + "a" * 4096 + ".py", "(.+)+Z"))

    def test_header_scan_is_bounded(self):
        with tempfile.NamedTemporaryFile("wb") as source:
            source.write(b"# repeated comment text\n" * 10_000)
            source.flush()
            lines = check_copyright.read_header_lines(source.name)

        self.assertLessEqual(len(lines), check_copyright.MAX_HEADER_LINES)
        self.assertLessEqual(sum(map(len, lines)), check_copyright.MAX_HEADER_BYTES)
        self.assertIsNone(check_copyright.get_copyright_years(lines)[0])

    def test_current_header_is_accepted(self):
        year = datetime.datetime.now().year
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as source:
            source.write(
                "# SPDX-FileCopyrightText: Copyright (c) "
                f"{year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
            )
            source.flush()
            self.assertEqual(check_copyright.check_copyright(source.name, False), [])

    def test_year_update_preserves_crlf_line_endings(self):
        year = datetime.datetime.now().year
        with tempfile.NamedTemporaryFile("w+b") as source:
            source.write(
                b"# SPDX-FileCopyrightText: Copyright (c) "
                + str(year - 1).encode()
                + b" NVIDIA CORPORATION & AFFILIATES\r\n"
                + b"# Keep this line unchanged.\r\n"
            )
            source.flush()
            self.assertEqual(check_copyright.check_copyright(source.name, True), [])
            source.seek(0)
            self.assertEqual(
                source.read(),
                b"# SPDX-FileCopyrightText: Copyright (c) "
                + str(year - 1).encode()
                + b"-"
                + str(year).encode()
                + b" NVIDIA CORPORATION & AFFILIATES\r\n"
                + b"# Keep this line unchanged.\r\n",
            )


if __name__ == "__main__":
    unittest.main()
