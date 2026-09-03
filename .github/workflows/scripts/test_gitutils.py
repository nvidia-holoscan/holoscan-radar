"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""

import unittest
from unittest.mock import patch

import gitutils


class GitUtilsTest(unittest.TestCase):
    def test_git_uses_argument_vector_without_a_shell(self):
        """Git ref text must remain an argument, never shell syntax."""
        ref = "main; touch should-not-run"
        with patch("gitutils.subprocess.check_output", return_value=b"output\n") as check_output:
            self.assertEqual(getattr(gitutils, "__git")("diff", ref), "output")

        check_output.assert_called_once_with(["git", "diff", ref])


if __name__ == "__main__":
    unittest.main()
