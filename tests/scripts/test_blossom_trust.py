# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authorization, checkout, and build dependencies for the Blossom scanner."""

import unittest
from pathlib import Path

import yaml


class BlossomTrustTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / ".github/workflows/blossom-ci.yml"
        self.jobs = yaml.safe_load(path.read_text())["jobs"]
        self.scan = self.jobs["Vulnerability-scan"]

    def test_checkout_uses_authorized_source_without_persisting_credentials(self):
        self.assertEqual(self.scan["needs"], ["Authorization"])
        checkout = self.scan["steps"][0]["with"]
        self.assertEqual(
            checkout["repository"], "${{ fromJson(needs.Authorization.outputs.args).repo }}"
        )
        self.assertEqual(checkout["ref"], "${{ fromJson(needs.Authorization.outputs.args).ref }}")
        self.assertEqual(checkout["persist-credentials"], False)

    def test_skipped_scan_cannot_trigger_the_internal_build(self):
        trigger = self.jobs["Job-trigger"]
        self.assertEqual(trigger["needs"], ["Vulnerability-scan"])
        self.assertNotIn("if", trigger)  # GitHub's default success() skips dependencies.


if __name__ == "__main__":
    unittest.main()
