# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security policy for the credential-bearing Blossom scanner job."""

import unittest
from pathlib import Path

import yaml


class BlossomTrustTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / ".github/workflows/blossom-ci.yml"
        self.jobs = yaml.safe_load(path.read_text())["jobs"]
        self.scan = self.jobs["Vulnerability-scan"]

    def test_source_repository_gate_precedes_all_credential_bearing_steps(self):
        # Require the guard at job level: guarding checkout alone still runs
        # the credential-bearing scanner when checkout is skipped.
        self.assertEqual(self.scan["needs"], ["Authorization"])
        self.assertEqual(
            self.scan["if"],
            "${{ fromJson(needs.Authorization.outputs.args).repo == github.repository }}",
        )
        checkout = self.scan["steps"][0]["with"]
        self.assertEqual(checkout["repository"], "${{ github.repository }}")
        self.assertEqual(checkout["persist-credentials"], False)

    def test_skipped_scan_cannot_trigger_the_internal_build(self):
        trigger = self.jobs["Job-trigger"]
        self.assertEqual(trigger["needs"], ["Vulnerability-scan"])
        self.assertNotIn("if", trigger)  # GitHub's default success() skips dependencies.

    def test_scanner_token_cannot_write_source_or_actions(self):
        # An explicit permissions map sets all omitted permissions to none.
        self.assertEqual(self.scan["permissions"], {"contents": "read", "statuses": "write"})


if __name__ == "__main__":
    unittest.main()
