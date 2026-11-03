# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer protocol identity/replay; synthetic host reports are not GPU evidence."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_benchmark as fixtures
from artifacts import read_json, write_json
from radar_perf import config_hash, load_session, main, normalize_config, timing_comparison
from test_optimization import config


class ConsumerSamplingTest(unittest.TestCase):
    def settings(self):
        value = config()
        value.update(
            measurement_policy_version=3,
            pairs=2,
            consumer_policies={"consumer": copy.deepcopy(value["policies"])},
            consumer_sampling={"consumer": {"cases": "baseline", "repeats": 1}},
        )
        value["sampling"]["cases"] = "primary-case"
        return normalize_config(value)

    def test_distinct_matrices_resolve_without_primary_case_inheritance(self):
        value = self.settings()
        self.assertEqual(value["sampling"]["cases"], "primary-case")
        self.assertEqual(value["consumer_sampling"]["consumer"]["cases"], "baseline")
        self.assertEqual(value["consumer_sampling"]["consumer"]["repeats"], 1)
        self.assertEqual(value["consumer_sampling"]["consumer"]["early_frames"], 50)

    def test_real_raw_replay_uses_consumer_protocol_and_rejects_mutation(self):
        fixture = fixtures.BenchmarkToolsTest(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        settings = self.settings()
        index = fixture.index(sampling=settings["consumer_sampling"]["consumer"])
        index["before"]["operator"] = "consumer"
        index["after"]["operator"] = "consumer"
        result = timing_comparison(settings, index, None, calibrate=True)
        self.assertTrue(result["insufficient_pair_budget"])
        self.assertFalse(result["passed"] and not result["insufficient_pair_budget"])
        self.assertTrue(all(row["case"][1] == "baseline" for row in result["results"]))
        changed = copy.deepcopy(settings)
        changed["consumer_sampling"]["consumer"]["frames"] = 100
        with self.assertRaisesRegex(ValueError, "sampling differs"):
            timing_comparison(changed, index, None, calibrate=True)
        # The primary must not accidentally use this consumer's protocol.
        index["before"]["operator"] = settings["operator"]
        index["after"]["operator"] = settings["operator"]
        with self.assertRaisesRegex(ValueError, "sampling differs"):
            timing_comparison(settings, index, None, calibrate=True)

    def test_undeclared_consumers_and_ambiguous_cases_rejected(self):
        original = self.settings()
        for field in (
            {"unknown": {"cases": "c"}},
            {"consumer": {"cases": "a,a"}},
            {"consumer": {"cases": ""}},
            {"consumer": {"cases": "a, b"}},
            {"consumer": {"cases": "a", "frames": 49}},
            {"consumer": []},
        ):
            value = copy.deepcopy(original)
            value["consumer_sampling"] = field
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_config(value)

    def test_measure_command_collects_the_declared_consumer_cases(self):
        fixture = fixtures.BenchmarkToolsTest(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        settings = self.settings()
        build = fixture.build()
        build["operator"] = "consumer"
        build_path = fixture.root / "build.json"
        config_path = fixture.root / "config.json"
        write_json(build_path, build)
        write_json(config_path, settings)
        output = fixture.root / "measured"
        with patch(
            "measurement.clock_state", return_value={"configuration": {}, "observations": {}}
        ):
            self.assertEqual(
                main(
                    [
                        "measure",
                        "--before",
                        str(build_path),
                        "--after",
                        str(build_path),
                        "--config",
                        str(config_path),
                        "--out",
                        str(output),
                    ]
                ),
                0,
            )
        index = read_json(output / "pairs.json")
        self.assertEqual(index["sampling"], settings["consumer_sampling"]["consumer"])
        result = timing_comparison(settings, index, None, calibrate=True)
        self.assertTrue(all(row["case"][1] == "baseline" for row in result["results"]))

    def test_legacy_config_has_no_implicit_consumer_protocol_upgrade(self):
        value = config()
        self.assertEqual(value["measurement_policy_version"], 2)
        self.assertEqual(value.get("consumer_sampling", {}), {})
        value["consumer_policies"] = {"consumer": copy.deepcopy(value["policies"])}
        value["consumer_sampling"] = {"consumer": {"cases": "c"}}
        with self.assertRaisesRegex(ValueError, "freshly confirmed version-3"):
            normalize_config(value)

    def test_absent_optional_map_preserves_historical_normalized_hashes(self):
        # Adding an unused optional field must not change a frozen CLI context.
        for version in (2, 3):
            value = config()
            value["measurement_policy_version"] = version
            historical = normalize_config(value)
            historical.pop("consumer_sampling", None)
            resolved = normalize_config(historical)
            self.assertNotIn("consumer_sampling", resolved)
            self.assertEqual(config_hash(resolved), config_hash(historical))

    def test_confirmed_hash_binds_consumer_protocol(self):
        value = self.settings()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            write_json(path, {"config": value, "confirmed_config_sha256": config_hash(value)})
            self.assertEqual(load_session(path)["config"], value)
            saved = read_json(path)
            saved["config"]["consumer_sampling"]["consumer"]["cases"] = "different"
            write_json(path, saved)
            with self.assertRaisesRegex(ValueError, "obtain new confirmation"):
                load_session(path)


if __name__ == "__main__":
    unittest.main()
