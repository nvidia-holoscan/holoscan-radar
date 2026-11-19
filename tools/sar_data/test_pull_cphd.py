# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline downloader checks: python3 -m unittest discover -s tools/sar_data."""

import contextlib
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import pull_cphd


class DownloadCliTest(unittest.TestCase):
    def test_catalog_rejects_unsafe_recipe_names(self):
        for name in ("../escape", "/tmp/escape", "nested\\name", "", None):
            with (
                self.subTest(name=name),
                mock.patch.object(
                    pull_cphd.yaml,
                    "safe_load",
                    return_value={"name": name, "source": {"bucket": "unused", "key": "unused"}},
                ),
                self.assertRaisesRegex(ValueError, "single filename component"),
            ):
                pull_cphd.dataset_catalog()

    def test_autofocus_uses_original_catalog_object_and_preset_path(self):
        output = io.StringIO()
        with (
            mock.patch("sys.argv", ["pull_cphd.py", "--app", "autofocus", "--list"]),
            mock.patch.object(pull_cphd, "pull_one") as pull,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(pull_cphd.main(), 0)
        pull.assert_not_called()
        self.assertIn("sar-data/tasks/Kourou%2C%20French%20Guiana/", output.getvalue())
        self.assertTrue(
            output.getvalue()
            .rstrip()
            .endswith(" -> data/sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd")
        )

    def test_all_custom_root_keeps_shared_collection_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "sar"
            jobs = pull_cphd.jobs_for("all", root)
            destinations = [job.dest_dir / Path(job.key).name for job in jobs]
            self.assertEqual(len(destinations), 8)
            self.assertEqual(len(set(destinations)), 8)
            self.assertIn(root / "2023-06-24-13-22-56_UMBRA-05_CPHD.cphd", destinations)
            self.assertTrue(all(p.parent == root for p in destinations))
            self.assertFalse(root.exists())

    def test_all_accepts_arbitrary_destination_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "collections"
            output = io.StringIO()
            with (
                mock.patch(
                    "sys.argv",
                    ["pull_cphd.py", "--app", "all", "--dest", str(destination), "--list"],
                ),
                mock.patch.object(pull_cphd, "prepare_one") as prepare,
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(pull_cphd.main(), 0)
            paths = [Path(line.split(" -> ")[1]) for line in output.getvalue().splitlines()]
            self.assertEqual(len(paths), 8)
            self.assertTrue(all(path.parent == destination for path in paths))
            prepare.assert_not_called()
            self.assertFalse(destination.exists())

    def test_every_shipped_config_has_an_exact_offline_plan(self):
        configs = sorted((pull_cphd.REPO_ROOT / "applications/sar").glob("*/config/*.yaml"))
        self.assertEqual(len(configs), 12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            for config in configs:
                with self.subTest(config=config.name):
                    spec = pull_cphd.yaml.safe_load(config.read_text())["cphd"]["path"]
                    inputs = [spec] if isinstance(spec, str) else spec
                    jobs = pull_cphd.jobs_for_config(config, root)
                    self.assertEqual({job.destination for job in jobs}, {root / p for p in inputs})
                    for job in jobs:
                        self.assertEqual(job.destination.parent, root / "sar")
                        if "houston" in config.name or "video_sar" in config.name:
                            self.assertEqual(job.recipe.name, "houston_1m.yaml")
                            self.assertEqual(job.destination.name, "houston_1p0m.cphd")
                        else:
                            self.assertIsNone(job.recipe)
                            self.assertEqual(Path(job.key).name, job.destination.name)
            self.assertFalse(root.exists())

    def test_invalid_config_never_downloads_or_creates_directories(self):
        known = "sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
        for paths in (
            [],
            None,
            42,
            [known, "sar/unknown.cphd"],
            [known, "wrong/" + Path(known).name],
            ["sar/collections/" + Path(known).name],
            [known, ""],
            [known, 42],
        ):
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as temp:
                config = Path(temp) / "config.yaml"
                config.write_text(pull_cphd.yaml.safe_dump({"cphd": {"path": paths}}))
                root = Path(temp) / "absent"
                with (
                    mock.patch(
                        "sys.argv",
                        ["pull_cphd.py", "--config", str(config), "--data-root", str(root)],
                    ),
                    mock.patch.object(pull_cphd, "prepare_one") as prepare,
                    mock.patch.object(Path, "mkdir") as mkdir,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as caught,
                ):
                    pull_cphd.main()
                self.assertEqual(caught.exception.code, 2)
                prepare.assert_not_called()
                mkdir.assert_not_called()
                self.assertFalse(root.exists())

    def test_multiple_inputs_deduplicate_and_honor_data_root(self):
        known = "sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yaml"
            config.write_text(pull_cphd.yaml.safe_dump({"cphd": {"path": [known, known]}}))
            root = Path(temp) / "data"
            with mock.patch.dict("os.environ", {"HOLOHUB_DATA_DIR": str(root)}):
                jobs = pull_cphd.jobs_for_config(config)
            self.assertEqual([job.destination for job in jobs], [root / known])
            self.assertFalse(root.exists())

    def test_manifest_with_wrong_output_is_rejected(self):
        catalog = pull_cphd.dataset_catalog()
        key = Path("sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd")
        catalog[key] = pull_cphd.replace(catalog[key], key="wrong.cphd")
        with (
            mock.patch.object(pull_cphd, "dataset_catalog", return_value=catalog),
            self.assertRaisesRegex(ValueError, "does not produce"),
        ):
            pull_cphd.jobs_for_config(pull_cphd.APPS["autofocus"])

    def test_malformed_config_is_reported_without_a_traceback(self):
        for text in ("cphd: null", "cphd: []", "cphd: [", "[]", ""):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temp:
                config = Path(temp) / "config.yaml"
                config.write_text(text)
                output = io.StringIO()
                with (
                    mock.patch("sys.argv", ["pull_cphd.py", "--config", str(config)]),
                    contextlib.redirect_stderr(output),
                    self.assertRaises(SystemExit) as caught,
                ):
                    pull_cphd.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertNotIn("Traceback", output.getvalue())

    def test_absolute_path_must_match_manifest_destination(self):
        known = "sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "data"
            config = Path(temp) / "config.yaml"
            config.write_text(pull_cphd.yaml.safe_dump({"cphd": {"path": str(root / known)}}))
            self.assertEqual(pull_cphd.jobs_for_config(config, root)[0].destination, root / known)
            with self.assertRaisesRegex(ValueError, "no manifest entry"):
                pull_cphd.jobs_for_config(config, root / "different")

    def test_existing_yaml_relative_file_takes_precedence(self):
        known = "sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yaml"
            config.write_text(pull_cphd.yaml.safe_dump({"cphd": {"path": known}}))
            shadow = config.parent / known
            shadow.parent.mkdir(parents=True)
            shadow.write_bytes(b"local input")
            with self.assertRaisesRegex(ValueError, "no manifest entry"):
                pull_cphd.jobs_for_config(config, Path(temp) / "data")

    def test_houston_list_is_offline_and_preparation_uses_recipe(self):
        output = io.StringIO()
        config = pull_cphd.APPS["video_sar"]
        with (
            mock.patch("sys.argv", ["pull_cphd.py", "--config", str(config), "--list"]),
            mock.patch.object(pull_cphd.subprocess, "run") as run,
            mock.patch.object(pull_cphd.urllib.request, "urlopen") as urlopen,
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(pull_cphd.main(), 0)
        run.assert_not_called()
        urlopen.assert_not_called()
        self.assertIn("houston_1m.yaml", output.getvalue())
        self.assertIn("houston_1p0m.cphd", output.getvalue())
        job = pull_cphd.jobs_for_config(config)[0]
        with (
            mock.patch.object(pull_cphd.shutil, "which", return_value="/usr/bin/uv"),
            mock.patch.object(pull_cphd.subprocess, "run") as run,
        ):
            self.assertEqual(pull_cphd.prepare_one(job, True, 1, 1), job.destination)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["uv", "run"])
        self.assertIn(str(job.recipe), command)
        self.assertIn(str(job.destination), command)
        self.assertIn("--force", command)

    def test_subset_transfer_size_is_forwarded(self):
        job = pull_cphd.jobs_for_config(pull_cphd.APPS["video_sar"])[0]
        with mock.patch.object(pull_cphd.subprocess, "run") as run:
            pull_cphd.prepare_one(job, False, 1, 1, io_chunk_bytes=16777216)
        command = run.call_args.args[0]
        position = command.index("--io-chunk-bytes")
        self.assertEqual(command[position + 1], "16777216")

    def test_duplicate_config_keys_fail_before_selection_or_side_effects(self):
        known = "sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
        for text in (
            f"cphd:\n  path: {known}\n  path: {known}\n",
            f"cphd: {{path: {known}}}\ncphd: {{path: {known}}}\n",
            f"cphd: {{<<: {{path: {known}, path: {known}}}}}\n",
            f"cphd: {{<<: [{{path: {known}, path: {known}}}]}}\n",
        ):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as temp:
                config = Path(temp) / "config.yaml"
                config.write_text(text)
                root = Path(temp) / "absent"
                output = io.StringIO()
                with (
                    mock.patch(
                        "sys.argv",
                        ["pull_cphd.py", "--config", str(config), "--data-root", str(root)],
                    ),
                    mock.patch.object(pull_cphd, "dataset_catalog") as catalog,
                    mock.patch.object(pull_cphd, "prepare_one") as prepare,
                    mock.patch.object(Path, "mkdir") as mkdir,
                    contextlib.redirect_stderr(output),
                    self.assertRaises(SystemExit) as caught,
                ):
                    pull_cphd.main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("duplicate key", output.getvalue())
                self.assertNotIn("Traceback", output.getvalue())
                catalog.assert_not_called()
                prepare.assert_not_called()
                mkdir.assert_not_called()
                self.assertFalse(root.exists())

    def test_yaml_merge_overrides_and_reused_anchors_remain_supported(self):
        known = "sar/2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yaml"
            config.write_text(
                "defaults: &defaults {path: unused.cphd}\n"
                f"cphd: &cphd {{<<: *defaults, path: {known}}}\n"
                "copy: {<<: *cphd}\n"
            )
            root = Path(temp) / "data"
            jobs = pull_cphd.jobs_for_config(config, root)
            self.assertEqual([job.destination for job in jobs], [root / known])
            self.assertFalse(root.exists())

    def test_houston_preparation_uses_current_python_when_uv_is_unavailable(self):
        job = pull_cphd.jobs_for_config(pull_cphd.APPS["video_sar"])[0]
        with (
            mock.patch.object(pull_cphd.shutil, "which", return_value=None),
            mock.patch.object(pull_cphd.subprocess, "run") as run,
        ):
            self.assertEqual(pull_cphd.prepare_one(job, False, 1, 1), job.destination)
        run.assert_called_once_with(
            [
                pull_cphd.sys.executable,
                str(pull_cphd.REPO_ROOT / "tools/sar_data/subset_cphd.py"),
                str(job.recipe),
                "--output",
                str(job.destination),
                "--progress-label",
                "[1/1] Houston 1 m subset",
            ],
            check=True,
        )

    def test_https_length_mismatch_preserves_destination(self):
        for expected in (0, 2, 8):
            for existing in (None, b"original collect"):
                with self.subTest(expected=expected, existing=existing):
                    with tempfile.TemporaryDirectory() as temp:
                        dest = Path(temp) / "collect.cphd"
                        if existing is not None:
                            dest.write_bytes(existing)
                        response = io.BytesIO(b"data")
                        response.headers = {"Content-Length": str(expected)}
                        # A prior HEAD length must not override the GET length.
                        progress = pull_cphd.DownloadProgress("test", 4)
                        with (
                            mock.patch.object(
                                pull_cphd.urllib.request, "urlopen", return_value=response
                            ),
                            contextlib.redirect_stderr(io.StringIO()),
                            self.assertRaisesRegex(RuntimeError, "download size mismatch"),
                        ):
                            pull_cphd.download_https("https://example.test/collect", dest, progress)
                        if existing is None:
                            self.assertFalse(dest.exists())
                        else:
                            self.assertEqual(dest.read_bytes(), existing)
                        self.assertFalse(pull_cphd.partial_path(dest).exists())

    def test_https_complete_response_publishes_destination(self):
        for headers in ({"Content-Length": "4"}, {}):
            with self.subTest(headers=headers):
                with tempfile.TemporaryDirectory() as temp:
                    dest = Path(temp) / "collect.cphd"
                    dest.write_bytes(b"original collect")
                    response = io.BytesIO(b"data")
                    response.headers = headers
                    with (
                        mock.patch.object(
                            pull_cphd.urllib.request, "urlopen", return_value=response
                        ),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        pull_cphd.download_https(
                            "https://example.test/collect",
                            dest,
                            pull_cphd.DownloadProgress("test", None),
                        )
                    self.assertEqual(dest.read_bytes(), b"data")
                    self.assertFalse(pull_cphd.partial_path(dest).exists())

    def test_http_404_reports_source_and_cleans_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"
            output = io.StringIO()
            with (
                mock.patch("sys.argv", ["pull_cphd.py", "--app", "autofocus", "--dest", temp]),
                mock.patch.object(pull_cphd.shutil, "which", return_value=None),
                mock.patch.object(pull_cphd, "https_size", return_value=None),
                mock.patch.object(
                    pull_cphd.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.HTTPError(
                        "https://example.test/missing", 404, "Not Found", {}, None
                    ),
                ),
                contextlib.redirect_stderr(output),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(pull_cphd.main(), 1)
            self.assertIn("HTTP Error 404", output.getvalue())
            self.assertIn("Source: https://umbra-open-data-catalog.", output.getvalue())
            self.assertIn(f"Destination: {dest}", output.getvalue())
            self.assertNotIn("Traceback", output.getvalue())
            self.assertFalse(dest.exists())
            self.assertFalse(pull_cphd.partial_path(dest).exists())


if __name__ == "__main__":
    unittest.main()
