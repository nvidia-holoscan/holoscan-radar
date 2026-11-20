# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled metadata pins: fail before parsing/publication, preserve arbitrary custom files."""

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unittest import mock

import numpy as np
import pull_cphd
from sar_data import integrity, subset
from test_subset_cphd import make_cphd, range_server


def pin_for(data):
    def region(offset, size):
        return {
            "offset": offset,
            "size": size,
            "sha256": hashlib.sha256(data[offset : offset + size]).hexdigest(),
        }

    return {"size_bytes": len(data), "header": region(0, 8), "xml": region(16, 13)}


class MetadataIntegrityTest(unittest.TestCase):
    data = b"header!!padding!<CPHD></CPHD>sample bytes"

    def test_every_bundled_dataset_has_a_complete_bounded_metadata_pin(self):
        catalog = pull_cphd.dataset_catalog()
        self.assertEqual(len(catalog), 10)
        for dataset in catalog.values():
            with self.subTest(scene=dataset.scene):
                pin = integrity.metadata_pin(f"s3://{dataset.bucket}/{dataset.key}")
                self.assertIsNotNone(pin)
                self.assertEqual(
                    pin,
                    integrity.metadata_pin(pull_cphd.https_url(dataset.https_prefix, dataset.key)),
                )
                for name in ("header", "xml"):
                    region = pin[name]
                    self.assertGreater(region["size"], 0)
                    self.assertGreaterEqual(region["offset"], 0)
                    self.assertLessEqual(region["offset"] + region["size"], pin["size_bytes"])
                    self.assertRegex(region["sha256"], r"^[0-9a-f]{64}$")

    def test_only_pinned_metadata_ranges_are_read_in_bounded_chunks(self):
        pin = pin_for(self.data)
        pin["size_bytes"] = 1 << 50  # A huge source must not imply a huge read.
        reads = []

        def read(offset, size):
            reads.append((offset, size))
            return self.data[offset : offset + size]

        with mock.patch.object(integrity, "_HASH_CHUNK_BYTES", 4):
            integrity.verify_metadata(pin, 1 << 50, read)
        self.assertEqual(reads, [(0, 4), (4, 4), (16, 4), (20, 4), (24, 4), (28, 1)])

    def test_header_xml_size_and_short_reads_are_rejected(self):
        for offset in (0, 17):
            changed = bytearray(self.data)
            changed[offset] ^= 1
            with (
                self.subTest(offset=offset),
                self.assertRaisesRegex(RuntimeError, "pinned metadata"),
            ):
                integrity.verify_metadata(
                    pin_for(self.data),
                    len(changed),
                    lambda start, size: changed[start : start + size],
                )
        read = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "source size"):
            integrity.verify_metadata(pin_for(self.data), len(self.data) + 1, read)
        read.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "short bundled"):
            integrity.verify_metadata(pin_for(self.data), len(self.data), lambda *_args: b"")

    def test_sample_changes_are_outside_the_declared_metadata_only_scope(self):
        changed = self.data[:-1] + b"X"
        integrity.verify_metadata(
            pin_for(self.data), len(changed), lambda start, size: changed[start : start + size]
        )

    def test_local_or_custom_sources_do_not_require_catalog_hashes(self):
        for source in (
            Path("2023-06-24-13-22-56_UMBRA-05_CPHD.cphd"),
            "https://custom.example/valid.cphd",
            pull_cphd.UMBRA_HTTPS + "arbitrary-valid-custom-file.cphd",
        ):
            with self.subTest(source=source):
                self.assertIsNone(integrity.metadata_pin(source))
                # No file is opened for sources outside the pinned example set.
                integrity.verify_metadata_file(Path("not-needed.cphd"), source)

    def test_bad_remote_metadata_fails_before_the_parser_and_preserves_outputs(self):
        changed = b"changed!" + self.data[8:]
        source = mock.Mock()
        source.size.return_value = len(changed)
        source.read.side_effect = lambda start, size: changed[start : start + size]
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "out.cphd"
            dest.write_bytes(b"original")
            checkpoint = Path(directory) / "out.cphd.part.json"
            checkpoint.write_bytes(b"resume state")
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            with (
                mock.patch.object(subset, "open_source", return_value=source),
                mock.patch.object(subset, "metadata_pin", return_value=pin_for(self.data)),
                mock.patch.object(subset, "_read_header") as parse,
                self.assertRaisesRegex(RuntimeError, "pinned metadata"),
            ):
                subset.subset_cphd("https://custom.example/input.cphd", dest, force=True)
            parse.assert_not_called()
            source.close.assert_called_once()
            self.assertEqual({p.name: p.read_bytes() for p in Path(directory).iterdir()}, before)

    def test_https_verifies_before_publish_and_preserves_an_existing_destination(self):
        for data in (self.data, b"changed!" + self.data[8:]):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                dest = Path(directory) / "out.cphd"
                dest.write_bytes(b"keep existing")
                response = io.BytesIO(data)
                response.headers = {"Content-Length": str(len(data))}
                with (
                    mock.patch.object(pull_cphd, "open_source_url", return_value=response),
                    mock.patch.object(integrity, "metadata_pin", return_value=pin_for(self.data)),
                    contextlib.redirect_stderr(io.StringIO()),
                ):

                    def action():
                        pull_cphd.download_https(
                            pull_cphd.UMBRA_HTTPS + "test",
                            dest,
                            pull_cphd.DownloadProgress("test", len(data)),
                        )

                    if data == self.data:
                        action()
                        self.assertEqual(dest.read_bytes(), data)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "pinned metadata"):
                            action()
                        self.assertEqual(dest.read_bytes(), b"keep existing")
                self.assertFalse(pull_cphd.partial_path(dest).exists())

    def test_aws_verifies_before_publish(self):
        for data in (self.data, b"changed!" + self.data[8:]):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                dest = Path(directory) / "out.cphd"
                dest.write_bytes(b"keep existing")

                def launch(*_args, **_kwargs):
                    pull_cphd.partial_path(dest).write_bytes(data)
                    process = mock.MagicMock()
                    process.__enter__.return_value = process
                    process.stdout = []
                    process.wait.return_value = 0
                    return process

                with (
                    mock.patch.object(pull_cphd.subprocess, "Popen", side_effect=launch),
                    mock.patch.object(integrity, "metadata_pin", return_value=pin_for(self.data)),
                    contextlib.redirect_stderr(io.StringIO()),
                ):

                    def action():
                        pull_cphd.download_aws(
                            "bucket", "key", dest, pull_cphd.DownloadProgress("test", len(data))
                        )

                    if data == self.data:
                        action()
                        self.assertEqual(dest.read_bytes(), data)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "pinned metadata"):
                            action()
                        self.assertEqual(dest.read_bytes(), b"keep existing")
                self.assertFalse(pull_cphd.partial_path(dest).exists())

    def test_pinned_remote_metadata_is_rechecked_while_signal_download_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, dest = Path(directory) / "source.cphd", Path(directory) / "out.cphd"
            values = make_cphd(path, None)
            data = path.read_bytes()
            local = subset.FileByteSource(path)
            try:
                header = subset._read_header(local)
            finally:
                local.close()
            header_size = data.index(b"\x0c\n") + 2
            xml_offset, xml_size = int(header["XML_BLOCK_BYTE_OFFSET"]), int(
                header["XML_BLOCK_SIZE"]
            )
            pin = {
                "size_bytes": len(data),
                "header": {
                    "offset": 0,
                    "size": header_size,
                    "sha256": hashlib.sha256(data[:header_size]).hexdigest(),
                },
                "xml": {
                    "offset": xml_offset,
                    "size": xml_size,
                    "sha256": hashlib.sha256(data[xml_offset : xml_offset + xml_size]).hexdigest(),
                },
            }
            with range_server(data) as (remote, state):
                origin = f"http://127.0.0.1:{remote._port}"
                kwargs = {
                    "samples": (2, 6),
                    "connections": 1,
                    "io_chunk_bytes": 64,
                    "allowed_source_origins": (origin,),
                }
                with (
                    mock.patch.object(subset, "metadata_pin", return_value=pin),
                    mock.patch.object(subset, "_FETCH_RETRIES", 1),
                ):
                    state.fail_from = 8192 + 4 * 64
                    with self.assertRaisesRegex(RuntimeError, "after 1 tries"):
                        subset.subset_cphd(origin + "/source.cphd", dest, **kwargs)
                    state.fail_from = None
                    state.requests.clear()
                    subset.subset_cphd(origin + "/source.cphd", dest, **kwargs)
                reads = [(start, end) for start, end, _etag in state.requests]
                self.assertIn((0, header_size - 1), reads)
                self.assertIn((xml_offset, xml_offset + xml_size - 1), reads)
                self.assertEqual(
                    sorted(start for start, _end in reads if start >= 8192),
                    [8192 + i * 64 + 16 for i in (4, 5)],
                )
            with dest.open("rb") as handle, subset.skcphd.Reader(handle) as reader:
                np.testing.assert_array_equal(reader.read_signal("Primary"), values[:, 2:6])

    def test_existing_pinned_file_is_checked_before_parse_or_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            job = pull_cphd.PullJob(
                "test", "key", "bucket", pull_cphd.UMBRA_HTTPS, Path(directory), False
            )
            job.destination.write_bytes(b"changed!" + self.data[8:])
            with (
                mock.patch.object(integrity, "metadata_pin", return_value=pin_for(self.data)),
                mock.patch.object(pull_cphd, "cphd_readable") as parse,
                mock.patch.object(pull_cphd, "download_https") as download,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "pinned metadata"),
            ):
                pull_cphd.pull_one(job, False, 1, 1)
            parse.assert_not_called()
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
