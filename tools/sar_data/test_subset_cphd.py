# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline subset transport/checkpoint checks; requires requirements.txt."""

import contextlib
import io
import json
import multiprocessing
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import lxml.etree
import numpy as np
import sarkit.cphd as skcphd
from sar_data import subset


@contextlib.contextmanager
def range_server(payload):
    state = SimpleNamespace(
        data=payload,
        statuses=[],
        requests=[],
        etag='"original"',
        wrong_range=False,
        short=False,
        fail_from=None,
        retry_after="7",
    )
    guard = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_GET(self):
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            with guard:
                state.requests.append((start, end, self.headers.get("If-Match")))
                status = state.statuses.pop(0) if state.statuses else 206
                if state.fail_from is not None and start >= state.fail_from:
                    status = 500
                if self.headers.get("If-Match", state.etag) != state.etag:
                    status = 412
            self.send_response(status)
            if status != 206:
                self.send_header("Content-Length", "0")
                self.send_header("Retry-After", state.retry_after)
                self.end_headers()
                return
            self.send_header(
                "Content-Range", f"bytes {start + state.wrong_range}-{end}/{len(state.data)}"
            )
            if state.etag is not None:
                self.send_header("ETag", state.etag)
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            data = state.data[start : end + 1]
            if state.short:
                state.short = False
                data = data[:-1]
                self.close_connection = True
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    source = subset.HttpByteSource(f"http://127.0.0.1:{server.server_port}/source.cphd")
    try:
        yield source, state
    finally:
        source.close()
        server.shutdown()
        server.server_close()
        thread.join()


class HttpRangeTest(unittest.TestCase):
    def test_transient_statuses_retry_the_same_range_and_honor_retry_after(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status), range_server(b"abcdefgh") as (source, state):
                state.statuses = [status]
                with mock.patch.object(subset.time, "sleep") as sleep:
                    self.assertEqual(source.read(2, 4), b"cdef")
                self.assertEqual([r[:2] for r in state.requests], [(2, 5), (2, 5)])
                sleep.assert_called_once_with(7.0)
                self.assertEqual(len(source._conns), 1)

    def test_retry_exhaustion_is_bounded_and_does_not_sleep_after_last_attempt(self):
        with range_server(b"abcdefgh") as (source, state):
            state.statuses = [500] * 5
            with mock.patch.object(subset.time, "sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "after 5 tries"):
                    source.read(0, 4)
            self.assertEqual(len(state.requests), 5)
            self.assertEqual(sleep.call_count, 4)

    def test_permanent_errors_and_ignored_or_incorrect_ranges_fail_immediately(self):
        for status in (200, 403, 404, 412, 416):
            with self.subTest(status=status), range_server(b"abcdefgh") as (source, state):
                state.statuses = [status]
                with mock.patch.object(subset.time, "sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        source.read(0, 4)
                self.assertEqual(len(state.requests), 1)
                sleep.assert_not_called()
        with range_server(b"abcdefgh") as (source, state):
            state.wrong_range = True
            with self.assertRaisesRegex(RuntimeError, "Content-Range"):
                source.read(0, 4)

    def test_short_body_reconnects_and_source_changes_are_not_mixed(self):
        with range_server(b"abcdefgh") as (source, state):
            state.short = True
            with mock.patch.object(subset.time, "sleep"):
                self.assertEqual(source.read(0, 4), b"abcd")
            self.assertEqual(len(state.requests), 2)
            state.etag = '"changed"'
            with self.assertRaisesRegex(RuntimeError, "HTTP 412"):
                source.read(4, 4)
            self.assertEqual(state.requests[-1][2], '"original"')


class ResumeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "source.bin"
        self.values = np.asarray(np.arange(80).reshape(10, 8) + 0.5j, dtype=">c8")
        self.source.write_bytes(self.values.tobytes())
        self.dest = self.root / "subset.cphd"
        self.part = self.dest.with_suffix(".cphd.part")
        self.journal = self.dest.with_suffix(".cphd.part.json")
        self.prefix = b"test CPHD header and metadata\0"
        patch = mock.patch.object(subset, "_SIGNAL_CHUNK_BYTES", 2 * 4 * 8)
        patch.start()
        self.addCleanup(patch.stop)

    def download(self, *, fail=False, force=False, source=None, **options):
        source = source or subset.FileByteSource(self.source)
        original = source.read
        reads = []

        def read(offset, size):
            reads.append((offset, size))
            if fail and offset >= 4 * 8 * 8:
                raise RuntimeError("injected interruption")
            return original(offset, size)

        try:
            with tempfile.TemporaryFile() as prefix, mock.patch.object(source, "read", read):
                prefix.write(self.prefix)
                prefix.flush()
                subset._download_signal(
                    self.dest,
                    self.source,
                    source,
                    prefix,
                    len(self.prefix),
                    pulses=(0, 10),
                    samples=(2, 6),
                    force=force,
                    signal_block=0,
                    channel_offset=0,
                    orig_num_samples=8,
                    pulse_start=0,
                    num_vectors=10,
                    sample_start=2,
                    num_samples=4,
                    connections=2,
                    progress=False,
                    **options,
                )
        finally:
            source.close()
        return reads

    def interrupt(self, *, force=False):
        with self.assertRaisesRegex(RuntimeError, "injected interruption"):
            self.download(fail=True, force=force)
        self.assertEqual(json.loads(self.journal.read_text())["written_vectors"], 4)
        self.assertTrue(self.part.exists())

    def test_resume_only_fetches_remaining_rows_and_discards_uncheckpointed_tail(self):
        self.interrupt()
        self.assertFalse(self.dest.exists())
        with self.part.open("ab") as handle:
            handle.write(b"an incomplete chunk")
        reads = self.download()
        self.assertEqual([offset for offset, _size in reads], [i * 64 + 16 for i in range(4, 10)])
        self.assertEqual(self.dest.read_bytes(), self.prefix + self.values[:, 2:6].tobytes())
        self.assertFalse(self.journal.exists())
        self.assertFalse(self.part.exists())
        self.assertEqual(
            json.loads(self.dest.with_suffix(".subset.json").read_text())["samples"], [2, 6]
        )

    def test_failed_force_keeps_previous_complete_output_and_next_call_resumes(self):
        self.dest.write_bytes(b"previous good product")
        self.interrupt(force=True)
        self.assertEqual(self.dest.read_bytes(), b"previous good product")
        self.assertEqual(len(self.download()), 6)

    def test_changed_source_is_refused_and_force_starts_again(self):
        self.interrupt()
        self.source.write_bytes((self.values + 1).astype(">c8").tobytes())
        before = self.part.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "source/window changed"):
            self.download()
        self.assertEqual(self.part.read_bytes(), before)
        self.assertEqual(len(self.download(force=True)), 10)

    def test_invalid_checkpoints_or_corrupt_partials_are_not_silently_reused(self):
        for damage in ("json", "count", "request", "truncate", "prefix", "tail", "missing"):
            with self.subTest(damage=damage):
                self.interrupt(force=True)
                state = json.loads(self.journal.read_text())
                if damage == "json":
                    self.journal.write_text("{")
                elif damage in ("count", "request"):
                    if damage == "count":
                        state["written_vectors"] = True
                    else:
                        state["request"]["samples"] = [0, 4]
                    self.journal.write_text(json.dumps(state))
                elif damage == "missing":
                    self.journal.unlink()
                else:
                    with self.part.open("r+b") as handle:
                        if damage == "truncate":
                            handle.truncate(len(self.prefix))
                        else:
                            handle.seek(
                                0 if damage == "prefix" else -1, 0 if damage == "prefix" else 2
                            )
                            handle.write(b"X")
                with self.assertRaisesRegex(RuntimeError, "--force"):
                    self.download()

    def test_crash_after_publication_recovers_without_fetching_signal_again(self):
        with mock.patch.object(subset, "_write_sidecar", side_effect=OSError("publication crash")):
            with self.assertRaisesRegex(OSError, "publication crash"):
                self.download()
        self.assertTrue(self.dest.exists())
        self.assertFalse(self.part.exists())
        self.assertEqual(self.download(), [])
        self.assertFalse(self.journal.exists())

    def test_second_writer_cannot_discard_an_active_download(self):
        self.interrupt()
        before = self.part.read_bytes()
        with subset._destination_lock(self.dest):
            with self.assertRaisesRegex(RuntimeError, "another download"):
                self.download(force=True)
        self.assertEqual(self.part.read_bytes(), before)

    def test_terminated_process_resumes_without_relying_on_exception_cleanup(self):
        context = multiprocessing.get_context("fork")
        ready = context.Event()
        wait = context.Event()
        original = subset._write_json_atomic

        def child():
            def checkpoint(path, state):
                original(path, state)
                if state.get("written_vectors") == 4:
                    ready.set()
                    wait.wait()

            with mock.patch.object(subset, "_write_json_atomic", checkpoint):
                self.download()

        process = context.Process(target=child)
        process.start()
        try:
            self.assertTrue(ready.wait(timeout=10), "Child did not reach its checkpoint")
        finally:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        self.assertLess(process.exitcode, 0)
        self.assertEqual(len(self.download()), 6)

    def test_changed_http_object_or_missing_validator_cannot_resume(self):
        for changed in (True, False):
            with (
                self.subTest(changed=changed),
                range_server(self.source.read_bytes()) as (source, state),
            ):
                source.read(0, 1)
                with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                    self.download(source=source, fail=True, force=True)
                # A new process learns the remote object's current identity.
                with range_server(self.source.read_bytes()) as (replacement, new_state):
                    new_state.etag = '"changed"' if changed else None
                    replacement.read(0, 1)
                    with self.assertRaisesRegex(RuntimeError, "--force"):
                        self.download(source=replacement)

    def test_http_workers_and_connections_remain_bounded_across_chunks(self):
        with range_server(self.source.read_bytes()) as (source, state):
            source.read(0, 1)  # Establish the same identity normally read with CPHD metadata.
            with mock.patch.object(source, "close"):
                self.download(source=source)
                self.assertLessEqual(len(source._conns), 3)
            self.assertEqual(len(state.requests), 11)


def make_cphd(path, namespace):
    fields = "".join(
        f"<{name}><Offset>{i}</Offset><Size>1</Size><Format>F8</Format></{name}>"
        for i, name in enumerate(("TxTime", "SC0", "SCSS"))
    )
    xml = lxml.etree.fromstring(
        (
            "<CPHD><CollectionID><Classification>UNCLASSIFIED</Classification>"
            "<ReleaseInfo>UNRESTRICTED</ReleaseInfo></CollectionID>"
            "<Data><SignalArrayFormat>CF8</SignalArrayFormat><NumBytesPVP>32</NumBytesPVP>"
            "<Channel><Identifier>Primary</Identifier><NumVectors>6</NumVectors><NumSamples>8</NumSamples>"
            "<SignalArrayByteOffset>0</SignalArrayByteOffset><PVPArrayByteOffset>0</PVPArrayByteOffset>"
            "</Channel></Data><Channel><Parameters><FxC>0</FxC><FxBW>0</FxBW></Parameters></Channel>"
            f"<PVP>{fields}</PVP></CPHD>"
        ).encode()
    )
    if namespace:
        for element in xml.iter():
            element.tag = f"{{{namespace}}}{element.tag}"
    encoded = lxml.etree.tostring(xml)
    pvps = np.zeros(
        6,
        dtype={
            "names": ["TxTime", "SC0", "SCSS"],
            "formats": [">f8"] * 3,
            "offsets": [0, 8, 16],
            "itemsize": 32,
        },
    )
    pvps.view("u1").reshape(6, 32)[:, 24:] = 0xAB
    pvps["TxTime"], pvps["SC0"], pvps["SCSS"] = np.arange(6), 9e9, 1e6
    values = np.asarray(np.arange(48).reshape(6, 8) + 0.25j, dtype=">c8")
    header = (
        f"CPHD/1.1.0\nXML_BLOCK_BYTE_OFFSET := 1024\nXML_BLOCK_SIZE := {len(encoded)}\n"
        "PVP_BLOCK_BYTE_OFFSET := 4096\nPVP_BLOCK_SIZE := 192\n"
        "SIGNAL_BLOCK_BYTE_OFFSET := 8192\nSIGNAL_BLOCK_SIZE := 384\n\f\n"
    ).encode()
    with path.open("wb") as handle:
        handle.write(header)
        handle.seek(1024)
        handle.write(encoded)
        handle.seek(4096)
        handle.write(pvps.tobytes())
        handle.seek(8192)
        handle.write(values.tobytes())
        handle.truncate(65536)  # Also supports the normal HTTP header probe.
    return values


class SubsetFileTest(unittest.TestCase):
    def test_real_writer_resumes_with_stable_padded_pvp_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "input.cphd", Path(directory) / "out.cphd"
            values = make_cphd(source, "http://api.nsgreg.nga.mil/schema/cphd/1.1.0")
            original = subset.FileByteSource.read

            def fail(reader, offset, size):
                if offset >= 8192 + 4 * 64:
                    raise RuntimeError("interrupted")
                return original(reader, offset, size)

            with (
                mock.patch.object(subset, "_validate_xmltree"),
                mock.patch.object(subset, "_SIGNAL_CHUNK_BYTES", 64),
            ):
                with mock.patch.object(subset.FileByteSource, "read", fail):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        subset.subset_cphd(source, dest, samples=(2, 6))
                subset.subset_cphd(source, dest, samples=(2, 6))
            with dest.open("rb") as handle, skcphd.Reader(handle) as reader:
                np.testing.assert_array_equal(reader.read_signal("Primary"), values[:, 2:6])

    def test_real_writer_and_minimal_fixture_outputs_preserve_samples_and_shift_pvps(self):
        for namespace in (None, "http://api.nsgreg.nga.mil/schema/cphd/1.1.0"):
            with self.subTest(namespace=namespace), tempfile.TemporaryDirectory() as directory:
                source, dest = Path(directory) / "input.cphd", Path(directory) / "out.cphd"
                values = make_cphd(source, namespace)
                # Tiny metadata tests the real sarkit binary writer/reader;
                # it deliberately omits schema-required scene/geometry fields.
                with mock.patch.object(subset, "_validate_xmltree"):
                    subset.subset_cphd(source, dest, samples=(2, 6))
                with dest.open("rb") as handle, skcphd.Reader(handle) as reader:
                    np.testing.assert_array_equal(reader.read_signal("Primary"), values[:, 2:6])
                    np.testing.assert_array_equal(reader.read_pvps("Primary")["SC0"], 9.002e9)
                self.assertTrue(subset.subset_cphd(source, dest, samples=(2, 6)).skipped)

    def test_http_failure_and_rerun_resume_through_the_public_subset_entry_point(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source.cphd", Path(directory) / "out.cphd"
            values = make_cphd(source, None)
            with range_server(source.read_bytes()) as (http_source, state):
                url = f"http://127.0.0.1:{http_source._port}/source.cphd"
                state.fail_from = 8192 + 4 * 64
                with (
                    mock.patch.object(subset, "_SIGNAL_CHUNK_BYTES", 64),
                    mock.patch.object(subset, "_FETCH_RETRIES", 1),
                ):
                    with self.assertRaisesRegex(RuntimeError, "after 1 tries"):
                        subset.subset_cphd(url, dest, samples=(2, 6), connections=2)
                    state.fail_from = None
                    state.requests.clear()
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        subset.subset_cphd(url, dest, samples=(2, 6), progress=True, connections=2)
                self.assertIn("4/6 vectors checkpointed", output.getvalue())
                signal_reads = [start for start, _end, _etag in state.requests if start >= 8192]
                self.assertEqual(sorted(signal_reads), [8192 + i * 64 + 16 for i in (4, 5)])
            with dest.open("rb") as handle, skcphd.Reader(handle) as reader:
                np.testing.assert_array_equal(reader.read_signal("Primary"), values[:, 2:6])


if __name__ == "__main__":
    unittest.main()
