# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subset a CPHD: pulse window and/or a contiguous sample (frequency) subband.

The video-SAR demo cannot stream Houston's 110 GiB signal block, so this
writes a local file the app can read. Two jobs, matching INPUT_DATA.md:

- a half-open pulse window (omit for the whole dwell)
- a contiguous central sample slice, which for FX-domain CPHD is a frequency
  subband. Centre frequency and bandwidth come from the *kept* sample grid
  (`SC0 + (i0 + (n-1)/2)*SCSS`, `n*SCSS`), not from the source `FxBand`.

A local path and an HTTPS URL share one byte-range reader. The Houston pull
is request-rate bound: 128 keep-alive connections is the measured optimum.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import http.client
import json
import logging
import os
import random
import re
import shutil
import ssl
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import BinaryIO, Protocol

import lxml.etree
import numpy as np
import sarkit.cphd as skcphd
import yaml
from sar_data.progress import DownloadProgress

# CF8 only: Houston and the CI smoke fixture are CF8, and a subband of CI4
# is a later demo's problem (Capella / tiled strip).
_CF8_BYTES = 8
_HEADER_PROBE = 64 * 1024
_DEFAULT_CONNECTIONS = 128
_HTTPS_TIMEOUT_S = 60.0
_FETCH_RETRIES = 5
_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# Host RAM on the demo box is 62 GiB; a 0.5 m Houston subband is ~52 GiB.
# Stream that many bytes through the writer rather than materialising them.
_SIGNAL_CHUNK_BYTES = 64 * 1024 * 1024


class ByteSource(Protocol):
    def read(self, offset: int, size: int) -> bytes: ...

    def close(self) -> None: ...

    def identity(self) -> dict: ...


@dataclass(frozen=True)
class SubsetResult:
    path: Path
    num_vectors: int
    num_samples: int
    skipped: bool
    dry_run: bool = False
    pulse_range: tuple[int, int] = (0, 0)
    sample_range: tuple[int, int] = (0, 0)


_ORIGINS = frozenset({"start", "center", "end"})


@dataclass(frozen=True)
class AxisWindow:
    """How to pick a pulse or sample window. Resolved after the header is read.

    Give exactly one of: explicit ``[start, stop)``, ``count``, ``bandwidth_hz``
    (samples), or ``duration_s`` (pulses). ``origin`` places a sized window
    in the source axis; it is ignored for an explicit span.
    """

    start: int | None = None
    stop: int | None = None
    count: int | None = None
    origin: str = "center"
    bandwidth_hz: float | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class SubsetRecipe:
    name: str
    source: str
    pulses: AxisWindow | None
    samples: AxisWindow | None
    connections: int = _DEFAULT_CONNECTIONS


class FileByteSource:
    """Per-thread file handles so parallel range reads do not share a seek."""

    def __init__(self, path: Path):
        self._path = path
        self._local = threading.local()
        self._handles: list[object] = []
        self._lock = threading.Lock()
        self._identity = self.identity()

    def identity(self) -> dict:
        stat = self._path.stat()
        identity = {
            "path": str(self._path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
        if hasattr(self, "_identity") and identity != self._identity:
            raise RuntimeError("source file changed while preparing the subset")
        return identity

    def read(self, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        handle = getattr(self._local, "handle", None)
        if handle is None:
            handle = self._path.open("rb")
            self._local.handle = handle
            with self._lock:
                self._handles.append(handle)
        handle.seek(offset)
        data = handle.read(size)
        if len(data) != size:
            raise RuntimeError(f"{self._path}: short read at {offset}: {len(data)} of {size} bytes")
        return data

    def close(self) -> None:
        with self._lock:
            for handle in self._handles:
                handle.close()
            self._handles.clear()


class HttpByteSource:
    """Keep-alive HTTP(S) Range GETs, one connection per calling thread."""

    def __init__(self, url: str, timeout: float = _HTTPS_TIMEOUT_S):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme: {url}")
        if not parsed.hostname:
            raise ValueError(f"URL has no host: {url}")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._path = parsed.path or "/"
        if parsed.query:
            self._path += "?" + parsed.query
        self._timeout = timeout
        self._local = threading.local()
        self._conns: list[http.client.HTTPConnection] = []
        self._lock = threading.Lock()
        self._identity: dict | None = None

    def identity(self) -> dict:
        with self._lock:
            return dict(self._identity or {})

    def _validate_response(self, response, offset: int, last: int) -> None:
        span = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.getheader("Content-Range", ""))
        if not span or tuple(map(int, span.groups()[:2])) != (offset, last) or int(span[3]) <= last:
            raise RuntimeError(f"incorrect Content-Range for bytes={offset}-{last}")
        identity = {
            "size": int(span[3]),
            "etag": response.getheader("ETag"),
            "last_modified": response.getheader("Last-Modified"),
        }
        with self._lock:
            if self._identity is None:
                self._identity = identity
            elif self._identity != identity:
                raise RuntimeError("HTTP source changed while preparing the subset")

    def _connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            return http.client.HTTPSConnection(
                self._host,
                self._port,
                timeout=self._timeout,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)

    def _connection(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            with self._lock:
                self._conns.append(conn)
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
            self._local.conn = None
            with self._lock:
                self._conns.remove(conn)

    def read(self, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        last = offset + size - 1
        headers = {
            "Host": self._host,
            "Range": f"bytes={offset}-{last}",
            "Connection": "keep-alive",
        }
        last_error: Exception | None = None
        for attempt in range(_FETCH_RETRIES):
            retry_after = None
            try:
                identity = self.identity()
                if identity.get("etag") and not identity["etag"].startswith("W/"):
                    headers["If-Match"] = identity["etag"]
                elif identity.get("last_modified"):
                    headers["If-Unmodified-Since"] = identity["last_modified"]
                conn = self._connection()
                conn.request("GET", self._path, headers=headers)
                response = conn.getresponse()
                if response.status in _RETRY_STATUSES:
                    retry_after = response.getheader("Retry-After")
                    raise OSError(f"HTTP {response.status} for Range bytes={offset}-{last}")
                if response.status == 200:
                    self._drop()
                    raise RuntimeError(
                        "server ignored Range and returned 200; refusing to "
                        "download the whole object"
                    )
                if response.status != 206:
                    status = response.status
                    self._drop()
                    raise RuntimeError(
                        f"HTTP {status} for Range bytes={offset}-{last} "
                        f"on {self._host}{self._path}"
                    )
                self._validate_response(response, offset, last)
                data = response.read(size + 1)
                if len(data) != size:
                    raise OSError(f"short HTTP body at {offset}: {len(data)} of {size} bytes")
                return data
            except RuntimeError:
                self._drop()
                raise
            except (http.client.HTTPException, OSError, TimeoutError) as exc:
                last_error = exc
                self._drop()
                if attempt + 1 < _FETCH_RETRIES:
                    delay = _retry_delay(attempt, retry_after)
                    logging.warning(
                        "%s; retry %s/%s in %.1fs", exc, attempt + 1, _FETCH_RETRIES - 1, delay
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"Range bytes={offset}-{last} failed after {_FETCH_RETRIES} tries"
        ) from last_error

    def close(self) -> None:
        with self._lock:
            for conn in self._conns:
                try:
                    conn.close()
                except OSError:
                    pass
            self._conns.clear()
        self._local.conn = None


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    delay = min(30.0, 2.0**attempt + random.uniform(0, 0.25))
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(retry_after) - datetime.now(UTC)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = 0.0
        delay = max(delay, min(60.0, seconds))
    return delay


def parse_index_range(value) -> tuple[int, int] | None:
    """Parse a half-open [start, stop) range from YAML, CLI, or None (= all)."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in ("", "all", ":"):
            return None
        if text.startswith("["):
            text = text.strip("[]() ")
            text = text.replace(",", ":")
        if ":" not in text:
            raise ValueError(f"range {value!r} is not start:stop or [start, stop)")
        start_s, stop_s = text.split(":", 1)
        return int(start_s), int(stop_s)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"range {value!r} is not start:stop or [start, stop)")


def parse_axis_window(value, *, axis: str) -> AxisWindow | None:
    """Parse a pulse or sample window from YAML or a CLI mapping."""
    if value is None:
        return None
    if isinstance(value, AxisWindow):
        return value
    if isinstance(value, str) or (
        isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(value[0], dict)
    ):
        span = parse_index_range(value)
        if span is None:
            return None
        return AxisWindow(start=span[0], stop=span[1])
    if not isinstance(value, dict):
        raise ValueError(
            f"{axis}: expected [start, stop) or a mapping with count / "
            f"bandwidth_hz / duration_s, not {value!r}"
        )
    origin = str(value.get("origin", "center"))
    if origin not in _ORIGINS:
        raise ValueError(f"{axis}.origin must be start, center, or end, not {origin!r}")
    start = value.get("start")
    stop = value.get("stop")
    count = value.get("count")
    bandwidth_hz = value.get("bandwidth_hz")
    duration_s = value.get("duration_s")
    has_span = start is not None or stop is not None
    sized = (count is not None, bandwidth_hz is not None, duration_s is not None)
    if has_span:
        if start is None or stop is None:
            raise ValueError(f"{axis}: start and stop must be given together")
        if any(sized):
            raise ValueError(
                f"{axis}: [start, stop) cannot mix with count / bandwidth_hz / duration_s"
            )
        return AxisWindow(start=int(start), stop=int(stop), origin=origin)
    if sum(sized) > 1:
        raise ValueError(f"{axis}: give only one of count, bandwidth_hz, duration_s")
    if sum(sized) == 0:
        raise ValueError(f"{axis}: mapping needs count, bandwidth_hz, duration_s, or start+stop")
    if duration_s is not None and axis != "pulses":
        raise ValueError("duration_s applies to pulses, not samples")
    if bandwidth_hz is not None and axis != "samples":
        raise ValueError("bandwidth_hz applies to samples, not pulses")
    return AxisWindow(
        count=int(count) if count is not None else None,
        origin=origin,
        bandwidth_hz=float(bandwidth_hz) if bandwidth_hz is not None else None,
        duration_s=float(duration_s) if duration_s is not None else None,
    )


def as_axis_window(value, *, axis: str) -> AxisWindow | None:
    """Accept None, ``(start, stop)``, or :class:`AxisWindow` at call sites."""
    if value is None or isinstance(value, AxisWindow):
        return value
    return parse_axis_window(value, axis=axis)


def resolve_range(span: tuple[int, int] | None, count: int, name: str) -> tuple[int, int]:
    if span is None:
        return 0, count
    start, stop = span
    if start < 0 or stop < 0:
        raise ValueError(f"{name}: negative indices are not accepted")
    if start >= stop:
        raise ValueError(f"{name}: empty range [{start}, {stop})")
    if stop > count:
        raise ValueError(f"{name}: stop {stop} exceeds {count}")
    return start, stop


def _place_count(n_keep: int, n_total: int, origin: str, name: str) -> tuple[int, int]:
    if n_keep < 1:
        raise ValueError(f"{name}: count must be >= 1, not {n_keep}")
    if n_keep > n_total:
        raise ValueError(f"{name}: count {n_keep} exceeds {n_total}")
    if origin == "start":
        return 0, n_keep
    if origin == "end":
        return n_total - n_keep, n_total
    start = (n_total - n_keep) // 2
    return start, start + n_keep


def resolve_axis_window(
    spec: AxisWindow | None,
    n_total: int,
    name: str,
    *,
    scss: float | None = None,
    tx_time: np.ndarray | None = None,
) -> tuple[int, int]:
    """Turn a window spec into a half-open index range on an axis of length ``n_total``."""
    if spec is None:
        return 0, n_total
    if spec.start is not None:
        return resolve_range((spec.start, spec.stop), n_total, name)
    n_keep = spec.count
    if spec.bandwidth_hz is not None:
        if scss is None or scss == 0.0:
            raise ValueError(f"{name}: bandwidth_hz needs a non-zero SCSS from the file")
        n_keep = int(round(spec.bandwidth_hz / scss))
    elif spec.duration_s is not None:
        if tx_time is None or len(tx_time) < 2:
            raise ValueError(f"{name}: duration_s needs at least two pulse times")
        pri = float(np.median(np.diff(np.asarray(tx_time, dtype=np.float64))))
        if pri <= 0.0:
            raise ValueError(f"{name}: pulse interval is not positive")
        n_keep = int(round(spec.duration_s / pri))
    if n_keep is None:
        raise ValueError(f"{name}: window spec has nothing to resolve")
    return _place_count(n_keep, n_total, spec.origin, name)


def load_subset_recipe(path: str | Path) -> SubsetRecipe:
    """Load a subset recipe YAML (Houston, or a local-file override)."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "name" not in raw:
        raise ValueError(f"{path}: recipe must be a mapping with a name")
    source_block = raw.get("source") or {}
    if isinstance(source_block, str):
        source = source_block
    elif source_block.get("path"):
        source = str(source_block["path"])
    elif source_block.get("url"):
        source = str(source_block["url"])
    elif source_block.get("key"):
        bucket = source_block.get("bucket", "umbra-open-data-catalog")
        key = str(source_block["key"])
        source = f"https://{bucket}.s3.us-west-2.amazonaws.com/" + urllib.parse.quote(key, safe="/")
    else:
        raise ValueError(f"{path}: source needs path, url, or bucket+key")
    connections = int(raw.get("connections", _DEFAULT_CONNECTIONS))
    if connections < 1:
        raise ValueError(f"{path}: connections must be >= 1")
    return SubsetRecipe(
        name=str(raw["name"]),
        source=source,
        pulses=parse_axis_window(raw.get("pulses"), axis="pulses"),
        samples=parse_axis_window(raw.get("samples"), axis="samples"),
        connections=connections,
    )


def open_source(spec: str | Path) -> ByteSource:
    text = str(spec)
    if text.startswith("http://") or text.startswith("https://"):
        return HttpByteSource(text)
    return FileByteSource(Path(text))


def _read_header(source: ByteSource) -> dict[str, str]:
    # Tiny generated fixtures can be shorter than the remote-object probe.
    # Keep strict exact reads for the actual XML, PVP, and signal blocks.
    local_size = source._path.stat().st_size if isinstance(source, FileByteSource) else None
    probe = source.read(
        0, min(_HEADER_PROBE, local_size) if local_size is not None else _HEADER_PROBE
    )
    if not probe.startswith(b"CPHD/"):
        raise RuntimeError("not a CPHD file (missing CPHD/ magic)")
    term = probe.find(skcphd.SECTION_TERMINATOR)
    if term < 0:
        probe = source.read(
            0, min(1024 * 1024, local_size) if local_size is not None else 1024 * 1024
        )
        term = probe.find(skcphd.SECTION_TERMINATOR)
        if term < 0:
            raise RuntimeError("ASCII header has no form-feed terminator")
    text = probe[:term].decode("ascii")
    kvps: dict[str, str] = {}
    for line in text.split("\n")[1:]:
        if not line:
            continue
        if " := " not in line:
            continue
        field, value = line.split(" := ", 1)
        kvps[field] = value
    for key in (
        "XML_BLOCK_BYTE_OFFSET",
        "XML_BLOCK_SIZE",
        "PVP_BLOCK_BYTE_OFFSET",
        "SIGNAL_BLOCK_BYTE_OFFSET",
    ):
        if key not in kvps:
            raise RuntimeError(f"CPHD header missing {key}")
    return kvps


def _channel_data(xmltree: lxml.etree.ElementTree):
    channels = xmltree.findall("{*}Data/{*}Channel")
    if len(channels) != 1:
        raise RuntimeError(f"subsetter requires exactly one CPHD channel, found {len(channels)}")
    return channels[0]


def _require_cf8(xmltree: lxml.etree.ElementTree) -> None:
    fmt = xmltree.findtext("{*}Data/{*}SignalArrayFormat")
    if fmt != "CF8":
        raise RuntimeError(f"subsetter writes CF8 only, this file is {fmt!r}")
    if _channel_data(xmltree).find("{*}CompressedSignalSize") is not None:
        raise RuntimeError("subsetter refuses compressed signal arrays")


def _set_text(node, tag: str, text: str) -> None:
    if node is None:
        raise RuntimeError(f"CPHD XML missing {tag}")
    child = node.find("{*}" + tag)
    if child is None:
        raise RuntimeError(f"CPHD XML missing {tag}")
    child.text = text


def _apply_subband_pvps(pvps: np.ndarray, sample_start: int, num_samples: int) -> np.ndarray:
    """Shift SC0 / FX1 / FX2 onto the kept sample grid. SCSS is unchanged."""
    out = pvps.copy()
    scss = out["SCSS"]
    sc0 = out["SC0"] + sample_start * scss
    out["SC0"] = sc0
    if "FX1" in out.dtype.names:
        out["FX1"] = sc0
    if "FX2" in out.dtype.names:
        out["FX2"] = sc0 + (num_samples - 1) * scss
    return out


def _apply_subset_xml(
    xmltree: lxml.etree.ElementTree,
    pvps: np.ndarray,
    *,
    pulse_start: int,
    orig_num_vectors: int,
    num_vectors: int,
    num_samples: int,
) -> None:
    channel = _channel_data(xmltree)
    _set_text(channel, "NumVectors", str(num_vectors))
    _set_text(channel, "NumSamples", str(num_samples))
    _set_text(channel, "SignalArrayByteOffset", "0")
    _set_text(channel, "PVPArrayByteOffset", "0")

    sc0 = np.asarray(pvps["SC0"], dtype=np.float64)
    scss = np.asarray(pvps["SCSS"], dtype=np.float64)
    fx1 = np.asarray(pvps["FX1"], dtype=np.float64) if "FX1" in pvps.dtype.names else sc0
    fx2 = (
        np.asarray(pvps["FX2"], dtype=np.float64)
        if "FX2" in pvps.dtype.names
        else sc0 + (num_samples - 1) * scss
    )
    scss0 = float(np.asarray(pvps["SCSS"], dtype=np.float64)[0])
    sc0_0 = float(np.asarray(pvps["SC0"], dtype=np.float64)[0])
    # INPUT_DATA.md: centre and bandwidth from the actual kept grid.
    fx_c = sc0_0 + 0.5 * (num_samples - 1) * scss0
    fx_bw = num_samples * scss0

    fx_band = xmltree.find("{*}Global/{*}FxBand")
    if fx_band is not None:
        _set_text(fx_band, "FxMin", repr(float(fx1.min())))
        _set_text(fx_band, "FxMax", repr(float(fx2.max())))

    timeline = xmltree.find("{*}Global/{*}Timeline")
    if timeline is not None:
        _set_text(timeline, "TxTime1", repr(float(pvps["TxTime"][0])))
        _set_text(timeline, "TxTime2", repr(float(pvps["TxTime"][-1])))

    params = xmltree.find("{*}Channel/{*}Parameters")
    _set_text(params, "FxC", repr(float(fx_c)))
    _set_text(params, "FxBW", repr(float(fx_bw)))
    old_ref_text = params.findtext("{*}RefVectorIndex")
    if old_ref_text is not None:
        old_ref = int(old_ref_text)
        if pulse_start <= old_ref < pulse_start + num_vectors:
            new_ref = old_ref - pulse_start
        else:
            new_ref = min(max(old_ref - pulse_start, 0), num_vectors - 1)
        _set_text(params, "RefVectorIndex", str(new_ref))

    if (pulse_start != 0 or (pulse_start + num_vectors) != orig_num_vectors) and (
        lxml.etree.QName(xmltree.getroot()).namespace is not None
    ):
        new_geom = skcphd.compute_reference_geometry(xmltree, pvps)
        root = xmltree.getroot()
        old_geom = root.find("{*}ReferenceGeometry")
        if old_geom is not None:
            root.replace(old_geom, new_geom)
        else:
            root.append(new_geom)


def _validate_xmltree(xmltree: lxml.etree.ElementTree) -> None:
    namespace = lxml.etree.QName(xmltree.getroot()).namespace
    info = skcphd.VERSION_INFO.get(namespace)
    if info is None:
        if namespace is None:
            # The C++ smoke fixture uses intentionally minimal unnamespaced XML.
            return
        raise RuntimeError(f"unsupported CPHD XML namespace {namespace!r}")
    schema = lxml.etree.XMLSchema(lxml.etree.parse(str(info["schema"])))
    if not schema.validate(xmltree):
        raise ValueError(f"subset CPHD XML failed schema validation: {schema.error_log}")


def _sidecar_path(dest: Path) -> Path:
    return dest.with_suffix(".subset.json")


def _window_record(source: str | Path, pulses: tuple[int, int], samples: tuple[int, int]) -> dict:
    return {
        "source": str(source),
        "pulses": [int(pulses[0]), int(pulses[1])],
        "samples": [int(samples[0]), int(samples[1])],
    }


def _dest_already_matches(
    dest: Path, source: str | Path, pulses: tuple[int, int], samples: tuple[int, int]
) -> bool:
    sidecar = _sidecar_path(dest)
    if not dest.is_file() or not sidecar.is_file():
        return False
    try:
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == _window_record(source, pulses, samples)


def _write_sidecar(
    dest: Path, source: str | Path, pulses: tuple[int, int], samples: tuple[int, int]
) -> None:
    _write_json_atomic(_sidecar_path(dest), _window_record(source, pulses, samples))


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _hash_region(handle: BinaryIO, offset: int, size: int) -> str:
    handle.seek(offset)
    digest = hashlib.sha256()
    while size:
        block = handle.read(min(size, 1024 * 1024))
        if not block:
            raise RuntimeError("partial download is truncated; use --force to restart")
        digest.update(block)
        size -= len(block)
    return digest.hexdigest()


@contextmanager
def _destination_lock(dest: Path):
    # Retain the tiny lock file: unlinking it lets a third process bypass a
    # waiter that still owns the previous inode. No lock is held after exit.
    with dest.with_name("." + dest.name + ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another download is already writing {dest}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _validate_checkpoint(handle: BinaryIO, state: dict, request: dict) -> int:
    if state.get("version") != 1 or state.get("request") != request:
        raise RuntimeError("partial download source/window changed; use --force to restart")
    count = state.get("written_vectors")
    start = state.get("last_chunk_start")
    if (
        type(count) is not int
        or not 0 <= count <= request["num_vectors"]
        or type(start) is not int
        or not 0 <= start <= count
    ):
        raise RuntimeError("invalid download checkpoint; use --force to restart")
    offset, row_size = request["dest_offset"], request["row_size"]
    if os.fstat(handle.fileno()).st_size < offset + count * row_size:
        raise RuntimeError("partial download is truncated; use --force to restart")
    if _hash_region(handle, 0, offset) != request["prefix_sha256"]:
        raise RuntimeError("partial download metadata changed; use --force to restart")
    if count and (
        start == count
        or _hash_region(handle, offset + start * row_size, (count - start) * row_size)
        != state.get("last_chunk_sha256")
    ):
        raise RuntimeError("partial download tail is corrupt; use --force to restart")
    return count


def _download_signal(
    dest: Path,
    source_name: str | Path,
    source: ByteSource,
    prefix: BinaryIO,
    dest_offset: int,
    *,
    pulses: tuple[int, int],
    samples: tuple[int, int],
    force: bool,
    **stream_options,
) -> None:
    part = dest.with_suffix(dest.suffix + ".part")
    journal = part.with_suffix(part.suffix + ".json")
    identity = source.identity()
    request = {
        **_window_record(source_name, pulses, samples),
        "source_identity": identity,
        "prefix_sha256": _hash_region(prefix, 0, dest_offset),
        "dest_offset": dest_offset,
        "num_vectors": stream_options["num_vectors"],
        "row_size": stream_options["num_samples"] * _CF8_BYTES,
    }
    with _destination_lock(dest):
        if force:
            part.unlink(missing_ok=True)
            journal.unlink(missing_ok=True)
        if journal.exists() or part.exists():
            etag = identity.get("etag")
            if isinstance(source, HttpByteSource) and not (
                (etag and not etag.startswith("W/")) or identity.get("last_modified")
            ):
                raise RuntimeError(
                    "server has no source validator for resume; use --force to restart"
                )
            try:
                state = json.loads(journal.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    raise ValueError("checkpoint is not an object")
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "missing/invalid download checkpoint; use --force to restart"
                ) from exc
            # Recover a crash between publishing the complete CPHD and writing
            # its sidecar. Never infer completion solely from a file's length.
            published = not part.exists() and state.get("written_vectors") == request["num_vectors"]
            candidate = dest if published else part
            if not candidate.is_file():
                raise RuntimeError("checkpoint has no partial file; use --force to restart")
            with candidate.open("rb") as handle:
                count = _validate_checkpoint(handle, state, request)
            if published:
                _write_sidecar(dest, source_name, pulses, samples)
                journal.unlink()
                _sync_directory(dest.parent)
                return
        else:
            state = {
                "version": 1,
                "request": request,
                "written_vectors": 0,
                "last_chunk_start": 0,
                "last_chunk_sha256": None,
            }
            count = 0
            prefix.seek(0)
            with part.open("xb") as handle:
                shutil.copyfileobj(prefix, handle, 1024 * 1024)
                handle.flush()
                os.fsync(handle.fileno())
            _write_json_atomic(journal, state)
        reporter = stream_options.get("reporter")
        if reporter is not None:
            reporter.phase(
                "Subset preparation",
                total=request["num_vectors"] * request["row_size"],
                initial=count * request["row_size"],
            )
            if count:
                reporter.message(f"resuming {count}/{request['num_vectors']} vectors checkpointed")
        with part.open("r+b") as handle:
            # Drop only uncheckpointed bytes from an interrupted chunk.
            handle.truncate(dest_offset + count * request["row_size"])

            def checkpoint(written: int, start: int, payload: np.ndarray) -> None:
                if source.identity() != identity:
                    raise RuntimeError("source changed while preparing the subset")
                handle.flush()
                os.fsync(handle.fileno())
                update = {
                    **state,
                    "written_vectors": written,
                    "last_chunk_start": start,
                    "last_chunk_sha256": hashlib.sha256(payload).hexdigest(),
                }
                _write_json_atomic(journal, update)
                state.update(update)

            try:
                _stream_signal(
                    handle,
                    source,
                    dest_offset=dest_offset,
                    start_vector=count,
                    checkpoint=checkpoint,
                    **stream_options,
                )
            except BaseException:
                if reporter is not None:
                    reporter.message(
                        f"Kept {state['written_vectors']}/{request['num_vectors']} vectors in {part}; "
                        "rerun the same command to resume."
                    )
                raise
        part.replace(dest)
        _sync_directory(dest.parent)
        _write_sidecar(dest, source_name, pulses, samples)
        journal.unlink()
        _sync_directory(dest.parent)


def _signal_row_offset(
    signal_block: int,
    channel_offset: int,
    vector: int,
    row_bytes: int,
    sample_start: int,
) -> int:
    return signal_block + channel_offset + vector * row_bytes + sample_start * _CF8_BYTES


def _fetch_workers(source: ByteSource, connections: int, num_vectors: int) -> int:
    # Local files are sequential strided reads; parallel seeks on OverlayFS
    # lose to one thread. HTTP Range stays request-rate bound.
    if isinstance(source, FileByteSource):
        return 1
    return max(1, min(connections, num_vectors))


def _fetch_signal(
    source: ByteSource,
    *,
    signal_block: int,
    channel_offset: int,
    orig_num_samples: int,
    pulse_start: int,
    num_vectors: int,
    sample_start: int,
    num_samples: int,
    connections: int,
    pool: ThreadPoolExecutor | None = None,
    reporter: DownloadProgress | None = None,
) -> np.ndarray:
    row_bytes = orig_num_samples * _CF8_BYTES
    row_size = num_samples * _CF8_BYTES
    out = np.empty((num_vectors, num_samples), dtype=np.dtype(">c8"))
    workers = _fetch_workers(source, connections, num_vectors)

    def fetch_one(index: int) -> None:
        vector = pulse_start + index
        offset = _signal_row_offset(signal_block, channel_offset, vector, row_bytes, sample_start)
        payload = source.read(offset, row_size)
        out[index] = np.frombuffer(payload, dtype=np.dtype(">c8"))
        if reporter is not None:
            reporter.add(len(payload))

    if workers == 1:
        for index in range(num_vectors):
            fetch_one(index)
        return out

    if pool is not None:
        for _ in pool.map(fetch_one, range(num_vectors)):
            pass
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in executor.map(fetch_one, range(num_vectors)):
                pass
    return out


def _stream_signal(
    handle: BinaryIO,
    source: ByteSource,
    *,
    dest_offset: int,
    signal_block: int,
    channel_offset: int,
    orig_num_samples: int,
    pulse_start: int,
    num_vectors: int,
    sample_start: int,
    num_samples: int,
    connections: int,
    start_vector: int = 0,
    checkpoint: Callable[[int, int, np.ndarray], None],
    reporter: DownloadProgress | None = None,
) -> None:
    """Write CF8 rows in bounded chunks. sarkit Writer has no partial API."""
    row_size = num_samples * _CF8_BYTES
    chunk_n = max(1, _SIGNAL_CHUNK_BYTES // row_size)
    expected = num_vectors * row_size
    handle.seek(dest_offset + start_vector * row_size)
    written_bytes = start_vector * row_size
    written_vectors = start_vector
    # Keep the same workers/connections across chunks, rather than retaining
    # another pool's sockets in HttpByteSource for every 64 MiB of the file.
    with ThreadPoolExecutor(max_workers=_fetch_workers(source, connections, num_vectors)) as pool:
        while written_vectors < num_vectors:
            n_keep = min(chunk_n, num_vectors - written_vectors)
            chunk = _fetch_signal(
                source,
                signal_block=signal_block,
                channel_offset=channel_offset,
                orig_num_samples=orig_num_samples,
                pulse_start=pulse_start + written_vectors,
                num_vectors=n_keep,
                sample_start=sample_start,
                num_samples=num_samples,
                connections=connections,
                pool=pool,
                reporter=reporter,
            )
            payload = np.ascontiguousarray(chunk)
            payload.tofile(handle)
            chunk_start = written_vectors
            written_bytes += int(payload.nbytes)
            written_vectors += n_keep
            checkpoint(written_vectors, chunk_start, payload)
    if written_bytes != expected:
        raise RuntimeError(f"streamed {written_bytes} signal bytes, expected {expected}")


def _read_support_arrays(
    source: ByteSource,
    xmltree: lxml.etree.ElementTree,
    kvps: dict[str, str],
) -> dict[str, np.ndarray]:
    if "SUPPORT_BLOCK_BYTE_OFFSET" not in kvps:
        return {}
    arrays: dict[str, np.ndarray] = {}
    support_base = int(kvps["SUPPORT_BLOCK_BYTE_OFFSET"])
    for node in xmltree.findall("{*}Data/{*}SupportArray"):
        ident = node.findtext("{*}Identifier")
        if ident is None:
            continue
        num_rows = int(node.findtext("{*}NumRows"))
        num_cols = int(node.findtext("{*}NumCols"))
        bytes_per = int(node.findtext("{*}BytesPerElement"))
        offset = support_base + int(node.findtext("{*}ArrayByteOffset"))
        payload = source.read(offset, num_rows * num_cols * bytes_per)
        sa_elem = xmltree.find(f"{{*}}SupportArray/*[{{*}}Identifier='{ident}']")
        if sa_elem is None:
            raise RuntimeError(f"support array {ident!r} missing from SupportArray")
        fmt = sa_elem.findtext("{*}ElementFormat")
        dtype = skcphd.binary_format_string_to_dtype(fmt).newbyteorder(">")
        arrays[ident] = np.frombuffer(payload, dtype=dtype).reshape(num_rows, num_cols)
    return arrays


def _write_minimal_prefix(
    handle: BinaryIO,
    xmltree: lxml.etree.ElementTree,
    pvps: np.ndarray,
    *,
    num_vectors: int,
    num_samples: int,
) -> int:
    """Write the C++ smoke fixture's intentionally unnamespaced CPHD dialect."""
    xml_bytes = lxml.etree.tostring(xmltree.getroot(), encoding="UTF-8")
    pvp_bytes = pvps.tobytes()

    def align(value: int) -> int:
        return (value + 1023) & ~1023

    xml_offset = 1024
    pvp_offset = align(xml_offset + len(xml_bytes))
    signal_offset = align(pvp_offset + len(pvp_bytes))
    header = (
        "CPHD/1.1.0\n"
        f"XML_BLOCK_BYTE_OFFSET := {xml_offset}\n"
        f"XML_BLOCK_SIZE := {len(xml_bytes)}\n"
        f"PVP_BLOCK_BYTE_OFFSET := {pvp_offset}\n"
        f"PVP_BLOCK_SIZE := {len(pvp_bytes)}\n"
        f"SIGNAL_BLOCK_BYTE_OFFSET := {signal_offset}\n"
        f"SIGNAL_BLOCK_SIZE := {num_vectors * num_samples * _CF8_BYTES}\n"
        "\f\n"
    ).encode("ascii")
    if len(header) > xml_offset:
        raise RuntimeError("minimal CPHD header exceeds its XML offset")
    handle.write(header)
    handle.seek(xml_offset)
    handle.write(xml_bytes)
    handle.seek(pvp_offset)
    handle.write(pvp_bytes)
    return signal_offset


def subset_cphd(
    source: str | Path,
    dest: str | Path,
    *,
    pulses: AxisWindow | tuple[int, int] | None = None,
    samples: AxisWindow | tuple[int, int] | None = None,
    connections: int = _DEFAULT_CONNECTIONS,
    force: bool = False,
    dry_run: bool = False,
    progress: bool = False,
    progress_label: str | None = None,
) -> SubsetResult:
    """Write a pulse-and/or-sample subset of ``source`` to ``dest``.

    ``progress`` is off by default so importing this as a library is quiet.
    The CLI passes ``progress=True``. Matching partial downloads resume
    automatically; ``force`` explicitly discards their checkpoint.
    """
    reporting = (
        DownloadProgress(progress_label or str(dest), None, stage="Subset preparation")
        if progress
        else nullcontext()
    )
    with reporting as reporter:
        result = _subset_cphd(
            source,
            dest,
            pulses=pulses,
            samples=samples,
            connections=connections,
            force=force,
            dry_run=dry_run,
            reporter=reporter,
        )
        if reporter is not None and result.skipped:
            reporter.message("Skipped: existing subset matches source and windows")
        return result


def _subset_cphd(
    source: str | Path,
    dest: str | Path,
    *,
    pulses: AxisWindow | tuple[int, int] | None,
    samples: AxisWindow | tuple[int, int] | None,
    connections: int,
    force: bool,
    dry_run: bool,
    reporter: DownloadProgress | None,
) -> SubsetResult:
    dest = Path(dest)
    pulse_spec = as_axis_window(pulses, axis="pulses")
    sample_spec = as_axis_window(samples, axis="samples")
    src = open_source(source)
    try:
        kvps = _read_header(src)
        xml_bytes = src.read(int(kvps["XML_BLOCK_BYTE_OFFSET"]), int(kvps["XML_BLOCK_SIZE"]))
        xmltree = lxml.etree.fromstring(xml_bytes).getroottree()
        _require_cf8(xmltree)
        channel = _channel_data(xmltree)
        orig_nv = int(channel.findtext("{*}NumVectors"))
        orig_ns = int(channel.findtext("{*}NumSamples"))
        channel_id = channel.findtext("{*}Identifier")
        if not channel_id:
            raise RuntimeError("CPHD channel Identifier is missing")
        signal_offset = int(channel.findtext("{*}SignalArrayByteOffset"))
        pvp_offset = int(channel.findtext("{*}PVPArrayByteOffset"))

        pvp_dtype = skcphd.get_pvp_dtype(xmltree).newbyteorder(">")
        # Some C++ fixtures omit PVP_BLOCK_SIZE; the single-channel XML still
        # gives the exact number of PVP records, excluding alignment padding.
        pvp_size = int(kvps.get("PVP_BLOCK_SIZE", str(pvp_offset + orig_nv * pvp_dtype.itemsize)))
        pvp_block = src.read(int(kvps["PVP_BLOCK_BYTE_OFFSET"]), pvp_size)
        itemsize = pvp_dtype.itemsize
        n_records = len(pvp_block) // itemsize
        pvps_all = np.frombuffer(pvp_block, dtype=pvp_dtype, count=n_records)
        first = pvp_offset // itemsize
        channel_pvps = pvps_all[first : first + orig_nv]
        if channel_pvps.shape[0] != orig_nv:
            raise RuntimeError(
                f"PVP block has {channel_pvps.shape[0]} records, XML declares {orig_nv}"
            )

        scss0 = float(np.asarray(channel_pvps["SCSS"], dtype=np.float64)[0])
        pulse_start, pulse_stop = resolve_axis_window(
            pulse_spec, orig_nv, "pulses", tx_time=channel_pvps["TxTime"]
        )
        sample_start, sample_stop = resolve_axis_window(sample_spec, orig_ns, "samples", scss=scss0)
        num_vectors = pulse_stop - pulse_start
        num_samples = sample_stop - sample_start
        pvps = _apply_subband_pvps(channel_pvps[pulse_start:pulse_stop], sample_start, num_samples)

        request_bytes = num_vectors * num_samples * _CF8_BYTES
        if reporter is not None:
            reporter.message(
                f"pulses [{pulse_start}, {pulse_stop}) of {orig_nv}; "
                f"samples [{sample_start}, {sample_stop}) of {orig_ns}; "
                f"{request_bytes} selected signal bytes"
            )
        if dry_run:
            print(
                f"pulses  [{pulse_start}, {pulse_stop}) of {orig_nv}\n"
                f"samples [{sample_start}, {sample_stop}) of {orig_ns}\n"
                f"subset {orig_nv} x {orig_ns} -> {num_vectors} x {num_samples} "
                f"({request_bytes / (1024**3):.2f} GiB, {num_vectors} range requests)"
            )
        result_kw = dict(
            path=dest,
            num_vectors=num_vectors,
            num_samples=num_samples,
            pulse_range=(pulse_start, pulse_stop),
            sample_range=(sample_start, sample_stop),
        )
        if dry_run:
            return SubsetResult(skipped=False, dry_run=True, **result_kw)

        dest.parent.mkdir(parents=True, exist_ok=True)
        if (
            dest.is_file()
            and not force
            and not dest.with_suffix(dest.suffix + ".part").exists()
            and not dest.with_suffix(dest.suffix + ".part.json").exists()
            and _dest_already_matches(
                dest, source, (pulse_start, pulse_stop), (sample_start, sample_stop)
            )
        ):
            return SubsetResult(skipped=True, **result_kw)

        support = _read_support_arrays(src, xmltree, kvps)

        xml_out = copy.deepcopy(xmltree)
        _apply_subset_xml(
            xml_out,
            pvps,
            pulse_start=pulse_start,
            orig_num_vectors=orig_nv,
            num_vectors=num_vectors,
            num_samples=num_samples,
        )
        if reporter is not None:
            reporter.phase("Validation")
            reporter.message("checking subset XML")
        _validate_xmltree(xml_out)

        # Structured dtype conversions can leave padding bytes uninitialised.
        # Encode directly in wire byte order with zeroed padding so rebuilding
        # the metadata on resume produces an identical prefix fingerprint.
        encoded_pvps = np.zeros(pvps.shape, dtype=skcphd.get_pvp_dtype(xml_out).newbyteorder(">"))
        for name in encoded_pvps.dtype.names:
            encoded_pvps[name] = pvps[name]

        # Encode only header/XML/PVP/support in a small temporary file. This
        # validates the current writer layout before touching a resumed file.
        with tempfile.TemporaryFile() as prefix:
            if lxml.etree.QName(xml_out.getroot()).namespace is None:
                dest_offset = _write_minimal_prefix(
                    prefix, xml_out, encoded_pvps, num_vectors=num_vectors, num_samples=num_samples
                )
            else:
                extra = {
                    key: value
                    for key, value in kvps.items()
                    if key not in skcphd.DEFINED_HEADER_KEYS
                }
                metadata = skcphd.Metadata(
                    xmltree=xml_out, file_header_part=skcphd.FileHeaderPart(additional_kvps=extra)
                )
                with skcphd.Writer(prefix, metadata) as writer:
                    writer.write_pvp(channel_id, encoded_pvps)
                    for ident, array in support.items():
                        writer.write_support_array(
                            ident, np.asarray(array, dtype=array.dtype.newbyteorder("="))
                        )
                    dest_offset = int(
                        writer._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]
                    ) + int(  # noqa: SLF001
                        writer._channel_size_offsets[channel_id]["signal_offset"]  # noqa: SLF001
                    )
                    # Transfer responsibility for the signal to our bounded,
                    # checkpointed stream; this writer encodes metadata only.
                    writer._signal_arrays_written.add(channel_id)  # noqa: SLF001
            prefix.truncate(dest_offset)
            prefix.flush()
            _download_signal(
                dest,
                source,
                src,
                prefix,
                dest_offset,
                pulses=(pulse_start, pulse_stop),
                samples=(sample_start, sample_stop),
                force=force,
                signal_block=int(kvps["SIGNAL_BLOCK_BYTE_OFFSET"]),
                channel_offset=signal_offset,
                orig_num_samples=orig_ns,
                pulse_start=pulse_start,
                num_vectors=num_vectors,
                sample_start=sample_start,
                num_samples=num_samples,
                connections=connections,
                reporter=reporter,
            )
        return SubsetResult(skipped=False, **result_kw)
    finally:
        src.close()
