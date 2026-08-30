# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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
import http.client
import json
import ssl
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import lxml.etree
import numpy as np
import sarkit.cphd as skcphd
import yaml

# CF8 only: Houston and the CI smoke fixture are CF8, and a subband of CI4
# is a later demo's problem (Capella / mode 3).
_CF8_BYTES = 8
_HEADER_PROBE = 64 * 1024
_DEFAULT_CONNECTIONS = 128
_HTTPS_TIMEOUT_S = 60.0
_FETCH_RETRIES = 5
# Host RAM on the demo box is 62 GiB; a 0.5 m Houston subband is ~52 GiB.
# Stream that many bytes through the writer rather than materialising them.
_SIGNAL_CHUNK_BYTES = 64 * 1024 * 1024


class ByteSource(Protocol):
    def read(self, offset: int, size: int) -> bytes: ...

    def close(self) -> None: ...


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
            try:
                conn = self._connection()
                conn.request("GET", self._path, headers=headers)
                response = conn.getresponse()
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
                data = response.read()
                if len(data) != size:
                    raise RuntimeError(f"short HTTP body at {offset}: {len(data)} of {size} bytes")
                return data
            except RuntimeError:
                raise
            except (http.client.HTTPException, OSError, TimeoutError) as exc:
                last_error = exc
                self._drop()
                time.sleep(0.25 * (2**attempt))
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
    probe = source.read(0, _HEADER_PROBE)
    if not probe.startswith(b"CPHD/"):
        raise RuntimeError("not a CPHD file (missing CPHD/ magic)")
    term = probe.find(skcphd.SECTION_TERMINATOR)
    if term < 0:
        probe = source.read(0, 1024 * 1024)
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
        "PVP_BLOCK_SIZE",
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

    fx1 = np.asarray(pvps["FX1"], dtype=np.float64)
    fx2 = np.asarray(pvps["FX2"], dtype=np.float64)
    scss0 = float(np.asarray(pvps["SCSS"], dtype=np.float64)[0])
    sc0_0 = float(np.asarray(pvps["SC0"], dtype=np.float64)[0])
    # INPUT_DATA.md: centre and bandwidth from the actual kept grid.
    fx_c = sc0_0 + 0.5 * (num_samples - 1) * scss0
    fx_bw = num_samples * scss0

    fx_band = xmltree.find("{*}Global/{*}FxBand")
    _set_text(fx_band, "FxMin", repr(float(fx1.min())))
    _set_text(fx_band, "FxMax", repr(float(fx2.max())))

    timeline = xmltree.find("{*}Global/{*}Timeline")
    _set_text(timeline, "TxTime1", repr(float(pvps["TxTime"][0])))
    _set_text(timeline, "TxTime2", repr(float(pvps["TxTime"][-1])))

    params = xmltree.find("{*}Channel/{*}Parameters")
    _set_text(params, "FxC", repr(float(fx_c)))
    _set_text(params, "FxBW", repr(float(fx_bw)))
    old_ref = int(params.findtext("{*}RefVectorIndex"))
    if pulse_start <= old_ref < pulse_start + num_vectors:
        new_ref = old_ref - pulse_start
    else:
        new_ref = min(max(old_ref - pulse_start, 0), num_vectors - 1)
    _set_text(params, "RefVectorIndex", str(new_ref))

    if pulse_start != 0 or (pulse_start + num_vectors) != orig_num_vectors:
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
    _sidecar_path(dest).write_text(
        json.dumps(_window_record(source, pulses, samples), indent=2) + "\n",
        encoding="utf-8",
    )


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
    progress: bool,
) -> np.ndarray:
    row_bytes = orig_num_samples * _CF8_BYTES
    row_size = num_samples * _CF8_BYTES
    out = np.empty((num_vectors, num_samples), dtype=np.dtype(">c8"))
    workers = _fetch_workers(source, connections, num_vectors)
    done = 0
    lock = threading.Lock()
    started = time.perf_counter()

    def fetch_one(index: int) -> None:
        nonlocal done
        vector = pulse_start + index
        offset = _signal_row_offset(signal_block, channel_offset, vector, row_bytes, sample_start)
        payload = source.read(offset, row_size)
        out[index] = np.frombuffer(payload, dtype=np.dtype(">c8"))
        if not progress:
            return
        with lock:
            done += 1
            current = done
        if current % 2000 == 0 or current == num_vectors:
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f"  {current}/{num_vectors} vectors ({current / elapsed:.0f} req/s)",
                flush=True,
            )

    if workers == 1:
        for index in range(num_vectors):
            fetch_one(index)
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch_one, range(num_vectors), chunksize=32):
            pass
    return out


def _stream_signal(
    writer: skcphd.Writer,
    source: ByteSource,
    *,
    channel_id: str,
    signal_block: int,
    channel_offset: int,
    orig_num_samples: int,
    pulse_start: int,
    num_vectors: int,
    sample_start: int,
    num_samples: int,
    connections: int,
    progress: bool,
) -> None:
    """Write CF8 rows in bounded chunks. sarkit Writer has no partial API."""
    row_size = num_samples * _CF8_BYTES
    chunk_n = max(1, _SIGNAL_CHUNK_BYTES // row_size)
    expected = num_vectors * row_size
    dest_off = int(writer._file_header_kvp["SIGNAL_BLOCK_BYTE_OFFSET"]) + int(  # noqa: SLF001
        writer._channel_size_offsets[channel_id]["signal_offset"]  # noqa: SLF001
    )
    handle = writer._file_object  # noqa: SLF001
    handle.seek(dest_off)
    written_bytes = 0
    written_vectors = 0
    started = time.perf_counter()
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
            progress=False,
        )
        payload = np.ascontiguousarray(chunk)
        payload.tofile(handle)
        written_bytes += int(payload.nbytes)
        written_vectors += n_keep
        if progress and (written_vectors % (chunk_n * 8) == 0 or written_vectors == num_vectors):
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                f"  {written_vectors}/{num_vectors} vectors "
                f"({written_vectors / elapsed:.0f} vec/s)",
                flush=True,
            )
    if written_bytes != expected:
        raise RuntimeError(f"streamed {written_bytes} signal bytes, expected {expected}")
    writer._signal_arrays_written.add(channel_id)  # noqa: SLF001


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


def subset_cphd(
    source: str | Path,
    dest: str | Path,
    *,
    pulses: AxisWindow | tuple[int, int] | None = None,
    samples: AxisWindow | tuple[int, int] | None = None,
    connections: int = _DEFAULT_CONNECTIONS,
    force: bool = False,
    dry_run: bool = False,
    progress: bool = True,
) -> SubsetResult:
    """Write a pulse-and/or-sample subset of ``source`` to ``dest``."""
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
        pvp_block = src.read(int(kvps["PVP_BLOCK_BYTE_OFFSET"]), int(kvps["PVP_BLOCK_SIZE"]))
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
        if progress or dry_run:
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
            and _dest_already_matches(
                dest, source, (pulse_start, pulse_stop), (sample_start, sample_stop)
            )
        ):
            print(
                f"skip {dest}: already pulses [{pulse_start}, {pulse_stop}) "
                f"samples [{sample_start}, {sample_stop})"
            )
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
        _validate_xmltree(xml_out)

        extra = {key: value for key, value in kvps.items() if key not in skcphd.DEFINED_HEADER_KEYS}
        metadata = skcphd.Metadata(
            xmltree=xml_out,
            file_header_part=skcphd.FileHeaderPart(additional_kvps=extra),
        )
        pvp_native = np.asarray(pvps, dtype=skcphd.get_pvp_dtype(xml_out))
        tmp = dest.with_suffix(dest.suffix + ".part")
        if tmp.exists():
            tmp.unlink()
        try:
            with tmp.open("wb") as handle, skcphd.Writer(handle, metadata) as writer:
                writer.write_pvp(channel_id, pvp_native)
                for ident, array in support.items():
                    writer.write_support_array(
                        ident, np.asarray(array, dtype=array.dtype.newbyteorder("="))
                    )
                _stream_signal(
                    writer,
                    src,
                    channel_id=channel_id,
                    signal_block=int(kvps["SIGNAL_BLOCK_BYTE_OFFSET"]),
                    channel_offset=signal_offset,
                    orig_num_samples=orig_ns,
                    pulse_start=pulse_start,
                    num_vectors=num_vectors,
                    sample_start=sample_start,
                    num_samples=num_samples,
                    connections=connections,
                    progress=progress,
                )
            tmp.replace(dest)
        except BaseException:
            if tmp.exists():
                tmp.unlink()
            raise
        _write_sidecar(dest, source, (pulse_start, pulse_stop), (sample_start, sample_stop))
        return SubsetResult(skipped=False, **result_kw)
    finally:
        src.close()
