# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subset a generated CPHD: pulse window and a contiguous sample subband.

CI runs this against the mode01_smoke fixture so the Houston pull is not a
test dependency. The identity assertion is against sarkit reading both
files, which is also how CPHDReaderOp's parser was pinned.
"""

from __future__ import annotations

import http.server
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(_TOOLS / "sar_sim") not in sys.path:
    sys.path.insert(0, str(_TOOLS / "sar_sim"))

pytest.importorskip("yaml", reason="sar_sim recipes are YAML")
pytest.importorskip("cupy", reason="fixture generation runs on the GPU")
skcphd = pytest.importorskip("sarkit.cphd", reason="subsetter writes CPHD via sarkit")

from sar_sim import generate  # noqa: E402
from sar_sim.subset import (  # noqa: E402
    AxisWindow,
    load_subset_recipe,
    parse_axis_window,
    parse_index_range,
    resolve_axis_window,
    subset_cphd,
)

SMOKE_RECIPE = _TOOLS / "sar_sim" / "recipes" / "mode01_smoke.yaml"
HOUSTON_RECIPE = _TOOLS / "sar_sim" / "recipes" / "mode02_houston.yaml"


@pytest.fixture(scope="module")
def smoke_cphd(tmp_path_factory):
    output = tmp_path_factory.mktemp("subset_src")
    result = generate(SMOKE_RECIPE, output)
    return result.cphd_path


def _read(path: Path):
    with path.open("rb") as handle:
        reader = skcphd.Reader(handle)
        channel_id = reader.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
        xml = reader.metadata.xmltree
        return reader.read_signal(channel_id), reader.read_pvps(channel_id), xml


def test_parse_index_range():
    assert parse_index_range(None) is None
    assert parse_index_range("all") is None
    assert parse_index_range("10:50") == (10, 50)
    assert parse_index_range("[33389, 37485)") == (33389, 37485)
    assert parse_index_range([64, 192]) == (64, 192)


def test_houston_recipe_is_the_input_data_pin():
    recipe = load_subset_recipe(HOUSTON_RECIPE)
    assert recipe.name == "mode02_houston"
    assert recipe.samples == AxisWindow(count=4096, origin="center")
    assert recipe.pulses is None
    assert recipe.connections == 128
    assert "Houston_TX" in recipe.source
    assert recipe.source.startswith("https://")
    # INPUT_DATA.md samples [33389, 37485) is the centred 4096 of 70875.
    assert resolve_axis_window(recipe.samples, 70875, "samples") == (33389, 37485)


def test_houston_gsd_recipes_are_scaled_from_the_4p09m_pin():
    recipes = _TOOLS / "sar_sim" / "recipes"
    half = load_subset_recipe(recipes / "mode02_houston_0p5m.yaml")
    metre = load_subset_recipe(recipes / "mode02_houston_1p0m.yaml")
    assert half.samples == AxisWindow(count=33505, origin="center")
    assert metre.samples == AxisWindow(count=16753, origin="center")
    # 4096 * 4.09 / 0.5 and / 1.0, centred on 70875.
    assert resolve_axis_window(half.samples, 70875, "samples") == (18685, 52190)
    assert resolve_axis_window(metre.samples, 70875, "samples") == (27061, 43814)


def test_parse_axis_window_forms():
    assert parse_axis_window(None, axis="samples") is None
    assert parse_axis_window([64, 192], axis="samples") == AxisWindow(start=64, stop=192)
    spec = parse_axis_window({"count": 4096, "origin": "center"}, axis="samples")
    assert spec == AxisWindow(count=4096, origin="center")
    bw = parse_axis_window({"bandwidth_hz": 61.6e6, "origin": "start"}, axis="samples")
    assert bw is not None
    assert bw.bandwidth_hz == pytest.approx(61.6e6)
    assert bw.origin == "start"
    with pytest.raises(ValueError, match="mix"):
        parse_axis_window({"start": 0, "stop": 10, "count": 4}, axis="samples")


def test_centered_count_on_smoke(smoke_cphd, tmp_path):
    dest = tmp_path / "center.cphd"
    result = subset_cphd(
        smoke_cphd,
        dest,
        samples=AxisWindow(count=128, origin="center"),
        connections=1,
        progress=False,
    )
    assert result.sample_range == (192, 320)
    src_sig, _, _ = _read(smoke_cphd)
    out_sig, _, _ = _read(dest)
    np.testing.assert_array_equal(out_sig, src_sig[:, 192:320])


def test_skip_distinguishes_equal_sized_windows(smoke_cphd, tmp_path):
    dest = tmp_path / "shift.cphd"
    first = subset_cphd(smoke_cphd, dest, samples=(0, 64), progress=False)
    shifted = subset_cphd(smoke_cphd, dest, samples=(64, 128), progress=False)
    assert not first.skipped
    assert not shifted.skipped
    assert shifted.sample_range == (64, 128)


def test_pulse_and_sample_subset_matches_sarkit_slice(smoke_cphd, tmp_path):
    dest = tmp_path / "subset.cphd"
    pulses, samples = (10, 50), (64, 192)
    result = subset_cphd(
        smoke_cphd,
        dest,
        pulses=pulses,
        samples=samples,
        connections=1,
        progress=False,
    )
    assert result.num_vectors == 40
    assert result.num_samples == 128
    assert not result.skipped

    src_sig, src_pvp, _ = _read(smoke_cphd)
    out_sig, out_pvp, out_xml = _read(dest)

    np.testing.assert_array_equal(out_sig, src_sig[10:50, 64:192])
    np.testing.assert_allclose(out_pvp["SC0"], src_pvp["SC0"][10:50] + 64 * src_pvp["SCSS"][10:50])
    np.testing.assert_array_equal(out_pvp["SCSS"], src_pvp["SCSS"][10:50])
    np.testing.assert_array_equal(out_pvp["TxTime"], src_pvp["TxTime"][10:50])

    n_sub = 128
    scss = float(out_pvp["SCSS"][0])
    sc0 = float(out_pvp["SC0"][0])
    assert out_xml.findtext("{*}Data/{*}Channel/{*}NumVectors") == "40"
    assert out_xml.findtext("{*}Data/{*}Channel/{*}NumSamples") == "128"
    # INPUT_DATA.md: FxC from the kept grid midpoint, FxBW = n * SCSS.
    fx_c = float(out_xml.findtext("{*}Channel/{*}Parameters/{*}FxC"))
    fx_bw = float(out_xml.findtext("{*}Channel/{*}Parameters/{*}FxBW"))
    assert fx_c == pytest.approx(sc0 + 0.5 * (n_sub - 1) * scss)
    assert fx_bw == pytest.approx(n_sub * scss)
    assert int(out_xml.findtext("{*}Channel/{*}Parameters/{*}RefVectorIndex")) == (
        _ref_index_in_window(smoke_cphd, 10, 50)
    )


def _ref_index_in_window(path: Path, pulse_start: int, pulse_stop: int) -> int:
    with path.open("rb") as handle:
        xml = skcphd.Reader(handle).metadata.xmltree
    old = int(xml.findtext("{*}Channel/{*}Parameters/{*}RefVectorIndex"))
    n_vec = pulse_stop - pulse_start
    if pulse_start <= old < pulse_stop:
        return old - pulse_start
    return min(max(old - pulse_start, 0), n_vec - 1)


def test_sample_only_keeps_every_pulse(smoke_cphd, tmp_path):
    dest = tmp_path / "samples.cphd"
    subset_cphd(smoke_cphd, dest, samples=(100, 200), connections=1, progress=False)
    src_sig, src_pvp, src_xml = _read(smoke_cphd)
    out_sig, out_pvp, out_xml = _read(dest)
    assert out_sig.shape == (src_sig.shape[0], 100)
    np.testing.assert_array_equal(out_sig, src_sig[:, 100:200])
    orig_ref = int(src_xml.findtext("{*}Channel/{*}Parameters/{*}RefVectorIndex"))
    new_ref = int(out_xml.findtext("{*}Channel/{*}Parameters/{*}RefVectorIndex"))
    assert new_ref == orig_ref
    np.testing.assert_allclose(out_pvp["SC0"], src_pvp["SC0"] + 100 * src_pvp["SCSS"])


def test_skip_when_counts_match(smoke_cphd, tmp_path):
    dest = tmp_path / "once.cphd"
    first = subset_cphd(smoke_cphd, dest, pulses=(0, 32), samples=(0, 64), progress=False)
    second = subset_cphd(smoke_cphd, dest, pulses=(0, 32), samples=(0, 64), progress=False)
    assert not first.skipped
    assert second.skipped
    forced = subset_cphd(
        smoke_cphd, dest, pulses=(0, 32), samples=(0, 64), force=True, progress=False
    )
    assert not forced.skipped


def test_rejects_out_of_bounds(smoke_cphd, tmp_path):
    with pytest.raises(ValueError, match="pulses"):
        subset_cphd(smoke_cphd, tmp_path / "bad.cphd", pulses=(0, 10_000), progress=False)


def test_http_range_matches_local_file(smoke_cphd, tmp_path):
    """The Houston path is HTTP Range; pin it against the local reader."""

    class RangeHandler(http.server.BaseHTTPRequestHandler):
        # Python 3.12 SimpleHTTPRequestHandler ignores Range (3.13+ honors it).
        protocol_version = "HTTP/1.1"
        directory = str(smoke_cphd.parent)

        def log_message(self, format, *args):  # noqa: A002
            del format, args

        def do_GET(self):
            path = Path(self.directory) / self.path.lstrip("/")
            if not path.is_file():
                self.send_error(404, "File not found")
                return
            file_size = path.stat().st_size
            start, stop = 0, file_size
            status = 200
            range_header = self.headers.get("Range")
            if range_header:
                if not range_header.startswith("bytes="):
                    self.send_error(400, "bad Range")
                    return
                spec = range_header[len("bytes=") :]
                first, _, last = spec.partition("-")
                start = int(first) if first else 0
                stop = int(last) + 1 if last else file_size
                if start < 0 or start >= stop or stop > file_size:
                    self.send_error(416, "Range not satisfiable")
                    return
                status = 206
            length = stop - start
            self.send_response(status)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Type", "application/octet-stream")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{stop - 1}/{file_size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/{smoke_cphd.name}"
        local_dest = tmp_path / "local.cphd"
        http_dest = tmp_path / "http.cphd"
        kwargs = dict(pulses=(8, 24), samples=(32, 96), connections=4, progress=False)
        subset_cphd(smoke_cphd, local_dest, **kwargs)
        subset_cphd(url, http_dest, **kwargs)
        local_sig, local_pvp, _ = _read(local_dest)
        http_sig, http_pvp, _ = _read(http_dest)
        np.testing.assert_array_equal(http_sig, local_sig)
        np.testing.assert_array_equal(http_pvp["SC0"], local_pvp["SC0"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
