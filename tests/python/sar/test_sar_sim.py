# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-checks for the in-repo SAR simulator (tools/sar_sim).

No CPHD binaries live in git, so every later SAR test stands on a fixture
this simulator generated and on the ground-truth sidecar it wrote beside it.
That makes the sidecar's claims load-bearing: if the predicted pixel indices
are wrong, every downstream assertion is wrong in the same direction and all
of them still pass.

So the assertion here is not that the simulator is self-consistent — it is
that an *independent* implementation, the reference backprojector below,
finds the targets where the sidecar says they are. That is the analytic
tautology guard `TESTING.md` asks for, applied to the simulator itself, and
it is also the reference `BackprojectionOp` will be measured against.

Host-only and GPU-free: none of this touches Holoscan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SIM_ROOT = Path(__file__).resolve().parents[3] / "tools" / "sar_sim"
if str(_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIM_ROOT))

# sarpy/sarkit and pyyaml are confined to the generate path and are not
# module runtime dependencies, so skip rather than fail where they are absent.
pytest.importorskip("yaml", reason="sar_sim recipes are YAML")
skcphd = pytest.importorskip("sarkit.cphd", reason="sar_sim writes CPHD via sarkit")

from sar_sim import generate, load_recipe  # noqa: E402
from sar_sim.geometry import SPEED_OF_LIGHT  # noqa: E402

SMOKE_RECIPE = _SIM_ROOT / "recipes" / "mode01_smoke.yaml"
DEMO_RECIPE = _SIM_ROOT / "recipes" / "mode01.yaml"


@pytest.fixture(scope="module")
def smoke_fixture(tmp_path_factory):
    """Generate the CI smoke collection once for the whole module."""
    output = tmp_path_factory.mktemp("sar_sim")
    result = generate(SMOKE_RECIPE, output)
    sidecar = json.loads(result.sidecar_path.read_text())
    return result, sidecar


def read_cphd(path: Path):
    """Read a CPHD back through sarkit, as an external consumer would."""
    with path.open("rb") as handle:
        reader = skcphd.Reader(handle)
        channel_id = reader.metadata.xmltree.findtext("{*}Data/{*}Channel/{*}Identifier")
        return reader.read_signal(channel_id), reader.read_pvps(channel_id)


def backproject(signal, pvps, lattice, *, oversample=8):
    """A plain time-domain backprojector, written to be obviously correct.

    Deliberately not the operator's algorithm and deliberately not fast: it
    exists to be an independent statement of the same geometry, so it stays
    a literal transcription of "compress in range, then sum each pulse's
    contribution at the delay each pixel implies".
    """
    num_samples = signal.shape[1]
    padded = num_samples * oversample
    # Matching the simulator's convention: the positive-exponent transform,
    # which numpy spells ifft. Using fft here mirrors the image.
    compressed = np.fft.ifft(signal, n=padded, axis=1) * padded

    sc0 = float(pvps["SC0"][0])
    range_step = SPEED_OF_LIGHT / (2.0 * padded * float(pvps["SCSS"][0]))

    origin = np.array(lattice["origin"])
    axis_v = np.array(lattice["axis_v"])
    axis_u = np.array(lattice["axis_u"])
    spacing_v, spacing_u = lattice["spacing"]
    num_v, num_u = lattice["counts"]

    grid_v = np.arange(num_v)[:, None, None] * spacing_v * axis_v[None, None, :]
    grid_u = np.arange(num_u)[None, :, None] * spacing_u * axis_u[None, None, :]
    pixels = (origin + grid_v + grid_u).reshape(-1, 3)

    image = np.zeros(pixels.shape[0], dtype=complex)
    for pulse in range(signal.shape[0]):
        apc = pvps["TxPos"][pulse]
        srp = pvps["SRPPos"][pulse]
        delta_r = np.linalg.norm(pixels - apc, axis=1) - np.linalg.norm(srp - apc)
        index = delta_r / range_step
        low = np.floor(index).astype(int)
        frac = index - low
        sample = (
            compressed[pulse][low % padded] * (1.0 - frac)
            + compressed[pulse][(low + 1) % padded] * frac
        )
        # Undo the carrier phase factored out by the range transform. The
        # demodulation frequency is SC0, the first sample's frequency, not
        # the band centre — that pairing is set by the frequency axis the
        # transform ran over.
        image += sample * np.exp(1j * 4.0 * np.pi * sc0 * delta_r / SPEED_OF_LIGHT)

    return image.reshape(num_v, num_u)


def test_recipes_load_and_are_consistent():
    for path in (SMOKE_RECIPE, DEMO_RECIPE):
        recipe = load_recipe(path)
        assert recipe.waveform.range_resolution > 0.0
        assert recipe.azimuth_resolution > 0.0
        # Every target has to sit inside the lattice the recipe implies, or
        # the fixture describes a scene it cannot image.
        for target in recipe.targets:
            assert abs(target.east_m) <= recipe.image_half_extent_m
            assert abs(target.north_m) <= recipe.image_half_extent_m


def test_generated_cphd_reads_back_through_sarkit(smoke_fixture):
    result, sidecar = smoke_fixture
    signal, pvps = read_cphd(result.cphd_path)

    assert signal.shape == (
        sidecar["collection"]["num_pulses"],
        sidecar["collection"]["num_samples"],
    )
    # CF8 is two big-endian float32 per sample. The byte order is not
    # incidental — CPHD stores signal arrays big-endian regardless of host,
    # so `CPHDReaderOp` has to swap on every little-endian machine it will
    # ever run on. Asserting it here means the fixture keeps stating the
    # requirement the reader is written against.
    assert signal.dtype.itemsize == 8
    assert signal.dtype.byteorder == ">"
    assert np.all(np.isfinite(signal))
    assert np.all(pvps["SIGNAL"] == 1)
    # TxTime must be non-negative and monotonic; CPHD requires the first and
    # the collection would be nonsense without the second.
    assert pvps["TxTime"][0] >= 0.0
    assert np.all(np.diff(pvps["TxTime"]) > 0.0)
    # Spotlight: the reference point does not move, which is what lets the
    # channel declare SRPFixed.
    assert np.allclose(pvps["SRPPos"], pvps["SRPPos"][0])


def test_self_check_locates_targets_in_range(smoke_fixture):
    result, _ = smoke_fixture
    checks = result.self_check
    assert checks["worst_range_error_m"] < 0.1 * checks["range_resolution_m"]


def test_backprojection_finds_targets_where_the_sidecar_says(smoke_fixture):
    """The assertion the rest of the SAR suite rests on.

    An independent backprojector, the sidecar's own lattice, and the recipe's
    target list have to agree on a pixel. A sign error anywhere in the chain
    — the phase model, `Global/SGN`, the image axes, the ground-range
    projection — moves a target somewhere else, and the off-axis targets are
    placed so that "somewhere else" is never another target's pixel.
    """
    result, sidecar = smoke_fixture
    signal, pvps = read_cphd(result.cphd_path)
    image = np.abs(backproject(signal, pvps, sidecar["sample_geometry"]))

    for target in sidecar["targets"]:
        expected_v, expected_u = target["image_index_vu_rounded"]
        window = 4
        v0, u0 = max(expected_v - window, 0), max(expected_u - window, 0)
        patch = image[v0 : expected_v + window + 1, u0 : expected_u + window + 1]
        assert patch.size > 0, f"{target['name']} predicted outside the lattice"
        offset = np.unravel_index(int(np.argmax(patch)), patch.shape)
        found = (v0 + offset[0], u0 + offset[1])
        assert found == (expected_v, expected_u), (
            f"{target['name']} peaked at {found}, sidecar predicted "
            f"{(expected_v, expected_u)}"
        )

    # Relative brightness tracks the authored amplitudes. The tolerance is
    # loose because off-centre targets lose a little to the interpolator,
    # which is a property of this reference backprojector rather than of the
    # fixture.
    peaks = {
        target["name"]: image[tuple(target["image_index_vu_rounded"])]
        for target in sidecar["targets"]
    }
    brightest = max(peaks.values())
    for target in sidecar["targets"]:
        expected = target["amplitude"] / sidecar["targets"][0]["amplitude"]
        assert peaks[target["name"]] / brightest == pytest.approx(expected, rel=0.15)


def test_regeneration_is_deterministic(tmp_path):
    """Same recipe, same bytes.

    Recipes are committed and fixtures are not, so a fixture regenerated on
    another machine has to be the one the assertions were written against.
    This covers a single toolchain only; cross-platform stability is an open
    question in INPUT_DATA.md.
    """
    first = generate(SMOKE_RECIPE, tmp_path / "first")
    second = generate(SMOKE_RECIPE, tmp_path / "second")
    assert first.cphd_path.read_bytes() == second.cphd_path.read_bytes()
    assert first.sidecar_path.read_text() == second.sidecar_path.read_text()
