# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPHD 1.1.0 metadata construction and file writing.

The XML is built explicitly rather than filled into a vendored template. It
is more code, but the fields a synthetic collection needs are exactly the
fields `CPHDReaderOp` will parse, and building them here means every one has
a stated origin instead of an inherited value nobody chose.

``CF8`` with no ``AmpSF`` by default, matching the pinned Atlanta collection,
so the reader takes one decode path for both the fixture and the first real
collect. A recipe may instead ask for ``CI4`` or ``CI2``, which is how the
integer-sample and per-vector-scale paths get covered in CI: the real
collections that use them (Capella) are gigabytes and cannot be fetched by a
test.
"""

from __future__ import annotations

from pathlib import Path

import lxml.etree
import numpy as np
import sarkit.cphd as skcphd

from .geometry import SPEED_OF_LIGHT, CollectionGeometry, ecef_to_llh
from .recipe import Recipe
from .signal import frequency_axis

CPHD_NAMESPACE = "http://api.nsgreg.nga.mil/schema/cphd/1.1.0"
CHANNEL_ID = "sim_channel_0"

# The Fourier sign convention declared in Global/SGN. Paired with the phase
# model in signal.py -- see that module's docstring before changing either.
FOURIER_SIGN = -1

# PVP fields, in the order they are laid out. Each entry is
# (name, num_words, format string). One 8-byte word per scalar. The order
# follows the schema's own sequence, which is where AmpSF belongs when a
# recipe asks for an integer format.
_PVP_BASE: list[tuple[str, int, str]] = [
    ("TxTime", 1, "F8"),
    ("TxPos", 3, "X=F8;Y=F8;Z=F8;"),
    ("TxVel", 3, "X=F8;Y=F8;Z=F8;"),
    ("RcvTime", 1, "F8"),
    ("RcvPos", 3, "X=F8;Y=F8;Z=F8;"),
    ("RcvVel", 3, "X=F8;Y=F8;Z=F8;"),
    ("SRPPos", 3, "X=F8;Y=F8;Z=F8;"),
    ("AmpSF", 1, "F8"),
    ("aFDOP", 1, "F8"),
    ("aFRR1", 1, "F8"),
    ("aFRR2", 1, "F8"),
    ("FX1", 1, "F8"),
    ("FX2", 1, "F8"),
    ("TOA1", 1, "F8"),
    ("TOA2", 1, "F8"),
    ("TDTropoSRP", 1, "F8"),
    ("SC0", 1, "F8"),
    ("SCSS", 1, "F8"),
    ("SIGNAL", 1, "I8"),
]


def pvp_layout(recipe: Recipe) -> list[tuple[str, int, str]]:
    """The PVP fields this recipe's file declares.

    AmpSF is present only for the integer formats. Emitting it as a constant 1
    for ``CF8`` would be legal but would hide a reader that ignores it, since
    ignoring a factor of one is indistinguishable from applying it.
    """
    if recipe.storage.is_integer:
        return list(_PVP_BASE)
    return [entry for entry in _PVP_BASE if entry[0] != "AmpSF"]


def quantize(signal: np.ndarray, recipe: Recipe) -> tuple[np.ndarray, np.ndarray]:
    """Scale each pulse into the integer range, returning samples and AmpSF.

    One scale factor per pulse, chosen so the loudest sample in that pulse
    lands at full scale — which is what a real sensor's AGC produces, and what
    makes AmpSF worth storing. Reading the file back and multiplying by AmpSF
    must return the original signal to within half a quantization step.
    """
    peak = recipe.storage.quantization_peak
    magnitude = np.maximum(np.abs(signal.real), np.abs(signal.imag))
    per_pulse = magnitude.max(axis=-1)
    # A silent pulse has no scale to derive; 1.0 keeps the record finite and
    # the samples zero either way.
    amp_sf = np.where(per_pulse > 0.0, per_pulse / peak, 1.0)
    scaled = np.rint(signal / amp_sf[:, None])
    dtype = np.int16 if recipe.storage.signal_format == "CI4" else np.int8
    # Field names are sarkit's, which is what its writer matches against.
    quantized = np.empty(signal.shape, dtype=np.dtype([("real", dtype), ("imag", dtype)]))
    quantized["real"] = np.clip(scaled.real, -peak, peak).astype(dtype)
    quantized["imag"] = np.clip(scaled.imag, -peak, peak).astype(dtype)
    return quantized, amp_sf


def _element(parent, tag: str, text: str | None = None, **attrib):
    node = lxml.etree.SubElement(parent, f"{{{CPHD_NAMESPACE}}}{tag}", **attrib)
    if text is not None:
        node.text = text
    return node


def _xyz(parent, tag: str, vector) -> None:
    node = _element(parent, tag)
    for name, value in zip("XYZ", np.asarray(vector, dtype=float), strict=True):
        _element(node, name, repr(float(value)))


def _latlon(parent, tag: str, lat: float, lon: float, **attrib) -> None:
    node = _element(parent, tag, **attrib)
    _element(node, "Lat", repr(float(lat)))
    _element(node, "Lon", repr(float(lon)))


def image_area_axes(geometry: CollectionGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Unit IAX / IAY vectors for the planar reference surface.

    IAX points in increasing ground range, away from the platform's ground
    track, and IAY completes a right-handed set with the local up. The
    scene's ENU authoring frame is unaffected: this pair only orients the
    image area the CPHD declares.
    """
    iax = -geometry.ground_range_axis
    iay = np.cross(geometry.up, iax)
    return iax / np.linalg.norm(iax), iay / np.linalg.norm(iay)


def build_pvps(
    recipe: Recipe,
    geometry: CollectionGeometry,
    amp_sf: np.ndarray | None = None,
) -> np.ndarray:
    """The per-vector parameter array, one record per pulse."""
    freqs = frequency_axis(recipe)
    fx1, fx2 = float(freqs[0]), float(freqs[-1])
    sample_spacing = float(freqs[1] - freqs[0])
    num_pulses = geometry.num_pulses

    # Monostatic and idealized: transmit and receive share the phase centre
    # and the pulse time. A recipe that wanted a bistatic or
    # stop-and-go-violating collection would have to say so, and none does.
    layout = pvp_layout(recipe)
    dtype = np.dtype(
        [
            (name, "f8" if fmt != "I8" else "i8", (words,) if words > 1 else ())
            for name, words, fmt in layout
        ]
    )
    pvps = np.zeros(num_pulses, dtype=dtype)
    if recipe.storage.is_integer:
        if amp_sf is None:
            raise ValueError(
                f"{recipe.storage.signal_format} needs the per-pulse AmpSF from "
                "quantize(); writing the field as zero would zero the collection"
            )
        pvps["AmpSF"] = amp_sf

    pvps["TxTime"] = geometry.tx_time
    pvps["RcvTime"] = geometry.tx_time
    pvps["TxPos"] = geometry.apc_ecef
    pvps["RcvPos"] = geometry.apc_ecef
    pvps["TxVel"] = geometry.apc_velocity
    pvps["RcvVel"] = geometry.apc_velocity
    pvps["SRPPos"] = geometry.iarp_ecef

    # No deramp residual to undo: the signal is synthesized already
    # compensated, so the along-vector Doppler and range-rate corrections are
    # identity. aFDOP = 0 with aFRR1 = 1 is the "no rescaling" case.
    pvps["aFDOP"] = 0.0
    pvps["aFRR1"] = 1.0
    pvps["aFRR2"] = 0.0

    pvps["FX1"] = fx1
    pvps["FX2"] = fx2
    pvps["SC0"] = fx1
    pvps["SCSS"] = sample_spacing

    # The saved TOA swath, symmetric about the reference point. It has to be
    # wide enough to contain every target's differential delay or the
    # collection would be declaring a swath it violates.
    toa_half_width = 0.5 / sample_spacing
    pvps["TOA1"] = -toa_half_width
    pvps["TOA2"] = toa_half_width

    pvps["TDTropoSRP"] = 0.0
    pvps["SIGNAL"] = 1
    return pvps


def build_xmltree(
    recipe: Recipe,
    geometry: CollectionGeometry,
    amp_sf: np.ndarray | None = None,
) -> lxml.etree.ElementTree:
    """Construct the CPHD 1.1.0 XML for a synthetic collection."""
    freqs = frequency_axis(recipe)
    fx1, fx2 = float(freqs[0]), float(freqs[-1])
    sample_spacing = float(freqs[1] - freqs[0])
    num_pulses = geometry.num_pulses
    num_samples = recipe.collection.num_samples
    lat, lon, hae = geometry.iarp_llh

    root = lxml.etree.Element(f"{{{CPHD_NAMESPACE}}}CPHD", nsmap={None: CPHD_NAMESPACE})

    collection_id = _element(root, "CollectionID")
    _element(collection_id, "CollectorName", "holoscan-radar-sar-sim")
    _element(collection_id, "CoreName", recipe.name)
    _element(collection_id, "CollectType", "MONOSTATIC")
    _element(_element(collection_id, "RadarMode"), "ModeType", "SPOTLIGHT")
    _element(collection_id, "Classification", "UNCLASSIFIED")
    _element(collection_id, "ReleaseInfo", "UNRESTRICTED")

    global_block = _element(root, "Global")
    _element(global_block, "DomainType", "FX")
    _element(global_block, "SGN", str(FOURIER_SIGN))
    timeline = _element(global_block, "Timeline")
    _element(timeline, "CollectionStart", "2026-01-01T00:00:00.000000Z")
    _element(timeline, "TxTime1", repr(float(geometry.tx_time[0])))
    _element(timeline, "TxTime2", repr(float(geometry.tx_time[-1])))
    fx_band = _element(global_block, "FxBand")
    _element(fx_band, "FxMin", repr(fx1))
    _element(fx_band, "FxMax", repr(fx2))
    toa_swath = _element(global_block, "TOASwath")
    _element(toa_swath, "TOAMin", repr(-0.5 / sample_spacing))
    _element(toa_swath, "TOAMax", repr(0.5 / sample_spacing))

    scene = _element(root, "SceneCoordinates")
    _element(scene, "EarthModel", "WGS_84")
    iarp = _element(scene, "IARP")
    _xyz(iarp, "ECF", geometry.iarp_ecef)
    llh_node = _element(iarp, "LLH")
    _element(llh_node, "Lat", repr(float(lat)))
    _element(llh_node, "Lon", repr(float(lon)))
    _element(llh_node, "HAE", repr(float(hae)))

    iax, iay = image_area_axes(geometry)
    planar = _element(_element(scene, "ReferenceSurface"), "Planar")
    _xyz(planar, "uIAX", iax)
    _xyz(planar, "uIAY", iay)

    extent = float(recipe.image_half_extent_m)
    image_area = _element(scene, "ImageArea")
    x1y1 = _element(image_area, "X1Y1")
    _element(x1y1, "X", repr(-extent))
    _element(x1y1, "Y", repr(-extent))
    x2y2 = _element(image_area, "X2Y2")
    _element(x2y2, "X", repr(extent))
    _element(x2y2, "Y", repr(extent))

    corners = _element(scene, "ImageAreaCornerPoints")
    for index, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1)), start=1):
        corner_ecef = geometry.iarp_ecef + extent * (sx * iax + sy * iay)
        corner_lat, corner_lon, _ = ecef_to_llh(corner_ecef)
        _latlon(corners, "IACP", corner_lat, corner_lon, index=str(index))

    layout = pvp_layout(recipe)
    num_bytes_pvp = 8 * sum(words for _, words, _ in layout)
    data = _element(root, "Data")
    _element(data, "SignalArrayFormat", recipe.storage.signal_format)
    _element(data, "NumBytesPVP", str(num_bytes_pvp))
    _element(data, "NumCPHDChannels", "1")
    channel_data = _element(data, "Channel")
    _element(channel_data, "Identifier", CHANNEL_ID)
    _element(channel_data, "NumVectors", str(num_pulses))
    _element(channel_data, "NumSamples", str(num_samples))
    _element(channel_data, "SignalArrayByteOffset", "0")
    _element(channel_data, "PVPArrayByteOffset", "0")
    _element(data, "NumSupportArrays", "0")

    channel = _element(root, "Channel")
    _element(channel, "RefChId", CHANNEL_ID)
    _element(channel, "FXFixedCPHD", "true")
    _element(channel, "TOAFixedCPHD", "true")
    _element(channel, "SRPFixedCPHD", "true")
    parameters = _element(channel, "Parameters")
    _element(parameters, "Identifier", CHANNEL_ID)
    _element(parameters, "RefVectorIndex", str(num_pulses // 2))
    _element(parameters, "FXFixed", "true")
    _element(parameters, "TOAFixed", "true")
    # Spotlight staring at the IARP: the reference point does not move, which
    # is the property the mode 1 demo depends on.
    _element(parameters, "SRPFixed", "true")
    _element(parameters, "SignalNormal", "true")
    polarization = _element(parameters, "Polarization")
    _element(polarization, "TxPol", "V")
    _element(polarization, "RcvPol", "V")
    _element(parameters, "FxC", repr(0.5 * (fx1 + fx2)))
    _element(parameters, "FxBW", repr(fx2 - fx1))
    _element(parameters, "TOASaved", repr(1.0 / sample_spacing))
    dwell_times = _element(parameters, "DwellTimes")
    _element(dwell_times, "CODId", "cod_0")
    _element(dwell_times, "DwellId", "dwell_0")

    pvp = _element(root, "PVP")
    offset = 0
    for name, words, fmt in layout:
        node = _element(pvp, name)
        _element(node, "Offset", str(offset))
        _element(node, "Size", str(words))
        _element(node, "Format", fmt)
        offset += words

    dwell = _element(root, "Dwell")
    _element(dwell, "NumCODTimes", "1")
    cod_time = _element(dwell, "CODTime")
    _element(cod_time, "Identifier", "cod_0")
    _poly2d(cod_time, "CODTimePoly", 0.0)
    _element(dwell, "NumDwellTimes", "1")
    dwell_time = _element(dwell, "DwellTime")
    _element(dwell_time, "Identifier", "dwell_0")
    _poly2d(dwell_time, "DwellTimePoly", float(geometry.tx_time[-1] - geometry.tx_time[0]))

    xmltree = lxml.etree.ElementTree(root)
    # sarkit derives every ReferenceGeometry angle from the metadata above
    # plus the PVPs, which keeps a dozen trigonometric fields from being a
    # dozen chances to disagree with the geometry actually written. It reads
    # only positions and times, so a placeholder AmpSF is enough here.
    if amp_sf is None:
        amp_sf = np.ones(num_pulses)
    root.append(skcphd.compute_reference_geometry(xmltree, build_pvps(recipe, geometry, amp_sf)))
    return xmltree


def _poly2d(parent, tag: str, constant: float) -> None:
    node = _element(parent, tag, order1="0", order2="0")
    _element(node, "Coef", repr(float(constant)), exponent1="0", exponent2="0")


def validate_xmltree(xmltree: lxml.etree.ElementTree) -> None:
    """Validate against the CPHD 1.1.0 XSD sarkit ships.

    Worth doing on every generate rather than only in tests: a schema-invalid
    fixture is a confusing reader bug later, and validation costs
    milliseconds.
    """
    schema_path = skcphd.VERSION_INFO[CPHD_NAMESPACE]["schema"]
    schema = lxml.etree.XMLSchema(lxml.etree.parse(str(schema_path)))
    if not schema.validate(xmltree):
        raise ValueError(f"generated CPHD XML failed schema validation: {schema.error_log}")


def write_cphd(
    path: str | Path,
    recipe: Recipe,
    geometry: CollectionGeometry,
    signal: np.ndarray,
) -> Path:
    """Write the signal and PVP arrays as a CPHD 1.1.0 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if recipe.storage.is_integer:
        samples, amp_sf = quantize(signal, recipe)
    else:
        samples, amp_sf = np.ascontiguousarray(signal, dtype=np.complex64), None

    xmltree = build_xmltree(recipe, geometry, amp_sf)
    validate_xmltree(xmltree)
    pvps = build_pvps(recipe, geometry, amp_sf)

    metadata = skcphd.Metadata(xmltree=xmltree)
    with path.open("wb") as handle, skcphd.Writer(handle, metadata) as writer:
        writer.write_signal(CHANNEL_ID, np.ascontiguousarray(samples))
        writer.write_pvp(CHANNEL_ID, pvps)
    return path


def slant_range_swath(recipe: Recipe) -> float:
    """Unambiguous slant-range extent implied by the frequency sampling."""
    freqs = frequency_axis(recipe)
    return SPEED_OF_LIGHT / (2.0 * float(freqs[1] - freqs[0]))
