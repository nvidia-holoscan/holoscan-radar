# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python parity tests for ESPRITEstimatorOp."""

from __future__ import annotations

import numpy as np
import pytest

holoscan = pytest.importorskip("holoscan")
cupy = pytest.importorskip("cupy")

from utils.angle_helpers import (  # noqa: E402
    direction_cosine_convention,
    make_planar_ura_geometry,
    make_steering_snapshot,
    make_ula_geometry,
    radar_boresight_convention,
)
from utils.harness import run_operator_multi, single_map, to_device_array  # noqa: E402

from holoscan.radar import (  # noqa: E402
    AngleConvention,
    AngleConventionPreset,
    ElevationReference,
    ESPRITEstimatorOpF32,
    ESPRITEstimatorOpF64,
    ShiftInvariantSubarrayPair,
    SignedAxis,
)

pytestmark = pytest.mark.requires_gpu


def make_block(geom, conv, az, el, snapshots, noise, seed):
    manifold = make_steering_snapshot(geom, conv, az, el)
    rng = np.random.default_rng(seed)
    result = np.empty((geom.num_elements(), snapshots), dtype=np.complex128)
    for sample in range(snapshots):
        phase = 2.0 * np.pi * sample / snapshots
        gain = np.exp(1j * phase)
        result[:, sample] = manifold * gain
        result[:, sample] += rng.normal(0, noise, geom.num_elements())
        result[:, sample] += 1j * rng.normal(0, noise, geom.num_elements())
    return result


def make_block_two(geom, conv, az0, az1, snapshots, noise, seed, *, coherent=False):
    manifold0 = make_steering_snapshot(geom, conv, az0, 0.0)
    manifold1 = make_steering_snapshot(geom, conv, az1, 0.0)
    rng = np.random.default_rng(seed)
    result = np.empty((geom.num_elements(), snapshots), dtype=np.complex128)
    for sample in range(snapshots):
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        phase1 = phase0 if coherent else rng.uniform(0.0, 2.0 * np.pi)
        result[:, sample] = manifold0 * np.exp(1j * phase0) + manifold1 * np.exp(1j * phase1)
        result[:, sample] += rng.normal(0, noise, geom.num_elements())
        result[:, sample] += 1j * rng.normal(0, noise, geom.num_elements())
    return result


def run_esprit(snapshot, *, op_cls=ESPRITEstimatorOpF64, **kwargs):
    dtype = np.complex64 if op_cls is ESPRITEstimatorOpF32 else np.complex128
    device = to_device_array(np.asarray(snapshot, dtype=dtype), dtype=dtype)

    def make_op(app):
        return op_cls(app, name="op_under_test", **kwargs)

    outputs = run_operator_multi(
        make_op,
        single_map(device, "snapshot"),
        input_port="snapshot",
        output_ports=[
            "aoa_estimates",
            "doa_multi",
            "source_count",
            "eigenvalues",
        ],
    )
    return {
        "aoa": outputs["aoa_estimates"][0],
        "doa_multi": outputs["doa_multi"][0],
        "source_count": outputs["source_count"][0],
        "eigenvalues": outputs["eigenvalues"][0],
    }


def make_block_two_azel(geom, conv, az0, el0, az1, el1, snapshots, noise, seed):
    manifold0 = make_steering_snapshot(geom, conv, az0, el0)
    manifold1 = make_steering_snapshot(geom, conv, az1, el1)
    rng = np.random.default_rng(seed)
    result = np.empty((geom.num_elements(), snapshots), dtype=np.complex128)
    for sample in range(snapshots):
        phase0 = rng.uniform(0.0, 2.0 * np.pi)
        phase1 = rng.uniform(0.0, 2.0 * np.pi)
        result[:, sample] = manifold0 * np.exp(1j * phase0) + manifold1 * np.exp(1j * phase1)
        result[:, sample] += rng.normal(0, noise, geom.num_elements())
        result[:, sample] += 1j * rng.normal(0, noise, geom.num_elements())
    return result


def assert_two(output, expected0, expected1, tolerance):
    actual = np.sort(output["doa_multi"][0, :2, 0])
    np.testing.assert_allclose(actual, np.sort([expected0, expected1]), atol=tolerance)


def test_RecoversUraSingleSourceAzEl():
    k_ny, k_nz, k_s = 6, 6, 256
    truth_az, truth_el = 0.15, -0.10
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, truth_el, k_s, 0.02, 7)
    out = run_esprit(
        block.reshape(1, geom.num_elements(), k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.03)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.03)


def test_RecoversUraTwoSourcesAzElPairing():
    k_ny, k_nz, k_s = 8, 8, 512
    # Distinct in BOTH az and el so a mispaired result is detectable.
    az0, el0 = 0.20, 0.12
    az1, el1 = -0.15, -0.10
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block_two_azel(geom, conv, az0, el0, az1, el1, k_s, 0.01, 79)
    out = run_esprit(
        block.reshape(1, geom.num_elements(), k_s),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=2,
    )
    assert out["doa_multi"].shape == (1, 2, 2)
    assert int(out["source_count"][0]) == 2

    dm = out["doa_multi"][0]

    def near(pair, taz, tel):
        return abs(pair[0] - taz) < 0.04 and abs(pair[1] - tel) < 0.04

    matched = (near(dm[0], az0, el0) and near(dm[1], az1, el1)) or (
        near(dm[0], az1, el1) and near(dm[1], az0, el0)
    )
    assert matched, f"pair0={dm[0]} pair1={dm[1]}"


def test_RecoversUraSingleSourceAzElFloat():
    k_ny, k_nz, k_s = 6, 6, 256
    truth_az, truth_el = -0.12, 0.08
    conv = radar_boresight_convention()
    geom = make_planar_ura_geometry(k_ny, k_nz, 0.5, 1.0)
    block = make_block(geom, conv, truth_az, truth_el, k_s, 0.02, 17)
    out = run_esprit(
        block.reshape(1, geom.num_elements(), k_s),
        op_cls=ESPRITEstimatorOpF32,
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert out["aoa"].shape == (1, 2)
    assert out["aoa"][0, 0] == pytest.approx(truth_az, abs=0.03)
    assert out["aoa"][0, 1] == pytest.approx(truth_el, abs=0.03)


@pytest.mark.parametrize("op_cls", [ESPRITEstimatorOpF32, ESPRITEstimatorOpF64])
def test_RecoversKnownDoaUlaFixedOneSource(op_cls):
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.08, 0.0, 64, 0.02, 3)
    output = run_esprit(
        block[None],
        op_cls=op_cls,
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    assert output["aoa"].shape == (1, 2)
    assert output["aoa"][0, 0] == pytest.approx(0.08, abs=0.03)
    assert output["aoa"][0, 1] == pytest.approx(0.0, abs=1e-8)


def test_CustomConventionMapsSignedAxes():
    geom = make_ula_geometry(8, 0.5, 1.0)
    radar = radar_boresight_convention()
    custom = AngleConvention()
    custom.preset = AngleConventionPreset.CUSTOM
    custom.boresight_axis = SignedAxis.PLUS_X
    custom.azimuth_positive_axis = SignedAxis.MINUS_Y
    custom.elevation_positive_axis = SignedAxis.MINUS_Z
    block = make_block(geom, radar, 0.18, 0.0, 64, 0.01, 176)

    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=custom,
        num_sources=1,
    )

    assert output["aoa"][0, 0] == pytest.approx(-0.18, abs=0.01)
    assert output["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_CustomConventionSupportsZenithReference():
    geom = make_ula_geometry(8, 0.5, 1.0)
    radar = radar_boresight_convention()
    custom = AngleConvention()
    custom.preset = AngleConventionPreset.CUSTOM
    custom.elevation_reference = ElevationReference.ZENITH
    block = make_block(geom, radar, 0.1, 0.0, 64, 0.01, 177)

    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=custom,
        num_sources=1,
    )

    assert output["aoa"][0, 0] == pytest.approx(0.1, abs=0.01)
    assert output["aoa"][0, 1] == pytest.approx(np.pi / 2.0, abs=1e-9)


def test_RejectsDegenerateCustomConvention():
    geom = make_ula_geometry(4, 0.5, 1.0)
    radar = radar_boresight_convention()
    custom = AngleConvention()
    custom.preset = AngleConventionPreset.CUSTOM
    custom.azimuth_positive_axis = custom.boresight_axis
    block = make_block(geom, radar, 0.1, 0.0, 8, 0.01, 178)

    with pytest.raises(ValueError, match="three distinct principal axes"):
        run_esprit(
            block[None],
            array_geometry=geom,
            angle_convention=custom,
            num_sources=1,
        )


def test_RecoversTwoSources():
    geom = make_ula_geometry(12, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block_two(geom, conv, -0.15, 0.20, 256, 0.02, 71)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=2,
    )
    assert int(output["source_count"][0]) == 2
    assert_two(output, -0.15, 0.20, 0.03)
    assert output["aoa"][0, 0] == output["doa_multi"][0, 0, 0]


def test_ResolvesCloselySpacedSources():
    geom = make_ula_geometry(16, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block_two(geom, conv, -0.035, 0.035, 512, 0.003, 73)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=2,
    )
    assert_two(output, -0.035, 0.035, 0.012)


def test_HighSnrLargeArrayEstimateRemainsFinite():
    geom = make_ula_geometry(32, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.17, 0.0, 128, 1e-6, 74)
    output = run_esprit(block[None], array_geometry=geom, angle_convention=conv, num_sources=1)
    assert np.isfinite(output["aoa"][0, 0])
    assert output["aoa"][0, 0] == pytest.approx(0.17, abs=1e-4)


def test_MaxSourcesCapsReportedDoas():
    geom = make_ula_geometry(12, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block_two(geom, conv, -0.15, 0.20, 256, 0.02, 78)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        num_sources=2,
        max_sources=1,
    )
    assert output["doa_multi"].shape == (1, 1, 2)
    assert int(output["source_count"][0]) == 1


def test_AutoNumSourcesRecoversOneSource():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.1, 0.0, 64, 0.02, 9)
    output = run_esprit(block[None], array_geometry=geom, angle_convention=conv, max_sources=4)
    assert int(output["source_count"][0]) == 1
    assert output["aoa"][0, 0] == pytest.approx(0.1, abs=0.02)


def test_RankDeficientAutoFallsBackToOneSource():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.1, 0.0, 4, 0.02, 12)
    output = run_esprit(block[None], array_geometry=geom, angle_convention=conv, max_sources=4)
    assert int(output["source_count"][0]) == 1


def test_ExplicitSubarrayPairWorks():
    geom = make_ula_geometry(4, 0.5, 1.0)
    geom.geometry_type = None
    geom.element_positions = [0, 0, 0, 0, 0.5, 0, 0, 1.5, 0, 0, 2.0, 0]
    pair = ShiftInvariantSubarrayPair()
    pair.reference_indices = [0, 1]
    pair.shifted_indices = [2, 3]
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.08, 0.0, 128, 0.01, 81)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        subarray_spec=pair,
        num_sources=1,
    )
    assert output["aoa"][0, 0] == pytest.approx(0.08, abs=0.02)


def test_ExplicitSubarrayPairRecoversTwoSources():
    geom = make_ula_geometry(8, 0.5, 1.0)
    geom.geometry_type = None
    reference_y = [0.0, 0.17, 0.39, 0.68]
    geom.element_positions = [
        coordinate
        for y in reference_y + [y + 0.5 for y in reference_y]
        for coordinate in (0.0, y, 0.0)
    ]
    pair = ShiftInvariantSubarrayPair()
    pair.reference_indices = [0, 1, 2, 3]
    pair.shifted_indices = [4, 5, 6, 7]
    conv = radar_boresight_convention()
    block = make_block_two(geom, conv, -0.16, 0.21, 256, 0.01, 206)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        subarray_spec=pair,
        num_sources=2,
        max_sources=2,
    )
    assert int(output["source_count"][0]) == 2
    assert_two(output, -0.16, 0.21, 0.03)


def test_RejectsInvalidAutoGeometry():
    conv = radar_boresight_convention()
    geom = make_ula_geometry(5, 0.5, 1.0)
    positions = list(geom.element_positions)
    positions[10] = 1.65
    geom.element_positions = positions
    block = make_block(geom, conv, 0.1, 0.0, 16, 0.02, 75)
    with pytest.raises(ValueError):
        run_esprit(block[None], array_geometry=geom, angle_convention=conv)


def test_RejectsUlaParallelToBoresight():
    geom = make_ula_geometry(4, 0.5, 1.0)
    geom.element_positions = [
        coordinate for element in range(4) for coordinate in (0.5 * element, 0.0, 0.0)
    ]
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.1, 0.0, 16, 0.02, 75)
    with pytest.raises(ValueError):
        run_esprit(block[None], array_geometry=geom, angle_convention=conv)


def test_RejectsExplicitSubarrayWithSmoothing():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    pair = ShiftInvariantSubarrayPair()
    pair.reference_indices = [0, 1, 2, 3]
    pair.shifted_indices = [1, 2, 3, 4]
    block = make_block(geom, conv, 0.1, 0.0, 32, 0.02, 5)
    with pytest.raises(ValueError):
        run_esprit(
            block[None],
            array_geometry=geom,
            angle_convention=conv,
            subarray_spec=pair,
            spatial_smoothing=[2],
        )


@pytest.mark.parametrize(
    ("reference", "shifted"),
    [
        ([0, 1, 2], [1, 2]),
        ([0, 1], [1, 3]),
        ([0, 1], [1, 6]),
    ],
)
def test_RejectsMalformedExplicitSubarray(reference, shifted):
    geom = make_ula_geometry(6, 0.5, 1.0)
    conv = radar_boresight_convention()
    pair = ShiftInvariantSubarrayPair()
    pair.reference_indices = reference
    pair.shifted_indices = shifted
    block = make_block(geom, conv, 0.1, 0.0, 16, 0.02, 84)
    with pytest.raises(ValueError):
        run_esprit(
            block[None],
            array_geometry=geom,
            angle_convention=conv,
            subarray_spec=pair,
        )


def test_EigenvaluesAscendingAndUnusedRowsNan():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.1, 0.0, 128, 0.02, 91)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
        max_sources=3,
    )
    assert np.all(np.diff(output["eigenvalues"][0]) >= -1e-9)
    assert np.all(output["eigenvalues"] >= -1e-6)
    assert np.all(np.isnan(output["doa_multi"][0, 1:]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_sources": 0},
        {"max_sources": 65},
        {"source_count_pfa": 0.0},
        {"source_count_pfa": 1.0},
        {"num_sources": 4},
    ],
)
def test_RejectsInvalidScalarParameters(kwargs):
    geom = make_ula_geometry(4, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.0, 0.0, 8, 0.02, 4)
    with pytest.raises(ValueError):
        run_esprit(block[None], array_geometry=geom, angle_convention=conv, **kwargs)


def test_RejectsSnapshotShapeAndDtypeMismatch():
    geom = make_ula_geometry(4, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.0, 0.0, 8, 0.02, 4)
    with pytest.raises(ValueError):
        run_esprit(block, array_geometry=geom, angle_convention=conv)
    with pytest.raises(ValueError):
        run_esprit(block[None, :3], array_geometry=geom, angle_convention=conv)

    device = to_device_array(block[None].astype(np.complex64), dtype=np.complex64)

    def make_op(app):
        return ESPRITEstimatorOpF64(app, array_geometry=geom, angle_convention=conv, name="op")

    with pytest.raises(ValueError):
        run_operator_multi(
            make_op,
            single_map(device, "snapshot"),
            input_port="snapshot",
            output_ports=["aoa_estimates"],
            drain_ports=["doa_multi", "source_count", "eigenvalues"],
        )


def test_FloatDoubleAgreeOnDoa():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.09, 0.0, 64, 0.05, 15)
    kwargs = dict(array_geometry=geom, angle_convention=conv, num_sources=1)
    single = run_esprit(block[None], op_cls=ESPRITEstimatorOpF32, **kwargs)
    double = run_esprit(block[None], **kwargs)
    assert single["aoa"][0, 0] == pytest.approx(float(double["aoa"][0, 0]), abs=1e-2)


def test_EmptyItemsProducesEmptyOutputs():
    geom = make_ula_geometry(4, 0.5, 1.0)
    output = run_esprit(
        np.empty((0, 4, 4), dtype=np.complex128),
        array_geometry=geom,
        angle_convention=radar_boresight_convention(),
        max_sources=3,
    )
    assert output["aoa"].shape == (0, 2)
    assert output["doa_multi"].shape == (0, 3, 2)
    assert output["eigenvalues"].shape == (0, 4)


def test_DirectionCosineMapsUlaProjection():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = direction_cosine_convention()
    block = make_block(geom, conv, 0.23, 0.0, 64, 0.01, 76)
    output = run_esprit(block[None], array_geometry=geom, angle_convention=conv, num_sources=1)
    assert output["aoa"][0, 0] == pytest.approx(0.23, abs=0.01)
    assert output["aoa"][0, 1] == pytest.approx(0.0, abs=1e-9)


def test_DirectionCosineNearEndfirePreservesSign():
    geom = make_ula_geometry(12, 0.5, 1.0)
    conv = direction_cosine_convention()
    blocks = [
        make_block(geom, conv, direction, 0.0, 128, 0.005, seed)
        for direction, seed in [(-0.92, 79), (0.92, 80)]
    ]
    output = run_esprit(
        np.stack(blocks),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    np.testing.assert_allclose(output["aoa"][:, 0], [-0.92, 0.92], atol=0.01)


def test_AllInvalidMaskBlanksDoa():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.1, 0.0, 16, 0.02, 33)
    snapshot = to_device_array(block[None], dtype=np.complex128)
    mask = to_device_array(np.zeros((1, 16), dtype=np.uint8), dtype=np.uint8)

    def make_op(app):
        return ESPRITEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=1,
            max_sources=2,
            name="op_under_test",
        )

    outputs = run_operator_multi(
        make_op,
        single_map(snapshot, "snapshot"),
        input_port="snapshot",
        output_ports=["aoa_estimates", "doa_multi", "source_count"],
        extra_inputs={"is_valid": [single_map(mask, "is_valid")]},
        drain_ports=["eigenvalues"],
    )
    assert np.all(np.isnan(outputs["aoa_estimates"][0]))
    assert np.all(np.isnan(outputs["doa_multi"][0]))
    assert int(outputs["source_count"][0][0]) == 0


def test_IsValidMaskExcludesWrongSnapshots():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    good = make_block(geom, conv, 0.1, 0.0, 16, 0.02, 31)
    bad = make_block(geom, conv, -0.2, 0.0, 16, 0.02, 32)
    block = np.concatenate([good, bad], axis=1)
    mask = np.zeros((1, 32), dtype=np.uint8)
    mask[0, :16] = 1
    snapshot = to_device_array(block[None], dtype=np.complex128)
    mask_device = to_device_array(mask, dtype=np.uint8)

    def make_op(app):
        return ESPRITEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=1,
            max_sources=2,
            name="op_under_test",
        )

    outputs = run_operator_multi(
        make_op,
        single_map(snapshot, "snapshot"),
        input_port="snapshot",
        output_ports=["aoa_estimates"],
        extra_inputs={"is_valid": [single_map(mask_device, "is_valid")]},
        drain_ports=["doa_multi", "source_count", "eigenvalues"],
    )
    assert outputs["aoa_estimates"][0][0, 0] == pytest.approx(0.1, abs=0.03)


def test_MultipleItemsRecoverIndependentDoa():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    first = make_block(geom, conv, 0.11, 0.0, 48, 0.02, 21)
    second = make_block(geom, conv, -0.07, 0.0, 48, 0.02, 22)
    output = run_esprit(
        np.stack([first, second]),
        array_geometry=geom,
        angle_convention=conv,
        num_sources=1,
    )
    np.testing.assert_allclose(output["aoa"][:, 0], [0.11, -0.07], atol=0.02)


def test_DeterministicAcrossRuns():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.09, 0.0, 64, 0.02, 171)
    kwargs = dict(array_geometry=geom, angle_convention=conv, num_sources=1)
    first = run_esprit(block[None], **kwargs)
    second = run_esprit(block[None], **kwargs)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_SpatialSmoothingRecoversCoherentSources():
    geom = make_ula_geometry(10, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block_two(geom, conv, -0.30, 0.30, 64, 0.02, 70, coherent=True)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        spatial_smoothing=[4],
        forward_backward=True,
        num_sources=2,
        max_sources=2,
    )
    assert output["eigenvalues"].shape == (1, 7)
    assert_two(output, -0.30, 0.30, 0.04)


def test_ForwardOnlySmoothingRecoversCoherentSources():
    geom = make_ula_geometry(10, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block_two(geom, conv, -0.30, 0.30, 128, 0.01, 205, coherent=True)
    output = run_esprit(
        block[None],
        array_geometry=geom,
        angle_convention=conv,
        spatial_smoothing=[4],
        forward_backward=False,
        num_sources=2,
        max_sources=2,
    )
    assert int(output["source_count"][0]) == 2
    assert_two(output, -0.30, 0.30, 0.04)


def test_FixedSourceCountWorksWithFewSnapshots():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.12, 0.0, 4, 0.001, 208)
    output = run_esprit(block[None], array_geometry=geom, angle_convention=conv, num_sources=1)
    assert int(output["source_count"][0]) == 1
    assert output["aoa"][0, 0] == pytest.approx(0.12, abs=0.02)


def test_LiveArrayGeometryStickyOverride():
    geom = make_ula_geometry(4, 0.5, 1.0)
    changed = make_ula_geometry(4, 0.5, 0.8)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.05, 0.0, 16, 0.05, 61)
    device = to_device_array(block[None], dtype=np.complex128)
    frame = single_map(device, "snapshot")

    def make_op(app):
        return ESPRITEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=1,
            name="op_under_test",
        )

    outputs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=["aoa_estimates"],
        overrides={"array_geometry": [geom, changed, geom]},
        drain_ports=["doa_multi", "source_count", "eigenvalues"],
    )["aoa_estimates"]
    assert outputs[0][0, 0] != pytest.approx(float(outputs[1][0, 0]), abs=1e-12)
    np.testing.assert_array_equal(outputs[0], outputs[2])


def test_LiveSourceCountPfaStickyOverride():
    geom = make_ula_geometry(8, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block(geom, conv, 0.08, 0.0, 40, 0.45, 191)
    device = to_device_array(block[None], dtype=np.complex128)
    frame = single_map(device, "snapshot")

    def make_op(app):
        return ESPRITEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=-1,
            source_count_pfa=1e-4,
            max_sources=4,
            name="op_under_test",
        )

    outputs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=["source_count"],
        overrides={"source_count_pfa": [1e-4, 0.5]},
        drain_ports=["aoa_estimates", "doa_multi", "eigenvalues"],
    )["source_count"]
    tight = int(outputs[0][0])
    loose = int(outputs[1][0])
    assert tight <= loose
    assert loose == int(outputs[2][0])


@pytest.mark.parametrize(
    ("port", "overrides", "output_port"),
    [
        ("num_sources", [1, 2], "source_count"),
        ("max_sources", [1, 3], "doa_multi"),
        ("spatial_smoothing", [[], [2]], "eigenvalues"),
    ],
)
def test_LiveOverridesAreSticky(port, overrides, output_port):
    geom = make_ula_geometry(12, 0.5, 1.0)
    conv = radar_boresight_convention()
    block = make_block_two(geom, conv, -0.15, 0.20, 128, 0.02, 101)
    device = to_device_array(block[None], dtype=np.complex128)
    frame = single_map(device, "snapshot")

    def make_op(app):
        return ESPRITEstimatorOpF64(
            app,
            array_geometry=geom,
            angle_convention=conv,
            num_sources=1 if port == "num_sources" else 2,
            max_sources=1 if port == "max_sources" else 4,
            name="op_under_test",
        )

    outputs = run_operator_multi(
        make_op,
        [frame, frame, frame],
        input_port="snapshot",
        output_ports=[output_port],
        overrides={port: overrides},
        drain_ports=[
            name
            for name in ["aoa_estimates", "doa_multi", "source_count", "eigenvalues"]
            if name != output_port
        ],
    )[output_port]
    assert len(outputs) == 3
    assert outputs[1].shape == outputs[2].shape
    if port == "num_sources":
        assert int(outputs[0][0]) == 1
        assert int(outputs[1][0]) == 2
        assert int(outputs[2][0]) == 2
    elif port == "max_sources":
        assert outputs[0].shape == (1, 1, 2)
        assert outputs[1].shape == (1, 3, 2)
    else:
        assert outputs[0].shape == (1, 12)
        assert outputs[1].shape == (1, 11)
