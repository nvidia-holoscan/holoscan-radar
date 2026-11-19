# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HSDK 5 registry and native constructor checks for Python operators."""

import ast
from pathlib import Path

import holoscan_radar as radar
import pytest

import holoscan as hs


def test_facade_stub_matches_public_operators():
    stub_path = Path(radar.__file__).with_suffix(".pyi")
    stub = ast.parse(stub_path.read_text(encoding="utf-8"))
    declared = {
        node.target.id
        for node in stub.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.annotation, ast.Name)
        and node.annotation.id == "RadarOperator"
    }
    assert declared == set(radar.__all__)


def test_all_public_precisions_are_registered():
    registered = {
        item.name
        for item in hs.registry.operators()
        if item.name.startswith("holoscan.radar.") and "._app." not in item.name
    }
    assert len(registered) == 67
    assert registered == {f"holoscan.radar.{name}" for name in radar.__all__}
    assert "holoscan.radar.SubarraySelectionOpF32" in registered
    assert "holoscan.radar.SubarraySelectionOpF64" in registered


@pytest.mark.parametrize("name", radar.__all__)
def test_public_facade_constructs_both_precisions(name):
    extents = {
        "max_samples_per_pulse": 8,
        "max_pulses": 8,
        "max_image_v": 8,
        "max_image_u": 8,
        "max_samples_per_chirp": 8,
        "max_chirps": 8,
        "max_range_bins": 8,
        "max_doppler_bins": 8,
        "max_rx": 8,
        "max_tx": 8,
        "max_interleaved_chirps": 8,
        "max_virtual_channels": 8,
        "max_channels": 8,
        "max_az": 8,
        "max_el": 8,
        "max_detections": 8,
        "max_points": 8,
        "max_items": 8,
        "max_snapshots": 8,
        "max_features": 8,
        "max_sources": 8,
        "max_clusters": 8,
        "max_subarray_channels": 8,
    }
    graph = hs.Graph(name)
    params = {"features": ["range"]} if name.startswith("FeatureAssembly") else {}
    handle = getattr(radar, name)(graph, "operator", extents=extents, **params)
    assert handle.name == "operator"


def test_subarray_ports_and_graph_constructor():
    operator_type = hs.registry.resolve("holoscan.radar.SubarraySelectionOpF32")
    assert set(operator_type.ports) == {
        "snapshot",
        "is_valid",
        "array_geometry_in",
        "subarray",
        "snapshot_out",
        "array_geometry",
        "is_valid_out",
    }
    graph = hs.Graph("subarray-selection")
    handle = radar.SubarraySelectionOpF32(
        graph,
        "select",
        extents={"max_items": 4, "max_snapshots": 8, "max_subarray_channels": 4},
        min_subarray_size=2,
    )
    assert handle.name == "select"
    assert set(handle.ports) == set(operator_type.ports)


def test_unsupported_live_enum_port_is_rejected_without_changing_cpp_contract():
    operator_type = hs.registry.resolve("holoscan.radar.FMCWRangeProcessorOpF32")
    assert "fft_norm" not in operator_type.ports
    assert "nfft" in operator_type.ports
    graph = hs.Graph("range-enum")
    handle = radar.FMCWRangeProcessorOpF32(
        graph,
        "range",
        extents={"max_chirps": 2, "max_samples_per_chirp": 8, "max_range_bins": 8},
    )
    with pytest.raises(AttributeError, match="fft_norm"):
        _ = handle.fft_norm


def test_bad_extents_are_rejected_by_native_thunk():
    graph = hs.Graph("bad-extents")
    with pytest.raises(hs.errors.InvalidArgumentError, match="positive integer"):
        radar.SubarraySelectionOpF32(
            graph,
            "select",
            extents={"max_items": 0, "max_snapshots": 8, "max_subarray_channels": 4},
        )


def test_feature_assembly_constructor_features():
    graph = hs.Graph("features")
    handle = radar.FeatureAssemblyOpF32(
        graph,
        "assemble",
        extents={"max_points": 8, "max_features": 2},
        features=["range", "velocity"],
    )
    assert handle.name == "assemble"
    with pytest.raises(TypeError, match="requires features"):
        radar.FeatureAssemblyOpF32(graph, "missing", extents={"max_points": 8})


@pytest.mark.parametrize("selection", [[], [0, 1, 2]])
def test_subarray_geometry_exact_join_to_beamscan(selection):
    graph = hs.Graph("subarray-geometry-join")
    extents = {
        "max_items": 4,
        "max_snapshots": 4,
        "max_subarray_channels": 3,
        "max_channels": 3,
        "max_az": 3,
        "max_el": 1,
    }
    geometry = {
        "element_positions": [0, 0, 0, 0, 0.5, 0, 0, 1, 0],
        "wavelength": 1,
        "geometry_type": "ula",
    }
    select = radar.SubarraySelectionOpF32(
        graph, "select", extents=extents, array_geometry=geometry, subarray=selection
    )
    beam = radar.BeamscanEstimatorOpF32(
        graph,
        "beam",
        extents=extents,
        array_geometry=geometry,
        az_coordinates=[-0.1, 0, 0.1],
        el_coordinates=[0],
    )
    graph.add_flow(select.snapshot_out, beam.snapshot)
    graph.add_flow(select.is_valid_out, beam.is_valid)
    graph.add_flow(select.array_geometry, beam.array_geometry)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    options = hs.CompileOptions()
    for operator_name, ports in (
        ("select", ("snapshot_out", "is_valid_out")),
        ("beam", ("aoa_estimates", "angle_spectrum")),
    ):
        for port in ports:
            options.deployment.bind_tensor_output_device(
                operator_path=operator_name,
                output_port=port,
                device=hs.execution.DeviceId(0),
            )
    assert hs.compile(graph, options).ok


def test_descriptor_validation_and_complex_pair_conversion():
    config_path = (
        Path(__file__).resolve().parents[2] / "applications/fmcw/siso/config/siso_smoke.yaml"
    )
    config = radar._radar._load_siso(str(config_path))
    waveform = config["range_processor"]["waveform"]
    waveform["reference_samples"] = [1 + 2j]
    graph = hs.Graph("complex-pair")
    handle = radar.FMCWRangeProcessorOpF32(
        graph, "range", extents=config["extents"], **config["range_processor"]
    )
    assert handle.name == "range"
    with pytest.raises(hs.errors.InvalidArgumentError, match="wavelength"):
        radar.SubarraySelectionOpF32(
            graph,
            "invalid_geometry",
            extents={"max_items": 4, "max_snapshots": 4, "max_subarray_channels": 3},
            array_geometry={
                "element_positions": [0, 0, 0, 0, 0.5, 0, 0, 1, 0],
                "wavelength": 0,
                "geometry_type": "ula",
            },
        )


def test_ddm_phase_code_complex_samples_are_interleaved():
    graph = hs.Graph("ddm-complex-codes")
    handle = radar.DDMDenseDemultiplexOpF32(
        graph,
        "demultiplex",
        extents={"max_tx": 2, "max_rx": 2, "max_doppler_bins": 8, "max_range_bins": 8},
        ddm_code={
            "num_tx": 2,
            "phase_codes": [[1 + 0j, 1j], [1 + 0j, -1j]],
            "guard_bands": 0,
        },
    )
    assert handle.name == "demultiplex"
