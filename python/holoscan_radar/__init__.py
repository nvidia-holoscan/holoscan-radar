# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Radar operators for HSDK 5 Graph authoring.

Pooled operators have the call shape ``OpF32(graph, name, *, extents, **params)``
(or ``OpF64``; geometry operators name both precisions explicitly).
``TensorDumpOp(graph, name, **params)`` is a dtype-independent sink. ``extents`` is a mapping of ``max_*`` pool limits. Descriptor
parameters use their Radar YAML spelling as mappings. Complex reference samples
cross the native boundary as ``[real, imaginary]`` pairs. DDM phase-code
matrices accept real or complex samples and become flat interleaved pairs. The C++ Radar codecs
validate descriptor values before the operator is added to the Graph.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Mapping
from typing import Any

from . import _radar as _radar


def _config_value(value: object) -> object:
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, enum.Enum):
        return _config_value(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _config_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config_value(item) for item in value]
    return value


def _ddm_code_value(value: object) -> object:
    if not isinstance(value, Mapping):
        return value
    samples = value.get("phase_codes")
    if not isinstance(samples, (list, tuple)) or not samples:
        return value
    if all(isinstance(sample, (int, float, Mapping)) for sample in samples):
        return value  # The C++ YAML representation is already interleaved.
    flattened: list[float] = []

    def append_samples(items: object) -> None:
        if isinstance(items, (list, tuple)):
            for item in items:
                append_samples(item)
        elif isinstance(items, complex):
            flattened.extend((items.real, items.imag))
        elif isinstance(items, (int, float)):
            flattened.extend((float(items), 0.0))
        else:
            raise TypeError("ddm_code.phase_codes must contain real or complex samples")

    append_samples(samples)
    return {**value, "phase_codes": flattened}


def _operator_call(
    type_name: str, graph: Any, name: str, *, extents: Mapping[str, int], **params: object
) -> Any:
    if not isinstance(extents, Mapping):
        raise TypeError("extents must be a mapping of max_* fields")
    if "ddm_code" in params:
        params = {**params, "ddm_code": _ddm_code_value(params["ddm_code"])}
    payload: dict[str, object] = {
        "extents": _config_value(extents),
        "params": _config_value(params),
    }
    if type_name.endswith("FeatureAssemblyOpF32") or type_name.endswith("FeatureAssemblyOpF64"):
        if "features" not in params:
            raise TypeError("FeatureAssemblyOp requires features")
        if set(params) != {"features"}:
            raise TypeError("FeatureAssemblyOp accepts only the features constructor argument")
        payload["features"] = _config_value(params["features"])
        payload["params"] = {}
    return graph.op(type_name, name, config_json=json.dumps(payload, allow_nan=False))


def _make_operator(type_name: str) -> Any:
    def operator(graph: Any, name: str, *, extents: Mapping[str, int], **params: object) -> Any:
        return _operator_call(type_name, graph, name, extents=extents, **params)

    operator.__name__ = type_name.removeprefix("holoscan.radar.")
    operator.__qualname__ = operator.__name__
    operator.__doc__ = f"Create {type_name} with Radar YAML parameters."
    operator.registry_name = type_name
    return operator


_REGISTRY_NAMES = (
    "CPHDReaderOpF32GeomF32",
    "CPHDReaderOpF32GeomF64",
    "CPHDReaderOpF64GeomF64",
    "ApertureManagerOpF32GeomF32",
    "ApertureManagerOpF32GeomF64",
    "ApertureManagerOpF64GeomF64",
    "BackprojectionOpF32GeomF32",
    "BackprojectionOpF32GeomF64",
    "BackprojectionOpF64GeomF64",
    "SARRangeCompressionOpF32",
    "SARRangeCompressionOpF64",
    "PhaseCorrectionOpF32",
    "PhaseCorrectionOpF64",
    "AutofocusOpF32",
    "AutofocusOpF64",
    "BeamscanEstimatorOpF32",
    "BeamscanEstimatorOpF64",
    "CFARDetectorOpF32",
    "CFARDetectorOpF64",
    "ClusterDBSCANOpF32",
    "ClusterDBSCANOpF64",
    "ClutterFilterOpF32",
    "ClutterFilterOpF64",
    "DDMDenseDemultiplexOpF32",
    "DDMDenseDemultiplexOpF64",
    "DDMPowerFoldOpF32",
    "DDMPowerFoldOpF64",
    "DDMSparseDemultiplexOpF32",
    "DDMSparseDemultiplexOpF64",
    "DetectionReductionOpF32",
    "DetectionReductionOpF64",
    "DopplerNotchOpF32",
    "DopplerNotchOpF64",
    "ESPRITEstimatorOpF32",
    "ESPRITEstimatorOpF64",
    "FMCWDopplerProcessorOpF32",
    "FMCWDopplerProcessorOpF64",
    "FMCWRangeDopplerProcessorOpF32",
    "FMCWRangeDopplerProcessorOpF64",
    "FMCWRangeProcessorOpF32",
    "FMCWRangeProcessorOpF64",
    "FeatureAssemblyOpF32",
    "FeatureAssemblyOpF64",
    "MIMODeinterleaveOpF32",
    "MIMODeinterleaveOpF64",
    "MUSICEstimatorOpF32",
    "MUSICEstimatorOpF64",
    "MVDREstimatorOpF32",
    "MVDREstimatorOpF64",
    "MultiSourceExpansionOpF32",
    "MultiSourceExpansionOpF64",
    "PhaseShiftBeamformerOpF32",
    "PhaseShiftBeamformerOpF64",
    "RankOneResidualScreenOpF32",
    "RankOneResidualScreenOpF64",
    "RootMUSICEstimatorOpF32",
    "RootMUSICEstimatorOpF64",
    "SnapshotGatherOpF32",
    "SnapshotGatherOpF64",
    "SubarraySelectionOpF32",
    "SubarraySelectionOpF64",
    "TDMDenseCompensationOpF32",
    "TDMDenseCompensationOpF64",
    "TDMSparseCompensationOpF32",
    "TDMSparseCompensationOpF64",
    "VirtualArrayAssemblyOpF32",
    "VirtualArrayAssemblyOpF64",
)
globals().update({name: _make_operator(f"holoscan.radar.{name}") for name in _REGISTRY_NAMES})


__all__ = list(_REGISTRY_NAMES)

# Source builds register the TensorDumpOp debugging tap; the wheel omits it.
if _radar._has_tensor_dump:

    def TensorDumpOp(graph: Any, name: str, **params: object) -> Any:
        """Create the dtype-independent tensor dump sink (no pool extents)."""
        return _operator_call("holoscan.radar.TensorDumpOp", graph, name, extents={}, **params)

    TensorDumpOp.registry_name = "holoscan.radar.TensorDumpOp"
    __all__.append("TensorDumpOp")


def plan_cfar_calibration(params: Mapping[str, object], input_shape: list[int]) -> dict[str, Any]:
    """Plan a CFAR LUT build, including a bounded GPU timing probe for Monte Carlo.

    The returned trial budget describes per-geometry relative Pfa coverage,
    not alpha error or simultaneous coverage across the complete LUT.
    """
    payload = {"params": _config_value(params)}
    return _radar._plan_cfar_calibration(json.dumps(payload, allow_nan=False), input_shape)


def calibrate_cfar(
    params: Mapping[str, object], input_shape: list[int], *, progress: Any = None
) -> dict[str, Any]:
    """Generate an offline alpha_lut and configuration/accuracy metadata on the GPU.

    The progress callback receives a dictionary after each bounded batch. Raising
    from it cancels the build. KeyboardInterrupt is checked between batches.
    Save the returned alpha_lut as an operator parameter to bypass calibration.
    """
    payload = {"params": _config_value(params)}
    return _radar._calibrate_cfar(json.dumps(payload, allow_nan=False), input_shape, progress)
