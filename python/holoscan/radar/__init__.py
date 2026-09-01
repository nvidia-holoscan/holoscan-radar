# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""holoscan.radar — GPU radar signal-processing operators for Holoscan."""

# radar_common must load first: op pybind modules use ArrayGeometry /
# AngleConvention / WindowSpec / PeakInterpolation as constructor defaults, and
# pybind11 requires those types to be registered before the op module imports.
# isort: off
from .radar_common import (  # noqa: F401
    DEFAULT_PROPAGATION_SPEED,
    AngleConvention,
    AngleConventionPreset,
    ApertureEmitMode,
    ArrayGeometry,
    ArrayGeometryType,
    BackprojectionBackend,
    ClutterFilterMethod,
    CoordinateFrame,
    CoordinateFrameSpec,
    CoordinateFrameType,
    DDMCodeDescriptor,
    ElevationReference,
    Ellipsoid,
    FftNorm,
    FormationControl,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    GeometryChange,
    LFMWaveformDescriptor,
    LFMWaveformSpec,
    MultiplexingScheme,
    OverlapPolicy,
    PeakInterpolation,
    PhaseCorrectionSense,
    RangeInterpolation,
    RepresentativePolicy,
    SampleGeometry,
    SampleGeometrySpec,
    ScoreMethod,
    SignedAxis,
    SnapshotEdgeMode,
    TXSchedule,
    WindowSpec,
    WindowType,
    make_device_tensor,
)

# isort: on
from .beamscan_estimator_op import BeamscanEstimatorOpF32  # noqa: F401
from .beamscan_estimator_op import BeamscanEstimatorOpF64
from .cfar_detector_op import CFARDetectorOpF32  # noqa: F401
from .cfar_detector_op import (
    CFARDetectorOpF64,
    CfarVariant,
    build_ca_alpha_lut,
    build_go_alpha_lut,
    build_os_alpha_lut,
    build_so_alpha_lut,
    ca_cfar_alpha,
    ca_cfar_pfa_from_alpha,
)
from .cluster_dbscan_op import ClusterDBSCANOpF32, ClusterDBSCANOpF64  # noqa: F401
from .clutter_filter_op import ClutterFilterOpF32, ClutterFilterOpF64  # noqa: F401
from .detection_reduction_op import DetectionReductionOpF32  # noqa: F401
from .detection_reduction_op import DetectionReductionOpF64
from .feature_assembly_op import AssemblyFeature  # noqa: F401
from .feature_assembly_op import FeatureAssemblyOpF32, FeatureAssemblyOpF64
from .fmcw_doppler_processor_op import (  # noqa: F401
    FMCWDopplerProcessorOpF32,
    FMCWDopplerProcessorOpF64,
)
from .fmcw_range_doppler_processor_op import (  # noqa: F401
    FMCWRangeDopplerProcessorOpF32,
    FMCWRangeDopplerProcessorOpF64,
)
from .fmcw_range_processor_op import FMCWRangeProcessorOpF32  # noqa: F401
from .fmcw_range_processor_op import FMCWRangeProcessorOpF64
from .mimo_coherent_compensation_op import (  # noqa: F401
    MIMOCoherentCompensationOpF32,
    MIMOCoherentCompensationOpF64,
)
from .mimo_deinterleave_op import MIMODeinterleaveOpF32  # noqa: F401
from .mimo_deinterleave_op import MIMODeinterleaveOpF64
from .mvdr_estimator_op import MVDREstimatorOpF32  # noqa: F401
from .mvdr_estimator_op import MVDREstimatorOpF64, retire_matx_solver_caches
from .phase_shift_beamformer_op import (  # noqa: F401
    PhaseShiftBeamformerOpF32,
    PhaseShiftBeamformerOpF64,
)
from .radar_app import CadenceMeter  # noqa: F401
from .radar_app import (
    BoresightViewOp,
    ClusterOverlayOp,
    ClusterReportOp,
    DetectionReportOp,
    NullSinkOp,
    RdChannelSliceOp,
    RdMapToImageOp,
    RollingPlotOp,
    cluster_palette,
    gui_available,
    make_turbo_lut,
)
from .snapshot_gather_op import SnapshotGatherOpF32, SnapshotGatherOpF64  # noqa: F401
from .virtual_array_assembly_op import (  # noqa: F401
    VirtualArrayAssemblyOpF32,
    VirtualArrayAssemblyOpF64,
)

# Convenience aliases: bare name → single precision (matches C++ demo apps).
BeamscanEstimatorOp = BeamscanEstimatorOpF32
CFARDetectorOp = CFARDetectorOpF32
ClusterDBSCANOp = ClusterDBSCANOpF32
ClutterFilterOp = ClutterFilterOpF32
DetectionReductionOp = DetectionReductionOpF32
FeatureAssemblyOp = FeatureAssemblyOpF32
FMCWDopplerProcessorOp = FMCWDopplerProcessorOpF32
FMCWRangeDopplerProcessorOp = FMCWRangeDopplerProcessorOpF32
FMCWRangeProcessorOp = FMCWRangeProcessorOpF32
MIMOCoherentCompensationOp = MIMOCoherentCompensationOpF32
MIMODeinterleaveOp = MIMODeinterleaveOpF32
MVDREstimatorOp = MVDREstimatorOpF32
PhaseShiftBeamformerOp = PhaseShiftBeamformerOpF32
SnapshotGatherOp = SnapshotGatherOpF32
VirtualArrayAssemblyOp = VirtualArrayAssemblyOpF32

__all__ = [
    "DEFAULT_PROPAGATION_SPEED",
    "AngleConvention",
    "AngleConventionPreset",
    "ApertureEmitMode",
    "ArrayGeometry",
    "ArrayGeometryType",
    "AssemblyFeature",
    "BackprojectionBackend",
    "BeamscanEstimatorOp",
    "BeamscanEstimatorOpF32",
    "BeamscanEstimatorOpF64",
    "BoresightViewOp",
    "CFARDetectorOp",
    "CFARDetectorOpF32",
    "CFARDetectorOpF64",
    "CadenceMeter",
    "CfarVariant",
    "ClusterDBSCANOp",
    "ClusterDBSCANOpF32",
    "ClusterDBSCANOpF64",
    "ClusterOverlayOp",
    "ClusterReportOp",
    "ClutterFilterMethod",
    "ClutterFilterOp",
    "ClutterFilterOpF32",
    "ClutterFilterOpF64",
    "CoordinateFrame",
    "CoordinateFrameSpec",
    "CoordinateFrameType",
    "DDMCodeDescriptor",
    "DetectionReductionOp",
    "DetectionReductionOpF32",
    "DetectionReductionOpF64",
    "DetectionReportOp",
    "ElevationReference",
    "Ellipsoid",
    "FMCWDopplerProcessorOp",
    "FMCWDopplerProcessorOpF32",
    "FMCWDopplerProcessorOpF64",
    "FMCWRangeDopplerProcessorOp",
    "FMCWRangeDopplerProcessorOpF32",
    "FMCWRangeDopplerProcessorOpF64",
    "FMCWRangeProcessorOp",
    "FMCWRangeProcessorOpF32",
    "FMCWRangeProcessorOpF64",
    "FMCWWaveformDescriptor",
    "FMCWWaveformSpec",
    "FeatureAssemblyOp",
    "FeatureAssemblyOpF32",
    "FeatureAssemblyOpF64",
    "FftNorm",
    "FormationControl",
    "GeometryChange",
    "LFMWaveformDescriptor",
    "LFMWaveformSpec",
    "MIMOCoherentCompensationOp",
    "MIMOCoherentCompensationOpF32",
    "MIMOCoherentCompensationOpF64",
    "MIMODeinterleaveOp",
    "MIMODeinterleaveOpF32",
    "MIMODeinterleaveOpF64",
    "MVDREstimatorOp",
    "MVDREstimatorOpF32",
    "MVDREstimatorOpF64",
    "retire_matx_solver_caches",
    "install_matx_cache_atexit",
    "MultiplexingScheme",
    "NullSinkOp",
    "OverlapPolicy",
    "PeakInterpolation",
    "PhaseCorrectionSense",
    "RangeInterpolation",
    "PhaseShiftBeamformerOp",
    "PhaseShiftBeamformerOpF32",
    "PhaseShiftBeamformerOpF64",
    "RdChannelSliceOp",
    "RdMapToImageOp",
    "RepresentativePolicy",
    "RollingPlotOp",
    "SampleGeometry",
    "SampleGeometrySpec",
    "ScoreMethod",
    "SignedAxis",
    "SnapshotEdgeMode",
    "SnapshotGatherOp",
    "SnapshotGatherOpF32",
    "SnapshotGatherOpF64",
    "TXSchedule",
    "VirtualArrayAssemblyOp",
    "VirtualArrayAssemblyOpF32",
    "VirtualArrayAssemblyOpF64",
    "WindowSpec",
    "WindowType",
    "ca_cfar_alpha",
    "ca_cfar_pfa_from_alpha",
    "build_ca_alpha_lut",
    "build_go_alpha_lut",
    "build_so_alpha_lut",
    "build_os_alpha_lut",
    "cluster_palette",
    "gui_available",
    "make_device_tensor",
    "make_turbo_lut",
]

_matx_atexit_installed = False


def install_matx_cache_atexit() -> None:
    """Retire MatX eig/inv plans on interpreter exit, before CuPy tears down CUDA.

    Call **after** ``import cupy`` (atexit is LIFO), and only after
    ``holoscan.radar`` has already been imported (CuPy-before-radar makes MatX
    eig ``Clear`` abort on this stack). ``tests/python/conftest.py`` does the
    safe order for pytest.
    """
    global _matx_atexit_installed
    if _matx_atexit_installed:
        return
    import atexit

    atexit.register(retire_matx_solver_caches)
    _matx_atexit_installed = True


# Prefer: import holoscan.radar, then cupy, then install_matx_cache_atexit().
# Do not auto-register here — CuPy-before-radar import order makes MatX eig
# Clear abort on teardown; callers (demos, conftest, perf) must opt in.
