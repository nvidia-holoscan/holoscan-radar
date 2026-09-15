# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""holoscan.radar — GPU radar signal-processing operators for Holoscan."""

# radar_common must load first: op pybind modules use ArrayGeometry /
# AngleConvention / WindowSpec / PeakInterpolation as constructor defaults, and
# pybind11 requires those types to be registered before the op module imports.
# isort: off
from .radar_common import (  # noqa: F401
    DEFAULT_PROPAGATION_SPEED,
    K_HALF_PI,
    K_PI,
    K_SPEED_OF_LIGHT,
    K_TWO_PI,
    AngleConvention,
    AngleConventionPreset,
    ArrayGeometry,
    ArrayGeometryType,
    ClutterFilterMethod,
    DDMCodeDescriptor,
    ElevationReference,
    FftNorm,
    FMCWWaveformDescriptor,
    FMCWWaveformSpec,
    MultiplexingScheme,
    OverlapPolicy,
    PeakInterpolation,
    RepresentativePolicy,
    ScoreMethod,
    ShiftInvariantSubarrayPair,
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
    CfarPeakMode,
    CfarPeakNeighborhood,
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
from .esprit_estimator_op import (  # noqa: F401
    ESPRITEstimatorOpF32,
    ESPRITEstimatorOpF64,
)
from .feature_assembly_op import AssemblyFeature  # noqa: F401
from .feature_assembly_op import FeatureAssemblyOpF32, FeatureAssemblyOpF64
from .music_estimator_op import MUSICEstimatorOpF32  # noqa: F401
from .music_estimator_op import (
    MUSICEstimatorOpF64,
    retire_music_matx_caches,
)
from .mvdr_estimator_op import MVDREstimatorOpF32  # noqa: F401
from .mvdr_estimator_op import MVDREstimatorOpF64, retire_matx_solver_caches
from .phase_shift_beamformer_op import (  # noqa: F401
    PhaseShiftBeamformerOpF32,
    PhaseShiftBeamformerOpF64,
)
from .radar_app import CadenceMeter  # noqa: F401
from .radar_app import ClusterReportOp, DetectionReportOp, NullSinkOp
from .root_music_estimator_op import (  # noqa: F401
    RootMUSICEstimatorOpF32,
    RootMUSICEstimatorOpF64,
)
from .snapshot_gather_op import SnapshotGatherOpF32, SnapshotGatherOpF64  # noqa: F401

# [EA HSDK5 ONLY] The FMCW range/Doppler/fused-RD and MIMO marshaling operators
# are not on the HSDK 5 port allow-list, so their extension modules are absent
# from an EA build. Import each optionally — one missing module must not take
# down `import holoscan.radar` for the operators that did port. Missing names
# are None, as in the Holoviz block below.
try:
    from .fmcw_doppler_processor_op import (  # noqa: F401
        FMCWDopplerProcessorOpF32,
        FMCWDopplerProcessorOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    FMCWDopplerProcessorOpF32 = None  # type: ignore[misc, assignment]
    FMCWDopplerProcessorOpF64 = None  # type: ignore[misc, assignment]

try:
    from .fmcw_range_doppler_processor_op import (  # noqa: F401
        FMCWRangeDopplerProcessorOpF32,
        FMCWRangeDopplerProcessorOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    FMCWRangeDopplerProcessorOpF32 = None  # type: ignore[misc, assignment]
    FMCWRangeDopplerProcessorOpF64 = None  # type: ignore[misc, assignment]

try:
    from .fmcw_range_processor_op import (  # noqa: F401
        FMCWRangeProcessorOpF32,
        FMCWRangeProcessorOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    FMCWRangeProcessorOpF32 = None  # type: ignore[misc, assignment]
    FMCWRangeProcessorOpF64 = None  # type: ignore[misc, assignment]

try:
    from .tdm_dense_compensation_op import (  # noqa: F401
        TDMDenseCompensationOpF32,
        TDMDenseCompensationOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    TDMDenseCompensationOpF32 = None  # type: ignore[misc, assignment]
    TDMDenseCompensationOpF64 = None  # type: ignore[misc, assignment]

try:
    from .ddm_dense_demultiplex_op import (  # noqa: F401
        DDMDenseDemultiplexOpF32,
        DDMDenseDemultiplexOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    DDMDenseDemultiplexOpF32 = None  # type: ignore[misc, assignment]
    DDMDenseDemultiplexOpF64 = None  # type: ignore[misc, assignment]

try:
    from .ddm_power_fold_op import (  # noqa: F401
        DDMPowerFoldOpF32,
        DDMPowerFoldOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    DDMPowerFoldOpF32 = None  # type: ignore[misc, assignment]
    DDMPowerFoldOpF64 = None  # type: ignore[misc, assignment]

try:
    from .ddm_sparse_demultiplex_op import (  # noqa: F401
        DDMSparseDemultiplexOpF32,
        DDMSparseDemultiplexOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    DDMSparseDemultiplexOpF32 = None  # type: ignore[misc, assignment]
    DDMSparseDemultiplexOpF64 = None  # type: ignore[misc, assignment]

try:
    from .mimo_deinterleave_op import (  # noqa: F401
        MIMODeinterleaveOpF32,
        MIMODeinterleaveOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    MIMODeinterleaveOpF32 = None  # type: ignore[misc, assignment]
    MIMODeinterleaveOpF64 = None  # type: ignore[misc, assignment]

try:
    from .virtual_array_assembly_op import (  # noqa: F401
        VirtualArrayAssemblyOpF32,
        VirtualArrayAssemblyOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    VirtualArrayAssemblyOpF32 = None  # type: ignore[misc, assignment]
    VirtualArrayAssemblyOpF64 = None  # type: ignore[misc, assignment]

try:
    from .tdm_sparse_compensation_op import (  # noqa: F401
        TDMSparseCompensationOpF32,
        TDMSparseCompensationOpF64,
    )
except ImportError:  # pragma: no cover - unported on HSDK 5 EA builds
    TDMSparseCompensationOpF32 = None  # type: ignore[misc, assignment]
    TDMSparseCompensationOpF64 = None  # type: ignore[misc, assignment]

# Holoviz-backed radar_app symbols are absent when Holoviz is gated off
# ([EA HSDK5 ONLY]); report sinks above still import.
try:
    from .radar_app import BoresightViewOp  # noqa: F401
    from .radar_app import (
        ClusterOverlayOp,
        PanelRect,
        RdAngleMaxOp,
        RdChannelSliceOp,
        RdMapToImageOp,
        RollingPlotOp,
        cluster_palette,
        gui_available,
        make_turbo_lut,
    )
except ImportError:  # pragma: no cover - Holoviz-gated EA builds
    BoresightViewOp = None  # type: ignore[misc, assignment]
    ClusterOverlayOp = None  # type: ignore[misc, assignment]
    PanelRect = None  # type: ignore[misc, assignment]
    RdAngleMaxOp = None  # type: ignore[misc, assignment]
    RdChannelSliceOp = None  # type: ignore[misc, assignment]
    RdMapToImageOp = None  # type: ignore[misc, assignment]
    RollingPlotOp = None  # type: ignore[misc, assignment]
    cluster_palette = None  # type: ignore[misc, assignment]
    gui_available = None  # type: ignore[misc, assignment]
    make_turbo_lut = None  # type: ignore[misc, assignment]

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
TDMDenseCompensationOp = TDMDenseCompensationOpF32
DDMDenseDemultiplexOp = DDMDenseDemultiplexOpF32
DDMPowerFoldOp = DDMPowerFoldOpF32
DDMSparseDemultiplexOp = DDMSparseDemultiplexOpF32
MIMODeinterleaveOp = MIMODeinterleaveOpF32
MUSICEstimatorOp = MUSICEstimatorOpF32
MVDREstimatorOp = MVDREstimatorOpF32
PhaseShiftBeamformerOp = PhaseShiftBeamformerOpF32
RootMUSICEstimatorOp = RootMUSICEstimatorOpF32
ESPRITEstimatorOp = ESPRITEstimatorOpF32
SnapshotGatherOp = SnapshotGatherOpF32
TDMSparseCompensationOp = TDMSparseCompensationOpF32
VirtualArrayAssemblyOp = VirtualArrayAssemblyOpF32

__all__ = [
    "DEFAULT_PROPAGATION_SPEED",
    "K_HALF_PI",
    "K_PI",
    "K_SPEED_OF_LIGHT",
    "K_TWO_PI",
    "AngleConvention",
    "AngleConventionPreset",
    "ArrayGeometry",
    "ArrayGeometryType",
    "AssemblyFeature",
    "BeamscanEstimatorOp",
    "BeamscanEstimatorOpF32",
    "BeamscanEstimatorOpF64",
    "BoresightViewOp",
    "CFARDetectorOp",
    "CFARDetectorOpF32",
    "CFARDetectorOpF64",
    "CadenceMeter",
    "CfarPeakMode",
    "CfarPeakNeighborhood",
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
    "DDMCodeDescriptor",
    "DetectionReductionOp",
    "DetectionReductionOpF32",
    "DetectionReductionOpF64",
    "DetectionReportOp",
    "ElevationReference",
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
    "DDMDenseDemultiplexOp",
    "DDMDenseDemultiplexOpF32",
    "DDMDenseDemultiplexOpF64",
    "DDMPowerFoldOp",
    "DDMPowerFoldOpF32",
    "DDMPowerFoldOpF64",
    "DDMSparseDemultiplexOp",
    "DDMSparseDemultiplexOpF32",
    "DDMSparseDemultiplexOpF64",
    "TDMDenseCompensationOp",
    "TDMDenseCompensationOpF32",
    "TDMDenseCompensationOpF64",
    "MIMODeinterleaveOp",
    "MIMODeinterleaveOpF32",
    "MIMODeinterleaveOpF64",
    "MUSICEstimatorOp",
    "MUSICEstimatorOpF32",
    "MUSICEstimatorOpF64",
    "MVDREstimatorOp",
    "MVDREstimatorOpF32",
    "MVDREstimatorOpF64",
    "retire_matx_solver_caches",
    "retire_music_matx_caches",
    "install_matx_cache_atexit",
    "MultiplexingScheme",
    "NullSinkOp",
    "OverlapPolicy",
    "PanelRect",
    "PeakInterpolation",
    "PhaseShiftBeamformerOp",
    "PhaseShiftBeamformerOpF32",
    "PhaseShiftBeamformerOpF64",
    "RdAngleMaxOp",
    "RdChannelSliceOp",
    "RdMapToImageOp",
    "RepresentativePolicy",
    "RootMUSICEstimatorOp",
    "RootMUSICEstimatorOpF32",
    "RootMUSICEstimatorOpF64",
    "ESPRITEstimatorOp",
    "ESPRITEstimatorOpF32",
    "ESPRITEstimatorOpF64",
    "RollingPlotOp",
    "ScoreMethod",
    "ShiftInvariantSubarrayPair",
    "SignedAxis",
    "SnapshotEdgeMode",
    "SnapshotGatherOp",
    "SnapshotGatherOpF32",
    "SnapshotGatherOpF64",
    "TDMSparseCompensationOp",
    "TDMSparseCompensationOpF32",
    "TDMSparseCompensationOpF64",
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
    atexit.register(retire_music_matx_caches)
    _matx_atexit_installed = True


# Prefer: import holoscan.radar, then cupy, then install_matx_cache_atexit().
# Do not auto-register here — CuPy-before-radar import order makes MatX eig
# Clear abort on teardown; callers (demos, conftest, perf) must opt in.
