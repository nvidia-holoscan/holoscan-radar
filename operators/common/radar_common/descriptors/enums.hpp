// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

namespace holoscan::radar {

// Window taper families supported by WindowSpec.
enum class WindowType {
  kNone,
  kHann,
  kHamming,
  kBlackman,
  kBlackmanHarris,
  kKaiser,
  kChebyshev,
  kTaylor,
  kFlatTop,
  kTukey,
  kCustom,
};

// FFT normalization convention. kBackward (unnormalized forward transform)
// is the radar standard: it keeps coherent-integration gain visible.
enum class FftNorm {
  kBackward,
  kOrtho,
  kForward,
};

// Slow-time clutter suppression method. kNone is only meaningful for the
// fused operator's optional inline clutter stage; ClutterFilterOp itself
// requires one of the active methods.
enum class ClutterFilterMethod {
  kNone,
  kMeanSubtraction,
  kSingleDelay,
  kDoubleDelay,
};

// Neighborhood edge behavior for SnapshotGatherOp along a range-Doppler axis.
enum class SnapshotEdgeMode {
  kWrap,
  kClamp,
  kShift,
};

// Informational array layout tag on ArrayGeometry (enables optimized kernels).
enum class ArrayGeometryType {
  kUla,
  kUra,
  kUca,
  kConformal,
};

// AngleConvention preset (see OPERATOR_API.md Supporting data structures).
enum class AngleConventionPreset {
  kRadarBoresight,
  kRosRep103,
  kIso8855Flu,
  kAerospaceNed,
  kPhysicsSpherical,
  kGeodeticEnu,
  kGeodeticNed,
  kDirectionCosine,
  kCustom,
};

// Signed Cartesian axis reference for AngleConvention::custom.
enum class SignedAxis {
  kPlusX,
  kMinusX,
  kPlusY,
  kMinusY,
  kPlusZ,
  kMinusZ,
};

enum class ElevationReference {
  kHorizon,
  kZenith,
};

// Sub-grid peak refinement for grid-scan angle estimators.
enum class PeakInterpolation {
  kNone,
  kParabolic,
};

// Virtual-element collision handling for VirtualArrayAssemblyOp.
enum class OverlapPolicy {
  kDedup,
  kAverage,
  kKeepAll,
};

// MIMO TX multiplexing scheme for TDMDenseCompensationOp / DDMDenseDemultiplexOp.
enum class MultiplexingScheme {
  kUnspecified,  // Holoscan Parameter default only; reject at compute().
  kTdm,
  kDdm,
};

// Cartesian frame family declared by CoordinateFrame. The frame is a
// boundary declaration only: formation kernels never read it.
enum class CoordinateFrameType {
  kUnspecified,  // Spec/Parameter default only; rejected at construction.
  kEcef,
  kEnu,
  kNed,
  kSceneLocal,
};

// Reference ellipsoid that CoordinateFrame::origin_llh is expressed against.
enum class Ellipsoid {
  kWgs84,
};

// BackprojectionOp.control vocabulary (D23). Wire values are uint8 in this
// table order; the enum is the named spelling for YAML, logs, and tests.
enum class FormationControl : std::uint8_t {
  kEmitAndContinue = 0,
  kEmitAndReset = 1,
  kResetWithoutEmit = 2,
  kResetThenEmit = 3,
};

// Range-profile interpolation at the computed range. Only kLinear is
// implemented this wave; the others are named so the enum does not grow
// later, and are rejected rather than silently falling back.
enum class RangeInterpolation {
  kNearest,
  kLinear,
  kSinc,
};

// BackprojectionOp.backend: legal kernel combinations, prefixed so a MatX
// wrap and an NVRadar copy cannot share a spelling. Default is the
// vectorized NVRadar LUT kernel (not the planar-only `_z0` variant).
enum class BackprojectionBackend {
  kMatxDouble,
  kMatxDoublePhaseLut,
  kMatxMixed,
  kMatxMixedPhaseLut,
  kMatxFloatFloat,
  kMatxFloat,
  kMatxFloatPhaseLut,
  kMatxTaylorFast,
  kNvradarCudaFp32,
  kNvradarCudaFp32Hq,
  kNvradarCudaFp32HqFast,
  kNvradarFp64Reference,
  kNvradarBlockAnchored,
  kNvradarBlockAnchoredFp64Trajectory,
  kNvradarBlockAnchoredRefinedThroughput,
  kNvradarBlockAnchoredRefinedQuality,
  kNvradarBlockAnchoredExpandedThroughput,
  kNvradarBlockAnchoredExpandedQuality,
  kNvradarBlockAnchoredPhaseLut,
  kNvradarBlockAnchoredRefinedQualityPhaseLut,
  kNvradarBlockAnchoredExpandedThroughputPhaseLut,
  kNvradarBlockAnchoredExpandedThroughputPhaseLutVec,
  kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0,
  kNvradarBlockAnchoredExpandedQualityPhaseLut,
  kNvradarBlockAnchoredPhaseLutLegacy,
  kNvradarBlockAnchoredFp64TrajectoryPhaseLut,
};

inline constexpr BackprojectionBackend kDefaultBackprojectionBackend = {
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVec,
};

// What an incoming sample geometry means for the accumulator.
enum class GeometryChange {
  kReset,
  kScroll,
};

// ApertureManagerOp.emit_mode: which FormationControl accompanies an emit.
// Structural (Static). YAML spelling `continue` is kContinue.
enum class ApertureEmitMode {
  kContinue,
  kReset,
  kReform,
};

// PhaseCorrectionOp.sense: multiply by exp(−iφ) or exp(+iφ). Static,
// required; kUnspecified is the Parameter placeholder and is rejected
// at initialize (the contract has no default).
enum class PhaseCorrectionSense {
  kUnspecified,
  kSubtract,
  kAdd,
};

// AutofocusOp.algorithm: wave-one estimator slot. Static; sole value
// is PGA (the stage itself defaults to absent from the spine).
enum class AutofocusAlgorithm {
  kPga,
};

// AutofocusOp.azimuth_axis: which image axis is slow-time-conjugate.
// Static and required; kUnspecified is the Parameter placeholder and
// is rejected at initialize (the contract has no default). Naming an
// axis is also the assertion that the image is conjugate (D20).
enum class AutofocusAzimuthAxis {
  kUnspecified,
  kU,
  kV,
};

// Canonical spellings (lowercase snake_case, for logs and future YAML
// configs).
constexpr std::string_view to_string(WindowType type) {
  switch (type) {
    case WindowType::kNone:
      return "none";
    case WindowType::kHann:
      return "hann";
    case WindowType::kHamming:
      return "hamming";
    case WindowType::kBlackman:
      return "blackman";
    case WindowType::kBlackmanHarris:
      return "blackman_harris";
    case WindowType::kKaiser:
      return "kaiser";
    case WindowType::kChebyshev:
      return "chebyshev";
    case WindowType::kTaylor:
      return "taylor";
    case WindowType::kFlatTop:
      return "flat_top";
    case WindowType::kTukey:
      return "tukey";
    case WindowType::kCustom:
      return "custom";
  }
  return "unknown";
}

constexpr std::string_view to_string(FftNorm norm) {
  switch (norm) {
    case FftNorm::kBackward:
      return "backward";
    case FftNorm::kOrtho:
      return "ortho";
    case FftNorm::kForward:
      return "forward";
  }
  return "unknown";
}

constexpr std::string_view to_string(ClutterFilterMethod method) {
  switch (method) {
    case ClutterFilterMethod::kNone:
      return "none";
    case ClutterFilterMethod::kMeanSubtraction:
      return "mean_subtraction";
    case ClutterFilterMethod::kSingleDelay:
      return "single_delay";
    case ClutterFilterMethod::kDoubleDelay:
      return "double_delay";
  }
  return "unknown";
}

constexpr std::string_view to_string(SnapshotEdgeMode mode) {
  switch (mode) {
    case SnapshotEdgeMode::kWrap:
      return "wrap";
    case SnapshotEdgeMode::kClamp:
      return "clamp";
    case SnapshotEdgeMode::kShift:
      return "shift";
  }
  return "unknown";
}

constexpr std::string_view to_string(ArrayGeometryType type) {
  switch (type) {
    case ArrayGeometryType::kUla:
      return "ula";
    case ArrayGeometryType::kUra:
      return "ura";
    case ArrayGeometryType::kUca:
      return "uca";
    case ArrayGeometryType::kConformal:
      return "conformal";
  }
  return "unknown";
}

constexpr std::string_view to_string(AngleConventionPreset preset) {
  switch (preset) {
    case AngleConventionPreset::kRadarBoresight:
      return "radar_boresight";
    case AngleConventionPreset::kRosRep103:
      return "ros_rep103";
    case AngleConventionPreset::kIso8855Flu:
      return "iso8855_flu";
    case AngleConventionPreset::kAerospaceNed:
      return "aerospace_ned";
    case AngleConventionPreset::kPhysicsSpherical:
      return "physics_spherical";
    case AngleConventionPreset::kGeodeticEnu:
      return "geodetic_enu";
    case AngleConventionPreset::kGeodeticNed:
      return "geodetic_ned";
    case AngleConventionPreset::kDirectionCosine:
      return "direction_cosine";
    case AngleConventionPreset::kCustom:
      return "custom";
  }
  return "unknown";
}

constexpr std::string_view to_string(SignedAxis axis) {
  switch (axis) {
    case SignedAxis::kPlusX:
      return "+x";
    case SignedAxis::kMinusX:
      return "-x";
    case SignedAxis::kPlusY:
      return "+y";
    case SignedAxis::kMinusY:
      return "-y";
    case SignedAxis::kPlusZ:
      return "+z";
    case SignedAxis::kMinusZ:
      return "-z";
  }
  return "unknown";
}

constexpr std::string_view to_string(ElevationReference ref) {
  switch (ref) {
    case ElevationReference::kHorizon:
      return "horizon";
    case ElevationReference::kZenith:
      return "zenith";
  }
  return "unknown";
}

constexpr std::string_view to_string(PeakInterpolation mode) {
  switch (mode) {
    case PeakInterpolation::kNone:
      return "none";
    case PeakInterpolation::kParabolic:
      return "parabolic";
  }
  return "unknown";
}

constexpr std::string_view to_string(OverlapPolicy policy) {
  switch (policy) {
    case OverlapPolicy::kDedup:
      return "dedup";
    case OverlapPolicy::kAverage:
      return "average";
    case OverlapPolicy::kKeepAll:
      return "keep_all";
  }
  return "unknown";
}

constexpr std::string_view to_string(MultiplexingScheme scheme) {
  switch (scheme) {
    case MultiplexingScheme::kUnspecified:
      return "unspecified";
    case MultiplexingScheme::kTdm:
      return "tdm";
    case MultiplexingScheme::kDdm:
      return "ddm";
  }
  return "unknown";
}

constexpr std::string_view to_string(CoordinateFrameType frame) {
  switch (frame) {
    case CoordinateFrameType::kUnspecified:
      return "unspecified";
    case CoordinateFrameType::kEcef:
      return "ecef";
    case CoordinateFrameType::kEnu:
      return "enu";
    case CoordinateFrameType::kNed:
      return "ned";
    case CoordinateFrameType::kSceneLocal:
      return "scene_local";
  }
  return "unknown";
}

constexpr std::string_view to_string(Ellipsoid ellipsoid) {
  switch (ellipsoid) {
    case Ellipsoid::kWgs84:
      return "wgs84";
  }
  return "unknown";
}

constexpr std::string_view to_string(FormationControl control) {
  switch (control) {
    case FormationControl::kEmitAndContinue:
      return "emit_and_continue";
    case FormationControl::kEmitAndReset:
      return "emit_and_reset";
    case FormationControl::kResetWithoutEmit:
      return "reset_without_emit";
    case FormationControl::kResetThenEmit:
      return "reset_then_emit";
  }
  return "unknown";
}

constexpr std::string_view to_string(RangeInterpolation mode) {
  switch (mode) {
    case RangeInterpolation::kNearest:
      return "nearest";
    case RangeInterpolation::kLinear:
      return "linear";
    case RangeInterpolation::kSinc:
      return "sinc";
  }
  return "unknown";
}

constexpr std::string_view to_string(BackprojectionBackend backend) {
  switch (backend) {
    case BackprojectionBackend::kMatxDouble:
      return "matx_double";
    case BackprojectionBackend::kMatxDoublePhaseLut:
      return "matx_double_phase_lut";
    case BackprojectionBackend::kMatxMixed:
      return "matx_mixed";
    case BackprojectionBackend::kMatxMixedPhaseLut:
      return "matx_mixed_phase_lut";
    case BackprojectionBackend::kMatxFloatFloat:
      return "matx_float_float";
    case BackprojectionBackend::kMatxFloat:
      return "matx_float";
    case BackprojectionBackend::kMatxFloatPhaseLut:
      return "matx_float_phase_lut";
    case BackprojectionBackend::kMatxTaylorFast:
      return "matx_taylor_fast";
    case BackprojectionBackend::kNvradarCudaFp32:
      return "nvradar_cuda_fp32";
    case BackprojectionBackend::kNvradarCudaFp32Hq:
      return "nvradar_cuda_fp32_hq";
    case BackprojectionBackend::kNvradarCudaFp32HqFast:
      return "nvradar_cuda_fp32_hq_fast";
    case BackprojectionBackend::kNvradarFp64Reference:
      return "nvradar_fp64_reference";
    case BackprojectionBackend::kNvradarBlockAnchored:
      return "nvradar_block_anchored";
    case BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory:
      return "nvradar_block_anchored_fp64_trajectory";
    case BackprojectionBackend::kNvradarBlockAnchoredRefinedThroughput:
      return "nvradar_block_anchored_refined_throughput";
    case BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality:
      return "nvradar_block_anchored_refined_quality";
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput:
      return "nvradar_block_anchored_expanded_throughput";
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedQuality:
      return "nvradar_block_anchored_expanded_quality";
    case BackprojectionBackend::kNvradarBlockAnchoredPhaseLut:
      return "nvradar_block_anchored_phase_lut";
    case BackprojectionBackend::kNvradarBlockAnchoredRefinedQualityPhaseLut:
      return "nvradar_block_anchored_refined_quality_phase_lut";
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLut:
      return "nvradar_block_anchored_expanded_throughput_phase_lut";
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVec:
      return "nvradar_block_anchored_expanded_throughput_phase_lut_vec";
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0:
      return "nvradar_block_anchored_expanded_throughput_phase_lut_vec_z0";
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedQualityPhaseLut:
      return "nvradar_block_anchored_expanded_quality_phase_lut";
    case BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy:
      return "nvradar_block_anchored_phase_lut_legacy";
    case BackprojectionBackend::kNvradarBlockAnchoredFp64TrajectoryPhaseLut:
      return "nvradar_block_anchored_fp64_trajectory_phase_lut";
  }
  return "unknown";
}

constexpr std::string_view to_string(ApertureEmitMode mode) {
  switch (mode) {
    case ApertureEmitMode::kContinue:
      return "continue";
    case ApertureEmitMode::kReset:
      return "reset";
    case ApertureEmitMode::kReform:
      return "reform";
  }
  return "unknown";
}

constexpr std::string_view to_string(GeometryChange change) {
  switch (change) {
    case GeometryChange::kReset:
      return "reset";
    case GeometryChange::kScroll:
      return "scroll";
  }
  return "unknown";
}

constexpr std::string_view to_string(PhaseCorrectionSense sense) {
  switch (sense) {
    case PhaseCorrectionSense::kUnspecified:
      return "unspecified";
    case PhaseCorrectionSense::kSubtract:
      return "subtract";
    case PhaseCorrectionSense::kAdd:
      return "add";
  }
  return "unknown";
}

constexpr std::string_view to_string(AutofocusAlgorithm algorithm) {
  switch (algorithm) {
    case AutofocusAlgorithm::kPga:
      return "pga";
  }
  return "unknown";
}

constexpr std::string_view to_string(AutofocusAzimuthAxis axis) {
  switch (axis) {
    case AutofocusAzimuthAxis::kUnspecified:
      return "unspecified";
    case AutofocusAzimuthAxis::kU:
      return "u";
    case AutofocusAzimuthAxis::kV:
      return "v";
  }
  return "unknown";
}

// Taylor expansion backends: every nvradar_block_anchored* value and
// matx_taylor_fast. Used by the locality warning (I7).
constexpr bool is_taylor_backend(BackprojectionBackend backend) {
  switch (backend) {
    case BackprojectionBackend::kMatxTaylorFast:
    case BackprojectionBackend::kNvradarBlockAnchored:
    case BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory:
    case BackprojectionBackend::kNvradarBlockAnchoredRefinedThroughput:
    case BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality:
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput:
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedQuality:
    case BackprojectionBackend::kNvradarBlockAnchoredPhaseLut:
    case BackprojectionBackend::kNvradarBlockAnchoredRefinedQualityPhaseLut:
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLut:
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVec:
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0:
    case BackprojectionBackend::kNvradarBlockAnchoredExpandedQualityPhaseLut:
    case BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy:
    case BackprojectionBackend::kNvradarBlockAnchoredFp64TrajectoryPhaseLut:
      return true;
    default:
      return false;
  }
}

}  // namespace holoscan::radar
