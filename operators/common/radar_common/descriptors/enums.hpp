// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

// Canonical spellings (lowercase snake_case, for logs and future YAML
// configs).
constexpr std::string_view to_string(WindowType type) {
  switch (type) {
    case WindowType::kNone: return "none";
    case WindowType::kHann: return "hann";
    case WindowType::kHamming: return "hamming";
    case WindowType::kBlackman: return "blackman";
    case WindowType::kBlackmanHarris: return "blackman_harris";
    case WindowType::kKaiser: return "kaiser";
    case WindowType::kChebyshev: return "chebyshev";
    case WindowType::kTaylor: return "taylor";
    case WindowType::kFlatTop: return "flat_top";
    case WindowType::kTukey: return "tukey";
    case WindowType::kCustom: return "custom";
  }
  return "unknown";
}

constexpr std::string_view to_string(FftNorm norm) {
  switch (norm) {
    case FftNorm::kBackward: return "backward";
    case FftNorm::kOrtho: return "ortho";
    case FftNorm::kForward: return "forward";
  }
  return "unknown";
}

constexpr std::string_view to_string(ClutterFilterMethod method) {
  switch (method) {
    case ClutterFilterMethod::kNone: return "none";
    case ClutterFilterMethod::kMeanSubtraction: return "mean_subtraction";
    case ClutterFilterMethod::kSingleDelay: return "single_delay";
    case ClutterFilterMethod::kDoubleDelay: return "double_delay";
  }
  return "unknown";
}

constexpr std::string_view to_string(SnapshotEdgeMode mode) {
  switch (mode) {
    case SnapshotEdgeMode::kWrap: return "wrap";
    case SnapshotEdgeMode::kClamp: return "clamp";
    case SnapshotEdgeMode::kShift: return "shift";
  }
  return "unknown";
}

constexpr std::string_view to_string(ArrayGeometryType type) {
  switch (type) {
    case ArrayGeometryType::kUla: return "ula";
    case ArrayGeometryType::kUra: return "ura";
    case ArrayGeometryType::kUca: return "uca";
    case ArrayGeometryType::kConformal: return "conformal";
  }
  return "unknown";
}

constexpr std::string_view to_string(AngleConventionPreset preset) {
  switch (preset) {
    case AngleConventionPreset::kRadarBoresight: return "radar_boresight";
    case AngleConventionPreset::kRosRep103: return "ros_rep103";
    case AngleConventionPreset::kIso8855Flu: return "iso8855_flu";
    case AngleConventionPreset::kAerospaceNed: return "aerospace_ned";
    case AngleConventionPreset::kPhysicsSpherical: return "physics_spherical";
    case AngleConventionPreset::kGeodeticEnu: return "geodetic_enu";
    case AngleConventionPreset::kGeodeticNed: return "geodetic_ned";
    case AngleConventionPreset::kDirectionCosine: return "direction_cosine";
    case AngleConventionPreset::kCustom: return "custom";
  }
  return "unknown";
}

constexpr std::string_view to_string(SignedAxis axis) {
  switch (axis) {
    case SignedAxis::kPlusX: return "+x";
    case SignedAxis::kMinusX: return "-x";
    case SignedAxis::kPlusY: return "+y";
    case SignedAxis::kMinusY: return "-y";
    case SignedAxis::kPlusZ: return "+z";
    case SignedAxis::kMinusZ: return "-z";
  }
  return "unknown";
}

constexpr std::string_view to_string(ElevationReference ref) {
  switch (ref) {
    case ElevationReference::kHorizon: return "horizon";
    case ElevationReference::kZenith: return "zenith";
  }
  return "unknown";
}

constexpr std::string_view to_string(PeakInterpolation mode) {
  switch (mode) {
    case PeakInterpolation::kNone: return "none";
    case PeakInterpolation::kParabolic: return "parabolic";
  }
  return "unknown";
}

constexpr std::string_view to_string(OverlapPolicy policy) {
  switch (policy) {
    case OverlapPolicy::kDedup: return "dedup";
    case OverlapPolicy::kAverage: return "average";
    case OverlapPolicy::kKeepAll: return "keep_all";
  }
  return "unknown";
}

constexpr std::string_view to_string(MultiplexingScheme scheme) {
  switch (scheme) {
    case MultiplexingScheme::kUnspecified: return "unspecified";
    case MultiplexingScheme::kTdm: return "tdm";
    case MultiplexingScheme::kDdm: return "ddm";
  }
  return "unknown";
}

}  // namespace holoscan::radar
