// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/coordinate_frame.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/lfm_waveform_descriptor.hpp"
#include "radar_common/descriptors/sample_geometry.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/reduce/reduction_enums.hpp"

// YAML::convert<> specializations for the shared config-plane descriptor
// types, enabling Holoscan's from_config() path for operator parameters
// (see arg_setters.hpp for the registration side). Lives in its own header
// so the descriptor and sim layers stay yaml-cpp-free.
//
// Spellings match the canonical to_string() forms in descriptors/enums.hpp.
// Decode failures throw with the offending key and the accepted values;
// Holoscan's YAMLNodeParser catches, logs the node, and falls back to the
// parameter default, matching SDK behavior for built-in types.

namespace holoscan::radar::detail {

// Parse an enum from its canonical spelling; `candidates` enumerates every
// legal value. Throws naming the type and accepted spellings.
template <typename Enum, size_t N>
Enum parse_enum(const std::string& text, const char* type_name,
                const Enum (&candidates)[N]) {
  for (Enum candidate : candidates) {
    if (text == to_string(candidate)) {
      return candidate;
    }
  }
  std::string accepted;
  for (Enum candidate : candidates) {
    accepted += accepted.empty() ? "" : ", ";
    accepted += to_string(candidate);
  }
  throw std::invalid_argument(std::string(type_name) + ": unknown value '" +
                              text + "' (accepted: " + accepted + ")");
}

inline constexpr WindowType kAllWindowTypes[] = {
    WindowType::kNone,      WindowType::kHann,
    WindowType::kHamming,   WindowType::kBlackman,
    WindowType::kBlackmanHarris, WindowType::kKaiser,
    WindowType::kChebyshev, WindowType::kTaylor,
    WindowType::kFlatTop,   WindowType::kTukey,
    WindowType::kCustom,
};

inline constexpr FftNorm kAllFftNorms[] = {
    FftNorm::kBackward,
    FftNorm::kOrtho,
    FftNorm::kForward,
};

inline constexpr ClutterFilterMethod kAllClutterFilterMethods[] = {
    ClutterFilterMethod::kNone,
    ClutterFilterMethod::kMeanSubtraction,
    ClutterFilterMethod::kSingleDelay,
    ClutterFilterMethod::kDoubleDelay,
};

inline constexpr SnapshotEdgeMode kAllSnapshotEdgeModes[] = {
    SnapshotEdgeMode::kWrap,
    SnapshotEdgeMode::kClamp,
    SnapshotEdgeMode::kShift,
};

inline constexpr RepresentativePolicy kAllRepresentativePolicies[] = {
    RepresentativePolicy::kPeak,
    RepresentativePolicy::kMedoid,
    RepresentativePolicy::kFirst,
};

inline constexpr OverlapPolicy kAllOverlapPolicies[] = {
    OverlapPolicy::kDedup,
    OverlapPolicy::kAverage,
    OverlapPolicy::kKeepAll,
};

inline constexpr MultiplexingScheme kAllMultiplexingSchemes[] = {
    MultiplexingScheme::kTdm,
    MultiplexingScheme::kDdm,
};

// kUnspecified is deliberately absent: it is the spec/Parameter default
// that construction rejects, so it must not be spellable in YAML.
inline constexpr CoordinateFrameType kAllCoordinateFrameTypes[] = {
    CoordinateFrameType::kEcef,
    CoordinateFrameType::kEnu,
    CoordinateFrameType::kNed,
    CoordinateFrameType::kSceneLocal,
};

inline constexpr Ellipsoid kAllEllipsoids[] = {
    Ellipsoid::kWgs84,
};

inline constexpr FormationControl kAllFormationControls[] = {
    FormationControl::kEmitAndContinue,
    FormationControl::kEmitAndReset,
    FormationControl::kResetWithoutEmit,
    FormationControl::kResetThenEmit,
};

inline constexpr RangeInterpolation kAllRangeInterpolations[] = {
    RangeInterpolation::kNearest,
    RangeInterpolation::kLinear,
    RangeInterpolation::kSinc,
};

inline constexpr BackprojectionBackend kAllBackprojectionBackends[] = {
    BackprojectionBackend::kMatxDouble,
    BackprojectionBackend::kMatxDoublePhaseLut,
    BackprojectionBackend::kMatxMixed,
    BackprojectionBackend::kMatxMixedPhaseLut,
    BackprojectionBackend::kMatxFloatFloat,
    BackprojectionBackend::kMatxFloat,
    BackprojectionBackend::kMatxFloatPhaseLut,
    BackprojectionBackend::kMatxTaylorFast,
    BackprojectionBackend::kNvradarCudaFp32,
    BackprojectionBackend::kNvradarCudaFp32Hq,
    BackprojectionBackend::kNvradarCudaFp32HqFast,
    BackprojectionBackend::kNvradarFp64Reference,
    BackprojectionBackend::kNvradarBlockAnchored,
    BackprojectionBackend::kNvradarBlockAnchoredFp64Trajectory,
    BackprojectionBackend::kNvradarBlockAnchoredRefinedThroughput,
    BackprojectionBackend::kNvradarBlockAnchoredRefinedQuality,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughput,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedQuality,
    BackprojectionBackend::kNvradarBlockAnchoredPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredRefinedQualityPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVec,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedThroughputPhaseLutVecZ0,
    BackprojectionBackend::kNvradarBlockAnchoredExpandedQualityPhaseLut,
    BackprojectionBackend::kNvradarBlockAnchoredPhaseLutLegacy,
    BackprojectionBackend::kNvradarBlockAnchoredFp64TrajectoryPhaseLut,
};

inline constexpr GeometryChange kAllGeometryChanges[] = {
    GeometryChange::kReset,
    GeometryChange::kScroll,
};

inline constexpr ApertureEmitMode kAllApertureEmitModes[] = {
    ApertureEmitMode::kContinue,
    ApertureEmitMode::kReset,
    ApertureEmitMode::kReform,
};

// Decode a fixed-length YAML sequence into std::array, rejecting a missing
// key or the wrong length by name. The descriptors' (3,) and (2,) fields are
// arrays rather than vectors, so their length is a type invariant and not
// something each decode site should re-check by hand.
//
// The IsDefined() check comes first because yaml-cpp's absent-key node is a
// zombie that throws YAML::InvalidNode on almost any inspection, including
// IsSequence(). Reaching it would report "invalid node; first invalid key"
// instead of naming the descriptor and the field the config is missing.
template <typename T, size_t N>
std::array<T, N> decode_fixed(const YAML::Node& node, const char* type_name,
                              const char* field) {
  if (!node.IsDefined()) {
    throw std::invalid_argument(std::string(type_name) + ": " + field +
                                " is required");
  }
  if (!node.IsSequence() || node.size() != N) {
    throw std::invalid_argument(std::string(type_name) + ": " + field +
                                " must be a sequence of length " +
                                std::to_string(N));
  }
  std::array<T, N> values{};
  for (size_t i = 0; i < N; ++i) {
    values[i] = node[i].as<T>();
  }
  return values;
}

inline constexpr ScoreMethod kAllScoreMethods[] = {
    ScoreMethod::kWeightedSum,
    ScoreMethod::kWeightedL2,
    ScoreMethod::kWeightedMax,
    ScoreMethod::kCustom,
};

inline constexpr ArrayGeometryType kAllArrayGeometryTypes[] = {
    ArrayGeometryType::kUla,
    ArrayGeometryType::kUra,
    ArrayGeometryType::kUca,
    ArrayGeometryType::kConformal,
};

inline constexpr AngleConventionPreset kAllAngleConventionPresets[] = {
    AngleConventionPreset::kRadarBoresight,
    AngleConventionPreset::kRosRep103,
    AngleConventionPreset::kIso8855Flu,
    AngleConventionPreset::kAerospaceNed,
    AngleConventionPreset::kPhysicsSpherical,
    AngleConventionPreset::kGeodeticEnu,
    AngleConventionPreset::kGeodeticNed,
    AngleConventionPreset::kDirectionCosine,
    AngleConventionPreset::kCustom,
};

inline constexpr SignedAxis kAllSignedAxes[] = {
    SignedAxis::kPlusX,  SignedAxis::kMinusX, SignedAxis::kPlusY,
    SignedAxis::kMinusY, SignedAxis::kPlusZ,  SignedAxis::kMinusZ,
};

inline constexpr ElevationReference kAllElevationReferences[] = {
    ElevationReference::kHorizon,
    ElevationReference::kZenith,
};

inline constexpr PeakInterpolation kAllPeakInterpolations[] = {
    PeakInterpolation::kNone,
    PeakInterpolation::kParabolic,
};

}  // namespace holoscan::radar::detail

namespace YAML {

template <>
struct convert<holoscan::radar::WindowType> {
  static Node encode(const holoscan::radar::WindowType& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::WindowType& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "WindowType",
        holoscan::radar::detail::kAllWindowTypes);
    return true;
  }
};

template <>
struct convert<holoscan::radar::FftNorm> {
  static Node encode(const holoscan::radar::FftNorm& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::FftNorm& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "FftNorm",
        holoscan::radar::detail::kAllFftNorms);
    return true;
  }
};

template <>
struct convert<holoscan::radar::ClutterFilterMethod> {
  static Node encode(const holoscan::radar::ClutterFilterMethod& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::ClutterFilterMethod& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "ClutterFilterMethod",
        holoscan::radar::detail::kAllClutterFilterMethods);
    return true;
  }
};

template <>
struct convert<holoscan::radar::SnapshotEdgeMode> {
  static Node encode(const holoscan::radar::SnapshotEdgeMode& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::SnapshotEdgeMode& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "SnapshotEdgeMode",
        holoscan::radar::detail::kAllSnapshotEdgeModes);
    return true;
  }
};

template <>
struct convert<holoscan::radar::RepresentativePolicy> {
  static Node encode(const holoscan::radar::RepresentativePolicy& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::RepresentativePolicy& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "RepresentativePolicy",
        holoscan::radar::detail::kAllRepresentativePolicies);
    return true;
  }
};

template <>
struct convert<holoscan::radar::ScoreMethod> {
  static Node encode(const holoscan::radar::ScoreMethod& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::ScoreMethod& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "ScoreMethod",
        holoscan::radar::detail::kAllScoreMethods);
    return true;
  }
};

// WindowSpec accepts either a bare window-type scalar (`range_window: hann`)
// or a map with any subset of the struct's fields; omitted fields keep their
// defaults.
template <>
struct convert<holoscan::radar::WindowSpec> {
  static Node encode(const holoscan::radar::WindowSpec& spec) {
    Node node;
    node["type"] = spec.type;
    node["sidelobe_attenuation_db"] = spec.sidelobe_attenuation_db;
    if (spec.kaiser_beta.has_value()) {
      node["kaiser_beta"] = *spec.kaiser_beta;
    }
    node["taylor_nbar"] = spec.taylor_nbar;
    node["tukey_alpha"] = spec.tukey_alpha;
    if (!spec.coefficients.empty()) {
      node["coefficients"] = spec.coefficients;
    }
    node["coherent_gain_normalization"] = spec.coherent_gain_normalization;
    return node;
  }
  static bool decode(const Node& node, holoscan::radar::WindowSpec& spec) {
    if (node.IsScalar()) {
      spec = holoscan::radar::WindowSpec{};
      spec.type = node.as<holoscan::radar::WindowType>();
      return true;
    }
    if (!node.IsMap()) {
      return false;
    }
    spec = holoscan::radar::WindowSpec{};
    if (const Node type = node["type"]) {
      spec.type = type.as<holoscan::radar::WindowType>();
    }
    if (const Node db = node["sidelobe_attenuation_db"]) {
      spec.sidelobe_attenuation_db = db.as<double>();
    }
    if (const Node beta = node["kaiser_beta"]) {
      if (!beta.IsNull()) {
        spec.kaiser_beta = beta.as<double>();
      }
    }
    if (const Node nbar = node["taylor_nbar"]) {
      spec.taylor_nbar = nbar.as<int>();
    }
    if (const Node alpha = node["tukey_alpha"]) {
      spec.tukey_alpha = alpha.as<double>();
    }
    if (const Node coeffs = node["coefficients"]) {
      spec.coefficients = coeffs.as<std::vector<double>>();
    }
    if (const Node norm = node["coherent_gain_normalization"]) {
      spec.coherent_gain_normalization = norm.as<bool>();
    }
    return true;
  }
};

// FMCWWaveformDescriptor decodes through FMCWWaveformSpec, so construction
// validation (exactly-one-of bandwidth / chirp_slope, positive scalars)
// applies to YAML configs identically to native construction. The optional
// measured `reference_samples` are not expressible in YAML; encode() emits
// the analytic fields only.
template <>
struct convert<holoscan::radar::FMCWWaveformDescriptor> {
  static Node encode(const holoscan::radar::FMCWWaveformDescriptor& wf) {
    Node node;
    node["carrier_frequency"] = wf.carrier_frequency();
    node["bandwidth"] = wf.bandwidth();
    node["chirp_duration"] = wf.chirp_duration();
    node["sample_rate"] = wf.sample_rate();
    node["chirp_repetition_interval"] = wf.chirp_repetition_interval();
    return node;
  }
  static bool decode(const Node& node,
                     holoscan::radar::FMCWWaveformDescriptor& wf) {
    if (!node.IsMap()) {
      return false;
    }
    holoscan::radar::FMCWWaveformSpec spec;
    if (const Node fc = node["carrier_frequency"]) {
      spec.carrier_frequency = fc.as<double>();
    }
    if (const Node bw = node["bandwidth"]) {
      spec.bandwidth = bw.as<double>();
    }
    if (const Node slope = node["chirp_slope"]) {
      spec.chirp_slope = slope.as<double>();
    }
    if (const Node duration = node["chirp_duration"]) {
      spec.chirp_duration = duration.as<double>();
    }
    if (const Node rate = node["sample_rate"]) {
      spec.sample_rate = rate.as<double>();
    }
    if (const Node cri = node["chirp_repetition_interval"]) {
      spec.chirp_repetition_interval = cri.as<double>();
    }
    // Throws std::invalid_argument naming the offending field on an
    // incomplete or inconsistent waveform block.
    wf = holoscan::radar::FMCWWaveformDescriptor(std::move(spec));
    return true;
  }
};

template <>
struct convert<holoscan::radar::ArrayGeometryType> {
  static Node encode(const holoscan::radar::ArrayGeometryType& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::ArrayGeometryType& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "ArrayGeometryType",
        holoscan::radar::detail::kAllArrayGeometryTypes);
    return true;
  }
};

template <>
struct convert<holoscan::radar::AngleConventionPreset> {
  static Node encode(const holoscan::radar::AngleConventionPreset& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::AngleConventionPreset& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "AngleConventionPreset",
        holoscan::radar::detail::kAllAngleConventionPresets);
    return true;
  }
};

template <>
struct convert<holoscan::radar::SignedAxis> {
  static Node encode(const holoscan::radar::SignedAxis& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::SignedAxis& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "SignedAxis",
        holoscan::radar::detail::kAllSignedAxes);
    return true;
  }
};

template <>
struct convert<holoscan::radar::ElevationReference> {
  static Node encode(const holoscan::radar::ElevationReference& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::ElevationReference& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "ElevationReference",
        holoscan::radar::detail::kAllElevationReferences);
    return true;
  }
};

template <>
struct convert<holoscan::radar::PeakInterpolation> {
  static Node encode(const holoscan::radar::PeakInterpolation& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::PeakInterpolation& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "PeakInterpolation",
        holoscan::radar::detail::kAllPeakInterpolations);
    return true;
  }
};

template <>
struct convert<holoscan::radar::OverlapPolicy> {
  static Node encode(const holoscan::radar::OverlapPolicy& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::OverlapPolicy& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "OverlapPolicy",
        holoscan::radar::detail::kAllOverlapPolicies);
    return true;
  }
};

template <>
struct convert<holoscan::radar::MultiplexingScheme> {
  static Node encode(const holoscan::radar::MultiplexingScheme& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::MultiplexingScheme& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "MultiplexingScheme",
        holoscan::radar::detail::kAllMultiplexingSchemes);
    return true;
  }
};

template <>
struct convert<holoscan::radar::DDMCodeDescriptor> {
  static Node encode(const holoscan::radar::DDMCodeDescriptor& code) {
    Node node;
    node["num_tx"] = code.num_tx;
    if (!code.phase_codes.empty()) {
      Node codes = Node(NodeType::Sequence);
      for (const auto& z : code.phase_codes) {
        codes.push_back(Node(z.real()));
        codes.push_back(Node(z.imag()));
      }
      node["phase_codes"] = codes;
    }
    if (!code.doppler_offsets.empty()) {
      node["doppler_offsets"] = code.doppler_offsets;
    }
    node["guard_bands"] = code.guard_bands;
    return node;
  }
  static bool decode(const Node& node,
                     holoscan::radar::DDMCodeDescriptor& code) {
    if (!node.IsMap()) {
      return false;
    }
    code = holoscan::radar::DDMCodeDescriptor{};
    if (const Node n = node["num_tx"]) {
      code.num_tx = n.as<int64_t>();
    }
    if (const Node codes = node["phase_codes"]) {
      if (!codes.IsSequence()) {
        return false;
      }
      // The sequence is flat re, im pairs, so an odd length means a code the
      // config meant to state is half missing. Dropping the stray element
      // would decode successfully into a shorter code table.
      if (codes.size() % 2 != 0) {
        throw std::invalid_argument(
            "DDMCodeDescriptor: phase_codes must be a flat sequence of "
            "interleaved re, im pairs (even length)");
      }
      code.phase_codes.reserve(codes.size() / 2);
      for (size_t i = 0; i + 1 < codes.size(); i += 2) {
        code.phase_codes.emplace_back(codes[i].as<double>(),
                                    codes[i + 1].as<double>());
      }
    }
    if (const Node offsets = node["doppler_offsets"]) {
      code.doppler_offsets = offsets.as<std::vector<double>>();
    }
    if (const Node guard = node["guard_bands"]) {
      code.guard_bands = guard.as<int64_t>();
    }
    code.validate();
    return true;
  }
};

template <>
struct convert<holoscan::radar::ArrayGeometry> {
  static Node encode(const holoscan::radar::ArrayGeometry& geom) {
    Node node;
    node["element_positions"] = geom.element_positions;
    node["wavelength"] = geom.wavelength;
    if (geom.geometry_type.has_value()) {
      node["geometry_type"] = *geom.geometry_type;
    }
    return node;
  }
  static bool decode(const Node& node, holoscan::radar::ArrayGeometry& geom) {
    if (!node.IsMap()) {
      return false;
    }
    geom = holoscan::radar::ArrayGeometry{};
    if (const Node pos = node["element_positions"]) {
      geom.element_positions = pos.as<std::vector<double>>();
    }
    if (const Node wl = node["wavelength"]) {
      geom.wavelength = wl.as<double>();
    }
    if (const Node gt = node["geometry_type"]) {
      geom.geometry_type = gt.as<holoscan::radar::ArrayGeometryType>();
    }
    geom.validate();
    return true;
  }
};

template <>
struct convert<holoscan::radar::AngleConvention> {
  static Node encode(const holoscan::radar::AngleConvention& conv) {
    Node node;
    node["preset"] = conv.preset;
    node["boresight_axis"] = conv.boresight_axis;
    node["azimuth_positive_axis"] = conv.azimuth_positive_axis;
    node["elevation_positive_axis"] = conv.elevation_positive_axis;
    node["elevation_reference"] = conv.elevation_reference;
    return node;
  }
  static bool decode(const Node& node, holoscan::radar::AngleConvention& conv) {
    if (!node.IsMap()) {
      return false;
    }
    conv = holoscan::radar::AngleConvention{};
    if (const Node preset = node["preset"]) {
      conv.preset = preset.as<holoscan::radar::AngleConventionPreset>();
    }
    if (const Node bore = node["boresight_axis"]) {
      conv.boresight_axis = bore.as<holoscan::radar::SignedAxis>();
    }
    if (const Node az = node["azimuth_positive_axis"]) {
      conv.azimuth_positive_axis = az.as<holoscan::radar::SignedAxis>();
    }
    if (const Node el = node["elevation_positive_axis"]) {
      conv.elevation_positive_axis = el.as<holoscan::radar::SignedAxis>();
    }
    if (const Node ref = node["elevation_reference"]) {
      conv.elevation_reference = ref.as<holoscan::radar::ElevationReference>();
    }
    return true;
  }
};

template <>
struct convert<holoscan::radar::TXSchedule> {
  static Node encode(const holoscan::radar::TXSchedule& schedule) {
    Node node;
    node["num_tx"] = schedule.num_tx;
    if (!schedule.tx_order.empty()) {
      node["tx_order"] = schedule.tx_order;
    }
    if (!schedule.chirp_tx_index.empty()) {
      node["chirp_tx_index"] = schedule.chirp_tx_index;
    }
    return node;
  }
  static bool decode(const Node& node, holoscan::radar::TXSchedule& schedule) {
    if (!node.IsMap()) {
      return false;
    }
    schedule = holoscan::radar::TXSchedule{};
    if (const Node ntx = node["num_tx"]) {
      schedule.num_tx = ntx.as<int64_t>();
    }
    if (const Node order = node["tx_order"]) {
      schedule.tx_order = order.as<std::vector<int64_t>>();
    }
    if (const Node idx = node["chirp_tx_index"]) {
      schedule.chirp_tx_index = idx.as<std::vector<int64_t>>();
    }
    schedule.validate();
    return true;
  }
};

template <>
struct convert<holoscan::radar::CoordinateFrameType> {
  static Node encode(const holoscan::radar::CoordinateFrameType& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::CoordinateFrameType& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "CoordinateFrameType",
        holoscan::radar::detail::kAllCoordinateFrameTypes);
    return true;
  }
};

template <>
struct convert<holoscan::radar::Ellipsoid> {
  static Node encode(const holoscan::radar::Ellipsoid& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node, holoscan::radar::Ellipsoid& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "Ellipsoid",
        holoscan::radar::detail::kAllEllipsoids);
    return true;
  }
};

// CoordinateFrame accepts either a bare frame-family scalar
// (`coordinate_frame: ecef`) or a map. The scalar shorthand covers the
// simulation and unit-test case, where scene_local carries no geodetic
// anchor and there is nothing else to say.
template <>
struct convert<holoscan::radar::CoordinateFrame> {
  static Node encode(const holoscan::radar::CoordinateFrame& frame) {
    Node node;
    node["frame"] = frame.frame();
    if (frame.origin_llh().has_value()) {
      node["origin_llh"] = std::vector<double>(frame.origin_llh()->begin(),
                                               frame.origin_llh()->end());
    }
    if (frame.epoch().has_value()) {
      node["epoch"] = *frame.epoch();
    }
    node["ellipsoid"] = frame.ellipsoid();
    return node;
  }
  static bool decode(const Node& node,
                     holoscan::radar::CoordinateFrame& frame) {
    holoscan::radar::CoordinateFrameSpec spec;
    if (node.IsScalar()) {
      spec.frame = node.as<holoscan::radar::CoordinateFrameType>();
      frame = holoscan::radar::CoordinateFrame(std::move(spec));
      return true;
    }
    if (!node.IsMap()) {
      return false;
    }
    if (const Node family = node["frame"]) {
      spec.frame = family.as<holoscan::radar::CoordinateFrameType>();
    }
    if (const Node llh = node["origin_llh"]) {
      spec.origin_llh = holoscan::radar::detail::decode_fixed<double, 3>(
          llh, "CoordinateFrame", "origin_llh");
    }
    if (const Node epoch = node["epoch"]) {
      if (!epoch.IsNull()) {
        spec.epoch = epoch.as<double>();
      }
    }
    if (const Node ellipsoid = node["ellipsoid"]) {
      spec.ellipsoid = ellipsoid.as<holoscan::radar::Ellipsoid>();
    }
    // Throws std::invalid_argument naming the offending field on a frame
    // block that omits `frame` or puts origin_llh out of range.
    frame = holoscan::radar::CoordinateFrame(std::move(spec));
    return true;
  }
};

// SampleGeometry decodes through SampleGeometrySpec, so axis normalization
// and the positive spacing / count checks apply to YAML configs identically
// to native construction. Every field is required: there is no sensible
// default image lattice.
template <>
struct convert<holoscan::radar::SampleGeometry> {
  static Node encode(const holoscan::radar::SampleGeometry& geometry) {
    Node node;
    node["origin"] =
        std::vector<double>(geometry.origin().begin(), geometry.origin().end());
    node["axis_u"] =
        std::vector<double>(geometry.axis_u().begin(), geometry.axis_u().end());
    node["axis_v"] =
        std::vector<double>(geometry.axis_v().begin(), geometry.axis_v().end());
    node["spacing"] = std::vector<double>(geometry.spacing().begin(),
                                          geometry.spacing().end());
    node["counts"] = std::vector<int64_t>(geometry.counts().begin(),
                                          geometry.counts().end());
    return node;
  }
  static bool decode(const Node& node,
                     holoscan::radar::SampleGeometry& geometry) {
    if (!node.IsMap()) {
      return false;
    }
    holoscan::radar::SampleGeometrySpec spec;
    spec.origin = holoscan::radar::detail::decode_fixed<double, 3>(
        node["origin"], "SampleGeometry", "origin");
    spec.axis_u = holoscan::radar::detail::decode_fixed<double, 3>(
        node["axis_u"], "SampleGeometry", "axis_u");
    spec.axis_v = holoscan::radar::detail::decode_fixed<double, 3>(
        node["axis_v"], "SampleGeometry", "axis_v");
    spec.spacing = holoscan::radar::detail::decode_fixed<double, 2>(
        node["spacing"], "SampleGeometry", "spacing");
    spec.counts = holoscan::radar::detail::decode_fixed<int64_t, 2>(
        node["counts"], "SampleGeometry", "counts");
    geometry = holoscan::radar::SampleGeometry(std::move(spec));
    return true;
  }
};

// LFMWaveformDescriptor decodes through LFMWaveformSpec, mirroring the
// FMCWWaveformDescriptor converter above field for field — the two types
// are intended to converge (D24), and the converters diverging would be a
// second thing to merge.
template <>
struct convert<holoscan::radar::LFMWaveformDescriptor> {
  static Node encode(const holoscan::radar::LFMWaveformDescriptor& wf) {
    Node node;
    node["carrier_frequency"] = wf.carrier_frequency();
    node["bandwidth"] = wf.bandwidth();
    node["chirp_duration"] = wf.chirp_duration();
    node["sample_rate"] = wf.sample_rate();
    node["chirp_repetition_interval"] = wf.chirp_repetition_interval();
    return node;
  }
  static bool decode(const Node& node,
                     holoscan::radar::LFMWaveformDescriptor& wf) {
    if (!node.IsMap()) {
      return false;
    }
    holoscan::radar::LFMWaveformSpec spec;
    if (const Node fc = node["carrier_frequency"]) {
      spec.carrier_frequency = fc.as<double>();
    }
    if (const Node bw = node["bandwidth"]) {
      spec.bandwidth = bw.as<double>();
    }
    if (const Node slope = node["chirp_slope"]) {
      spec.chirp_slope = slope.as<double>();
    }
    if (const Node duration = node["chirp_duration"]) {
      spec.chirp_duration = duration.as<double>();
    }
    if (const Node rate = node["sample_rate"]) {
      spec.sample_rate = rate.as<double>();
    }
    if (const Node cri = node["chirp_repetition_interval"]) {
      spec.chirp_repetition_interval = cri.as<double>();
    }
    // Throws std::invalid_argument naming the offending field on an
    // incomplete or inconsistent waveform block.
    wf = holoscan::radar::LFMWaveformDescriptor(std::move(spec));
    return true;
  }
};

template <>
struct convert<holoscan::radar::FormationControl> {
  static Node encode(const holoscan::radar::FormationControl& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::FormationControl& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "FormationControl",
        holoscan::radar::detail::kAllFormationControls);
    return true;
  }
};

template <>
struct convert<holoscan::radar::RangeInterpolation> {
  static Node encode(const holoscan::radar::RangeInterpolation& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::RangeInterpolation& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "RangeInterpolation",
        holoscan::radar::detail::kAllRangeInterpolations);
    return true;
  }
};

template <>
struct convert<holoscan::radar::BackprojectionBackend> {
  static Node encode(const holoscan::radar::BackprojectionBackend& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::BackprojectionBackend& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "BackprojectionBackend",
        holoscan::radar::detail::kAllBackprojectionBackends);
    return true;
  }
};

template <>
struct convert<holoscan::radar::GeometryChange> {
  static Node encode(const holoscan::radar::GeometryChange& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::GeometryChange& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "GeometryChange",
        holoscan::radar::detail::kAllGeometryChanges);
    return true;
  }
};

template <>
struct convert<holoscan::radar::ApertureEmitMode> {
  static Node encode(const holoscan::radar::ApertureEmitMode& value) {
    return Node(std::string(holoscan::radar::to_string(value)));
  }
  static bool decode(const Node& node,
                     holoscan::radar::ApertureEmitMode& value) {
    if (!node.IsScalar()) {
      return false;
    }
    value = holoscan::radar::detail::parse_enum(
        node.as<std::string>(), "ApertureEmitMode",
        holoscan::radar::detail::kAllApertureEmitModes);
    return true;
  }
};

}  // namespace YAML
