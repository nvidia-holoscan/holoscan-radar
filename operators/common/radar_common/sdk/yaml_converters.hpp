// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"

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

}  // namespace YAML
