// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "utils/perf_payloads.hpp"
#include "utils/perf_profiles.hpp"

namespace holoscan::radar::perf {

inline WindowType recipe_window_type(std::string_view value) {
  if (value == "none")
    return WindowType::kNone;
  if (value == "hann")
    return WindowType::kHann;
  if (value == "hamming")
    return WindowType::kHamming;
  throw std::invalid_argument("unknown recipe window: " + std::string(value));
}

inline WindowSpec recipe_window(WindowType type) {
  WindowSpec result;
  result.type = type;
  return result;
}

inline FftNorm recipe_fft_norm(std::string_view value) {
  if (value == "backward")
    return FftNorm::kBackward;
  if (value == "ortho")
    return FftNorm::kOrtho;
  if (value == "forward")
    return FftNorm::kForward;
  throw std::invalid_argument("unknown recipe FFT normalization: " + std::string(value));
}

inline SnapshotEdgeMode recipe_snapshot_edge(std::string_view value) {
  if (value == "wrap")
    return SnapshotEdgeMode::kWrap;
  if (value == "clamp")
    return SnapshotEdgeMode::kClamp;
  if (value == "shift")
    return SnapshotEdgeMode::kShift;
  throw std::invalid_argument("unknown snapshot edge mode: " + std::string(value));
}

inline PeakInterpolation recipe_peak_interpolation(std::string_view value) {
  if (value == "none")
    return PeakInterpolation::kNone;
  if (value == "parabolic")
    return PeakInterpolation::kParabolic;
  throw std::invalid_argument("unknown peak interpolation: " + std::string(value));
}

inline ArrayGeometry recipe_ula_geometry(const YAML::Node& node, int64_t channels) {
  if (node["factory"].as<std::string>() != "ula_geometry")
    throw std::invalid_argument("unknown recipe array geometry factory");
  const auto spacing = node["spacing_wavelengths"].as<double>();
  const auto wavelength = node["wavelength"].as<double>();
  if (channels <= 0 || !std::isfinite(spacing) || spacing <= 0.0 || !std::isfinite(wavelength) ||
      wavelength <= 0.0)
    throw std::invalid_argument("recipe ULA needs positive channels, spacing, and wavelength");
  return make_ula_geometry(channels, spacing, wavelength);
}

inline std::vector<double> recipe_coordinates(const YAML::Node& node,
                                              const std::map<std::string, int64_t>& values,
                                              double shift = 0.0) {
  if (node["factory"].as<std::string>() != "linspace")
    throw std::invalid_argument("unknown recipe coordinate factory");
  const auto axis = node["count"].as<std::string>();
  const auto start = node["start"].as<double>() + shift;
  const auto stop = node["stop"].as<double>() + shift;
  if (!std::isfinite(start) || !std::isfinite(stop))
    throw std::invalid_argument("recipe coordinates need finite endpoints");
  return linspace(start, stop, values.at(axis));
}

inline FMCWWaveformDescriptor recipe_fmcw_waveform(
    const YAML::Node& node, std::optional<double> pri_override = std::nullopt) {
  require_keys(
      node,
      {"factory", "carrier_frequency", "bandwidth", "chirp_duration", "sample_rate", "pri_seconds"},
      "waveform");
  if (node["factory"].as<std::string>() != "valid_fmcw_waveform")
    throw std::invalid_argument("unknown recipe FMCW waveform factory");
  FMCWWaveformSpec spec;
  spec.carrier_frequency = node["carrier_frequency"].as<double>();
  const auto bandwidth = node["bandwidth"].as<double>();
  spec.bandwidth = bandwidth;
  spec.chirp_duration = node["chirp_duration"].as<double>();
  spec.sample_rate = node["sample_rate"].as<double>();
  spec.chirp_repetition_interval = pri_override.value_or(node["pri_seconds"].as<double>());
  if (!std::isfinite(spec.carrier_frequency) || spec.carrier_frequency <= 0.0 ||
      !std::isfinite(bandwidth) || bandwidth <= 0.0 || !std::isfinite(spec.chirp_duration) ||
      spec.chirp_duration <= 0.0 || !std::isfinite(spec.sample_rate) || spec.sample_rate <= 0.0 ||
      !std::isfinite(spec.chirp_repetition_interval) ||
      spec.chirp_repetition_interval <= spec.chirp_duration)
    throw std::invalid_argument("recipe FMCW waveform needs positive fields and PRI after chirp");
  return FMCWWaveformDescriptor(spec);
}

}  // namespace holoscan::radar::perf
