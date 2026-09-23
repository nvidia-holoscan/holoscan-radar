// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "radar_app/app/multi_profile_source.hpp"
#include "radar_app/fmcw/mimo_config.hpp"

namespace holoscan::radar::app::multi_waveform {

struct Config : mimo::TdmConfig {
  explicit Config(mimo::TdmConfig base) : mimo::TdmConfig(std::move(base)) {}

  // Resolved hardware/DSP descriptors for every authored waveform.
  std::vector<TdmTransmitProfile> profiles;
  // Indices into profiles, preserving the authored cyclic schedule.
  std::vector<std::size_t> profile_sequence;
  // Fixed dashboard selection and its longest recurrence interval.
  std::size_t visualization_profile_index{0U};
  std::size_t visualization_profile_max_gap_frames{1U};
  // Bounds and cadence for the independently ticking TX/RX sources.
  std::size_t max_frames_in_flight{4U};
  double rx_poll_period{0.001};
  int simulated_receive_polls{2};
};

// Number of source activations Holoviz must retain the selected profile.
inline std::size_t maximum_cyclic_profile_gap(const std::vector<std::size_t>& sequence,
                                              std::size_t profile_index) {
  std::vector<std::size_t> positions;
  for (std::size_t i = 0; i < sequence.size(); ++i) {
    if (sequence[i] == profile_index) {
      positions.push_back(i);
    }
  }
  if (positions.empty()) {
    throw std::runtime_error("app.visualization.profile must appear in app.profile_sequence");
  }
  std::size_t max_gap = positions.front() + sequence.size() - positions.back();
  for (std::size_t i = 1; i < positions.size(); ++i) {
    max_gap = std::max(max_gap, positions[i] - positions[i - 1U]);
  }
  return max_gap;
}

inline double require_scalar(const YAML::Node& node, const char* field) {
  if (!node[field]) {
    throw std::runtime_error(std::string("app.profiles[].") + field + " is required");
  }
  const double value = node[field].as<double>();
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::runtime_error(std::string("app.profiles[].") + field + " must be finite and > 0");
  }
  return value;
}

inline int64_t require_integer(const YAML::Node& node, const char* field) {
  if (!node[field]) {
    throw std::runtime_error(std::string("app.profiles[].") + field + " is required");
  }
  const int64_t value = node[field].as<int64_t>();
  if (value <= 0) {
    throw std::runtime_error(std::string("app.profiles[].") + field + " must be > 0");
  }
  return value;
}

inline Config load_config(const std::filesystem::path& path) {
  const YAML::Node root = YAML::LoadFile(path.string());
  const YAML::Node app = root["app"];
  const YAML::Node profiles = app["profiles"];
  const YAML::Node sequence = app["profile_sequence"];
  // Profiles are the sole authority for waveform and cube shape.
  if (!profiles || !profiles.IsSequence() || profiles.size() == 0U) {
    throw std::runtime_error("app.profiles must be a non-empty sequence");
  }
  if (!sequence || !sequence.IsSequence() || sequence.size() == 0U) {
    throw std::runtime_error("app.profile_sequence must be a non-empty sequence");
  }
  if (app["waveform"] || app["cube"] ||
      (app["tdm"] && app["tdm"]["physical_chirp_repetition_interval"])) {
    throw std::runtime_error(
        "multi-profile waveform, cube, and physical PRI belong only in app.profiles");
  }
  // Decode source coordination bounds independently of DSP parameters.
  const std::size_t max_frames_in_flight =
      app["max_frames_in_flight"] ? app["max_frames_in_flight"].as<std::size_t>() : 4U;
  if (max_frames_in_flight == 0U) {
    throw std::runtime_error("app.max_frames_in_flight must be > 0");
  }
  const double rx_poll_period = app["rx_poll_period"] ? app["rx_poll_period"].as<double>() : 0.001;
  mimo::require_finite_positive("app.rx_poll_period", rx_poll_period);
  const int simulated_receive_polls =
      app["simulated_receive_polls"] ? app["simulated_receive_polls"].as<int>() : 2;
  if (simulated_receive_polls < 0) {
    throw std::runtime_error("app.simulated_receive_polls must be >= 0");
  }

  // Resolve profiles once, derive both physical and effective-TDM timing,
  // and accumulate pool maxima across every possible frame shape.
  const int64_t num_tx = app["tx_array"]["num_tx"].as<int64_t>();
  std::unordered_map<std::string, std::size_t> by_name;
  std::vector<TdmTransmitProfile> decoded_profiles;
  std::vector<std::size_t> decoded_sequence;
  int64_t max_chirps = 0;
  int64_t max_samples = 0;
  double longest_cpi = 0.0;
  double carrier = 0.0;
  decoded_profiles.reserve(profiles.size());
  for (const YAML::Node& authored : profiles) {
    const std::string name = authored["name"] ? authored["name"].as<std::string>() : "";
    if (name.empty() || by_name.count(name) != 0U) {
      throw std::runtime_error("app.profiles names must be non-empty and unique");
    }
    const double fc = require_scalar(authored, "carrier_frequency");
    const double bandwidth = require_scalar(authored, "bandwidth");
    const double duration = require_scalar(authored, "chirp_duration");
    const double sample_rate = require_scalar(authored, "sample_rate");
    const double physical_pri = require_scalar(authored, "physical_chirp_repetition_interval");
    const int64_t chirps = require_integer(authored, "chirps_per_tx");
    const int64_t samples = require_integer(authored, "samples_per_chirp");
    if (physical_pri < duration) {
      throw std::runtime_error("profile physical PRI must be >= chirp duration");
    }
    const double expected_samples = sample_rate * duration;
    if (std::abs(expected_samples - static_cast<double>(samples)) > 1e-9) {
      throw std::runtime_error("profile samples_per_chirp must equal sample_rate * chirp_duration");
    }
    if (carrier == 0.0) {
      carrier = fc;
    } else if (fc != carrier) {
      throw std::runtime_error("all profiles must use the same carrier frequency");
    }

    TdmTransmitProfile profile;
    profile.name = name;
    profile.waveform = FMCWWaveformDescriptor(FMCWWaveformSpec{
        .carrier_frequency = fc,
        .bandwidth = bandwidth,
        .chirp_duration = duration,
        .sample_rate = sample_rate,
        .chirp_repetition_interval = physical_pri * static_cast<double>(num_tx),
    });
    profile.physical_waveform = sim::SimWaveform{
        .carrier_frequency = fc,
        .bandwidth = bandwidth,
        .chirp_duration = duration,
        .sample_rate = sample_rate,
        .chirp_repetition_interval = physical_pri,
    };
    profile.chirps_per_tx = chirps;
    profile.samples_per_chirp = samples;
    by_name.emplace(name, decoded_profiles.size());
    decoded_profiles.push_back(std::move(profile));
    max_chirps = std::max(max_chirps, chirps);
    max_samples = std::max(max_samples, samples);
    longest_cpi = std::max(longest_cpi, physical_pri * static_cast<double>(num_tx * chirps));
  }
  // Convert the authored name schedule to stable profile indices.
  for (const YAML::Node& entry : sequence) {
    const std::string name = entry.as<std::string>();
    const auto found = by_name.find(name);
    if (found == by_name.end()) {
      throw std::runtime_error("app.profile_sequence references unknown profile: " + name);
    }
    decoded_sequence.push_back(found->second);
  }

  // Reuse the ordinary TDM loader with the first profile as a temporary
  // structural baseline; per-frame waveform inputs replace it below.
  mimo::AppParams baseline;
  baseline.waveform = decoded_profiles.front().waveform;
  baseline.cube = mimo::CubeConfig{.num_chirps = decoded_profiles.front().chirps_per_tx,
                                   .samples_per_chirp = decoded_profiles.front().samples_per_chirp};
  baseline.tdm = mimo::TdmSimulationConfig{
      .physical_chirp_repetition_interval =
          decoded_profiles.front().physical_waveform.chirp_repetition_interval,
      .tx_order = {}};
  Config result{mimo::load_tdm_config(path, std::move(baseline))};
  // Resolve fixed dashboard selection and retention from the full cycle.
  if (result.visualization.profile.empty()) {
    result.visualization.profile = decoded_profiles[decoded_sequence.front()].name;
  }
  const auto selected = by_name.find(result.visualization.profile);
  if (selected == by_name.end()) {
    throw std::runtime_error("app.visualization.profile references unknown profile: " +
                             result.visualization.profile);
  }
  result.visualization_profile_index = selected->second;
  result.visualization_profile_max_gap_frames =
      maximum_cyclic_profile_gap(decoded_sequence, result.visualization_profile_index);

  result.profiles = std::move(decoded_profiles);
  result.profile_sequence = std::move(decoded_sequence);
  result.max_frames_in_flight = max_frames_in_flight;
  result.rx_poll_period = rx_poll_period;
  result.simulated_receive_polls = simulated_receive_polls;
  if (result.frame_period <= longest_cpi) {
    throw std::runtime_error("app.run.frame_period must exceed the longest complete TDM CPI");
  }

  // Clear static waveform defaults so the Exact TX-applied descriptor is
  // the only waveform authority for every processing activation.
  auto range_params = result.range_params.value();
  range_params.waveform = FMCWWaveformDescriptor{};
  result.range_params = holoscan::detail::PreparedParamsAccess::make(
      std::move(range_params),
      holoscan::detail::PreparedParamsAccess::evidence(result.range_params));
  auto doppler_params = result.doppler_params.value();
  doppler_params.waveform = FMCWWaveformDescriptor{};
  result.doppler_params = holoscan::detail::PreparedParamsAccess::make(
      std::move(doppler_params),
      holoscan::detail::PreparedParamsAccess::evidence(result.doppler_params));
  auto compensation_params = result.compensation_params.value();
  compensation_params.waveform = FMCWWaveformDescriptor{};
  result.compensation_params = holoscan::detail::PreparedParamsAccess::make(
      std::move(compensation_params),
      holoscan::detail::PreparedParamsAccess::evidence(result.compensation_params));

  // Tensor pools must admit the largest shape from any scheduled profile.
  result.extents.max_chirps = max_chirps;
  result.extents.max_interleaved_chirps = num_tx * max_chirps;
  result.extents.max_samples_per_chirp = max_samples;
  result.extents.max_range_bins = max_samples;
  result.extents.max_doppler_bins = max_chirps;
  const int64_t max_cells = max_chirps * max_samples;
  result.extents.max_detections = max_cells;
  result.extents.max_items = max_cells;
  result.extents.max_points = max_cells;
  return result;
}

}  // namespace holoscan::radar::app::multi_waveform
