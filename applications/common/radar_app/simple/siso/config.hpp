// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/config/io.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/error.hpp>

#include "clustering/cluster_dbscan_op/params_config.hpp"
#include "detection/cfar_detector_op/params_config.hpp"
#include "detection/detection_reduction_op/params_config.hpp"
#include "fmcw/clutter_filter_op/params_config.hpp"
#include "fmcw/fmcw_doppler_processor_op/params_config.hpp"
#include "fmcw/fmcw_range_processor_op/params_config.hpp"
#include "radar_app/app/run_config.hpp"
#include "radar_app/simple/siso/config_schema.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sim/scene.hpp"

namespace holoscan::radar::app::siso {

struct RuntimeConfig : CommonRunConfig {
  sim::SimScene scene;
  MaxTensorExtents extents;
  ::holoscan::radar::app::VisualizationConfig visualization;
};

struct Config : RuntimeConfig {
  holoscan::PreparedParams<FMCWRangeProcessorParams> range_params;
  holoscan::PreparedParams<ClutterFilterParams> clutter_params;
  holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler_params;
  holoscan::PreparedParams<CFARParams> cfar_params;
  holoscan::PreparedParams<FeatureAssemblyParams> assembly_params;
  holoscan::PreparedParams<ClusterDBSCANParams> dbscan_params;
  holoscan::PreparedParams<DetectionReductionParams> reduction_params;
};

template <typename T>
inline void require_positive(std::string_view key, T value) {
  if (value <= 0) {
    throw std::runtime_error(std::string(key) + " must be > 0, got " +
                             std::to_string(value));
  }
}

inline void validate_frame_period(double seconds) {
  if (!std::isfinite(seconds)) {
    throw std::runtime_error("app.run.frame_period must be finite");
  }
  const double nanoseconds = seconds * 1e9;
  if (nanoseconds < 1.0 ||
      nanoseconds > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "app.run.frame_period must be within [1 ns, INT64_MAX ns], got " +
        std::to_string(seconds) + " s");
  }
}

template <typename Params>
inline holoscan::PreparedParams<Params> decode_params(
    const holoscan::config::Document& document, std::string_view path,
    Params baseline = Params{}) {
  auto decoded = holoscan::params::try_decode<Params>(
      document, path, std::move(baseline));
  if (!decoded) {
    throw std::runtime_error(
        std::string(path) + ": " +
        holoscan::params::format_error(decoded.error()));
  }
  return std::move(*decoded);
}

inline void reject_authored(const holoscan::config::Document& document,
                            std::string_view path) {
  auto value = document.find(path);
  if (!value) {
    throw std::runtime_error(holoscan::config::format_error(value.error()));
  }
  if (value->has_value()) {
    throw std::runtime_error(
        std::string(path) + " is app-derived and must not be authored");
  }
}

inline holoscan::config::Document load_document(
    const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("config file not found: " + path.string());
  }
  auto loaded = holoscan::config_io::load(
      path, holoscan::config_io::Format::kYaml);
  if (!loaded) {
    throw std::runtime_error(holoscan::config::format_error(loaded.error()));
  }
  return std::move(*loaded);
}

template <typename Enum>
inline holoscan::config::Document normalize_enum_field(
    holoscan::config::Document document, std::string_view path) {
  auto found = document.find(path);
  if (!found) {
    throw std::runtime_error(holoscan::config::format_error(found.error()));
  }
  if (!found->has_value()) {
    return document;
  }
  auto decoded = document.get<Enum>(path);
  if (!decoded) {
    throw std::runtime_error(
        holoscan::config::format_error(decoded.error()));
  }
  using Underlying = std::underlying_type_t<Enum>;
  auto replaced = document.replace(path, static_cast<Underlying>(*decoded));
  if (!replaced) {
    throw std::runtime_error(
        holoscan::config::format_error(replaced.error()));
  }
  return std::move(*replaced);
}

inline holoscan::config::Document normalize_operator_enums(
    holoscan::config::Document document) {
  document = normalize_enum_field<FftNorm>(
      std::move(document), "/range_processor/fft_norm");
  document = normalize_enum_field<ClutterFilterMethod>(
      std::move(document), "/clutter_filter/method");
  document = normalize_enum_field<FftNorm>(
      std::move(document), "/doppler_processor/fft_norm");
  document = normalize_enum_field<CfarVariant>(
      std::move(document), "/cfar_detector/cfar_variant");
  document = normalize_enum_field<CfarPeakMode>(
      std::move(document), "/cfar_detector/peak_mode");
  document = normalize_enum_field<RepresentativePolicy>(
      std::move(document), "/reduction/representative");
  document = normalize_enum_field<ScoreMethod>(
      std::move(document), "/reduction/score_method");
  return document;
}

inline RuntimeConfig make_runtime_config(const AppParams& authored) {
  RuntimeConfig result;
  result.num_frames = authored.run.num_frames;
  result.frame_period = authored.run.frame_period;
  result.meter_cadence = authored.run.meter_cadence;
  result.meter_log_every = authored.run.meter_log_every;
  result.visualization = authored.visualization;
  result.visualize = authored.visualization.enabled;
  result.viz.window_width = authored.visualization.window_width;
  result.viz.window_height = authored.visualization.window_height;

  require_positive("app.run.num_frames", result.num_frames);
  validate_frame_period(result.frame_period);
  if (result.meter_log_every < 0) {
    throw std::runtime_error("app.run.meter_log_every must be >= 0");
  }
  require_positive("app.cube.num_chirps", authored.cube.num_chirps);
  require_positive("app.cube.samples_per_chirp",
                   authored.cube.samples_per_chirp);
  require_positive("app.visualization.window_width",
                   result.visualization.window_width);
  require_positive("app.visualization.window_height",
                   result.visualization.window_height);
  if (result.visualization.window_width >
          std::numeric_limits<std::uint32_t>::max() ||
      result.visualization.window_height >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(
        "app.visualization window dimensions exceed uint32");
  }
  require_positive("app.visualization.display_rate_hz",
                   result.visualization.display_rate_hz);
  require_positive("app.visualization.dynamic_range_db",
                   result.visualization.dynamic_range_db);
  require_positive("app.visualization.detection_radius",
                   result.visualization.detection_radius);
  require_positive("app.visualization.history_frames",
                   result.visualization.history_frames);
  require_positive("app.visualization.detection_trail_frames",
                   result.visualization.detection_trail_frames);
  if (result.visualization.enabled &&
      (result.visualization.window_width < 640 ||
       result.visualization.window_height < 360)) {
    throw std::runtime_error(
        "app.visualization dashboard requires at least a 640x360 window");
  }
  if (result.visualization.history_frames < 2) {
    throw std::runtime_error(
        "app.visualization.history_frames must be >= 2");
  }
  if (result.visualization.window_title.empty()) {
    throw std::runtime_error(
        "app.visualization.window_title must not be empty");
  }
  if (!std::isfinite(result.visualization.display_rate_hz) ||
      !std::isfinite(result.visualization.dynamic_range_db) ||
      !std::isfinite(result.visualization.detection_radius)) {
    throw std::runtime_error(
        "app.visualization floating-point values must be finite");
  }

  const auto& waveform = authored.waveform;
  result.scene.waveform.carrier_frequency = waveform.carrier_frequency();
  result.scene.waveform.bandwidth = waveform.bandwidth();
  result.scene.waveform.chirp_duration = waveform.chirp_duration();
  result.scene.waveform.sample_rate = waveform.sample_rate();
  result.scene.waveform.chirp_repetition_interval =
      waveform.chirp_repetition_interval();
  result.scene.num_chirps = authored.cube.num_chirps;
  result.scene.samples_per_chirp = authored.cube.samples_per_chirp;
  result.scene.snr_db = authored.noise_snr_db;

  result.scene.targets.reserve(authored.targets.size());
  for (const TargetConfig& target : authored.targets) {
    if (!std::isfinite(target.range) || target.range <= 0.0 ||
        !std::isfinite(target.velocity) ||
        !std::isfinite(target.amplitude) || target.amplitude <= 0.0) {
      throw std::runtime_error(
          "app.targets entries require finite range > 0, finite velocity, "
          "and finite amplitude > 0");
    }
    result.scene.targets.push_back(sim::PointTarget{
        sim::Vec3{target.range, 0.0, 0.0},
        sim::Vec3{target.velocity, 0.0, 0.0},
        {target.amplitude, 0.0}});
  }
  if (result.scene.targets.empty()) {
    throw std::runtime_error("siso: app.targets must not be empty");
  }
  return result;
}

inline MaxTensorExtents make_extents(
    const RuntimeConfig& runtime, const FMCWRangeProcessorParams& range,
    const FMCWDopplerProcessorParams& doppler,
    const ClusterDBSCANParams& dbscan, int64_t max_features) {
  MaxTensorExtents result;
  const int64_t num_range =
      range.nfft > 0 ? range.nfft : runtime.scene.samples_per_chirp;
  const int64_t num_doppler =
      doppler.nfft > 0 ? doppler.nfft : runtime.scene.num_chirps;
  const int64_t map_cells = num_doppler * num_range;
  result.max_chirps = runtime.scene.num_chirps;
  result.max_samples_per_chirp = runtime.scene.samples_per_chirp;
  result.max_range_bins = num_range;
  result.max_doppler_bins = num_doppler;
  result.max_detections = map_cells;
  result.max_items = map_cells;
  result.max_points = map_cells;
  result.max_features = max_features;
  result.max_clusters =
      dbscan.max_clusters > 0 ? dbscan.max_clusters : map_cells;
  return result;
}

inline Config load_config(const std::filesystem::path& path) {
  auto document = normalize_operator_enums(load_document(path));
  const auto app_params = decode_params<AppParams>(document, "/app");
  RuntimeConfig runtime = make_runtime_config(*app_params);
  auto range = decode_params<FMCWRangeProcessorParams>(
      document, "/range_processor");
  auto clutter = decode_params<ClutterFilterParams>(
      document, "/clutter_filter");
  auto doppler = decode_params<FMCWDopplerProcessorParams>(
      document, "/doppler_processor");
  reject_authored(document, "/cfar_detector/nci_fold");
  CFARParams cfar_baseline;
  cfar_baseline.nci_fold = 1;
  auto cfar = decode_params<CFARParams>(
      document, "/cfar_detector", std::move(cfar_baseline));
  auto assembly = decode_params<FeatureAssemblyParams>(
      document, "/feature_assembly");
  auto dbscan = decode_params<ClusterDBSCANParams>(
      document, "/cluster_dbscan");
  auto reduction = decode_params<DetectionReductionParams>(
      document, "/reduction");

  if (cfar->peak_mode != CfarPeakMode::kNone) {
    throw std::runtime_error("cfar_detector.peak_mode must be none");
  }
  if (!cfar->nci_axes.empty()) {
    throw std::runtime_error("cfar_detector.nci_axes must be empty");
  }
  const std::vector<AssemblyFeature> expected_features{
      AssemblyFeature::kRange, AssemblyFeature::kVelocity};
  if (assembly->features != expected_features) {
    throw std::runtime_error(
        "feature_assembly.features must be [range, velocity]");
  }
  if (dbscan->epsilon.size() != 2U) {
    throw std::runtime_error(
        "cluster_dbscan.epsilon must contain range and velocity radii");
  }
  if (reduction->representative != RepresentativePolicy::kMedoid) {
    throw std::runtime_error("reduction.representative must be medoid");
  }
  runtime.extents = make_extents(
      runtime, *range, *doppler, *dbscan,
      static_cast<int64_t>(assembly->features.size()));

  return Config{
      std::move(runtime), std::move(range), std::move(clutter),
      std::move(doppler), std::move(cfar), std::move(assembly),
      std::move(dbscan), std::move(reduction)};
}

}  // namespace holoscan::radar::app::siso
