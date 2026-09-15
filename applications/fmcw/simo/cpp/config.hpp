// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/config/io.hpp>
#include <holoscan/core/parameter_input.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/error.hpp>

#include "angle_estimation/beamscan_estimator_op/params_config.hpp"
#include "angle_estimation/phase_shift_beamformer_op/params_config.hpp"
#include "angle_estimation/snapshot_gather_op/params_config.hpp"
#include "clustering/cluster_dbscan_op/params_config.hpp"
#include "detection/cfar_detector_op/params_config.hpp"
#include "detection/detection_reduction_op/params_config.hpp"
#include "fmcw/clutter_filter_op/params_config.hpp"
#include "fmcw/fmcw_doppler_processor_op/params_config.hpp"
#include "fmcw/fmcw_range_processor_op/params_config.hpp"
#include "radar_app/app/run_config.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_common/sim/scene.hpp"
#include "config_schema.hpp"  // NOLINT(build/include_subdir): app-private

namespace holoscan::radar::app::simo {

enum class Workflow {
  kDetectThenBeamform,
  kBeamformThenDetect,
};

struct RuntimeConfig : CommonRunConfig {
  sim::SimScene scene;
  ArrayGeometry array_geometry;
  MaxTensorExtents extents;
  MaxTensorExtents cfar_extents;
  ::holoscan::radar::app::simo::VisualizationConfig visualization;
};

struct DetectThenBeamformConfig : RuntimeConfig {
  holoscan::PreparedParams<FMCWRangeProcessorParams> range_params;
  holoscan::PreparedParams<ClutterFilterParams> clutter_params;
  holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler_params;
  holoscan::PreparedParams<CFARParams> cfar_params;
  holoscan::PreparedParams<SnapshotGatherParams> gather_params;
  holoscan::PreparedParams<BeamscanParams> beamscan_params;
  holoscan::PreparedParams<ClusterDBSCANParams> dbscan_params;
  holoscan::PreparedParams<DetectionReductionParams> reduction_params;
};

struct BeamformThenDetectConfig : RuntimeConfig {
  holoscan::PreparedParams<FMCWRangeProcessorParams> range_params;
  holoscan::PreparedParams<ClutterFilterParams> clutter_params;
  holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler_params;
  holoscan::PreparedParams<PhaseShiftBeamformerParams> beamformer_params;
  holoscan::PreparedParams<CFARParams> cfar_params;
};

template <typename T>
inline void require_positive(const char* key, T value) {
  if (value <= 0) {
    throw std::runtime_error(std::string(key) + " must be > 0, got " +
                             std::to_string(value));
  }
}

inline void validate_frame_period(double seconds) {
  if (!std::isfinite(seconds)) {
    throw std::runtime_error("app.run.frame_period must be finite");
  }
  const double ns = seconds * 1e9;
  if (ns < 1.0 ||
      ns > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    throw std::runtime_error(
        "app.run.frame_period must be within [1 ns, INT64_MAX ns], got " +
        std::to_string(seconds) + " s");
  }
}

inline void require_angle_grid(std::string_view key,
                               const std::vector<double>& grid) {
  if (grid.empty()) {
    throw std::runtime_error(std::string(key) + " must not be empty");
  }
}

template <typename Params>
holoscan::PreparedParams<Params> decode_params(
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

inline RuntimeConfig make_runtime_config(const AppParams& authored,
                                         const char* workflow_name) {
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
  require_positive("app.run.frame_period", result.frame_period);
  validate_frame_period(result.frame_period);
  if (result.meter_log_every < 0) {
    throw std::runtime_error("app.run.meter_log_every must be >= 0");
  }
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
  require_positive("app.cube.num_chirps", authored.cube.num_chirps);
  require_positive("app.cube.samples_per_chirp",
                   authored.cube.samples_per_chirp);

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

  const auto& rx = authored.rx_array;
  if (rx.az_elements < 2 || rx.el_elements < 2) {
    throw std::runtime_error(
        "app.rx_array az_elements and el_elements must be >= 2");
  }
  result.array_geometry = sim::make_ura_geometry(
      rx.az_elements, rx.el_elements, rx.spacing, waveform.wavelength());
  result.scene.rx_positions = sim::make_ura_rx_positions(
      rx.az_elements, rx.el_elements, rx.spacing, waveform.wavelength());

  result.scene.targets.reserve(authored.targets.size());
  for (const TargetConfig& target : authored.targets) {
    const sim::Vec3 direction{
        std::cos(target.elevation) * std::cos(target.azimuth),
        std::cos(target.elevation) * std::sin(target.azimuth),
        std::sin(target.elevation)};
    result.scene.targets.push_back(sim::PointTarget{
        sim::Vec3{target.range * direction[0],
                  target.range * direction[1],
                  target.range * direction[2]},
        target.velocity,
        {target.amplitude, 0.0}});
  }
  if (result.scene.targets.empty()) {
    throw std::runtime_error(std::string(workflow_name) +
                             ": app.targets must not be empty");
  }
  return result;
}

inline int64_t snapshot_count(const std::vector<int64_t>& window) {
  const int64_t doppler_half = window.empty() ? 0 : window[0];
  const int64_t range_half =
      window.size() > 1 ? window[1] : doppler_half;
  return (2 * doppler_half + 1) * (2 * range_half + 1);
}

template <typename AngleParams>
inline MaxTensorExtents make_extents(
    const RuntimeConfig& runtime, Workflow workflow,
    const FMCWRangeProcessorParams& range,
    const FMCWDopplerProcessorParams& doppler,
    const AngleParams& angle, const SnapshotGatherParams* gather,
    const ClusterDBSCANParams* dbscan) {
  MaxTensorExtents result;
  const int64_t num_rx = runtime.array_geometry.num_elements();
  const int64_t num_chirps = runtime.scene.num_chirps;
  const int64_t num_samples = runtime.scene.samples_per_chirp;
  const int64_t num_range =
      range.nfft > 0 ? range.nfft : num_samples;
  const int64_t num_doppler =
      doppler.nfft > 0 ? doppler.nfft : num_chirps;
  const int64_t num_az = static_cast<int64_t>(angle.az_angles.size());
  const int64_t num_el = static_cast<int64_t>(angle.el_angles.size());
  const int64_t map_cells = num_doppler * num_range;
  const int64_t max_points =
      workflow == Workflow::kDetectThenBeamform
          ? map_cells
          : num_az * num_el * map_cells;

  result.max_rx = num_rx;
  result.max_chirps = num_chirps;
  result.max_samples_per_chirp = num_samples;
  result.max_range_bins = num_range;
  result.max_doppler_bins = num_doppler;
  result.max_channels =
      workflow == Workflow::kDetectThenBeamform ? num_rx : num_az * num_el;
  result.max_detections = max_points;
  result.max_items = max_points;
  result.max_points = max_points;
  // Four physical features plus the CFAR score.
  result.max_features = 5;
  result.max_snapshots =
      gather == nullptr ? 1 : snapshot_count(gather->snapshot_window);
  result.max_az = num_az;
  result.max_el = num_el;
  result.max_clusters =
      dbscan != nullptr && dbscan->max_clusters > 0
          ? dbscan->max_clusters
          : max_points;
  return result;
}

inline MaxTensorExtents make_dense_cfar_extents(
    const MaxTensorExtents& base) {
  MaxTensorExtents result = base;
  result.max_tx = base.max_az;
  result.max_rx = base.max_el;
  result.max_channels = std::nullopt;
  return result;
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

// The pinned HSDK 5 Params decoder currently lowers enum schema fields through
// their integer representation even when a portable ConfigCodec is present.
// Accept the public canonical strings, then publish an immutable normalized
// document so PreparedParams still retains document-backed evidence.
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
  auto replaced =
      document.replace(path, static_cast<Underlying>(*decoded));
  if (!replaced) {
    throw std::runtime_error(
        holoscan::config::format_error(replaced.error()));
  }
  return std::move(*replaced);
}

inline holoscan::config::Document normalize_operator_enums(
    holoscan::config::Document document, Workflow workflow) {
  document = normalize_enum_field<FftNorm>(
      std::move(document), "/range_processor/fft_norm");
  document = normalize_enum_field<ClutterFilterMethod>(
      std::move(document), "/clutter_filter/method");
  document = normalize_enum_field<FftNorm>(
      std::move(document), "/doppler_processor/fft_norm");
  const std::string base =
      workflow == Workflow::kDetectThenBeamform
          ? "/detect_then_beamform"
          : "/beamform_then_detect";
  document = normalize_enum_field<CfarVariant>(
      std::move(document), base + "/cfar_detector/cfar_variant");
  document = normalize_enum_field<CfarPeakMode>(
      std::move(document), base + "/cfar_detector/peak_mode");
  document = normalize_enum_field<CfarPeakNeighborhood>(
      std::move(document),
      base + "/cfar_detector/peak_neighborhood");
  document = normalize_enum_field<RepresentativePolicy>(
      std::move(document), base + "/reduction/representative");
  document = normalize_enum_field<ScoreMethod>(
      std::move(document), base + "/reduction/score_method");
  if (workflow == Workflow::kDetectThenBeamform) {
    document = normalize_enum_field<SnapshotEdgeMode>(
        std::move(document),
        base + "/snapshot_gather/doppler_edge");
    document = normalize_enum_field<SnapshotEdgeMode>(
        std::move(document),
        base + "/snapshot_gather/range_edge");
    document = normalize_enum_field<PeakInterpolation>(
        std::move(document),
        base + "/beamscan_estimator/peak_interpolation");
  }
  return document;
}

inline DetectThenBeamformConfig load_detect_then_beamform_config(
    const std::filesystem::path& path) {
  auto document = normalize_operator_enums(
      load_document(path), Workflow::kDetectThenBeamform);
  const auto app_params = decode_params<AppParams>(document, "/app");
  RuntimeConfig runtime =
      make_runtime_config(*app_params, "detect_then_beamform");

  auto range = decode_params<FMCWRangeProcessorParams>(
      document, "/range_processor");
  auto clutter = decode_params<ClutterFilterParams>(
      document, "/clutter_filter");
  auto doppler = decode_params<FMCWDopplerProcessorParams>(
      document, "/doppler_processor");
  reject_authored(
      document, "/detect_then_beamform/cfar_detector/nci_fold");
  CFARParams cfar_baseline;
  cfar_baseline.nci_fold = runtime.array_geometry.num_elements();
  auto cfar = decode_params<CFARParams>(
      document, "/detect_then_beamform/cfar_detector",
      std::move(cfar_baseline));
  auto gather = decode_params<SnapshotGatherParams>(
      document, "/detect_then_beamform/snapshot_gather");
  reject_authored(
      document, "/detect_then_beamform/beamscan_estimator/array_geometry");
  BeamscanParams beamscan_baseline;
  beamscan_baseline.array_geometry = runtime.array_geometry;
  auto beamscan = decode_params<BeamscanParams>(
      document, "/detect_then_beamform/beamscan_estimator",
      std::move(beamscan_baseline));
  auto dbscan = decode_params<ClusterDBSCANParams>(
      document, "/detect_then_beamform/cluster_dbscan");
  auto reduction = decode_params<DetectionReductionParams>(
      document, "/detect_then_beamform/reduction");

  if (cfar->peak_mode != CfarPeakMode::kNone) {
    throw std::runtime_error(
        "detect_then_beamform.cfar_detector.peak_mode must be none");
  }
  if (cfar->nci_axes != std::vector<int64_t>{0}) {
    throw std::runtime_error(
        "detect_then_beamform.cfar_detector.nci_axes must be [0]");
  }
  require_angle_grid(
      "detect_then_beamform.beamscan_estimator.az_angles",
      beamscan->az_angles);
  require_angle_grid(
      "detect_then_beamform.beamscan_estimator.el_angles",
      beamscan->el_angles);
  runtime.extents = make_extents(
      runtime, Workflow::kDetectThenBeamform, *range, *doppler, *beamscan,
      &gather.value(), &dbscan.value());
  runtime.cfar_extents = runtime.extents;

  return DetectThenBeamformConfig{
      std::move(runtime), std::move(range), std::move(clutter),
      std::move(doppler), std::move(cfar), std::move(gather),
      std::move(beamscan), std::move(dbscan), std::move(reduction)};
}

inline BeamformThenDetectConfig load_beamform_then_detect_config(
    const std::filesystem::path& path) {
  auto document = normalize_operator_enums(
      load_document(path), Workflow::kBeamformThenDetect);
  const auto app_params = decode_params<AppParams>(document, "/app");
  RuntimeConfig runtime =
      make_runtime_config(*app_params, "beamform_then_detect");

  auto range = decode_params<FMCWRangeProcessorParams>(
      document, "/range_processor");
  auto clutter = decode_params<ClutterFilterParams>(
      document, "/clutter_filter");
  auto doppler = decode_params<FMCWDopplerProcessorParams>(
      document, "/doppler_processor");
  reject_authored(
      document,
      "/beamform_then_detect/phase_shift_beamformer/array_geometry");
  PhaseShiftBeamformerParams beamformer_baseline;
  beamformer_baseline.array_geometry = runtime.array_geometry;
  auto beamformer = decode_params<PhaseShiftBeamformerParams>(
      document, "/beamform_then_detect/phase_shift_beamformer",
      std::move(beamformer_baseline));
  auto cfar = decode_params<CFARParams>(
      document, "/beamform_then_detect/cfar_detector");
  if (cfar->peak_mode != CfarPeakMode::kAll) {
    throw std::runtime_error(
        "beamform_then_detect.cfar_detector.peak_mode must be all");
  }
  if (!cfar->nci_axes.empty()) {
    throw std::runtime_error(
        "beamform_then_detect.cfar_detector.nci_axes must be empty");
  }
  require_angle_grid(
      "beamform_then_detect.phase_shift_beamformer.az_angles",
      beamformer->az_angles);
  require_angle_grid(
      "beamform_then_detect.phase_shift_beamformer.el_angles",
      beamformer->el_angles);
  runtime.extents = make_extents(
      runtime, Workflow::kBeamformThenDetect, *range, *doppler,
      *beamformer, nullptr, nullptr);
  runtime.cfar_extents = make_dense_cfar_extents(runtime.extents);

  return BeamformThenDetectConfig{
      std::move(runtime), std::move(range), std::move(clutter),
      std::move(doppler), std::move(beamformer), std::move(cfar)};
}

}  // namespace holoscan::radar::app::simo
