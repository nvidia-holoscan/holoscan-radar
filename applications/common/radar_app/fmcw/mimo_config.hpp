// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/config/io.hpp>
#include <holoscan/core/parameter_input.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/error.hpp>

#include "angle_estimation/beamscan_estimator_op/params_config.hpp"
#include "angle_estimation/snapshot_gather_op/params_config.hpp"
#include "clustering/cluster_dbscan_op/params_config.hpp"
#include "detection/cfar_detector_op/params_config.hpp"
#include "detection/detection_reduction_op/params_config.hpp"
#include "fmcw/clutter_filter_op/params_config.hpp"
#include "fmcw/doppler_notch_op/doppler_notch_op.hpp"
#include "fmcw/fmcw_doppler_processor_op/params_config.hpp"
#include "fmcw/fmcw_range_processor_op/params_config.hpp"
#include "fmcw/tdm_sparse_compensation_op/tdm_sparse_compensation_op.hpp"
#include "radar_app/app/run_config.hpp"
#include "radar_app/app/config_utils.hpp"
#include "radar_app/fmcw/config_loader.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/dsp/ddm_code_generators.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/dsp/virtual_array.hpp"
#include "radar_common/math/uniform_grid.hpp"
#include "radar_common/sim/array_builders.hpp"
#include "radar_common/sim/scene.hpp"
#include "radar_app/fmcw/mimo_config_schema.hpp"

namespace holoscan::radar::app::mimo {

using config::decode_params;
using config::inject_derived_params;
using config::load_inherited_document;
using config::normalize_enum_field;
using config::normalize_scan_coordinate_field;
using config::reject_authored;
using config::require_finite_positive;
using config::require_positive;
using config::validate_frame_period;

struct RuntimeConfig : CommonRunConfig {
  sim::SimScene scene;
  sim::SceneVisibility scene_visibility{};
  ArrayGeometry array_geometry;
  std::vector<double> tx_positions;
  std::vector<double> rx_positions;
  MaxTensorExtents extents;
  ::holoscan::radar::app::VisualizationConfig visualization;
  std::vector<int64_t> report_index_columns;
};

struct TdmConfig : RuntimeConfig {
  TXSchedule tx_schedule;
  holoscan::PreparedParams<MIMODeinterleaveParams> deinterleave_params;
  holoscan::PreparedParams<FMCWRangeProcessorParams> range_params;
  holoscan::PreparedParams<ClutterFilterParams> clutter_params;
  holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler_params;
  holoscan::PreparedParams<TDMSparseCompensationParams> compensation_params;
  holoscan::PreparedParams<CFARParams> cfar_params;
  holoscan::PreparedParams<BeamscanParams> beamscan_params;
  holoscan::PreparedParams<FeatureAssemblyParams> assembly_params;
  holoscan::PreparedParams<ClusterDBSCANParams> dbscan_params;
  holoscan::PreparedParams<DetectionReductionParams> reduction_params;
};

struct DdmConfig : RuntimeConfig {
  DDMCodeDescriptor ddm_code;
  std::vector<double> doppler_offsets;
  bool correlate{false};
  holoscan::PreparedParams<FMCWRangeProcessorParams> range_params;
  holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler_params;
  holoscan::PreparedParams<DDMDenseDemultiplexParams> demultiplex_params;
  holoscan::PreparedParams<DopplerNotchParams> notch_params;
  holoscan::PreparedParams<VirtualArrayAssemblyParams> vaa_params;
  holoscan::PreparedParams<CFARParams> cfar_params;
  holoscan::PreparedParams<SnapshotGatherParams> gather_params;
  holoscan::PreparedParams<BeamscanParams> beamscan_params;
  holoscan::PreparedParams<FeatureAssemblyParams> assembly_params;
  holoscan::PreparedParams<ClusterDBSCANParams> dbscan_params;
  holoscan::PreparedParams<DetectionReductionParams> reduction_params;
};

inline holoscan::config::Document normalize_operator_fields(holoscan::config::Document document,
                                                            bool tdm) {
  document = normalize_enum_field<FftNorm>(std::move(document), "/range_processor/fft_norm");
  document =
      normalize_enum_field<ClutterFilterMethod>(std::move(document), "/clutter_filter/method");
  document = normalize_enum_field<FftNorm>(std::move(document), "/doppler_processor/fft_norm");
  const std::string middle = tdm ? "/tdm_sparse_compensation" : "/virtual_array_assembly";
  document = normalize_enum_field<OverlapPolicy>(std::move(document), middle + "/overlap_policy");
  document = normalize_enum_field<CfarVariant>(std::move(document), "/cfar_detector/cfar_variant");
  document = normalize_enum_field<CfarPeakMode>(std::move(document), "/cfar_detector/peak_mode");
  const std::string snapshot = tdm ? "/tdm_sparse_compensation" : "/snapshot_gather";
  document =
      normalize_enum_field<SnapshotEdgeMode>(std::move(document), snapshot + "/doppler_edge");
  document = normalize_enum_field<SnapshotEdgeMode>(std::move(document), snapshot + "/range_edge");
  document = normalize_enum_field<PeakInterpolation>(std::move(document),
                                                     "/beamscan_estimator/peak_interpolation");
  document =
      normalize_enum_field<RepresentativePolicy>(std::move(document), "/reduction/representative");
  document = normalize_enum_field<ScoreMethod>(std::move(document), "/reduction/score_method");
  document =
      normalize_scan_coordinate_field(std::move(document), "/beamscan_estimator/az_coordinates");
  document =
      normalize_scan_coordinate_field(std::move(document), "/beamscan_estimator/el_coordinates");
  return document;
}

inline RuntimeConfig make_runtime_config(const AppParams& authored, const char* workflow_name) {
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
  result.visualization.validate_layout();
  require_positive("app.visualization.window_width", result.visualization.window_width);
  require_positive("app.visualization.window_height", result.visualization.window_height);
  require_finite_positive("app.visualization.display_rate_hz",
                          result.visualization.display_rate_hz);
  require_finite_positive("app.visualization.dynamic_range_db",
                          result.visualization.dynamic_range_db);
  require_finite_positive("app.visualization.detection_radius",
                          result.visualization.detection_radius);
  require_positive("app.visualization.history_frames", result.visualization.history_frames);
  require_positive("app.visualization.detection_trail_frames",
                   result.visualization.detection_trail_frames);
  if (result.visualization.history_frames < 2) {
    throw std::runtime_error("app.visualization.history_frames must be >= 2");
  }
  if (result.visualization.enabled &&
      (result.visualization.window_width < 640 || result.visualization.window_height < 360)) {
    throw std::runtime_error("app.visualization dashboard requires at least a 640x360 window");
  }
  if (result.visualization.window_title.empty()) {
    throw std::runtime_error("app.visualization.window_title must not be empty");
  }

  require_positive("app.cube.num_chirps", authored.cube.num_chirps);
  require_positive("app.cube.samples_per_chirp", authored.cube.samples_per_chirp);
  require_positive("app.tx_array.num_tx", authored.tx_array.num_tx);
  require_finite_positive("app.tx_array.spacing", authored.tx_array.spacing);
  if (authored.rx_array.az_elements < 2 || authored.rx_array.el_elements < 2) {
    throw std::runtime_error("app.rx_array az_elements and el_elements must be >= 2");
  }
  require_finite_positive("app.rx_array.spacing", authored.rx_array.spacing);

  const auto& waveform = authored.waveform;
  result.scene.waveform.carrier_frequency = waveform.carrier_frequency();
  result.scene.waveform.bandwidth = waveform.bandwidth();
  result.scene.waveform.chirp_duration = waveform.chirp_duration();
  result.scene.waveform.sample_rate = waveform.sample_rate();
  result.scene.waveform.chirp_repetition_interval = waveform.chirp_repetition_interval();
  result.scene.num_chirps = authored.cube.num_chirps;
  result.scene.samples_per_chirp = authored.cube.samples_per_chirp;
  result.scene.snr_db = authored.noise_snr_db;
  ::holoscan::radar::app::configure_range_loss(authored.range_loss, &result.scene);
  result.scene_visibility =
      ::holoscan::radar::app::configure_scene_visibility(authored.scene_visibility);
  result.scene.tx_positions = sim::make_linear_positions(
      authored.tx_array.num_tx, authored.tx_array.spacing, waveform.wavelength());
  result.scene.rx_positions =
      sim::make_ura_rx_positions(authored.rx_array.az_elements, authored.rx_array.el_elements,
                                 authored.rx_array.spacing, waveform.wavelength());
  result.tx_positions = sim::flatten_positions(result.scene.tx_positions);
  result.rx_positions = sim::flatten_positions(result.scene.rx_positions);

  result.scene.targets.reserve(authored.targets.size());
  for (const TargetConfig& target : authored.targets) {
    if (!std::isfinite(target.range) || target.range <= 0.0 || !std::isfinite(target.azimuth) ||
        !std::isfinite(target.elevation) || !std::isfinite(target.amplitude)) {
      throw std::runtime_error("app.targets range/amplitude/angles must be finite and range > 0");
    }
    const sim::Vec3 direction{std::cos(target.elevation) * std::cos(target.azimuth),
                              std::cos(target.elevation) * std::sin(target.azimuth),
                              std::sin(target.elevation)};
    result.scene.targets.push_back(
        sim::PointTarget{sim::Vec3{target.range * direction[0], target.range * direction[1],
                                   target.range * direction[2]},
                         target.velocity,
                         {target.amplitude, 0.0}});
  }
  if (result.scene.targets.empty()) {
    throw std::runtime_error(std::string(workflow_name) + ": app.targets must not be empty");
  }
  return result;
}

inline int64_t snapshot_count(const std::vector<int64_t>& window) {
  const int64_t doppler_half = window.empty() ? 0 : window[0];
  const int64_t range_half = window.size() > 1 ? window[1] : doppler_half;
  return (2 * doppler_half + 1) * (2 * range_half + 1);
}

inline std::vector<int64_t> report_index_columns(const FeatureAssemblyParams& assembly) {
  const auto column = [&](AssemblyFeature feature) {
    const auto iter = std::find(assembly.features.begin(), assembly.features.end(), feature);
    if (iter == assembly.features.end()) {
      throw std::runtime_error("feature_assembly.features is missing " +
                               std::string(to_string(feature)));
    }
    return static_cast<int64_t>(std::distance(assembly.features.begin(), iter));
  };
  return {column(AssemblyFeature::kDopplerBin), column(AssemblyFeature::kRangeBin)};
}

inline void validate_cluster_tail(const FeatureAssemblyParams& assembly,
                                  const ClusterDBSCANParams& dbscan,
                                  const DetectionReductionParams& reduction) {
  const std::vector<AssemblyFeature> expected{
      AssemblyFeature::kRange,      AssemblyFeature::kVelocity, AssemblyFeature::kAzEstimate,
      AssemblyFeature::kElEstimate, AssemblyFeature::kScore,    AssemblyFeature::kRangeBin,
      AssemblyFeature::kDopplerBin};
  if (assembly.features != expected) {
    throw std::runtime_error(
        "feature_assembly.features must be [range, velocity, az_estimate, "
        "el_estimate, score, range_bin, doppler_bin]");
  }
  if (dbscan.epsilon.size() != expected.size() ||
      !std::all_of(dbscan.epsilon.begin(), dbscan.epsilon.begin() + 5,
                   [](double value) {
                     return value == 0.0;
                   }) ||
      dbscan.epsilon[5] <= 0.0 || dbscan.epsilon[6] <= 0.0) {
    throw std::runtime_error(
        "cluster_dbscan.epsilon must ignore the five physical/score "
        "columns and use positive range-bin and Doppler-bin radii");
  }
  if (reduction.representative != RepresentativePolicy::kWeightedMean ||
      reduction.score_method != ScoreMethod::kWeightedSum ||
      reduction.feature_weights != std::vector<double>{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}) {
    throw std::runtime_error(
        "reduction must use weighted_mean/weighted_sum with a one-hot score "
        "weight matching the clustered SIMO tail");
  }
}

inline MaxTensorExtents make_extents(const RuntimeConfig& runtime, int64_t report_doppler,
                                     const FMCWRangeProcessorParams& range,
                                     const FMCWDopplerProcessorParams& doppler,
                                     const std::vector<int64_t>& snapshot_window,
                                     const BeamscanParams& beamscan,
                                     const ClusterDBSCANParams& dbscan, int64_t num_features,
                                     bool tdm) {
  MaxTensorExtents result;
  const int64_t num_tx = static_cast<int64_t>(runtime.scene.tx_positions.size());
  const int64_t num_rx = static_cast<int64_t>(runtime.scene.rx_positions.size());
  const int64_t chirps = runtime.scene.num_chirps;
  const int64_t samples = runtime.scene.samples_per_chirp;
  const int64_t range_bins = range.nfft > 0 ? range.nfft : samples;
  const int64_t pre_doppler = doppler.nfft > 0 ? doppler.nfft : chirps;
  const int64_t max_doppler = std::max(pre_doppler, report_doppler);
  const int64_t map_cells = report_doppler * range_bins;

  result.max_tx = num_tx;
  result.max_rx = num_rx;
  result.max_chirps = chirps;
  if (tdm) {
    result.max_interleaved_chirps = num_tx * chirps;
  }
  result.max_samples_per_chirp = samples;
  result.max_range_bins = range_bins;
  result.max_doppler_bins = max_doppler;
  result.max_virtual_channels = runtime.array_geometry.num_elements();
  result.max_channels = runtime.array_geometry.num_elements();
  result.max_detections = map_cells;
  result.max_items = map_cells;
  result.max_points = map_cells;
  result.max_features = num_features;
  result.max_snapshots = snapshot_count(snapshot_window);
  result.max_az = static_cast<int64_t>(beamscan.az_coordinates.size());
  result.max_el = static_cast<int64_t>(beamscan.el_coordinates.size());
  result.max_clusters = dbscan.max_clusters > 0 ? dbscan.max_clusters : map_cells;
  return result;
}

inline void validate_common_operator_params(const CFARParams& cfar, const BeamscanParams& beamscan,
                                            const FMCWDopplerProcessorParams& doppler,
                                            const std::vector<int64_t>& expected_nci_axes) {
  if (cfar.peak_mode != CfarPeakMode::kNone) {
    throw std::runtime_error("cfar_detector.peak_mode must be none");
  }
  if (cfar.nci_axes != expected_nci_axes) {
    throw std::runtime_error(expected_nci_axes.size() == 2
                                 ? "cfar_detector.nci_axes must be [0, 1] for TDM"
                                 : "cfar_detector.nci_axes must be [0] for DDM");
  }
  if (!doppler.center_zero_doppler) {
    throw std::runtime_error("doppler_processor.center_zero_doppler must be true");
  }
  if (beamscan.az_coordinates.empty() || beamscan.el_coordinates.empty()) {
    throw std::runtime_error("beamscan_estimator scan coordinate axes must not be empty");
  }
}

struct CommonPrepared {
  RuntimeConfig runtime;
  holoscan::PreparedParams<FMCWRangeProcessorParams> range;
  holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler;
  holoscan::PreparedParams<VirtualArrayAssemblyParams> vaa;
  holoscan::PreparedParams<CFARParams> cfar;
  holoscan::PreparedParams<SnapshotGatherParams> gather;
  holoscan::PreparedParams<BeamscanParams> beamscan;
  holoscan::PreparedParams<FeatureAssemblyParams> assembly;
  holoscan::PreparedParams<ClusterDBSCANParams> dbscan;
  holoscan::PreparedParams<DetectionReductionParams> reduction;
};

inline CommonPrepared prepare_common(const holoscan::config::Document& document,
                                     const AppParams& authored, const char* workflow_name) {
  RuntimeConfig runtime = make_runtime_config(authored, workflow_name);
  auto range = decode_params<FMCWRangeProcessorParams>(document, "/range_processor");
  auto doppler = decode_params<FMCWDopplerProcessorParams>(document, "/doppler_processor");

  reject_authored(document, "/virtual_array_assembly/tx_positions");
  reject_authored(document, "/virtual_array_assembly/rx_positions");
  reject_authored(document, "/virtual_array_assembly/wavelength");
  VirtualArrayAssemblyParams vaa_baseline;
  vaa_baseline.tx_positions = runtime.tx_positions;
  vaa_baseline.rx_positions = runtime.rx_positions;
  vaa_baseline.wavelength = authored.waveform.wavelength();
  auto vaa = decode_params<VirtualArrayAssemblyParams>(document, "/virtual_array_assembly",
                                                       std::move(vaa_baseline));
  const detail::VirtualAssemblyPlan plan = detail::build_virtual_plan(
      runtime.tx_positions, runtime.rx_positions,
      static_cast<int64_t>(runtime.scene.tx_positions.size()),
      static_cast<int64_t>(runtime.scene.rx_positions.size()), vaa->overlap_policy, workflow_name);
  runtime.array_geometry = detail::make_array_geometry(plan, authored.waveform.wavelength());

  reject_authored(document, "/cfar_detector/nci_fold");
  CFARParams cfar_baseline;
  cfar_baseline.nci_fold = runtime.array_geometry.num_elements();
  auto cfar = decode_params<CFARParams>(document, "/cfar_detector", std::move(cfar_baseline));
  auto gather = decode_params<SnapshotGatherParams>(document, "/snapshot_gather");
  reject_authored(document, "/beamscan_estimator/array_geometry");
  BeamscanParams beamscan_baseline;
  beamscan_baseline.array_geometry = runtime.array_geometry;
  auto beamscan =
      decode_params<BeamscanParams>(document, "/beamscan_estimator", std::move(beamscan_baseline));
  auto assembly = decode_params<FeatureAssemblyParams>(document, "/feature_assembly");
  auto dbscan = decode_params<ClusterDBSCANParams>(document, "/cluster_dbscan");
  auto reduction = decode_params<DetectionReductionParams>(document, "/reduction");

  validate_common_operator_params(*cfar, *beamscan, *doppler, std::vector<int64_t>{0});
  validate_cluster_tail(*assembly, *dbscan, *reduction);
  runtime.report_index_columns = report_index_columns(*assembly);
  return CommonPrepared{std::move(runtime),  std::move(range),    std::move(doppler),
                        std::move(vaa),      std::move(cfar),     std::move(gather),
                        std::move(beamscan), std::move(assembly), std::move(dbscan),
                        std::move(reduction)};
}

inline TdmConfig load_tdm_config_impl(const std::filesystem::path& path,
                                      std::optional<AppParams> app_baseline) {
  auto document = normalize_operator_fields(load_inherited_document(path), true);
  const auto app_params = app_baseline.has_value()
                              ? decode_params<AppParams>(document, "/app", std::move(*app_baseline))
                              : decode_params<AppParams>(document, "/app");
  if (!app_params->tdm.has_value()) {
    throw std::runtime_error("tdm: app.tdm.physical_chirp_repetition_interval is required");
  }

  RuntimeConfig runtime = make_runtime_config(*app_params, "tdm");

  const double physical_pri = app_params->tdm->physical_chirp_repetition_interval;
  require_finite_positive("app.tdm.physical_chirp_repetition_interval", physical_pri);
  const double expected_effective_pri =
      physical_pri * static_cast<double>(app_params->tx_array.num_tx);
  const double tolerance = 1e-12 * std::max(1.0, std::abs(expected_effective_pri));
  if (std::abs(app_params->waveform.chirp_repetition_interval() - expected_effective_pri) >
      tolerance) {
    throw std::runtime_error(
        "tdm: app.waveform.chirp_repetition_interval must equal "
        "app.tx_array.num_tx * physical_chirp_repetition_interval");
  }
  runtime.scene.waveform.chirp_repetition_interval = physical_pri;

  TXSchedule schedule;
  schedule.num_tx = app_params->tx_array.num_tx;
  schedule.tx_order = app_params->tdm->tx_order;
  schedule.validate();
  (void)schedule.expand_chirp_tx_index(schedule.num_tx * runtime.scene.num_chirps);

  reject_authored(document, "/mimo_deinterleave/tx_schedule");
  MIMODeinterleaveParams deinterleave_baseline;
  deinterleave_baseline.tx_schedule = schedule;
  document = inject_derived_params(std::move(document), "/mimo_deinterleave",
                                   std::move(deinterleave_baseline));
  auto deinterleave = decode_params<MIMODeinterleaveParams>(document, "/mimo_deinterleave");

  auto range = decode_params<FMCWRangeProcessorParams>(document, "/range_processor");
  auto clutter = decode_params<ClutterFilterParams>(document, "/clutter_filter");
  auto doppler = decode_params<FMCWDopplerProcessorParams>(document, "/doppler_processor");

  reject_authored(document, "/tdm_sparse_compensation/waveform");
  reject_authored(document, "/tdm_sparse_compensation/tx_schedule");
  reject_authored(document, "/tdm_sparse_compensation/tx_positions");
  reject_authored(document, "/tdm_sparse_compensation/rx_positions");
  TDMSparseCompensationParams compensation_baseline;
  compensation_baseline.waveform = app_params->waveform;
  compensation_baseline.tx_schedule = schedule;
  compensation_baseline.tx_positions = runtime.tx_positions;
  compensation_baseline.rx_positions = runtime.rx_positions;
  auto compensation = decode_params<TDMSparseCompensationParams>(
      document, "/tdm_sparse_compensation", std::move(compensation_baseline));

  const int64_t num_tx = static_cast<int64_t>(runtime.scene.tx_positions.size());
  const int64_t num_rx = static_cast<int64_t>(runtime.scene.rx_positions.size());
  const detail::VirtualAssemblyPlan plan =
      detail::build_virtual_plan(runtime.tx_positions, runtime.rx_positions, num_tx, num_rx,
                                 compensation->overlap_policy, "tdm");
  runtime.array_geometry = detail::make_array_geometry(plan, app_params->waveform.wavelength());

  reject_authored(document, "/cfar_detector/nci_fold");
  CFARParams cfar_baseline;
  cfar_baseline.nci_fold = num_tx * num_rx;
  auto cfar = decode_params<CFARParams>(document, "/cfar_detector", std::move(cfar_baseline));
  reject_authored(document, "/beamscan_estimator/array_geometry");
  BeamscanParams beamscan_baseline;
  beamscan_baseline.array_geometry = runtime.array_geometry;
  auto beamscan =
      decode_params<BeamscanParams>(document, "/beamscan_estimator", std::move(beamscan_baseline));
  auto assembly = decode_params<FeatureAssemblyParams>(document, "/feature_assembly");
  auto dbscan = decode_params<ClusterDBSCANParams>(document, "/cluster_dbscan");
  auto reduction = decode_params<DetectionReductionParams>(document, "/reduction");

  validate_common_operator_params(*cfar, *beamscan, *doppler, std::vector<int64_t>{0, 1});
  validate_cluster_tail(*assembly, *dbscan, *reduction);
  runtime.report_index_columns = report_index_columns(*assembly);
  runtime.extents = make_extents(runtime, runtime.scene.num_chirps, *range, *doppler,
                                 compensation->snapshot_window, *beamscan, *dbscan,
                                 static_cast<int64_t>(assembly->features.size()), true);

  return TdmConfig{std::move(runtime),      std::move(schedule), std::move(deinterleave),
                   std::move(range),        std::move(clutter),  std::move(doppler),
                   std::move(compensation), std::move(cfar),     std::move(beamscan),
                   std::move(assembly),     std::move(dbscan),   std::move(reduction)};
}

inline TdmConfig load_tdm_config(const std::filesystem::path& path) {
  return load_tdm_config_impl(path, std::nullopt);
}

inline TdmConfig load_tdm_config(const std::filesystem::path& path, AppParams app_baseline) {
  return load_tdm_config_impl(path, std::move(app_baseline));
}

inline DdmConfig load_ddm_config(const std::filesystem::path& path) {
  auto document = normalize_operator_fields(load_inherited_document(path), false);
  const auto app_params = decode_params<AppParams>(document, "/app");
  if (app_params->tdm.has_value()) {
    throw std::runtime_error("ddm: app.tdm must not be authored");
  }
  reject_authored(document, "/clutter_filter");
  CommonPrepared common = prepare_common(document, *app_params, "ddm");

  DDMDenseDemultiplexParams demultiplex_baseline;
  std::optional<std::string> generator;
  auto root_ddm = document.try_find("/ddm");
  if (!root_ddm) {
    throw std::runtime_error(holoscan::config::format_error(root_ddm.error()));
  }
  if (root_ddm->has_value()) {
    auto decoded = document.try_get<DdmSimulationConfig>("/ddm");
    if (!decoded) {
      throw std::runtime_error(holoscan::config::format_error(decoded.error()));
    }
    generator = decoded->generate_phase_codes;
  }
  if (generator.has_value()) {
    reject_authored(document, "/ddm_dense_demultiplex/ddm_code");
    if (*generator == "zadoff_chu" || *generator == "zc") {
      demultiplex_baseline.ddm_code =
          make_zadoff_chu_ddm_code(app_params->tx_array.num_tx, common.runtime.scene.num_chirps);
    } else if (*generator == "sin") {
      demultiplex_baseline.ddm_code = make_sinusoidal_ddm_stress_code(
          app_params->tx_array.num_tx, common.runtime.scene.num_chirps);
    } else {
      throw std::runtime_error("ddm.generate_phase_codes must be zadoff_chu, zc, or sin");
    }
    document = inject_derived_params(std::move(document), "/ddm_dense_demultiplex",
                                     std::move(demultiplex_baseline));
  }
  auto demultiplex = decode_params<DDMDenseDemultiplexParams>(document, "/ddm_dense_demultiplex");
  auto notch = decode_params<DopplerNotchParams>(document, "/doppler_notch");
  DDMCodeDescriptor code = demultiplex->ddm_code;
  code.validate();
  if (code.num_tx != app_params->tx_array.num_tx) {
    throw std::runtime_error(
        "ddm_dense_demultiplex.ddm_code.num_tx must match "
        "app.tx_array.num_tx");
  }

  const bool correlate = detail::ddm_classify_codes(code) == detail::DdmCodeMode::kNonLinear;
  std::vector<double> offsets;
  int64_t report_doppler = common.runtime.scene.num_chirps;
  if (correlate) {
    if (code.guard_bands != 0) {
      throw std::runtime_error("nonlinear DDM phase codes require guard_bands == 0");
    }
    const int64_t code_chirps = static_cast<int64_t>(code.phase_codes.size()) / code.num_tx;
    if (code_chirps != common.runtime.scene.num_chirps) {
      throw std::runtime_error("nonlinear DDM phase-code length must match app.cube.num_chirps");
    }
  } else {
    offsets = detail::ddm_resolve_offsets(code);
    report_doppler = detail::ddm_num_doppler_per_tx(common.runtime.scene.num_chirps, code.num_tx,
                                                    code.guard_bands);
    (void)detail::ddm_exact_band_indices(code, common.runtime.scene.num_chirps);
  }

  common.runtime.extents =
      make_extents(common.runtime, report_doppler, *common.range, *common.doppler,
                   common.gather->snapshot_window, *common.beamscan, *common.dbscan,
                   static_cast<int64_t>(common.assembly->features.size()), false);

  return DdmConfig{std::move(common.runtime),  std::move(code),
                   std::move(offsets),         correlate,
                   std::move(common.range),    std::move(common.doppler),
                   std::move(demultiplex),     std::move(notch),
                   std::move(common.vaa),      std::move(common.cfar),
                   std::move(common.gather),   std::move(common.beamscan),
                   std::move(common.assembly), std::move(common.dbscan),
                   std::move(common.reduction)};
}

}  // namespace holoscan::radar::app::mimo
