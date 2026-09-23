// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "angle_estimation/esprit_estimator_op/esprit_estimator_op.hpp"
#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"
#include "angle_estimation/music_estimator_op/music_estimator_op.hpp"
#include "angle_estimation/mvdr_estimator_op/mvdr_estimator_op.hpp"
#include "angle_estimation/root_music_estimator_op/root_music_estimator_op.hpp"
#include "radar_app/fmcw/mimo_config.hpp"

namespace holoscan::radar::app::adaptive_aoa {

enum class Algorithm { kMusic, kRootMusic, kEsprit, kMvdr };

// Decode the one structural choice owned by this application. Estimator
// selection is fixed at graph construction; it is not a live operator option.
inline Algorithm parse_algorithm(const YAML::Node& app) {
  const std::string value = app["aoa_algorithm"] ? app["aoa_algorithm"].as<std::string>() : "music";
  if (value == "music")
    return Algorithm::kMusic;
  if (value == "root_music")
    return Algorithm::kRootMusic;
  if (value == "esprit")
    return Algorithm::kEsprit;
  if (value == "mvdr")
    return Algorithm::kMvdr;
  throw std::runtime_error("app.aoa_algorithm must be music, root_music, esprit, or mvdr");
}

inline const char* algorithm_name(Algorithm value) {
  switch (value) {
    case Algorithm::kMusic:
      return "MUSIC";
    case Algorithm::kRootMusic:
      return "Root-MUSIC";
    case Algorithm::kEsprit:
      return "ESPRIT";
    case Algorithm::kMvdr:
      return "MVDR";
  }
  return "unknown";
}

struct Config : mimo::RuntimeConfig {
  Config(mimo::RuntimeConfig runtime, Algorithm selected_algorithm, TXSchedule schedule,
         holoscan::PreparedParams<MIMODeinterleaveParams> deinterleave,
         holoscan::PreparedParams<FMCWRangeProcessorParams> range,
         holoscan::PreparedParams<ClutterFilterParams> clutter,
         holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler,
         holoscan::PreparedParams<TDMSparseCompensationParams> compensation,
         holoscan::PreparedParams<CFARParams> cfar,
         holoscan::PreparedParams<BeamscanParams> beamscan,
         holoscan::PreparedParams<RankOneResidualScreenParams> screen,
         holoscan::PreparedParams<mimo::FeatureAssemblyParams> assembly)
  : mimo::RuntimeConfig(std::move(runtime)),
    algorithm(selected_algorithm),
    tx_schedule(std::move(schedule)),
    deinterleave_params(std::move(deinterleave)),
    range_params(std::move(range)),
    clutter_params(std::move(clutter)),
    doppler_params(std::move(doppler)),
    compensation_params(std::move(compensation)),
    cfar_params(std::move(cfar)),
    beamscan_params(std::move(beamscan)),
    screen_params(std::move(screen)),
    assembly_params(std::move(assembly)) {}

  Algorithm algorithm{Algorithm::kMusic};
  // Derived TDM schedule shared by the source and deinterleaver.
  TXSchedule tx_schedule;
  // Prepared operator parameters preserve Holoscan's native YAML boundary.
  holoscan::PreparedParams<MIMODeinterleaveParams> deinterleave_params;
  holoscan::PreparedParams<FMCWRangeProcessorParams> range_params;
  holoscan::PreparedParams<ClutterFilterParams> clutter_params;
  holoscan::PreparedParams<FMCWDopplerProcessorParams> doppler_params;
  holoscan::PreparedParams<TDMSparseCompensationParams> compensation_params;
  holoscan::PreparedParams<CFARParams> cfar_params;
  holoscan::PreparedParams<BeamscanParams> beamscan_params;
  holoscan::PreparedParams<RankOneResidualScreenParams> screen_params;
  holoscan::PreparedParams<mimo::FeatureAssemblyParams> assembly_params;
  // Exactly one estimator parameter set is populated for a configured graph.
  std::optional<holoscan::PreparedParams<MUSICParams>> music_params;
  std::optional<holoscan::PreparedParams<RootMUSICParams>> root_music_params;
  std::optional<holoscan::PreparedParams<ESPRITParams>> esprit_params;
  std::optional<holoscan::PreparedParams<MVDRParams>> mvdr_params;
};

inline void validate_assembly(const mimo::FeatureAssemblyParams& assembly) {
  const std::vector<AssemblyFeature> expected{
      AssemblyFeature::kRange, AssemblyFeature::kVelocity, AssemblyFeature::kAzEstimate,
      AssemblyFeature::kElEstimate, AssemblyFeature::kScore};
  if (assembly.features != expected) {
    throw std::runtime_error(
        "feature_assembly.features must be [range, velocity, az_estimate, "
        "el_estimate, score]");
  }
}

template <typename Params>
inline void validate_covariance_params(const Params& params, const BeamscanParams& beamscan) {
  // All estimator outputs must share one angle boundary and one demonstrated
  // smoothing design so they remain interchangeable in the graph slot.
  if (params.angle_convention != beamscan.angle_convention) {
    throw std::runtime_error("selected estimator angle_convention must match beamscan_estimator");
  }
  if (params.max_sources < 2) {
    throw std::runtime_error("selected estimator max_sources must be >= 2");
  }
  if (params.spatial_smoothing != std::vector<int64_t>{2, 1} || !params.forward_backward) {
    throw std::runtime_error(
        "selected estimator requires spatial_smoothing [2, 1] and "
        "forward_backward: true");
  }
}

inline Config load_config(const std::filesystem::path& path) {
  // The app section owns estimator selection. Operator blocks remain decoded
  // through Holoscan Params rather than being parsed by application code.
  const YAML::Node yaml = YAML::LoadFile(path.string());
  const Algorithm algorithm = parse_algorithm(yaml["app"]);
  auto document = mimo::normalize_operator_fields(mimo::load_document(path), true);
  const auto app_params = mimo::decode_params<mimo::AppParams>(document, "/app");
  if (!app_params->tdm.has_value()) {
    throw std::runtime_error(
        "adaptive_aoa: app.tdm.physical_chirp_repetition_interval is required");
  }

  // Reuse common scene, array, run, visualization, and tensor-bound parsing.
  mimo::RuntimeConfig runtime = mimo::make_runtime_config(*app_params, "adaptive_aoa");
  // YAML authors the effective per-TX DSP PRI and the physical on-wire slot
  // PRI. The source simulator uses physical timing; downstream DSP keeps the
  // authored effective descriptor.
  const double physical_pri = app_params->tdm->physical_chirp_repetition_interval;
  mimo::require_finite_positive("app.tdm.physical_chirp_repetition_interval", physical_pri);
  const double effective_pri = physical_pri * static_cast<double>(app_params->tx_array.num_tx);
  const double tolerance = 1e-12 * std::max(1.0, std::abs(effective_pri));
  if (std::abs(app_params->waveform.chirp_repetition_interval() - effective_pri) > tolerance) {
    throw std::runtime_error("adaptive_aoa: waveform PRI must equal num_tx times physical PRI");
  }
  runtime.scene.waveform.chirp_repetition_interval = physical_pri;

  // This example depends on one specific collision-free filled virtual URA,
  // rather than claiming that every TDM layout supports all four estimators.
  if (app_params->tx_array.num_tx != 2 || app_params->tx_array.spacing != 2.0 ||
      app_params->rx_array.az_elements != 4 || app_params->rx_array.el_elements != 4 ||
      app_params->rx_array.spacing != 0.5) {
    throw std::runtime_error(
        "adaptive_aoa requires two TX at 2 lambda and a 4x4 RX URA at "
        "half-wavelength spacing");
  }

  // Derive the schedule once and inject it at every configuration boundary.
  TXSchedule schedule;
  schedule.num_tx = app_params->tx_array.num_tx;
  schedule.tx_order = app_params->tdm->tx_order;
  schedule.validate();
  (void)schedule.expand_chirp_tx_index(schedule.num_tx * runtime.scene.num_chirps);

  // TX schedule is application-derived and must not be duplicated in an
  // operator block where it could disagree with the source.
  mimo::reject_authored(document, "/mimo_deinterleave/tx_schedule");
  MIMODeinterleaveParams deinterleave_baseline;
  deinterleave_baseline.tx_schedule = schedule;
  document = mimo::inject_derived_params(std::move(document), "/mimo_deinterleave",
                                         std::move(deinterleave_baseline));
  auto deinterleave = mimo::decode_params<MIMODeinterleaveParams>(document, "/mimo_deinterleave");
  auto range = mimo::decode_params<FMCWRangeProcessorParams>(document, "/range_processor");
  auto clutter = mimo::decode_params<ClutterFilterParams>(document, "/clutter_filter");
  auto doppler = mimo::decode_params<FMCWDopplerProcessorParams>(document, "/doppler_processor");

  // Compensation descriptors describe the generated tensor for this tick;
  // inject the waveform, schedule, and physical arrays from the app authority.
  mimo::reject_authored(document, "/tdm_sparse_compensation/waveform");
  mimo::reject_authored(document, "/tdm_sparse_compensation/tx_schedule");
  mimo::reject_authored(document, "/tdm_sparse_compensation/tx_positions");
  mimo::reject_authored(document, "/tdm_sparse_compensation/rx_positions");
  TDMSparseCompensationParams compensation_baseline;
  compensation_baseline.waveform = app_params->waveform;
  compensation_baseline.tx_schedule = schedule;
  compensation_baseline.tx_positions = runtime.tx_positions;
  compensation_baseline.rx_positions = runtime.rx_positions;
  auto compensation = mimo::decode_params<TDMSparseCompensationParams>(
      document, "/tdm_sparse_compensation", std::move(compensation_baseline));

  const int64_t num_tx = static_cast<int64_t>(runtime.scene.tx_positions.size());
  const int64_t num_rx = static_cast<int64_t>(runtime.scene.rx_positions.size());
  // Resolve and validate the filled 8x4 virtual geometry before constructing
  // any angle estimator parameters.
  const detail::VirtualAssemblyPlan virtual_plan =
      detail::build_virtual_plan(runtime.tx_positions, runtime.rx_positions, num_tx, num_rx,
                                 compensation->overlap_policy, "adaptive_aoa");
  runtime.array_geometry =
      detail::make_array_geometry(virtual_plan, app_params->waveform.wavelength());
  if (runtime.array_geometry.num_elements() != 32) {
    throw std::runtime_error("adaptive_aoa derived virtual array must be a filled 8x4 URA");
  }

  // CFAR folds the named TX/RX axes; the fold size follows the derived array.
  mimo::reject_authored(document, "/cfar_detector/nci_fold");
  CFARParams cfar_baseline;
  cfar_baseline.nci_fold = num_tx * num_rx;
  auto cfar = mimo::decode_params<CFARParams>(document, "/cfar_detector", std::move(cfar_baseline));

  // Beamscan and the routing screen consume the same derived geometry. Their
  // operator YAML owns only algorithm parameters and presentation grids.
  mimo::reject_authored(document, "/beamscan_estimator/array_geometry");
  BeamscanParams beamscan_baseline;
  beamscan_baseline.array_geometry = runtime.array_geometry;
  auto beamscan = mimo::decode_params<BeamscanParams>(document, "/beamscan_estimator",
                                                      std::move(beamscan_baseline));
  mimo::reject_authored(document, "/rank_one_residual_screen/array_geometry");
  RankOneResidualScreenParams screen_baseline;
  screen_baseline.array_geometry = runtime.array_geometry;
  auto screen = mimo::decode_params<RankOneResidualScreenParams>(
      document, "/rank_one_residual_screen", std::move(screen_baseline));
  auto assembly = mimo::decode_params<mimo::FeatureAssemblyParams>(document, "/feature_assembly");

  // Fail closed if configuration changes would invalidate the demonstration's
  // peak-only, multi-look, no-clustering workflow.
  if (cfar->peak_mode != CfarPeakMode::kRangeDoppler ||
      cfar->nci_axes != std::vector<int64_t>{0, 1}) {
    throw std::runtime_error(
        "adaptive_aoa CFAR requires peak_mode range_doppler and "
        "nci_axes [0, 1]");
  }
  if (!doppler->center_zero_doppler) {
    throw std::runtime_error("doppler_processor.center_zero_doppler must be true");
  }
  if (compensation->overlap_policy != OverlapPolicy::kDedup ||
      compensation->snapshot_window != std::vector<int64_t>{1, 1} ||
      compensation->doppler_edge != SnapshotEdgeMode::kWrap ||
      compensation->range_edge != SnapshotEdgeMode::kShift) {
    throw std::runtime_error(
        "adaptive_aoa compensation requires dedup, snapshot_window [1, 1], "
        "Doppler wrap, and range shift");
  }
  if (beamscan->az_coordinates.empty() || beamscan->el_coordinates.empty()) {
    throw std::runtime_error("beamscan scan grids must not be empty");
  }
  if (screen->angle_convention != beamscan->angle_convention) {
    throw std::runtime_error("rank_one_residual_screen angle_convention must match Beamscan");
  }
  validate_assembly(*assembly);

  Config result{
      std::move(runtime), algorithm,           std::move(schedule), std::move(deinterleave),
      std::move(range),   std::move(clutter),  std::move(doppler),  std::move(compensation),
      std::move(cfar),    std::move(beamscan), std::move(screen),   std::move(assembly)};

  // Decode only the selected estimator. This both enforces startup selection
  // and avoids allocating unused estimator workspaces in the graph.
  const auto decode_geometry = [&](auto tag, const char* section) {
    using Params = decltype(tag);
    mimo::reject_authored(document, std::string(section) + "/array_geometry");
    Params baseline;
    baseline.array_geometry = result.array_geometry;
    return mimo::decode_params<Params>(document, section, std::move(baseline));
  };
  int64_t max_sources = 0;
  switch (algorithm) {
    case Algorithm::kMusic: {
      document = mimo::normalize_scan_coordinate_field(std::move(document),
                                                       "/music_estimator/az_coordinates");
      document = mimo::normalize_scan_coordinate_field(std::move(document),
                                                       "/music_estimator/el_coordinates");
      auto params = decode_geometry(MUSICParams{}, "/music_estimator");
      validate_covariance_params(*params, *result.beamscan_params);
      if (params->num_sources >= 0 || params->az_coordinates.empty() ||
          params->el_coordinates.empty()) {
        throw std::runtime_error("MUSIC requires num_sources: auto and non-empty scan grids");
      }
      max_sources = params->max_sources;
      result.music_params = std::move(params);
      break;
    }
    case Algorithm::kRootMusic: {
      auto params = decode_geometry(RootMUSICParams{}, "/root_music_estimator");
      validate_covariance_params(*params, *result.beamscan_params);
      if (params->num_sources >= 0) {
        throw std::runtime_error("Root-MUSIC requires num_sources: auto");
      }
      max_sources = params->max_sources;
      result.root_music_params = std::move(params);
      break;
    }
    case Algorithm::kEsprit: {
      auto params = decode_geometry(ESPRITParams{}, "/esprit_estimator");
      validate_covariance_params(*params, *result.beamscan_params);
      if (params->num_sources >= 0) {
        throw std::runtime_error("ESPRIT requires num_sources: auto");
      }
      max_sources = params->max_sources;
      result.esprit_params = std::move(params);
      break;
    }
    case Algorithm::kMvdr: {
      document = mimo::normalize_scan_coordinate_field(std::move(document),
                                                       "/mvdr_estimator/az_coordinates");
      document = mimo::normalize_scan_coordinate_field(std::move(document),
                                                       "/mvdr_estimator/el_coordinates");
      auto params = decode_geometry(MVDRParams{}, "/mvdr_estimator");
      validate_covariance_params(*params, *result.beamscan_params);
      if (params->az_coordinates.empty() || params->el_coordinates.empty()) {
        throw std::runtime_error("MVDR requires non-empty scan grids");
      }
      max_sources = params->max_sources;
      result.mvdr_params = std::move(params);
      break;
    }
  }

  // Reuse the common MIMO extent derivation. No DBSCAN operator is created;
  // the placeholder only supplies the helper's legacy max-cluster argument.
  ClusterDBSCANParams unused_cluster;
  unused_cluster.max_clusters = 1;
  result.extents = mimo::make_extents(
      result, result.scene.num_chirps, *result.range_params, *result.doppler_params,
      result.compensation_params->snapshot_window, *result.beamscan_params, unused_cluster,
      static_cast<int64_t>(result.assembly_params->features.size()), true);
  // Fine MUSIC/MVDR scan grids may exceed the coarser Beamscan presentation
  // grid and therefore participate independently in pool sizing.
  if (result.music_params.has_value()) {
    result.extents.max_az =
        std::max(*result.extents.max_az,
                 static_cast<int64_t>(result.music_params->value().az_coordinates.size()));
    result.extents.max_el =
        std::max(*result.extents.max_el,
                 static_cast<int64_t>(result.music_params->value().el_coordinates.size()));
  }
  if (result.mvdr_params.has_value()) {
    result.extents.max_az =
        std::max(*result.extents.max_az,
                 static_cast<int64_t>(result.mvdr_params->value().az_coordinates.size()));
    result.extents.max_el =
        std::max(*result.extents.max_el,
                 static_cast<int64_t>(result.mvdr_params->value().el_coordinates.size()));
  }
  // In the worst case every input detection expands to max_sources rows.
  const int64_t input_capacity = *result.extents.max_detections;
  result.extents.max_sources = max_sources;
  result.extents.max_points = input_capacity * max_sources;
  return result;
}

}  // namespace holoscan::radar::app::adaptive_aoa
