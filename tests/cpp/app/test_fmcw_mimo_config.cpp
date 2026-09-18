// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_app/simple/mimo/config.hpp"
#include "radar_common/descriptors/window_spec.hpp"

namespace {

namespace mimo = holoscan::radar::app::mimo;
namespace radar = holoscan::radar;

std::filesystem::path config_path(const char* name) {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
         "applications/fmcw" / name;
}

void expect_smoke_matches_production(const char* production_name,
                                     const char* smoke_name) {
  YAML::Node expected = YAML::LoadFile(config_path(production_name).string());
  const YAML::Node smoke = YAML::LoadFile(config_path(smoke_name).string());
  expected["app"]["run"]["num_frames"] = 2;
  expected["app"]["visualization"]["enabled"] = false;
  expected["app"].remove("range_loss");
  expected["app"].remove("scene_visibility");
  YAML::Node targets(YAML::NodeType::Sequence);
  targets.push_back(expected["app"]["targets"][0]);
  expected["app"]["targets"] = targets;
  EXPECT_EQ(YAML::Dump(expected), YAML::Dump(smoke));
}

class MutatedConfig {
 public:
  MutatedConfig(const char* source_name, const std::string& old_text,
                const std::string& new_text)
      : MutatedConfig(
            source_name,
            std::vector<std::pair<std::string, std::string>>{
                {old_text, new_text}}) {}

  MutatedConfig(
      const char* source_name,
      const std::vector<std::pair<std::string, std::string>>& replacements) {
    std::ifstream input(config_path(source_name));
    if (!input) {
      throw std::runtime_error("cannot read source config");
    }
    std::string contents{std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>()};
    for (const auto& [old_text, new_text] : replacements) {
      const std::size_t position = contents.find(old_text);
      if (position == std::string::npos ||
          contents.find(old_text, position + old_text.size()) !=
              std::string::npos) {
        throw std::runtime_error("config mutation anchor must occur once");
      }
      contents.replace(position, old_text.size(), new_text);
    }
    path_ = std::filesystem::temp_directory_path() /
            ("holoscan_radar_mimo_config_" + std::to_string(next_id_++) +
             ".yaml");
    std::ofstream output(path_);
    output << contents;
    if (!output) {
      throw std::runtime_error("cannot write temporary config");
    }
  }

  ~MutatedConfig() { std::filesystem::remove(path_); }

  MutatedConfig(const MutatedConfig&) = delete;
  MutatedConfig& operator=(const MutatedConfig&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  inline static int64_t next_id_{0};
  std::filesystem::path path_;
};

void expect_common_geometry_and_tail(const mimo::RuntimeConfig& config,
                                     int64_t max_az, int64_t max_clusters) {
  EXPECT_EQ(config.array_geometry.num_elements(), 32);
  EXPECT_EQ(config.extents.max_tx, 2);
  EXPECT_EQ(config.extents.max_rx, 16);
  EXPECT_EQ(config.extents.max_virtual_channels, 32);
  EXPECT_EQ(config.extents.max_channels, 32);
  EXPECT_EQ(config.extents.max_features, 7);
  EXPECT_EQ(config.extents.max_snapshots, 1);
  EXPECT_EQ(config.extents.max_az, max_az);
  EXPECT_EQ(config.extents.max_el, max_az);
  EXPECT_EQ(config.extents.max_clusters, max_clusters);
  EXPECT_EQ(config.report_index_columns, (std::vector<int64_t>{6, 5}));
}

TEST(FmcwMimoConfig, DecodesTdmDefault) {
  const auto config = mimo::load_tdm_config(
      config_path("mimo_tdm/config/mimo_tdm.yaml"));

  EXPECT_EQ(config.num_frames, 250);
  EXPECT_EQ(config.scene.range_loss.model,
            radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
  EXPECT_EQ(config.scene_visibility.model,
            radar::sim::SceneVisibilityModel::kFrontHalfSpace);
  EXPECT_EQ(config.tx_schedule.num_tx, 2);
  EXPECT_EQ(config.tx_schedule.tx_order, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.deinterleave_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->tx_positions, config.tx_positions);
  EXPECT_EQ(config.compensation_params->rx_positions, config.rx_positions);
  EXPECT_EQ(config.compensation_params->overlap_policy,
            holoscan::radar::OverlapPolicy::kKeepAll);
  EXPECT_EQ(config.compensation_params->snapshot_window,
            (std::vector<int64_t>{0, 0}));
  EXPECT_EQ(config.compensation_params->doppler_edge,
            holoscan::radar::SnapshotEdgeMode::kWrap);
  EXPECT_EQ(config.compensation_params->range_edge,
            holoscan::radar::SnapshotEdgeMode::kShift);
  EXPECT_DOUBLE_EQ(
      config.compensation_params->waveform.chirp_repetition_interval(),
      50.0e-6);
  EXPECT_DOUBLE_EQ(config.scene.waveform.chirp_repetition_interval, 25.0e-6);
  EXPECT_EQ(config.clutter_params->method,
            holoscan::radar::ClutterFilterMethod::kMeanSubtraction);
  EXPECT_EQ(config.clutter_params->doppler_window.type,
            holoscan::radar::WindowType::kHann);
  EXPECT_EQ(config.clutter_params->doppler_window,
            config.doppler_params->doppler_window);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.beamscan_params->array_geometry, config.array_geometry);
  EXPECT_EQ(config.extents.max_chirps, 64);
  ASSERT_TRUE(config.extents.max_interleaved_chirps.has_value());
  EXPECT_EQ(*config.extents.max_interleaved_chirps, 128);
  EXPECT_EQ(config.extents.max_doppler_bins, 64);
  EXPECT_EQ(config.extents.max_detections, 64 * 256);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesTdmSmoke) {
  const auto config = mimo::load_tdm_config(
      config_path("mimo_tdm/config/mimo_tdm_smoke.yaml"));

  EXPECT_EQ(config.num_frames, 2);
  EXPECT_FALSE(config.visualize);
  EXPECT_EQ(config.tx_schedule.num_tx, 2);
  EXPECT_EQ(config.tx_schedule.tx_order, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.deinterleave_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->tx_schedule, config.tx_schedule);
  EXPECT_EQ(config.compensation_params->overlap_policy,
            holoscan::radar::OverlapPolicy::kKeepAll);
  EXPECT_EQ(config.compensation_params->snapshot_window,
            (std::vector<int64_t>{0, 0}));
  EXPECT_DOUBLE_EQ(config.scene.waveform.chirp_repetition_interval, 25.0e-6);
  EXPECT_EQ(config.clutter_params->method,
            holoscan::radar::ClutterFilterMethod::kMeanSubtraction);
  EXPECT_EQ(config.clutter_params->doppler_window,
            config.doppler_params->doppler_window);
  EXPECT_EQ(config.doppler_params->doppler_window.type,
            holoscan::radar::WindowType::kHann);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.extents.max_chirps, 64);
  ASSERT_TRUE(config.extents.max_interleaved_chirps.has_value());
  EXPECT_EQ(*config.extents.max_interleaved_chirps, 128);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesLinearDdmDefault) {
  const auto config = mimo::load_ddm_config(
      config_path("mimo_ddm/config/mimo_ddm_linear.yaml"));

  EXPECT_FALSE(config.correlate);
  EXPECT_EQ(config.scene.range_loss.model,
            radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
  EXPECT_EQ(config.scene_visibility.model,
            radar::sim::SceneVisibilityModel::kFrontHalfSpace);
  EXPECT_EQ(config.ddm_code.num_tx, 2);
  EXPECT_EQ(config.doppler_offsets, (std::vector<double>{0.0, 0.5}));
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_DOUBLE_EQ(config.notch_params->center_velocity, 0.0);
  EXPECT_DOUBLE_EQ(config.notch_params->stop_half_width, 0.61);
  EXPECT_DOUBLE_EQ(config.notch_params->transition_width, 1.22);
  EXPECT_TRUE(config.notch_params->cyclic_velocity_axis);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.beamscan_params->array_geometry, config.array_geometry);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 60 * 256);
  EXPECT_FALSE(config.extents.max_interleaved_chirps.has_value());
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesLinearDdmSmoke) {
  const auto config = mimo::load_ddm_config(
      config_path(
          "mimo_ddm/config/mimo_ddm_linear_smoke.yaml"));

  EXPECT_FALSE(config.correlate);
  EXPECT_FALSE(config.visualize);
  EXPECT_EQ(config.doppler_offsets, (std::vector<double>{0.0, 0.5}));
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_DOUBLE_EQ(config.notch_params->stop_half_width, 0.61);
  EXPECT_DOUBLE_EQ(config.notch_params->transition_width, 1.22);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 60 * 256);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesDocumentedNonlinearDdmPreset) {
  const auto config = mimo::load_ddm_config(
      config_path("mimo_ddm/config/mimo_ddm_nonlinear.yaml"));

  EXPECT_TRUE(config.correlate);
  EXPECT_EQ(config.scene.range_loss.model,
            radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
  EXPECT_EQ(config.scene_visibility.model,
            radar::sim::SceneVisibilityModel::kFrontHalfSpace);
  EXPECT_TRUE(config.doppler_offsets.empty());
  EXPECT_EQ(config.ddm_code.num_tx, 2);
  EXPECT_EQ(config.ddm_code.phase_codes.size(), 2U * 128U);
  EXPECT_EQ(config.ddm_code.guard_bands, 0);
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_DOUBLE_EQ(config.notch_params->stop_half_width, 0.61);
  EXPECT_DOUBLE_EQ(config.notch_params->transition_width, 1.22);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 128 * 256);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesNonlinearDdmGeneratorAliases) {
  for (const char* generator : {"zc", "sin"}) {
    const MutatedConfig preset(
        "mimo_ddm/config/mimo_ddm_nonlinear_smoke.yaml",
        "generate_phase_codes: zadoff_chu",
        "generate_phase_codes: " + std::string(generator));
    const auto config = mimo::load_ddm_config(preset.path());

    EXPECT_TRUE(config.correlate);
    EXPECT_EQ(config.ddm_code.num_tx, 2);
    EXPECT_EQ(config.ddm_code.phase_codes.size(), 2U * 128U);
    EXPECT_TRUE(config.ddm_code.doppler_offsets.empty());
    EXPECT_EQ(config.ddm_code.guard_bands, 0);
  }
}

TEST(FmcwMimoConfig, SmokeConfigsOnlyApplyAllowedOverrides) {
  expect_smoke_matches_production(
      "mimo_tdm/config/mimo_tdm.yaml",
      "mimo_tdm/config/mimo_tdm_smoke.yaml");
  expect_smoke_matches_production(
      "mimo_ddm/config/mimo_ddm_linear.yaml",
      "mimo_ddm/config/mimo_ddm_linear_smoke.yaml");
  expect_smoke_matches_production(
      "mimo_ddm/config/mimo_ddm_nonlinear.yaml",
      "mimo_ddm/config/mimo_ddm_nonlinear_smoke.yaml");
}

TEST(FmcwMimoConfig, RejectsWrongLoaderSection) {
  EXPECT_THROW(
      mimo::load_tdm_config(
          config_path("mimo_ddm/config/mimo_ddm_linear.yaml")),
      std::runtime_error);
  EXPECT_THROW(
      mimo::load_ddm_config(
          config_path("mimo_tdm/config/mimo_tdm.yaml")),
      std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsAuthoredDeinterleaveSchedule) {
  const MutatedConfig config(
      "mimo_tdm/config/mimo_tdm_smoke.yaml", "mimo_deinterleave: {}",
      "mimo_deinterleave: {tx_schedule: {num_tx: 2}}");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsAuthoredDerivedSparseFields) {
  {
    const MutatedConfig config(
        "mimo_tdm/config/mimo_tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    waveform: {}\n    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
  {
    const MutatedConfig config(
        "mimo_tdm/config/mimo_tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    tx_schedule: {num_tx: 2}\n    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
  {
    const MutatedConfig config(
        "mimo_tdm/config/mimo_tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    tx_positions: [0.0, 0.0, 0.0]\n"
        "    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
  {
    const MutatedConfig config(
        "mimo_tdm/config/mimo_tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    rx_positions: [0.0, 0.0, 0.0]\n"
        "    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
}

TEST(FmcwMimoConfig, RejectsTdmPriMismatch) {
  const MutatedConfig config(
      "mimo_tdm/config/mimo_tdm_smoke.yaml", "chirp_repetition_interval: 50.0e-6",
      "chirp_repetition_interval: 51.0e-6");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsDdmTxMismatch) {
  const MutatedConfig config("mimo_ddm/config/mimo_ddm_linear_smoke.yaml",
                             "ddm_code:\n        num_tx: 2",
                             "ddm_code:\n        num_tx: 3");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsDdmClutterFilter) {
  const MutatedConfig config(
      "mimo_ddm/config/mimo_ddm_linear_smoke.yaml", "virtual_array_assembly:\n",
      "clutter_filter:\n    method: mean_subtraction\n"
      "virtual_array_assembly:\n");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsMalformedSchemeSpecificConfig) {
  const MutatedConfig config(
      "mimo_ddm/config/mimo_ddm_linear.yaml", "    targets:\n",
      "    ddm:\n"
      "        generate_phase_codes: zadoff_chu\n"
      "    targets:\n");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsInvalidClusteredTailShape) {
  const MutatedConfig config(
      "mimo_tdm/config/mimo_tdm_smoke.yaml",
      "        - score\n        - range_bin\n        - doppler_bin\n",
      "        - score\n        - doppler_bin\n        - range_bin\n");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, DecodesSharedRangeLossBlock) {
  const MutatedConfig preset(
      "mimo_tdm/config/mimo_tdm_smoke.yaml", "    tx_array:\n",
      "    range_loss:\n"
      "        model: monostatic_free_space\n"
      "        reference_range: 40.0\n"
      "    tx_array:\n");
  const auto config = mimo::load_tdm_config(preset.path());
  EXPECT_EQ(config.scene.range_loss.model,
            radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 40.0);
}

}  // namespace
