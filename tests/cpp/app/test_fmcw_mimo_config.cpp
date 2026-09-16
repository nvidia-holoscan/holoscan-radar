// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"  // NOLINT(build/include_subdir): app-private header
#include "radar_common/descriptors/window_spec.hpp"

namespace {

namespace mimo = holoscan::radar::app::mimo;

std::filesystem::path config_path(const char* name) {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
         "applications/fmcw/mimo/config" / name;
}

class MutatedConfig {
 public:
  MutatedConfig(const char* source_name, const std::string& old_text,
                const std::string& new_text) {
    std::ifstream input(config_path(source_name));
    if (!input) {
      throw std::runtime_error("cannot read source config");
    }
    std::string contents{std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>()};
    const std::size_t position = contents.find(old_text);
    if (position == std::string::npos ||
        contents.find(old_text, position + old_text.size()) !=
            std::string::npos) {
      throw std::runtime_error("config mutation anchor must occur once");
    }
    contents.replace(position, old_text.size(), new_text);
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
  const auto config = mimo::load_tdm_config(config_path("tdm.yaml"));

  EXPECT_EQ(config.num_frames, 250);
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
  const auto config = mimo::load_tdm_config(config_path("tdm_smoke.yaml"));

  EXPECT_EQ(config.num_frames, 2);
  EXPECT_FALSE(config.visualize);
  EXPECT_EQ(config.tx_schedule.num_tx, 2);
  EXPECT_TRUE(config.tx_schedule.tx_order.empty());
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
            holoscan::radar::WindowSpec{});
  EXPECT_EQ(config.doppler_params->doppler_window.type,
            holoscan::radar::WindowType::kHann);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.extents.max_chirps, 32);
  ASSERT_TRUE(config.extents.max_interleaved_chirps.has_value());
  EXPECT_EQ(*config.extents.max_interleaved_chirps, 64);
  expect_common_geometry_and_tail(config, 5, 16);
}

TEST(FmcwMimoConfig, DecodesLinearDdmDefault) {
  const auto config = mimo::load_ddm_config(config_path("ddm.yaml"));

  EXPECT_FALSE(config.correlate);
  EXPECT_EQ(config.ddm_code.num_tx, 2);
  EXPECT_EQ(config.doppler_offsets, (std::vector<double>{0.0, 0.5}));
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0}));
  EXPECT_EQ(config.cfar_params->nci_fold, 32);
  EXPECT_EQ(config.beamscan_params->array_geometry, config.array_geometry);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 60 * 256);
  EXPECT_FALSE(config.extents.max_interleaved_chirps.has_value());
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, DecodesLinearDdmSmoke) {
  const auto config = mimo::load_ddm_config(config_path("ddm_smoke.yaml"));

  EXPECT_FALSE(config.correlate);
  EXPECT_FALSE(config.visualize);
  EXPECT_EQ(config.doppler_offsets, (std::vector<double>{0.0, 0.5}));
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_EQ(config.extents.max_doppler_bins, 64);
  EXPECT_EQ(config.extents.max_detections, 28 * 256);
  expect_common_geometry_and_tail(config, 5, 16);
}

TEST(FmcwMimoConfig, DecodesNonlinearDdmPhaseCodes) {
  const auto config =
      mimo::load_ddm_config(config_path("ddm_phase_codes.yaml"));

  EXPECT_TRUE(config.correlate);
  EXPECT_TRUE(config.doppler_offsets.empty());
  EXPECT_EQ(config.ddm_code.num_tx, 2);
  EXPECT_EQ(config.ddm_code.phase_codes.size(), 2U * 128U);
  EXPECT_EQ(config.ddm_code.guard_bands, 0);
  EXPECT_TRUE(config.doppler_params->waveform.is_valid());
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 128 * 256);
  expect_common_geometry_and_tail(config, 13, 32);
}

TEST(FmcwMimoConfig, RejectsWrongLoaderSection) {
  EXPECT_THROW(mimo::load_tdm_config(config_path("ddm.yaml")),
               std::runtime_error);
  EXPECT_THROW(mimo::load_ddm_config(config_path("tdm.yaml")),
               std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsAuthoredDeinterleaveSchedule) {
  const MutatedConfig config(
      "tdm_smoke.yaml", "mimo_deinterleave: {}",
      "mimo_deinterleave: {tx_schedule: {num_tx: 2}}");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsAuthoredDerivedSparseFields) {
  {
    const MutatedConfig config(
        "tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    waveform: {}\n    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
  {
    const MutatedConfig config(
        "tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    tx_schedule: {num_tx: 2}\n    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
  {
    const MutatedConfig config(
        "tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    tx_positions: [0.0, 0.0, 0.0]\n"
        "    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
  {
    const MutatedConfig config(
        "tdm_smoke.yaml", "    overlap_policy: keep_all",
        "    rx_positions: [0.0, 0.0, 0.0]\n"
        "    overlap_policy: keep_all");
    EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
  }
}

TEST(FmcwMimoConfig, RejectsTdmPriMismatch) {
  const MutatedConfig config(
      "tdm_smoke.yaml", "chirp_repetition_interval: 50.0e-6",
      "chirp_repetition_interval: 51.0e-6");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsDdmTxMismatch) {
  const MutatedConfig config("ddm_smoke.yaml",
                             "ddm_code:\n        num_tx: 2",
                             "ddm_code:\n        num_tx: 3");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsDdmClutterFilter) {
  const MutatedConfig config(
      "ddm_smoke.yaml", "virtual_array_assembly:\n",
      "clutter_filter:\n    method: mean_subtraction\n"
      "virtual_array_assembly:\n");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsMalformedSchemeSpecificConfig) {
  const MutatedConfig config(
      "ddm_phase_codes.yaml", "ddm_dense_demultiplex: {}",
      "ddm_dense_demultiplex: {ddm_code: {num_tx: 2, "
      "doppler_offsets: [0.0, 0.5]}}");
  EXPECT_THROW(mimo::load_ddm_config(config.path()), std::runtime_error);
}

TEST(FmcwMimoConfig, RejectsInvalidClusteredTailShape) {
  const MutatedConfig config(
      "tdm_smoke.yaml",
      "        - score\n        - range_bin\n        - doppler_bin\n",
      "        - score\n        - doppler_bin\n        - range_bin\n");
  EXPECT_THROW(mimo::load_tdm_config(config.path()), std::runtime_error);
}

}  // namespace
