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

namespace {

namespace adaptive = holoscan::radar::app::adaptive_aoa;
using holoscan::radar::CfarPeakMode;
using holoscan::radar::OverlapPolicy;
using holoscan::radar::SnapshotEdgeMode;

std::filesystem::path source_config() {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
         "applications/fmcw/adaptive_aoa/config/adaptive_aoa.yaml";
}

class SelectedConfig {
 public:
  explicit SelectedConfig(const std::string& name) {
    path_ = std::filesystem::temp_directory_path() /
            ("adaptive_aoa_" + std::to_string(next_id_++) + ".yaml");
    std::ofstream output(path_);
    output << "extends: " << source_config().string() << "\napp:\n    aoa_algorithm: " << name
           << "\n";
    if (!output)
      throw std::runtime_error("cannot write adaptive config");
  }

  ~SelectedConfig() { std::filesystem::remove(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  inline static int64_t next_id_{0};
  std::filesystem::path path_;
};

TEST(AdaptiveAoAConfig, DecodesTopologyAndFoldedPeakCfar) {
  const auto config = adaptive::load_config(source_config());
  EXPECT_EQ(config.algorithm, adaptive::Algorithm::kMusic);
  EXPECT_EQ(config.array_geometry.num_elements(), 32);
  EXPECT_EQ(config.tx_schedule.tx_order, (std::vector<int64_t>{0, 1}));
  EXPECT_DOUBLE_EQ(config.scene.waveform.chirp_repetition_interval, 25.0e-6);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0, 1}));
  EXPECT_EQ(config.cfar_params->peak_mode, CfarPeakMode::kRangeDoppler);
  EXPECT_EQ(config.compensation_params->overlap_policy, OverlapPolicy::kDedup);
  EXPECT_EQ(config.compensation_params->snapshot_window, (std::vector<int64_t>{1, 1}));
  EXPECT_EQ(config.compensation_params->doppler_edge, SnapshotEdgeMode::kWrap);
  EXPECT_EQ(config.compensation_params->range_edge, SnapshotEdgeMode::kShift);
  EXPECT_EQ(config.extents.max_snapshots, 9);
  EXPECT_EQ(config.extents.max_sources, 2);
  EXPECT_EQ(config.extents.max_points, *config.extents.max_detections * 2);
  ASSERT_TRUE(config.music_params.has_value());
  EXPECT_FALSE(config.root_music_params.has_value());
  EXPECT_FALSE(config.esprit_params.has_value());
  EXPECT_FALSE(config.mvdr_params.has_value());
}

TEST(AdaptiveAoAConfig, InstantiatesOnlySelectedEstimatorParameters) {
  for (const auto& [name, expected] : std::vector<std::pair<std::string, adaptive::Algorithm>>{
           {"music", adaptive::Algorithm::kMusic},
           {"root_music", adaptive::Algorithm::kRootMusic},
           {"esprit", adaptive::Algorithm::kEsprit},
           {"mvdr", adaptive::Algorithm::kMvdr}}) {
    const SelectedConfig selected(name);
    const auto config = adaptive::load_config(selected.path());
    EXPECT_EQ(config.algorithm, expected) << name;
    EXPECT_EQ(config.music_params.has_value(), expected == adaptive::Algorithm::kMusic) << name;
    EXPECT_EQ(config.root_music_params.has_value(), expected == adaptive::Algorithm::kRootMusic)
        << name;
    EXPECT_EQ(config.esprit_params.has_value(), expected == adaptive::Algorithm::kEsprit) << name;
    EXPECT_EQ(config.mvdr_params.has_value(), expected == adaptive::Algorithm::kMvdr) << name;
  }
}

TEST(AdaptiveAoAConfig, RejectsUnknownAlgorithm) {
  const SelectedConfig invalid("beam_magic");
  EXPECT_THROW(adaptive::load_config(invalid.path()), std::runtime_error);
}

}  // namespace
