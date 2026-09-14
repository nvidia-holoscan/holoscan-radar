// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <vector>

#include <holoscan/config/io.hpp>

#include "config.hpp"  // NOLINT(build/include_subdir): app-private header

namespace {

namespace simo = holoscan::radar::app::simo;

std::filesystem::path config_path(const char* name) {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
         "applications/fmcw/simo/config" / name;
}

TEST(FmcwSimoConfig, DecodesDetectThenBeamformSmoke) {
  const auto config =
      simo::load_detect_then_beamform_config(config_path("smoke.yaml"));

  EXPECT_EQ(config.num_frames, 5);
  EXPECT_EQ(config.scene.num_chirps, 64);
  EXPECT_EQ(config.scene.samples_per_chirp, 256);
  EXPECT_EQ(config.array_geometry.num_elements(), 16);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0}));
  EXPECT_EQ(config.cfar_params->nci_fold, 16);
  EXPECT_EQ(config.gather_params->doppler_edge,
            holoscan::radar::SnapshotEdgeMode::kWrap);
  EXPECT_EQ(config.gather_params->range_edge,
            holoscan::radar::SnapshotEdgeMode::kShift);
  EXPECT_EQ(config.beamscan_params->array_geometry,
            config.array_geometry);
  EXPECT_EQ(config.extents.max_snapshots, 1);
  EXPECT_EQ(config.extents.max_az, 5);
  EXPECT_EQ(config.extents.max_el, 5);
  EXPECT_EQ(config.reduction_params->representative,
            holoscan::radar::RepresentativePolicy::kMedoid);
  EXPECT_FALSE(config.visualize);
}

TEST(FmcwSimoConfig, DecodesBeamformThenDetectSmoke) {
  const auto config =
      simo::load_beamform_then_detect_config(config_path("smoke.yaml"));

  EXPECT_EQ(config.num_frames, 5);
  EXPECT_TRUE(config.cfar_params->nci_axes.empty());
  EXPECT_EQ(config.beamformer_params->array_geometry,
            config.array_geometry);
  EXPECT_EQ(config.extents.max_channels, 25);
  EXPECT_EQ(config.cfar_extents.max_tx, 5);
  EXPECT_EQ(config.cfar_extents.max_rx, 5);
  EXPECT_FALSE(config.cfar_extents.max_channels.has_value());
  EXPECT_EQ(config.reduction_params->representative,
            holoscan::radar::RepresentativePolicy::kMedoid);
}

TEST(FmcwSimoConfig, DecodesBothDefaultWorkflowBlocks) {
  const auto detect =
      simo::load_detect_then_beamform_config(config_path("default.yaml"));
  const auto beamform =
      simo::load_beamform_then_detect_config(config_path("default.yaml"));

  EXPECT_EQ(detect.num_frames, 250);
  EXPECT_EQ(detect.scene.targets.size(), 1U);
  EXPECT_EQ(detect.extents.max_az, 13);
  EXPECT_EQ(beamform.extents.max_el, 13);
  EXPECT_EQ(beamform.dbscan_params->min_points, 2);
  EXPECT_TRUE(detect.visualize);
  EXPECT_DOUBLE_EQ(detect.visualization.dynamic_range_db, 60.0);
  EXPECT_DOUBLE_EQ(detect.visualization.detection_radius, 0.010);
  EXPECT_EQ(detect.visualization.history_frames, 120);
}

TEST(FmcwSimoConfig, RejectsAuthoredDerivedFields) {
  auto document = holoscan::config_io::parse(
      "{derived: 1}", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_THROW(simo::reject_authored(*document, "/derived"),
               std::runtime_error);
  EXPECT_NO_THROW(simo::reject_authored(*document, "/missing"));
}

}  // namespace
