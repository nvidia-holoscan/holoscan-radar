// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/config/io.hpp>

#include "radar_app/simple/siso/config.hpp"

namespace {

namespace siso = holoscan::radar::app::siso;
namespace radar = holoscan::radar;

std::filesystem::path config_path(const char* name) {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
         "applications/fmcw/siso/config" / name;
}

TEST(FmcwSisoConfig, DecodesProductionAndDerivesRankTwoExtents) {
  const auto config = siso::load_config(config_path("siso.yaml"));
  EXPECT_EQ(config.num_frames, 250);
  EXPECT_DOUBLE_EQ(config.frame_period, 0.1);
  EXPECT_EQ(config.scene.num_chirps, 128);
  EXPECT_EQ(config.scene.samples_per_chirp, 256);
  EXPECT_EQ(config.scene.targets.size(), 5U);
  ASSERT_TRUE(config.scene.snr_db.has_value());
  EXPECT_DOUBLE_EQ(*config.scene.snr_db, 20.0);
  ASSERT_EQ(config.scene.rx_positions.size(), 1U);
  EXPECT_EQ(config.scene.rx_positions[0],
            (radar::sim::Vec3{0.0, 0.0, 0.0}));
  EXPECT_EQ(config.cfar_params->nci_fold, 1);
  EXPECT_TRUE(config.cfar_params->nci_axes.empty());
  EXPECT_EQ(config.assembly_params->features,
            (std::vector<radar::AssemblyFeature>{
                radar::AssemblyFeature::kRange,
                radar::AssemblyFeature::kVelocity}));
  EXPECT_EQ(config.dbscan_params->epsilon,
            (std::vector<double>{3.0, 2.0}));
  EXPECT_EQ(config.reduction_params->representative,
            radar::RepresentativePolicy::kMedoid);
  EXPECT_EQ(config.extents.max_chirps, 128);
  EXPECT_EQ(config.extents.max_samples_per_chirp, 256);
  EXPECT_EQ(config.extents.max_range_bins, 256);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 32768);
  EXPECT_EQ(config.extents.max_features, 2);
  EXPECT_EQ(config.extents.max_clusters, 32);
  EXPECT_FALSE(config.extents.max_channels.has_value());
  EXPECT_TRUE(config.visualize);
}

TEST(FmcwSisoConfig, DecodesSmoke) {
  const auto config = siso::load_config(config_path("siso_smoke.yaml"));
  EXPECT_EQ(config.num_frames, 2);
  ASSERT_EQ(config.scene.targets.size(), 1U);
  EXPECT_DOUBLE_EQ(config.scene.targets[0].position[0], 245.0);
  EXPECT_DOUBLE_EQ(config.scene.targets[0].velocity[0], -6.5006);
  EXPECT_FALSE(config.visualize);
}

TEST(FmcwSisoConfig, SmokeOnlyAppliesAllowedOverrides) {
  YAML::Node expected = YAML::LoadFile(config_path("siso.yaml").string());
  const YAML::Node smoke =
      YAML::LoadFile(config_path("siso_smoke.yaml").string());
  expected["app"]["run"]["num_frames"] = 2;
  expected["app"]["visualization"]["enabled"] = false;
  YAML::Node targets(YAML::NodeType::Sequence);
  targets.push_back(expected["app"]["targets"][0]);
  expected["app"]["targets"] = targets;
  EXPECT_EQ(YAML::Dump(expected), YAML::Dump(smoke));
}

TEST(FmcwSisoConfig, RejectsAuthoredDerivedFields) {
  auto document = holoscan::config_io::parse(
      "{derived: 1}", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_THROW(siso::reject_authored(*document, "/derived"),
               std::runtime_error);
  EXPECT_NO_THROW(siso::reject_authored(*document, "/missing"));
}

TEST(FmcwSisoConfig, RejectsMalformedApplicationParameters) {
  siso::AppParams params;
  params.cube.num_chirps = 128;
  params.cube.samples_per_chirp = 256;
  params.targets.push_back(siso::TargetConfig{
      .range = -1.0, .velocity = 0.0, .amplitude = 1.0});
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
}

}  // namespace
