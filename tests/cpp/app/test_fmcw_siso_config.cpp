// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/config/io.hpp>

#include "config.hpp"  // NOLINT(build/include_subdir): app-private header
#include "radar_common/math/constants.hpp"

namespace {

namespace siso = holoscan::radar::app::siso;
namespace radar = holoscan::radar;

std::filesystem::path config_path(const char* name) {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) / "applications/fmcw/siso/config" / name;
}

TEST(FmcwSisoConfig, DecodesProductionAndDerivesRankTwoExtents) {
  const auto config = siso::load_config(config_path("siso.yaml"));
  EXPECT_EQ(config.num_frames, 250);
  EXPECT_DOUBLE_EQ(config.frame_period, 0.1);
  EXPECT_EQ(config.scene.num_chirps, 128);
  EXPECT_EQ(config.scene.samples_per_chirp, 256);
  EXPECT_EQ(config.scene.targets.size(), 5U);
  ASSERT_TRUE(config.scene.snr_db.has_value());
  EXPECT_DOUBLE_EQ(*config.scene.snr_db, 5.0);
  EXPECT_EQ(config.scene.range_loss.model, radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
  ASSERT_EQ(config.scene.rx_positions.size(), 1U);
  EXPECT_EQ(config.scene.rx_positions[0], (radar::sim::Vec3{0.0, 0.0, 0.0}));
  EXPECT_EQ(config.cfar_params->nci_fold, 1);
  EXPECT_TRUE(config.cfar_params->nci_axes.empty());
  EXPECT_EQ(config.assembly_params->features,
            (std::vector<radar::AssemblyFeature>{radar::AssemblyFeature::kRange,
                                                 radar::AssemblyFeature::kVelocity}));
  EXPECT_EQ(config.dbscan_params->epsilon, (std::vector<double>{3.0, 2.0}));
  EXPECT_EQ(config.reduction_params->representative, radar::RepresentativePolicy::kMedoid);
  EXPECT_EQ(config.extents.max_chirps, 128);
  EXPECT_EQ(config.extents.max_samples_per_chirp, 256);
  EXPECT_EQ(config.extents.max_range_bins, 256);
  EXPECT_EQ(config.extents.max_doppler_bins, 128);
  EXPECT_EQ(config.extents.max_detections, 32768);
  EXPECT_EQ(config.extents.max_features, 2);
  EXPECT_EQ(config.extents.max_clusters, 32);
  EXPECT_FALSE(config.extents.max_channels.has_value());
  EXPECT_TRUE(config.visualize);
  EXPECT_EQ(config.scene_visibility.model, radar::sim::SceneVisibilityModel::kFrontHalfSpace);
}

TEST(FmcwSisoConfig, DecodesSmoke) {
  const auto config = siso::load_config(config_path("siso_smoke.yaml"));
  EXPECT_EQ(config.num_frames, 2);
  ASSERT_EQ(config.scene.targets.size(), 1U);
  EXPECT_DOUBLE_EQ(config.scene.targets[0].position[0], 245.0);
  EXPECT_DOUBLE_EQ(config.scene.targets[0].velocity[0], -6.5006);
  EXPECT_EQ(config.scene.range_loss.model, radar::sim::RangeLossModel::kNone);
  EXPECT_FALSE(config.visualize);
}

TEST(FmcwSisoConfig, SmokeOnlyAppliesAllowedOverrides) {
  YAML::Node expected = YAML::LoadFile(config_path("siso.yaml").string());
  const YAML::Node smoke = YAML::LoadFile(config_path("siso_smoke.yaml").string());
  expected["app"]["run"]["num_frames"] = 2;
  expected["app"]["visualization"]["enabled"] = false;
  expected["app"].remove("range_loss");
  expected["app"].remove("scene_visibility");
  expected["app"]["targets"] = smoke["app"]["targets"];
  EXPECT_EQ(YAML::Dump(expected), YAML::Dump(smoke));
}

TEST(FmcwSisoConfig, RejectsAuthoredDerivedFields) {
  auto document =
      holoscan::config_io::try_parse("{derived: 1}", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_THROW(siso::reject_authored(*document, "/derived"), std::runtime_error);
  EXPECT_NO_THROW(siso::reject_authored(*document, "/missing"));
}

TEST(FmcwSisoConfig, RejectsMalformedApplicationParameters) {
  siso::AppParams params;
  params.cube.num_chirps = 128;
  params.cube.samples_per_chirp = 256;
  params.targets.push_back(siso::TargetConfig{.range = -1.0, .velocity = 0.0, .amplitude = 1.0});
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
}

TEST(FmcwSisoConfig, ValidatesRangeLossConfiguration) {
  siso::AppParams params;
  params.cube.num_chirps = 2;
  params.cube.samples_per_chirp = 2;
  params.targets.push_back(siso::TargetConfig{.range = 40.0, .velocity = 0.0, .amplitude = 1.0});

  params.range_loss =
      radar::app::RangeLossConfig{.model = "monostatic_free_space", .reference_range = 40.0};
  const auto configured = siso::make_runtime_config(params);
  EXPECT_EQ(configured.scene.range_loss.model, radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(configured.scene.range_loss.reference_range, 40.0);

  params.range_loss->reference_range.reset();
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
  params.range_loss = radar::app::RangeLossConfig{.model = "unknown"};
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
  params.range_loss = radar::app::RangeLossConfig{.model = "none", .reference_range = -1.0};
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
}

TEST(FmcwSisoConfig, ValidatesSceneVisibilityConfiguration) {
  siso::AppParams params;
  params.cube.num_chirps = 2;
  params.cube.samples_per_chirp = 2;
  params.targets.push_back(siso::TargetConfig{.range = 40.0, .velocity = 0.0, .amplitude = 1.0});

  auto configured = siso::make_runtime_config(params);
  EXPECT_EQ(configured.scene_visibility.model, radar::sim::SceneVisibilityModel::kFrontHalfSpace);

  params.scene_visibility.model = "all";
  configured = siso::make_runtime_config(params);
  EXPECT_EQ(configured.scene_visibility.model, radar::sim::SceneVisibilityModel::kAll);

  params.scene_visibility.model = "frustum";
  params.scene_visibility.horizontal_fov = radar::kPi / 2.0;
  params.scene_visibility.vertical_fov = radar::kPi / 3.0;
  configured = siso::make_runtime_config(params);
  EXPECT_EQ(configured.scene_visibility.model, radar::sim::SceneVisibilityModel::kFrustum);
  EXPECT_DOUBLE_EQ(*configured.scene_visibility.horizontal_fov, radar::kPi / 2.0);

  params.scene_visibility.vertical_fov.reset();
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
  params.scene_visibility.vertical_fov = radar::kPi / 3.0;
  params.scene_visibility.up_direction = {1.0, 0.0, 0.0};
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
  params.scene_visibility.model = "unknown";
  EXPECT_THROW(siso::make_runtime_config(params), std::runtime_error);
}

}  // namespace
