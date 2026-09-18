// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/config/io.hpp>

#include "radar_app/simple/simo/config.hpp"

namespace {

namespace simo = holoscan::radar::app::simo;
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
  YAML::Node targets(YAML::NodeType::Sequence);
  targets.push_back(expected["app"]["targets"][0]);
  expected["app"]["targets"] = targets;
  EXPECT_EQ(YAML::Dump(expected), YAML::Dump(smoke));
}

TEST(FmcwSimoConfig, DecodesDetectThenBeamformSmoke) {
  const auto config =
      simo::load_detect_then_beamform_config(
          config_path("simo_detect_then_beamform/config/simo_detect_then_beamform_smoke.yaml"));

  EXPECT_EQ(config.num_frames, 2);
  EXPECT_DOUBLE_EQ(config.frame_period, 0.1);
  EXPECT_EQ(config.scene.num_chirps, 128);
  EXPECT_EQ(config.scene.samples_per_chirp, 256);
  EXPECT_EQ(config.scene.targets.size(), 1U);
  ASSERT_TRUE(config.scene.snr_db.has_value());
  EXPECT_DOUBLE_EQ(*config.scene.snr_db, 5.0);
  EXPECT_EQ(config.clutter_params->method,
            holoscan::radar::ClutterFilterMethod::kMeanSubtraction);
  EXPECT_EQ(config.array_geometry.num_elements(), 64);
  EXPECT_EQ(config.cfar_params->nci_axes, (std::vector<int64_t>{0}));
  EXPECT_EQ(config.cfar_params->nci_fold, 64);
  EXPECT_EQ(config.gather_params->doppler_edge,
            holoscan::radar::SnapshotEdgeMode::kWrap);
  EXPECT_EQ(config.gather_params->range_edge,
            holoscan::radar::SnapshotEdgeMode::kShift);
  EXPECT_EQ(config.beamscan_params->array_geometry,
            config.array_geometry);
  EXPECT_EQ(config.extents.max_snapshots, 1);
  EXPECT_EQ(config.extents.max_features, 7);
  EXPECT_EQ(config.assembly_params->features,
            (std::vector<holoscan::radar::AssemblyFeature>{
                holoscan::radar::AssemblyFeature::kRange,
                holoscan::radar::AssemblyFeature::kVelocity,
                holoscan::radar::AssemblyFeature::kAzEstimate,
                holoscan::radar::AssemblyFeature::kElEstimate,
                holoscan::radar::AssemblyFeature::kScore,
                holoscan::radar::AssemblyFeature::kRangeBin,
                holoscan::radar::AssemblyFeature::kDopplerBin}));
  EXPECT_EQ(config.dbscan_params->epsilon,
            (std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5}));
  EXPECT_EQ(config.reduction_params->feature_weights,
            (std::vector<double>{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}));
  EXPECT_EQ(config.extents.max_az, 13);
  EXPECT_EQ(config.extents.max_el, 13);
  EXPECT_EQ(config.beamscan_params->az_coordinates.size(), 13U);
  EXPECT_EQ(config.beamscan_params->el_coordinates.size(), 13U);
  EXPECT_EQ(config.reduction_params->representative,
            holoscan::radar::RepresentativePolicy::kWeightedMean);
  EXPECT_FALSE(config.visualize);
}

TEST(FmcwSimoConfig, DecodesBeamformThenDetectSmoke) {
  const auto config =
      simo::load_beamform_then_detect_config(config_path(
          "simo_beamform_then_detect/config/"
          "simo_beamform_then_detect_smoke.yaml"));

  EXPECT_EQ(config.num_frames, 2);
  EXPECT_TRUE(config.cfar_params->nci_axes.empty());
  EXPECT_EQ(config.beamformer_params->array_geometry,
            config.array_geometry);
  EXPECT_EQ(config.extents.max_channels, 169);
  EXPECT_EQ(config.cfar_extents.max_tx, 13);
  EXPECT_EQ(config.cfar_extents.max_rx, 13);
  EXPECT_FALSE(config.cfar_extents.max_channels.has_value());
  EXPECT_EQ(config.cfar_params->peak_mode,
            holoscan::radar::CfarPeakMode::kAll);
  EXPECT_EQ(config.cfar_params->peak_radius,
            (std::vector<int64_t>{2, 2, 2, 2}));
  EXPECT_EQ(config.extents.max_features, 9);
  EXPECT_EQ(config.assembly_params->features,
            (std::vector<holoscan::radar::AssemblyFeature>{
                holoscan::radar::AssemblyFeature::kRange,
                holoscan::radar::AssemblyFeature::kVelocity,
                holoscan::radar::AssemblyFeature::kAz,
                holoscan::radar::AssemblyFeature::kEl,
                holoscan::radar::AssemblyFeature::kScore,
                holoscan::radar::AssemblyFeature::kRangeBin,
                holoscan::radar::AssemblyFeature::kDopplerBin,
                holoscan::radar::AssemblyFeature::kAzBin,
                holoscan::radar::AssemblyFeature::kElBin}));
}

TEST(FmcwSimoConfig, DecodesBothDefaultWorkflowBlocks) {
  const auto detect =
      simo::load_detect_then_beamform_config(
          config_path("simo_detect_then_beamform/config/simo_detect_then_beamform.yaml"));
  const auto beamform =
      simo::load_beamform_then_detect_config(config_path(
          "simo_beamform_then_detect/config/"
          "simo_beamform_then_detect.yaml"));

  EXPECT_EQ(detect.num_frames, 250);
  EXPECT_EQ(detect.scene.targets.size(), 5U);
  EXPECT_EQ(detect.extents.max_az, 13);
  EXPECT_EQ(beamform.extents.max_el, 13);
  ASSERT_EQ(detect.beamscan_params->az_coordinates.size(), 13U);
  ASSERT_EQ(beamform.beamformer_params->el_coordinates.size(), 13U);
  for (std::size_t index = 0; index < 13U; ++index) {
    const double expected = -0.3 + static_cast<double>(index) * 0.05;
    EXPECT_NEAR(detect.beamscan_params->az_coordinates[index], expected, 1.0e-14);
  }
  for (std::size_t index = 0; index < 13U; ++index) {
    const double expected = -0.3 + static_cast<double>(index) * 0.05;
    EXPECT_NEAR(beamform.beamformer_params->el_coordinates[index], expected,
                1.0e-14);
  }
  EXPECT_DOUBLE_EQ(detect.beamscan_params->az_coordinates.back(), 0.3);
  EXPECT_DOUBLE_EQ(beamform.beamformer_params->el_coordinates.back(), 0.3);
  EXPECT_EQ(detect.cfar_params->peak_mode,
            holoscan::radar::CfarPeakMode::kNone);
  EXPECT_EQ(beamform.cfar_params->peak_mode,
            holoscan::radar::CfarPeakMode::kAll);
  EXPECT_EQ(beamform.cfar_params->peak_radius,
            (std::vector<int64_t>{2, 2, 2, 2}));
  EXPECT_EQ(detect.reduction_params->representative,
            holoscan::radar::RepresentativePolicy::kWeightedMean);
  EXPECT_EQ(detect.reduction_params->feature_weights,
            (std::vector<double>{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}));
  EXPECT_EQ(detect.dbscan_params->epsilon,
            (std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5}));
  EXPECT_EQ(detect.extents.max_features, 7);
  EXPECT_EQ(beamform.extents.max_features, 9);
  EXPECT_TRUE(detect.visualize);
  EXPECT_DOUBLE_EQ(detect.visualization.dynamic_range_db, 60.0);
  EXPECT_DOUBLE_EQ(detect.visualization.detection_radius, 0.010);
  EXPECT_EQ(detect.visualization.history_frames, 120);
  EXPECT_EQ(detect.visualization.detection_trail_frames, 50);
}

TEST(FmcwSimoConfig, SmokeConfigsOnlyApplyAllowedOverrides) {
  expect_smoke_matches_production(
      "simo_detect_then_beamform/config/simo_detect_then_beamform.yaml",
      "simo_detect_then_beamform/config/simo_detect_then_beamform_smoke.yaml");
  expect_smoke_matches_production(
      "simo_beamform_then_detect/config/simo_beamform_then_detect.yaml",
      "simo_beamform_then_detect/config/simo_beamform_then_detect_smoke.yaml");
}

TEST(FmcwSimoConfig, RejectsAuthoredDerivedFields) {
  auto document = holoscan::config_io::parse(
      "{derived: 1}", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_THROW(simo::reject_authored(*document, "/derived"),
               std::runtime_error);
  EXPECT_NO_THROW(simo::reject_authored(*document, "/missing"));
}

TEST(FmcwSimoConfig, RejectsInvalidScanCoordinateRangeMapping) {
  auto document = holoscan::config_io::parse(
      "{grid: {min: 0.0, max: 1.0, step: 0.3}}",
      holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_THROW(
      simo::normalize_scan_coordinate_field(std::move(*document), "/grid"),
      std::runtime_error);
}

}  // namespace
