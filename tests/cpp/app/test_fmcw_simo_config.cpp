// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/config/io.hpp>

#include "config.hpp"  // NOLINT(build/include_subdir): app-private header
#include "radar_app/app/graph_params.hpp"

namespace {

namespace simo = holoscan::radar::app::simo;
namespace radar = holoscan::radar;

std::filesystem::path config_path(const char* name) {
  return std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
         "applications/fmcw/simo/config" / name;
}

void expect_grid_near(const std::vector<double>& actual,
                      const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1.0e-14);
  }
}

TEST(FmcwSimoConfig, DecodesDetectThenBeamformSmoke) {
  const auto config =
      simo::load_detect_then_beamform_config(config_path("smoke.yaml"));

  EXPECT_EQ(config.num_frames, 5);
  EXPECT_DOUBLE_EQ(config.frame_period, 0.1);
  EXPECT_EQ(config.scene.num_chirps, 64);
  EXPECT_EQ(config.scene.samples_per_chirp, 256);
  EXPECT_EQ(config.scene.targets.size(), 1U);
  ASSERT_TRUE(config.scene.snr_db.has_value());
  EXPECT_DOUBLE_EQ(*config.scene.snr_db, 5.0);
  EXPECT_EQ(config.clutter_params->method,
            holoscan::radar::ClutterFilterMethod::kNone);
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
  EXPECT_EQ(config.extents.max_az, 5);
  EXPECT_EQ(config.extents.max_el, 5);
  EXPECT_EQ(config.beamscan_params->az_angles,
            (std::vector<double>{-0.3, -0.15, 0.0, 0.15, 0.3}));
  EXPECT_EQ(config.beamscan_params->el_angles,
            (std::vector<double>{-0.3, -0.15, 0.0, 0.15, 0.3}));
  EXPECT_EQ(config.reduction_params->representative,
            holoscan::radar::RepresentativePolicy::kWeightedMean);
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
      simo::load_detect_then_beamform_config(config_path("default.yaml"));
  const auto beamform =
      simo::load_beamform_then_detect_config(config_path("default.yaml"));

  EXPECT_EQ(detect.num_frames, 250);
  EXPECT_EQ(detect.scene.targets.size(), 5U);
  EXPECT_EQ(detect.extents.max_az, 13);
  EXPECT_EQ(beamform.extents.max_el, 13);
  ASSERT_EQ(detect.beamscan_params->az_angles.size(), 13U);
  ASSERT_EQ(beamform.beamformer_params->el_angles.size(), 13U);
  for (std::size_t index = 0; index < 13U; ++index) {
    const double expected = -0.3 + static_cast<double>(index) * 0.05;
    EXPECT_NEAR(detect.beamscan_params->az_angles[index], expected, 1.0e-14);
    EXPECT_NEAR(beamform.beamformer_params->el_angles[index], expected,
                1.0e-14);
  }
  EXPECT_DOUBLE_EQ(detect.beamscan_params->az_angles.back(), 0.3);
  EXPECT_DOUBLE_EQ(beamform.beamformer_params->el_angles.back(), 0.3);
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

TEST(FmcwSimoConfig, RejectsAuthoredDerivedFields) {
  auto document = holoscan::config_io::parse(
      "{derived: 1}", holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_THROW(simo::reject_authored(*document, "/derived"),
               std::runtime_error);
  EXPECT_NO_THROW(simo::reject_authored(*document, "/missing"));
}

TEST(FmcwSimoConfig, RejectsInvalidAngleRangeMapping) {
  auto document = holoscan::config_io::parse(
      "{grid: {min: 0.0, max: 1.0, step: 0.3}}",
      holoscan::config_io::Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_THROW(
      simo::normalize_angle_grid_field(std::move(*document), "/grid"),
      std::runtime_error);
}

TEST(GraphParamsAngleGrid, LoadsBothYamlFormsForAllScanOperators) {
  const YAML::Node node = YAML::Load(R"(
az_angles: {min: -0.2, max: 0.2, step: 0.1}
el_angles: [0.0]
)");
  const std::vector<double> expected_az{-0.2, -0.1, 0.0, 0.1, 0.2};
  const std::vector<double> expected_el{0.0};

  radar::BeamscanParams beamscan;
  radar::app::load_beamscan_params(node, &beamscan);
  expect_grid_near(beamscan.az_angles, expected_az);
  EXPECT_EQ(beamscan.el_angles, expected_el);

  radar::PhaseShiftBeamformerParams beamformer;
  radar::app::load_phase_shift_beamformer_params(node, &beamformer);
  expect_grid_near(beamformer.az_angles, expected_az);
  EXPECT_EQ(beamformer.el_angles, expected_el);

  radar::MVDRParams mvdr;
  radar::app::load_mvdr_params(node, &mvdr);
  expect_grid_near(mvdr.az_angles, expected_az);
  EXPECT_EQ(mvdr.el_angles, expected_el);

  radar::MUSICParams music;
  radar::app::load_music_params(node, &music);
  expect_grid_near(music.az_angles, expected_az);
  EXPECT_EQ(music.el_angles, expected_el);
}

}  // namespace
