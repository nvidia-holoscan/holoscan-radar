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
  EXPECT_EQ(detect.visualization.detection_trail_frames, 12);
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
