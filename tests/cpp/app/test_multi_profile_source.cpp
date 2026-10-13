// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/app/multi_profile_source.hpp"

namespace {

namespace app = holoscan::radar::app;
namespace workflow = holoscan::radar::app::multi_waveform;

holoscan::TimePoint time_at(int64_t timestamp) {
  return holoscan::TimePoint{timestamp, holoscan::ClockId{7}};
}

class MutatedConfig {
 public:
  MutatedConfig(const std::string& old_text, const std::string& new_text) {
    const auto source = std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
                        "applications/fmcw/multi_waveform/config/"
                        "multi_waveform_smoke.yaml";
    std::ifstream input(source);
    std::string contents{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const std::size_t position = contents.find(old_text);
    if (position == std::string::npos ||
        contents.find(old_text, position + old_text.size()) != std::string::npos) {
      throw std::runtime_error("mutation anchor must occur exactly once");
    }
    contents.replace(position, old_text.size(), new_text);
    path_ = std::filesystem::temp_directory_path() /
            ("multi_profile_config_" + std::to_string(next_id_++) + ".yaml");
    std::ofstream output(path_);
    output << contents;
  }

  ~MutatedConfig() { std::filesystem::remove(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  inline static int64_t next_id_{0};
  std::filesystem::path path_;
};

TEST(TransmitFrameCoordinator, PreservesTransmitStampAcrossDifferentSourceActivations) {
  app::TransmitFrameCoordinator coordinator(2U);
  coordinator.register_transmit(app::TransmitReceipt{11U, time_at(100), 1U});

  const auto early = coordinator.claim(11U);
  EXPECT_EQ(early.status, app::TransmitFrameCoordinator::ClaimStatus::kNotReady);
  EXPECT_EQ(early.transmit_time, time_at(100));
  const holoscan::TimePoint rx_activation_time = time_at(900);
  EXPECT_NE(early.transmit_time, rx_activation_time);

  coordinator.mark_ready(11U);
  const auto ready = coordinator.claim(11U);
  EXPECT_EQ(ready.status, app::TransmitFrameCoordinator::ClaimStatus::kReady);
  EXPECT_EQ(ready.transmit_time, time_at(100));
  EXPECT_EQ(ready.profile_index, 1U);
  EXPECT_EQ(coordinator.outstanding(), 0U);
  EXPECT_THROW(coordinator.claim(11U), std::runtime_error);
}

TEST(TransmitFrameCoordinator, ExpiresAndFinalizesUnmatchedFrames) {
  app::TransmitFrameCoordinator coordinator(2U);
  coordinator.register_transmit(app::TransmitReceipt{4U, time_at(40), 0U});
  EXPECT_THROW(coordinator.finalize(), std::runtime_error);
  coordinator.advance_poll(1U);
  EXPECT_THROW(coordinator.advance_poll(1U), std::runtime_error);
  coordinator.cancel(4U);
  EXPECT_NO_THROW(coordinator.finalize());
}

TEST(TransmitFrameCoordinator, RejectsDuplicatesOverflowAndDoubleReady) {
  app::TransmitFrameCoordinator coordinator(1U);
  coordinator.register_transmit(app::TransmitReceipt{1U, time_at(1), 0U});
  EXPECT_THROW(coordinator.register_transmit(app::TransmitReceipt{1U, time_at(2), 0U}),
               std::runtime_error);
  EXPECT_THROW(coordinator.register_transmit(app::TransmitReceipt{2U, time_at(2), 0U}),
               std::runtime_error);
  coordinator.mark_ready(1U);
  EXPECT_THROW(coordinator.mark_ready(1U), std::runtime_error);
  coordinator.cancel(1U);
  EXPECT_EQ(coordinator.outstanding(), 0U);
}

TEST(TransmitFrameCoordinator, RejectsPendingReceiveAtFinalization) {
  app::TransmitFrameCoordinator coordinator(1U);
  coordinator.register_receive(7U);
  EXPECT_THROW(coordinator.finalize(), std::runtime_error);
  EXPECT_THROW(coordinator.register_receive(7U), std::runtime_error);
  EXPECT_THROW(coordinator.register_receive(8U), std::runtime_error);
  coordinator.complete_receive(7U);
  EXPECT_THROW(coordinator.complete_receive(7U), std::runtime_error);
  EXPECT_NO_THROW(coordinator.finalize());
}

TEST(MultiProfileConfig, DecodesProductionRangeLoss) {
  const auto path = std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
                    "applications/fmcw/multi_waveform/config/"
                    "multi_waveform.yaml";
  const workflow::Config config = workflow::load_config(path);
  EXPECT_EQ(config.scene.range_loss.model,
            holoscan::radar::sim::RangeLossModel::kMonostaticFreeSpace);
  EXPECT_DOUBLE_EQ(config.scene.range_loss.reference_range, 100.0);
}

TEST(MultiProfileConfig, DecodesProfilesAndMaximumExtents) {
  const auto path = std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
                    "applications/fmcw/multi_waveform/config/"
                    "multi_waveform_smoke.yaml";
  const workflow::Config config = workflow::load_config(path);
  ASSERT_EQ(config.profiles.size(), 2U);
  EXPECT_EQ(config.profiles[0].name, "near");
  EXPECT_EQ(config.profiles[1].name, "far");
  EXPECT_EQ(config.profile_sequence, (std::vector<std::size_t>{0U, 1U}));
  EXPECT_EQ(config.visualization.profile, "near");
  EXPECT_EQ(config.visualization_profile_index, 0U);
  EXPECT_EQ(config.visualization_profile_max_gap_frames, 2U);
  EXPECT_EQ(config.extents.max_samples_per_chirp, 512);
  EXPECT_EQ(config.extents.max_chirps, 64);
  EXPECT_EQ(config.extents.max_interleaved_chirps, 128);
  EXPECT_DOUBLE_EQ(config.profiles[0].waveform.chirp_repetition_interval(), 50.0e-6);
  EXPECT_DOUBLE_EQ(config.profiles[1].waveform.chirp_repetition_interval(), 90.0e-6);
  EXPECT_DOUBLE_EQ(config.profiles[0].physical_waveform.chirp_repetition_interval, 25.0e-6);
  EXPECT_DOUBLE_EQ(config.profiles[1].physical_waveform.chirp_repetition_interval, 45.0e-6);
  EXPECT_EQ(config.scene.range_loss.model, holoscan::radar::sim::RangeLossModel::kNone);
  EXPECT_EQ(config.scene.analog_if.sampling_mode, holoscan::radar::sim::IfSamplingMode::kComplexIq);
  EXPECT_EQ(config.scene.analog_if.response,
            holoscan::radar::sim::AnalogIfResponseType::kRaisedCosine);
  EXPECT_DOUBLE_EQ(config.scene.analog_if.passband_fraction, 0.9);
  EXPECT_DOUBLE_EQ(config.scene.analog_if.stopband_fraction, 1.0);
  EXPECT_EQ(config.range_params->nfft, 0);
  EXPECT_EQ(config.doppler_params->nfft, 0);
  EXPECT_FALSE(config.range_params->waveform.is_valid());
  EXPECT_FALSE(config.doppler_params->waveform.is_valid());
  EXPECT_FALSE(config.compensation_params->waveform.is_valid());
}

TEST(MultiProfileConfig, DefaultsVisualizationToFirstScheduledProfile) {
  const MutatedConfig authored("        profile: near\n", "");
  const workflow::Config config = workflow::load_config(authored.path());
  EXPECT_EQ(config.visualization.profile, "near");
  EXPECT_EQ(config.visualization_profile_index, 0U);
  EXPECT_EQ(config.visualization_profile_max_gap_frames, 2U);
}

TEST(MultiProfileConfig, SelectsNamedVisualizationProfile) {
  const MutatedConfig authored("profile: near", "profile: far");
  const workflow::Config config = workflow::load_config(authored.path());
  EXPECT_EQ(config.visualization.profile, "far");
  EXPECT_EQ(config.visualization_profile_index, 1U);
  EXPECT_EQ(config.visualization_profile_max_gap_frames, 2U);
}

TEST(MultiProfileConfig, RejectsUnknownVisualizationProfileWhenDisabled) {
  const MutatedConfig authored("profile: near", "profile: missing");
  EXPECT_THROW(workflow::load_config(authored.path()), std::runtime_error);
}

TEST(MultiProfileConfig, ComputesMaximumCyclicVisualizationGap) {
  EXPECT_EQ(workflow::maximum_cyclic_profile_gap({0U, 1U, 1U, 0U, 1U}, 0U), 3U);
  EXPECT_THROW(workflow::maximum_cyclic_profile_gap({0U, 0U}, 1U), std::runtime_error);
}

TEST(MultiProfileConfig, RejectsDuplicateNames) {
  const MutatedConfig config("name: far, carrier_frequency", "name: near, carrier_frequency");
  EXPECT_THROW(workflow::load_config(config.path()), std::runtime_error);
}

TEST(MultiProfileConfig, RejectsUnknownSequenceProfile) {
  const MutatedConfig config("profile_sequence: [near, far]", "profile_sequence: [near, missing]");
  EXPECT_THROW(workflow::load_config(config.path()), std::runtime_error);
}

TEST(MultiProfileConfig, RejectsNonIntegralSamples) {
  const MutatedConfig config("samples_per_chirp: 512}", "samples_per_chirp: 511}");
  EXPECT_THROW(workflow::load_config(config.path()), std::runtime_error);
}

TEST(MultiProfileConfig, RejectsDuplicatedLegacyWaveformAuthoring) {
  const MutatedConfig config("app:\n", "app:\n    waveform: {carrier_frequency: 77.0e9}\n");
  EXPECT_THROW(workflow::load_config(config.path()), std::runtime_error);
}

}  // namespace
