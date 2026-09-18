// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17): repo builds with C++17
#include <stdexcept>
#include <vector>

#include "config.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/app/multi_profile_source.hpp"

namespace {

namespace app = holoscan::radar::app;
namespace workflow = holoscan::radar::app::mimo_tdm_multi_profile;

holoscan::TimePoint time_at(int64_t timestamp) {
  return holoscan::TimePoint{timestamp, holoscan::ClockId{7}};
}

TEST(TransmitFrameCoordinator, HoldsUntilPublicationAndConsumesOnce) {
  app::TransmitFrameCoordinator coordinator(2U);
  coordinator.register_transmit(app::TransmitReceipt{11U, time_at(100), 1U});

  const auto early = coordinator.claim(11U);
  EXPECT_EQ(early.status,
            app::TransmitFrameCoordinator::ClaimStatus::kNotReady);
  EXPECT_EQ(early.transmit_time, time_at(100));

  coordinator.mark_ready(11U);
  const auto ready = coordinator.claim(11U);
  EXPECT_EQ(ready.status, app::TransmitFrameCoordinator::ClaimStatus::kReady);
  EXPECT_EQ(ready.transmit_time, time_at(100));
  EXPECT_EQ(ready.profile_index, 1U);
  EXPECT_EQ(coordinator.outstanding(), 0U);
  EXPECT_EQ(coordinator.claim(11U).status,
            app::TransmitFrameCoordinator::ClaimStatus::kMissing);
}

TEST(TransmitFrameCoordinator, RejectsDuplicatesOverflowAndDoubleReady) {
  app::TransmitFrameCoordinator coordinator(1U);
  coordinator.register_transmit(app::TransmitReceipt{1U, time_at(1), 0U});
  EXPECT_THROW(coordinator.register_transmit(
                   app::TransmitReceipt{1U, time_at(2), 0U}),
               std::runtime_error);
  EXPECT_THROW(coordinator.register_transmit(
                   app::TransmitReceipt{2U, time_at(2), 0U}),
               std::runtime_error);
  coordinator.mark_ready(1U);
  EXPECT_THROW(coordinator.mark_ready(1U), std::runtime_error);
  coordinator.cancel(1U);
  EXPECT_EQ(coordinator.outstanding(), 0U);
}

TEST(MultiProfileConfig, DecodesProfilesAndMaximumExtents) {
  const auto path = std::filesystem::path(HOLOSCAN_RADAR_SOURCE_DIR) /
                    "applications/fmcw/mimo_tdm_multi_profile/config/"
                    "mimo_tdm_multi_profile_smoke.yaml";
  const workflow::Config config = workflow::load_config(path);
  ASSERT_EQ(config.profiles.size(), 2U);
  EXPECT_EQ(config.profiles[0].name, "near");
  EXPECT_EQ(config.profiles[1].name, "far");
  EXPECT_EQ(config.profile_sequence, (std::vector<std::size_t>{0U, 1U}));
  EXPECT_EQ(config.extents.max_samples_per_chirp, 512);
  EXPECT_EQ(config.extents.max_chirps, 64);
  EXPECT_EQ(config.extents.max_interleaved_chirps, 128);
  EXPECT_DOUBLE_EQ(
      config.profiles[0].waveform.chirp_repetition_interval(), 50.0e-6);
  EXPECT_DOUBLE_EQ(
      config.profiles[1].waveform.chirp_repetition_interval(), 90.0e-6);
  EXPECT_DOUBLE_EQ(
      config.profiles[1].physical_waveform.chirp_repetition_interval,
      45.0e-6);
  EXPECT_EQ(config.range_params->nfft, 0);
  EXPECT_EQ(config.doppler_params->nfft, 0);
  EXPECT_FALSE(config.range_params->waveform.is_valid());
  EXPECT_FALSE(config.doppler_params->waveform.is_valid());
  EXPECT_FALSE(config.compensation_params->waveform.is_valid());
}

}  // namespace
