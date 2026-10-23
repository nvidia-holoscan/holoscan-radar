// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_common/sar/formation_hud.hpp"
#include "utils/tensor_fixtures.hpp"
#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_fixture.hpp"  // NOLINT(build/include_subdir)

namespace {
namespace fs = std::filesystem;
using holoscan::radar::cphd::CphdFile;
using holoscan::radar::testing::cphd_fixture::make_fixture;

TEST(CphdReaderOp, UnmeteredWaitsOnFullQueueUnlessPaced) {
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::cphd_reader_waits_on_full;
  CphdReaderParams params;
  EXPECT_TRUE(cphd_reader_waits_on_full(params));
  params.playback_rate = 1.0;
  EXPECT_FALSE(cphd_reader_waits_on_full(params));
  params.wait_for_downstream = true;
  EXPECT_TRUE(cphd_reader_waits_on_full(params));
  params.wait_for_downstream = false;
  params.playback_rate = 0.0;
  EXPECT_TRUE(cphd_reader_waits_on_full(params));
}

TEST(CphdReaderOp, DutyCycleQuietTimeScalesWithPlaybackRate) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::FormationHud;
  using holoscan::radar::MaxTensorExtents;
  const fs::path path = fs::temp_directory_path() / "holoscan_sar_reader_cooldown.cphd";
  make_fixture(path);
  struct Case {
    double rate;
    double duty;
    double quiet_s;
  };
  // Five pulses at 10 ms spacing span 40 ms. A 1/26 duty cycle adds
  // 1000 ms of radar quiet time, scaled along with playback (except rate 0).
  const std::array<Case, 5> cases{{{2.0, 1.0 / 26.0, 0.5},
                                   {0.5, 1.0 / 26.0, 2.0},
                                   {1.0, 1.0 / 26.0, 1.0},
                                   {0.0, 1.0 / 26.0, 1.0},
                                   {2.0, 1.0, 0.0}}};
  for (const auto& sample : cases) {
    SCOPED_TRACE(sample.rate);
    SCOPED_TRACE(sample.duty);
    MaxTensorExtents extents;
    extents.max_pulses = 5;
    extents.max_samples_per_pulse = 8;
    auto hud = std::make_shared<FormationHud>();
    auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents, hud);
    CphdReaderParams params;
    params.path = holoscan::radar::CollectionPaths(path.string());
    params.coordinate_frame =
        CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
    params.pulses_per_tick = 5;
    params.loop = true;
    params.playback_rate = sample.rate;
    params.duty_cycle = sample.duty;
    fixture.set_params(params);
    fixture.capture_snapshot<holoscan::Tensor>("phase_history", 2U,
                                               5U * 8U * sizeof(std::complex<double>));
    fixture.capture_snapshot<holoscan::Tensor>("positions", 2U, 5U * 3U * sizeof(double));
    fixture.capture_snapshot<holoscan::Tensor>("reference_position", 2U, 5U * 3U * sizeof(double));
    fixture.capture<uint8_t>("stream_boundary", 2U);
    for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                             "pulse_times", "velocities"}) {
      fixture.output_device(port, holoscan::DeviceId{0});
    }
    auto running = fixture.run_async({.expect_stop = true});
    const bool observed = running.wait_for_output("stream_boundary", 2U, std::chrono::seconds{5});
    running.request_stop();
    const auto result = running.finish();
    ASSERT_TRUE(observed) << result.describe_failure();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    const auto snapshot = hud->snapshot();
    ASSERT_GE(snapshot.spans.size(), 2U);
    const auto& first = snapshot.spans.front();
    if (sample.quiet_s == 0.0) {
      EXPECT_FALSE(first.quiet_start.has_value());
      EXPECT_FALSE(first.quiet_end.has_value());
    } else {
      ASSERT_TRUE(first.quiet_start.has_value());
      ASSERT_TRUE(first.quiet_end.has_value());
      const double actual =
          std::chrono::duration<double>(*first.quiet_end - *first.quiet_start).count();
      // Observe the actual sleep interval, with slack for host scheduling jitter.
      EXPECT_NEAR(actual, sample.quiet_s, 0.15);
    }
  }
  fs::remove(path);
}

TEST(CphdFile, CppFixtureDecodesHeaderPvpAndSignal) {
  const fs::path path = fs::temp_directory_path() / "holoscan_sar_cphd_cpp_fixture.cphd";
  make_fixture(path);
  const CphdFile file(path.string());
  ASSERT_EQ(file.num_channels(), 1);
  EXPECT_EQ(file.channel(0).num_vectors, 5);
  EXPECT_EQ(file.channel(0).num_samples, 8);
  EXPECT_EQ(file.sample_format(), holoscan::radar::cphd::SampleFormat::kCF8);
  const auto pvp = file.read_pvp_block(0, 1, 3);
  const auto pos = pvp.field("TxPos", 3);
  ASSERT_EQ(pos.size(), 9U);
  EXPECT_DOUBLE_EQ(pos[1], 1.0);
  EXPECT_DOUBLE_EQ(pos[7], 3.0);
  const auto sc0 = pvp.field("SC0", 1);
  ASSERT_EQ(sc0.size(), 3U);
  EXPECT_DOUBLE_EQ(sc0[0], 9.0e9);
  const auto signal = file.read_signal(0, 1, 3);
  ASSERT_EQ(signal.size(), 24U);
  EXPECT_NEAR(signal[0].real(), std::cos(0.1), 1e-6);
  EXPECT_NEAR(signal[0].imag(), std::sin(0.1), 1e-6);
  EXPECT_NEAR(signal[23].real(), std::cos(1.7), 1e-6);
  fs::remove(path);
}
TEST(CphdReaderOp, AcceptsRecordedGridRampButRejectsFrequencyStep) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::MaxTensorExtents;
  const fs::path ramp = fs::temp_directory_path() / "holoscan_sar_reader_ramp.cphd";
  const fs::path step = fs::temp_directory_path() / "holoscan_sar_reader_step.cphd";
  make_fixture(ramp, 5.0, 0.03);
  make_fixture(step, 110000.0);
  const auto run_reader = [](const fs::path& path) {
    MaxTensorExtents extents;
    extents.max_pulses = 5;
    extents.max_samples_per_pulse = 8;
    auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents);
    CphdReaderParams params;
    params.path = holoscan::radar::CollectionPaths(path.string());
    params.coordinate_frame =
        CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
    params.pulses_per_tick = 5;
    fixture.set_params(params);
    fixture.capture_snapshot<holoscan::Tensor>("phase_history", 1U,
                                               5U * 8U * sizeof(std::complex<double>));
    fixture.capture_snapshot<holoscan::Tensor>("positions", 1U, 5U * 3U * sizeof(double));
    fixture.capture_snapshot<holoscan::Tensor>("reference_position", 1U, 5U * 3U * sizeof(double));
    fixture.capture<uint8_t>("stream_boundary", 1U);
    for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                             "pulse_times", "velocities"}) {
      fixture.output_device(port, holoscan::DeviceId{0});
    }
    // The reader parks after D22 so a downstream aperture can drain; HSDK
    // request_stop does not. Isolated operator tests must cancel, matching
    // EmitsEachBatchAndTypedEndBoundary.
    auto running = fixture.run_async({.expect_stop = true});
    (void)running.wait_for_output("stream_boundary", 1U, std::chrono::seconds{5});
    running.request_stop();
    return running.finish();
  };
  const auto accepted = run_reader(ramp);
  ASSERT_TRUE(accepted.ok()) << accepted.describe_failure();
  const auto boundaries = accepted.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 1U);
  EXPECT_EQ(boundaries[0].data, 2U);
  const auto rejected = run_reader(step);
  EXPECT_FALSE(rejected.ok());
  EXPECT_NE(rejected.describe_failure().find("SC0 varies across a pulse batch"), std::string::npos);
  fs::remove(ramp);
  fs::remove(step);
}

TEST(CphdReaderOp, EmitsEachBatchAndTypedEndBoundary) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::MaxTensorExtents;
  const fs::path path = fs::temp_directory_path() / "holoscan_sar_reader_native.cphd";
  make_fixture(path);
  MaxTensorExtents extents;
  extents.max_pulses = 2;
  extents.max_samples_per_pulse = 8;
  auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<float>>(extents);
  CphdReaderParams params;
  params.path = holoscan::radar::CollectionPaths(path.string());
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
  params.pulses_per_tick = 2;
  fixture.set_params(params);
  fixture.capture_snapshot<holoscan::Tensor>("phase_history", 3U,
                                             2U * 8U * sizeof(std::complex<float>));
  fixture.capture_snapshot<holoscan::Tensor>("positions", 3U, 2U * 3U * sizeof(float));
  fixture.capture_snapshot<holoscan::Tensor>("reference_position", 3U, 2U * 3U * sizeof(float));
  fixture.capture<uint8_t>("stream_boundary", 3U);
  for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                           "pulse_times", "velocities"}) {
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("stream_boundary", 3U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto boundaries = result.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 3U);
  EXPECT_EQ(boundaries[0].data, 0U);
  EXPECT_EQ(boundaries[1].data, 0U);
  EXPECT_EQ(boundaries[2].data, 2U);
  const auto outputs = result.snapshots<holoscan::Tensor>("phase_history");
  ASSERT_EQ(outputs.size(), 3U);
  EXPECT_EQ(outputs[0].shape(), (std::vector<int64_t>{2, 8}));
  EXPECT_EQ(outputs[2].shape(), (std::vector<int64_t>{1, 8}));
  const auto expected_signal = CphdFile(path.string()).read_signal(0, 0, 2);
  const auto host = holoscan::radar::testing::to_host(outputs[0]);
  ASSERT_EQ(host.size(), static_cast<int64_t>(expected_signal.size()));
  const auto* samples = host.as<std::complex<float>>();
  for (std::size_t i = 0; i < expected_signal.size(); ++i) {
    EXPECT_NEAR(samples[i].real(), expected_signal[i].real(), 1e-5F) << i;
    EXPECT_NEAR(samples[i].imag(), expected_signal[i].imag(), 1e-5F) << i;
  }
  fs::remove(path);
}
TEST(CphdReaderOp, MarksFinalBatchOfEachCollection) {
  using holoscan::radar::CPHDReaderOp;
  using holoscan::radar::CphdReaderParams;
  using holoscan::radar::CoordinateFrame;
  using holoscan::radar::CoordinateFrameSpec;
  using holoscan::radar::CoordinateFrameType;
  using holoscan::radar::MaxTensorExtents;
  const fs::path first = fs::temp_directory_path() / "holoscan_sar_reader_first.cphd";
  const fs::path second = fs::temp_directory_path() / "holoscan_sar_reader_second.cphd";
  make_fixture(first);
  make_fixture(second);
  MaxTensorExtents extents;
  extents.max_pulses = 2;
  extents.max_samples_per_pulse = 8;
  auto fixture = holoscan::testing::create_operator_test<CPHDReaderOp<double>>(extents);
  CphdReaderParams params;
  params.path =
      holoscan::radar::CollectionPaths(std::vector<std::string>{first.string(), second.string()});
  params.coordinate_frame =
      CoordinateFrame(CoordinateFrameSpec{.frame = CoordinateFrameType::kEcef});
  params.pulses_per_tick = 2;
  fixture.set_params(params);
  fixture.capture_snapshot<holoscan::Tensor>("phase_history", 6U,
                                             2U * 8U * sizeof(std::complex<double>));
  fixture.capture_snapshot<holoscan::Tensor>("positions", 6U, 2U * 3U * sizeof(double));
  fixture.capture_snapshot<holoscan::Tensor>("reference_position", 6U, 2U * 3U * sizeof(double));
  fixture.capture<uint8_t>("stream_boundary", 6U);
  for (const char* port : {"phase_history", "positions", "reference_position", "sample_axis",
                           "pulse_times", "velocities"}) {
    fixture.output_device(port, holoscan::DeviceId{0});
  }
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("stream_boundary", 6U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto boundaries = result.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 6U);
  const std::vector<uint8_t> expected{0U, 0U, 1U, 0U, 0U, 2U};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(boundaries[i].data, expected[i]) << "batch " << i;
  const auto outputs = result.snapshots<holoscan::Tensor>("phase_history");
  ASSERT_EQ(outputs.size(), 6U);
  const auto expected_signal = CphdFile(first.string()).read_signal(0, 0, 2);
  const auto host = holoscan::radar::testing::to_host(outputs[0]);
  ASSERT_EQ(host.size(), static_cast<int64_t>(expected_signal.size()));
  const auto* samples = host.as<std::complex<double>>();
  for (std::size_t i = 0; i < expected_signal.size(); ++i) {
    EXPECT_NEAR(samples[i].real(), expected_signal[i].real(), 1e-5) << i;
    EXPECT_NEAR(samples[i].imag(), expected_signal[i].imag(), 1e-5) << i;
  }
  fs::remove(first);
  fs::remove(second);
}

}  // namespace
