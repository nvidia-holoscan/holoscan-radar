// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "io/cphd_reader_op/cphd_reader_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"
#include "io/cphd_reader_op/cphd_file.hpp"
#include "io/cphd_fixture.hpp"  // NOLINT(build/include_subdir)

namespace {
namespace fs = std::filesystem;
using holoscan::radar::cphd::CphdFile;
using holoscan::radar::testing::cphd_fixture::make_fixture;

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
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    auto fixture_result = running.finish();
    completion.check(fixture_result);
    return fixture_result;
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
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "stream_boundary", 3U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
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
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "stream_boundary", 6U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto boundaries = result.samples<uint8_t>("stream_boundary");
  ASSERT_EQ(boundaries.size(), 6U);
  const std::vector<uint8_t> expected{0U, 0U, 1U, 0U, 0U, 2U};
  for (std::size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(boundaries[i].data, expected[i]) << "batch " << i;
  fs::remove(first);
  fs::remove(second);
}

}  // namespace
