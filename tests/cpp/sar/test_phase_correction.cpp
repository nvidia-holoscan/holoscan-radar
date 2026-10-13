// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdint>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/phase_correction_op/phase_correction_op.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::MaxTensorExtents;
using holoscan::radar::PhaseCorrectionOp;
using holoscan::radar::PhaseCorrectionParams;
using holoscan::radar::PhaseCorrectionSense;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

void check_phase_sign(PhaseCorrectionSense sense, float expected_imag) {
  using Op = PhaseCorrectionOp<float>;
  MaxTensorExtents extents;
  extents.max_pulses = 2;
  extents.max_range_bins = 3;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.set_params(PhaseCorrectionParams{.sense = sense});
  const std::vector<std::complex<float>> profiles(6U, {1.0F, 0.0F});
  const std::vector<float> correction{0.0F, 1.5707963268F};
  auto input = to_device_tensor(profiles, {2, 3}, complex_dtype<float>());
  auto phase = to_device_tensor(correction, {2}, real_dtype<float>());
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::nanoseconds{1}, holoscan::testing::input("range_profiles", *input),
      holoscan::testing::input("phase_correction", *phase)));
  fixture.capture_snapshot<holoscan::Tensor>("range_profiles_out", 1U, input->nbytes());
  fixture.input_source_device("range_profiles", holoscan::DeviceId{0});
  fixture.input_source_device("phase_correction", holoscan::DeviceId{0});
  fixture.output_device("range_profiles_out", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("range_profiles_out", 1U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto output = to_host(result.snapshot<holoscan::Tensor>("range_profiles_out"));
  ASSERT_EQ(output.shape, (std::vector<int64_t>{2, 3}));
  const auto* values = output.as<std::complex<float>>();
  for (int64_t bin = 0; bin < 3; ++bin) {
    EXPECT_NEAR(values[bin].real(), 1.0F, 1e-5F);
    EXPECT_NEAR(values[bin].imag(), 0.0F, 1e-5F);
    EXPECT_NEAR(values[3 + bin].real(), 0.0F, 1e-5F);
    EXPECT_NEAR(values[3 + bin].imag(), expected_imag, 1e-5F);
  }
}

TEST(PhaseCorrectionOp, SubtractAppliesNegativePulsePhase) {
  check_phase_sign(PhaseCorrectionSense::kSubtract, -1.0F);
}
TEST(PhaseCorrectionOp, AddAppliesPositivePulsePhase) {
  check_phase_sign(PhaseCorrectionSense::kAdd, 1.0F);
}
}  // namespace
