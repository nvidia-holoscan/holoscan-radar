// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "radar_common/sim/sar_phase_history_sim.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::MaxTensorExtents;
using holoscan::radar::SARRangeCompressionOp;
using holoscan::radar::complex_dtype;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

TEST(SARRangeCompressionOp, ConstantSpectrumHasCentredUnitImpulse) {
  for (const std::uint64_t budget : {0ULL, 768ULL, 2147483648ULL}) {
    SCOPED_TRACE(budget);
    using Op = SARRangeCompressionOp<float>;
    MaxTensorExtents extents;
    extents.max_pulses = 1;
    extents.max_samples_per_pulse = 8;
    extents.max_range_bins = 8;
    auto fixture = holoscan::testing::create_operator_test<Op>(extents);
    holoscan::radar::SARRangeCompressionParams params;
    params.cache_budget_bytes = budget;
    params.range_window.type = holoscan::radar::WindowType::kCustom;
    params.range_window.coefficients.assign(8U, 1.0);
    fixture.set_params(params);
    const std::vector<std::complex<float>> spectrum(8U, {1.0F, 0.0F});
    auto input = to_device_tensor(spectrum, {1, 8}, complex_dtype<float>());
    fixture.push(&Op::phase_history, *input);
    fixture.capture_snapshot<holoscan::Tensor>("range_profiles", 1U, input->nbytes());
    fixture.input_source_device("phase_history", holoscan::DeviceId{0});
    fixture.output_device("range_profiles", holoscan::DeviceId{0});
    fixture.output_device("range_bins", holoscan::DeviceId{0});
    holoscan::radar::testing::FixtureCompletion completion;
    auto running = fixture.run_async(completion.run_options());
    const bool observed = completion.wait_for_output(running, "range_profiles", 1U);
    running.request_stop();
    const auto result = running.finish();
    completion.check(result);
    ASSERT_TRUE(observed) << result.describe_failure();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    const auto output = to_host(result.snapshot<holoscan::Tensor>("range_profiles"));
    ASSERT_EQ(output.shape, (std::vector<int64_t>{1, 8}));
    const auto* values = output.as<std::complex<float>>();
    for (int64_t bin = 0; bin < 8; ++bin) {
      EXPECT_NEAR(values[bin].real(), bin == 4 ? 1.0F : 0.0F, 1e-5F);
      EXPECT_NEAR(values[bin].imag(), 0.0F, 1e-5F);
    }
  }
}

TEST(SARRangeCompressionOp, AnalyticOffCentrePointPreservesRangeAndPhase) {
  using Op = SARRangeCompressionOp<double>;
  namespace sim = holoscan::radar::sim;
  sim::SarPhaseHistoryConfig config;
  config.num_pulses = 1;
  config.num_samples = 16;
  config.bandwidth = 15.0e6;
  const double spacing = sim::sar_range_bin_spacing(config, config.num_samples);
  const double delta_range = 3.0 * spacing;
  const std::complex<double> amplitude{0.6, -0.8};
  const auto signal = sim::sar_phase_history(
      config, {sim::SarPointTarget{.differential_range = delta_range, .amplitude = amplitude}});
  MaxTensorExtents extents;
  extents.max_pulses = 1;
  extents.max_samples_per_pulse = config.num_samples;
  extents.max_range_bins = config.num_samples;
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  holoscan::radar::SARRangeCompressionParams params;
  params.cache_budget_bytes = 0;
  fixture.set_params(params);
  auto input = to_device_tensor(signal, {1, config.num_samples}, complex_dtype<double>());
  fixture.push(&Op::phase_history, *input);
  fixture.capture_snapshot<holoscan::Tensor>("range_profiles", 1U, input->nbytes());
  fixture.input_source_device("phase_history", holoscan::DeviceId{0});
  fixture.output_device("range_profiles", holoscan::DeviceId{0});
  fixture.output_device("range_bins", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "range_profiles", 1U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto output = to_host(result.snapshot<holoscan::Tensor>("range_profiles"));
  const auto* values = output.as<std::complex<double>>();
  const int64_t peak = config.num_samples / 2 + 3;
  const auto axis = sim::sar_frequency_axis(config);
  const double phase = -4.0 * holoscan::radar::kPi * axis[config.num_samples / 2] * delta_range /
                       config.propagation_speed;
  const std::complex<double> expected =
      amplitude * std::complex<double>{std::cos(phase), std::sin(phase)};
  for (int64_t bin = 0; bin < config.num_samples; ++bin) {
    const std::complex<double> want = bin == peak ? expected : std::complex<double>{0.0, 0.0};
    EXPECT_NEAR(std::abs(values[bin] - want), 0.0, 1e-6) << "range bin " << bin;
  }
}

}  // namespace
