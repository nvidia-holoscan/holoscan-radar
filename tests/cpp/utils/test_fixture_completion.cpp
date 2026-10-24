// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <complex>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/fixture_completion.hpp"
#include "utils/tensor_fixtures.hpp"
#include "utils/test_extents.hpp"

namespace {

namespace testing = holoscan::radar::testing;
using Op = holoscan::radar::ClutterFilterOp<float>;
constexpr const char* kVariable = "HOLOSCAN_RADAR_TEST_TIMEOUT_SECONDS";

class TimeoutEnvironment {
 public:
  TimeoutEnvironment() {
    if (const char* value = std::getenv(kVariable))
      saved_ = value;
  }
  ~TimeoutEnvironment() {
    if (saved_)
      ::setenv(kVariable, saved_->c_str(), 1);
    else
      ::unsetenv(kVariable);
  }

 private:
  std::optional<std::string> saved_;
};

void stage_fixture(holoscan::testing::OperatorTestHarness<Op>& fixture, bool invalid = false) {
  fixture.set_params(
      holoscan::radar::ClutterFilterParams{.method = holoscan::radar::ClutterFilterMethod::kNone});
  auto input = invalid
                   ? testing::to_device_tensor(std::vector<float>(8U, 1.0F), {4, 2},
                                               holoscan::radar::real_dtype<float>())
                   : testing::to_device_tensor(std::vector<std::complex<float>>(8U, {1.0F, 0.0F}),
                                               {4, 2}, holoscan::radar::complex_dtype<float>());
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{1}, holoscan::testing::input("range_profiles", *input)));
  fixture.capture_snapshot<holoscan::Tensor>("range_profiles_out", 1U, 1024U);
  fixture.input_source_device("range_profiles", holoscan::DeviceId{0});
  fixture.output_device("range_profiles_out", holoscan::DeviceId{0});
}

TEST(FixtureCompletion, ConfigurableBudgetAndManualReleasePolicy) {
  TimeoutEnvironment restore;
  ::unsetenv(kVariable);
  EXPECT_EQ(testing::fixture_output_timeout(), std::chrono::seconds{30});
  ::setenv(kVariable, "75", 1);
  testing::FixtureCompletion completion;
  const auto options = completion.run_options(holoscan::testing::FrameRelease::kManual);
  EXPECT_EQ(options.timeout, std::chrono::seconds{150});
  EXPECT_TRUE(options.expect_stop);
  EXPECT_EQ(options.frames, holoscan::testing::FrameRelease::kManual);
}

TEST(FixtureCompletion, RejectsInvalidConfigurationBeforeRunning) {
  TimeoutEnvironment restore;
  for (const char* value : {"", "0", "-1", "2s", " 3", "999999999999999999999"}) {
    SCOPED_TRACE(value);
    ::setenv(kVariable, value, 1);
    EXPECT_THROW((void)testing::FixtureCompletion{}, std::invalid_argument);
  }
}

TEST(FixtureCompletion, ObservationBudgetDoesNotRestartAfterExpiry) {
  testing::FixtureCompletion completion(std::chrono::milliseconds{10});
  EXPECT_GT(completion.remaining().count(), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  EXPECT_EQ(completion.remaining(), std::chrono::milliseconds::zero());
  EXPECT_EQ(completion.remaining(), std::chrono::milliseconds::zero());
}

TEST(FixtureCompletion, MissingCaptureCannotBecomeSuccessfulRequestedStop) {
  auto fixture = holoscan::testing::create_operator_test<Op>(testing::default_test_extents());
  stage_fixture(fixture);
  testing::FixtureCompletion completion(std::chrono::milliseconds{10});
  auto options = completion.run_options(holoscan::testing::FrameRelease::kManual);
  // Isolate the caller observation timeout from the SDK watchdog in this regression.
  options.timeout = std::chrono::seconds{60};
  auto running = fixture.run_async(options);
  EXPECT_FALSE(completion.wait_for_output(running, "range_profiles_out", 1U));
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_THROW(completion.check(result), std::runtime_error);
}

TEST(FixtureCompletion, PreservesExpectedOperatorErrorDiagnostics) {
  auto fixture = holoscan::testing::create_operator_test<Op>(testing::default_test_extents());
  stage_fixture(fixture, true);
  testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  EXPECT_FALSE(completion.wait_for_output(running, "range_profiles_out", 1U));
  running.request_stop();
  const auto result = running.finish();
  ASSERT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
  EXPECT_NO_THROW(completion.check(result));
}

TEST(FixtureCompletion, AcceptsCapturedPrefixAfterManualRelease) {
  auto fixture = holoscan::testing::create_operator_test<Op>(testing::default_test_extents());
  stage_fixture(fixture);
  testing::FixtureCompletion completion;
  auto running =
      fixture.run_async(completion.run_options(holoscan::testing::FrameRelease::kManual));
  const auto released = running.release_frame(completion.remaining());
  const bool observed = completion.wait_for_output(running, "range_profiles_out", 1U);
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(released) << released.error().message;
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_NO_THROW(completion.check(result));
  EXPECT_EQ(result.snapshots<holoscan::Tensor>("range_profiles_out").size(), 1U);
}

}  // namespace
