// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include <holoscan/testing/operator_test_harness.hpp>

namespace holoscan::radar::testing {

// Correctness observation budget, independent of performance acceptance thresholds.
inline std::chrono::milliseconds fixture_output_timeout() {
  constexpr const char* kVariable = "HOLOSCAN_RADAR_TEST_TIMEOUT_SECONDS";
  const char* value = std::getenv(kVariable);
  if (value == nullptr)
    return std::chrono::seconds{30};
  const std::string_view text(value);
  int seconds = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), seconds);
  if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || seconds <= 0) {
    throw std::invalid_argument(std::string(kVariable) + " must be a positive integer");
  }
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds{seconds});
}

// Observation policy only: native fixtures still own staging, execution, stop and cleanup.
// Call check() after the SDK's finish() and before inspecting captured payloads.
class FixtureCompletion {
 public:
  explicit FixtureCompletion(std::chrono::milliseconds timeout = fixture_output_timeout())
  : timeout_(timeout) {
    if (timeout <= std::chrono::milliseconds::zero() ||
        timeout > std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::duration::max()) /
                      4) {
      throw std::invalid_argument("fixture output timeout must be positive and representable");
    }
  }

  holoscan::testing::TestRunOptions run_options(
      holoscan::testing::FrameRelease frames = holoscan::testing::FrameRelease::kAutomatic) const {
    return {.timeout = timeout_ * 2, .expect_stop = true, .frames = frames};
  }

  // All output prefixes and manual releases share one deadline, rather than one per port.
  std::chrono::milliseconds remaining() {
    const auto now = std::chrono::steady_clock::now();
    if (!deadline_)
      deadline_ = now + timeout_;
    if (now >= *deadline_)
      return std::chrono::milliseconds::zero();
    return std::chrono::ceil<std::chrono::milliseconds>(*deadline_ - now);
  }

  bool wait_for_output(const holoscan::testing::RunningFixture& running, std::string_view port,
                       std::size_t minimum) {
    const bool observed = running.wait_for_output(port, minimum, remaining());
    if (!observed && missing_.empty()) {
      missing_ = std::string(port) + " (required prefix " + std::to_string(minimum) + ")";
    }
    return observed;
  }

  void check(const holoscan::testing::TestResult& result) const {
    // Expected operator/compile errors retain native diagnostics and caller assertions.
    if (result.ok() && !missing_.empty()) {
      throw std::runtime_error("test fixture did not observe " + missing_ + " within " +
                               std::to_string(timeout_.count()) +
                               " ms before stop; configure HOLOSCAN_RADAR_TEST_TIMEOUT_SECONDS "
                               "for this correctness-test deployment");
    }
  }

 private:
  std::chrono::milliseconds timeout_;
  std::optional<std::chrono::steady_clock::time_point> deadline_;
  std::string missing_;
};

}  // namespace holoscan::radar::testing
