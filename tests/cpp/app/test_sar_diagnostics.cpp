// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <initializer_list>
#include <string>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>

#include "radar_app/sar/diagnostics.hpp"

namespace {

TEST(SarDiagnostics, BareQueueAndPoolErrorsTriggerBanner) {
  for (const auto code :
       {holoscan::ErrorCode::kBackpressured, holoscan::ErrorCode::kResourceExhausted}) {
    holoscan::Diagnostic diagnostic;
    diagnostic.detail.emplace();
    diagnostic.detail->operator_path = "ecef_to_enu";
    diagnostic.detail->error = holoscan::Error{code};
    EXPECT_TRUE(holoscan::radar::app::sar::is_backpressure_failure(diagnostic));
    // Error wording does not affect classification.
    diagnostic.detail->error = holoscan::Error{code, "arbitrary SDK or operator message"};
    EXPECT_TRUE(holoscan::radar::app::sar::is_backpressure_failure(diagnostic));
  }
}

TEST(SarDiagnostics, WrappedBackpressureTriggersBanner) {
  holoscan::Diagnostic diagnostic;
  diagnostic.detail.emplace();
  diagnostic.detail->error = holoscan::Error{holoscan::ErrorCode::kFailure};
  diagnostic.detail->causes.push_back(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
  for (const auto code :
       {holoscan::ErrorCode::kBackpressured, holoscan::ErrorCode::kResourceExhausted}) {
    diagnostic.detail->causes.push_back(holoscan::Error{code});
    EXPECT_TRUE(holoscan::radar::app::sar::is_backpressure_failure(diagnostic));
    diagnostic.detail->causes.pop_back();
  }
}

TEST(SarDiagnostics, UnrelatedFailuresWarningsAndMissingDetailsDoNotTriggerBanner) {
  holoscan::Diagnostic diagnostic;
  EXPECT_FALSE(holoscan::radar::app::sar::is_backpressure_failure(diagnostic));
  diagnostic.detail.emplace();
  diagnostic.detail->error =
      holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "output queue is full"};
  EXPECT_FALSE(holoscan::radar::app::sar::is_backpressure_failure(diagnostic));
  diagnostic.detail->error = holoscan::Error{holoscan::ErrorCode::kBackpressured};
  diagnostic.severity = holoscan::DiagnosticSeverity::kWarning;
  EXPECT_FALSE(holoscan::radar::app::sar::is_backpressure_failure(diagnostic));
}

TEST(SarDiagnostics, VulkanInitializationFailureTriggersVisualizationBanner) {
  holoscan::Diagnostic diagnostic;
  diagnostic.detail.emplace();
  diagnostic.detail->operator_path = "holoviz";
  diagnostic.detail->operation = holoscan::DiagnosticOperation::kLifecycle;
  diagnostic.detail->error = holoscan::Error{
      holoscan::ErrorCode::kFailure,
      "create_device_for_surface failed (HoloViz kVulkanError): [holoviz] Vulkan error at "
      "vk_device.cpp:507 - VK_ERROR_INITIALIZATION_FAILED (-3)"};
  EXPECT_TRUE(holoscan::radar::app::sar::is_vulkan_failure(diagnostic));
  EXPECT_FALSE(holoscan::radar::app::sar::is_backpressure_failure(diagnostic));
}

TEST(SarDiagnostics, WrappedVulkanErrorsRetainTheirBackendMarker) {
  holoscan::Diagnostic diagnostic;
  diagnostic.detail.emplace();
  diagnostic.detail->causes.push_back(holoscan::Error{
      holoscan::ErrorCode::kFailure, "[holoviz] Vulkan error: VK_ERROR_DEVICE_LOST"});
  EXPECT_TRUE(holoscan::radar::app::sar::is_vulkan_failure(diagnostic));
  diagnostic.detail->causes.back().message =
      "[holoviz] Vulkan instance does not support the required extension: VK_KHR_surface";
  EXPECT_TRUE(holoscan::radar::app::sar::is_vulkan_failure(diagnostic));
}

TEST(SarDiagnostics, OtherBackendsWarningsAndMissingDetailsAreNotVulkanFailures) {
  holoscan::Diagnostic diagnostic;
  EXPECT_FALSE(holoscan::radar::app::sar::is_vulkan_failure(diagnostic));
  diagnostic.detail.emplace();
  diagnostic.detail->error = holoscan::Error{
      holoscan::ErrorCode::kFailure, "create_stream failed (HoloViz kCudaError): CUDA failure"};
  EXPECT_FALSE(holoscan::radar::app::sar::is_vulkan_failure(diagnostic));
  diagnostic.detail->error.message = "arbitrary generic failure";
  EXPECT_FALSE(holoscan::radar::app::sar::is_vulkan_failure(diagnostic));
  diagnostic.detail->error.message = "create_device_for_surface failed (HoloViz kVulkanError)";
  diagnostic.severity = holoscan::DiagnosticSeverity::kWarning;
  EXPECT_FALSE(holoscan::radar::app::sar::is_vulkan_failure(diagnostic));
}

TEST(SarDiagnostics, FailureBannersKeepEightyColumnBordersAndScopedColors) {
  namespace sar = holoscan::radar::app::sar;
  const auto plain = sar::format_failure_banner("VULKAN: VISUALIZATION FAILED", "details", false);
  EXPECT_EQ(plain.find('\n', 2U), 82U);
  EXPECT_EQ(plain.substr(plain.size() - 81U), std::string(80U, '=') + "\n");
  EXPECT_EQ(plain.find('\033'), std::string::npos);
  EXPECT_NE(plain.find("\ndetails\n"), std::string::npos);
  const auto colored = sar::format_failure_banner("VULKAN: VISUALIZATION FAILED", "details", true);
  EXPECT_NE(colored.find("\033[0;37m\ndetails\n\033[1;31m"), std::string::npos);
  EXPECT_EQ(colored.substr(colored.size() - 5U), "\033[0m\n");
  const auto backpressure = sar::format_failure_banner("BACKPRESSURE: RUN STOPPED", "", false);
  EXPECT_EQ(backpressure.substr(2U, 80U),
            "========================== BACKPRESSURE: RUN STOPPED ===========================");
}

}  // namespace
