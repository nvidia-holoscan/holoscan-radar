// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <initializer_list>

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

}  // namespace
