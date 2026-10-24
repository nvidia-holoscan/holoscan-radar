// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>

namespace holoscan::radar::app::sar {

inline bool is_backpressure_failure(const Diagnostic& diagnostic) {
  if (diagnostic.severity != DiagnosticSeverity::kError || !diagnostic.detail)
    return false;
  const auto is_pressure = [](ErrorCode code) {
    return code == ErrorCode::kBackpressured || code == ErrorCode::kResourceExhausted;
  };
  const auto& detail = *diagnostic.detail;
  return is_pressure(detail.error.code) ||
         std::any_of(detail.causes.begin(), detail.causes.end(), [&](const Error& cause) {
           return is_pressure(cause.code);
         });
}

}  // namespace holoscan::radar::app::sar
