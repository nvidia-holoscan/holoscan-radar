// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

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

inline bool is_vulkan_failure(const Diagnostic& diagnostic) {
  if (diagnostic.severity != DiagnosticSeverity::kError || !diagnostic.detail)
    return false;
  // HoloViz's Vulkan category becomes Core kFailure; only its explicit backend
  // markers survive translation. Do not treat every generic failure as Vulkan.
  const auto is_vulkan = [](const Error& error) {
    return error.message.find("HoloViz kVulkanError") != std::string::npos ||
           error.message.find("[holoviz] Vulkan ") != std::string::npos;
  };
  const auto& detail = *diagnostic.detail;
  return is_vulkan(detail.error) ||
         std::any_of(detail.causes.begin(), detail.causes.end(), is_vulkan);
}

inline std::string format_failure_banner(std::string_view title, std::string_view body,
                                         bool color) {
  constexpr std::size_t kWidth = 80;
  std::string heading = " " + std::string(title) + " ";
  if (heading.size() < kWidth) {
    const auto left = (kWidth - heading.size()) / 2U;
    heading = std::string(left, '=') + heading + std::string(kWidth - heading.size() - left, '=');
  }
  std::string banner = "\n\n";
  if (color)
    banner += "\033[1;31m";
  banner += heading;
  if (color)
    banner += "\033[0;37m";
  banner += "\n";
  banner += body;
  if (banner.back() != '\n')
    banner += "\n";
  if (color)
    banner += "\033[1;31m";
  banner += std::string(kWidth, '=');
  if (color)
    banner += "\033[0m";
  banner += "\n";
  return banner;
}

}  // namespace holoscan::radar::app::sar
