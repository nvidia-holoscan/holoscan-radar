// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/visualization/holoviz_op.hpp>

namespace holoscan::radar::app::viz::detail {

// SDK colors are packed ABGR.
inline constexpr std::uint32_t kDashboardTextPrimary = 0xFFF3EDE6U;
inline constexpr std::uint32_t kDashboardTextSecondary = 0xFFC2AC9BU;
inline constexpr std::uint32_t kDashboardTextMuted = 0xFF9C8471U;

inline constexpr std::size_t kDashboardTextRows = 128U;
inline constexpr std::size_t kDashboardTextBytes = 192U;
static_assert(visualization::kTextRunHeaderBytes == 16U);
inline constexpr auto kDashboardTextRowBytes = 16U + kDashboardTextBytes;

// Serialize the SDK's plain uint8 Tensor format. Each sample owns its text bytes;
// the renderer may consume it on a separate partition after the next radar tick.
inline void append_dashboard_text(std::vector<std::uint8_t>* records, std::int64_t width,
                                  std::int64_t /*height*/, float x, float y, std::string_view text,
                                  float size_px, std::uint32_t color = kDashboardTextPrimary) {
  if (records->size() / kDashboardTextRowBytes >= kDashboardTextRows) {
    throw std::invalid_argument("FMCW dashboard exceeded its text run budget");
  }
  // Keep long workflow/profile names inside the canvas. Native monospace advances
  // approximately 7 px at 13 px height; reserve a small right gutter.
  const std::size_t available = static_cast<std::size_t>(
      std::max(0.0F, (static_cast<float>(width) - x - 12.0F) / (size_px * 7.0F / 13.0F)));
  const std::size_t length = std::min({text.size(), kDashboardTextBytes - 1U, available});
  if (length == 0U) {
    return;
  }
  std::string display(text.substr(0, length));
  if (length < text.size() && length >= 3U) {
    display.replace(length - 3U, 3U, "...");
  }
  const holoscan::visualization::TextRunHeader header{
      .x = x,
      .y = y,
      .size_px = size_px,
      .color = color,
  };
  const std::size_t start = records->size();
  records->resize(start + kDashboardTextRowBytes, 0U);
  std::memcpy(records->data() + start, &header, sizeof(header));
  std::memcpy(records->data() + start + sizeof(header), display.data(), display.size());
}

}  // namespace holoscan::radar::app::viz::detail
