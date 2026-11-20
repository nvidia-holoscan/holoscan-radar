// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

namespace holoscan::radar::app::viz {

// Resize feedback is application diagnostics, never a data-plane descriptor.
// The immutable bounds size pools; one atomic snapshot sizes the visible panel.
class FrameViewport {
 public:
  FrameViewport(int width, int height, int max_width, int max_height)
  : max_width(max_width), max_height(max_height) {
    set(width, height);
  }

  void set(int width, int height) {
    const auto w = static_cast<std::uint32_t>(std::clamp(width, 1, max_width));
    const auto h = static_cast<std::uint32_t>(std::clamp(height, 1, max_height));
    extent_.store((static_cast<std::uint64_t>(w) << 32U) | h);
  }
  [[nodiscard]] std::pair<int, int> get() const {
    const auto extent = extent_.load();
    return {static_cast<int>(extent >> 32U), static_cast<int>(extent & 0xffffffffU)};
  }

  void drawn(std::int64_t stamp, int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    history_[next_] = Entry{stamp, width, height};
    next_ = (next_ + 1) % history_.size();
  }
  [[nodiscard]] std::optional<std::pair<int, int>> drawn(std::int64_t stamp) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t i = 0; i < history_.size(); ++i) {
      const auto& entry = history_[(next_ + history_.size() - 1 - i) % history_.size()];
      if (entry && entry->stamp == stamp)
        return std::pair{entry->width, entry->height};
    }
    return std::nullopt;
  }

  const int max_width, max_height;

 private:
  std::atomic<std::uint64_t> extent_{};
  struct Entry {
    std::int64_t stamp;
    int width, height;
  };
  mutable std::mutex mutex_;
  std::array<std::optional<Entry>, 64> history_{};
  std::size_t next_{};
};

}  // namespace holoscan::radar::app::viz
