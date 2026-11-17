// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>

namespace holoscan::radar::app {
struct FrameProgress {
  std::atomic<std::int64_t> capture_ns{-1};
  std::atomic<std::int64_t> expected_ns{-1};
  std::atomic<std::int64_t> input_ns{-1};
};
// The authoring thread hands each graph its own progress owner. Workers only
// access captured owners, never this thread-local construction slot.
inline std::shared_ptr<FrameProgress>& authoring_frame_progress() {
  static thread_local auto progress = std::make_shared<FrameProgress>();
  return progress;
}
inline constexpr auto kDisplayDrainTimeout = std::chrono::seconds{10};
}  // namespace holoscan::radar::app
