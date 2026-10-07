// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <thread>

#include <holoscan/core/execution_context.hpp>

namespace holoscan::radar {

/// Sleep up to \p seconds, returning early when the run requests cooperative stop.
inline void cooperative_sleep(const ExecutionContext& context, double seconds) {
  if (!(seconds > 0.0))
    return;
  using clock = std::chrono::steady_clock;
  const auto deadline = clock::now() + std::chrono::duration<double>(seconds);
  constexpr auto kSlice = std::chrono::milliseconds(50);
  while (clock::now() < deadline) {
    if (context.cancellation().stop_requested())
      return;
    const auto remaining = deadline - clock::now();
    std::this_thread::sleep_for(remaining < kSlice ? remaining : kSlice);
  }
}

}  // namespace holoscan::radar
