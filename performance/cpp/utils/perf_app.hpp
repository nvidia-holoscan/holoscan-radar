// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared timing plumbing for the performance harness. TimingCollector is the
// single object the source and terminal sink write into; the scenario runner
// (in each perf_<operator>.cpp) reads it back after app->run() to build the
// report rows. Kept operator-agnostic so a second consumer can reuse it.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

namespace holoscan::radar::perf {

using Clock = std::chrono::steady_clock;

inline double to_ms(Clock::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

// One collector per scenario run. The data source records a start timestamp
// per emitted frame and the graph-ready instant on its first tick; the
// terminal sink records an end timestamp per completed frame (after a device
// sync). Single-threaded GreedyScheduler guarantees one start and one end per
// frame in emission order, so the vectors align by index.
struct TimingCollector {
  Clock::time_point t_construct{};   // set before make_application
  Clock::time_point t_first_tick{};  // set on the source's first compute
  bool first_tick_set{false};

  std::vector<Clock::time_point> starts;
  std::vector<Clock::time_point> ends;

  // ms from graph construction to the first tick: setup / "ready-to-tick".
  double setup_ms() const { return to_ms(t_first_tick - t_construct); }

  // Per-frame end-to-end latency, aligned by frame index.
  std::vector<double> frame_latencies_ms() const {
    const size_t n = std::min(starts.size(), ends.size());
    std::vector<double> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      out.push_back(to_ms(ends[i] - starts[i]));
    }
    return out;
  }
};

}  // namespace holoscan::radar::perf
