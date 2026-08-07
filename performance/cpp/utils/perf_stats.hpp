// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Latency aggregation for the performance harness: percentiles over a set of
// per-frame samples, plus median-of-medians across repeats. Deliberately
// dependency-free (no Google Benchmark, no boost).

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace holoscan::radar::perf {

struct LatencyStats {
  double min_ms{std::numeric_limits<double>::quiet_NaN()};
  double median_ms{std::numeric_limits<double>::quiet_NaN()};
  double p90_ms{std::numeric_limits<double>::quiet_NaN()};
  double p99_ms{std::numeric_limits<double>::quiet_NaN()};
  double max_ms{std::numeric_limits<double>::quiet_NaN()};
  int64_t samples{0};
};

// Nearest-rank percentile on an already-sorted, non-empty vector. q in [0,1].
inline double percentile_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double rank = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<size_t>(std::floor(rank));
  const auto hi = static_cast<size_t>(std::ceil(rank));
  const double frac = rank - static_cast<double>(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

inline LatencyStats summarize(std::vector<double> samples) {
  LatencyStats s;
  s.samples = static_cast<int64_t>(samples.size());
  if (samples.empty()) {
    return s;
  }
  std::sort(samples.begin(), samples.end());
  s.min_ms = samples.front();
  s.max_ms = samples.back();
  s.median_ms = percentile_sorted(samples, 0.50);
  s.p90_ms = percentile_sorted(samples, 0.90);
  s.p99_ms = percentile_sorted(samples, 0.99);
  return s;
}

}  // namespace holoscan::radar::perf
