// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace holoscan::radar {

inline constexpr std::size_t kMaxUniformGridPoints = 1'000'000U;

// Expand an endpoint-aligned, inclusive uniform grid. The interval count must
// be integral within 64 * epsilon * max(1, abs(intervals)).
inline std::vector<double> make_uniform_grid(double min, double max, double step,
                                             std::string_view name = "uniform grid",
                                             std::size_t max_points = kMaxUniformGridPoints) {
  const std::string prefix{name};
  if (!std::isfinite(min) || !std::isfinite(max) || !std::isfinite(step)) {
    throw std::invalid_argument(prefix + ": min, max, and step must be finite");
  }
  if (step <= 0.0) {
    throw std::invalid_argument(prefix + ": step must be > 0");
  }
  if (min > max) {
    throw std::invalid_argument(prefix + ": min must be <= max");
  }
  if (max_points == 0U) {
    throw std::invalid_argument(prefix + ": max_points must be > 0");
  }
  if (min == max) {
    return {min};
  }

  const double intervals = (max - min) / step;
  const double rounded = std::round(intervals);
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(intervals));
  if (!std::isfinite(intervals) || std::abs(intervals - rounded) > tolerance) {
    throw std::invalid_argument(prefix + ": (max - min) must be an integer multiple of step");
  }
  if (rounded < 1.0 || rounded >= static_cast<double>(max_points) ||
      rounded >= static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(prefix + ": generated point count exceeds limit");
  }

  const std::size_t count = static_cast<std::size_t>(rounded) + 1U;
  std::vector<double> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    result[index] = min + static_cast<double>(index) * step;
  }
  result.back() = max;
  return result;
}

}  // namespace holoscan::radar
