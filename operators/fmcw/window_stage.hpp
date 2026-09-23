// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/window_spec.hpp"
#include "radar_common/dsp/window.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/tensor/matx_utils.hpp"

namespace holoscan::radar::detail {

// A generated window and the sum of the coefficients after casting to device
// precision. Keeping the sum with the taps lets clutter projection respect
// coherent_gain_normalization without assuming any particular gain.
template <typename T>
struct CachedWindow {
  matx::tensor_t<T, 1> taps;
  double sum;
  double absolute_sum;
};

template <typename T>
const CachedWindow<T>& get_cached_window(const WindowSpec& spec, int64_t length,
                                         ConfigCache<CachedWindow<T>>& cache, const char* what,
                                         cudaStream_t stream) {
  return cache.get_or_build(window_cache_key(spec, length), what, [&]() {
    std::vector<double> host = generate_window(spec, length);
    double sum = 0.0;
    double absolute_sum = 0.0;
    for (const double coefficient : host) {
      const double device_coefficient = static_cast<double>(static_cast<T>(coefficient));
      sum += device_coefficient;
      absolute_sum += std::abs(device_coefficient);
    }
    return CachedWindow<T>{
        .taps = upload_coefficients<T>(host, stream), .sum = sum, .absolute_sum = absolute_sum};
  });
}

// Weighted projection requires a meaningful denominator. In particular,
// custom unnormalized taps may legitimately reach generation with zero sum,
// unlike coherent-gain-normalized taps, which generation already rejects.
template <typename T>
void require_nonzero_window_sum(const WindowSpec& spec, const CachedWindow<T>& window) {
  const double tolerance = 8.0 * static_cast<double>(std::numeric_limits<T>::epsilon()) *
                           static_cast<double>(window.taps.Size(0)) * window.absolute_sum;
  if (!std::isfinite(window.sum) || std::abs(window.sum) <= tolerance) {
    const char* kind = spec.type == WindowType::kCustom ? "custom window" : "window";
    throw std::invalid_argument(std::string("clutter stage: ") + kind +
                                " coefficient sum is numerically zero; weighted mean is undefined");
  }
}

}  // namespace holoscan::radar::detail
