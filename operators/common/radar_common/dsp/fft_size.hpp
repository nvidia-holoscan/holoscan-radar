// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace holoscan::radar {

// Smallest 5-smooth integer >= n (2^a * 3^b * 5^c). Matches
// numpy.fft.next_fast_len and the range-compression OPEN_ITEMS
// example (4099 -> 4320). cuFFT's cheap path is a radix
// decomposition on these factors (7 is also cheap in some
// versions; this helper does not require it). Throws when n < 1.
inline int64_t next_fast_fft_size(int64_t n) {
  if (n < 1) {
    throw std::invalid_argument("next_fast_fft_size: n must be >= 1");
  }
  const int64_t kMax = std::numeric_limits<int64_t>::max();
  int64_t best = kMax;
  for (int64_t p2 = 1; p2 <= best;) {
    for (int64_t p23 = p2; p23 <= best;) {
      int64_t p235 = p23;
      while (p235 < n) {
        if (p235 > kMax / 5) {
          p235 = kMax;
          break;
        }
        p235 *= 5;
      }
      if (p235 >= n && p235 < best) {
        best = p235;
      }
      if (p23 > kMax / 3) {
        break;
      }
      p23 *= 3;
    }
    if (p2 > kMax / 2) {
      break;
    }
    p2 *= 2;
  }
  return best;
}

}  // namespace holoscan::radar
