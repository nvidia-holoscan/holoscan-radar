// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// next_fast_fft_size: 5-smooth pad used as the cuFFT radix length.

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "radar_common/dsp/fft_size.hpp"

namespace {

using holoscan::radar::next_fast_fft_size;

bool is_five_smooth(int64_t n) {
  if (n < 1) {
    return false;
  }
  while (n % 2 == 0) {
    n /= 2;
  }
  while (n % 3 == 0) {
    n /= 3;
  }
  while (n % 5 == 0) {
    n /= 5;
  }
  return n == 1;
}

}  // namespace

TEST(NextFastFftSize, IdentityOnFiveSmooth) {
  EXPECT_EQ(next_fast_fft_size(1), 1);
  EXPECT_EQ(next_fast_fft_size(8), 8);
  EXPECT_EQ(next_fast_fft_size(9), 9);
  EXPECT_EQ(next_fast_fft_size(12), 12);
  EXPECT_EQ(next_fast_fft_size(32768), 32768);
}

TEST(NextFastFftSize, PadsPrimesToRadixLength) {
  EXPECT_EQ(next_fast_fft_size(7), 8);
  EXPECT_EQ(next_fast_fft_size(11), 12);
  EXPECT_EQ(next_fast_fft_size(13), 15);
  // OPEN_ITEMS.md: pad 4099 to 4320, not a merely-sufficient prime.
  EXPECT_EQ(next_fast_fft_size(4099), 4320);
  // Ohio Umbra FX is one below 5-smooth (22499 or 16183 in docs).
  EXPECT_EQ(next_fast_fft_size(22499), 22500);
  EXPECT_EQ(next_fast_fft_size(16183), 16200);
}

TEST(NextFastFftSize, ResultIsFiveSmoothAndAtLeastN) {
  for (int64_t n = 1; n <= 4096; ++n) {
    const int64_t f = next_fast_fft_size(n);
    EXPECT_GE(f, n);
    EXPECT_TRUE(is_five_smooth(f)) << n << " -> " << f;
  }
}

TEST(NextFastFftSize, RejectsNonPositive) {
  EXPECT_THROW(next_fast_fft_size(0), std::invalid_argument);
  EXPECT_THROW(next_fast_fft_size(-1), std::invalid_argument);
}
