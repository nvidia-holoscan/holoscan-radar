// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace holoscan::radar::app {

// Entry count of the RD-map color lookup table; RdMapToImageOp maps log-
// magnitude onto [0, kLutSize - 1] and Holoviz / X11 viewers index into it.
constexpr size_t kLutSize = 256;

// kLutSize-entry RGBA lookup table for the turbo colormap, built from the
// published 5th-order polynomial fit of the Google turbo palette.
inline std::vector<std::vector<float>> make_turbo_lut() {
  constexpr float kR[6] = {0.13572138F, 4.61539260F, -42.66032258F,
                           132.13108234F, -152.94239396F, 59.28637943F};
  constexpr float kG[6] = {0.09140261F, 2.19418839F, 4.84296658F,
                           -14.18503333F, 4.27729857F, 2.82956604F};
  constexpr float kB[6] = {0.10667330F, 12.64194608F, -60.58204836F,
                           110.36276771F, -89.90310912F, 27.34824973F};
  auto poly = [](const float (&c)[6], float t) {
    float v = c[5];
    for (int i = 4; i >= 0; --i) {
      v = v * t + c[i];
    }
    return std::clamp(v, 0.0F, 1.0F);
  };
  std::vector<std::vector<float>> lut;
  lut.reserve(kLutSize);
  for (size_t i = 0; i < kLutSize; ++i) {
    const float t = static_cast<float>(i) / (kLutSize - 1);
    lut.push_back({poly(kR, t), poly(kG, t), poly(kB, t), 1.0F});
  }
  return lut;
}

}  // namespace holoscan::radar::app
