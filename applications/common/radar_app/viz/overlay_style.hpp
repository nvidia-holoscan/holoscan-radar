// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// [EA HSDK5 ONLY] Holoviz-free overlay styling for the X11 RD viewer: the
// cluster color palette, the rank-based cluster→color assignment, and numeric
// tick formatting. These mirror the Holoviz path's cluster_palette() /
// rep_color_ranks() / plot_format_tick() in viz/viz.hpp so a cluster reads the
// same on either backend; viz.hpp keeps its own copies because it cannot be
// included without Holoviz. Fold back into one header when Holoviz returns.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace holoscan::radar::app {

// Palette size; a cluster's color is its range rank modulo this count.
constexpr int kNumOverlayClusterColors = 30;

struct OverlayRgb {
  std::uint8_t r{0};
  std::uint8_t g{0};
  std::uint8_t b{0};
};

// Muted gray for DBSCAN noise points (label -1).
constexpr OverlayRgb kOverlayNoiseColor{150U, 150U, 150U};

// Categorical palette: hues step by the golden angle (maximally spread on the
// wheel for any count) with saturation/value alternating so hue-adjacent
// entries stay distinct and every dot reads bright against the turbo image.
inline std::vector<OverlayRgb> overlay_cluster_palette() {
  constexpr float kGolden = 0.61803398875F;  // golden-ratio conjugate
  std::vector<OverlayRgb> palette;
  palette.reserve(static_cast<std::size_t>(kNumOverlayClusterColors));
  for (int i = 0; i < kNumOverlayClusterColors; ++i) {
    const float h6 = std::fmod(static_cast<float>(i) * kGolden, 1.0F) * 6.0F;
    const float s = 0.70F + 0.30F * static_cast<float>(i % 2);       // .70/1.0
    const float v = 1.0F - 0.25F * static_cast<float>((i / 2) % 2);  // 1.0/.75
    const int hi = static_cast<int>(h6) % 6;
    const float f = h6 - std::floor(h6);
    const float p = v * (1.0F - s);
    const float q = v * (1.0F - s * f);
    const float t = v * (1.0F - s * (1.0F - f));
    float r = v;
    float g = t;
    float b = p;  // hi == 0
    switch (hi) {
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
      default: break;
    }
    const auto to_byte = [](float channel) {
      return static_cast<std::uint8_t>(
          std::clamp(channel, 0.0F, 1.0F) * 255.0F + 0.5F);
    };
    palette.push_back(OverlayRgb{to_byte(r), to_byte(g), to_byte(b)});
  }
  return palette;
}

// Frame-stable cluster ordering: cluster indices sorted by representative
// range bin (Doppler bin, then cluster index as tiebreaks). `rep` is the
// (num_clusters, 2) representative cells with columns (doppler_bin,
// range_bin). The DBSCAN label is SIZE-ranked, so it names a different
// physical cluster whenever two clusters trade point counts — anything the
// eye tracks across frames (a color, a legend column) must key off this
// ordering instead, which only shifts when clusters actually reorder in
// range or the set changes.
inline std::vector<int> overlay_rep_rank_order(
    const std::vector<std::int64_t>& rep, std::int64_t num_clusters) {
  const std::size_t n =
      static_cast<std::size_t>(std::max<std::int64_t>(num_clusters, 0));
  std::vector<int> order(n);
  for (std::size_t i = 0; i < n; ++i) {
    order[i] = static_cast<int>(i);
  }
  std::sort(order.begin(), order.end(), [&rep](int a, int b) {
    const std::int64_t ra = rep[static_cast<std::size_t>(a) * 2 + 1];  // range
    const std::int64_t rb = rep[static_cast<std::size_t>(b) * 2 + 1];
    if (ra != rb) {
      return ra < rb;
    }
    const std::int64_t da = rep[static_cast<std::size_t>(a) * 2 + 0];
    const std::int64_t db = rep[static_cast<std::size_t>(b) * 2 + 0];
    if (da != db) {
      return da < db;
    }
    return a < b;
  });
  return order;
}

// Per-cluster color bucket: the range rank above, modulo the palette size.
// The result is row-aligned with `rep` (bucket[label] = that cluster's color).
inline std::vector<int> overlay_rep_color_ranks(
    const std::vector<std::int64_t>& rep, std::int64_t num_clusters) {
  const std::vector<int> order = overlay_rep_rank_order(rep, num_clusters);
  std::vector<int> bucket(order.size());
  for (std::size_t rank = 0; rank < order.size(); ++rank) {
    bucket[static_cast<std::size_t>(order[rank])] =
        static_cast<int>(rank % kNumOverlayClusterColors);
  }
  return bucket;
}

// Fixed-point tick / status formatting (no locale, no iostream). A positive
// `width` space-pads to a fixed field so readouts drawn in a monospace font
// keep their digit positions as the value crosses a power of ten.
inline std::string overlay_format(double value, int decimals, int width = 0) {
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%*.*f", std::clamp(width, 0, 16),
                std::clamp(decimals, 0, 3), value);
  return std::string(buffer);
}

// Signed form for velocity readouts ("+1.2" / "-6.7").
inline std::string overlay_format_signed(double value, int decimals,
                                         int width = 0) {
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%+*.*f", std::clamp(width, 0, 16),
                std::clamp(decimals, 0, 3), value);
  return std::string(buffer);
}

}  // namespace holoscan::radar::app
