// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/visualization/holoviz_op.hpp>

#include "radar_app/sar/hud_format.hpp"

namespace holoscan::radar::app::sar {

// Host HUD sampled beside the formed-image raster. Matches the source-branch
// on-image chrome (name, lattice, BP rate, lag) plus an APC look chevron as
// Holoviz text/geometry layers. The N-up compass inset is omitted: normalized
// coordinates squash it on a landscape strip.
struct FormationDiag {
  double bp_compute_s{0.0};
  int64_t pulses{0};
  int64_t pixels{0};
  int64_t bp_num_v{0};
  int64_t bp_num_u{0};
  std::optional<std::chrono::steady_clock::time_point> emit_at;
  std::optional<std::chrono::steady_clock::time_point> last_packet_at;
  bool playback_valid{false};
  bool playback_paced{false};
  double playback_behind_s{0.0};
  double playback_target_s{0.0};
  double playback_wall_s{0.0};
  bool apc_valid{false};
  double raster_u{0.0};
  double raster_v{0.0};
  std::optional<double> form_latency_s;
  std::string rate_line;
};

struct OverlayConfig {
  std::function<FormationDiag()> diag;
  std::vector<std::string> scene_names;
  bool show_playback_lag{false};
  // Docked output keeps the status and image in the same timestamped canvas.
  int status_width{0}, footer_height{0}, canvas_width{0}, image_height{0};
};

using holoscan::visualization::kTextRunHeaderBytes;
inline constexpr std::uint32_t kHudRuns = 6U;
inline constexpr std::uint32_t kHudChars = 64U;
inline constexpr std::size_t kHudRowBytes = kTextRunHeaderBytes + kHudChars;
inline constexpr std::uint32_t kHudMaxGlyphs = kHudRuns * kHudChars;
inline constexpr std::uint32_t kHudMaxLines = 48U;
inline constexpr std::size_t kHudLineFloats = 4U;

// Packed 0xAABBGGRR (Holoviz text-layer colour).
constexpr std::uint32_t hud_rgba(std::uint32_t r, std::uint32_t g, std::uint32_t b,
                                 std::uint32_t a = 255U) {
  return (a << 24U) | (b << 16U) | (g << 8U) | r;
}

inline void write_hud_run(std::uint8_t* row, const holoscan::visualization::TextRunHeader& header,
                          std::string_view text) {
  std::memcpy(row, &header, sizeof(header));
  std::uint8_t* chars = row + kTextRunHeaderBytes;
  std::memset(chars, 0, kHudChars);
  std::memcpy(chars, text.data(), std::min<std::size_t>(text.size(), kHudChars));
}

inline void add_hud_line(std::vector<float>* lines, float x0, float y0, float x1, float y1) {
  if (lines->size() / kHudLineFloats >= kHudMaxLines)
    return;
  lines->insert(lines->end(), {x0, y0, x1, y1});
}

// Border chevron in display coordinates (+x right, +y down). `east`/`north`
// are already swapped for landscape rasters.
inline void add_chevron_lines(std::vector<float>* lines, double east, double north) {
  const double len = std::hypot(east, north);
  if (!(len > 0.0))
    return;
  const double dx = east / len;
  const double dy = -north / len;
  constexpr double kCx = 0.5;
  constexpr double kCy = 0.5;
  constexpr double kX0 = 0.03;
  constexpr double kY0 = 0.03;
  constexpr double kX1 = 0.97;
  constexpr double kY1 = 0.97;
  double t_hit = 1.0e300;
  double hx = kCx;
  double hy = kCy;
  auto consider_x = [&](double xe) {
    if (dx == 0.0)
      return;
    const double t = (xe - kCx) / dx;
    if (!(t > 0.0) || t >= t_hit)
      return;
    const double y = kCy + t * dy;
    if (y >= kY0 && y <= kY1) {
      t_hit = t;
      hx = xe;
      hy = y;
    }
  };
  auto consider_y = [&](double ye) {
    if (dy == 0.0)
      return;
    const double t = (ye - kCy) / dy;
    if (!(t > 0.0) || t >= t_hit)
      return;
    const double x = kCx + t * dx;
    if (x >= kX0 && x <= kX1) {
      t_hit = t;
      hx = x;
      hy = ye;
    }
  };
  consider_x(kX0);
  consider_x(kX1);
  consider_y(kY0);
  consider_y(kY1);
  if (!(t_hit < 1.0e299))
    return;
  const double px = -dy;
  const double py = dx;
  const float tx = static_cast<float>(hx);
  const float ty = static_cast<float>(hy);
  const float l0 = static_cast<float>(hx - dx * 0.035 + px * 0.018);
  const float l1 = static_cast<float>(hy - dy * 0.035 + py * 0.018);
  const float r0 = static_cast<float>(hx - dx * 0.035 - px * 0.018);
  const float r1 = static_cast<float>(hy - dy * 0.035 - py * 0.018);
  add_hud_line(lines, tx, ty, l0, l1);
  add_hud_line(lines, tx, ty, r0, r1);
  add_hud_line(lines, l0, l1, r0, r1);
}

inline void fill_hud_geometry(std::vector<float>* lines, const FormationDiag& diag,
                              bool transpose_axes) {
  lines->clear();
  if (!diag.apc_valid)
    return;
  const double chev_e = transpose_axes ? diag.raster_v : diag.raster_u;
  const double chev_n = transpose_axes ? -diag.raster_u : -diag.raster_v;
  add_chevron_lines(lines, chev_e, chev_n);
}

inline void write_hud_rows(std::uint8_t* rows, const std::vector<std::string>& lines,
                           const FormationDiag& diag, bool show_lag, int64_t window_h) {
  using holoscan::visualization::TextRunHeader;
  std::memset(rows, 0, kHudRuns * kHudRowBytes);
  const auto gold = hud_rgba(255, 220, 80);
  // y is a fraction of the framebuffer; size_px is pixels. A short strip
  // window is a few hundred pixels tall, so a fixed 0.038 step is smaller
  // than the glyph and the lines collide.
  constexpr float kLabelPx = 18.0F;
  constexpr float kLabelGapPx = 6.0F;
  const float dy = std::max(
      0.038F, (kLabelPx + kLabelGapPx) / static_cast<float>(std::max<int64_t>(1, window_h)));
  for (std::size_t i = 0; i < lines.size() && i < 3U; ++i) {
    write_hud_run(rows + i * kHudRowBytes,
                  TextRunHeader{.x = 0.02F,
                                .y = 0.02F + dy * static_cast<float>(i),
                                .size_px = kLabelPx,
                                .color = gold},
                  lines[i]);
  }
  if (show_lag && diag.playback_valid && diag.playback_paced) {
    const std::string lag =
        format_lag_line(diag.playback_behind_s, diag.playback_target_s, diag.playback_wall_s);
    const auto color = diag.playback_behind_s > 0.2 ? hud_rgba(255, 96, 80) : hud_rgba(80, 200, 80);
    write_hud_run(rows + 4U * kHudRowBytes,
                  TextRunHeader{.x = 0.02F, .y = 0.94F, .size_px = 16.0F, .color = color}, lag);
  }
}

// A bounded left-hand status panel. The timeline never regenerates these rows:
// they are part of the completed image whose acquisition stamp they describe.
inline void write_footer_hud_rows(std::uint8_t* rows, const std::vector<std::string>& lines,
                                  const FormationDiag& diag, bool show_lag,
                                  const OverlayConfig& layout) {
  using holoscan::visualization::TextRunHeader;
  std::memset(rows, 0, kHudRuns * kHudRowBytes);
  const int total_height = layout.image_height + layout.footer_height;
  auto write = [&](std::size_t index, std::string text, float pixels, std::uint32_t color) {
    const auto limit = static_cast<std::size_t>(std::max(
        3,
        std::min<int>(kHudChars, static_cast<int>((layout.status_width - 20) / (pixels * 0.65F)))));
    if (text.size() > limit)
      text = text.substr(0, limit - 3) + "...";
    write_hud_run(rows + index * kHudRowBytes,
                  TextRunHeader{.x = 10.0F / layout.canvas_width,
                                .y = (layout.image_height + 7.0F + index * 16.0F) / total_height,
                                .size_px = pixels,
                                .color = color},
                  text);
  };
  const auto title_color = hud_rgba(235, 239, 244);
  const auto detail_color = hud_rgba(165, 175, 188);
  const auto rate_color = hud_rgba(194, 211, 228);
  if (!lines.empty()) {
    const auto split = lines[0].rfind("  ");
    write(0, split == std::string::npos ? "SAR image" : lines[0].substr(0, split), 14, title_color);
    write(1, "Image " + (split == std::string::npos ? lines[0] : lines[0].substr(split + 2)), 12,
          detail_color);
  }
  if (lines.size() > 1) {
    const auto count = static_cast<std::size_t>(std::max(4, (layout.status_width - 20) / 8));
    const auto split = lines[1].size() > count ? lines[1].rfind(' ', count) : std::string::npos;
    if (split != std::string::npos) {
      write(2, lines[1].substr(0, split), 12, rate_color);
      write(3, lines[1].substr(split + 1), 12, rate_color);
    } else {
      write(2, lines[1], 12, rate_color);
    }
  }
  if (lines.size() > 2)
    write(4, lines[2], 12, detail_color);
  if (show_lag && diag.playback_valid && diag.playback_paced)
    write(5, format_lag_line(diag.playback_behind_s, diag.playback_target_s, diag.playback_wall_s),
          11, diag.playback_behind_s > 0.2 ? hud_rgba(255, 120, 105) : hud_rgba(115, 200, 150));
}

}  // namespace holoscan::radar::app::sar
