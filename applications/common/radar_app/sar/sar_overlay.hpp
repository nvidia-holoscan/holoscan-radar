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
// on-image chrome (name, lattice, BP rate, lag) plus a sensor look annotation as
// native text and cached display pixels. The N-up compass inset is omitted: normalized
// coordinates squash it on a landscape strip.
struct FormationDiag {
  double bp_compute_s{0.0};
  std::optional<double> collection_elapsed_s;
  std::optional<double> budget_s;
  std::optional<double> budget_playback_rate;
  std::optional<int64_t> capture_time_ns;
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
  std::function<FormationDiag(int64_t)> diag_at_capture;
  bool strip_update_metrics{false};
  std::vector<std::string> scene_names;
  bool show_playback_lag{false};
  // Docked output keeps the status and image in the same timestamped canvas.
  int status_width{0}, footer_height{0}, canvas_width{0}, image_height{0};
};

using holoscan::visualization::kTextRunHeaderBytes;
inline constexpr std::uint32_t kHudRuns = 7U;
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

// Pixel-space display annotation. The cone is a direction cue, not a beamwidth.
// Keep its short dimensions in pixels so wide strip images do not squash it.
struct SensorLookAnnotation {
  float origin_x{}, origin_y{};
  float label_x{}, label_y{}, label_width{}, label_height{22};
  float left{}, top{}, right{}, bottom{};
  float segments[7][4]{};
};
inline constexpr std::string_view kSensorLookLabel = "Sensor look angle";
inline constexpr float kSensorLookFont = 14;

inline std::optional<SensorLookAnnotation> sensor_look_annotation(const FormationDiag& diag,
                                                                  bool transpose_axes, int width,
                                                                  int height) {
  if (!diag.apc_valid || width < 180 || height < 100)
    return std::nullopt;
  const double x = transpose_axes ? diag.raster_v : diag.raster_u;
  const double y = transpose_axes ? diag.raster_u : diag.raster_v;
  const double length = std::hypot(x, y);
  if (!std::isfinite(length) || length <= 0)
    return std::nullopt;
  const double dx = x / length, dy = y / length;
  constexpr double margin = 16;
  const double cx = width * 0.5, cy = height * 0.5;
  const double tx = dx == 0 ? 1.0e30 : (cx - margin) / std::abs(dx);
  const double ty = dy == 0 ? 1.0e30 : (cy - margin) / std::abs(dy);
  const double t = std::min(tx, ty);
  SensorLookAnnotation result;
  result.origin_x = static_cast<float>(cx + dx * t);
  result.origin_y = static_cast<float>(cy + dy * t);
  const float ox = result.origin_x, oy = result.origin_y;
  // At most 54px long: substantially shorter than the preview's viewing cone.
  const float reach = std::clamp(std::min(width, height) * 0.07F, 24.0F, 54.0F);
  const float tip_x = ox - dx * reach, tip_y = oy - dy * reach;
  const float px = -dy, py = dx;
  const float spread = reach * 0.18F;
  result.label_width = kSensorLookLabel.size() * kSensorLookFont * 7 / 13 + 12;
  result.label_x = std::clamp(ox - result.label_width * (dx >= 0 ? 1.0F : 0.0F), 6.0F,
                              width - result.label_width - 6);
  // Above/below-edge cues need the label beyond the arrow tip, not over it.
  const float label_offset = dy < -0.5          ? reach + 12
                             : dy > 0.5         ? -reach - result.label_height - 12
                             : oy < height - 56 ? 24
                                                : -46;
  result.label_y = std::clamp(oy + label_offset, 6.0F, height - result.label_height - 6);
  const float anchor_x =
      std::clamp(ox, result.label_x + 6, result.label_x + result.label_width - 6);
  const float anchor_y =
      result.label_y > oy ? result.label_y : result.label_y + result.label_height;
  const auto segment = [&](int i, float ax, float ay, float bx, float by) {
    result.segments[i][0] = ax;
    result.segments[i][1] = ay;
    result.segments[i][2] = bx;
    result.segments[i][3] = by;
  };
  segment(0, ox, oy, tip_x + px * spread, tip_y + py * spread);
  segment(1, ox, oy, tip_x - px * spread, tip_y - py * spread);
  segment(2, ox, oy, tip_x, tip_y);
  segment(3, tip_x, tip_y, tip_x + dx * 6 + px * 3, tip_y + dy * 6 + py * 3);
  segment(4, tip_x, tip_y, tip_x + dx * 6 - px * 3, tip_y + dy * 6 - py * 3);
  segment(5, ox, oy, anchor_x, anchor_y + (anchor_y > oy ? -8 : 8));
  segment(6, anchor_x, anchor_y + (anchor_y > oy ? -8 : 8), anchor_x, anchor_y);
  result.left = std::min(result.label_x, ox - reach - 3);
  result.top = std::min(result.label_y, oy - reach - 3);
  result.right = std::max(result.label_x + result.label_width, ox + reach + 3);
  result.bottom = std::max(result.label_y + result.label_height, oy + reach + 3);
  return result;
}

inline void write_sensor_look_row(std::uint8_t* rows, const SensorLookAnnotation& look,
                                  int canvas_width, int total_height) {
  write_hud_run(rows + 6U * kHudRowBytes,
                holoscan::visualization::TextRunHeader{.x = (look.label_x + 6) / canvas_width,
                                                       .y = (look.label_y + 4) / total_height,
                                                       .size_px = kSensorLookFont,
                                                       .color = hud_rgba(157, 211, 219)},
                kSensorLookLabel);
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
                                  const FormationDiag& diag, bool /*show_lag*/,
                                  const OverlayConfig& layout) {
  using holoscan::visualization::TextRunHeader;
  std::memset(rows, 0, kHudRuns * kHudRowBytes);
  const int total_height = layout.image_height + layout.footer_height;
  auto write = [&](std::size_t index, std::string text, float y, float pixels,
                   std::uint32_t color) {
    const auto limit = static_cast<std::size_t>(std::max(
        3,
        std::min<int>(kHudChars, static_cast<int>((layout.status_width - 20) / (pixels * 0.65F)))));
    if (text.size() > limit)
      text = text.substr(0, limit - 3) + "...";
    write_hud_run(rows + index * kHudRowBytes,
                  TextRunHeader{.x = 8.0F / layout.canvas_width,
                                .y = (layout.image_height + y) / total_height,
                                .size_px = pixels,
                                .color = color},
                  text);
  };
  const auto title_color = hud_rgba(75, 205, 220);
  const auto detail_color = hud_rgba(205, 217, 229);
  const auto rate_color = hud_rgba(230, 237, 243);
  std::string dimensions;
  if (!lines.empty()) {
    const auto split = lines[0].rfind("  ");
    write(0, split == std::string::npos ? "SAR image" : lines[0].substr(0, split), 2, 18,
          title_color);
    dimensions = split == std::string::npos ? lines[0] : lines[0].substr(split + 2);
  }
  std::string pulses = diag.pulses > 0 ? std::to_string(diag.pulses) : "--";
  if (diag.pulses > 0) {
    for (std::size_t end = pulses.size(); end > 3; end -= 3)
      pulses.insert(end - 3, ",");
  }
  write(1,
        (dimensions.empty() ? "--" : dimensions) + " pixels" +
            (layout.strip_update_metrics ? " (mosaic)" : "  " + pulses + " pulses"),
        25, 14, detail_color);
  write(2, "BP work: " + format_bp_giga_work(diag.pulses, diag.pixels) + " Gpx*pulses",
        layout.strip_update_metrics ? 43 : 45, 14, rate_color);
  const bool timed = std::isfinite(diag.bp_compute_s) && diag.bp_compute_s > 0;
  write(3,
        "Latency: " + (timed ? format_seconds(diag.bp_compute_s) : "--") + " / " +
            format_playback_budget(diag.budget_s, diag.budget_playback_rate),
        layout.strip_update_metrics ? 61 : 65, 14, rate_color);
  write(4,
        "Throughput: " + format_bp_giga_throughput(diag.bp_compute_s, diag.pulses, diag.pixels) +
            " Gpx*pulses/sec",
        layout.strip_update_metrics ? 79 : 85, 14, rate_color);
  if (layout.strip_update_metrics)
    write(5, "Last strip update: " + pulses + " pulses", 97, 12, detail_color);
  // Acquisition/elapsed progress stays in the adjacent live panel. These
  // rows remain frozen with the completed image; no timeline tick redraws them.
}

// Static startup text uses the existing native GPU text layer and one exact
// stamp with its charcoal image. Live elapsed/stage text belongs to the small
// timeline panel, so no clock tick needs to recreate this completed canvas.
inline void write_startup_hud_rows(std::uint8_t* rows, const OverlayConfig& layout) {
  using holoscan::visualization::TextRunHeader;
  std::memset(rows, 0, kHudRuns * kHudRowBytes);
  const float width = layout.canvas_width;
  const float height = layout.image_height;
  const float total = height + layout.footer_height;
  auto centered = [&](std::size_t row, std::string text, float y, float size, std::uint32_t color) {
    const auto limit = static_cast<std::size_t>(
        std::clamp((width - 24) / (size * 0.6F), 3.0F, static_cast<float>(kHudChars)));
    if (text.size() > limit)
      text = text.substr(0, limit - 3) + "...";
    const float x = std::max(4.0F, (width - text.size() * size * 7 / 13) / 2);
    write_hud_run(
        rows + row * kHudRowBytes,
        TextRunHeader{
            .x = x / width, .y = std::max(0.0F, y) / total, .size_px = size, .color = color},
        text);
  };
  const float title_size = std::clamp((width - 24) / 14, 12.0F, 26.0F);
  centered(0, "PREPARING", height / 2 - 45, 14, hud_rgba(75, 205, 220));
  centered(1, "Preparing first SAR image", height / 2 - 17, title_size, hud_rgba(230, 237, 243));
  if (height >= 100 && !layout.scene_names.empty())
    centered(2, layout.scene_names.front(), height / 2 + 24, 16, hud_rgba(145, 163, 181));
  if (layout.status_width > 0) {
    write_hud_run(rows + 3 * kHudRowBytes,
                  TextRunHeader{.x = 10 / width,
                                .y = (height + 8) / total,
                                .size_px = 16,
                                .color = hud_rgba(75, 205, 220)},
                  "STARTING UP");
    const std::string text = layout.status_width >= 220 ? "Waiting for first image" : "Waiting...";
    write_hud_run(rows + 4 * kHudRowBytes,
                  TextRunHeader{.x = 10 / width,
                                .y = (height + 32) / total,
                                .size_px = 14,
                                .color = hud_rgba(165, 175, 188)},
                  text);
  }
}

}  // namespace holoscan::radar::app::sar
