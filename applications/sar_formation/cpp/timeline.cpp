// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "custom_ops.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/viz/font_5x7.hpp"
#include "radar_app/viz/viz.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void fill_rect(std::vector<uint8_t>* img, int64_t w, int64_t h,
               int64_t x0, int64_t y0, int64_t x1, int64_t y1,
               const std::array<uint8_t, 3>& c) {
  if (x0 > x1) {
    std::swap(x0, x1);
  }
  if (y0 > y1) {
    std::swap(y0, y1);
  }
  x0 = std::max(int64_t{0}, x0);
  y0 = std::max(int64_t{0}, y0);
  x1 = std::min(w, x1);
  y1 = std::min(h, y1);
  for (int64_t y = y0; y < y1; ++y) {
    for (int64_t x = x0; x < x1; ++x) {
      holoscan::radar::app::plot_put_px(img, w, h, x, y, c);
    }
  }
}

int64_t time_to_x(Clock::time_point t, Clock::time_point origin,
                  double window_s, int64_t w) {
  const double age =
      std::chrono::duration<double>(t - origin).count();
  const double frac = age / window_s;
  return static_cast<int64_t>(std::llround(frac * static_cast<double>(w)));
}

void draw_bar(std::vector<uint8_t>* img, int64_t w, int64_t h,
              Clock::time_point t0, Clock::time_point t1,
              Clock::time_point origin, double window_s,
              int64_t y0, int64_t y1,
              const std::array<uint8_t, 3>& c) {
  const int64_t x0 = time_to_x(t0, origin, window_s, w);
  const int64_t x1 = time_to_x(t1, origin, window_s, w);
  if (x1 <= 0 || x0 >= w) {
    return;
  }
  fill_rect(img, w, h, x0, y0, x1, y1, c);
}

void draw_tick(std::vector<uint8_t>* img, int64_t w, int64_t h,
               Clock::time_point t, Clock::time_point origin,
               double window_s, int64_t y0, int64_t y1,
               const std::array<uint8_t, 3>& c) {
  const int64_t x = time_to_x(t, origin, window_s, w);
  fill_rect(img, w, h, x - 1, y0, x + 2, y1, c);
}

std::array<uint8_t, 3> collect_color(int file_index) {
  static constexpr std::array<std::array<uint8_t, 3>, 8> k{{
      {{36, 158, 186}},
      {{222, 118, 36}},
      {{62, 168, 82}},
      {{156, 92, 198}},
      {{214, 78, 98}},
      {{204, 168, 40}},
      {{72, 128, 220}},
      {{150, 178, 48}},
  }};
  if (file_index < 0) {
    return {{50, 140, 190}};
  }
  return k[static_cast<size_t>(file_index) % k.size()];
}

int span_file_at(const std::vector<radar::CollectionSpan>& spans,
                 Clock::time_point t) {
  int idx = -1;
  bool have = false;
  Clock::time_point start{};
  for (const auto& span : spans) {
    if (span.dwell_start > t) {
      continue;
    }
    if (!have || span.dwell_start >= start) {
      have = true;
      start = span.dwell_start;
      idx = span.file_index;
    }
  }
  return idx;
}

std::string format_latency_ms(double s) {
  if (!std::isfinite(s)) {
    return {};
  }
  const bool neg = s < 0.0;
  const int ms = static_cast<int>(
      std::llround(std::abs(s) * 1000.0));
  std::string out = neg ? "-" : "";
  out += std::to_string(ms);
  out += "ms";
  return out;
}

std::string format_tenths_s(double s) {
  const int scaled =
      static_cast<int>(std::llround(std::abs(s) * 10.0));
  return std::to_string(scaled / 10) + "." +
         std::to_string(scaled % 10) + "s";
}

std::string format_lag(const radar::PlaybackStatus& st) {
  if (!st.valid || !st.paced || !std::isfinite(st.behind_s)) {
    return {};
  }
  if (std::abs(st.behind_s) < 0.05) {
    return "on time";
  }
  if (st.behind_s > 0.0) {
    return "+" + format_tenths_s(st.behind_s) + " behind";
  }
  return format_tenths_s(-st.behind_s) + " ahead";
}

std::array<uint8_t, 3> lag_color(const radar::PlaybackStatus& st) {
  if (st.behind_s > 0.2) {
    return {{255, 80, 80}};
  }
  if (st.behind_s > 0.05) {
    return {{255, 180, 60}};
  }
  return {{80, 200, 80}};
}

std::string fit_label(const std::string& name, int64_t max_w,
                      int scale) {
  if (max_w <= 0) {
    return {};
  }
  if (holoscan::radar::app::plot_text_width_5x7(name, scale) <=
      max_w) {
    return name;
  }
  std::string s = name;
  while (!s.empty() &&
         holoscan::radar::app::plot_text_width_5x7(s, scale) >
             max_w) {
    s.pop_back();
  }
  return s;
}

}  // namespace

void TimelineRasterOp::compute(
    [[maybe_unused]] holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context) {
  const char* who = "TimelineRasterOp";
  if (!(window_s > 0.0) || width <= 0 || height <= 0) {
    throw std::invalid_argument(
        std::string(who) +
        ": window_s, width, and height must be positive");
  }
  std::vector<radar::CollectionSpan> collected;
  if (spans) {
    collected = spans();
  }
  const auto now = Clock::now();
  const auto origin = now - std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(window_s));
  if (last_image_emit) {
    const auto at = last_image_emit();
    if (at.has_value() &&
        (!last_seen_emit_.has_value() || *at != *last_seen_emit_)) {
      last_seen_emit_ = at;
      ImageMark mark;
      mark.at = *at;
      Clock::time_point color_at = *at;
      if (last_packet_at) {
        const auto pkt = last_packet_at();
        if (pkt.has_value()) {
          color_at = *pkt;
          mark.latency_s = std::chrono::duration<double>(
                               *at - *pkt)
                               .count();
        }
      }
      // Last-packet span is the formed collect. emit_at often
      // lands in the next dwell (reader overlaps BP).
      mark.file_index = span_file_at(collected, color_at);
      image_emits_.push_back(mark);
    }
  }
  // Visible window, not a count. 64 marks was ~29 s at 2 Hz
  // and only ~7 s at one-tick emits (~8.7 Hz).
  const auto keep = std::find_if(
      image_emits_.begin(), image_emits_.end(),
      [&](const ImageMark& m) { return m.at >= origin; });
  image_emits_.erase(image_emits_.begin(), keep);
  const int64_t w = width;
  const int64_t h = height;
  std::vector<uint8_t> img(static_cast<size_t>(w * h * 4), 0);
  const std::array<uint8_t, 3> bg{16, 16, 16};
  fill_rect(&img, w, h, 0, 0, w, h, bg);

  const std::array<uint8_t, 3> quiet_c{90, 90, 90};
  const std::array<uint8_t, 3> fg{220, 220, 220};
  const std::array<uint8_t, 3> axis_c{80, 80, 80};
  const std::array<uint8_t, 3> head_c{255, 80, 80};

  constexpr int kScale = 2;
  const int64_t label_h = holoscan::radar::app::kFont5x7Rows * kScale;
  const int64_t top = 4;
  const int64_t lane_gap = 3;
  const int64_t axis_gap = 4;
  const int64_t lab_gap = 6;
  const int64_t bot = 4;
  const int64_t axis_block =
      axis_gap + 1 + lab_gap + label_h + bot;
  const int64_t usable =
      std::max(int64_t{24}, h - top - lane_gap - axis_block);
  const int64_t plat_h = usable / 2;
  const int64_t plat_y0 = top;
  const int64_t plat_y1 = plat_y0 + plat_h;
  const int64_t prod_y0 = plat_y1 + lane_gap;
  const int64_t prod_y1 = prod_y0 + (usable - plat_h);
  const int64_t axis_y = prod_y1 + axis_gap;
  fill_rect(&img, w, h, 0, plat_y0, w, plat_y1,
            std::array<uint8_t, 3>{28, 28, 28});
  fill_rect(&img, w, h, 0, prod_y0, w, prod_y1,
            std::array<uint8_t, 3>{28, 28, 28});

  for (const auto& span : collected) {
    const auto dwell_end = span.dwell_end.value_or(now);
    const auto dwell_c = collect_color(span.file_index);
    draw_bar(&img, w, h, span.dwell_start, dwell_end, origin,
             window_s, plat_y0, plat_y1, dwell_c);
    if (span.quiet_start.has_value()) {
      const auto quiet_end = span.quiet_end.value_or(now);
      draw_bar(&img, w, h, *span.quiet_start, quiet_end, origin,
               window_s, plat_y0, plat_y1, quiet_c);
    }
    std::string name;
    if (!scene_names.empty() && span.file_index >= 0) {
      const size_t idx = static_cast<size_t>(span.file_index) %
                         scene_names.size();
      name = scene_names[idx];
    }
    if (!name.empty()) {
      const int64_t x0 = time_to_x(span.dwell_start, origin,
                                   window_s, w);
      const int64_t x1 = time_to_x(dwell_end, origin, window_s, w);
      const int64_t vis0 = std::max(int64_t{0}, x0);
      const int64_t vis1 = std::min(w, x1);
      name = fit_label(name, vis1 - vis0 - 8, kScale);
      if (!name.empty()) {
        const int64_t ty =
            plat_y0 + (plat_y1 - plat_y0 - label_h) / 2;
        holoscan::radar::app::plot_draw_text_5x7(
            &img, w, h, vis0 + 4, ty, name, fg, kScale);
      }
    }
  }
  const int64_t lat_y =
      prod_y0 + (prod_y1 - prod_y0 - label_h) / 2;
  for (const auto& mark : image_emits_) {
    const auto tick_c = collect_color(mark.file_index);
    draw_tick(&img, w, h, mark.at, origin, window_s, prod_y0,
              prod_y1, tick_c);
    if (!mark.latency_s.has_value()) {
      continue;
    }
    const std::string lab = format_latency_ms(*mark.latency_s);
    if (lab.empty()) {
      continue;
    }
    const int64_t x = time_to_x(mark.at, origin, window_s, w);
    const int64_t tw =
        holoscan::radar::app::plot_text_width_5x7(lab, kScale);
    int64_t tx = x + 4;
    if (tx + tw > w - 4) {
      tx = x - 4 - tw;
    }
    if (tx < 2) {
      tx = 2;
    }
    holoscan::radar::app::plot_draw_text_5x7(
        &img, w, h, tx, lat_y, lab, tick_c, kScale);
  }

  fill_rect(&img, w, h, 0, axis_y, w, axis_y + 1, axis_c);
  const int n_ticks = 5;
  for (int i = 0; i <= n_ticks; ++i) {
    const double frac = static_cast<double>(i) /
                        static_cast<double>(n_ticks);
    const int64_t x = static_cast<int64_t>(
        std::llround(frac * static_cast<double>(w)));
    fill_rect(&img, w, h, x, axis_y - 2, x + 1, axis_y + 3, axis_c);
    const double ago = window_s * (1.0 - frac);
    std::string lab;
    if (i == n_ticks) {
      lab = "now";
    } else {
      const int sec = static_cast<int>(std::llround(ago));
      lab = std::to_string(sec) + "s";
    }
    const int64_t tw =
        holoscan::radar::app::plot_text_width_5x7(lab, kScale);
    int64_t tx = x - tw / 2;
    if (tx < 2) {
      tx = 2;
    }
    if (tx + tw > w - 2) {
      tx = w - 2 - tw;
    }
    holoscan::radar::app::plot_draw_text_5x7(
        &img, w, h, tx, axis_y + lab_gap, lab, fg, kScale);
  }
  fill_rect(&img, w, h, w - 2, 0, w, h, head_c);

  if (playback) {
    const auto st = playback();
    const std::string lab = format_lag(st);
    if (!lab.empty()) {
      const int64_t tw =
          holoscan::radar::app::plot_text_width_5x7(lab, kScale);
      const int64_t pad = 4;
      const int64_t tx = w - 8 - tw;
      const int64_t ty =
          plat_y0 + (plat_y1 - plat_y0 - label_h) / 2;
      fill_rect(&img, w, h, tx - pad, plat_y0, w - 2, plat_y1,
                bg);
      holoscan::radar::app::plot_draw_text_5x7(
          &img, w, h, tx, ty, lab, lag_color(st), kScale);
    }
  }

  auto tensor = radar::make_device_tensor(
      {h, w, 4}, DLDataType{kDLUInt, 8, 1});
  if (cudaMemcpy(tensor->data(), img.data(), img.size(),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error(
        std::string(who) + ": timeline H2D failed");
  }
  holoscan::TensorMap map;
  map.insert({"timeline", std::move(tensor)});
  op_output.emit(map, "timeline");
}
