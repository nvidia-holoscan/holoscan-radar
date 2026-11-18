// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_app/sar/timeline_raster.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "radar_app/sar/plot_raster.hpp"
#include "radar_common/sar/formation_hud.hpp"
#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar::app::sar {
namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kMaxRects = 2048;
constexpr size_t kTextRows = 256;
constexpr size_t kTextBytes = 128;
struct Rect {
  int64_t x0, y0, x1, y1;
  uint8_t r, g, b;
};
struct TimelineCommands {
  std::vector<Rect> rectangles;
  std::array<uint8_t, kTextRows * kTextBytes> text{};
  size_t rows{0};
};

// The host creates only bounded drawing commands and text records. Pixels and
// glyphs are rendered on the GPU, never through a full-frame host raster.
void fill_rect(TimelineCommands* img, int64_t, int64_t, int64_t x0, int64_t y0, int64_t x1,
               int64_t y1, const std::array<uint8_t, 3>& c) {
  if (img->rectangles.size() < kMaxRects)
    img->rectangles.push_back(
        {std::min(x0, x1), std::min(y0, y1), std::max(x0, x1), std::max(y0, y1), c[0], c[1], c[2]});
}

void draw_text(TimelineCommands* img, int64_t, int64_t, int64_t x, int64_t y,
               const std::string& text, const std::array<uint8_t, 3>& color, int scale) {
  if (img->rows >= kTextRows)
    return;
  auto* row = img->text.data() + img->rows++ * kTextBytes;
  const visualization::TextRunHeader header{
      .x = static_cast<float>(x),
      .y = static_cast<float>(y),
      .size_px = static_cast<float>(7 * scale),
      .color = 0xff000000U | (static_cast<uint32_t>(color[2]) << 16U) |
               (static_cast<uint32_t>(color[1]) << 8U) | color[0]};
  std::memcpy(row, &header, sizeof(header));
  std::memcpy(row + sizeof(header), text.data(),
              std::min(text.size(), kTextBytes - sizeof(header) - 1));
}

__global__ void render_timeline(uint8_t* output, int64_t width, int64_t height,
                                const Rect* rectangles, size_t count) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= width * height)
    return;
  const int64_t x = pixel % width, y = pixel / width;
  uint8_t r = 16, g = 16, b = 16;
  for (size_t i = 0; i < count; ++i) {
    const auto rect = rectangles[i];
    if (x >= rect.x0 && x < rect.x1 && y >= rect.y0 && y < rect.y1) {
      r = rect.r;
      g = rect.g;
      b = rect.b;
    }
  }
  output[pixel * 4] = r;
  output[pixel * 4 + 1] = g;
  output[pixel * 4 + 2] = b;
  output[pixel * 4 + 3] = 255;
}

int64_t time_to_x(Clock::time_point t, Clock::time_point origin, double window_s, int64_t w) {
  const double age = std::chrono::duration<double>(t - origin).count();
  const double frac = age / window_s;
  return static_cast<int64_t>(std::llround(frac * static_cast<double>(w)));
}

void draw_bar(TimelineCommands* img, int64_t w, int64_t h, Clock::time_point t0,
              Clock::time_point t1, Clock::time_point origin, double window_s, int64_t y0,
              int64_t y1, const std::array<uint8_t, 3>& c) {
  const int64_t x0 = time_to_x(t0, origin, window_s, w);
  const int64_t x1 = time_to_x(t1, origin, window_s, w);
  if (x1 <= 0 || x0 >= w)
    return;
  fill_rect(img, w, h, x0, y0, x1, y1, c);
}

void draw_tick(TimelineCommands* img, int64_t w, int64_t h, Clock::time_point t,
               Clock::time_point origin, double window_s, int64_t y0, int64_t y1,
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
  if (file_index < 0)
    return {{50, 140, 190}};
  return k[static_cast<size_t>(file_index) % k.size()];
}

int span_file_at(const std::vector<radar::CollectionSpan>& spans, Clock::time_point t) {
  int idx = -1;
  bool have = false;
  Clock::time_point start{};
  for (const auto& span : spans) {
    if (span.dwell_start > t)
      continue;
    if (!have || span.dwell_start >= start) {
      have = true;
      start = span.dwell_start;
      idx = span.file_index;
    }
  }
  return idx;
}

std::string format_tenths_s(double s) {
  const int scaled = static_cast<int>(std::llround(std::abs(s) * 10.0));
  return std::to_string(scaled / 10) + "." + std::to_string(scaled % 10) + "s";
}

enum class LagBand : int { kUnknown = 0, kOnTime = 1, kBehind = 2, kAhead = 3 };

LagBand step_lag_band(LagBand cur, double behind) {
  if (!std::isfinite(behind))
    return cur;
  const double mag = std::abs(behind);
  if (cur == LagBand::kUnknown) {
    if (mag < 0.15)
      return LagBand::kOnTime;
    return behind > 0.0 ? LagBand::kBehind : LagBand::kAhead;
  }
  if (cur == LagBand::kOnTime) {
    if (behind >= 0.2)
      return LagBand::kBehind;
    if (behind <= -0.2)
      return LagBand::kAhead;
    return LagBand::kOnTime;
  }
  if (cur == LagBand::kBehind) {
    if (behind <= -0.2)
      return LagBand::kAhead;
    if (behind <= 0.05)
      return LagBand::kOnTime;
    return LagBand::kBehind;
  }
  if (behind >= 0.2)
    return LagBand::kBehind;
  if (behind >= -0.05)
    return LagBand::kOnTime;
  return LagBand::kAhead;
}

std::string format_lag_band(LagBand band, double behind) {
  if (band == LagBand::kUnknown || band == LagBand::kOnTime)
    return "on time";
  if (band == LagBand::kBehind)
    return "+" + format_tenths_s(behind) + " behind";
  return format_tenths_s(-behind) + " ahead";
}

std::array<uint8_t, 3> lag_band_color(LagBand band, double behind) {
  if (band == LagBand::kBehind && behind > 0.2)
    return {{255, 80, 80}};
  if (band == LagBand::kBehind)
    return {{255, 180, 60}};
  return {{80, 200, 80}};
}

std::string format_latency_ms(double s) {
  if (!std::isfinite(s) || s < 0.0)
    return {};
  if (s < 1.0) {
    const int ms = static_cast<int>(std::llround(s * 1000.0));
    return std::to_string(ms) + "ms";
  }
  return format_tenths_s(s);
}

std::string fit_label(const std::string& name, int64_t max_w, int scale) {
  if (max_w <= 0)
    return {};
  if (plot_text_width_5x7(name, scale) <= max_w)
    return name;
  std::string s = name;
  while (!s.empty() && plot_text_width_5x7(s, scale) > max_w)
    s.pop_back();
  return s;
}

std::string pulse_count(int64_t count) {
  auto text = std::to_string(count);
  for (auto position = static_cast<int64_t>(text.size()) - 3; position > 0; position -= 3)
    text.insert(static_cast<std::size_t>(position), ",");
  return text;
}

std::string next_image_eta(const radar::sar_progress::View& view) {
  if (view.image_complete)
    return "Image ready";
  if (!view.eta_s)
    return "Next image: Estimating...";
  // Rounded approximate wall time, not a promised countdown to GPU completion.
  const auto seconds = static_cast<int64_t>(std::ceil(std::min(*view.eta_s, 359999.0)));
  if (seconds >= 3600)
    return "Next image: ~" + std::to_string(seconds / 3600) + "h " +
           std::to_string((seconds % 3600) / 60) + "m";
  if (seconds >= 60)
    return "Next image: ~" + std::to_string(seconds / 60) + "m " + std::to_string(seconds % 60) +
           "s";
  return "Next image: ~" + std::to_string(seconds) + "s";
}

void draw_pulse_progress(TimelineCommands* img, int64_t width, int64_t height,
                         const radar::sar_progress::View& view,
                         const std::vector<std::string>& names, bool compact) {
  const std::array<uint8_t, 3> detail{180, 194, 207}, received{75, 205, 220}, formed{115, 200, 150};
  const auto fit = [](std::string text, int64_t pixels) {
    // Native 14px text advances about 7.5px per character; leave a margin.
    const auto chars = static_cast<std::size_t>(std::max(int64_t{0}, pixels / 8));
    if (text.size() > chars)
      text = chars >= 3 ? text.substr(0, chars - 3) + "..." : "";
    return text;
  };
  const auto row = [&](const std::string& label, int64_t value, int64_t y,
                       const std::array<uint8_t, 3>& color) {
    const auto count = view.valid ? pulse_count(value) + "/" + pulse_count(view.total) : "--";
    const int64_t bar_x = compact ? std::max(int64_t{180}, width - 80) : 8;
    const int64_t bar_y = compact ? y + 5 : y + 18;
    const int64_t bar_w = std::max(int64_t{1}, width - bar_x - 8);
    draw_text(img, width, height, 8, y, fit(label + " " + count, compact ? bar_x - 16 : width - 16),
              color, 2);
    fill_rect(img, width, height, bar_x, bar_y, bar_x + bar_w, bar_y + 4, {40, 49, 58});
    if (view.total > 0)
      fill_rect(img, width, height, bar_x, bar_y, bar_x + bar_w * value / view.total, bar_y + 4,
                color);
  };
  if (compact) {
    row("Received", view.received, 0, received);
    row("Formed", view.formed, 14, formed);
    draw_text(img, width, height, 8, 28, fit(next_image_eta(view), width - 16), detail, 2);
  } else {
    std::string name = "Pulse progress";
    if (view.valid && view.file_index >= 0 &&
        static_cast<std::size_t>(view.file_index) < names.size())
      name = names[view.file_index];
    draw_text(img, width, height, 8, 3, fit(name, width - 16), detail, 2);
    draw_text(img, width, height, 8, 23, fit(next_image_eta(view), width - 16), detail, 2);
    row("Received", view.received, 46, received);
    row("Formed", view.formed, 78, formed);
    fill_rect(img, width, height, width - 1, 0, width, height, {45, 55, 65});
  }
}

}  // namespace

TimelineRasterOp::TimelineRasterOp(std::shared_ptr<radar::FormationHud> hud,
                                   std::vector<std::string> scene_names, bool show_playback_lag,
                                   double rate, int64_t width, int64_t height,
                                   std::shared_ptr<FrameProgress> progress)
: hud_(std::move(hud)),
  scene_names_(std::move(scene_names)),
  show_playback_lag_(show_playback_lag),
  progress_(std::move(progress)),
  period_(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / rate))),
  width_(width),
  height_(height) {
  if (!std::isfinite(rate) || rate <= 0.0 || rate > 1000.0 || width < 1 || height < 1)
    throw std::invalid_argument("Invalid timeline dimensions/rate");
  if (!hud_)
    throw std::invalid_argument("TimelineRasterOp: FormationHud mailbox is required");
}

void TimelineRasterOp::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation rgba{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 3U};
  spec.output(frame)
      .tensor_field(schema::ImageTensorFields::data)
      .produces_tensor(pooled_tensor_output(
          rgba, 3U, static_cast<uint64_t>(width_) * static_cast<uint64_t>(height_) * 4U));
  constexpr TensorRepresentation text{.memory_kind = MemoryKind::kHost,
                                      .dtype = DLDataType{kDLUInt, 8U, 1U},
                                      .rank = 2U,
                                      .strides = TensorStridePolicy::kContiguousOnly};
  spec.output(labels).produces_tensor(pooled_tensor_output(text, 2U, kTextRows * kTextBytes));
}

Contract TimelineRasterOp::contract() const {
  Contract result;
  result.trigger(OnClock{.period = period_});
  return result;
}

void TimelineRasterOp::start() {
  started_at_ = Clock::now();
  pulse_progress_ = hud_->progress();
  if (cudaMalloc(&commands_, kMaxRects * sizeof(Rect)) != cudaSuccess)
    throw std::runtime_error("Timeline command allocation failed");
}
void TimelineRasterOp::stop() {
  if (commands_)
    (void)cudaFree(commands_);
  commands_ = nullptr;
}

expected<void, Error> TimelineRasterOp::compute(ExecutionContext& context) {
  const auto snap = hud_->snapshot();
  const auto now = Clock::now();
  const bool starting = progress_ && progress_->capture_ns.load() < 0;
  const bool hud_due = !hud_held_ || std::chrono::duration<double>(now - hud_at_).count() >= 1.0;
  if (pulse_progress_ && hud_due)
    pulse_view_ = pulse_progress_->snapshot(now);
  if (starting && hud_due) {
    const std::string phase = snap.bp.emit_at       ? "Preparing display"
                              : snap.reader_stopped ? "Forming first image"
                              : snap.last_packet_at ? "Receiving collection"
                                                    : "Starting up";
    startup_line_ =
        phase + "  " +
        std::to_string(
            std::chrono::duration_cast<std::chrono::seconds>(now - started_at_).count()) +
        "s elapsed";
  }
  const auto origin =
      now - std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(kWindowS));
  if (snap.bp.emit_at.has_value() &&
      (!last_seen_emit_.has_value() || *snap.bp.emit_at != *last_seen_emit_)) {
    last_seen_emit_ = snap.bp.emit_at;
    ImageMark mark;
    mark.at = *snap.bp.emit_at;
    Clock::time_point color_at = *snap.bp.emit_at;
    const auto packet_at =
        snap.bp.last_packet_at.has_value() ? snap.bp.last_packet_at : snap.last_packet_at;
    if (packet_at.has_value()) {
      color_at = *packet_at;
      mark.latency_s = std::chrono::duration<double>(*snap.bp.emit_at - *packet_at).count();
    }
    mark.file_index = span_file_at(snap.spans, color_at);
    if (image_emits_.size() >= 256)
      image_emits_.erase(image_emits_.begin());
    image_emits_.push_back(mark);
  }
  const auto keep = std::find_if(image_emits_.begin(), image_emits_.end(), [&](const ImageMark& m) {
    return m.at >= origin;
  });
  image_emits_.erase(image_emits_.begin(), keep);

  const bool compact_progress = pulse_progress_ && width_ < 620;
  const int64_t progress_width = pulse_progress_ && !compact_progress ? 264 : 0;
  const int64_t progress_height = compact_progress ? 42 : 0;
  const int64_t w = width_ - progress_width;
  const int64_t h = height_ - progress_height;
  TimelineCommands img;
  const std::array<uint8_t, 3> bg{16, 16, 16};
  fill_rect(&img, w, h, 0, 0, w, h, bg);

  const std::array<uint8_t, 3> quiet_c{90, 90, 90};
  const std::array<uint8_t, 3> fg{220, 220, 220};
  const std::array<uint8_t, 3> axis_c{80, 80, 80};
  const std::array<uint8_t, 3> head_c{255, 80, 80};

  constexpr int kScale = 2;
  const int64_t label_h = kFont5x7Rows * kScale;
  const int64_t top = 4;
  const int64_t lane_gap = 3;
  const int64_t axis_gap = 4;
  const int64_t lab_gap = 6;
  const int64_t bot = 4;
  const int64_t axis_block = axis_gap + 1 + lab_gap + label_h + bot;
  const int64_t label_gap = 3;
  const int64_t usable =
      std::max(int64_t{4}, h - top - lane_gap - axis_block - 2 * (label_h + label_gap));
  const int64_t plat_h = usable / 2;
  const int64_t plat_label_y = top;
  const int64_t plat_y0 = plat_label_y + label_h + label_gap;
  const int64_t plat_y1 = plat_y0 + plat_h;
  const int64_t prod_label_y = plat_y1 + lane_gap;
  const int64_t prod_y0 = prod_label_y + label_h + label_gap;
  const int64_t prod_y1 = prod_y0 + (usable - plat_h);
  const int64_t axis_y = prod_y1 + axis_gap;
  fill_rect(&img, w, h, 0, plat_y0, w, plat_y1, std::array<uint8_t, 3>{28, 28, 28});
  fill_rect(&img, w, h, 0, prod_y0, w, prod_y1, std::array<uint8_t, 3>{28, 28, 28});

  const int64_t hud_slot = 14 * kFont5x7Advance * kScale;
  auto draw_hud = [&](int64_t y0, int64_t y1, const std::string& lab,
                      const std::array<uint8_t, 3>& color) {
    if (lab.empty())
      return;
    const int64_t pad = 4;
    const int64_t tx = w - 8 - hud_slot;
    const int64_t ty = y0 + (y1 - y0 - label_h) / 2;
    fill_rect(&img, w, h, tx - pad, y0, w - 2, y1, bg);
    draw_text(&img, w, h, tx, ty, lab, color, kScale);
  };

  for (const auto& span : snap.spans) {
    const auto dwell_end = span.dwell_end.value_or(now);
    const auto dwell_c = collect_color(span.file_index);
    draw_bar(&img, w, h, span.dwell_start, dwell_end, origin, kWindowS, plat_y0, plat_y1, dwell_c);
    if (span.quiet_start.has_value()) {
      const auto quiet_end = span.quiet_end.value_or(now);
      draw_bar(&img, w, h, *span.quiet_start, quiet_end, origin, kWindowS, plat_y0, plat_y1,
               quiet_c);
    }
    std::string name;
    if (!scene_names_.empty() && span.file_index >= 0) {
      const size_t idx = static_cast<size_t>(span.file_index) % scene_names_.size();
      name = scene_names_[idx];
    }
    if (!name.empty()) {
      const int64_t x0 = time_to_x(span.dwell_start, origin, kWindowS, w);
      const int64_t x1 = time_to_x(dwell_end, origin, kWindowS, w);
      const int64_t vis0 = std::max(int64_t{0}, x0);
      const int64_t vis1 = std::min(w, x1);
      name = fit_label(name, vis1 - vis0 - 8, kScale);
      if (!name.empty()) {
        draw_text(&img, w, h, vis0 + 4, plat_label_y, name, fg, kScale);
      }
    }
  }
  const int64_t lat_y = prod_label_y;
  int64_t last_label_right = -8;
  for (const auto& mark : image_emits_) {
    const auto tick_c = collect_color(mark.file_index);
    draw_tick(&img, w, h, mark.at, origin, kWindowS, prod_y0, prod_y1, tick_c);
    if (starting || !mark.latency_s.has_value())
      continue;
    const std::string lab = format_latency_ms(*mark.latency_s);
    if (lab.empty())
      continue;
    const int64_t x = time_to_x(mark.at, origin, kWindowS, w);
    if (x < 0 || x >= w)
      continue;
    const int64_t tw = plot_text_width_5x7(lab, kScale);
    int64_t tx = x + 4;
    if (tx + tw > w - 4)
      tx = x - 4 - tw;
    // Off-screen or overlapping the previous label: keep the tick, skip the
    // text. Pinning to x=2 made the number sit on the left edge while the
    // tick scrolled away (continuous strip parked this HUD for the same overlap).
    if (tx < 2 || tx + tw > w - 2 || tx < last_label_right + 4)
      continue;
    draw_text(&img, w, h, tx, lat_y, lab, tick_c, kScale);
    last_label_right = tx + tw;
  }

  if (starting && !pulse_progress_)
    draw_text(&img, w, h, 4, prod_label_y, fit_label(startup_line_, w - 8, kScale),
              std::array<uint8_t, 3>{145, 180, 195}, kScale);
  fill_rect(&img, w, h, 0, axis_y, w, axis_y + 1, axis_c);
  const int n_ticks = 5;
  for (int i = 0; i <= n_ticks; ++i) {
    const double frac = static_cast<double>(i) / static_cast<double>(n_ticks);
    const int64_t x = static_cast<int64_t>(std::llround(frac * static_cast<double>(w)));
    fill_rect(&img, w, h, x, axis_y - 2, x + 1, axis_y + 3, axis_c);
    std::string lab;
    if (i == n_ticks) {
      lab = "now";
    } else {
      const double ago = kWindowS * (1.0 - frac);
      lab = std::to_string(static_cast<int>(std::llround(ago))) + "s";
    }
    const int64_t tw = plot_text_width_5x7(lab, kScale);
    int64_t tx = x - tw / 2;
    if (tx < 2)
      tx = 2;
    if (tx + tw > w - 2)
      tx = w - 2 - tw;
    draw_text(&img, w, h, tx, axis_y + lab_gap, lab, fg, kScale);
  }
  fill_rect(&img, w, h, w - 2, 0, w, h, head_c);

  if (hud_due) {
    hud_at_ = now;
    hud_held_ = true;
    if (show_playback_lag_ && snap.playback.valid && snap.playback.paced &&
        std::isfinite(snap.playback.behind_s)) {
      const auto band = step_lag_band(static_cast<LagBand>(lag_band_), snap.playback.behind_s);
      lag_band_ = static_cast<int>(band);
      lag_hud_ = format_lag_band(band, snap.playback.behind_s);
      lag_hud_fg_ = lag_band_color(band, snap.playback.behind_s);
    }
  }
  if (show_playback_lag_)
    draw_hud(plat_y0, plat_y1, lag_hud_, lag_hud_fg_);

  // Shift only bounded commands/text, never pixels or the completed SAR image.
  if (pulse_progress_) {
    for (auto& rect : img.rectangles) {
      rect.x0 = std::clamp(rect.x0, int64_t{0}, w) + progress_width;
      rect.x1 = std::clamp(rect.x1, int64_t{0}, w) + progress_width;
      rect.y0 = std::clamp(rect.y0, int64_t{0}, h) + progress_height;
      rect.y1 = std::clamp(rect.y1, int64_t{0}, h) + progress_height;
    }
    for (std::size_t row = 0; row < img.rows; ++row) {
      visualization::TextRunHeader header;
      auto* record = img.text.data() + row * kTextBytes;
      std::memcpy(&header, record, sizeof(header));
      header.x += progress_width;
      header.y += progress_height;
      std::memcpy(record, &header, sizeof(header));
    }
    draw_pulse_progress(&img, compact_progress ? width_ : progress_width, height_, pulse_view_,
                        scene_names_, compact_progress);
  }
  auto loan = frame.allocate_tensor(TensorLoanRequest{
      .shape = std::vector<int64_t>{height_, width_, 4}, .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!loan)
    return make_unexpected(std::move(loan).error());
  auto writer = loan->write(context.cuda_stream());
  if (!writer)
    return make_unexpected(std::move(writer).error());
  if (cudaMemcpyAsync(commands_, img.rectangles.data(), img.rectangles.size() * sizeof(Rect),
                      cudaMemcpyHostToDevice, context.cuda_stream().get()) != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "Timeline commands H2D failed"});
  // The bounded host command vector may be destroyed after this callback.
  // Finish its H2D read before releasing it; pixels remain GPU-rendered.
  if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "Timeline command transfer failed"});
  render_timeline<<<(width_ * height_ + 255) / 256, 256, 0, context.cuda_stream().get()>>>(
      static_cast<uint8_t*>(writer->data()), width_, height_, static_cast<const Rect*>(commands_),
      img.rectangles.size());
  if (cudaPeekAtLastError() != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "Timeline GPU raster failed"});
  HOLOSCAN_RETURN_IF_ERROR(std::move(*writer).commit());
  auto text = labels.allocate_tensor(
      TensorLoanRequest{.shape = std::array<int64_t, 2>{kTextRows, kTextBytes},
                        .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!text)
    return make_unexpected(std::move(text).error());
  HOLOSCAN_RETURN_IF_ERROR(text->write_host_and_commit(
      [&](const TensorOutputWriteGuard& guard) -> expected<void, Error> {
        std::memcpy(guard.data(), img.text.data(), img.text.size());
        return {};
      }));
  HOLOSCAN_RETURN_IF_ERROR(
      labels.emit(std::move(*text), EmitOptions{.capture_time = context.activation_time()}));
  schema::ImageT descriptor{};
  descriptor.height = static_cast<std::int32_t>(height_);
  descriptor.width = static_cast<std::int32_t>(width_);
  descriptor.encoding = schema::ImageEncoding_RGBA8;
  descriptor.color_space = schema::ColorSpace_UNSPECIFIED;
  return frame.emit_tensor(std::move(*loan), descriptor,
                           EmitOptions{.capture_time = context.activation_time()});
}

}  // namespace holoscan::radar::app::sar
