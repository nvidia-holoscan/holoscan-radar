// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_app/sar/timeline_raster.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
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
#include "radar_app/sar/hud_format.hpp"
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
  float line_x{0}, slope{0}, half_width{0};
  bool operator==(const Rect&) const = default;
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

void draw_text_px(TimelineCommands* img, int64_t x, int64_t y, const std::string& text,
                  const std::array<uint8_t, 3>& color, float pixels) {
  if (img->rows >= kTextRows)
    return;
  auto* row = img->text.data() + img->rows++ * kTextBytes;
  const visualization::TextRunHeader header{
      .x = static_cast<float>(x),
      .y = static_cast<float>(y),
      .size_px = pixels,
      .color = 0xff000000U | (static_cast<uint32_t>(color[2]) << 16U) |
               (static_cast<uint32_t>(color[1]) << 8U) | color[0]};
  std::memcpy(row, &header, sizeof(header));
  std::memcpy(row + sizeof(header), text.data(),
              std::min(text.size(), kTextBytes - sizeof(header) - 1));
}

// Merge only drawing coverage. Original batch pairs were already matched in
// the diagnostics mailbox; sorting independent lane events cannot match them.
void draw_intervals(TimelineCommands* img, int64_t w, int64_t h,
                    std::vector<std::pair<int64_t, int64_t>> intervals, int64_t y0, int64_t y1,
                    const std::array<uint8_t, 3>& color) {
  std::sort(intervals.begin(), intervals.end());
  int64_t left = -1, right = -1;
  const auto emit = [&] {
    if (left >= 0)
      fill_rect(img, w, h, left, y0, right, y1, color);
  };
  for (const auto& interval : intervals) {
    if (left < 0 || interval.first > right) {
      emit();
      left = interval.first;
      right = interval.second;
    } else {
      right = std::max(right, interval.second);
    }
  }
  emit();
}

// One bounded command per connection. The GPU evaluates a thin line directly,
// rather than expanding a diagonal into many host-generated rectangles.
void draw_connection(TimelineCommands* img, int64_t left, int64_t right, int64_t x0, int64_t y0,
                     int64_t x1, int64_t y1, const std::array<uint8_t, 3>& color) {
  if (img->rectangles.size() == kMaxRects)
    return;
  const float slope = static_cast<float>(x1 - x0) / static_cast<float>(y1 - y0);
  img->rectangles.push_back({left, y0, right, y1 + 1, color[0], color[1], color[2],
                             static_cast<float>(x0), slope,
                             0.55F * std::sqrt(1.0F + slope * slope)});
}

__global__ void render_timeline(uint8_t* output, int64_t width, int64_t height,
                                const Rect* rectangles, size_t count, int64_t visible_width,
                                int64_t visible_height) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= width * height)
    return;
  const int64_t x = pixel % width, y = pixel / width;
  uint8_t r = 0, g = 0, b = 0;
  for (size_t i = 0; x < visible_width && y < visible_height && i < count; ++i) {
    const auto rect = rectangles[i];
    if (x >= rect.x0 && x < rect.x1 && y >= rect.y0 && y < rect.y1) {
      if (rect.half_width > 0 &&
          fabsf(static_cast<float>(x) -
                (rect.line_x + static_cast<float>(y - rect.y0) * rect.slope)) > rect.half_width)
        continue;
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

// This is a checkpoint, not a live countdown: the GPU may be busy between
// presentations. The monitor cannot distinguish auxiliary BP from autofocus,
// so do not infer a more specific active stage from received pulse counts.
std::string image_checkpoint(const radar::sar_progress::View& view) {
  if (view.image_complete)
    return "Image ready";
  return view.valid && view.received > 0 ? "Forming image" : "Receiving pulses";
}

void draw_pulse_progress(TimelineCommands* img, int64_t width, int64_t height,
                         const radar::sar_progress::View& view,
                         const std::vector<std::string>& names, bool compact,
                         const radar::AutofocusHudStatus& autofocus) {
  const std::array<uint8_t, 3> detail{205, 217, 229}, formed{115, 200, 150}, received{75, 205, 220};
  const auto fit = [](std::string text, int64_t width, float pixels = 14) {
    const auto chars = static_cast<std::size_t>(std::max(0.0F, width / (pixels * 0.6F)));
    if (text.size() > chars)
      text = chars >= 3 ? text.substr(0, chars - 3) + "..." : "";
    return text;
  };
  const auto percent = [&](int64_t value) {
    return view.valid && view.total > 0
               ? std::to_string(static_cast<int>(100.0 * std::clamp(value, int64_t{0}, view.total) /
                                                 view.total)) +
                     "%"
               : "--";
  };
  std::string title = "waiting";
  if (view.valid && view.file_index >= 0 &&
      static_cast<std::size_t>(view.file_index) < names.size())
    title = names[view.file_index];
  const bool has_stages = autofocus.outcome != radar::AutofocusHudOutcome::kDisabled;
  const bool has_outcome = has_stages && autofocus.outcome != radar::AutofocusHudOutcome::kPending;
  const std::string detail_line = has_outcome  ? format_autofocus_status(autofocus, true, true)
                                  : has_stages ? "Last reported stage"
                                               : "Updates on completion";
  std::string work = "--";
  if (view.giga_pixel_pulses) {
    char digits[48]{};
    const auto formatted = std::to_chars(digits, digits + sizeof(digits), *view.giga_pixel_pulses,
                                         std::chars_format::fixed, 1);
    if (formatted.ec == std::errc())
      work.assign(digits, formatted.ptr);
  }
  const std::string work_line = "BP work: " + work + " Gpx*pulses";
  const auto bar = [&](int64_t bar_x, int64_t bar_y, int64_t bar_height) {
    const int64_t bar_w = std::max(int64_t{1}, width - bar_x - 8);
    fill_rect(img, width, height, bar_x, bar_y, bar_x + bar_w, bar_y + bar_height, {40, 49, 58});
    if (view.valid && view.total > 0) {
      const auto filled =
          static_cast<int64_t>(static_cast<double>(bar_w) *
                               std::clamp(view.formed, int64_t{0}, view.total) / view.total);
      fill_rect(img, width, height, bar_x, bar_y, bar_x + filled, bar_y + bar_height, formed);
      const auto admitted =
          static_cast<int64_t>(static_cast<double>(bar_w) *
                               std::clamp(view.received, int64_t{0}, view.total) / view.total);
      const auto marker = bar_x + std::min(admitted, std::max(int64_t{0}, bar_w - 2));
      // Draw last and beyond the track height so the marker remains visible
      // when both frontiers meet, including zero and full completion.
      fill_rect(img, width, height, marker, bar_y - 2, marker + 2, bar_y + bar_height + 2,
                received);
    }
  };
  const auto timing_text = has_stages ? format_autofocus_stage(autofocus) : image_checkpoint(view);
  const std::array<uint8_t, 3> primary{230, 237, 243}, warning{255, 180, 60};
  const auto timing_color = autofocus.outcome == radar::AutofocusHudOutcome::kDeclined ||
                                    autofocus.outcome == radar::AutofocusHudOutcome::kUnavailable
                                ? warning
                            : view.image_complete ? formed
                                                  : primary;
  const auto progress_labels = [&](int64_t y) {
    const auto receipt = "Received " + percent(view.received);
    const auto receipt_width = static_cast<int64_t>(std::ceil(receipt.size() * 12 * 0.6));
    const auto receipt_x = std::max(width / 2, width - 8 - receipt_width);
    draw_text_px(img, 8, y, fit("Last completed " + percent(view.formed), receipt_x - 16, 12),
                 formed, 12);
    draw_text_px(img, receipt_x, y, receipt, received, 12);
  };
  if (compact && has_stages) {
    // Keep the history's full 56px below. A reported outcome takes the
    // redundant scene-title row (also present in the completed-image card).
    draw_text_px(img, 8, 0, fit(timing_text, width - 16, 12), primary, 12);
    draw_text_px(img, 8, 13, fit(has_outcome ? detail_line : title, width - 16, 14),
                 has_outcome ? timing_color : detail, 14);
    draw_text_px(img, 8, 27, fit(work_line, width - 16, 14), detail, 14);
    progress_labels(41);
    bar(8, 53, 3);
  } else if (compact) {
    const int64_t split = width / 2;
    draw_text_px(img, 8, 0, fit(timing_text, split - 16, 12), timing_color, 12);
    draw_text_px(img, split, 0, fit(detail_line, width - split - 8, 12), detail, 12);
    const int64_t work_width = static_cast<int64_t>(std::ceil(work_line.size() * 14 * 0.6)) + 8;
    const int64_t work_x = std::max(int64_t{8}, width - 8 - work_width);
    draw_text_px(img, 8, 14, fit(title, work_x - 16, 14), detail, 14);
    draw_text_px(img, work_x, 14, fit(work_line, width - work_x - 8, 14), detail, 14);
    progress_labels(30);
    bar(8, 46, 4);
  } else {
    draw_text_px(img, 8, 25, fit(title, width - 16, 14), detail, 14);
    draw_text_px(img, 8, 45, fit(work_line, width - 16, 14), detail, 14);
    const float headline_px = has_stages ? 16 : 18;
    draw_text_px(img, 8, 2, fit(timing_text, width - 16, headline_px),
                 has_stages ? primary : timing_color, headline_px);
    progress_labels(63);
    bar(8, 82, 6);
    draw_text_px(img, 8, 96, fit(detail_line, width - 16, 12), has_stages ? timing_color : detail,
                 12);
    fill_rect(img, width, height, width - 1, 0, width, height, {45, 55, 65});
  }
}

}  // namespace

struct TimelineRasterOp::RenderCache {
  TimelineCommands commands;
  int64_t width{0}, height{0};
  Clock::time_point history_end{};
};

TimelineRasterOp::~TimelineRasterOp() = default;

TimelineRasterOp::TimelineRasterOp(std::shared_ptr<radar::FormationHud> hud,
                                   std::vector<std::string> scene_names, bool show_playback_lag,
                                   double rate, int64_t width, int64_t height,
                                   std::shared_ptr<FrameProgress> progress,
                                   std::shared_ptr<viz::FrameViewport> viewport)
: hud_(std::move(hud)),
  scene_names_(std::move(scene_names)),
  show_playback_lag_(show_playback_lag),
  progress_(std::move(progress)),
  period_(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / rate))),
  width_(width),
  height_(height),
  viewport_(std::move(viewport)) {
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
  rendered_.reset();
  pulse_progress_ = hud_->progress();
  if (cudaMalloc(&commands_, kMaxRects * sizeof(Rect)) != cudaSuccess)
    throw std::runtime_error("Timeline command allocation failed");
}
void TimelineRasterOp::stop() {
  rendered_.reset();
  if (commands_)
    (void)cudaFree(commands_);
  commands_ = nullptr;
}

expected<void, Error> TimelineRasterOp::compute(ExecutionContext& context) {
  const auto [layout_width, layout_height] =
      viewport_ ? viewport_->get() : std::pair{static_cast<int>(width_), static_cast<int>(height_)};
  const auto snap = hud_->snapshot();
  const auto now = Clock::now();
  const auto batch_events = pulse_progress_ ? pulse_progress_->batch_history()
                                            : std::vector<radar::sar_progress::BatchEvent>{};

  const auto batch_spans = pulse_progress_ ? pulse_progress_->batch_spans()
                                           : std::vector<radar::sar_progress::BatchSpan>{};
  // Event time, not poll time: held history remains fixed during long GPU
  // work and after EOF. Exact spans may be newer than the coalesced snapshot.
  std::optional<Clock::time_point> latest;
  const auto observe = [&](Clock::time_point at) {
    latest = latest ? std::max(*latest, at) : at;
  };
  for (const auto& event : batch_events)
    observe(event.at);
  for (const auto& span : batch_spans) {
    observe(span.received);
    if (span.processed)
      observe(*span.processed);
  }
  const auto history_end = latest.value_or(Clock::time_point{});
  const bool hud_due = !hud_held_ || !rendered_ || rendered_->history_end != history_end ||
                       std::chrono::duration<double>(now - hud_at_).count() >= 1.0;
  if (pulse_progress_ && hud_due)
    pulse_view_ = pulse_progress_->snapshot(now);
  const auto origin = history_end - std::chrono::duration_cast<Clock::duration>(
                                        std::chrono::duration<double>(kWindowS));
  const bool compact_progress = pulse_progress_ && layout_width < 620;
  const int64_t progress_width = pulse_progress_ && !compact_progress ? 264 : 0;
  const int64_t progress_height = compact_progress ? std::min(56, layout_height / 2) : 0;
  const int64_t w = layout_width - progress_width;
  const int64_t h = layout_height - progress_height;
  TimelineCommands img;
  const std::array<uint8_t, 3> bg{0, 0, 0}, fg{205, 217, 229}, track{53, 64, 76};
  const std::array<uint8_t, 3> connector{75, 94, 113};
  const std::array<uint8_t, 3> received_c{75, 205, 220}, processed_c{115, 200, 150};
  fill_rect(&img, w, h, 0, 0, w, h, bg);
  const bool compact_history = h < 80;
  // Keep matching glyph metrics across lane titles and time bookends.
  const float font = compact_history ? 14 : 16;
  const int64_t gutter = compact_history ? 40 : 64;
  const int64_t plot_left = std::min(gutter, std::max(int64_t{0}, w - 2));
  constexpr std::string_view latest_label = "Latest batch";
  const auto latest_width = static_cast<int64_t>(std::ceil(latest_label.size() * font * 7 / 13));
  const int64_t right_gutter = latest_width + 16;
  const int64_t plot_right = std::max(plot_left + 1, w - right_gutter);
  const int64_t plot_width = plot_right - plot_left;
  const int64_t received_y = compact_history ? 18 : 36;
  const int64_t processed_y = compact_history ? 34 : 70;
  const int64_t axis_y = (received_y + processed_y - static_cast<int64_t>(font)) / 2;
  const auto raw_x_at = [&](Clock::time_point at) {
    return plot_left + time_to_x(at, origin, kWindowS, plot_width);
  };
  const auto x_at = [&](Clock::time_point at) {
    return std::clamp(raw_x_at(at), plot_left, plot_right - 1);
  };
  std::vector<std::pair<int64_t, int64_t>> connections;
  for (const auto& span : batch_spans) {
    // A pending batch has no known completion endpoint. Keep its receipt tick
    // without drawing a fictitious completion at the right endpoint.
    if (!span.processed || *span.processed < origin || *span.processed > history_end ||
        span.received > history_end)
      continue;
    connections.emplace_back(raw_x_at(span.received), raw_x_at(*span.processed));
  }
  std::sort(connections.begin(), connections.end());
  connections.erase(std::unique(connections.begin(), connections.end()), connections.end());
  // All event ticks remain. At dense rates, display representative real pairs
  // across history, bounded independently of the source batch rate.
  constexpr std::size_t kMaxConnections = 64;
  const auto count = std::min(connections.size(), kMaxConnections);
  for (std::size_t i = 0; i < count; ++i) {
    const auto index = count <= 1 ? 0 : i * (connections.size() - 1) / (count - 1);
    const auto [received_x, processed_x] = connections[index];
    draw_connection(&img, plot_left, plot_right, received_x, received_y, processed_x, processed_y,
                    connector);
  }
  for (const auto y : {received_y, processed_y})
    fill_rect(&img, w, h, plot_left, y, plot_right, y + 1, track);
  std::array<std::vector<std::pair<int64_t, int64_t>>, 2> ticks;
  for (const auto& event : batch_events) {
    if (event.at < origin || event.at > history_end)
      continue;
    const auto x = x_at(event.at);
    ticks[event.processed ? 1 : 0].emplace_back(std::max(plot_left, x - 1),
                                                std::min(plot_right, x + 2));
  }
  draw_intervals(&img, w, h, std::move(ticks[0]), received_y - 4, received_y + 5, received_c);
  draw_intervals(&img, w, h, std::move(ticks[1]), processed_y - 2, processed_y + 3, processed_c);
  draw_text_px(&img, 8, compact_history ? 0 : 2, "Pulse batch received", fg, font);
  draw_text_px(&img, 8, compact_history ? 41 : 80, "Pulse batch processed", fg, font);
  draw_text_px(&img, 8, axis_y, "-" + std::to_string(static_cast<int>(kWindowS)) + "s", fg, font);
  draw_text_px(&img, w - 8 - latest_width, axis_y, std::string(latest_label), fg, font);
  if (!compact_history) {
    const std::string caption = "Time relative to last batch update";
    const auto caption_width = static_cast<int64_t>(std::ceil(caption.size() * 12 * 7.0 / 13));
    draw_text_px(&img, std::max(int64_t{8}, (w - caption_width) / 2), h - 14, caption, fg, 12);
  }
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
  if (show_playback_lag_ && !lag_hud_.empty()) {
    const float lag_font = font;
    const float lag_advance = lag_font * 7 / 13;
    const auto title_end = 8 + 20 * font * 7 / 13;
    const auto chars = static_cast<std::size_t>(std::max(0.0F, (w - title_end - 24) / lag_advance));
    const auto text = lag_hud_.substr(0, chars);
    draw_text_px(&img, w - 8 - static_cast<int64_t>(text.size() * lag_advance),
                 compact_history ? 0 : 8, text, lag_hud_fg_, lag_font);
  }

  // Shift only bounded commands/text, never pixels or the completed SAR image.
  if (pulse_progress_) {
    for (auto& rect : img.rectangles) {
      rect.x0 = std::clamp(rect.x0, int64_t{0}, w) + progress_width;
      rect.x1 = std::clamp(rect.x1, int64_t{0}, w) + progress_width;
      rect.y0 = std::clamp(rect.y0, int64_t{0}, h) + progress_height;
      rect.y1 = std::clamp(rect.y1, int64_t{0}, h) + progress_height;
      rect.line_x += static_cast<float>(progress_width);
    }
    for (std::size_t row = 0; row < img.rows; ++row) {
      visualization::TextRunHeader header;
      auto* record = img.text.data() + row * kTextBytes;
      std::memcpy(&header, record, sizeof(header));
      header.x += progress_width;
      header.y += progress_height;
      std::memcpy(record, &header, sizeof(header));
    }
    draw_pulse_progress(&img, compact_progress ? layout_width : progress_width, layout_height,
                        pulse_view_, scene_names_, compact_progress, snap.autofocus);
  }
  // Polling still services completion proof and resize feedback. An identical
  // bounded command/text snapshot needs no new GPU frame or display download.
  if (rendered_ && rendered_->width == layout_width && rendered_->height == layout_height &&
      rendered_->commands.rectangles == img.rectangles && rendered_->commands.text == img.text) {
    rendered_->history_end = history_end;
    return {};
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
      img.rectangles.size(), layout_width, layout_height);
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
  if (viewport_)
    viewport_->drawn(context.activation_time().timestamp_ns, layout_width, layout_height);
  HOLOSCAN_RETURN_IF_ERROR(frame.emit_tensor(
      std::move(*loan), descriptor, EmitOptions{.capture_time = context.activation_time()}));
  if (!rendered_)
    rendered_ = std::make_unique<RenderCache>();
  rendered_->commands = std::move(img);
  rendered_->width = layout_width;
  rendered_->height = layout_height;
  rendered_->history_end = history_end;
  return {};
}

}  // namespace holoscan::radar::app::sar
