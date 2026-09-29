// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_app/viz/fmcw_dashboard_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_app/app/angle_presentation.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app::viz {
namespace {

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kInt64{kDLInt, 64U, 1U};
constexpr DLDataType kRgba8{kDLUInt, 8U, 1U};

__device__ unsigned int positive_float_bits(float value) {
  return __float_as_uint(fmaxf(value, 0.0F));
}

__global__ void reduce_positive_peak(const float* values, std::int64_t count,
                                     unsigned int* peak_bits) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && isfinite(values[i])) {
    atomicMax(peak_bits, positive_float_bits(values[i]));
  }
}

__device__ unsigned char channel(float value) {
  return static_cast<unsigned char>(fminf(fmaxf(value * 255.0F, 0.0F), 255.0F));
}

__global__ void power_to_rgba(const float* power, std::int64_t num_planes, std::int64_t source_rows,
                              std::int64_t source_cols, const float* peak, float dynamic_range_db,
                              std::int64_t canvas_width, int plot_x, int plot_y, int plot_width,
                              int plot_height, std::uint8_t* rgba) {
  const std::int64_t output_pixel =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t output_pixels = static_cast<std::int64_t>(plot_width) * plot_height;
  if (output_pixel >= output_pixels) {
    return;
  }
  const std::int64_t output_row = output_pixel / plot_width;
  const std::int64_t output_col = output_pixel % plot_width;
  const std::int64_t source_row = output_col * source_rows / plot_width;
  const std::int64_t source_col = (plot_height - 1 - output_row) * source_cols / plot_height;
  const std::int64_t source_pixel = source_row * source_cols + source_col;
  const std::int64_t source_pixels = source_rows * source_cols;
  float value = 0.0F;
  for (std::int64_t plane = 0; plane < num_planes; ++plane) {
    value = fmaxf(value, power[plane * source_pixels + source_pixel]);
  }
  const float reference = fmaxf(*peak, FLT_MIN);
  const float db = 10.0F * log10f(fmaxf(value, FLT_MIN) / reference);
  const float t = fminf(fmaxf(1.0F + db / dynamic_range_db, 0.0F), 1.0F);

  // Compact perceptual-style blue -> cyan -> yellow -> red ramp.
  const float red = fminf(fmaxf(3.0F * t - 1.0F, 0.0F), 1.0F);
  const float green = fminf(fmaxf(1.5F - fabsf(3.0F * t - 1.5F), 0.0F), 1.0F);
  const float blue = fminf(fmaxf(2.0F - 3.0F * t, 0.0F), 1.0F);
  const std::int64_t out = 4 * ((output_row + plot_y) * canvas_width + output_col + plot_x);
  rgba[out] = channel(red);
  rgba[out + 1] = channel(green);
  rgba[out + 2] = channel(blue);
  rgba[out + 3] = 255U;
}

__global__ void apply_annotation(const std::uint8_t* annotation, std::int64_t pixels,
                                 std::uint8_t* rgba) {
  const std::int64_t pixel = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel < pixels && annotation[4 * pixel + 3] != 0U) {
    rgba[4 * pixel] = annotation[4 * pixel];
    rgba[4 * pixel + 1] = annotation[4 * pixel + 1];
    rgba[4 * pixel + 2] = annotation[4 * pixel + 2];
    rgba[4 * pixel + 3] = annotation[4 * pixel + 3];
  }
}

expected<void, Error> cuda_failure(const char* who, const char* step) {
  return make_unexpected(Error{ErrorCode::kFailure, std::string(who) + ": " + step + " failed"});
}

expected<std::vector<std::int64_t>, Error> copy_int64_to_host(const Tensor& tensor,
                                                              cudaStream_t stream,
                                                              const char* name) {
  if (!dldatatype_equals(tensor.dtype(), kInt64) || !tensor.is_contiguous()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("FmcwDashboardOp.") + name + ": expected a compact int64 Tensor"});
  }
  const std::size_t count = static_cast<std::size_t>(tensor.nbytes()) / sizeof(std::int64_t);
  std::vector<std::int64_t> result(count);
  if (!result.empty() && cudaMemcpyAsync(result.data(), tensor.data(), tensor.nbytes(),
                                         cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure, std::string("FmcwDashboardOp.") + name +
                                                          ": device-to-host copy failed"});
  }
  return result;
}

expected<std::vector<float>, Error> copy_float_to_host(const Tensor& tensor, cudaStream_t stream,
                                                       const char* name) {
  if (!dldatatype_equals(tensor.dtype(), kFloat32) || !tensor.is_contiguous()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string("FmcwDashboardOp.") + name + ": expected a compact float32 Tensor"});
  }
  const std::size_t count = static_cast<std::size_t>(tensor.nbytes()) / sizeof(float);
  std::vector<float> result(count);
  if (!result.empty() && cudaMemcpyAsync(result.data(), tensor.data(), tensor.nbytes(),
                                         cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure, std::string("FmcwDashboardOp.") + name +
                                                          ": device-to-host copy failed"});
  }
  return result;
}

void validate_dashboard_inputs(const Tensor& detection_idx, const Tensor& points,
                               const Tensor& labels, const Tensor& sizes, const Tensor& range_bins,
                               const Tensor& velocity_bins, const Tensor& reduced_points,
                               bool show_angles) {
  const std::int64_t min_features = show_angles ? 4 : 2;
  if (detection_idx.shape().size() != 2U || detection_idx.shape()[1] < 2 ||
      points.shape().size() != 2U || points.shape()[1] < min_features ||
      points.shape()[0] != detection_idx.shape()[0] || labels.shape().size() != 1U ||
      labels.shape()[0] != detection_idx.shape()[0] || sizes.shape().size() != 1U ||
      range_bins.shape().size() != 1U || range_bins.shape()[0] <= 0 ||
      velocity_bins.shape().size() != 1U || velocity_bins.shape()[0] <= 0 ||
      reduced_points.shape().size() != 2U || reduced_points.shape()[0] != sizes.shape()[0] ||
      reduced_points.shape()[1] < min_features) {
    throw std::invalid_argument("FmcwDashboardOp: incompatible detection, cluster, or axis shapes");
  }
}

void validate_detection_inputs(const Tensor& detection_idx, const Tensor& points,
                               const Tensor& range_bins, const Tensor& velocity_bins,
                               bool show_angles) {
  const std::int64_t min_features = show_angles ? 4 : 2;
  if (detection_idx.shape().size() != 2U || detection_idx.shape()[1] < 2 ||
      points.shape().size() != 2U || points.shape()[1] < min_features ||
      points.shape()[0] != detection_idx.shape()[0] || range_bins.shape().size() != 1U ||
      range_bins.shape()[0] <= 0 || velocity_bins.shape().size() != 1U ||
      velocity_bins.shape()[0] <= 0) {
    throw std::invalid_argument("FmcwDashboardOp: incompatible raw detection or axis shapes");
  }
}

std::array<std::uint8_t, 7> glyph(char value) {
  switch (static_cast<char>(std::toupper(static_cast<unsigned char>(value)))) {
    case 'A':
      return {14, 17, 17, 31, 17, 17, 17};
    case 'B':
      return {30, 17, 17, 30, 17, 17, 30};
    case 'C':
      return {14, 17, 16, 16, 16, 17, 14};
    case 'D':
      return {30, 17, 17, 17, 17, 17, 30};
    case 'E':
      return {31, 16, 16, 30, 16, 16, 31};
    case 'F':
      return {31, 16, 16, 30, 16, 16, 16};
    case 'G':
      return {14, 17, 16, 23, 17, 17, 15};
    case 'H':
      return {17, 17, 17, 31, 17, 17, 17};
    case 'I':
      return {14, 4, 4, 4, 4, 4, 14};
    case 'J':
      return {7, 2, 2, 2, 18, 18, 12};
    case 'K':
      return {17, 18, 20, 24, 20, 18, 17};
    case 'L':
      return {16, 16, 16, 16, 16, 16, 31};
    case 'M':
      return {17, 27, 21, 21, 17, 17, 17};
    case 'N':
      return {17, 25, 21, 19, 17, 17, 17};
    case 'O':
      return {14, 17, 17, 17, 17, 17, 14};
    case 'P':
      return {30, 17, 17, 30, 16, 16, 16};
    case 'Q':
      return {14, 17, 17, 17, 21, 18, 13};
    case 'R':
      return {30, 17, 17, 30, 20, 18, 17};
    case 'S':
      return {15, 16, 16, 14, 1, 1, 30};
    case 'T':
      return {31, 4, 4, 4, 4, 4, 4};
    case 'U':
      return {17, 17, 17, 17, 17, 17, 14};
    case 'V':
      return {17, 17, 17, 17, 17, 10, 4};
    case 'W':
      return {17, 17, 17, 21, 21, 21, 10};
    case 'X':
      return {17, 17, 10, 4, 10, 17, 17};
    case 'Y':
      return {17, 17, 10, 4, 4, 4, 4};
    case 'Z':
      return {31, 1, 2, 4, 8, 16, 31};
    case '0':
      return {14, 17, 19, 21, 25, 17, 14};
    case '1':
      return {4, 12, 4, 4, 4, 4, 14};
    case '2':
      return {14, 17, 1, 2, 4, 8, 31};
    case '3':
      return {30, 1, 1, 14, 1, 1, 30};
    case '4':
      return {2, 6, 10, 18, 31, 2, 2};
    case '5':
      return {31, 16, 16, 30, 1, 1, 30};
    case '6':
      return {14, 16, 16, 30, 17, 17, 14};
    case '7':
      return {31, 1, 2, 4, 8, 8, 8};
    case '8':
      return {14, 17, 17, 14, 17, 17, 14};
    case '9':
      return {14, 17, 17, 15, 1, 1, 14};
    case '-':
      return {0, 0, 0, 31, 0, 0, 0};
    case '=':
      return {0, 31, 0, 31, 0, 0, 0};
    case '.':
      return {0, 0, 0, 0, 0, 12, 12};
    case '/':
      return {1, 2, 2, 4, 8, 8, 16};
    case '(':
      return {2, 4, 8, 8, 8, 4, 2};
    case ')':
      return {8, 4, 2, 2, 2, 4, 8};
    case '+':
      return {0, 4, 4, 31, 4, 4, 0};
    case ':':
      return {0, 12, 12, 0, 12, 12, 0};
    default:
      return {};
  }
}

void set_pixel(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height,
               std::int64_t x, std::int64_t y, const std::array<std::uint8_t, 4>& color) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return;
  }
  const std::size_t offset = 4U * static_cast<std::size_t>(y * width + x);
  std::copy(color.begin(), color.end(), image->begin() + offset);
}

void draw_text(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height,
               std::int64_t x, std::int64_t y, std::string_view text, int scale) {
  constexpr std::array<std::uint8_t, 4> kWhite{255U, 255U, 255U, 255U};
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto rows = glyph(text[i]);
    for (int row = 0; row < 7; ++row) {
      for (int col = 0; col < 5; ++col) {
        if ((rows[row] & (1U << (4 - col))) == 0U) {
          continue;
        }
        for (int sy = 0; sy < scale; ++sy) {
          for (int sx = 0; sx < scale; ++sx) {
            set_pixel(image, width, height,
                      x + static_cast<std::int64_t>(i) * 6 * scale + col * scale + sx,
                      y + row * scale + sy, kWhite);
          }
        }
      }
    }
  }
}

void draw_text_ccw(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height,
                   std::int64_t x, std::int64_t y, std::string_view text, int scale) {
  constexpr std::array<std::uint8_t, 4> kWhite{255U, 255U, 255U, 255U};
  const std::int64_t text_width = static_cast<std::int64_t>(text.size()) * 6 * scale;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const auto rows = glyph(text[i]);
    for (int row = 0; row < 7; ++row) {
      for (int col = 0; col < 5; ++col) {
        if ((rows[row] & (1U << (4 - col))) == 0U) {
          continue;
        }
        for (int sy = 0; sy < scale; ++sy) {
          for (int sx = 0; sx < scale; ++sx) {
            const std::int64_t normal_x =
                static_cast<std::int64_t>(i) * 6 * scale + col * scale + sx;
            set_pixel(image, width, height, x + row * scale + sy, y + text_width - 1 - normal_x,
                      kWhite);
          }
        }
      }
    }
  }
}

void draw_line(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height, int x0,
               int y0, int x1, int y1, const std::array<std::uint8_t, 4>& color) {
  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  while (true) {
    set_pixel(image, width, height, x0, y0, color);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int doubled = 2 * error;
    if (doubled >= dy) {
      error += dy;
      x0 += sx;
    }
    if (doubled <= dx) {
      error += dx;
      y0 += sy;
    }
  }
}

std::string format_value(float value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(value));
  return buffer;
}

std::string format_degrees(float radians) {
  constexpr float kRadiansToDegrees = 57.2957795130823208768F;
  const float degrees = radians * kRadiansToDegrees;
  char buffer[32];
  if (std::abs(degrees) < 0.5F) {
    std::snprintf(buffer, sizeof(buffer), "0");
  } else {
    std::snprintf(buffer, sizeof(buffer), "%+.0f", static_cast<double>(degrees));
  }
  return buffer;
}

struct PixelRect {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
};

struct DashboardLayout {
  PixelRect rd_plot;
  PixelRect azimuth_range_panel;
  PixelRect boresight_panel;
  std::array<PixelRect, 4> history_panels;
};

DashboardLayout make_dashboard_layout(std::int64_t width, std::int64_t height, bool show_angles) {
  constexpr int kOuter = 10;
  constexpr int kGap = 10;
  const int canvas_width = static_cast<int>(width);
  const int canvas_height = static_cast<int>(height);
  const int side_width = std::max(360, canvas_width * 38 / 100);
  const int side_x = canvas_width - kOuter - side_width;
  const int rd_right = side_x - kGap;

  DashboardLayout result;
  result.rd_plot = PixelRect{
      .x = 58,
      .y = 38,
      .width = rd_right - 66,
      .height = canvas_height - 82,
  };
  const int panel_width = (side_width - kGap) / 2;
  const int available_height = canvas_height - 2 * kOuter;
  const int panel_height = (available_height - 2 * kGap) / 3;
  result.azimuth_range_panel = PixelRect{side_x, kOuter, panel_width, panel_height};
  result.boresight_panel =
      PixelRect{side_x + panel_width + kGap, kOuter, panel_width, panel_height};

  if (!show_angles) {
    const int history_height = (available_height - kGap) / 2;
    result.history_panels = {
        PixelRect{side_x, kOuter, side_width, history_height},
        PixelRect{side_x, kOuter + history_height + kGap, side_width, history_height},
        PixelRect{},
        PixelRect{},
    };
    return result;
  }

  const int second_row_y = kOuter + panel_height + kGap;
  const int third_row_y = second_row_y + panel_height + kGap;
  result.history_panels = {
      PixelRect{side_x, second_row_y, panel_width, panel_height},
      PixelRect{side_x + panel_width + kGap, second_row_y, panel_width, panel_height},
      PixelRect{side_x, third_row_y, panel_width, panel_height},
      PixelRect{side_x + panel_width + kGap, third_row_y, panel_width, panel_height},
  };
  return result;
}

constexpr std::array<std::array<std::uint8_t, 4>, 8> kClusterColors{{
    {0U, 255U, 255U, 255U},
    {255U, 210U, 0U, 255U},
    {255U, 90U, 180U, 255U},
    {100U, 255U, 100U, 255U},
    {255U, 130U, 60U, 255U},
    {120U, 170U, 255U, 255U},
    {220U, 120U, 255U, 255U},
    {255U, 255U, 255U, 255U},
}};

void fill_rect(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height,
               const PixelRect& rect, const std::array<std::uint8_t, 4>& color) {
  for (int y = rect.y; y < rect.y + rect.height; ++y) {
    for (int x = rect.x; x < rect.x + rect.width; ++x) {
      set_pixel(image, width, height, x, y, color);
    }
  }
}

void draw_disc(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height, int cx,
               int cy, int radius, const std::array<std::uint8_t, 4>& color) {
  for (int y = -radius; y <= radius; ++y) {
    for (int x = -radius; x <= radius; ++x) {
      if (x * x + y * y <= radius * radius) {
        set_pixel(image, width, height, cx + x, cy + y, color);
      }
    }
  }
}

void fill_oval(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height, float cx,
               float cy, float rx, float ry, const std::array<std::uint8_t, 4>& color) {
  const int x_radius = static_cast<int>(std::ceil(rx));
  const int y_radius = static_cast<int>(std::ceil(ry));
  for (int y = -y_radius; y <= y_radius; ++y) {
    for (int x = -x_radius; x <= x_radius; ++x) {
      const float normalized_x = static_cast<float>(x) / rx;
      const float normalized_y = static_cast<float>(y) / ry;
      if (normalized_x * normalized_x + normalized_y * normalized_y <= 1.0F) {
        set_pixel(image, width, height, static_cast<int>(std::lround(cx)) + x,
                  static_cast<int>(std::lround(cy)) + y, color);
      }
    }
  }
}

void draw_oval(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height, float cx,
               float cy, float rx, float ry, const std::array<std::uint8_t, 4>& color,
               int segments) {
  constexpr float kTau = 6.28318530717958647692F;
  for (int segment = 0; segment < segments; ++segment) {
    const float a0 = kTau * segment / segments;
    const float a1 = kTau * (segment + 1) / segments;
    draw_line(image, width, height, static_cast<int>(std::lround(cx + rx * std::cos(a0))),
              static_cast<int>(std::lround(cy + ry * std::sin(a0))),
              static_cast<int>(std::lround(cx + rx * std::cos(a1))),
              static_cast<int>(std::lround(cy + ry * std::sin(a1))), color);
  }
}

void draw_polar_arc(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height,
                    float cx, float cy, float radius, float start_azimuth, float end_azimuth,
                    const std::array<std::uint8_t, 4>& color, int segments) {
  for (int segment = 0; segment < segments; ++segment) {
    const float fraction0 = static_cast<float>(segment) / segments;
    const float fraction1 = static_cast<float>(segment + 1) / segments;
    const float azimuth0 = start_azimuth + fraction0 * (end_azimuth - start_azimuth);
    const float azimuth1 = start_azimuth + fraction1 * (end_azimuth - start_azimuth);
    draw_line(image, width, height, static_cast<int>(std::lround(cx + radius * std::sin(azimuth0))),
              static_cast<int>(std::lround(cy - radius * std::cos(azimuth0))),
              static_cast<int>(std::lround(cx + radius * std::sin(azimuth1))),
              static_cast<int>(std::lround(cy - radius * std::cos(azimuth1))), color);
  }
}

std::array<std::uint8_t, 4> aged_color(const std::array<std::uint8_t, 4>& color, std::int64_t age,
                                       std::int64_t lifetime) {
  const float fraction = std::clamp(
      static_cast<float>(age) / static_cast<float>(std::max<std::int64_t>(1, lifetime - 1)), 0.0F,
      1.0F);
  const float scale = 1.0F - 0.75F * fraction;
  return {
      static_cast<std::uint8_t>(scale * color[0]),
      static_cast<std::uint8_t>(scale * color[1]),
      static_cast<std::uint8_t>(scale * color[2]),
      color[3],
  };
}

enum class SpatialView {
  kAzimuthRange,
  kBoresight,
};

void draw_spatial_panel(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height,
                        const PixelRect& panel, std::string_view title, SpatialView view,
                        const std::deque<HistorySample>& history,
                        const std::array<float, 2>& range_bounds, std::int64_t frame_index,
                        const FmcwDashboardConfig& config) {
  constexpr std::array<std::uint8_t, 4> kBackground{12U, 12U, 18U, 255U};
  constexpr std::array<std::uint8_t, 4> kGrid{70U, 70U, 82U, 255U};
  constexpr std::array<std::uint8_t, 4> kWhite{255U, 255U, 255U, 255U};
  fill_rect(image, width, height, panel, kBackground);
  const bool is_polar = view == SpatialView::kAzimuthRange;
  const PixelRect plot{
      .x = panel.x + (is_polar ? 12 : 38),
      .y = panel.y + 22,
      .width = panel.width - (is_polar ? 24 : 46),
      .height = panel.height - 48,
  };
  if (plot.width <= 1 || plot.height <= 1) {
    return;
  }
  draw_text(image, width, height, panel.x + (panel.width - static_cast<int>(title.size()) * 6) / 2,
            panel.y + 5, title, 1);
  if (!is_polar) {
    constexpr std::string_view kXLabel = "AZ (RAD)";
    constexpr std::string_view kYLabel = "EL (RAD)";
    draw_text(image, width, height,
              plot.x + (plot.width - static_cast<int>(kXLabel.size()) * 6) / 2,
              plot.y + plot.height + 4, kXLabel, 1);
    draw_text_ccw(image, width, height, panel.x + 2,
                  plot.y + (plot.height - static_cast<int>(kYLabel.size()) * 6) / 2, kYLabel, 1);
  }

  const float max_range = std::max(1.0F, range_bounds[1]);
  const float circle_cx = plot.x + 0.5F * (plot.width - 1);
  const bool use_sector = is_polar && detail::use_azimuth_sector(config.az_bounds);
  const float display_center_azimuth =
      use_sector ? 0.5F * (config.az_bounds[0] + config.az_bounds[1]) : 0.0F;
  const float display_half_span =
      use_sector ? 0.5F * (config.az_bounds[1] - config.az_bounds[0]) : 0.0F;
  const float circle_cy = use_sector ? static_cast<float>(plot.y + plot.height - 12)
                                     : plot.y + 0.5F * (plot.height - 1);
  const float projected_half_width = std::max(0.001F, std::sin(display_half_span));
  const float circle_radius =
      use_sector ? std::min(0.5F * (plot.width - 2) / projected_half_width,
                            static_cast<float>(plot.height - 20))
                 : 0.5F * static_cast<float>(std::min(plot.width, plot.height) - 2);
  if (view == SpatialView::kAzimuthRange) {
    if (use_sector) {
      draw_polar_arc(image, width, height, circle_cx, circle_cy, circle_radius, -display_half_span,
                     display_half_span, kGrid, 48);
      draw_polar_arc(image, width, height, circle_cx, circle_cy, 0.5F * circle_radius,
                     -display_half_span, display_half_span, kGrid, 32);
      const std::string left_label = format_degrees(config.az_bounds[0]);
      const std::string center_label = format_degrees(display_center_azimuth);
      const std::string right_label = format_degrees(config.az_bounds[1]);
      const std::string degree_legend =
          "AZ: " + left_label + " / " + center_label + " / " + right_label + " DEG";
      draw_text(image, width, height,
                panel.x + (panel.width - static_cast<int>(degree_legend.size()) * 6) / 2,
                panel.y + 14, degree_legend, 1);
    } else {
      draw_oval(image, width, height, circle_cx, circle_cy, circle_radius, circle_radius, kGrid,
                48);
      draw_oval(image, width, height, circle_cx, circle_cy, 0.5F * circle_radius,
                0.5F * circle_radius, kGrid, 32);
    }
    draw_line(image, width, height, static_cast<int>(circle_cx), static_cast<int>(circle_cy),
              static_cast<int>(circle_cx), static_cast<int>(circle_cy - circle_radius), kGrid);
    for (float azimuth : config.az_bounds) {
      const float display_azimuth = azimuth - display_center_azimuth;
      draw_line(image, width, height, static_cast<int>(circle_cx), static_cast<int>(circle_cy),
                static_cast<int>(circle_cx + circle_radius * std::sin(display_azimuth)),
                static_cast<int>(circle_cy - circle_radius * std::cos(display_azimuth)), kGrid);
    }
  } else {
    for (int tick = 0; tick < 3; ++tick) {
      const float fraction = static_cast<float>(tick) / 2.0F;
      const int x = plot.x + static_cast<int>(fraction * (plot.width - 1));
      const int y = plot.y + static_cast<int>(fraction * (plot.height - 1));
      draw_line(image, width, height, x, plot.y, x, plot.y + plot.height - 1, kGrid);
      draw_line(image, width, height, plot.x, y, plot.x + plot.width - 1, y, kGrid);
    }
  }

  const auto to_pixel = [&](const std::array<float, 4>& features) {
    if (view == SpatialView::kAzimuthRange) {
      const float normalized_range = std::clamp(features[0] / max_range, 0.0F, 1.0F);
      const float display_azimuth = features[2] - display_center_azimuth;
      return std::array<int, 2>{
          static_cast<int>(circle_cx +
                           normalized_range * circle_radius * std::sin(display_azimuth)),
          static_cast<int>(circle_cy -
                           normalized_range * circle_radius * std::cos(display_azimuth)),
      };
    }
    const float x_fraction = std::clamp(
        (features[2] - config.az_bounds[0]) / (config.az_bounds[1] - config.az_bounds[0]), 0.0F,
        1.0F);
    const float y_fraction = std::clamp(
        (features[3] - config.el_bounds[0]) / (config.el_bounds[1] - config.el_bounds[0]), 0.0F,
        1.0F);
    return std::array<int, 2>{
        plot.x + static_cast<int>(x_fraction * (plot.width - 1)),
        plot.y + plot.height - 1 - static_cast<int>(y_fraction * (plot.height - 1)),
    };
  };

  const std::int64_t first_truth_frame =
      std::max<std::int64_t>(0, frame_index - config.detection_trail_frames + 1);
  for (const TruthTarget& target : config.truth_targets) {
    const auto& color =
        kClusterColors[static_cast<std::size_t>(target.color_bucket) % kClusterColors.size()];
    std::array<int, 2> previous{};
    bool have_previous = false;
    for (std::int64_t frame = first_truth_frame; frame <= frame_index; ++frame) {
      const double truth_time = static_cast<double>(frame) * config.frame_period;
      if (!detail::truth_target_is_visible_at_time(target, truth_time, config.scene_visibility)) {
        have_previous = false;
        continue;
      }
      const auto pixel = to_pixel(detail::truth_features_at_time(target, truth_time));
      if (have_previous) {
        draw_line(image, width, height, previous[0], previous[1], pixel[0], pixel[1], color);
      }
      previous = pixel;
      have_previous = true;
    }
  }

  for (const HistorySample& sample : history) {
    const auto& base_color =
        kClusterColors[static_cast<std::size_t>(sample.color_bucket) % kClusterColors.size()];
    const std::int64_t age = frame_index - sample.frame;
    const auto color = aged_color(base_color, age, config.detection_trail_frames);
    const auto pixel = to_pixel(sample.features);
    const int radius = age == 0 ? 4 : 2;
    draw_disc(image, width, height, pixel[0], pixel[1], radius, color);
    if (age == 0) {
      draw_oval(image, width, height, static_cast<float>(pixel[0]), static_cast<float>(pixel[1]),
                7.0F, 7.0F, kWhite, 16);
    }
  }
}

void draw_history_panel(std::vector<std::uint8_t>* image, std::int64_t width, std::int64_t height,
                        const PixelRect& panel, std::string_view title, int feature,
                        std::array<float, 2> bounds, const std::deque<HistorySample>& history,
                        std::int64_t frame_index, const FmcwDashboardConfig& config) {
  constexpr std::array<std::uint8_t, 4> kBackground{12U, 12U, 18U, 255U};
  constexpr std::array<std::uint8_t, 4> kGrid{70U, 70U, 82U, 255U};
  fill_rect(image, width, height, panel, kBackground);

  const PixelRect plot{
      .x = panel.x + 38,
      .y = panel.y + 22,
      .width = panel.width - 46,
      .height = panel.height - 48,
  };
  if (plot.width <= 1 || plot.height <= 1) {
    return;
  }
  draw_text(image, width, height, panel.x + (panel.width - static_cast<int>(title.size()) * 6) / 2,
            panel.y + 5, title, 1);

  for (int tick = 0; tick < 3; ++tick) {
    const float fraction = static_cast<float>(tick) / 2.0F;
    const int x = plot.x + static_cast<int>(fraction * (plot.width - 1));
    const int y = plot.y + static_cast<int>(fraction * (plot.height - 1));
    draw_line(image, width, height, x, plot.y, x, plot.y + plot.height - 1, kGrid);
    draw_line(image, width, height, plot.x, y, plot.x + plot.width - 1, y, kGrid);

    const float value = bounds[1] - fraction * (bounds[1] - bounds[0]);
    const std::string y_label = format_value(value);
    draw_text(image, width, height, plot.x - 4 - static_cast<int>(y_label.size()) * 6, y - 3,
              y_label, 1);

    const double seconds =
        -(1.0 - fraction) * static_cast<double>(config.history_frames - 1) * config.frame_period;
    const std::string x_label = format_value(static_cast<float>(seconds));
    draw_text(image, width, height, x - static_cast<int>(x_label.size()) * 3,
              plot.y + plot.height + 4, x_label, 1);
  }

  const std::int64_t first_truth_frame =
      std::max<std::int64_t>(0, frame_index - config.history_frames + 1);
  for (const TruthTarget& target : config.truth_targets) {
    const auto& color =
        kClusterColors[static_cast<std::size_t>(target.color_bucket) % kClusterColors.size()];
    std::array<int, 2> previous{};
    bool have_previous = false;
    for (std::int64_t frame = first_truth_frame; frame <= frame_index; ++frame) {
      const double truth_time = static_cast<double>(frame) * config.frame_period;
      if (!detail::truth_target_is_visible_at_time(target, truth_time, config.scene_visibility)) {
        have_previous = false;
        continue;
      }
      const HistorySample truth_sample{
          .frame = frame, .features = detail::truth_features_at_time(target, truth_time)};
      const auto position = detail::history_plot_position(truth_sample, frame_index,
                                                          config.history_frames, feature, bounds);
      const std::array<int, 2> pixel{
          plot.x + static_cast<int>(position[0] * (plot.width - 1)),
          plot.y + plot.height - 1 - static_cast<int>(position[1] * (plot.height - 1))};
      if (have_previous) {
        draw_line(image, width, height, previous[0], previous[1], pixel[0], pixel[1], color);
      }
      previous = pixel;
      have_previous = true;
    }
    if (have_previous && first_truth_frame == frame_index) {
      draw_disc(image, width, height, previous[0], previous[1], 1, color);
    }
  }

  for (const HistorySample& sample : history) {
    const auto position =
        detail::history_plot_position(sample, frame_index, config.history_frames, feature, bounds);
    const int x = plot.x + static_cast<int>(position[0] * (plot.width - 1));
    const int y = plot.y + plot.height - 1 - static_cast<int>(position[1] * (plot.height - 1));
    const auto& color =
        kClusterColors[static_cast<std::size_t>(sample.color_bucket) % kClusterColors.size()];
    draw_disc(image, width, height, x, y, 2, color);
  }
}

std::vector<std::uint8_t> make_annotation(
    std::int64_t rows, std::int64_t cols, const std::vector<DetectionMarker>& markers,
    const std::deque<HistorySample>& history, const std::deque<HistorySample>& detection_history,
    const std::vector<float>& ranges, const std::vector<float>& velocities,
    const FmcwDashboardConfig& config, std::int64_t frame_index) {
  const std::int64_t width = config.width;
  const std::int64_t height = config.height;
  const DashboardLayout layout = make_dashboard_layout(width, height, config.show_angles);
  const PixelRect& plot = layout.rd_plot;
  std::vector<std::uint8_t> image(static_cast<std::size_t>(width * height * 4), 0U);
  constexpr std::array<std::uint8_t, 4> kWhite{255U, 255U, 255U, 255U};

  draw_line(&image, width, height, plot.x - 1, plot.y - 1, plot.x + plot.width, plot.y - 1, kWhite);
  draw_line(&image, width, height, plot.x - 1, plot.y + plot.height, plot.x + plot.width,
            plot.y + plot.height, kWhite);
  draw_line(&image, width, height, plot.x - 1, plot.y - 1, plot.x - 1, plot.y + plot.height,
            kWhite);
  draw_line(&image, width, height, plot.x + plot.width, plot.y - 1, plot.x + plot.width,
            plot.y + plot.height, kWhite);

  const std::string title =
      config.title +
      " | DT: " + format_value(static_cast<float>(frame_index * config.frame_period)) + " S";
  const std::int64_t title_width = static_cast<std::int64_t>(title.size()) * 12;
  draw_text(&image, width, height,
            std::max<std::int64_t>(2, plot.x + (plot.width - title_width) / 2), 9, title, 2);
  constexpr std::string_view kDopplerLabel = "VELOCITY (M/S)";
  draw_text(&image, width, height,
            plot.x + (plot.width - static_cast<std::int64_t>(kDopplerLabel.size()) * 6) / 2,
            plot.y + plot.height + 25, kDopplerLabel, 1);
  constexpr std::string_view kRangeLabel = "RANGE (M)";
  draw_text_ccw(
      &image, width, height, 2,
      std::max<std::int64_t>(
          0, plot.y + (plot.height - static_cast<std::int64_t>(kRangeLabel.size()) * 6) / 2),
      kRangeLabel, 1);

  const std::array<std::int64_t, 3> x_positions{plot.x, plot.x + plot.width / 2,
                                                plot.x + plot.width - 1};
  const std::array<std::size_t, 3> x_indices{0U, velocities.size() / 2U, velocities.size() - 1U};
  for (std::size_t i = 0; i < x_positions.size(); ++i) {
    const std::string tick = format_value(velocities[x_indices[i]]);
    draw_text(&image, width, height, x_positions[i] - static_cast<std::int64_t>(tick.size()) * 3,
              plot.y + plot.height + 5, tick, 1);
  }
  const std::array<std::int64_t, 3> y_positions{plot.y, plot.y + plot.height / 2,
                                                plot.y + plot.height - 7};
  const std::array<std::size_t, 3> y_indices{ranges.size() - 1U, ranges.size() / 2U, 0U};
  for (std::size_t i = 0; i < y_positions.size(); ++i) {
    const std::string tick = format_value(ranges[y_indices[i]]);
    draw_text(&image, width, height, plot.x - 4 - static_cast<std::int64_t>(tick.size()) * 6,
              y_positions[i], tick, 1);
  }

  constexpr std::array<std::uint8_t, 4> kBlack{0U, 0U, 0U, 255U};
  for (const DetectionMarker& marker : markers) {
    const auto& color =
        kClusterColors[static_cast<std::size_t>(marker.color_bucket) % kClusterColors.size()];
    const auto position =
        detail::rd_plot_position(marker.doppler_bin, marker.range_bin, rows, cols);
    const float x = plot.x + position[0] * plot.width;
    const float y = plot.y + position[1] * plot.height;
    const float outer_rx = std::max(4.0F, 0.65F * config.detection_radius * plot.width);
    const float outer_ry = std::max(4.0F, 0.65F * config.detection_radius * plot.height);
    fill_oval(&image, width, height, x, y, outer_rx, outer_ry, kBlack);
    fill_oval(&image, width, height, x, y, std::max(1.0F, outer_rx - 2.0F),
              std::max(1.0F, outer_ry - 2.0F), color);
  }

  const auto range_bounds = std::minmax_element(ranges.begin(), ranges.end());
  const auto velocity_bounds = std::minmax_element(velocities.begin(), velocities.end());
  const std::array<float, 2> physical_range_bounds{*range_bounds.first, *range_bounds.second};
  if (config.show_angles) {
    draw_spatial_panel(&image, width, height, layout.azimuth_range_panel, "AZIMUTH-RANGE",
                       SpatialView::kAzimuthRange, detection_history, physical_range_bounds,
                       frame_index, config);
    draw_spatial_panel(&image, width, height, layout.boresight_panel, "BORESIGHT",
                       SpatialView::kBoresight, detection_history, physical_range_bounds,
                       frame_index, config);
  }
  draw_history_panel(&image, width, height, layout.history_panels[0], "RANGE (M)", 0,
                     physical_range_bounds, history, frame_index, config);
  draw_history_panel(&image, width, height, layout.history_panels[1], "VELOCITY (M/S)", 1,
                     {*velocity_bounds.first, *velocity_bounds.second}, history, frame_index,
                     config);
  if (config.show_angles) {
    draw_history_panel(&image, width, height, layout.history_panels[2], "AZIMUTH (RAD)", 2,
                       config.az_bounds, history, frame_index, config);
    draw_history_panel(&image, width, height, layout.history_panels[3], "ELEVATION (RAD)", 3,
                       config.el_bounds, history, frame_index, config);
  }
  return image;
}

}  // namespace

FmcwDashboardOp::FmcwDashboardOp(FmcwDashboardConfig config) : config_(std::move(config)) {
  sim::validate_scene_visibility(config_.scene_visibility);
  if (!std::isfinite(config_.dynamic_range_db) || config_.dynamic_range_db <= 0.0F ||
      !std::isfinite(config_.detection_radius) || config_.detection_radius <= 0.0F ||
      !std::isfinite(config_.frame_period) || config_.frame_period <= 0.0 ||
      config_.history_frames < 2 || config_.detection_trail_frames < 1 || config_.width < 640 ||
      config_.height < 360 ||
      (config_.selected_profile.empty() == config_.selected_waveform.is_valid()) ||
      (config_.show_angles && (!(config_.az_bounds[0] < config_.az_bounds[1]) ||
                               !(config_.el_bounds[0] < config_.el_bounds[1])))) {
    throw std::invalid_argument("FmcwDashboardOp: invalid scales, history, bounds, or canvas size");
  }
  const std::vector<std::int64_t> max_shape{config_.height, config_.width, 4};
  max_byte_span_ = tensor_byte_span(max_shape, kRgba8);
  peak_power_ = make_device_tensor({1}, kFloat32);
  annotation_ = make_device_tensor(max_shape, kRgba8);
}

void FmcwDashboardOp::setup(holoscan::OperatorSpec& spec) {
  constexpr TensorRepresentation kPower{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = kFloat32,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kImage{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = kRgba8,
      .rank = 3U,
      .strides = TensorStridePolicy::kContiguousOnly,
  };
  spec.input(cfar_power).queue_depth(2U).expects_tensor(tensor_input(kPower));
  spec.input(detection_idx).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(points).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  if (config_.mode == DashboardMode::kClusters) {
    spec.input(labels).queue_depth(2U).expects_tensor(tensor_input(kDevice));
    spec.input(sizes).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  }
  spec.input(range_bins).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(velocity_bins).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  if (config_.mode == DashboardMode::kClusters) {
    spec.input(reduced_points).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  }
  if (!config_.selected_profile.empty()) {
    spec.input(waveform_in).queue_depth(2U);
  }
  spec.output(rgba).produces_tensor(pooled_tensor_output(kImage, 3U, max_byte_span_));
}

holoscan::Contract FmcwDashboardOp::contract() const {
  holoscan::Contract result;
  if (config_.mode == DashboardMode::kClusters) {
    result
        .trigger(holoscan::OnAligned{
            .ports = holoscan::ports(cfar_power, detection_idx, points, labels, sizes, range_bins,
                                     velocity_bins, reduced_points),
            .window = std::chrono::nanoseconds{0}})
        .input(cfar_power, holoscan::Exact{})
        .input(detection_idx, holoscan::Exact{})
        .input(points, holoscan::Exact{})
        .input(labels, holoscan::Exact{})
        .input(sizes, holoscan::Exact{})
        .input(range_bins, holoscan::Exact{})
        .input(velocity_bins, holoscan::Exact{})
        .input(reduced_points, holoscan::Exact{});
  } else {
    result
        .trigger(holoscan::OnAligned{
            .ports = holoscan::ports(cfar_power, detection_idx, points, range_bins, velocity_bins),
            .window = std::chrono::nanoseconds{0}})
        .input(cfar_power, holoscan::Exact{})
        .input(detection_idx, holoscan::Exact{})
        .input(points, holoscan::Exact{})
        .input(range_bins, holoscan::Exact{})
        .input(velocity_bins, holoscan::Exact{});
  }
  if (!config_.selected_profile.empty()) {
    result.input(waveform_in, holoscan::Exact{}).on_missing(waveform_in, holoscan::SafeDefault{});
  }
  return result;
}

expected<void, Error> FmcwDashboardOp::compute(holoscan::ExecutionContext& context) {
  constexpr const char* kWho = "FmcwDashboardOp";
  HOLOSCAN_TRY(auto power, receive_tensor(cfar_power, kWho));
  HOLOSCAN_TRY(auto idx, receive_tensor(detection_idx, kWho));
  HOLOSCAN_TRY(auto feature_points, receive_tensor(points, kWho));
  HOLOSCAN_TRY(auto ranges, receive_tensor(range_bins, kWho));
  HOLOSCAN_TRY(auto velocities, receive_tensor(velocity_bins, kWho));
  Tensor cluster_labels;
  Tensor cluster_sizes;
  Tensor reduced;
  if (config_.mode == DashboardMode::kClusters) {
    HOLOSCAN_TRY(cluster_labels, receive_tensor(labels, kWho));
    HOLOSCAN_TRY(cluster_sizes, receive_tensor(sizes, kWho));
    HOLOSCAN_TRY(reduced, receive_tensor(reduced_points, kWho));
  }
  std::optional<wire::FMCWWaveformT> wire_waveform;
  if (!config_.selected_profile.empty()) {
    HOLOSCAN_TRY(wire_waveform, receive_optional_value(waveform_in));
  }
  HOLOSCAN_TRY(bool render_profile,
               detail::selected_profile_matches(wire_waveform, config_.selected_profile,
                                                config_.selected_waveform));
  if (!render_profile) {
    ++frame_index_;
    return {};
  }
  const std::int64_t current_frame = frame_index_++;
  BatchedShape layout;
  try {
    layout = collapse_batch(power, "FmcwDashboardOp.cfar_power");
    if (config_.mode == DashboardMode::kClusters) {
      validate_dashboard_inputs(idx, feature_points, cluster_labels, cluster_sizes, ranges,
                                velocities, reduced, config_.show_angles);
    } else {
      validate_detection_inputs(idx, feature_points, ranges, velocities, config_.show_angles);
    }
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }
  if (!dldatatype_equals(power.dtype(), kFloat32)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(kWho) + ": cfar_power must be float32"});
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported, std::string(kWho) + ": no CUDA stream"});
  }

  HOLOSCAN_TRY(auto host_idx, copy_int64_to_host(idx, stream, "detection_idx"));
  HOLOSCAN_TRY(auto host_points, copy_float_to_host(feature_points, stream, "points"));
  HOLOSCAN_TRY(auto host_ranges, copy_float_to_host(ranges, stream, "range_bins"));
  HOLOSCAN_TRY(auto host_velocities, copy_float_to_host(velocities, stream, "velocity_bins"));
  std::vector<std::int64_t> host_labels;
  std::vector<std::int64_t> host_sizes;
  std::vector<float> host_reduced;
  if (config_.mode == DashboardMode::kClusters) {
    HOLOSCAN_TRY(host_labels, copy_int64_to_host(cluster_labels, stream, "labels"));
    HOLOSCAN_TRY(host_sizes, copy_int64_to_host(cluster_sizes, stream, "sizes"));
    HOLOSCAN_TRY(host_reduced, copy_float_to_host(reduced, stream, "reduced_points"));
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return cuda_failure(kWho, "annotation input synchronization");
  }

  std::vector<std::uint8_t> host_annotation;
  try {
    std::vector<DetectionMarker> markers;
    if (config_.mode == DashboardMode::kClusters) {
      markers = detail::make_cluster_markers(host_idx, host_points, host_labels, host_sizes,
                                             idx.shape()[1], feature_points.shape()[1],
                                             config_.angle_convention, config_.show_angles);
      detail::apply_reduced_features(&markers, host_reduced, reduced.shape()[1],
                                     config_.angle_convention, config_.show_angles);
    } else {
      markers = detail::make_raw_detection_markers(host_idx, host_points, idx.shape()[1],
                                                   feature_points.shape()[1],
                                                   config_.angle_convention, config_.show_angles);
    }
    detail::append_history(&history_, markers, current_frame, config_.history_frames);
    detail::append_history(&detection_history_, markers, current_frame,
                           config_.detection_trail_frames);
    host_annotation =
        make_annotation(layout.rows, layout.cols, markers, history_, detection_history_,
                        host_ranges, host_velocities, config_, current_frame);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }
  if (cudaMemcpyAsync(annotation_->mutable_data().value(), host_annotation.data(),
                      host_annotation.size(), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return cuda_failure(kWho, "annotation upload");
  }

  const std::int64_t pixels = layout.rows * layout.cols;
  const std::int64_t count = layout.batch * pixels;
  if (cudaMemsetAsync(peak_power_->mutable_data().value(), 0, sizeof(float), stream) !=
      cudaSuccess) {
    return cuda_failure(kWho, "peak reset");
  }
  constexpr int kBlock = 256;
  const int reduce_grid = static_cast<int>((count + kBlock - 1) / kBlock);
  reduce_positive_peak<<<reduce_grid, kBlock, 0, stream>>>(
      static_cast<const float*>(power.data()), count,
      static_cast<unsigned int*>(peak_power_->mutable_data().value()));
  if (cudaGetLastError() != cudaSuccess) {
    return cuda_failure(kWho, "peak reduction launch");
  }

  const std::int64_t canvas_width = config_.width;
  const std::int64_t canvas_height = config_.height;
  const std::int64_t canvas_pixels = canvas_width * canvas_height;
  const DashboardLayout dashboard =
      make_dashboard_layout(canvas_width, canvas_height, config_.show_angles);
  const std::int64_t plot_pixels =
      static_cast<std::int64_t>(dashboard.rd_plot.width) * dashboard.rd_plot.height;
  const std::vector<std::int64_t> shape{canvas_height, canvas_width, 4};
  HOLOSCAN_TRY(
      auto loan,
      prepare_device_tensor(
          rgba, context, shape, kRgba8, [&](void* dst, std::size_t bytes) -> expected<void, Error> {
            if (cudaMemsetAsync(dst, 0, bytes, stream) != cudaSuccess) {
              return cuda_failure(kWho, "canvas reset");
            }
            const int grid = static_cast<int>((plot_pixels + kBlock - 1) / kBlock);
            power_to_rgba<<<grid, kBlock, 0, stream>>>(
                static_cast<const float*>(power.data()), layout.batch, layout.rows, layout.cols,
                static_cast<const float*>(peak_power_->data()), config_.dynamic_range_db,
                canvas_width, dashboard.rd_plot.x, dashboard.rd_plot.y, dashboard.rd_plot.width,
                dashboard.rd_plot.height, static_cast<std::uint8_t*>(dst));
            if (cudaGetLastError() != cudaSuccess) {
              return cuda_failure(kWho, "color mapping launch");
            }
            const int annotation_grid = static_cast<int>((canvas_pixels + kBlock - 1) / kBlock);
            apply_annotation<<<annotation_grid, kBlock, 0, stream>>>(
                static_cast<const std::uint8_t*>(annotation_->data()), canvas_pixels,
                static_cast<std::uint8_t*>(dst));
            if (cudaGetLastError() != cudaSuccess) {
              return cuda_failure(kWho, "annotation launch");
            }
            return {};
          }));
  // annotation_ is reused on the next tick and may otherwise be overwritten
  // from a different activation stream before this composition has read it.
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return cuda_failure(kWho, "composition synchronization");
  }
  retained_input_ = std::move(power);
  return emit_prepared_tensor(rgba, context, std::move(loan));
}

namespace detail {

expected<bool, Error> selected_profile_matches(
    const std::optional<wire::FMCWWaveformT>& applied_waveform, std::string_view selected_profile,
    const FMCWWaveformDescriptor& selected_waveform) {
  if (selected_profile.empty()) {
    return true;
  }
  if (!applied_waveform.has_value()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "FmcwDashboardOp: selected profile requires an Exact waveform input"});
  }
  auto decoded = from_wire(*applied_waveform);
  if (!decoded) {
    return make_unexpected(std::move(decoded).error());
  }
  return *decoded == selected_waveform;
}

std::vector<std::uint8_t> make_heatmap_rgba_reference(const std::vector<float>& power,
                                                      const std::vector<std::int64_t>& shape,
                                                      float dynamic_range_db) {
  if (shape.size() < 2U || !std::isfinite(dynamic_range_db) || dynamic_range_db <= 0.0F) {
    throw std::invalid_argument("make_heatmap_rgba_reference: invalid shape or dynamic range");
  }
  std::int64_t planes = 1;
  for (std::size_t i = 0; i + 2 < shape.size(); ++i) {
    if (shape[i] <= 0) {
      throw std::invalid_argument("make_heatmap_rgba_reference: shape must be positive");
    }
    planes *= shape[i];
  }
  const std::int64_t pixels = shape[shape.size() - 2] * shape.back();
  if (pixels <= 0 || power.size() != static_cast<std::size_t>(planes * pixels)) {
    throw std::invalid_argument("make_heatmap_rgba_reference: power size does not match shape");
  }
  const float peak = *std::max_element(power.begin(), power.end());
  const float reference = std::max(peak, FLT_MIN);
  std::vector<std::uint8_t> result(static_cast<std::size_t>(pixels) * 4U);
  for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
    float value = 0.0F;
    for (std::int64_t plane = 0; plane < planes; ++plane) {
      value = std::max(value, power[static_cast<std::size_t>(plane * pixels + pixel)]);
    }
    const float db = 10.0F * std::log10(std::max(value, FLT_MIN) / reference);
    const float t = std::clamp(1.0F + db / dynamic_range_db, 0.0F, 1.0F);
    const float red = std::clamp(3.0F * t - 1.0F, 0.0F, 1.0F);
    const float green = std::clamp(1.5F - std::abs(3.0F * t - 1.5F), 0.0F, 1.0F);
    const float blue = std::clamp(2.0F - 3.0F * t, 0.0F, 1.0F);
    const std::size_t out = static_cast<std::size_t>(pixel) * 4U;
    result[out] = static_cast<std::uint8_t>(red * 255.0F);
    result[out + 1] = static_cast<std::uint8_t>(green * 255.0F);
    result[out + 2] = static_cast<std::uint8_t>(blue * 255.0F);
    result[out + 3] = 255U;
  }
  return result;
}

std::vector<DetectionMarker> make_cluster_markers(const std::vector<std::int64_t>& detection_idx,
                                                  const std::vector<float>& points,
                                                  const std::vector<std::int64_t>& labels,
                                                  const std::vector<std::int64_t>& sizes,
                                                  std::int64_t idx_cols, std::int64_t point_cols,
                                                  const AngleConvention& angle_convention,
                                                  bool show_angles) {
  const std::int64_t min_features = show_angles ? 4 : 2;
  if (idx_cols < 2 || point_cols < min_features ||
      detection_idx.size() % static_cast<std::size_t>(idx_cols) != 0U) {
    throw std::invalid_argument("make_cluster_markers: invalid column counts");
  }
  const std::size_t rows = detection_idx.size() / static_cast<std::size_t>(idx_cols);
  if (labels.size() != rows || points.size() != rows * static_cast<std::size_t>(point_cols)) {
    throw std::invalid_argument("make_cluster_markers: row counts do not match");
  }

  struct Accumulator {
    double doppler_bin{0.0};
    double range_bin{0.0};
    std::array<double, 4> features{};
    PhysicalAngleCentroid angle_centroid;
    std::int64_t count{0};
  };
  std::vector<Accumulator> accumulators(sizes.size());
  for (std::size_t row = 0; row < rows; ++row) {
    const std::int64_t label = labels[row];
    if (label < 0) {
      continue;
    }
    if (label >= static_cast<std::int64_t>(sizes.size())) {
      throw std::invalid_argument("make_cluster_markers: label is out of bounds");
    }
    Accumulator& accumulator = accumulators[static_cast<std::size_t>(label)];
    const std::size_t idx_offset = row * static_cast<std::size_t>(idx_cols);
    accumulator.doppler_bin +=
        static_cast<double>(detection_idx[idx_offset + static_cast<std::size_t>(idx_cols - 2)]);
    accumulator.range_bin +=
        static_cast<double>(detection_idx[idx_offset + static_cast<std::size_t>(idx_cols - 1)]);
    const std::size_t point_offset = row * static_cast<std::size_t>(point_cols);
    for (std::size_t feature = 0; feature < 2U; ++feature) {
      accumulator.features[feature] += static_cast<double>(points[point_offset + feature]);
    }
    if (show_angles) {
      accumulator.angle_centroid.add(angle_convention, points[point_offset + 2U],
                                     points[point_offset + 3U]);
    }
    ++accumulator.count;
  }

  std::vector<DetectionMarker> result(sizes.size());
  for (std::size_t cluster = 0; cluster < sizes.size(); ++cluster) {
    const Accumulator& accumulator = accumulators[cluster];
    if (sizes[cluster] <= 0 || accumulator.count != sizes[cluster]) {
      throw std::invalid_argument("make_cluster_markers: cluster size does not match labels");
    }
    const double inv = 1.0 / static_cast<double>(accumulator.count);
    result[cluster].doppler_bin = static_cast<float>(accumulator.doppler_bin * inv);
    result[cluster].range_bin = static_cast<float>(accumulator.range_bin * inv);
    for (std::size_t feature = 0; feature < 2U; ++feature) {
      result[cluster].features[feature] = static_cast<float>(accumulator.features[feature] * inv);
    }
    if (show_angles) {
      const PhysicalAngles angles = accumulator.angle_centroid.mean();
      result[cluster].features[2] = static_cast<float>(angles.az);
      result[cluster].features[3] = static_cast<float>(angles.el);
    }
  }

  std::vector<std::size_t> range_order(result.size());
  for (std::size_t i = 0; i < range_order.size(); ++i) {
    range_order[i] = i;
  }
  std::stable_sort(range_order.begin(), range_order.end(),
                   [&result](std::size_t lhs, std::size_t rhs) {
                     return result[lhs].features[0] < result[rhs].features[0];
                   });
  for (std::size_t rank = 0; rank < range_order.size(); ++rank) {
    result[range_order[rank]].color_bucket = static_cast<int>(rank % kClusterColors.size());
  }
  return result;
}

std::vector<DetectionMarker> make_raw_detection_markers(
    const std::vector<std::int64_t>& detection_idx, const std::vector<float>& points,
    std::int64_t idx_cols, std::int64_t point_cols, const AngleConvention& angle_convention,
    bool show_angles) {
  const std::int64_t min_features = show_angles ? 4 : 2;
  if (idx_cols < 2 || point_cols < min_features ||
      detection_idx.size() % static_cast<std::size_t>(idx_cols) != 0U) {
    throw std::invalid_argument("make_raw_detection_markers: invalid column counts");
  }
  const std::size_t rows = detection_idx.size() / static_cast<std::size_t>(idx_cols);
  if (points.size() != rows * static_cast<std::size_t>(point_cols)) {
    throw std::invalid_argument("make_raw_detection_markers: row counts do not match");
  }

  std::vector<DetectionMarker> result(rows);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t idx_offset = row * static_cast<std::size_t>(idx_cols);
    result[row].doppler_bin =
        static_cast<float>(detection_idx[idx_offset + static_cast<std::size_t>(idx_cols - 2)]);
    result[row].range_bin =
        static_cast<float>(detection_idx[idx_offset + static_cast<std::size_t>(idx_cols - 1)]);
    const std::size_t point_offset = row * static_cast<std::size_t>(point_cols);
    std::copy_n(points.begin() + static_cast<std::ptrdiff_t>(point_offset), 2,
                result[row].features.begin());
    if (show_angles) {
      const PhysicalAngles angles = to_physical_angles(angle_convention, points[point_offset + 2U],
                                                       points[point_offset + 3U]);
      result[row].features[2] = static_cast<float>(angles.az);
      result[row].features[3] = static_cast<float>(angles.el);
    }
  }

  std::vector<std::size_t> range_order(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    range_order[i] = i;
  }
  std::stable_sort(range_order.begin(), range_order.end(),
                   [&result](std::size_t lhs, std::size_t rhs) {
                     return result[lhs].features[0] < result[rhs].features[0];
                   });
  for (std::size_t rank = 0; rank < range_order.size(); ++rank) {
    result[range_order[rank]].color_bucket = static_cast<int>(rank % kClusterColors.size());
  }
  return result;
}

void apply_reduced_features(std::vector<DetectionMarker>* markers,
                            const std::vector<float>& reduced_points, std::int64_t reduced_cols,
                            const AngleConvention& angle_convention, bool show_angles) {
  const std::int64_t min_features = show_angles ? 4 : 2;
  if (markers == nullptr || reduced_cols < min_features ||
      reduced_points.size() != markers->size() * static_cast<std::size_t>(reduced_cols)) {
    throw std::invalid_argument("apply_reduced_features: incompatible reduced points");
  }
  for (std::size_t cluster = 0; cluster < markers->size(); ++cluster) {
    for (std::size_t feature = 0; feature < 2U; ++feature) {
      (*markers)[cluster].features[feature] =
          reduced_points[cluster * static_cast<std::size_t>(reduced_cols) + feature];
    }
    if (show_angles) {
      const std::size_t offset = cluster * static_cast<std::size_t>(reduced_cols);
      const PhysicalAngles angles = to_physical_angles(
          angle_convention, reduced_points[offset + 2U], reduced_points[offset + 3U]);
      (*markers)[cluster].features[2] = static_cast<float>(angles.az);
      (*markers)[cluster].features[3] = static_cast<float>(angles.el);
    }
  }
}

void append_history(std::deque<HistorySample>* history, const std::vector<DetectionMarker>& markers,
                    std::int64_t frame, std::int64_t history_frames) {
  if (history == nullptr || history_frames < 1) {
    throw std::invalid_argument("append_history: invalid history");
  }
  for (const DetectionMarker& marker : markers) {
    history->push_back(HistorySample{
        .frame = frame,
        .doppler_bin = marker.doppler_bin,
        .range_bin = marker.range_bin,
        .features = marker.features,
        .color_bucket = marker.color_bucket,
    });
  }
  const std::int64_t oldest = frame - history_frames + 1;
  while (!history->empty() && history->front().frame < oldest) {
    history->pop_front();
  }
}

std::array<float, 2> rd_plot_position(float doppler_bin, float range_bin, std::int64_t rows,
                                      std::int64_t cols) {
  if (rows <= 0 || cols <= 0) {
    throw std::invalid_argument("rd_plot_position: invalid map shape");
  }
  const float x = (doppler_bin + 0.5F) / static_cast<float>(rows);
  const float y = 1.0F - (range_bin + 0.5F) / static_cast<float>(cols);
  return {std::clamp(x, 0.0F, 1.0F), std::clamp(y, 0.0F, 1.0F)};
}

bool use_azimuth_sector(const std::array<float, 2>& az_bounds) {
  constexpr float kPi = 3.14159265358979323846F;
  constexpr float kTolerance = 1.0e-5F;
  return std::isfinite(az_bounds[0]) && std::isfinite(az_bounds[1]) &&
         az_bounds[0] <= az_bounds[1] && az_bounds[1] - az_bounds[0] <= kPi + kTolerance;
}

std::array<float, 2> history_plot_position(const HistorySample& sample, std::int64_t current_frame,
                                           std::int64_t history_frames, int feature,
                                           std::array<float, 2> bounds) {
  if (history_frames < 2 || feature < 0 || feature >= 4 || !(bounds[0] < bounds[1])) {
    throw std::invalid_argument("history_plot_position: invalid arguments");
  }
  const std::int64_t oldest = current_frame - history_frames + 1;
  const float x =
      static_cast<float>(sample.frame - oldest) / static_cast<float>(history_frames - 1);
  const float y =
      (sample.features[static_cast<std::size_t>(feature)] - bounds[0]) / (bounds[1] - bounds[0]);
  return {std::clamp(x, 0.0F, 1.0F), std::clamp(y, 0.0F, 1.0F)};
}

bool truth_target_is_visible_at_time(const TruthTarget& target, double time,
                                     const sim::SceneVisibility& visibility) {
  if (!std::isfinite(time)) {
    throw std::invalid_argument("truth_target_is_visible_at_time: non-finite time");
  }
  sim::Vec3 position{};
  for (std::size_t axis = 0; axis < position.size(); ++axis) {
    position[axis] = target.position[axis] + time * target.velocity[axis];
  }
  return sim::target_is_visible(position, visibility);
}

std::array<float, 4> truth_features_at_time(const TruthTarget& target, double time) {
  if (!std::isfinite(time)) {
    throw std::invalid_argument("truth_features_at_time: non-finite time");
  }
  std::array<double, 3> position{};
  double range_squared = 0.0;
  for (std::size_t axis = 0; axis < position.size(); ++axis) {
    position[axis] = target.position[axis] + time * target.velocity[axis];
    range_squared += position[axis] * position[axis];
  }
  const double range = std::sqrt(range_squared);
  if (!std::isfinite(range) || range <= 0.0) {
    throw std::invalid_argument("truth_features_at_time: target position has no finite direction");
  }
  const double radial_velocity =
      (target.velocity[0] * position[0] + target.velocity[1] * position[1] +
       target.velocity[2] * position[2]) /
      range;
  const double azimuth = std::atan2(position[1], position[0]);
  const double elevation = std::atan2(position[2], std::hypot(position[0], position[1]));
  return {static_cast<float>(range), static_cast<float>(radial_velocity),
          static_cast<float>(azimuth), static_cast<float>(elevation)};
}

}  // namespace detail
}  // namespace holoscan::radar::app::viz
