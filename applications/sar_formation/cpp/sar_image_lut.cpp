// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "custom_ops.hpp"             // NOLINT(build/include_subdir)
#include "sar_image_lut_kernels.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/app/holoviz_layout.hpp"
#include "radar_app/viz/viz.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace {

int64_t downsample_factor(int64_t height, int64_t width, int64_t max_side) {
  if (max_side <= 0) {
    throw std::invalid_argument("SarImageToLutOp: max_side must be positive");
  }
  const int64_t longest = std::max(height, width);
  if (longest <= max_side) {
    return 1;
  }
  return (longest + max_side - 1) / max_side;
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("SarImageToLutOp: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

std::string format_seconds(double s) {
  if (!std::isfinite(s) || s < 0.0) {
    return "0ms";
  }
  if (s < 1.0) {
    const int ms = static_cast<int>(std::llround(s * 1000.0));
    return std::to_string(ms) + "ms";
  }
  const int scaled = static_cast<int>(std::llround(s * 10.0));
  return std::to_string(scaled / 10) + "." + std::to_string(scaled % 10) + "s";
}

std::string format_sci(double v) {
  if (!std::isfinite(v) || v <= 0.0) {
    return "0";
  }
  int exp = static_cast<int>(std::floor(std::log10(v)));
  double m = v / std::pow(10.0, static_cast<double>(exp));
  int scaled = static_cast<int>(std::llround(m * 100.0));
  if (scaled >= 1000) {
    scaled = 100;
    ++exp;
  }
  const int whole = scaled / 100;
  const int frac = scaled % 100;
  std::string out = std::to_string(whole) + ".";
  if (frac < 10) {
    out += '0';
  }
  out += std::to_string(frac);
  out += 'e';
  if (exp < 0) {
    out += '-';
    exp = -exp;
  }
  out += std::to_string(exp);
  return out;
}

std::shared_ptr<holoscan::Tensor> overlay_tensor(const std::vector<std::string>& lines,
                                                 cudaStream_t stream) {
  constexpr int kScale = 2;
  constexpr int64_t kPad = 3;
  constexpr int64_t kGap = 2;
  const int64_t width = holoscan::radar::app::kSarOverlayWidth;
  const int64_t height = holoscan::radar::app::kSarOverlayHeight;
  std::vector<uint8_t> img(static_cast<size_t>(width * height * 4), 0);
  const std::array<uint8_t, 3> bg{16, 16, 16};
  const std::array<uint8_t, 3> fg{255, 220, 80};
  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      holoscan::radar::app::plot_put_px(&img, width, height, x, y, bg);
    }
  }
  const int64_t line_h = 5 * kScale;
  int64_t y = kPad;
  for (const auto& line : lines) {
    if (y + line_h > height) {
      break;
    }
    holoscan::radar::app::plot_draw_text(&img, width, height, kPad, y, line, fg, kScale);
    y += line_h + kGap;
  }
  auto tensor = radar::make_device_tensor({height, width, 4}, DLDataType{kDLUInt, 8, 1});
  check_cuda(
      cudaMemcpyAsync(tensor->data(), img.data(), img.size(), cudaMemcpyHostToDevice, stream),
      "overlay H2D failed");
  return tensor;
}

void put_rgba(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t x, int64_t y,
              const std::array<uint8_t, 4>& c) {
  if (x < 0 || x >= w || y < 0 || y >= h) {
    return;
  }
  const size_t o = static_cast<size_t>((y * w + x) * 4);
  (*img)[o + 0] = c[0];
  (*img)[o + 1] = c[1];
  (*img)[o + 2] = c[2];
  (*img)[o + 3] = c[3];
}

void draw_line_rgba(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t x0, int64_t y0,
                    int64_t x1, int64_t y1, const std::array<uint8_t, 4>& c) {
  int64_t dx = std::abs(x1 - x0);
  int64_t dy = -std::abs(y1 - y0);
  const int64_t sx = x0 < x1 ? 1 : -1;
  const int64_t sy = y0 < y1 ? 1 : -1;
  int64_t err = dx + dy;
  int64_t x = x0;
  int64_t y = y0;
  while (true) {
    put_rgba(img, w, h, x, y, c);
    put_rgba(img, w, h, x + 1, y, c);
    put_rgba(img, w, h, x, y + 1, c);
    if (x == x1 && y == y1) {
      break;
    }
    const int64_t e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
}

void fill_disc_rgba(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t cx, int64_t cy,
                    int64_t r, const std::array<uint8_t, 4>& c) {
  const int64_t r2 = r * r;
  for (int64_t y = cy - r; y <= cy + r; ++y) {
    for (int64_t x = cx - r; x <= cx + r; ++x) {
      const int64_t dx = x - cx;
      const int64_t dy = y - cy;
      if (dx * dx + dy * dy <= r2) {
        put_rgba(img, w, h, x, y, c);
      }
    }
  }
}

void fill_triangle_rgba(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t x0, int64_t y0,
                        int64_t x1, int64_t y1, int64_t x2, int64_t y2,
                        const std::array<uint8_t, 4>& c) {
  const int64_t xmin = std::min(x0, std::min(x1, x2));
  const int64_t xmax = std::max(x0, std::max(x1, x2));
  const int64_t ymin = std::min(y0, std::min(y1, y2));
  const int64_t ymax = std::max(y0, std::max(y1, y2));
  const auto edge = [](int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t px, int64_t py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
  };
  const int64_t a01 = edge(x0, y0, x1, y1, x2, y2);
  if (a01 == 0) {
    return;
  }
  for (int64_t y = ymin; y <= ymax; ++y) {
    for (int64_t x = xmin; x <= xmax; ++x) {
      const int64_t w0 = edge(x1, y1, x2, y2, x, y);
      const int64_t w1 = edge(x2, y2, x0, y0, x, y);
      const int64_t w2 = edge(x0, y0, x1, y1, x, y);
      if (a01 > 0) {
        if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
          put_rgba(img, w, h, x, y, c);
        }
      } else if (w0 <= 0 && w1 <= 0 && w2 <= 0) {
        put_rgba(img, w, h, x, y, c);
      }
    }
  }
}

std::shared_ptr<holoscan::Tensor> compass_tensor(bool valid, double east, double north,
                                                 cudaStream_t stream) {
  constexpr int64_t kSide = holoscan::radar::app::kSarCompassSide;
  constexpr int64_t kCx = 40;
  constexpr int64_t kCy = 40;
  constexpr int64_t kDisc = 36;
  constexpr int64_t kShaft = 22;
  std::vector<uint8_t> img(static_cast<size_t>(kSide * kSide * 4), 0);
  const std::array<uint8_t, 4> bg{16, 16, 16, 255};
  const std::array<uint8_t, 4> ring{90, 90, 90, 255};
  const std::array<uint8_t, 4> fg{255, 220, 80, 255};
  for (int64_t y = 0; y < kSide; ++y) {
    for (int64_t x = 0; x < kSide; ++x) {
      put_rgba(&img, kSide, kSide, x, y, bg);
    }
  }
  fill_disc_rgba(&img, kSide, kSide, kCx, kCy, kDisc, ring);
  fill_disc_rgba(&img, kSide, kSide, kCx, kCy, kDisc - 2, bg);
  const std::array<uint8_t, 3> n_rgb{220, 220, 220};
  holoscan::radar::app::plot_draw_text(&img, kSide, kSide, kCx - 1, 8, "N", n_rgb, 1);
  if (valid) {
    const double len = std::hypot(east, north);
    if (len > 0.0) {
      const double ux = east / len;
      const double uy = -north / len;
      const int64_t tx = kCx + static_cast<int64_t>(std::llround(ux * kShaft));
      const int64_t ty = kCy + static_cast<int64_t>(std::llround(uy * kShaft));
      draw_line_rgba(&img, kSide, kSide, kCx, kCy, tx, ty, fg);
      const double px = -uy;
      const double py = ux;
      const int64_t bx = tx - static_cast<int64_t>(std::llround(ux * 9.0));
      const int64_t by = ty - static_cast<int64_t>(std::llround(uy * 9.0));
      const int64_t l0 = bx + static_cast<int64_t>(std::llround(px * 5.0));
      const int64_t l1 = by + static_cast<int64_t>(std::llround(py * 5.0));
      const int64_t r0 = bx - static_cast<int64_t>(std::llround(px * 5.0));
      const int64_t r1 = by - static_cast<int64_t>(std::llround(py * 5.0));
      fill_triangle_rgba(&img, kSide, kSide, tx, ty, l0, l1, r0, r1, fg);
    }
  }
  fill_disc_rgba(&img, kSide, kSide, kCx, kCy, 2, fg);
  auto tensor = radar::make_device_tensor({kSide, kSide, 4}, DLDataType{kDLUInt, 8, 1});
  check_cuda(
      cudaMemcpyAsync(tensor->data(), img.data(), img.size(), cudaMemcpyHostToDevice, stream),
      "compass H2D failed");
  return tensor;
}

void put_u8(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t x, int64_t y, uint8_t v) {
  if (x < 0 || x >= w || y < 0 || y >= h) {
    return;
  }
  (*img)[static_cast<size_t>(y * w + x)] = v;
}

void fill_triangle_u8(std::vector<uint8_t>* img, int64_t w, int64_t h, int64_t x0, int64_t y0,
                      int64_t x1, int64_t y1, int64_t x2, int64_t y2, uint8_t v) {
  const int64_t xmin = std::min(x0, std::min(x1, x2));
  const int64_t xmax = std::max(x0, std::max(x1, x2));
  const int64_t ymin = std::min(y0, std::min(y1, y2));
  const int64_t ymax = std::max(y0, std::max(y1, y2));
  const auto edge = [](int64_t ax, int64_t ay, int64_t bx, int64_t by, int64_t px, int64_t py) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
  };
  const int64_t a01 = edge(x0, y0, x1, y1, x2, y2);
  if (a01 == 0) {
    return;
  }
  for (int64_t y = ymin; y <= ymax; ++y) {
    for (int64_t x = xmin; x <= xmax; ++x) {
      const int64_t w0 = edge(x1, y1, x2, y2, x, y);
      const int64_t w1 = edge(x2, y2, x0, y0, x, y);
      const int64_t w2 = edge(x0, y0, x1, y1, x, y);
      if (a01 > 0) {
        if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
          put_u8(img, w, h, x, y, v);
        }
      } else if (w0 <= 0 && w1 <= 0 && w2 <= 0) {
        put_u8(img, w, h, x, y, v);
      }
    }
  }
}

void draw_border_chevron(std::vector<uint8_t>* img, int64_t h, int64_t w, double east,
                         double north) {
  const double len = std::hypot(east, north);
  if (!(len > 0.0) || h < 32 || w < 32) {
    return;
  }
  const double dx = east / len;
  const double dy = -north / len;
  const double cx = 0.5 * static_cast<double>(w - 1);
  const double cy = 0.5 * static_cast<double>(h - 1);
  const double x0 = 16.0;
  const double y0 = 16.0;
  const double x1 = static_cast<double>(w - 17);
  const double y1 = static_cast<double>(h - 17);
  double t_hit = 1.0e300;
  double hx = cx;
  double hy = cy;
  auto consider_x = [&](double xe) {
    if (dx == 0.0) {
      return;
    }
    const double t = (xe - cx) / dx;
    if (!(t > 0.0) || t >= t_hit) {
      return;
    }
    const double y = cy + t * dy;
    if (y >= y0 && y <= y1) {
      t_hit = t;
      hx = xe;
      hy = y;
    }
  };
  auto consider_y = [&](double ye) {
    if (dy == 0.0) {
      return;
    }
    const double t = (ye - cy) / dy;
    if (!(t > 0.0) || t >= t_hit) {
      return;
    }
    const double x = cx + t * dx;
    if (x >= x0 && x <= x1) {
      t_hit = t;
      hx = x;
      hy = ye;
    }
  };
  consider_x(x0);
  consider_x(x1);
  consider_y(y0);
  consider_y(y1);
  if (!(t_hit < 1.0e299)) {
    return;
  }
  const int64_t tx = static_cast<int64_t>(std::llround(hx));
  const int64_t ty = static_cast<int64_t>(std::llround(hy));
  const double px = -dy;
  const double py = dx;
  const int64_t bx = tx - static_cast<int64_t>(std::llround(dx * 16.0));
  const int64_t by = ty - static_cast<int64_t>(std::llround(dy * 16.0));
  const int64_t l0 = bx + static_cast<int64_t>(std::llround(px * 8.0));
  const int64_t l1 = by + static_cast<int64_t>(std::llround(py * 8.0));
  const int64_t r0 = bx - static_cast<int64_t>(std::llround(px * 8.0));
  const int64_t r1 = by - static_cast<int64_t>(std::llround(py * 8.0));
  fill_triangle_u8(img, w, h, tx, ty, l0, l1, r0, r1, 0);
  const int64_t bx2 = tx - static_cast<int64_t>(std::llround(dx * 12.0));
  const int64_t by2 = ty - static_cast<int64_t>(std::llround(dy * 12.0));
  const int64_t l2 = bx2 + static_cast<int64_t>(std::llround(px * 5.0));
  const int64_t l3 = by2 + static_cast<int64_t>(std::llround(py * 5.0));
  const int64_t r2 = bx2 - static_cast<int64_t>(std::llround(px * 5.0));
  const int64_t r3 = by2 - static_cast<int64_t>(std::llround(py * 5.0));
  fill_triangle_u8(img, w, h, tx, ty, l2, l3, r2, r3, 255);
}

}  // namespace

void SarImageToLutOp::compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
                              [[maybe_unused]] holoscan::ExecutionContext& context) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto elapsed = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };
  const char* who = "SarImageToLutOp";
  auto image = radar::receive_single(op_input, "image", who);
  radar::require_cuda_accessible(*image, "image", who);
  radar::require_contiguous(*image, "image", who);
  const auto dtype = image->dtype();
  const bool is_c64 = dtype.code == kDLComplex && dtype.bits == 64 && dtype.lanes == 1;
  const bool is_c128 = dtype.code == kDLComplex && dtype.bits == 128 && dtype.lanes == 1;
  if (!is_c64 && !is_c128) {
    throw std::invalid_argument(std::string(who) + ": image must be complex64 or complex128");
  }
  const auto shape = image->shape();
  if (shape.size() != 2) {
    throw std::invalid_argument(std::string(who) + ": image must be rank-2 (num_v, num_u)");
  }
  const int64_t num_v = shape[0];
  const int64_t num_u = shape[1];
  if (num_v <= 0 || num_u <= 0) {
    throw std::invalid_argument(std::string(who) + ": image axes must be positive");
  }
  if (!(floor_db < 0.0) || !std::isfinite(floor_db)) {
    throw std::invalid_argument(std::string(who) + ": floor_db must be finite and < 0");
  }
  if (!(high_percentile > 0.0) || high_percentile >= 100.0 || !std::isfinite(high_percentile)) {
    throw std::invalid_argument(std::string(who) + ": high_percentile must be in (0, 100)");
  }
  if (low_percentile < 0.0 || low_percentile >= high_percentile || !std::isfinite(low_percentile)) {
    throw std::invalid_argument(std::string(who) +
                                ": low_percentile must be in [0, high_percentile)");
  }

  cudaStream_t stream = op_input.receive_cuda_stream("image");
  const size_t count = static_cast<size_t>(num_v) * static_cast<size_t>(num_u);
  auto mag = radar::make_device_tensor({num_v, num_u}, radar::real_dtype<float>());
  if (is_c64) {
    sar_image_magnitude_c64(image->data(), static_cast<float*>(mag->data()), count, stream);
  } else {
    sar_image_magnitude_c128(image->data(), static_cast<float*>(mag->data()), count, stream);
  }
  check_cuda(cudaGetLastError(), "magnitude kernel failed");
  int64_t disp_u = num_u;
  if (crop_nu > 0) {
    if (crop_u0 < 0 || crop_nu < 1 || crop_u0 + crop_nu > num_u) {
      throw std::invalid_argument(std::string(who) + ": crop_u [" + std::to_string(crop_u0) + ", " +
                                  std::to_string(crop_u0 + crop_nu) + ") is outside image u " +
                                  std::to_string(num_u));
    }
    auto cropped = radar::make_device_tensor({num_v, crop_nu}, radar::real_dtype<float>());
    const size_t dst_pitch = static_cast<size_t>(crop_nu) * sizeof(float);
    const size_t src_pitch = static_cast<size_t>(num_u) * sizeof(float);
    const float* src = static_cast<const float*>(mag->data()) + crop_u0;
    check_cuda(cudaMemcpy2DAsync(cropped->data(), dst_pitch, src, src_pitch, dst_pitch,
                                 static_cast<size_t>(num_v), cudaMemcpyDeviceToDevice, stream),
               "u-crop memcpy failed");
    mag = cropped;
    disp_u = crop_nu;
  }
  const bool landscape = num_v > disp_u;
  int64_t grid_v = num_v;
  int64_t grid_u = disp_u;
  std::shared_ptr<holoscan::Tensor> transposed;
  if (landscape) {
    transposed = radar::make_device_tensor({disp_u, num_v}, radar::real_dtype<float>());
    sar_image_transpose(static_cast<float*>(mag->data()), static_cast<float*>(transposed->data()),
                        num_v, disp_u, stream);
    check_cuda(cudaGetLastError(), "transpose kernel failed");
    mag = transposed;
    grid_v = disp_u;
    grid_u = num_v;
  }
  const double mag_s = elapsed();

  const int64_t factor = downsample_factor(grid_v, grid_u, max_side);
  const int64_t out_v = grid_v / factor;
  const int64_t out_u = grid_u / factor;
  if (out_v <= 0 || out_u <= 0) {
    throw std::invalid_argument(std::string(who) + ": max_side leaves an empty display grid");
  }
  const size_t out_count = static_cast<size_t>(out_v) * static_cast<size_t>(out_u);
  std::shared_ptr<holoscan::Tensor> displayed;
  float* grid = static_cast<float*>(mag->data());
  if (factor > 1) {
    displayed = radar::make_device_tensor({out_v, out_u}, radar::real_dtype<float>());
    sar_image_box_average(grid, static_cast<float*>(displayed->data()), grid_v, grid_u, factor,
                          stream);
    check_cuda(cudaGetLastError(), "box-average kernel failed");
    grid = static_cast<float*>(displayed->data());
  }
  const double box_s = elapsed() - mag_s;

  float high = 0.0F;
  float low_pct = 0.0F;
  size_t n_valid = 0;
  check_cuda(sar_image_percentiles(grid, out_count, high_percentile, low_percentile, &high,
                                   &low_pct, &n_valid, stream),
             "GPU percentile failed");
  if (n_valid == 0 || !(high > 0.0F)) {
    throw std::runtime_error(std::string(who) + ": image is identically zero");
  }
  const float low_from_db = high * std::pow(10.0F, static_cast<float>(floor_db) / 20.0F);
  float low = low_from_db;
  if (low_percentile > 0.0) {
    low = std::max(low_pct, low_from_db);
  }
  if (!(low > 0.0F) || !(low < high)) {
    low = low_from_db;
  }
  const float floor = 20.0F * std::log10(low / high);
  if (!(floor < 0.0F)) {
    throw std::runtime_error(std::string(who) + ": display window is empty");
  }
  const double pct_s = elapsed() - mag_s - box_s;
  if (factor > 1) {
    HOLOSCAN_LOG_INFO(
        "{}: formed {} x {}, display {} x {}, high={:.4e}, "
        "window={:.1f} dB",
        who, num_v, num_u, out_v, out_u, static_cast<double>(high), static_cast<double>(floor));
  } else {
    HOLOSCAN_LOG_INFO("{}: formed {} x {}, high={:.4e}, window={:.1f} dB", who, num_v, num_u,
                      static_cast<double>(high), static_cast<double>(floor));
  }

  auto tensor = radar::make_device_tensor({out_v, out_u, 1}, DLDataType{kDLUInt, 8, 1});
  sar_image_apply_lut(grid, static_cast<std::uint8_t*>(tensor->data()), out_count, high, floor,
                      stream);
  check_cuda(cudaGetLastError(), "LUT kernel failed");
  FormationDiag diag{};
  if (formation_diag) {
    diag = formation_diag();
  }
  if (diag.apc_valid) {
    std::vector<uint8_t> host(out_count);
    check_cuda(
        cudaMemcpyAsync(host.data(), tensor->data(), out_count, cudaMemcpyDeviceToHost, stream),
        "LUT D2H for APC mark failed");
    check_cuda(cudaStreamSynchronize(stream), "LUT D2H sync failed");
    // Chevron in display pixels: +x is right. Landscape swaps
    // (u,v) so along-track (v) is horizontal.
    const double chev_e = landscape ? diag.raster_v : diag.raster_u;
    const double chev_n = landscape ? -diag.raster_u : -diag.raster_v;
    draw_border_chevron(&host, out_v, out_u, chev_e, chev_n);
    check_cuda(
        cudaMemcpyAsync(tensor->data(), host.data(), out_count, cudaMemcpyHostToDevice, stream),
        "LUT H2D for APC mark failed");
  }
  std::vector<std::string> lines;
  const std::string size = std::to_string(num_v) + "x" + std::to_string(num_u);
  if (!scene_names.empty()) {
    const size_t idx = static_cast<size_t>(overlay_index) % scene_names.size();
    lines.push_back(scene_names[idx] + "  " + size);
  } else {
    lines.push_back(size);
  }
  ++overlay_index;
  std::string rate_line;
  if (diag.bp_compute_s > 0.0 && diag.pulses > 0 && diag.pixels > 0) {
    const double rate =
        static_cast<double>(diag.pixels) * static_cast<double>(diag.pulses) / diag.bp_compute_s;
    rate_line = format_seconds(diag.bp_compute_s) + "  " + format_sci(rate) + " px*pls/s";
    // Mosaic canvas is the LUT input; BP formed a tile. Name the
    // lattice the rate is for so the parent strip is not read
    // as the tile throughput.
    if (diag.bp_num_v > 0 && diag.bp_num_u > 0 &&
        (diag.bp_num_v != num_v || diag.bp_num_u != num_u)) {
      rate_line =
          std::to_string(diag.bp_num_v) + "x" + std::to_string(diag.bp_num_u) + "  " + rate_line;
    }
  }
  const auto overlay_now = std::chrono::steady_clock::now();
  if (!rate_line.empty() &&
      (!have_overlay_rate_ ||
       std::chrono::duration<double>(overlay_now - overlay_rate_at_).count() >= 1.0)) {
    overlay_rate_at_ = overlay_now;
    have_overlay_rate_ = true;
    held_rate_line_ = rate_line;
  }
  if (!held_rate_line_.empty()) {
    lines.push_back(held_rate_line_);
  }
  std::optional<double> form_latency;
  if (diag.last_packet_at.has_value()) {
    const auto emit_at = diag.emit_at.value_or(t0);
    form_latency = std::chrono::duration<double>(emit_at - *diag.last_packet_at).count();
  }
  std::shared_ptr<holoscan::Tensor> label = held_overlay_;
  if (!held_overlay_ || lines != last_overlay_lines_) {
    last_overlay_lines_ = lines;
    held_overlay_ = overlay_tensor(lines, stream);
    label = held_overlay_;
  }
  auto look = compass_tensor(diag.apc_valid, diag.compass_east, diag.compass_north, stream);
  check_cuda(cudaStreamSynchronize(stream), "LUT H2D sync failed");
  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "sar_image");
  }
  holoscan::TensorMap map;
  map.insert({"sar_image", std::move(tensor)});
  map.insert({"size_label", std::move(label)});
  map.insert({"apc_look", std::move(look)});
  op_output.emit(map, "sar_image");
  const auto now = std::chrono::steady_clock::now();
  if (have_last_emit_) {
    HOLOSCAN_LOG_INFO("{}: {:.3f} s wall since previous image", who,
                      std::chrono::duration<double>(now - last_emit_at_).count());
  }
  last_emit_at_ = now;
  have_last_emit_ = true;
  if (form_latency.has_value()) {
    HOLOSCAN_LOG_INFO("{}: image {:.3f} s after last pulse packet", who, *form_latency);
  }
  if (diag.playback.valid && diag.playback.paced) {
    HOLOSCAN_LOG_INFO(
        "{}: playback collect {:.3f} s  wall {:.3f} s  "
        "behind {:.3f} s",
        who, diag.playback.collect_s, diag.playback.wall_s, diag.playback.behind_s);
  }
  if (diag.apc_valid) {
    HOLOSCAN_LOG_INFO("{}: APC look  E {:.1f} km  N {:.1f} km", who, diag.compass_east / 1000.0,
                      diag.compass_north / 1000.0);
  }
  const double total_s = elapsed();
  const double lut_s = total_s - mag_s - box_s - pct_s;
  HOLOSCAN_LOG_INFO(
      "{}: {}x{} display {}x{} in {:.3f} s (mag {:.3f}, box {:.3f}, "
      "pct {:.3f}, lut {:.3f})",
      who, num_v, num_u, out_v, out_u, total_s, mag_s, box_s, pct_s, lut_s);
}
