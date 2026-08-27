// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
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

#include "custom_ops.hpp"  // NOLINT(build/include_subdir)
#include "sar_image_lut_kernels.hpp"  // NOLINT(build/include_subdir)
#include "radar_app/viz/viz.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace {

int64_t downsample_factor(int64_t height, int64_t width,
                          int64_t max_side) {
  if (max_side <= 0) {
    throw std::invalid_argument(
        "SarImageToLutOp: max_side must be positive");
  }
  const int64_t longest = std::max(height, width);
  if (longest <= max_side) {
    return 1;
  }
  return (longest + max_side - 1) / max_side;
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("SarImageToLutOp: ") + what +
                             ": " + cudaGetErrorString(err));
  }
}

std::string format_seconds(double s) {
  if (!std::isfinite(s) || s < 0.0) {
    return "0.0s";
  }
  const int scaled =
      static_cast<int>(std::llround(s * 10.0));
  return std::to_string(scaled / 10) + "." +
         std::to_string(scaled % 10) + "s";
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

std::shared_ptr<holoscan::Tensor> overlay_tensor(
    const std::vector<std::string>& lines, cudaStream_t stream) {
  constexpr int kScale = 1;
  constexpr int64_t kPad = 3;
  constexpr int64_t kGap = 2;
  int64_t tw = 0;
  for (const auto& line : lines) {
    tw = std::max(tw, holoscan::radar::app::plot_text_width(
                          line, kScale));
  }
  const int64_t n = static_cast<int64_t>(lines.size());
  const int64_t line_h = 5 * kScale;
  const int64_t th =
      n * line_h + (n > 1 ? (n - 1) * kGap : 0);
  const int64_t width = std::max(int64_t{1}, tw + 2 * kPad);
  const int64_t height = std::max(int64_t{1}, th + 2 * kPad);
  std::vector<uint8_t> img(
      static_cast<size_t>(width * height * 4), 0);
  const std::array<uint8_t, 3> bg{16, 16, 16};
  const std::array<uint8_t, 3> fg{255, 220, 80};
  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      holoscan::radar::app::plot_put_px(&img, width, height, x, y,
                                        bg);
    }
  }
  int64_t y = kPad;
  for (const auto& line : lines) {
    holoscan::radar::app::plot_draw_text(
        &img, width, height, kPad, y, line, fg, kScale);
    y += line_h + kGap;
  }
  auto tensor = radar::make_device_tensor(
      {height, width, 4}, DLDataType{kDLUInt, 8, 1});
  check_cuda(
      cudaMemcpyAsync(tensor->data(), img.data(), img.size(),
                      cudaMemcpyHostToDevice, stream),
      "overlay H2D failed");
  return tensor;
}

}  // namespace

void SarImageToLutOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto elapsed = [&]() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - t0)
        .count();
  };
  const char* who = "SarImageToLutOp";
  auto image = radar::receive_single(op_input, "image", who);
  radar::require_cuda_accessible(*image, "image", who);
  radar::require_contiguous(*image, "image", who);
  const auto dtype = image->dtype();
  const bool is_c64 =
      dtype.code == kDLComplex && dtype.bits == 64 && dtype.lanes == 1;
  const bool is_c128 =
      dtype.code == kDLComplex && dtype.bits == 128 &&
      dtype.lanes == 1;
  if (!is_c64 && !is_c128) {
    throw std::invalid_argument(
        std::string(who) + ": image must be complex64 or complex128");
  }
  const auto shape = image->shape();
  if (shape.size() != 2) {
    throw std::invalid_argument(
        std::string(who) + ": image must be rank-2 (num_v, num_u)");
  }
  const int64_t num_v = shape[0];
  const int64_t num_u = shape[1];
  if (num_v <= 0 || num_u <= 0) {
    throw std::invalid_argument(
        std::string(who) + ": image axes must be positive");
  }
  if (!(floor_db < 0.0) || !std::isfinite(floor_db)) {
    throw std::invalid_argument(
        std::string(who) + ": floor_db must be finite and < 0");
  }
  if (!(high_percentile > 0.0) || high_percentile >= 100.0 ||
      !std::isfinite(high_percentile)) {
    throw std::invalid_argument(
        std::string(who) +
        ": high_percentile must be in (0, 100)");
  }
  if (low_percentile < 0.0 || low_percentile >= high_percentile ||
      !std::isfinite(low_percentile)) {
    throw std::invalid_argument(
        std::string(who) +
        ": low_percentile must be in [0, high_percentile)");
  }

  cudaStream_t stream = op_input.receive_cuda_stream("image");
  const size_t count =
      static_cast<size_t>(num_v) * static_cast<size_t>(num_u);
  auto mag = radar::make_device_tensor({num_v, num_u},
                                       radar::real_dtype<float>());
  if (is_c64) {
    sar_image_magnitude_c64(image->data(),
                            static_cast<float*>(mag->data()), count,
                            stream);
  } else {
    sar_image_magnitude_c128(image->data(),
                             static_cast<float*>(mag->data()), count,
                             stream);
  }
  check_cuda(cudaGetLastError(), "magnitude kernel failed");
  const double mag_s = elapsed();

  const int64_t factor =
      downsample_factor(num_v, num_u, max_side);
  const int64_t out_v = num_v / factor;
  const int64_t out_u = num_u / factor;
  if (out_v <= 0 || out_u <= 0) {
    throw std::invalid_argument(
        std::string(who) + ": max_side leaves an empty display grid");
  }
  const size_t out_count =
      static_cast<size_t>(out_v) * static_cast<size_t>(out_u);
  std::shared_ptr<holoscan::Tensor> displayed;
  float* grid = static_cast<float*>(mag->data());
  if (factor > 1) {
    displayed = radar::make_device_tensor(
        {out_v, out_u}, radar::real_dtype<float>());
    sar_image_box_average(grid,
                          static_cast<float*>(displayed->data()),
                          num_v, num_u, factor, stream);
    check_cuda(cudaGetLastError(), "box-average kernel failed");
    grid = static_cast<float*>(displayed->data());
  }
  const double box_s = elapsed() - mag_s;

  float high = 0.0F;
  float low_pct = 0.0F;
  size_t n_valid = 0;
  check_cuda(sar_image_percentiles(grid, out_count, high_percentile,
                                   low_percentile, &high, &low_pct,
                                   &n_valid, stream),
             "GPU percentile failed");
  if (n_valid == 0 || !(high > 0.0F)) {
    throw std::runtime_error(
        std::string(who) + ": image is identically zero");
  }
  const float low_from_db =
      high * std::pow(10.0F, static_cast<float>(floor_db) / 20.0F);
  float low = low_from_db;
  if (low_percentile > 0.0) {
    low = std::max(low_pct, low_from_db);
  }
  if (!(low > 0.0F) || !(low < high)) {
    low = low_from_db;
  }
  const float floor = 20.0F * std::log10(low / high);
  if (!(floor < 0.0F)) {
    throw std::runtime_error(
        std::string(who) + ": display window is empty");
  }
  const double pct_s = elapsed() - mag_s - box_s;
  if (factor > 1) {
    HOLOSCAN_LOG_INFO(
        "{}: formed {} x {}, display {} x {}, high={:.4e}, "
        "window={:.1f} dB",
        who, num_v, num_u, out_v, out_u,
        static_cast<double>(high), static_cast<double>(floor));
  } else {
    HOLOSCAN_LOG_INFO(
        "{}: formed {} x {}, high={:.4e}, window={:.1f} dB",
        who, num_v, num_u, static_cast<double>(high),
        static_cast<double>(floor));
  }

  auto tensor = radar::make_device_tensor(
      {out_v, out_u, 1}, DLDataType{kDLUInt, 8, 1});
  sar_image_apply_lut(grid, static_cast<std::uint8_t*>(tensor->data()),
                      out_count, high, floor, stream);
  check_cuda(cudaGetLastError(), "LUT kernel failed");
  std::vector<std::string> lines;
  const std::string size = std::to_string(num_v) + "x" +
                           std::to_string(num_u);
  if (!scene_names.empty()) {
    const size_t idx =
        static_cast<size_t>(overlay_index) % scene_names.size();
    lines.push_back(scene_names[idx] + "  " + size);
  } else {
    lines.push_back(size);
  }
  FormationDiag diag{};
  if (formation_diag) {
    diag = formation_diag();
  }
  ++overlay_index;
  if (diag.bp_compute_s > 0.0 && diag.pulses > 0 &&
      diag.pixels > 0) {
    const double rate = static_cast<double>(diag.pixels) *
                        static_cast<double>(diag.pulses) /
                        diag.bp_compute_s;
    lines.push_back(format_seconds(diag.bp_compute_s) + "  " +
                    format_sci(rate) + " px*pls/s");
  }
  std::optional<double> form_latency;
  if (diag.last_packet_at.has_value()) {
    const auto emit_at = diag.emit_at.value_or(t0);
    form_latency = std::chrono::duration<double>(
                       emit_at - *diag.last_packet_at)
                       .count();
  }
  auto label = overlay_tensor(lines, stream);
  check_cuda(cudaStreamSynchronize(stream), "LUT H2D sync failed");
  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "sar_image");
  }
  holoscan::TensorMap map;
  map.insert({"sar_image", std::move(tensor)});
  map.insert({"size_label", std::move(label)});
  op_output.emit(map, "sar_image");
  const auto now = std::chrono::steady_clock::now();
  if (have_last_emit_) {
    HOLOSCAN_LOG_INFO(
        "{}: {:.3f} s wall since previous image",
        who,
        std::chrono::duration<double>(now - last_emit_at_).count());
  }
  last_emit_at_ = now;
  have_last_emit_ = true;
  if (form_latency.has_value()) {
    HOLOSCAN_LOG_INFO(
        "{}: image {:.3f} s after last pulse packet",
        who, *form_latency);
  }
  const double total_s = elapsed();
  const double lut_s = total_s - mag_s - box_s - pct_s;
  HOLOSCAN_LOG_INFO(
      "{}: {}x{} display {}x{} in {:.3f} s (mag {:.3f}, box {:.3f}, "
      "pct {:.3f}, lut {:.3f})",
      who, num_v, num_u, out_v, out_u, total_s, mag_s, box_s, pct_s,
      lut_s);
}
