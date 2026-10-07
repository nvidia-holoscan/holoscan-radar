// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar_image_viz.hpp"          // NOLINT(build/include_subdir)
#include "sar_image_lut_kernels.hpp"  // NOLINT(build/include_subdir)

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>

#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar::app::sar {
namespace {
template <typename T>
__global__ void sample_magnitude(const T* input, int64_t source_v, int64_t source_u, float* output,
                                 int64_t height, int64_t width, bool transpose_axes) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= height * width)
    return;
  const int64_t y = i / width, x = i % width;
  // The scrolling strip places along-track v on the horizontal display axis.
  const int64_t display_v = transpose_axes ? x : y;
  const int64_t display_u = transpose_axes ? y : x;
  const int64_t display_v_extent = transpose_axes ? width : height;
  const int64_t display_u_extent = transpose_axes ? height : width;
  const int64_t v0 = display_v * source_v / display_v_extent;
  const int64_t v1 = max(v0 + 1, (display_v + 1) * source_v / display_v_extent);
  const int64_t u0 = display_u * source_u / display_u_extent;
  const int64_t u1 = max(u0 + 1, (display_u + 1) * source_u / display_u_extent);
  double sum = 0.0;
  int64_t count = 0;
  for (int64_t sv = v0; sv < v1; ++sv) {
    for (int64_t su = u0; su < u1; ++su) {
      const T z = input[sv * source_u + su];
      const double magnitude = hypot(static_cast<double>(z.x), static_cast<double>(z.y));
      if (isfinite(magnitude) && magnitude > 0.0) {
        sum += magnitude;
        ++count;
      }
    }
  }
  output[i] = count ? static_cast<float>(sum / static_cast<double>(count)) : 0.0f;
}
__global__ void render_gray(const float* magnitude, std::uint8_t* output, int64_t count, float high,
                            float floor_db) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float db = 20.0f * log10f(max(magnitude[i], 1.0e-20f) / high);
  const auto gray =
      static_cast<std::uint8_t>(255.0f * fminf(1.0f, fmaxf(0.0f, (db - floor_db) / -floor_db)));
  output[4 * i] = gray;
  output[4 * i + 1] = gray;
  output[4 * i + 2] = gray;
  output[4 * i + 3] = 255U;
}
}  // namespace

SarImageVizOp::SarImageVizOp(int64_t max_v, int64_t max_u, int64_t display_side,
                             bool transpose_axes, OverlayConfig overlay)
: max_v_(max_v),
  max_u_(max_u),
  display_side_(display_side),
  transpose_axes_(transpose_axes),
  overlay_(std::move(overlay)) {
  if (max_v_ < 1 || max_u_ < 1 || display_side_ < 1)
    throw std::invalid_argument("SarImageVizOp: invalid image extents");
}
SarImageVizOp::~SarImageVizOp() {
  if (sampled_magnitude_ != nullptr)
    (void)cudaFree(sampled_magnitude_);
}
void SarImageVizOp::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  constexpr TensorRepresentation rgba{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 3U};
  constexpr TensorRepresentation hud_text{
      .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLUInt, 8U, 1U}, .rank = 2U};
  constexpr TensorRepresentation hud_geom{
      .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLFloat, 32U, 1U}, .rank = 1U};
  spec.input(image).queue_depth(4U).expects_tensor(tensor_input(device));
  spec.output(frame)
      .tensor_field(schema::ImageTensorFields::data)
      .produces_tensor(pooled_tensor_output(
          rgba, 3U,
          static_cast<uint64_t>(display_side_) * static_cast<uint64_t>(display_side_) * 4U));
  spec.output(labels).produces_tensor(
      pooled_tensor_output(hud_text, 2U, static_cast<uint64_t>(kHudRuns) * kHudRowBytes));
  spec.output(overlay).produces_tensor(pooled_tensor_output(
      hud_geom, 1U, static_cast<uint64_t>(kHudMaxLines) * kHudLineFloats * sizeof(float)));
}
Contract SarImageVizOp::contract() const {
  Contract result;
  result.trigger(OnEach{image});
  return result;
}
void SarImageVizOp::start() {
  if (sampled_magnitude_ == nullptr &&
      cudaMalloc(&sampled_magnitude_,
                 static_cast<size_t>(display_side_ * display_side_) * sizeof(float)) != cudaSuccess)
    throw std::runtime_error("SarImageVizOp: sampled magnitude allocation failed");
}
expected<void, Error> SarImageVizOp::compute(ExecutionContext& context) {
  // Receive and validate inputs
  HOLOSCAN_TRY(auto input, receive_tensor(image, "SarImageVizOp"));
  const auto shape = input.shape();
  const auto dtype = input.dtype();
  if (shape.size() != 2U || shape[0] < 1 || shape[1] < 1 || shape[0] > max_v_ ||
      shape[1] > max_u_ || dtype.code != kDLComplex || (dtype.bits != 64U && dtype.bits != 128U))
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "SarImageVizOp: expected bounded complex image"});
  const int64_t longest = std::max(shape[0], shape[1]);
  const int64_t source_height = transpose_axes_ ? shape[1] : shape[0];
  const int64_t source_width = transpose_axes_ ? shape[0] : shape[1];
  const int64_t height = std::max<int64_t>(1, source_height * display_side_ / longest);
  const int64_t width = std::max<int64_t>(1, source_width * display_side_ / longest);
  const auto stream = context.cuda_stream().get();
  constexpr int kThreads = 256;
  const int64_t output_count = height * width;
  // Execute the device stage
  if (dtype.bits == 128U)
    sample_magnitude<<<(output_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const double2*>(input.data()), shape[0], shape[1], sampled_magnitude_, height,
        width, transpose_axes_);
  else
    sample_magnitude<<<(output_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const float2*>(input.data()), shape[0], shape[1], sampled_magnitude_, height,
        width, transpose_axes_);
  if (cudaPeekAtLastError() != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "SarImageVizOp: sample kernel failed"});
  float high = 0.0f, low = 0.0f;
  size_t valid = 0;
  if (sar_image_percentiles(sampled_magnitude_, static_cast<size_t>(output_count), 99.8, 25.0,
                            &high, &low, &valid, stream) != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "SarImageVizOp: percentile sort failed"});
  high = valid ? std::max(high, 1.0e-20f) : 1.0f;
  const float floor_db =
      valid ? std::min(-0.001f, std::max(-30.0f, 20.0f * log10f(low / high))) : -30.0f;
  auto loan = frame.allocate_tensor(TensorLoanRequest{
      .shape = std::vector<int64_t>{height, width, 4}, .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!loan)
    return make_unexpected(std::move(loan).error());
  auto writer = loan->write(context.cuda_stream());
  if (!writer)
    return make_unexpected(std::move(writer).error());
  render_gray<<<(output_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      sampled_magnitude_, static_cast<std::uint8_t*>(writer->data()), output_count, high, floor_db);
  if (cudaPeekAtLastError() != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "SarImageVizOp: render kernel failed"});
  HOLOSCAN_RETURN_IF_ERROR(std::move(*writer).commit());

  // Publish outputs: HUD tensors before the ImageT trigger
  FormationDiag diag{};
  if (overlay_.diag)
    diag = overlay_.diag();
  std::vector<std::string> banner;
  const std::string size = std::to_string(shape[0]) + "x" + std::to_string(shape[1]);
  if (!overlay_.scene_names.empty()) {
    const size_t idx = static_cast<size_t>(overlay_index_) % overlay_.scene_names.size();
    banner.push_back(overlay_.scene_names[idx] + "  " + size);
  } else {
    banner.push_back(size);
  }
  ++overlay_index_;
  std::string rate_line;
  if (diag.bp_compute_s > 0.0 && diag.pulses > 0 && diag.pixels > 0) {
    const double rate =
        static_cast<double>(diag.pixels) * static_cast<double>(diag.pulses) / diag.bp_compute_s;
    rate_line = format_seconds(diag.bp_compute_s) + "  " + format_sci(rate) + " px*pls/s";
    if (diag.bp_num_v > 0 && diag.bp_num_u > 0 &&
        (diag.bp_num_v != shape[0] || diag.bp_num_u != shape[1])) {
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
  if (!held_rate_line_.empty())
    banner.push_back(held_rate_line_);
  if (diag.form_latency_s.has_value()) {
    const auto latency = format_form_latency(*diag.form_latency_s);
    if (!latency.empty())
      banner.push_back(latency);
  }

  const std::array<int64_t, 2> label_shape{static_cast<int64_t>(kHudRuns),
                                           static_cast<int64_t>(kHudRowBytes)};
  auto records = labels.allocate_tensor(
      TensorLoanRequest{.shape = label_shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!records)
    return make_unexpected(std::move(records).error());
  auto labels_filled = records->write_host_and_commit(
      [&](const TensorOutputWriteGuard& guard) -> expected<void, Error> {
        auto* rows = guard.data_as<std::uint8_t>();
        if (rows == nullptr || guard.byte_size() < kHudRuns * kHudRowBytes)
          return make_unexpected(
              Error{ErrorCode::kFailure, "SarImageVizOp: text record loan is too small"});
        write_hud_rows(rows, banner, diag, overlay_.show_playback_lag, height);
        return {};
      });
  if (!labels_filled)
    return make_unexpected(std::move(labels_filled).error());

  std::vector<float> geom;
  fill_hud_geometry(&geom, diag, transpose_axes_);
  if (geom.empty())
    geom.assign(kHudLineFloats, 0.0F);
  const std::array<int64_t, 1> overlay_shape{static_cast<int64_t>(geom.size())};
  auto lines = overlay.allocate_tensor(
      TensorLoanRequest{.shape = overlay_shape, .dtype = DLDataType{kDLFloat, 32U, 1U}});
  if (!lines)
    return make_unexpected(std::move(lines).error());
  auto overlay_filled = lines->write_host_and_commit(
      [&](const TensorOutputWriteGuard& guard) -> expected<void, Error> {
        auto* out = guard.data_as<float>();
        if (out == nullptr || guard.byte_size() < geom.size() * sizeof(float))
          return make_unexpected(
              Error{ErrorCode::kFailure, "SarImageVizOp: overlay loan is too small"});
        std::memcpy(out, geom.data(), geom.size() * sizeof(float));
        return {};
      });
  if (!overlay_filled)
    return make_unexpected(std::move(overlay_filled).error());

  const EmitOptions options{.capture_time = context.activation_time()};
  HOLOSCAN_RETURN_IF_ERROR(labels.emit(std::move(*records), options));
  HOLOSCAN_RETURN_IF_ERROR(overlay.emit(std::move(*lines), options));
  schema::ImageT descriptor{};
  descriptor.height = static_cast<std::int32_t>(height);
  descriptor.width = static_cast<std::int32_t>(width);
  descriptor.encoding = schema::ImageEncoding_RGBA8;
  descriptor.color_space = schema::ColorSpace_UNSPECIFIED;
  HOLOSCAN_RETURN_IF_ERROR(frame.emit_tensor(std::move(*loan), descriptor, options));
  HOLOSCAN_LOG_INFO("SarImageVizOp: {}x{} -> {}x{} display", shape[0], shape[1], height, width);
  return {};
}
}  // namespace holoscan::radar::app::sar
