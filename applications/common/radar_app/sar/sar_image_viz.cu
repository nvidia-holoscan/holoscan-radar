// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar_image_viz.hpp"          // NOLINT(build/include_subdir)
#include "sar_image_lut_kernels.hpp"  // NOLINT(build/include_subdir)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>

#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar::app::sar {
namespace {
template <typename T>
__global__ void sample_magnitude(const T* input, int64_t source_v, int64_t source_u, float* output,
                                 int64_t height, int64_t width) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= height * width)
    return;
  const int64_t y = i / width, x = i % width;
  const int64_t y0 = y * source_v / height;
  const int64_t y1 = max(y0 + 1, (y + 1) * source_v / height);
  const int64_t x0 = x * source_u / width;
  const int64_t x1 = max(x0 + 1, (x + 1) * source_u / width);
  double sum = 0.0;
  int64_t count = 0;
  for (int64_t sy = y0; sy < y1; ++sy) {
    for (int64_t sx = x0; sx < x1; ++sx) {
      const T z = input[sy * source_u + sx];
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

SarImageVizOp::SarImageVizOp(int64_t max_v, int64_t max_u, int64_t display_side, double hold_s)
: max_v_(max_v), max_u_(max_u), display_side_(display_side), hold_s_(hold_s) {
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
  spec.input(image).queue_depth(4U).expects_tensor(tensor_input(device));
  spec.output(frame)
      .tensor_field(schema::ImageTensorFields::data)
      .produces_tensor(pooled_tensor_output(
          rgba, 3U,
          static_cast<uint64_t>(display_side_) * static_cast<uint64_t>(display_side_) * 4U));
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
  HOLOSCAN_TRY(auto input, receive_tensor(image, "SarImageVizOp"));
  const auto shape = input.shape();
  const auto dtype = input.dtype();
  if (shape.size() != 2U || shape[0] < 1 || shape[1] < 1 || shape[0] > max_v_ ||
      shape[1] > max_u_ || dtype.code != kDLComplex || (dtype.bits != 64U && dtype.bits != 128U))
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "SarImageVizOp: expected bounded complex image"});
  const int64_t longest = std::max(shape[0], shape[1]);
  const int64_t height = std::max<int64_t>(1, shape[0] * display_side_ / longest);
  const int64_t width = std::max<int64_t>(1, shape[1] * display_side_ / longest);
  const auto stream = context.cuda_stream().get();
  constexpr int kThreads = 256;
  const int64_t output_count = height * width;
  if (dtype.bits == 128U)
    sample_magnitude<<<(output_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const double2*>(input.data()), shape[0], shape[1], sampled_magnitude_, height,
        width);
  else
    sample_magnitude<<<(output_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const float2*>(input.data()), shape[0], shape[1], sampled_magnitude_, height,
        width);
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
  schema::ImageT descriptor{};
  descriptor.height = static_cast<std::int32_t>(height);
  descriptor.width = static_cast<std::int32_t>(width);
  descriptor.encoding = schema::ImageEncoding_RGBA8;
  descriptor.color_space = schema::ColorSpace_UNSPECIFIED;
  HOLOSCAN_RETURN_IF_ERROR(frame.emit_tensor(
      std::move(*loan), descriptor, EmitOptions{.capture_time = context.activation_time()}));
  if (hold_s_ > 0.0)
    std::this_thread::sleep_for(std::chrono::duration<double>(hold_s_));
  return {};
}
}  // namespace holoscan::radar::app::sar
