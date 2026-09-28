// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar_image_viz.hpp"  // NOLINT(build/include_subdir)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
__global__ void find_peak(const T* input, int64_t count, std::uint32_t* bits) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const double re = static_cast<double>(input[i].x);
  const double im = static_cast<double>(input[i].y);
  const float magnitude = static_cast<float>(sqrt(re * re + im * im));
  if (isfinite(magnitude))
    atomicMax(bits, __float_as_uint(magnitude));
}
template <typename T>
__global__ void render_gray(const T* input, int64_t source_v, int64_t source_u,
                            std::uint8_t* output, int64_t height, int64_t width,
                            const std::uint32_t* max_bits) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= height * width)
    return;
  const int64_t y = i / width, x = i % width;
  const int64_t sy = min(source_v - 1, (y * source_v) / height);
  const int64_t sx = min(source_u - 1, (x * source_u) / width);
  const T sample = input[sy * source_u + sx];
  const float magnitude = hypotf(static_cast<float>(sample.x), static_cast<float>(sample.y));
  const float peak = max(__uint_as_float(*max_bits), 1.0e-20f);
  const float db = 20.0f * log10f(max(magnitude, 1.0e-20f) / peak);
  const auto gray =
      static_cast<std::uint8_t>(255.0f * fminf(1.0f, fmaxf(0.0f, (db + 30.0f) / 30.0f)));
  output[4 * i] = gray;
  output[4 * i + 1] = gray;
  output[4 * i + 2] = gray;
  output[4 * i + 3] = 255U;
}
}  // namespace

SarImageVizOp::SarImageVizOp(int64_t max_v, int64_t max_u, int64_t display_side)
: max_v_(max_v), max_u_(max_u), display_side_(display_side) {
  if (max_v_ < 1 || max_u_ < 1 || display_side_ < 1)
    throw std::invalid_argument("SarImageVizOp: invalid image extents");
}
SarImageVizOp::~SarImageVizOp() {
  if (max_bits_ != nullptr)
    (void)cudaFree(max_bits_);
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
  if (max_bits_ == nullptr && cudaMalloc(&max_bits_, sizeof(std::uint32_t)) != cudaSuccess)
    throw std::runtime_error("SarImageVizOp: peak allocation failed");
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
  if (cudaMemsetAsync(max_bits_, 0, sizeof(std::uint32_t), stream) != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "SarImageVizOp: peak clear failed"});
  constexpr int kThreads = 256;
  const int64_t source_count = shape[0] * shape[1];
  const int64_t output_count = height * width;
  if (dtype.bits == 128U)
    find_peak<<<(source_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const double2*>(input.data()), source_count, max_bits_);
  else
    find_peak<<<(source_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const float2*>(input.data()), source_count, max_bits_);
  auto loan = frame.allocate_tensor(TensorLoanRequest{
      .shape = std::vector<int64_t>{height, width, 4}, .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!loan)
    return make_unexpected(std::move(loan).error());
  auto writer = loan->write(context.cuda_stream());
  if (!writer)
    return make_unexpected(std::move(writer).error());
  if (dtype.bits == 128U)
    render_gray<<<(output_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const double2*>(input.data()), shape[0], shape[1],
        static_cast<std::uint8_t*>(writer->data()), height, width, max_bits_);
  else
    render_gray<<<(output_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const float2*>(input.data()), shape[0], shape[1],
        static_cast<std::uint8_t*>(writer->data()), height, width, max_bits_);
  if (cudaPeekAtLastError() != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, "SarImageVizOp: render kernel failed"});
  HOLOSCAN_RETURN_IF_ERROR(std::move(*writer).commit());
  schema::ImageT descriptor{};
  descriptor.height = static_cast<std::int32_t>(height);
  descriptor.width = static_cast<std::int32_t>(width);
  descriptor.encoding = schema::ImageEncoding_RGBA8;
  descriptor.color_space = schema::ColorSpace_UNSPECIFIED;
  return frame.emit_tensor(std::move(*loan), descriptor,
                           EmitOptions{.capture_time = context.activation_time()});
}
}  // namespace holoscan::radar::app::sar
