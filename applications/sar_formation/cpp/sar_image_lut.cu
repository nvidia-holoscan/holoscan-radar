// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "custom_ops.hpp"  // NOLINT(build/include_subdir)
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace {

constexpr int kThreads = 256;

__global__ void magnitude_c64_kernel(const float2* in, float* out,
                                     size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kThreads +
                   static_cast<size_t>(threadIdx.x);
  if (i >= count) {
    return;
  }
  const float2 z = in[i];
  out[i] = hypotf(z.x, z.y);
}

__global__ void magnitude_c128_kernel(const double2* in, float* out,
                                      size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * kThreads +
                   static_cast<size_t>(threadIdx.x);
  if (i >= count) {
    return;
  }
  const double2 z = in[i];
  out[i] = static_cast<float>(hypot(z.x, z.y));
}

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

std::vector<float> box_average(const std::vector<float>& in, int64_t in_v,
                               int64_t in_u, int64_t factor) {
  const int64_t out_v = in_v / factor;
  const int64_t out_u = in_u / factor;
  std::vector<float> out(static_cast<size_t>(out_v * out_u));
  const float n = static_cast<float>(factor * factor);
  for (int64_t y = 0; y < out_v; ++y) {
    for (int64_t x = 0; x < out_u; ++x) {
      float sum = 0.0F;
      for (int64_t dy = 0; dy < factor; ++dy) {
        for (int64_t dx = 0; dx < factor; ++dx) {
          const int64_t iy = y * factor + dy;
          const int64_t ix = x * factor + dx;
          sum += in[static_cast<size_t>(iy * in_u + ix)];
        }
      }
      out[static_cast<size_t>(y * out_u + x)] = sum / n;
    }
  }
  return out;
}

}  // namespace

void SarImageToLutOp::compute(
    holoscan::InputContext& op_input,
    holoscan::OutputContext& op_output,
    [[maybe_unused]] holoscan::ExecutionContext& context) {
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

  cudaStream_t stream = op_input.receive_cuda_stream("image");
  const size_t count =
      static_cast<size_t>(num_v) * static_cast<size_t>(num_u);
  auto mag = radar::make_device_tensor({num_v, num_u},
                                       radar::real_dtype<float>());
  const int blocks = static_cast<int>(
      (count + static_cast<size_t>(kThreads) - 1) /
      static_cast<size_t>(kThreads));
  if (is_c64) {
    magnitude_c64_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float2*>(image->data()),
        static_cast<float*>(mag->data()), count);
  } else {
    magnitude_c128_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const double2*>(image->data()),
        static_cast<float*>(mag->data()), count);
  }
  check_cuda(cudaGetLastError(), "magnitude kernel failed");

  std::vector<float> host(count);
  check_cuda(
      cudaMemcpyAsync(host.data(), mag->data(), count * sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
      "magnitude D2H failed");
  check_cuda(cudaStreamSynchronize(stream),
             "magnitude D2H sync failed");

  std::vector<float> valid;
  valid.reserve(host.size());
  for (float value : host) {
    if (std::isfinite(value) && value > 0.0F) {
      valid.push_back(value);
    }
  }
  if (valid.empty()) {
    throw std::runtime_error(
        std::string(who) + ": image is identically zero");
  }
  const double frac = high_percentile / 100.0;
  auto idx = static_cast<size_t>(
      std::floor(static_cast<double>(valid.size() - 1) * frac));
  if (idx >= valid.size()) {
    idx = valid.size() - 1;
  }
  std::nth_element(valid.begin(),
                   valid.begin() + static_cast<std::ptrdiff_t>(idx),
                   valid.end());
  const float high = valid[idx];
  if (!(high > 0.0F)) {
    throw std::runtime_error(
        std::string(who) + ": image is identically zero");
  }

  const int64_t factor =
      downsample_factor(num_v, num_u, max_side);
  const int64_t out_v = num_v / factor;
  const int64_t out_u = num_u / factor;
  if (out_v <= 0 || out_u <= 0) {
    throw std::invalid_argument(
        std::string(who) + ": max_side leaves an empty display grid");
  }
  const std::vector<float> displayed =
      factor > 1 ? box_average(host, num_v, num_u, factor) : host;
  const size_t out_count =
      static_cast<size_t>(out_v) * static_cast<size_t>(out_u);

  const float floor = static_cast<float>(floor_db);
  const float span = -floor;
  constexpr float kLutMax = 255.0F;
  std::vector<float> lut(out_count);
  for (size_t i = 0; i < out_count; ++i) {
    const float mag_v = displayed[i];
    const float db =
        20.0F * std::log10(mag_v / high + 1.0e-12F);
    const float clipped = std::min(0.0F, std::max(floor, db));
    lut[i] = (clipped - floor) / span * kLutMax;
  }

  auto tensor = radar::make_device_tensor(
      {out_v, out_u, 1}, radar::real_dtype<float>());
  check_cuda(cudaMemcpyAsync(tensor->data(), lut.data(),
                             out_count * sizeof(float),
                             cudaMemcpyHostToDevice, stream),
             "LUT H2D failed");
  check_cuda(cudaStreamSynchronize(stream), "LUT H2D sync failed");
  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "sar_image");
  }
  radar::emit_single(op_output, "sar_image", tensor);
}
