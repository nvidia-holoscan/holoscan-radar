// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "viz/simo_viz_ops.hpp"  // NOLINT(build/include_subdir)

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app::simo::viz {
namespace {

constexpr DLDataType kFloat32{kDLFloat, 32U, 1U};
constexpr DLDataType kInt64{kDLInt, 64U, 1U};
constexpr DLDataType kRgba8{kDLUInt, 8U, 1U};

__device__ unsigned int positive_float_bits(float value) {
  return __float_as_uint(fmaxf(value, 0.0F));
}

__global__ void reduce_positive_peak(const float* values, std::int64_t count,
                                     unsigned int* peak_bits) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count && isfinite(values[i])) {
    atomicMax(peak_bits, positive_float_bits(values[i]));
  }
}

__device__ unsigned char channel(float value) {
  return static_cast<unsigned char>(
      fminf(fmaxf(value * 255.0F, 0.0F), 255.0F));
}

__global__ void power_to_rgba(const float* power, std::int64_t num_planes,
                              std::int64_t pixels, const float* peak,
                              float dynamic_range_db, std::uint8_t* rgba) {
  const std::int64_t pixel =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixels) {
    return;
  }
  float value = 0.0F;
  for (std::int64_t plane = 0; plane < num_planes; ++plane) {
    value = fmaxf(value, power[plane * pixels + pixel]);
  }
  const float reference = fmaxf(*peak, FLT_MIN);
  const float db = 10.0F * log10f(fmaxf(value, FLT_MIN) / reference);
  const float t = fminf(fmaxf(1.0F + db / dynamic_range_db, 0.0F), 1.0F);

  // Compact perceptual-style blue -> cyan -> yellow -> red ramp.
  const float red = fminf(fmaxf(3.0F * t - 1.0F, 0.0F), 1.0F);
  const float green = fminf(fmaxf(1.5F - fabsf(3.0F * t - 1.5F), 0.0F), 1.0F);
  const float blue = fminf(fmaxf(2.0F - 3.0F * t, 0.0F), 1.0F);
  rgba[4 * pixel] = channel(red);
  rgba[4 * pixel + 1] = channel(green);
  rgba[4 * pixel + 2] = channel(blue);
  rgba[4 * pixel + 3] = 255U;
}

expected<void, Error> cuda_failure(const char* who, const char* step) {
  return make_unexpected(
      Error{ErrorCode::kFailure, std::string(who) + ": " + step + " failed"});
}

expected<std::vector<std::int64_t>, Error> copy_int64_to_host(
    const Tensor& tensor, cudaStream_t stream, const char* name) {
  if (!dldatatype_equals(tensor.dtype(), kInt64) || !tensor.is_contiguous()) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string("SimoClusterGeometryOp.") + name +
            ": expected a compact int64 Tensor"});
  }
  const std::size_t count =
      static_cast<std::size_t>(tensor.nbytes()) / sizeof(std::int64_t);
  std::vector<std::int64_t> result(count);
  if (!result.empty() &&
      cudaMemcpyAsync(result.data(), tensor.data(), tensor.nbytes(),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string("SimoClusterGeometryOp.") + name +
            ": device-to-host copy failed"});
  }
  return result;
}

expected<TensorOutputLoan, Error> prepare_host_floats(
    Output<Tensor>& output, const std::vector<float>& values) {
  const std::vector<std::int64_t> shape{
      static_cast<std::int64_t>(values.size())};
  auto loan =
      output.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = kFloat32});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto written = loan->write_host_and_commit(
      [&values](const TensorOutputWriteGuard& writer) {
        std::copy(values.begin(), values.end(), writer.data_as<float>());
      });
  if (!written) {
    return make_unexpected(std::move(written).error());
  }
  return std::move(*loan);
}

void validate_geometry_inputs(const Tensor& detection_idx, const Tensor& labels,
                              const Tensor& rep_idx, const Tensor& range_bins,
                              const Tensor& velocity_bins) {
  if (detection_idx.shape().size() != 2U ||
      detection_idx.shape()[1] < 2 ||
      labels.shape().size() != 1U ||
      labels.shape()[0] != detection_idx.shape()[0] ||
      rep_idx.shape().size() != 2U || rep_idx.shape()[1] < 2 ||
      range_bins.shape().size() != 1U || range_bins.shape()[0] <= 0 ||
      velocity_bins.shape().size() != 1U || velocity_bins.shape()[0] <= 0) {
    throw std::invalid_argument(
        "SimoClusterGeometryOp: incompatible index, label, or axis shapes");
  }
}

}  // namespace

SimoRdHeatmapOp::SimoRdHeatmapOp(const MaxTensorExtents& extents,
                                 RdHeatmapConfig config)
    : config_(config) {
  if (!std::isfinite(config_.dynamic_range_db) ||
      config_.dynamic_range_db <= 0.0F) {
    throw std::invalid_argument(
        "SimoRdHeatmapOp: dynamic_range_db must be finite and > 0");
  }
  const std::int64_t max_d = require_extent(
      extents.max_doppler_bins, "max_doppler_bins", "SimoRdHeatmapOp");
  const std::int64_t max_r = require_extent(
      extents.max_range_bins, "max_range_bins", "SimoRdHeatmapOp");
  max_byte_span_ =
      tensor_byte_span(std::vector<std::int64_t>{max_d, max_r, 4}, kRgba8);
  peak_power_ = make_device_tensor({1}, kFloat32);
}

void SimoRdHeatmapOp::setup(holoscan::OperatorSpec& spec) {
  constexpr TensorRepresentation kPower{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = kFloat32,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kImage{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = kRgba8,
      .rank = 3U,
  };
  spec.input(cfar_power).queue_depth(2U).expects_tensor(tensor_input(kPower));
  spec.output(rgba).produces_tensor(
      pooled_tensor_output(kImage, 3U, max_byte_span_));
}

holoscan::Contract SimoRdHeatmapOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{cfar_power});
  return result;
}

expected<void, Error> SimoRdHeatmapOp::compute(
    holoscan::ExecutionContext& context) {
  constexpr const char* kWho = "SimoRdHeatmapOp";
  HOLOSCAN_TRY(auto power, receive_tensor(cfar_power, kWho));
  BatchedShape layout;
  try {
    layout = collapse_batch(power, "SimoRdHeatmapOp.cfar_power");
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }
  if (!dldatatype_equals(power.dtype(), kFloat32)) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kWho) + ": cfar_power must be float32"});
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, std::string(kWho) + ": no CUDA stream"});
  }

  const std::int64_t pixels = layout.rows * layout.cols;
  const std::int64_t count = layout.batch * pixels;
  if (cudaMemsetAsync(peak_power_->mutable_data().value(), 0, sizeof(float),
                      stream) != cudaSuccess) {
    return cuda_failure(kWho, "peak reset");
  }
  constexpr int kBlock = 256;
  const int reduce_grid =
      static_cast<int>((count + kBlock - 1) / kBlock);
  reduce_positive_peak<<<reduce_grid, kBlock, 0, stream>>>(
      static_cast<const float*>(power.data()), count,
      static_cast<unsigned int*>(peak_power_->mutable_data().value()));
  if (cudaGetLastError() != cudaSuccess) {
    return cuda_failure(kWho, "peak reduction launch");
  }

  const std::vector<std::int64_t> shape{layout.rows, layout.cols, 4};
  HOLOSCAN_TRY(auto loan,
               prepare_device_tensor(
                   rgba, context, shape, kRgba8,
                   [&](void* dst, std::size_t) -> expected<void, Error> {
                     const int grid =
                         static_cast<int>((pixels + kBlock - 1) / kBlock);
                     power_to_rgba<<<grid, kBlock, 0, stream>>>(
                         static_cast<const float*>(power.data()), layout.batch,
                         pixels, static_cast<const float*>(peak_power_->data()),
                         config_.dynamic_range_db,
                         static_cast<std::uint8_t*>(dst));
                     if (cudaGetLastError() != cudaSuccess) {
                       return cuda_failure(kWho, "color mapping launch");
                     }
                     return {};
                   }));
  retained_input_ = std::move(power);
  return emit_prepared_tensor(rgba, context, std::move(loan));
}

SimoClusterGeometryOp::SimoClusterGeometryOp(
    const MaxTensorExtents& extents, ClusterGeometryConfig config)
    : config_(config) {
  if (!std::isfinite(config_.detection_radius) ||
      config_.detection_radius <= 0.0F ||
      !std::isfinite(config_.representative_size) ||
      config_.representative_size <= 0.0F) {
    throw std::invalid_argument(
        "SimoClusterGeometryOp: marker dimensions must be finite and > 0");
  }
  const std::int64_t max_detections = require_extent(
      extents.max_detections, "max_detections", "SimoClusterGeometryOp");
  const std::int64_t max_clusters = require_extent(
      extents.max_clusters, "max_clusters", "SimoClusterGeometryOp");
  max_ovals_byte_span_ = tensor_byte_span(
      std::vector<std::int64_t>{max_detections * 4}, kFloat32);
  max_crosses_byte_span_ = tensor_byte_span(
      std::vector<std::int64_t>{max_clusters * 3}, kFloat32);
}

void SimoClusterGeometryOp::setup(holoscan::OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorRepresentation kHostFloats{
      .memory_kind = MemoryKind::kHost,
      .dtype = kFloat32,
      .rank = 1U,
  };
  spec.input(detection_idx).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(labels).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(rep_idx).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(range_bins).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(velocity_bins).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(detection_ovals)
      .produces_tensor(
          pooled_tensor_output(kHostFloats, 1U, max_ovals_byte_span_));
  spec.output(representative_crosses)
      .produces_tensor(
          pooled_tensor_output(kHostFloats, 1U, max_crosses_byte_span_));
}

holoscan::Contract SimoClusterGeometryOp::contract() const {
  holoscan::Contract result;
  result
      .trigger(holoscan::OnAligned{
          .ports = holoscan::ports(detection_idx, labels, rep_idx, range_bins,
                                   velocity_bins),
          .window = std::chrono::nanoseconds{0}})
      .input(detection_idx, holoscan::Exact{})
      .input(labels, holoscan::Exact{})
      .input(rep_idx, holoscan::Exact{})
      .input(range_bins, holoscan::Exact{})
      .input(velocity_bins, holoscan::Exact{});
  return result;
}

expected<void, Error> SimoClusterGeometryOp::compute(
    holoscan::ExecutionContext& context) {
  constexpr const char* kWho = "SimoClusterGeometryOp";
  HOLOSCAN_TRY(auto idx, receive_tensor(detection_idx, kWho));
  HOLOSCAN_TRY(auto cluster_labels, receive_tensor(labels, kWho));
  HOLOSCAN_TRY(auto reps, receive_tensor(rep_idx, kWho));
  HOLOSCAN_TRY(auto ranges, receive_tensor(range_bins, kWho));
  HOLOSCAN_TRY(auto velocities, receive_tensor(velocity_bins, kWho));
  try {
    validate_geometry_inputs(idx, cluster_labels, reps, ranges, velocities);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, std::string(kWho) + ": no CUDA stream"});
  }
  HOLOSCAN_TRY(auto host_idx, copy_int64_to_host(idx, stream, "detection_idx"));
  HOLOSCAN_TRY(
      auto host_labels, copy_int64_to_host(cluster_labels, stream, "labels"));
  HOLOSCAN_TRY(auto host_reps, copy_int64_to_host(reps, stream, "rep_idx"));
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    return cuda_failure(kWho, "copy synchronization");
  }

  std::vector<float> ovals;
  std::vector<float> crosses;
  try {
    ovals = detail::make_cluster_ovals(
        host_idx, host_labels, idx.shape()[1], velocities.shape()[0],
        ranges.shape()[0], config_.detection_radius);
    crosses = detail::make_representative_crosses(
        host_reps, reps.shape()[1], velocities.shape()[0], ranges.shape()[0],
        config_.representative_size);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }

  HOLOSCAN_TRY(auto oval_loan,
               prepare_host_floats(detection_ovals, ovals));
  HOLOSCAN_TRY(auto cross_loan,
               prepare_host_floats(representative_crosses, crosses));
  HOLOSCAN_TRY(emit_prepared_tensor(detection_ovals, context,
                                    std::move(oval_loan)));
  return emit_prepared_tensor(representative_crosses, context,
                              std::move(cross_loan));
}

namespace detail {

std::vector<std::uint8_t> make_heatmap_rgba_reference(
    const std::vector<float>& power, const std::vector<std::int64_t>& shape,
    float dynamic_range_db) {
  if (shape.size() < 2U || !std::isfinite(dynamic_range_db) ||
      dynamic_range_db <= 0.0F) {
    throw std::invalid_argument(
        "make_heatmap_rgba_reference: invalid shape or dynamic range");
  }
  std::int64_t planes = 1;
  for (std::size_t i = 0; i + 2 < shape.size(); ++i) {
    if (shape[i] <= 0) {
      throw std::invalid_argument(
          "make_heatmap_rgba_reference: shape must be positive");
    }
    planes *= shape[i];
  }
  const std::int64_t pixels = shape[shape.size() - 2] * shape.back();
  if (pixels <= 0 ||
      power.size() != static_cast<std::size_t>(planes * pixels)) {
    throw std::invalid_argument(
        "make_heatmap_rgba_reference: power size does not match shape");
  }
  const float peak = *std::max_element(power.begin(), power.end());
  const float reference = std::max(peak, FLT_MIN);
  std::vector<std::uint8_t> result(
      static_cast<std::size_t>(pixels) * 4U);
  for (std::int64_t pixel = 0; pixel < pixels; ++pixel) {
    float value = 0.0F;
    for (std::int64_t plane = 0; plane < planes; ++plane) {
      value = std::max(
          value, power[static_cast<std::size_t>(plane * pixels + pixel)]);
    }
    const float db =
        10.0F * std::log10(std::max(value, FLT_MIN) / reference);
    const float t =
        std::clamp(1.0F + db / dynamic_range_db, 0.0F, 1.0F);
    const float red = std::clamp(3.0F * t - 1.0F, 0.0F, 1.0F);
    const float green =
        std::clamp(1.5F - std::abs(3.0F * t - 1.5F), 0.0F, 1.0F);
    const float blue = std::clamp(2.0F - 3.0F * t, 0.0F, 1.0F);
    const std::size_t out = static_cast<std::size_t>(pixel) * 4U;
    result[out] = static_cast<std::uint8_t>(red * 255.0F);
    result[out + 1] = static_cast<std::uint8_t>(green * 255.0F);
    result[out + 2] = static_cast<std::uint8_t>(blue * 255.0F);
    result[out + 3] = 255U;
  }
  return result;
}

std::vector<float> make_cluster_ovals(
    const std::vector<std::int64_t>& detection_idx,
    const std::vector<std::int64_t>& labels, std::int64_t idx_cols,
    std::int64_t num_doppler, std::int64_t num_range, float radius) {
  if (idx_cols < 2 || detection_idx.size() % idx_cols != 0U ||
      labels.size() != detection_idx.size() /
                           static_cast<std::size_t>(idx_cols) ||
      num_doppler <= 0 || num_range <= 0) {
    throw std::invalid_argument("make_cluster_ovals: invalid input dimensions");
  }
  std::vector<float> result;
  result.reserve(labels.size() * 4U);
  for (std::size_t row = 0; row < labels.size(); ++row) {
    if (labels[row] < 0) {
      continue;
    }
    const std::size_t offset =
        row * static_cast<std::size_t>(idx_cols);
    const std::int64_t doppler =
        detection_idx[offset + static_cast<std::size_t>(idx_cols - 2)];
    const std::int64_t range =
        detection_idx[offset + static_cast<std::size_t>(idx_cols - 1)];
    if (doppler < 0 || doppler >= num_doppler || range < 0 ||
        range >= num_range) {
      throw std::invalid_argument(
          "make_cluster_ovals: detection index is out of bounds");
    }
    result.insert(result.end(),
                  {(static_cast<float>(range) + 0.5F) /
                       static_cast<float>(num_range),
                   (static_cast<float>(doppler) + 0.5F) /
                       static_cast<float>(num_doppler),
                   radius, radius});
  }
  return result;
}

std::vector<float> make_representative_crosses(
    const std::vector<std::int64_t>& rep_idx, std::int64_t idx_cols,
    std::int64_t num_doppler, std::int64_t num_range, float size) {
  if (idx_cols < 2 || rep_idx.size() % idx_cols != 0U || num_doppler <= 0 ||
      num_range <= 0) {
    throw std::invalid_argument(
        "make_representative_crosses: invalid input dimensions");
  }
  const std::size_t rows =
      rep_idx.size() / static_cast<std::size_t>(idx_cols);
  std::vector<float> result;
  result.reserve(rows * 3U);
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t offset =
        row * static_cast<std::size_t>(idx_cols);
    const std::int64_t doppler =
        rep_idx[offset + static_cast<std::size_t>(idx_cols - 2)];
    const std::int64_t range =
        rep_idx[offset + static_cast<std::size_t>(idx_cols - 1)];
    if (doppler < 0 || doppler >= num_doppler || range < 0 ||
        range >= num_range) {
      throw std::invalid_argument(
          "make_representative_crosses: representative index is out of bounds");
    }
    result.insert(result.end(),
                  {(static_cast<float>(range) + 0.5F) /
                       static_cast<float>(num_range),
                   (static_cast<float>(doppler) + 0.5F) /
                       static_cast<float>(num_doppler),
                   size});
  }
  return result;
}

}  // namespace detail
}  // namespace holoscan::radar::app::simo::viz
