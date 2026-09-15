// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "custom_ops.hpp"  // NOLINT(build/include_subdir)

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app::simo {

namespace {

constexpr const char* kOpName = "PeakIdentityOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

__global__ void identity_labels_kernel(int64_t* labels, int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    labels[index] = index;
  }
}

__global__ void identity_sizes_kernel(int64_t* sizes, int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    sizes[index] = 1;
  }
}

int grid_for(int64_t count) {
  constexpr int kBlock = 256;
  return static_cast<int>((count + kBlock - 1) / kBlock);
}

expected<void, Error> fail_cuda(const char* operation) {
  return make_unexpected(Error{
      ErrorCode::kFailure,
      std::string(kOpName) + ": " + operation + " failed"});
}

}  // namespace

namespace detail {

void launch_peak_identity_labels(int64_t* labels, int64_t count,
                                 cudaStream_t stream) {
  if (count > 0) {
    identity_labels_kernel<<<grid_for(count), 256, 0, stream>>>(labels, count);
  }
}

void launch_peak_identity_sizes(int64_t* sizes, int64_t count,
                                cudaStream_t stream) {
  if (count > 0) {
    identity_sizes_kernel<<<grid_for(count), 256, 0, stream>>>(sizes, count);
  }
}

}  // namespace detail

PeakIdentityOp::PeakIdentityOp(const MaxTensorExtents& extents) {
  const int64_t max_detections =
      require_extent(extents.max_detections, "max_detections", kOpName);
  vector_max_byte_span_ =
      tensor_byte_span(std::vector<int64_t>{max_detections}, kInt64Dtype);
  index_max_byte_span_ =
      tensor_byte_span(
          std::vector<int64_t>{max_detections, 4}, kInt64Dtype);
}

void PeakIdentityOp::setup(holoscan::OperatorSpec& spec) {
  constexpr holoscan::TensorRepresentation kDevice{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  spec.input(detection_idx)
      .queue_depth(2U)
      .expects_tensor(holoscan::radar::tensor_input(kDevice));
  spec.output(labels)
      .produces_tensor(holoscan::radar::pooled_tensor_output(
          kDevice, 2U, vector_max_byte_span_));
  spec.output(sizes)
      .produces_tensor(holoscan::radar::pooled_tensor_output(
          kDevice, 2U, vector_max_byte_span_));
  spec.output(rep_idx)
      .produces_tensor(holoscan::radar::pooled_tensor_output(
          kDevice, 2U, index_max_byte_span_));
}

holoscan::Contract PeakIdentityOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{detection_idx});
  return result;
}

expected<void, Error> PeakIdentityOp::compute(
    holoscan::ExecutionContext& context) {
  HOLOSCAN_TRY(auto indices, receive_tensor(detection_idx, kOpName));
  try {
    require_cuda_accessible(indices, "detection_idx", kOpName);
    require_contiguous(indices, "detection_idx", kOpName);
    require_dtype(indices, kInt64Dtype, "detection_idx", kOpName);
  } catch (const std::invalid_argument& error) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, error.what()});
  }
  const auto& shape = indices.shape();
  if (shape.size() != 2U || shape[1] != 4) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "PeakIdentityOp: detection_idx must have shape (num_peaks, 4)"});
  }

  const int64_t count = shape[0];
  cudaStream_t stream = context.cuda_stream().get();
  HOLOSCAN_TRY(
      auto labels_loan,
      prepare_device_tensor(
          labels, context, std::vector<int64_t>{count}, kInt64Dtype,
          [count, stream](void* output, std::size_t) -> expected<void, Error> {
            detail::launch_peak_identity_labels(
                static_cast<int64_t*>(output), count, stream);
            if (cudaPeekAtLastError() != cudaSuccess) {
              return fail_cuda("labels kernel launch");
            }
            return {};
          }));
  HOLOSCAN_TRY(
      auto sizes_loan,
      prepare_device_tensor(
          sizes, context, std::vector<int64_t>{count}, kInt64Dtype,
          [count, stream](void* output, std::size_t) -> expected<void, Error> {
            detail::launch_peak_identity_sizes(
                static_cast<int64_t*>(output), count, stream);
            if (cudaPeekAtLastError() != cudaSuccess) {
              return fail_cuda("sizes kernel launch");
            }
            return {};
          }));
  HOLOSCAN_TRY(
      auto rep_idx_loan,
      prepare_device_tensor(
          rep_idx, context, std::vector<int64_t>{count, 4}, kInt64Dtype,
          [&indices, stream](void* output,
                             std::size_t bytes) -> expected<void, Error> {
            if (bytes > 0 &&
                cudaMemcpyAsync(output, indices.data(), bytes,
                                cudaMemcpyDeviceToDevice,
                                stream) != cudaSuccess) {
              return fail_cuda("detection_idx copy");
            }
            return {};
          }));

  HOLOSCAN_RETURN_IF_ERROR(
      emit_prepared_tensor(labels, context, std::move(labels_loan)));
  HOLOSCAN_RETURN_IF_ERROR(
      emit_prepared_tensor(sizes, context, std::move(sizes_loan)));
  return emit_prepared_tensor(rep_idx, context, std::move(rep_idx_loan));
}

}  // namespace holoscan::radar::app::simo
