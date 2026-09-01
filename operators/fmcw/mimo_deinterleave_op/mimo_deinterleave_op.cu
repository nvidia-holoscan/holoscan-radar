// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_stage.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/extents_layout.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

size_t elem_bytes_for(DLDataType dtype) {
  return static_cast<size_t>(dtype.bits * dtype.lanes / 8);
}

}  // namespace

template <typename T>
MIMODeinterleaveOp<T>::MIMODeinterleaveOp(
    const MaxTensorExtents& extents) {
  const char* who = "MIMODeinterleaveOp";
  const std::int64_t max_tx =
      require_extent(extents.max_tx, "max_tx", who);
  const std::int64_t max_rx =
      require_extent(extents.max_rx, "max_rx", who);
  const std::int64_t max_chirps =
      require_extent(
          extents.max_chirps, "max_chirps", who);
  const std::int64_t max_samples =
      require_extent(extents.max_samples_per_chirp,
                     "max_samples_per_chirp", who);
  extents_ = extents;
  max_rank_ = 4U;
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_tx, max_rx, max_chirps, max_samples}, who);
}

template <typename T>
MIMODeinterleaveOp<T>::~MIMODeinterleaveOp() {
  if (d_gather_map_ != nullptr) {
    cudaFree(d_gather_map_);
    d_gather_map_ = nullptr;
  }
}

template <typename T>
void MIMODeinterleaveOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  spec.input(adc_cube).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(adc_cube_out)
      .produces_tensor(pooled_tensor_output(kDevice, max_rank_, max_byte_span_));
}

template <typename T>
Contract MIMODeinterleaveOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{adc_cube});
  return result;
}

template <typename T>
void MIMODeinterleaveOp<T>::ensure_device_gather_map(
    int64_t slow_time, const TXSchedule& schedule, cudaStream_t stream) {
  if (slow_time == cached_slow_time_ && schedule == cached_schedule_ &&
      d_gather_map_ != nullptr) {
    return;
  }
  cached_slow_time_ = slow_time;
  cached_schedule_ = schedule;
  cached_gather_host_ = schedule.build_gather_map(slow_time);
  const size_t count = cached_gather_host_.size();
  if (count > d_gather_capacity_) {
    if (d_gather_map_ != nullptr) {
      cudaFree(d_gather_map_);
      d_gather_map_ = nullptr;
    }
    if (cudaMalloc(&d_gather_map_, count * sizeof(int64_t)) != cudaSuccess) {
      throw std::runtime_error(
          "MIMODeinterleaveOp: gather-map device allocation failed");
    }
    d_gather_capacity_ = count;
  }
  if (cudaMemcpyAsync(d_gather_map_, cached_gather_host_.data(),
                      count * sizeof(int64_t), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("MIMODeinterleaveOp: gather-map H2D failed");
  }
}

template <typename T>
expected<void, Error> MIMODeinterleaveOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto input,
               receive_tensor(adc_cube, "MIMODeinterleaveOp"));

  try {
    classify_input_dtype<T>(input, "adc_cube", /*allow_real=*/true);
    require_cuda_accessible(input, "adc_cube", "MIMODeinterleaveOp");
    require_contiguous(input, "adc_cube", "MIMODeinterleaveOp");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const DLDataType dtype = input.dtype();
  detail::AdcCubeLayout layout;
  try {
    layout = detail::parse_adc_cube_layout(input, "adc_cube");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const MIMODeinterleaveParams& pinned =
      context.params<MIMODeinterleaveParams>();
  TXSchedule schedule = pinned.tx_schedule;
  try {
    schedule.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const int64_t num_tx = schedule.num_tx;
  if (layout.slow_time % num_tx != 0) {
    return fail_invalid(
        "MIMODeinterleaveOp: slow_time must be divisible by "
        "tx_schedule.num_tx");
  }
  const int64_t chirps_per_tx = layout.slow_time / num_tx;

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "MIMODeinterleaveOp: no CUDA compute stream"});
  }

  try {
    ensure_device_gather_map(layout.slow_time, schedule, stream);
  } catch (const std::runtime_error& e) {
    return make_unexpected(Error{ErrorCode::kFailure, e.what()});
  }

  const std::vector<int64_t> out_shape =
      detail::deinterleaved_adc_shape(layout, num_tx, chirps_per_tx);

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(emit_device_tensor(
      adc_cube_out, context, out_shape, dtype,
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          detail::run_deinterleave_stage(
              input_ptr, out_ptr, layout, num_tx, chirps_per_tx,
              d_gather_map_, elem_bytes_for(dtype), stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      }));

  retained_input_ = std::move(input);
  return {};
}

template class MIMODeinterleaveOp<float>;
template class MIMODeinterleaveOp<double>;

}  // namespace holoscan::radar
