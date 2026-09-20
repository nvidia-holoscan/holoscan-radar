// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_stage.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

constexpr const char* kOpName = "MIMODeinterleaveOp";

size_t elem_bytes_for(DLDataType dtype) {
  return static_cast<size_t>(dtype.bits * dtype.lanes / 8);
}

// Stage knobs (the static schedule) plus this tick's parsed layout and
// emit geometry. There is no resolve_params: no [live] ports and the
// empty-schedule sentinel is rejected here, not mapped to an optional.
struct DeinterleaveStagePlan {
  TXSchedule schedule;
  detail::AdcCubeLayout layout;
  int64_t num_tx{0};
  int64_t chirps_per_tx{0};
  std::vector<int64_t> out_shape;
};

expected<DeinterleaveStagePlan, Error> build_deinterleave_stage_plan(
    TXSchedule schedule, const Tensor& input,
    const MaxTensorExtents& extents) {
  DeinterleaveStagePlan plan;
  try {
    plan.layout = detail::parse_adc_cube_layout(input, "adc_cube");
    schedule.validate();
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }

  if (plan.layout.slow_time % schedule.num_tx != 0) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": slow_time must be divisible by tx_schedule.num_tx"});
  }

  plan.schedule = std::move(schedule);
  plan.num_tx = plan.schedule.num_tx;
  plan.chirps_per_tx = plan.layout.slow_time / plan.num_tx;
  if (plan.num_tx > *extents.max_tx ||
      plan.layout.num_rx > *extents.max_rx ||
      plan.chirps_per_tx > *extents.max_chirps ||
      plan.layout.samples > *extents.max_samples_per_chirp) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": output (M_tx, N_rx, L, samples) exceeds MaxTensorExtents"});
  }
  plan.out_shape = detail::deinterleaved_adc_shape(
      plan.layout, plan.num_tx, plan.chirps_per_tx);
  return plan;
}

}  // namespace

template <typename T>
MIMODeinterleaveOp<T>::MIMODeinterleaveOp(
    const MaxTensorExtents& extents) {
  const char* who = kOpName;
  const std::int64_t max_tx =
      require_extent(extents.max_tx, "max_tx", who);
  const std::int64_t max_rx =
      require_extent(extents.max_rx, "max_rx", who);
  const std::int64_t max_chirps =
      require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_samples =
      require_extent(extents.max_samples_per_chirp,
                     "max_samples_per_chirp", who);
  extents_ = extents;
  max_rank_ = 4U;
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_tx, max_rx, max_chirps, max_samples}, who);

  gather_map_capacity_ =
      static_cast<std::size_t>(max_tx) * static_cast<std::size_t>(max_chirps);
  gather_map_ = make_device_tensor(
      {max_tx * max_chirps}, DLDataType{kDLInt, 64, 1});
}

template <typename T>
void MIMODeinterleaveOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds fat_bounds{.max_rank = max_rank_,
                                .max_byte_span = max_byte_span_};

  spec.input(adc_cube).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(adc_cube_out)
      .produces_tensor(pooled_tensor_output(kDevice, fat_bounds));
}

template <typename T>
Contract MIMODeinterleaveOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{adc_cube});
  return result;
}

template <typename T>
expected<void, Error> MIMODeinterleaveOp<T>::ensure_device_gather_map(
    int64_t slow_time, const TXSchedule& schedule, cudaStream_t stream) {
  if (slow_time == cached_slow_time_ && schedule == cached_schedule_) {
    return {};
  }

  std::vector<int64_t> host;
  try {
    host = schedule.build_gather_map(slow_time);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }
  if (host.size() > gather_map_capacity_) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": gather map length exceeds max_tx * max_chirps"});
  }

  auto mut = gather_map_->mutable_data();
  if (!mut) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": gather-map not writable"});
  }
  if (cudaMemcpyAsync(*mut, host.data(), host.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": gather-map H2D failed"});
  }

  cached_slow_time_ = slow_time;
  cached_schedule_ = schedule;
  return {};
}

template <typename T>
expected<void, Error> MIMODeinterleaveOp<T>::compute(
    ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  // Receive adc_cube; require a device-contiguous cube (complex or real IF).
  HOLOSCAN_TRY(auto input, receive_tensor(adc_cube, kOpName));
  try {
    require_cuda_accessible(input, "adc_cube", kOpName);
    require_contiguous(input, "adc_cube", kOpName);
    classify_input_dtype<T>(input, "adc_cube", /*allow_real=*/true);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Trailing rank is (N_rx, M·L, samples), not an FMCW (chirps, bins)
  // cube — layout parse lives in the plan builder. No [live] merge.
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const MIMODeinterleaveParams& pinned =
      context.params<MIMODeinterleaveParams>();
  // Build and validate the bounded execution plan for this input shape.
  HOLOSCAN_TRY(auto plan,
               build_deinterleave_stage_plan(pinned.tx_schedule, input,
                                             extents_));

  // The row-copy kernel runs on the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              std::string(kOpName) + ": no CUDA compute stream"});
  }

  HOLOSCAN_TRY(ensure_device_gather_map(
      plan.layout.slow_time, plan.schedule, stream));

  // Required adc_cube_out: strided gather, then publish.
  const void* input_ptr = input.data();
  const int64_t* d_map =
      static_cast<const int64_t*>(gather_map_->data());
  // Publish coherent outputs in dependency order and retain borrowed inputs.
  HOLOSCAN_TRY(emit_device_tensor(
      adc_cube_out, context, plan.out_shape, input.dtype(),
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          // Execute the device stage on the input runtime stream.
          detail::run_deinterleave_stage(
              input_ptr, out_ptr, plan.layout, plan.num_tx,
              plan.chirps_per_tx, d_map, elem_bytes_for(input.dtype()),
              stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      }));

  // Keep adc_cube alive until stream-ordered reads complete.
  retained_input_ = std::move(input);
  return {};
}

template class MIMODeinterleaveOp<float>;
template class MIMODeinterleaveOp<double>;

}  // namespace holoscan::radar
