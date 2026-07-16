// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"

#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

size_t elem_bytes_for(DLDataType dtype) {
  return static_cast<size_t>(dtype.bits * dtype.lanes / 8);
}

}  // namespace

template <typename T>
MIMODeinterleaveOp<T>::~MIMODeinterleaveOp() {
  if (d_gather_map_ != nullptr) {
    cudaFree(d_gather_map_);
    d_gather_map_ = nullptr;
  }
}

template <typename T>
void MIMODeinterleaveOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void MIMODeinterleaveOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("adc_cube");
  spec.output<holoscan::TensorMap>("adc_cube");

  spec.param(tx_schedule_, "tx_schedule", "TDM TX firing schedule",
             "TXSchedule describing how the interleaved slow-time axis "
             "splits into (M_tx, L). Required; static for the instance.",
             TXSchedule{});
}

template <typename T>
void MIMODeinterleaveOp<T>::ensure_device_gather_map(int64_t slow_time,
                                                     const TXSchedule& schedule,
                                                     cudaStream_t stream) {
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
void MIMODeinterleaveOp<T>::compute(InputContext& op_input,
                                    OutputContext& op_output,
                                    ExecutionContext& context) {
  (void)context;
  constexpr const char* kOpName = "MIMODeinterleaveOp";
  std::shared_ptr<holoscan::Tensor> input =
      receive_single(op_input, "adc_cube", kOpName);

  classify_input_dtype<T>(*input, "adc_cube", /*allow_real=*/true);
  require_cuda_accessible(*input, "adc_cube", kOpName);
  require_contiguous(*input, "adc_cube", kOpName);

  const DLDataType dtype = input->dtype();
  const detail::AdcCubeLayout layout =
      detail::parse_adc_cube_layout(*input, "adc_cube");

  TXSchedule schedule = tx_schedule_.get();
  schedule.validate();
  const int64_t num_tx = schedule.num_tx;
  if (layout.slow_time % num_tx != 0) {
    throw std::invalid_argument(
        std::string(kOpName) +
        ": slow_time must be divisible by tx_schedule.num_tx");
  }
  const int64_t chirps_per_tx = layout.slow_time / num_tx;

  cudaStream_t stream = op_input.receive_cuda_stream("adc_cube");
  ensure_device_gather_map(layout.slow_time, schedule, stream);

  auto output = make_device_tensor(
      detail::deinterleaved_adc_shape(layout, num_tx, chirps_per_tx), dtype);

  detail::run_deinterleave_stage(
      input->data(), output->data(), layout, num_tx, chirps_per_tx,
      d_gather_map_, elem_bytes_for(dtype), stream);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "adc_cube");
  }
  holoscan::TensorMap out_map;
  out_map.insert({"adc_cube", output});
  op_output.emit(out_map, "adc_cube");

  retained_input_ = std::move(input);
}

template class MIMODeinterleaveOp<float>;
template class MIMODeinterleaveOp<double>;

}  // namespace holoscan::radar
