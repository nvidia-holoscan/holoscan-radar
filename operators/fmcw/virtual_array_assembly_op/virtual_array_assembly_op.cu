// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

bool plan_key_changed(const std::vector<double>& tx,
                      const std::vector<double>& rx, OverlapPolicy policy,
                      int64_t num_tx, int64_t num_rx,
                      const std::vector<double>& cached_tx,
                      const std::vector<double>& cached_rx,
                      OverlapPolicy cached_policy, int64_t cached_num_tx,
                      int64_t cached_num_rx) {
  return tx != cached_tx || rx != cached_rx || policy != cached_policy ||
         num_tx != cached_num_tx || num_rx != cached_num_rx;
}

}  // namespace

template <typename T>
VirtualArrayAssemblyOp<T>::~VirtualArrayAssemblyOp() {
  if (d_src_channel_ != nullptr) {
    cudaFree(d_src_channel_);
    d_src_channel_ = nullptr;
  }
  if (d_member_offsets_ != nullptr) {
    cudaFree(d_member_offsets_);
    d_member_offsets_ = nullptr;
  }
  if (d_member_indices_ != nullptr) {
    cudaFree(d_member_indices_);
    d_member_indices_ = nullptr;
  }
}

template <typename T>
void VirtualArrayAssemblyOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
  effective_wavelength_ = wavelength_.get();
}

template <typename T>
void VirtualArrayAssemblyOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("rd_maps");
  spec.output<holoscan::TensorMap>("rd_maps");
  spec.output<ArrayGeometry>("array_geometry");

  spec.param(tx_positions_, "tx_positions", "TX element positions",
             "TX element positions (m), row-major (num_tx, 3). Static.",
             std::vector<double>{});
  spec.param(rx_positions_, "rx_positions", "RX element positions",
             "RX element positions (m), row-major (num_rx, 3). Static.",
             std::vector<double>{});
  spec.param(wavelength_, "wavelength", "Operating wavelength",
             "Operating wavelength (m) for the emitted array_geometry.",
             0.0);
  add_override_port<double>(spec, "wavelength");
  spec.param(overlap_policy_, "overlap_policy", "Virtual overlap policy",
             "How coincident virtual elements are handled: dedup, average, "
             "or keep_all.",
             OverlapPolicy::kDedup);
}

template <typename T>
void VirtualArrayAssemblyOp<T>::ensure_device_plan(
    const detail::VirtualAssemblyPlan& plan, cudaStream_t stream) {
  const size_t src_count = plan.src_channel.size();
  if (src_count > d_src_capacity_) {
    if (d_src_channel_ != nullptr) {
      cudaFree(d_src_channel_);
      d_src_channel_ = nullptr;
    }
    if (cudaMalloc(&d_src_channel_, src_count * sizeof(int64_t)) !=
        cudaSuccess) {
      throw std::runtime_error(
          "VirtualArrayAssemblyOp: src_channel device allocation failed");
    }
    d_src_capacity_ = src_count;
  }
  if (!plan.src_channel.empty()) {
    if (cudaMemcpyAsync(d_src_channel_, plan.src_channel.data(),
                        src_count * sizeof(int64_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      throw std::runtime_error(
          "VirtualArrayAssemblyOp: src_channel H2D failed");
    }
  }

  const size_t offset_count = plan.member_offsets.size();
  if (offset_count > d_member_offsets_capacity_) {
    if (d_member_offsets_ != nullptr) {
      cudaFree(d_member_offsets_);
      d_member_offsets_ = nullptr;
    }
    if (cudaMalloc(&d_member_offsets_, offset_count * sizeof(int64_t)) !=
        cudaSuccess) {
      throw std::runtime_error(
          "VirtualArrayAssemblyOp: member_offsets device allocation failed");
    }
    d_member_offsets_capacity_ = offset_count;
  }
  if (!plan.member_offsets.empty()) {
    if (cudaMemcpyAsync(d_member_offsets_, plan.member_offsets.data(),
                        offset_count * sizeof(int64_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      throw std::runtime_error(
          "VirtualArrayAssemblyOp: member_offsets H2D failed");
    }
  }

  const size_t index_count = plan.member_indices.size();
  if (index_count > d_member_indices_capacity_) {
    if (d_member_indices_ != nullptr) {
      cudaFree(d_member_indices_);
      d_member_indices_ = nullptr;
    }
    if (cudaMalloc(&d_member_indices_, index_count * sizeof(int64_t)) !=
        cudaSuccess) {
      throw std::runtime_error(
          "VirtualArrayAssemblyOp: member_indices device allocation failed");
    }
    d_member_indices_capacity_ = index_count;
  }
  if (!plan.member_indices.empty()) {
    if (cudaMemcpyAsync(d_member_indices_, plan.member_indices.data(),
                        index_count * sizeof(int64_t), cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      throw std::runtime_error(
          "VirtualArrayAssemblyOp: member_indices H2D failed");
    }
  }
}

template <typename T>
void VirtualArrayAssemblyOp<T>::compute(InputContext& op_input,
                                      OutputContext& op_output,
                                      ExecutionContext& context) {
  (void)context;
  constexpr const char* kOpName = "VirtualArrayAssemblyOp";

  std::shared_ptr<holoscan::Tensor> input =
      receive_single(op_input, "rd_maps", kOpName);

  classify_input_dtype<T>(*input, "rd_maps", /*allow_real=*/false);
  require_cuda_accessible(*input, "rd_maps", kOpName);
  require_contiguous(*input, "rd_maps", kOpName);

  effective_wavelength_ =
      resolve_override_param(op_input, "wavelength", effective_wavelength_,
                             /*changed=*/nullptr);
  if (!(effective_wavelength_ > 0.0) ||
      !std::isfinite(effective_wavelength_)) {
    throw std::invalid_argument(std::string(kOpName) +
                                ": wavelength must be a positive finite "
                                "scalar (build-time or [live] override)");
  }

  const detail::RdMapsLayout layout =
      detail::parse_rd_maps_layout(*input, "rd_maps");

  const std::vector<double> tx_positions = tx_positions_.get();
  const std::vector<double> rx_positions = rx_positions_.get();
  const OverlapPolicy policy = overlap_policy_.get();

  if (plan_key_changed(tx_positions, rx_positions, policy, layout.num_tx,
                       layout.num_rx, cached_tx_positions_,
                       cached_rx_positions_, cached_policy_, cached_num_tx_,
                       cached_num_rx_)) {
    cached_tx_positions_ = tx_positions;
    cached_rx_positions_ = rx_positions;
    cached_policy_ = policy;
    cached_num_tx_ = layout.num_tx;
    cached_num_rx_ = layout.num_rx;
    cached_plan_ = detail::build_virtual_plan(
        cached_tx_positions_, cached_rx_positions_, layout.num_tx,
        layout.num_rx, cached_policy_);
  }

  cudaStream_t stream = op_input.receive_cuda_stream("rd_maps");
  ensure_device_plan(cached_plan_, stream);

  const DLDataType dtype = input->dtype();
  auto output = make_device_tensor(
      detail::assembled_rd_shape(layout, cached_plan_.num_virtual), dtype);

  detail::run_virtual_assembly_stage<T>(
      input->data(), output->data(), layout, cached_plan_, d_src_channel_,
      d_member_offsets_, d_member_indices_, stream);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "rd_maps");
  }
  holoscan::TensorMap out_map;
  out_map.insert({"rd_maps", output});
  op_output.emit(out_map, "rd_maps");

  const ArrayGeometry geometry =
      detail::make_array_geometry(cached_plan_, effective_wavelength_);
  op_output.emit(geometry, "array_geometry");

  retained_input_ = std::move(input);
}

template class VirtualArrayAssemblyOp<float>;
template class VirtualArrayAssemblyOp<double>;

}  // namespace holoscan::radar
