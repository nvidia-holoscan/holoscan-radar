// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_stage.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/extents_layout.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

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
VirtualArrayAssemblyOp<T>::VirtualArrayAssemblyOp(
    const MaxTensorExtents& extents) {
  const char* who = "VirtualArrayAssemblyOp";
  const std::int64_t max_virtual =
      extents.max_virtual_channels.has_value()
          ? require_extent(extents.max_virtual_channels,
                           "max_virtual_channels", who)
          : require_extent(extents.max_tx, "max_tx", who) *
                require_extent(extents.max_rx, "max_rx", who);
  const std::int64_t max_doppler =
      require_extent(extents.max_doppler_bins,
                     "max_doppler_bins", who);
  const std::int64_t max_range =
      require_extent(extents.max_range_bins,
                     "max_range_bins", who);
  extents_ = extents;
  max_rank_ = 3U;
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_virtual, max_doppler, max_range}, who);
  geom_byte_span_ = static_cast<std::size_t>(
      (max_virtual + 1) * 3) * sizeof(double);
}

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
void VirtualArrayAssemblyOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorAllocationBounds bounds{
      .max_rank = max_rank_,
      .max_byte_span = max_byte_span_};
  const TensorAllocationBounds geom_bounds{
      .max_rank = 2U,
      .max_byte_span = geom_byte_span_};
  spec.input(rd_maps).queue_depth(2U).expects_tensor(kDevice);
  spec.output(rd_maps_out).produces_tensor(kDevice).tensor_allocation(bounds);
  // Packed ArrayGeometry (host-planned; H2D each tick).
  spec.output(array_geometry)
      .produces_tensor(kDevice)
      .tensor_allocation(geom_bounds);
  // [live] POD sticky override. Leaving unconnected is legal — compute()
  // seeds from Params. Descriptor / vector live params are Params-only (§9).
  spec.input(wavelength).queue_depth(2U);
}

template <typename T>
Contract VirtualArrayAssemblyOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  maybe_add_live_override_to_contract(
      result, wavelength, params().expect_wavelength);
  return result;
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
                        offset_count * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
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
                        index_count * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error(
          "VirtualArrayAssemblyOp: member_indices H2D failed");
    }
  }
}

template <typename T>
expected<void, Error> VirtualArrayAssemblyOp<T>::compute(
    ExecutionContext& context) {
  constexpr const char* kOpName = "VirtualArrayAssemblyOp";

  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));

  try {
    classify_input_dtype<T>(input, "rd_maps", /*allow_real=*/false);
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const VirtualArrayAssemblyParams& pinned =
      context.params<VirtualArrayAssemblyParams>();

  HOLOSCAN_TRY(auto effective_wavelength,
               resolve_expected_override(pinned.expect_wavelength, wavelength,
                                         pinned.wavelength, sticky_wavelength_,
                                         pinned.sticky_wavelength));

  if (!(effective_wavelength > 0.0) || !std::isfinite(effective_wavelength)) {
    return fail_invalid(
        std::string(kOpName) +
        ": wavelength must be a positive finite "
        "scalar (build-time or [live] override)");
  }

  const detail::RdMapsLayout layout =
      detail::parse_rd_maps_layout(input, "rd_maps");

  const std::vector<double>& tx_positions = pinned.tx_positions;
  const std::vector<double>& rx_positions = pinned.rx_positions;
  const OverlapPolicy policy = pinned.overlap_policy;

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

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "VirtualArrayAssemblyOp: no CUDA compute stream"});
  }

  try {
    ensure_device_plan(cached_plan_, stream);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  const DLDataType dtype = input.dtype();
  const std::vector<int64_t> out_shape =
      detail::assembled_rd_shape(layout, cached_plan_.num_virtual);

  // Prepare both output loans before either emit so a late fill failure
  // cannot leave one port already published.
  const void* input_ptr = input.data();
  HOLOSCAN_TRY(
      auto rd_loan,
      prepare_device_tensor(
          rd_maps_out, context, out_shape, dtype,
          [&](void* out_ptr, std::size_t) -> expected<void, Error> {
            try {
              detail::run_virtual_assembly_stage<T>(
                  input_ptr, out_ptr, layout, cached_plan_, d_src_channel_,
                  d_member_offsets_, d_member_indices_, stream);
            } catch (const std::invalid_argument& e) {
              return fail_invalid(e.what());
            }
            return {};
          }));

  const ArrayGeometry geom =
      detail::make_array_geometry(cached_plan_, effective_wavelength);
  const std::vector<double> packed = pack_array_geometry(geom);
  const std::vector<int64_t> geom_shape = array_geometry_tensor_shape(geom);
  const DLDataType f64 = real_dtype<double>();
  HOLOSCAN_TRY(
      auto geom_loan,
      prepare_device_tensor(
          array_geometry, context, geom_shape, f64,
          [&](void* dst, std::size_t nbytes) -> expected<void, Error> {
            if (nbytes != packed.size() * sizeof(double)) {
              return fail_runtime(
                  "VirtualArrayAssemblyOp: array_geometry size mismatch");
            }
            if (cudaMemcpyAsync(dst, packed.data(), nbytes,
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
              return fail_runtime(
                  "VirtualArrayAssemblyOp: array_geometry H2D failed");
            }
            return {};
          }));

  HOLOSCAN_TRY(emit_prepared_tensor(rd_maps_out, context,
                                    std::move(rd_loan)));
  HOLOSCAN_TRY(emit_prepared_tensor(array_geometry, context,
                                    std::move(geom_loan)));

  retained_input_ = std::move(input);
  return {};
}

template class VirtualArrayAssemblyOp<float>;
template class VirtualArrayAssemblyOp<double>;

}  // namespace holoscan::radar
