// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
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

constexpr const char* kOpName = "VirtualArrayAssemblyOp";

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

expected<void, Error> copy_index_table(const std::shared_ptr<Tensor>& dst,
                                       std::size_t capacity,
                                       const std::vector<int64_t>& host,
                                       const char* what,
                                       cudaStream_t stream) {
  if (host.size() > capacity) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": " + what +
            " length exceeds ctor-sized index table"});
  }
  if (host.empty()) {
    return {};
  }
  auto mut = dst->mutable_data();
  if (!mut) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": " + what + " not writable"});
  }
  if (cudaMemcpyAsync(*mut, host.data(), host.size() * sizeof(int64_t),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": " + what + " H2D failed"});
  }
  return {};
}

// Stage knobs plus this tick's parsed layout, gather plan, and emit
// geometry. Trailing rank is (M_tx, N_rx, D, R), not an FMCW (chirps, bins)
// cube, so this does not call collapse_named_tx_rx_batch.
struct VirtualAssemblyStagePlan {
  detail::VirtualAssemblyStageConfig stage;
  detail::RdMapsLayout layout;
  detail::VirtualAssemblyPlan gather;
  std::vector<int64_t> out_shape;
  std::vector<int64_t> geom_shape;
};

// Merge the Exact [live] wavelength onto pinned Params. Map the
// non-positive sentinel by rejecting it (wavelength is required).
// tx_positions, rx_positions, and overlap_policy are Params-only.
expected<detail::VirtualAssemblyStageConfig, Error> resolve_params(
    Input<double>& wavelength_in,
    const VirtualArrayAssemblyParams& pinned) {
  HOLOSCAN_TRY(auto wavelength,
               resolve_live_override(wavelength_in, pinned.wavelength));
  if (!(wavelength > 0.0) || !std::isfinite(wavelength)) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": wavelength must be a positive finite "
            "scalar (build-time or [live] override)"});
  }
  return detail::VirtualAssemblyStageConfig{
      .tx_positions = pinned.tx_positions,
      .rx_positions = pinned.rx_positions,
      .wavelength = wavelength,
      .overlap_policy = pinned.overlap_policy};
}

// Bind the stage config to this input: parse rank-4 layout, group virtual
// elements, and size both outputs.
expected<VirtualAssemblyStagePlan, Error> build_assembly_stage_plan(
    detail::VirtualAssemblyStageConfig stage, const Tensor& input,
    const MaxTensorExtents& extents) {
  VirtualAssemblyStagePlan plan;
  plan.stage = std::move(stage);
  try {
    plan.layout = detail::parse_rd_maps_layout(input, "rd_maps");
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }

  if (plan.layout.num_tx > *extents.max_tx ||
      plan.layout.num_rx > *extents.max_rx ||
      plan.layout.num_doppler > *extents.max_doppler_bins ||
      plan.layout.num_range > *extents.max_range_bins) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": input (M_tx, N_rx, D, R) exceeds MaxTensorExtents"});
  }

  try {
    plan.gather = detail::build_virtual_plan(
        plan.stage.tx_positions, plan.stage.rx_positions, plan.layout.num_tx,
        plan.layout.num_rx, plan.stage.overlap_policy);
  } catch (const std::invalid_argument& e) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, e.what()});
  }

  const int64_t max_virtual =
      extents.max_virtual_channels.has_value()
          ? *extents.max_virtual_channels
          : (*extents.max_tx) * (*extents.max_rx);
  if (plan.gather.num_virtual > max_virtual) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": num_virtual exceeds MaxTensorExtents"});
  }

  plan.out_shape =
      detail::assembled_rd_shape(plan.layout, plan.gather.num_virtual);
  plan.geom_shape = {plan.gather.num_virtual + 1, 3};
  return plan;
}

}  // namespace

template <typename T>
VirtualArrayAssemblyOp<T>::VirtualArrayAssemblyOp(
    const MaxTensorExtents& extents) {
  const char* who = kOpName;
  const std::int64_t max_tx =
      require_extent(extents.max_tx, "max_tx", who);
  const std::int64_t max_rx =
      require_extent(extents.max_rx, "max_rx", who);
  const std::int64_t max_doppler =
      require_extent(extents.max_doppler_bins, "max_doppler_bins", who);
  const std::int64_t max_range =
      require_extent(extents.max_range_bins, "max_range_bins", who);
  const std::int64_t max_virtual =
      extents.max_virtual_channels.has_value()
          ? require_extent(extents.max_virtual_channels,
                           "max_virtual_channels", who)
          : max_tx * max_rx;
  extents_ = extents;
  max_rank_ = 3U;
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_virtual, max_doppler, max_range}, who);
  geom_byte_span_ = static_cast<std::size_t>(
      (max_virtual + 1) * 3) * sizeof(double);

  src_channel_capacity_ = static_cast<std::size_t>(max_virtual);
  member_offsets_capacity_ =
      static_cast<std::size_t>(max_virtual) + 1U;
  member_indices_capacity_ =
      static_cast<std::size_t>(max_tx) * static_cast<std::size_t>(max_rx);
  src_channel_ = make_device_tensor(
      {max_virtual}, DLDataType{kDLInt, 64, 1});
  member_offsets_ = make_device_tensor(
      {max_virtual + 1}, DLDataType{kDLInt, 64, 1});
  member_indices_ = make_device_tensor(
      {max_tx * max_rx}, DLDataType{kDLInt, 64, 1});
}

template <typename T>
void VirtualArrayAssemblyOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds bounds{
      .max_rank = max_rank_,
      .max_byte_span = max_byte_span_};
  const TensorBounds geom_bounds{
      .max_rank = 2U,
      .max_byte_span = geom_byte_span_};

  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(rd_maps_out)
      .produces_tensor(pooled_tensor_output(kDevice, bounds));
  spec.output(array_geometry)
      .produces_tensor(pooled_tensor_output(kDevice, geom_bounds));

  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(wavelength).queue_depth(2U);
}

template <typename T>
Contract VirtualArrayAssemblyOp<T>::contract() const {
  // Trigger on the data tensor. The Exact override is declared whether or
  // not the application wires it: an unwired one compiles to a warning and
  // reads as a miss.
  Contract result;
  result.trigger(OnEach{rd_maps});
  add_live_override_to_contract(result, wavelength);
  return result;
}

template <typename T>
expected<void, Error> VirtualArrayAssemblyOp<T>::ensure_device_index_tables(
    const detail::VirtualAssemblyStageConfig& stage,
    const detail::VirtualAssemblyPlan& plan, cudaStream_t stream) {
  if (!plan_key_changed(stage.tx_positions, stage.rx_positions,
                        stage.overlap_policy, plan.num_tx, plan.num_rx,
                        cached_tx_positions_, cached_rx_positions_,
                        cached_policy_, cached_num_tx_, cached_num_rx_)) {
    return {};
  }

  HOLOSCAN_TRY(copy_index_table(src_channel_, src_channel_capacity_,
                                plan.src_channel, "src_channel", stream));
  HOLOSCAN_TRY(copy_index_table(member_offsets_, member_offsets_capacity_,
                                plan.member_offsets, "member_offsets",
                                stream));
  HOLOSCAN_TRY(copy_index_table(member_indices_, member_indices_capacity_,
                                plan.member_indices, "member_indices",
                                stream));

  cached_tx_positions_ = stage.tx_positions;
  cached_rx_positions_ = stage.rx_positions;
  cached_policy_ = stage.overlap_policy;
  cached_num_tx_ = plan.num_tx;
  cached_num_rx_ = plan.num_rx;
  return {};
}

template <typename T>
expected<void, Error> VirtualArrayAssemblyOp<T>::compute(
    ExecutionContext& context) {
  // Receive rd_maps; require a device-contiguous complex cube.
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));
  try {
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
    require_complex<T>(input, "rd_maps");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // This tick's Params: pinned defaults with any matched [live] override.
  const VirtualArrayAssemblyParams& pinned =
      context.params<VirtualArrayAssemblyParams>();
  HOLOSCAN_TRY(auto stage, resolve_params(wavelength, pinned));

  // Trailing rank is (M_tx, N_rx, D, R), not an FMCW (chirps, bins)
  // cube — layout parse lives in the plan builder.
  HOLOSCAN_TRY(auto plan,
               build_assembly_stage_plan(std::move(stage), input, extents_));

  // Gather and geometry H2D run on the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              std::string(kOpName) + ": no CUDA compute stream"});
  }

  HOLOSCAN_TRY(ensure_device_index_tables(plan.stage, plan.gather, stream));

  // Prepare both required outputs before either emit so a late fill
  // failure cannot leave one port already published.
  const void* input_ptr = input.data();
  const int64_t* d_src =
      static_cast<const int64_t*>(src_channel_->data());
  const int64_t* d_off =
      static_cast<const int64_t*>(member_offsets_->data());
  const int64_t* d_idx =
      static_cast<const int64_t*>(member_indices_->data());
  HOLOSCAN_TRY(
      auto rd_loan,
      prepare_device_tensor(
          rd_maps_out, context, plan.out_shape, input.dtype(),
          [&](void* out_ptr, std::size_t) -> expected<void, Error> {
            try {
              detail::run_virtual_assembly_stage<T>(
                  input_ptr, out_ptr, plan.layout, plan.gather, d_src, d_off,
                  d_idx, stream);
            } catch (const std::invalid_argument& e) {
              return fail_invalid(e.what());
            } catch (const std::runtime_error& e) {
              return fail_runtime(e.what());
            }
            return {};
          }));

  const ArrayGeometry geom =
      detail::make_array_geometry(plan.gather, plan.stage.wavelength);
  const std::vector<double> packed = pack_array_geometry(geom);
  const DLDataType f64 = real_dtype<double>();
  HOLOSCAN_TRY(
      auto geom_loan,
      prepare_device_tensor(
          array_geometry, context, plan.geom_shape, f64,
          [&](void* dst, std::size_t nbytes) -> expected<void, Error> {
            if (nbytes != packed.size() * sizeof(double)) {
              return fail_runtime(
                  std::string(kOpName) + ": array_geometry size mismatch");
            }
            if (cudaMemcpyAsync(dst, packed.data(), nbytes,
                                cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
              return fail_runtime(
                  std::string(kOpName) + ": array_geometry H2D failed");
            }
            return {};
          }));

  HOLOSCAN_TRY(emit_prepared_tensor(rd_maps_out, context,
                                    std::move(rd_loan)));
  HOLOSCAN_TRY(emit_prepared_tensor(array_geometry, context,
                                    std::move(geom_loan)));

  // Keep rd_maps alive until stream-ordered reads complete.
  retained_input_ = std::move(input);
  return {};
}

template class VirtualArrayAssemblyOp<float>;
template class VirtualArrayAssemblyOp<double>;

}  // namespace holoscan::radar
