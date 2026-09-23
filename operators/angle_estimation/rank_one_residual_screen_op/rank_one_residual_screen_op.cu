// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "angle_estimation/is_valid_mask.hpp"
#include "angle_estimation/rank_one_residual_screen_op/rank_one_residual_screen_stage.hpp"
#include "detection/cfar_detector_op/cfar_compact.cuh"
#include "radar_common/dsp/angle_convention.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"

namespace holoscan::radar {

template <typename T>
struct RankOneResidualScreenOp<T>::RuntimeState {
  detail::DetectionIdxCompactState compact;
  bool has_geometry{false};
  ArrayGeometry geometry_key{};
  std::shared_ptr<Tensor> positions;
};

namespace {

constexpr const char* kOpName = "RankOneResidualScreenOp";

expected<void, Error> invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void*, Error> writable(Tensor& tensor, const char* name) {
  auto result = tensor.mutable_data();
  if (!result) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": " + name + " is not writable"});
  }
  return *result;
}

template <typename T>
expected<void, Error> validate_inputs(const Tensor& snapshot,
                                      const Tensor& baseline, int64_t max_items,
                                      int64_t max_channels,
                                      int64_t max_snapshots) {
  try {
    require_cuda_accessible(snapshot, "snapshot", kOpName);
    require_contiguous(snapshot, "snapshot", kOpName);
    require_complex<T>(snapshot, "snapshot");
    require_cuda_accessible(baseline, "baseline_aoa", kOpName);
    require_contiguous(baseline, "baseline_aoa", kOpName);
    require_dtype(baseline, real_dtype<T>(), "baseline_aoa", kOpName);
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  }
  const auto& shape = snapshot.shape();
  if (shape.size() != 3U || shape[0] < 0 || shape[1] <= 0 || shape[2] <= 0) {
    return invalid(
        "RankOneResidualScreenOp: snapshot must have shape "
        "(num_items, num_channels, num_snapshots) with positive trailing "
        "extents");
  }
  if (baseline.shape() != std::vector<int64_t>{shape[0], 2}) {
    return invalid(
        "RankOneResidualScreenOp: baseline_aoa must have shape "
        "(num_items, 2) matching snapshot");
  }
  if (shape[0] > max_items || shape[1] > max_channels ||
      shape[2] > max_snapshots) {
    return invalid("RankOneResidualScreenOp: snapshot exceeds MaxTensorExtents");
  }
  return {};
}

}  // namespace

template <typename T>
RankOneResidualScreenOp<T>::RankOneResidualScreenOp(
    const MaxTensorExtents& extents) {
  max_items_ = require_extent(extents.max_items, "max_items", kOpName);
  max_channels_ =
      require_extent(extents.max_channels, "max_channels", kOpName);
  max_snapshots_ =
      require_extent(extents.max_snapshots, "max_snapshots", kOpName);
  const std::vector<int64_t> snapshot_bounds{
      max_items_, max_channels_, max_snapshots_};
  const std::vector<int64_t> item_bounds{max_items_};
  const std::vector<int64_t> valid_bounds{max_items_, max_snapshots_};
  snapshot_max_byte_span_ =
      tensor_byte_span(snapshot_bounds, complex_dtype<T>());
  selected_idx_max_byte_span_ = tensor_byte_span(
      item_bounds, DLDataType{kDLInt, 64U, 1U});
  valid_max_byte_span_ = tensor_byte_span(
      valid_bounds, DLDataType{kDLUInt, 8U, 1U});
  score_max_byte_span_ = tensor_byte_span(item_bounds, real_dtype<T>());
  mask_max_byte_span_ = tensor_byte_span(
      item_bounds, DLDataType{kDLUInt, 8U, 1U});
  state_ = std::make_shared<RuntimeState>();
}

template <typename T>
void RankOneResidualScreenOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{.memory_kind = MemoryKind::kCudaDevice,
                                        .dtype = std::nullopt,
                                        .rank = std::nullopt};
  spec.input(snapshot).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(baseline_aoa)
      .queue_depth(2U)
      .expects_tensor(tensor_input(device));
  spec.input(is_valid)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(device));
  spec.input(array_geometry).optional().queue_depth(2U);
  spec.output(snapshot_out)
      .produces_tensor(
          pooled_tensor_output(device, 3U, snapshot_max_byte_span_));
  spec.output(selected_idx)
      .produces_tensor(
          pooled_tensor_output(device, 1U, selected_idx_max_byte_span_));
  spec.output(is_valid_out)
      .produces_tensor(pooled_tensor_output(device, 2U, valid_max_byte_span_));
  spec.output(residual_score)
      .produces_tensor(pooled_tensor_output(device, 1U, score_max_byte_span_));
  spec.output(selection_mask)
      .produces_tensor(pooled_tensor_output(device, 1U, mask_max_byte_span_));
}

template <typename T>
Contract RankOneResidualScreenOp<T>::contract() const {
  Contract result;
  result
      .trigger(OnAligned{.ports = ports(snapshot, baseline_aoa),
                         .window = std::chrono::nanoseconds{0}})
      .input(snapshot, Exact{})
      .input(baseline_aoa, Exact{})
      .input(is_valid, Exact{})
      .on_missing(is_valid, SafeDefault{});
  add_live_override_to_contract(result, array_geometry);
  return result;
}

template <typename T>
expected<void, Error> RankOneResidualScreenOp<T>::compute(
    ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));
  HOLOSCAN_TRY(auto aoa, receive_tensor(baseline_aoa, kOpName));
  HOLOSCAN_RETURN_IF_ERROR(validate_inputs<T>(
      snap, aoa, max_items_, max_channels_, max_snapshots_));
  const auto& shape = snap.shape();
  const int64_t items = shape[0];
  const int64_t channels = shape[1];
  const int64_t snapshots = shape[2];

  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const auto& params = context.params<RankOneResidualScreenParams>();
  if (!std::isfinite(params.residual_threshold) ||
      params.residual_threshold < 0.0 || params.residual_threshold > 1.0) {
    return invalid(
        "RankOneResidualScreenOp: residual_threshold must be within [0, 1]");
  }
  HOLOSCAN_TRY(
      auto geometry,
      resolve_live_override_via(
          array_geometry, params.array_geometry,
          [](const WireArrayGeometry& value) { return from_wire(value); }));
  try {
    geometry.validate();
    (void)boundary_to_canonical(params.angle_convention, 0.0, 0.0);
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  }
  if (geometry.num_elements() != channels) {
    return invalid(
        "RankOneResidualScreenOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  const unsigned char* valid_ptr = nullptr;
  if (maybe_valid.has_value()) {
    try {
      detail::check_is_valid_mask(*maybe_valid, items, snapshots, kOpName);
    } catch (const std::invalid_argument& error) {
      return invalid(error.what());
    }
    valid_ptr = static_cast<const unsigned char*>(maybe_valid->data());
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "RankOneResidualScreenOp: no CUDA stream"});
  }
  if (!state_->has_geometry || state_->geometry_key != geometry) {
    state_->positions =
        make_device_tensor({channels, 3}, DLDataType{kDLFloat, 64U, 1U});
    HOLOSCAN_TRY(auto position_ptr,
                 writable(*state_->positions, "positions workspace"));
    if (cudaMemcpyAsync(position_ptr, geometry.element_positions.data(),
                        static_cast<size_t>(channels * 3) * sizeof(double),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure,
                                   "RankOneResidualScreenOp: geometry H2D failed"});
    }
    state_->geometry_key = geometry;
    state_->has_geometry = true;
  }

  const std::vector<int64_t> item_shape{items};
  // Allocate exact-sized outputs before launching device work.
  auto score_loan = residual_score.allocate_tensor(
      TensorLoanRequest{.shape = item_shape, .dtype = real_dtype<T>()});
  auto mask_loan = selection_mask.allocate_tensor(
      TensorLoanRequest{.shape = item_shape,
                        .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!score_loan) return make_unexpected(std::move(score_loan).error());
  if (!mask_loan) return make_unexpected(std::move(mask_loan).error());
  HOLOSCAN_TRY(auto score_writer, score_loan->write(context.cuda_stream()));
  HOLOSCAN_TRY(auto mask_writer, mask_loan->write(context.cuda_stream()));

  if (items > 0) {
    constexpr int block = 128;
    const int grid = static_cast<int>((items + block - 1) / block);
    detail::residual_screen_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<const cuda::std::complex<T>*>(snap.data()),
        static_cast<const T*>(aoa.data()), valid_ptr,
        static_cast<const double*>(state_->positions->data()), items, channels,
        snapshots, geometry.wavelength, params.angle_convention,
        static_cast<T>(params.residual_threshold),
        static_cast<T*>(score_writer.data()),
        static_cast<unsigned char*>(mask_writer.data()));
  }
  check_cuda("residual screen", kOpName);

  int64_t selected = 0;
  try {
    selected = detail::count_mask_detections(
        reinterpret_cast<const bool*>(mask_writer.data()), items,
        &state_->compact, stream);
    if (items > 0) {
      // Execute the device stage on the input runtime stream.
      detail::compact_detail::launch_exclusive_sum(
          state_->compact.d_offsets, items, &state_->compact, stream);
    }
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 std::string(kOpName) + ": " + error.what()});
  }

  const std::vector<int64_t> snapshot_shape{
      selected, channels, snapshots};
  const std::vector<int64_t> selected_shape{selected};
  const std::vector<int64_t> valid_shape{selected, snapshots};
  auto snapshot_loan = snapshot_out.allocate_tensor(TensorLoanRequest{
      .shape = snapshot_shape, .dtype = complex_dtype<T>()});
  auto idx_loan = selected_idx.allocate_tensor(TensorLoanRequest{
      .shape = selected_shape, .dtype = DLDataType{kDLInt, 64U, 1U}});
  auto valid_loan = is_valid_out.allocate_tensor(TensorLoanRequest{
      .shape = valid_shape,
      .dtype = DLDataType{kDLUInt, 8U, 1U}});
  if (!snapshot_loan) return make_unexpected(std::move(snapshot_loan).error());
  if (!idx_loan) return make_unexpected(std::move(idx_loan).error());
  if (!valid_loan) return make_unexpected(std::move(valid_loan).error());
  HOLOSCAN_TRY(auto snapshot_writer,
               snapshot_loan->write(context.cuda_stream()));
  HOLOSCAN_TRY(auto idx_writer, idx_loan->write(context.cuda_stream()));
  HOLOSCAN_TRY(auto valid_writer, valid_loan->write(context.cuda_stream()));

  if (selected > 0) {
    detail::compact_screen_kernel<T><<<static_cast<int>(items), 256, 0, stream>>>(
        static_cast<const cuda::std::complex<T>*>(snap.data()), valid_ptr,
        static_cast<const unsigned char*>(mask_writer.data()),
        state_->compact.d_offsets, items, channels, snapshots,
        static_cast<cuda::std::complex<T>*>(snapshot_writer.data()),
        static_cast<unsigned char*>(valid_writer.data()),
        static_cast<int64_t*>(idx_writer.data()));
    check_cuda("screen compaction", kOpName);
  }

  HOLOSCAN_TRY(std::move(score_writer).commit());
  HOLOSCAN_TRY(std::move(mask_writer).commit());
  HOLOSCAN_TRY(std::move(snapshot_writer).commit());
  HOLOSCAN_TRY(std::move(idx_writer).commit());
  HOLOSCAN_TRY(std::move(valid_writer).commit());
  // Publish coherent outputs in dependency order and retain borrowed inputs.
  HOLOSCAN_TRY(emit_prepared_tensor(
      selected_idx, context, std::move(*idx_loan)));
  (void)emit_prepared_tensor(is_valid_out, context, std::move(*valid_loan));
  (void)emit_prepared_tensor(residual_score, context, std::move(*score_loan));
  (void)emit_prepared_tensor(selection_mask, context, std::move(*mask_loan));
  HOLOSCAN_TRY(emit_prepared_tensor(
      snapshot_out, context, std::move(*snapshot_loan)));

  retained_snapshot_ = std::move(snap);
  retained_aoa_ = std::move(aoa);
  if (maybe_valid.has_value()) retained_valid_ = std::move(*maybe_valid);
  return {};
}

template class RankOneResidualScreenOp<float>;
template class RankOneResidualScreenOp<double>;

}  // namespace holoscan::radar
