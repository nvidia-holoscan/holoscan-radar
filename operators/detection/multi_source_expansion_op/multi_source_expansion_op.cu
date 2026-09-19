// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "detection/cfar_detector_op/cfar_compact.cuh"
#include "detection/multi_source_expansion_op/multi_source_expansion_stage.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

template <typename T>
struct MultiSourceExpansionOp<T>::RuntimeState {
  detail::DetectionIdxCompactState compact;
  std::shared_ptr<Tensor> selected_ordinal;
  std::shared_ptr<Tensor> error;
};

namespace {

constexpr const char* kOpName = "MultiSourceExpansionOp";
constexpr DLDataType kInt64Dtype{kDLInt, 64U, 1U};
constexpr DLDataType kInt32Dtype{kDLInt, 32U, 1U};
constexpr std::int64_t kMaxIndexColumns = 10;

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

expected<void, Error> require_device_tensor(const Tensor& tensor,
                                            const char* port,
                                            DLDataType dtype) {
  try {
    require_cuda_accessible(tensor, port, kOpName);
    require_contiguous(tensor, port, kOpName);
    require_dtype(tensor, dtype, port, kOpName);
  } catch (const std::invalid_argument& error) {
    return invalid(error.what());
  }
  return {};
}

const char* expansion_error_message(int32_t code) {
  switch (static_cast<detail::ExpansionError>(code)) {
    case detail::ExpansionError::kSelectedOrder:
      return "MultiSourceExpansionOp: selected_idx must be strictly ascending, "
             "unique, and in range";
    case detail::ExpansionError::kSourceCount:
      return "MultiSourceExpansionOp: source_count must be within "
             "[0, doa_multi.shape(1)]";
    case detail::ExpansionError::kNonFiniteDoa:
      return "MultiSourceExpansionOp: confirmed source angles must be finite";
    case detail::ExpansionError::kNone:
      break;
  }
  return "MultiSourceExpansionOp: expansion validation failed";
}

}  // namespace

template <typename T>
MultiSourceExpansionOp<T>::MultiSourceExpansionOp(
    const MaxTensorExtents& extents) {
  max_items_ = require_extent(extents.max_items, "max_items", kOpName);
  max_points_ = require_extent(extents.max_points, "max_points", kOpName);
  max_sources_ =
      require_extent(extents.max_sources, "max_sources", kOpName);
  const std::vector<int64_t> idx_bounds{
      max_points_, kMaxIndexColumns};
  const std::vector<int64_t> aoa_bounds{max_points_, 2};
  idx_max_byte_span_ = tensor_byte_span(idx_bounds, kInt64Dtype);
  aoa_max_byte_span_ = tensor_byte_span(aoa_bounds, real_dtype<T>());
  state_ = std::make_shared<RuntimeState>();
  state_->selected_ordinal = make_device_tensor({max_items_}, kInt64Dtype);
  state_->error = make_device_tensor({1}, kInt32Dtype);
}

template <typename T>
void MultiSourceExpansionOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{.memory_kind = MemoryKind::kCudaDevice,
                                        .dtype = std::nullopt,
                                        .rank = std::nullopt};
  spec.input(detection_idx).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(baseline_aoa).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(selected_idx).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(resolved_aoa).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(doa_multi).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(source_count).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.output(detection_idx_out)
      .produces_tensor(pooled_tensor_output(device, 2U, idx_max_byte_span_));
  spec.output(aoa_estimates_out)
      .produces_tensor(pooled_tensor_output(device, 2U, aoa_max_byte_span_));
}

template <typename T>
Contract MultiSourceExpansionOp<T>::contract() const {
  Contract result;
  result
      .trigger(OnAligned{.ports = ports(detection_idx, baseline_aoa,
                                        selected_idx, resolved_aoa, doa_multi,
                                        source_count),
                         .window = std::chrono::nanoseconds{0}})
      .input(detection_idx, Exact{})
      .input(baseline_aoa, Exact{})
      .input(selected_idx, Exact{})
      .input(resolved_aoa, Exact{})
      .input(doa_multi, Exact{})
      .input(source_count, Exact{});
  return result;
}

template <typename T>
expected<void, Error> MultiSourceExpansionOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto idx, receive_tensor(detection_idx, kOpName));
  HOLOSCAN_TRY(auto baseline, receive_tensor(baseline_aoa, kOpName));
  HOLOSCAN_TRY(auto selected, receive_tensor(selected_idx, kOpName));
  HOLOSCAN_TRY(auto resolved, receive_tensor(resolved_aoa, kOpName));
  HOLOSCAN_TRY(auto multi, receive_tensor(doa_multi, kOpName));
  HOLOSCAN_TRY(auto counts, receive_tensor(source_count, kOpName));

  HOLOSCAN_RETURN_IF_ERROR(require_device_tensor(idx, "detection_idx",
                                                 kInt64Dtype));
  HOLOSCAN_RETURN_IF_ERROR(require_device_tensor(baseline, "baseline_aoa",
                                                 real_dtype<T>()));
  HOLOSCAN_RETURN_IF_ERROR(require_device_tensor(selected, "selected_idx",
                                                 kInt64Dtype));
  HOLOSCAN_RETURN_IF_ERROR(require_device_tensor(resolved, "resolved_aoa",
                                                 real_dtype<T>()));
  HOLOSCAN_RETURN_IF_ERROR(require_device_tensor(multi, "doa_multi",
                                                 real_dtype<T>()));
  HOLOSCAN_RETURN_IF_ERROR(require_device_tensor(counts, "source_count",
                                                 kInt32Dtype));

  const auto& idx_shape = idx.shape();
  if (idx_shape.size() != 2U || idx_shape[0] < 0 || idx_shape[1] < 2 ||
      idx_shape[1] > kMaxIndexColumns) {
    return invalid(
        "MultiSourceExpansionOp: detection_idx must have shape "
        "(num_items, 2 [+ n_batch]) with at most 10 columns");
  }
  const int64_t items = idx_shape[0];
  const int64_t idx_cols = idx_shape[1];
  if (items > max_items_) {
    return invalid("MultiSourceExpansionOp: num_items exceeds max_items");
  }
  if (baseline.shape() != std::vector<int64_t>{items, 2}) {
    return invalid(
        "MultiSourceExpansionOp: baseline_aoa must have shape "
        "(num_items, 2)");
  }
  if (selected.shape().size() != 1U || selected.shape()[0] < 0) {
    return invalid(
        "MultiSourceExpansionOp: selected_idx must have shape "
        "(num_selected,)");
  }
  const int64_t num_selected = selected.shape()[0];
  if (num_selected > items) {
    return invalid(
        "MultiSourceExpansionOp: num_selected cannot exceed num_items");
  }
  if (resolved.shape() != std::vector<int64_t>{num_selected, 2}) {
    return invalid(
        "MultiSourceExpansionOp: resolved_aoa must have shape "
        "(num_selected, 2)");
  }
  const auto& multi_shape = multi.shape();
  if (multi_shape.size() != 3U || multi_shape[0] != num_selected ||
      multi_shape[1] <= 0 || multi_shape[1] > max_sources_ ||
      multi_shape[2] != 2) {
    return invalid(
        "MultiSourceExpansionOp: doa_multi must have shape "
        "(num_selected, max_sources, 2) within ctor bounds");
  }
  if (counts.shape() != std::vector<int64_t>{num_selected}) {
    return invalid(
        "MultiSourceExpansionOp: source_count must have shape "
        "(num_selected,)");
  }
  const int64_t source_slots = multi_shape[1];

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "MultiSourceExpansionOp: no CUDA stream"});
  }
  try {
    state_->compact.ensure_offsets(items);
    state_->compact.ensure_count_scalar();
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 std::string(kOpName) + ": " + error.what()});
  }
  HOLOSCAN_TRY(auto ordinal_ptr,
               writable(*state_->selected_ordinal, "selected ordinal"));
  HOLOSCAN_TRY(auto error_ptr, writable(*state_->error, "error scalar"));
  detail::build_expansion_plan_kernel<T><<<1, 1, 0, stream>>>(
      static_cast<const int64_t*>(selected.data()), num_selected,
      static_cast<const T*>(resolved.data()), static_cast<const T*>(multi.data()),
      static_cast<const int32_t*>(counts.data()), items, source_slots,
      state_->compact.d_offsets, static_cast<int64_t*>(ordinal_ptr),
      state_->compact.d_count, static_cast<int32_t*>(error_ptr));
  check_cuda("build expansion plan", kOpName);

  int64_t output_rows = 0;
  int32_t host_error = 0;
  if (cudaMemcpyAsync(&output_rows, state_->compact.d_count, sizeof(output_rows),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaMemcpyAsync(&host_error, error_ptr, sizeof(host_error),
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              "MultiSourceExpansionOp: validation readback failed"});
  }
  if (host_error != 0) return invalid(expansion_error_message(host_error));
  if (output_rows > max_points_) {
    return invalid(
        "MultiSourceExpansionOp: expanded rows exceed max_points");
  }
  try {
    if (items > 0) {
      detail::compact_detail::launch_exclusive_sum(
          state_->compact.d_offsets, items, &state_->compact, stream);
    }
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 std::string(kOpName) + ": " + error.what()});
  }

  const std::vector<int64_t> idx_output_shape{output_rows, idx_cols};
  const std::vector<int64_t> aoa_output_shape{output_rows, 2};
  auto idx_loan = detection_idx_out.allocate_tensor(TensorLoanRequest{
      .shape = idx_output_shape, .dtype = kInt64Dtype});
  auto aoa_loan = aoa_estimates_out.allocate_tensor(TensorLoanRequest{
      .shape = aoa_output_shape, .dtype = real_dtype<T>()});
  if (!idx_loan) return make_unexpected(std::move(idx_loan).error());
  if (!aoa_loan) return make_unexpected(std::move(aoa_loan).error());
  HOLOSCAN_TRY(auto idx_writer, idx_loan->write(context.cuda_stream()));
  HOLOSCAN_TRY(auto aoa_writer, aoa_loan->write(context.cuda_stream()));
  if (items > 0) {
    detail::scatter_expanded_rows_kernel<T>
        <<<static_cast<int>(items), 128, 0, stream>>>(
            static_cast<const int64_t*>(idx.data()),
            static_cast<const T*>(baseline.data()),
            static_cast<const int64_t*>(ordinal_ptr),
            static_cast<const T*>(resolved.data()),
            static_cast<const T*>(multi.data()),
            static_cast<const int32_t*>(counts.data()),
            state_->compact.d_offsets, items, idx_cols, source_slots,
            static_cast<int64_t*>(idx_writer.data()),
            static_cast<T*>(aoa_writer.data()));
    check_cuda("scatter expanded rows", kOpName);
  }

  HOLOSCAN_TRY(std::move(idx_writer).commit());
  HOLOSCAN_TRY(std::move(aoa_writer).commit());
  HOLOSCAN_TRY(emit_prepared_tensor(
      aoa_estimates_out, context, std::move(*aoa_loan)));
  HOLOSCAN_TRY(emit_prepared_tensor(
      detection_idx_out, context, std::move(*idx_loan)));

  retained_detection_idx_ = std::move(idx);
  retained_baseline_ = std::move(baseline);
  retained_selected_ = std::move(selected);
  retained_resolved_ = std::move(resolved);
  retained_multi_ = std::move(multi);
  retained_count_ = std::move(counts);
  return {};
}

template class MultiSourceExpansionOp<float>;
template class MultiSourceExpansionOp<double>;

}  // namespace holoscan::radar
