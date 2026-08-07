// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/dsp/subarray_selection.hpp"
#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {
constexpr const char* kOpName = "SubarraySelectionOp";
}  // namespace

template <typename T>
void SubarraySelectionOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
}

template <typename T>
void SubarraySelectionOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("snapshot");
  spec.input<holoscan::TensorMap>("is_valid")
      .condition(holoscan::ConditionType::kNone);

  spec.output<holoscan::TensorMap>("snapshot");
  spec.output<ArrayGeometry>("array_geometry");
  spec.output<holoscan::TensorMap>("is_valid")
      .condition(holoscan::ConditionType::kNone);

  spec.param(array_geometry_, "array_geometry", "Array geometry",
             "Holey virtual-array element positions and wavelength",
             ArrayGeometry{});
  add_override_port<ArrayGeometry>(spec, "array_geometry");
  spec.param(subarray_, "subarray", "Explicit sub-array",
             "Explicit channel indices (empty = auto largest filled ULA/URA)",
             std::vector<int64_t>{});
  add_override_port<std::vector<int64_t>>(spec, "subarray");
  spec.param(min_subarray_size_, "min_subarray_size", "Min sub-array size",
             "Auto-selection errors below this many channels", int64_t{2});
}

template <typename T>
void SubarraySelectionOp<T>::compute(InputContext& op_input,
                                     OutputContext& op_output,
                                     ExecutionContext&) {
  if (!seeded_) {
    effective_geometry_ = array_geometry_.get();
    effective_geometry_.validate();
    effective_subarray_ = subarray_.get();
    effective_min_ = min_subarray_size_.get();
    seeded_ = true;
  }

  bool geom_changed = false;
  bool subarray_changed = false;
  effective_geometry_ = resolve_override_param(
      op_input, "array_geometry", effective_geometry_, &geom_changed);
  effective_geometry_.validate();
  effective_subarray_ = resolve_override_param(op_input, "subarray",
                                               effective_subarray_,
                                               &subarray_changed);

  auto snapshot = receive_single(op_input, "snapshot", kOpName);
  require_cuda_accessible(*snapshot, "snapshot", kOpName);
  require_contiguous(*snapshot, "snapshot", kOpName);
  require_dtype(*snapshot, complex_dtype<T>(), "snapshot", kOpName);

  const auto& shape = snapshot->shape();
  if (shape.size() != 2 && shape.size() != 3) {
    throw std::invalid_argument(
        "SubarraySelectionOp: snapshot must have shape (num_items, "
        "num_channels) or (num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape.size() == 3 ? shape[2] : 1;
  if (num_channels != effective_geometry_.num_elements()) {
    throw std::invalid_argument(
        "SubarraySelectionOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  auto is_valid = receive_optional(op_input, "is_valid", kOpName);
  auto stream = op_input.receive_cuda_stream("snapshot");

  // (Re)build the selection + upload the channel-index table only when the
  // geometry / selection config changes.
  std::ostringstream ks;
  ks << effective_geometry_.wavelength << '|';
  for (double v : effective_geometry_.element_positions) {
    ks << v << ',';
  }
  ks << '|';
  for (int64_t v : effective_subarray_) {
    ks << v << ',';
  }
  ks << '|' << effective_min_;
  const std::string key = ks.str();
  if (key != sel_key_) {
    selection_ = build_subarray_selection(effective_geometry_,
                                          effective_subarray_, effective_min_);
    sel_key_ = key;
    if (selection_.reduced) {
      d_channel_index_ = make_device_tensor({selection_.subarray_size},
                                            DLDataType{kDLInt, 32, 1});
      if (cudaMemcpyAsync(d_channel_index_->data(),
                          selection_.channel_index.data(),
                          selection_.channel_index.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        throw std::runtime_error(
            "SubarraySelectionOp: channel-index H2D failed");
      }
      HOLOSCAN_LOG_INFO(
          "SubarraySelectionOp: selected {}/{} channels (uniform sub-array)",
          selection_.subarray_size, num_channels);
    }
  }

  std::shared_ptr<holoscan::Tensor> out_snapshot;
  ArrayGeometry out_geometry;
  if (!selection_.reduced) {
    // Already a filled ULA/URA — pass through unchanged.
    out_snapshot = snapshot;
    out_geometry = effective_geometry_;
  } else {
    const int64_t l = selection_.subarray_size;
    const std::vector<int64_t> out_shape =
        shape.size() == 3 ? std::vector<int64_t>{num_items, l, num_snapshots}
                          : std::vector<int64_t>{num_items, l};
    out_snapshot = make_device_tensor(out_shape, complex_dtype<T>());
    run_spatial_smoothing_gather<T>(
        static_cast<const cuda::std::complex<T>*>(snapshot->data()),
        /*mask=*/nullptr, num_items, num_channels, num_snapshots,
        static_cast<const int32_t*>(d_channel_index_->data()),
        /*fb_flip=*/nullptr, /*num_subarrays=*/1, l,
        /*forward_backward=*/false,
        static_cast<cuda::std::complex<T>*>(out_snapshot->data()),
        /*ss_mask=*/nullptr, stream);
    check_cuda("subarray gather", kOpName);
    out_geometry = selection_.reduced_geometry;
  }

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "snapshot");
    if (is_valid) {
      op_output.set_cuda_stream(stream, "is_valid");
    }
  }
  emit_single(op_output, "snapshot", out_snapshot);
  op_output.emit(out_geometry, "array_geometry");
  if (is_valid) {
    emit_single(op_output, "is_valid", is_valid);
  }
}

template class SubarraySelectionOp<float>;
template class SubarraySelectionOp<double>;

}  // namespace holoscan::radar
