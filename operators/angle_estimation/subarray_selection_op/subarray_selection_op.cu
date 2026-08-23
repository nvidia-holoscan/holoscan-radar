// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/logger/logger.hpp>

#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/dsp/spatial_smoothing_device.hpp"
#include "radar_common/dsp/subarray_selection.hpp"
#include "radar_common/sdk/config_tensor.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "SubarraySelectionOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

expected<void*, Error> writable_ptr(holoscan::Tensor& tensor,
                                    const char* what) {
  auto mut = tensor.mutable_data();
  if (!mut) {
    return make_unexpected(Error{
        ErrorCode::kFailure,
        std::string(kOpName) + ": " + what + " is not writable: " +
            mut.error().message});
  }
  return *mut;
}

}  // namespace

template <typename T>
SubarraySelectionOp<T>::SubarraySelectionOp(
    const MaxTensorExtents& extents) {
  extents_ = extents;
  if (extents.max_subarray_channels.has_value()) {
    subarray_max_extent_ = require_extent(
        extents.max_subarray_channels, "max_subarray_channels",
        "SubarraySelectionOp");
  } else if (extents.max_virtual_channels.has_value()) {
    subarray_max_extent_ = require_extent(
        extents.max_virtual_channels, "max_virtual_channels",
        "SubarraySelectionOp");
  } else if (extents.max_channels.has_value()) {
    subarray_max_extent_ = require_extent(
        extents.max_channels, "max_channels", "SubarraySelectionOp");
  }
  if (!extents.max_items.has_value() &&
      !extents.max_subarray_channels.has_value() &&
      !extents.max_snapshots.has_value()) {
    return;  // Python / non-pooled: leave byte spans at 0.
  }
  const char* who = "SubarraySelectionOp";
  const std::int64_t max_items =
      require_extent(extents.max_items,
                     "max_items", who);
  const std::int64_t max_sub =
      require_extent(extents.max_subarray_channels,
                     "max_subarray_channels", who);
  const std::int64_t max_snap =
      require_extent(extents.max_snapshots,
                     "max_snapshots", who);
  snapshot_max_byte_span_ = byte_span_complex_cube<T>(
      {max_items, max_sub, max_snap}, who);
  geom_max_byte_span_ = byte_span_elems(
      product_extents({max_sub + 1, 3}, who),
      sizeof(double), who);
  is_valid_max_byte_span_ = byte_span_elems(
      product_extents({max_items, max_snap}, who), 1U, who);
}

template <typename T>
void SubarraySelectionOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  constexpr TensorPortLayout kHostIndexVector{
      .memory_kind = MemoryKind::kHost,
      .dtype = DLDataType{kDLInt, 64U, 1U},
      .rank = 1U,
  };
  spec.input(snapshot).queue_depth(2U).expects_tensor(kDevice);
  // Optional is_valid: omitted from contract so unconnected graphs compile.
  spec.input(is_valid).queue_depth(2U).expects_tensor(kDevice);
  spec.input(array_geometry_in).queue_depth(2U);
  spec.input(subarray)
      .queue_depth(2U)
      .expects_tensor(kHostIndexVector);
  spec.output(snapshot_out)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = max_rank_,
          .max_byte_span = snapshot_max_byte_span_});
  // Packed ArrayGeometry (host-planned; H2D each rebuild / tick).
  spec.output(array_geometry)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = 2U,
          .max_byte_span = geom_max_byte_span_});
  spec.output(is_valid_out)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = max_rank_,
          .max_byte_span = is_valid_max_byte_span_});
}

template <typename T>
Contract SubarraySelectionOp<T>::contract() const {
  // Trigger on the data tensor only. Optional is_valid is omitted so
  // unconnected graphs do not yield TEMPORAL_INPUT_UNCONNECTED (same pattern
  // as BeamscanEstimatorOp).
  Contract result;
  result.trigger(OnEach{snapshot});
  maybe_add_live_override_to_contract(
      result, array_geometry_in, params().expect_array_geometry);
  maybe_add_live_override_to_contract(
      result, subarray, params().expect_subarray);
  return result;
}

template <typename T>
expected<void, Error> SubarraySelectionOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));

  try {
    require_cuda_accessible(snap, "snapshot", kOpName);
    require_contiguous(snap, "snapshot", kOpName);
    require_dtype(snap, complex_dtype<T>(), "snapshot", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto& shape = snap.shape();
  if (shape.size() != 2 && shape.size() != 3) {
    return fail_invalid(
        "SubarraySelectionOp: snapshot must have shape (num_items, "
        "num_channels) or (num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  const int64_t num_snapshots = shape.size() == 3 ? shape[2] : 1;
  const SubarraySelectionParams& pinned =
      context.params<SubarraySelectionParams>();
  HOLOSCAN_TRY(
      auto geometry,
      resolve_expected_override_via(
          pinned.expect_array_geometry, array_geometry_in,
          pinned.array_geometry, sticky_array_geometry_,
          pinned.sticky_array_geometry,
          [](const WireArrayGeometry& value) { return from_wire(value); }));
  const int64_t live_subarray_bound =
      subarray_max_extent_ > 0 ? subarray_max_extent_ : num_channels;
  HOLOSCAN_TRY(
      auto subarray_value,
      resolve_expected_override_via(
          pinned.expect_subarray, subarray, pinned.subarray,
          sticky_subarray_, pinned.sticky_subarray,
          [live_subarray_bound](const Tensor& value) {
            return read_host_config_vector<int64_t>(
                value, DLDataType{kDLInt, 64U, 1U},
                live_subarray_bound, "subarray");
          }));
  try {
    geometry.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const int64_t min_size = pinned.min_subarray_size;
  if (num_channels != geometry.num_elements()) {
    return fail_invalid(
        "SubarraySelectionOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  if (maybe_valid.has_value()) {
    // The mask is forwarded verbatim, so only its transport shape matters
    // here; slot contents are the consuming estimator's business.
    const DLDataType vdtype = maybe_valid->dtype();
    const bool mask_dtype_ok =
        vdtype.lanes == 1 && vdtype.bits == 8 &&
        (vdtype.code == kDLBool || vdtype.code == kDLUInt);
    if (!mask_dtype_ok) {
      return fail_invalid(
          "SubarraySelectionOp: is_valid must have dtype bool or uint8");
    }
    const auto& vshape = maybe_valid->shape();
    if (vshape.size() != 2 || vshape[0] != num_items ||
        vshape[1] != num_snapshots) {
      return fail_invalid(
          "SubarraySelectionOp: is_valid must have shape (num_items, "
          "num_snapshots) matching snapshot");
    }
    try {
      require_cuda_accessible(*maybe_valid, "is_valid", kOpName);
      require_contiguous(*maybe_valid, "is_valid", kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "SubarraySelectionOp: no CUDA compute stream"});
  }

  // (Re)build the selection + upload the channel-index table only when the
  // geometry / selection config changes.
  std::ostringstream ks;
  ks << geometry.wavelength << '|';
  for (double v : geometry.element_positions) {
    ks << v << ',';
  }
  ks << '|';
  for (int64_t v : subarray_value) {
    ks << v << ',';
  }
  ks << '|' << min_size;
  const std::string key = ks.str();
  if (key != sel_key_) {
    try {
      selection_ =
          build_subarray_selection(geometry, subarray_value, min_size);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    sel_key_ = key;
    if (selection_.reduced) {
      d_channel_index_ = make_device_tensor({selection_.subarray_size},
                                            DLDataType{kDLInt, 32, 1});
      HOLOSCAN_TRY(auto idx_ptr,
                   writable_ptr(*d_channel_index_, "channel_index"));
      if (cudaMemcpyAsync(idx_ptr, selection_.channel_index.data(),
                          selection_.channel_index.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return fail_runtime(
            "SubarraySelectionOp: channel-index H2D failed");
      }
      HOLOSCAN_LOG_INFO(
          "SubarraySelectionOp: selected {}/{} channels (uniform sub-array)",
          selection_.subarray_size, num_channels);
    }
  }

  const ArrayGeometry& out_geometry =
      selection_.reduced ? selection_.reduced_geometry : geometry;
  const int64_t out_channels =
      selection_.reduced ? selection_.subarray_size : num_channels;
  const std::vector<int64_t> out_shape =
      shape.size() == 3
          ? std::vector<int64_t>{num_items, out_channels, num_snapshots}
          : std::vector<int64_t>{num_items, out_channels};

  const DLDataType cplx = complex_dtype<T>();
  const std::size_t snap_bytes = tensor_byte_span(out_shape, cplx);

  // Prepare every output loan before any emit so a late fill failure cannot
  // leave snapshot_out / array_geometry already published.
  // TensorOutputLoan is move-only — prepare via a local lambda, not a
  // default-constructed expected<>.
  auto prepare_snap = [&]() -> expected<TensorOutputLoan, Error> {
    if (!selection_.reduced) {
      return prepare_device_tensor(
          snapshot_out, context, out_shape, cplx,
          [&](void* dst, std::size_t nbytes) -> expected<void, Error> {
            if (nbytes != snap_bytes) {
              return fail_runtime(
                  "SubarraySelectionOp: snapshot pass-through size mismatch");
            }
            if (cudaMemcpyAsync(dst, snap.data(), nbytes,
                                cudaMemcpyDeviceToDevice, stream) !=
                cudaSuccess) {
              return fail_runtime(
                  "SubarraySelectionOp: snapshot pass-through D2D failed");
            }
            return {};
          });
    }
    return prepare_device_tensor(
        snapshot_out, context, out_shape, cplx,
        [&](void* dst, std::size_t) -> expected<void, Error> {
          try {
            run_spatial_smoothing_gather<T>(
                static_cast<const cuda::std::complex<T>*>(snap.data()),
                /*mask=*/nullptr, num_items, num_channels, num_snapshots,
                static_cast<const int32_t*>(d_channel_index_->data()),
                /*fb_flip=*/nullptr, /*num_subarrays=*/1, out_channels,
                /*forward_backward=*/false,
                static_cast<cuda::std::complex<T>*>(dst),
                /*ss_mask=*/nullptr, stream);
            check_cuda("subarray gather", kOpName);
          } catch (const std::runtime_error& e) {
            return fail_runtime(e.what());
          }
          return {};
        });
  };
  HOLOSCAN_TRY(auto snap_loan, prepare_snap());

  const std::vector<double> packed = pack_array_geometry(out_geometry);
  const std::vector<int64_t> geom_shape =
      array_geometry_tensor_shape(out_geometry);
  const DLDataType f64 = real_dtype<double>();
  HOLOSCAN_TRY(auto geom_loan,
               prepare_device_tensor(
                   array_geometry, context, geom_shape, f64,
                   [&](void* dst, std::size_t nbytes) -> expected<void, Error> {
                     if (nbytes != packed.size() * sizeof(double)) {
                       return fail_runtime(
                           "SubarraySelectionOp: array_geometry size mismatch");
                     }
                     if (cudaMemcpyAsync(dst, packed.data(), nbytes,
                                         cudaMemcpyHostToDevice, stream) !=
                         cudaSuccess) {
                       return fail_runtime(
                           "SubarraySelectionOp: array_geometry H2D failed");
                     }
                     return {};
                   }));

  std::optional<TensorOutputLoan> valid_loan;
  if (maybe_valid.has_value()) {
    const auto& valid = *maybe_valid;
    const auto& vshape = valid.shape();
    const DLDataType vdtype = valid.dtype();
    const std::size_t vbytes = tensor_byte_span(vshape, vdtype);
    HOLOSCAN_TRY(auto prepared,
                 prepare_device_tensor(
                     is_valid_out, context, vshape, vdtype,
                     [&](void* dst, std::size_t nbytes) -> expected<void, Error> {
                       if (nbytes != vbytes) {
                         return fail_runtime(
                             "SubarraySelectionOp: is_valid forward size "
                             "mismatch");
                       }
                       if (cudaMemcpyAsync(dst, valid.data(), nbytes,
                                           cudaMemcpyDeviceToDevice, stream) !=
                           cudaSuccess) {
                         return fail_runtime(
                             "SubarraySelectionOp: is_valid forward D2D failed");
                       }
                       return {};
                     }));
    valid_loan.emplace(std::move(prepared));
  }

  HOLOSCAN_TRY(
      emit_prepared_tensor(snapshot_out, context, std::move(snap_loan)));
  HOLOSCAN_TRY(
      emit_prepared_tensor(array_geometry, context, std::move(geom_loan)));
  if (valid_loan.has_value()) {
    HOLOSCAN_TRY(emit_prepared_tensor(is_valid_out, context,
                                      std::move(*valid_loan)));
    retained_is_valid_ = std::move(*maybe_valid);
  }

  retained_snapshot_ = std::move(snap);
  return {};
}

template class SubarraySelectionOp<float>;
template class SubarraySelectionOp<double>;

}  // namespace holoscan::radar
