// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/clutter_filter_op/clutter_filter_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/clutter_filter_op/clutter_stage.hpp"
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

}  // namespace

template <typename T>
ClutterFilterOp<T>::ClutterFilterOp(const MaxTensorExtents& extents) {
  const char* who = "ClutterFilterOp";
  const std::int64_t max_chirps =
      require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_range_bins =
      require_extent(extents.max_range_bins, "max_range_bins", who);
  std::int64_t max_tx = 1;
  std::int64_t max_rx = 1;
  require_tx_rx_pool_extents(extents, who, &max_tx, &max_rx);
  extents_ = extents;
  max_rank_ =
      static_cast<std::uint32_t>(2U + named_tx_rx_leading_rank(extents_));
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_tx, max_rx, max_chirps, max_range_bins}, who);
}

template <typename T>
void ClutterFilterOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  // layout.rank stays unset (exact-rank optional); max_rank on allocation
  // admits the named-axis ceiling from extents. Opaque leading batch is
  // rejected in compute().
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  spec.input(range_profiles).queue_depth(2U).expects_tensor(kDevice);
  spec.output(range_profiles_out)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = max_rank_,
          .max_byte_span = max_byte_span_});
  // [live] method override port. Wired graphs may push ClutterFilterMethod
  // samples; the operator caches the last value (sticky). Leaving the port
  // unconnected is legal — compute() seeds from Params.
  spec.input(method).queue_depth(2U);
}

template <typename T>
Contract ClutterFilterOp<T>::contract() const {
  // Trigger on the data tensor only. The [live] method input is intentionally
  // omitted from the temporal contract so an unconnected override port does
  // not yield TEMPORAL_INPUT_UNCONNECTED, and so a second OnClock source is
  // not required (HSDK 5 EA multi-partition device binding is still rough —
  // sticky live-override coverage is exercised via Params + operator sticky
  // state when a sample does arrive).
  Contract result;
  result.trigger(OnEach{range_profiles});
  return result;
}

template <typename T>
expected<void, Error> ClutterFilterOp<T>::compute(ExecutionContext& context) {
  HOLOSCAN_TRY(auto input,
               receive_tensor(range_profiles, "ClutterFilterOp"));

  const ClutterFilterParams& pinned = context.params<ClutterFilterParams>();
  // Best-effort sticky override: if the port is wired and a sample is waiting,
  // absorb it; otherwise keep sticky / fall back to Params.
  HOLOSCAN_TRY(auto method_value,
               resolve_sticky_override(method, pinned.method, sticky_method_));
  if (method_value == ClutterFilterMethod::kNone) {
    return fail_invalid(
        "ClutterFilterOp: method 'none' is not supported by the standalone "
        "operator (it exists only for the fused operator's inline clutter "
        "stage); use one of {mean_subtraction, single_delay, double_delay}");
  }

  try {
    classify_input_dtype<T>(input, "range_profiles", /*allow_real=*/false);
    require_cuda_accessible(input, "range_profiles", "ClutterFilterOp");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  BatchedShape batched;
  try {
    batched = collapse_batch(input, "range_profiles");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const std::size_t max_leading = named_tx_rx_leading_rank(extents_);
  if (batched.leading.size() > max_leading) {
    return fail_invalid(unexpected_leading_message(
        "ClutterFilterOp", batched.leading.size(), max_leading));
  }
  const std::int64_t max_rx = extent_rx_or_channels(extents_);
  const std::int64_t max_tx = extent_or_one(extents_.max_tx);
  if (batched.batch > max_tx * max_rx) {
    return fail_invalid(
        "ClutterFilterOp: leading product " + std::to_string(batched.batch) +
        " exceeds max_tx*max_rx from MaxTensorExtents");
  }
  if (batched.rows > *extents_.max_chirps ||
      batched.cols > *extents_.max_range_bins) {
    return fail_invalid(
        "ClutterFilterOp: input chirps/range_bins exceed MaxTensorExtents");
  }

  const int64_t num_chirps = batched.rows;
  if (method_value == ClutterFilterMethod::kSingleDelay && num_chirps < 2) {
    return fail_invalid(
        "ClutterFilterOp: method 'single_delay' needs num_chirps >= 2, got " +
        std::to_string(num_chirps));
  }
  if (method_value == ClutterFilterMethod::kDoubleDelay && num_chirps < 3) {
    return fail_invalid(
        "ClutterFilterOp: method 'double_delay' needs num_chirps >= 3, got " +
        std::to_string(num_chirps));
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "ClutterFilterOp: no CUDA compute stream"});
  }

  const int64_t num_chirps_out =
      detail::clutter_num_chirps_out(method_value, num_chirps);
  const std::vector<int64_t> out_shape =
      full_rank_shape(batched, num_chirps_out, batched.cols);
  const DLDataType dtype = complex_dtype<T>();

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(emit_device_tensor(
      range_profiles_out, context, out_shape, dtype,
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          detail::run_clutter_stage<T>(method_value, input_ptr, batched.batch,
                                       num_chirps, batched.cols, out_ptr,
                                       stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      }));

  retained_input_ = std::move(input);
  return {};
}

template class ClutterFilterOp<float>;
template class ClutterFilterOp<double>;

}  // namespace holoscan::radar
