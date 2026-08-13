// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "angle_estimation/beamscan_estimator_op/beamscan_stage.hpp"
#include "angle_estimation/beamscan_estimator_op/is_valid_mask.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "BeamscanEstimatorOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

template <typename T>
std::vector<T> cast_vector(const std::vector<double>& in) {
  std::vector<T> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = static_cast<T>(in[i]);
  }
  return out;
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
void BeamscanEstimatorOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorAllocationBounds bounds{
      .max_rank = kBeamscanEstimatorMaxRank,
      .max_byte_span = kBeamscanEstimatorMaxByteSpan};
  spec.input(snapshot).queue_depth(2U).expects_tensor(kDevice);
  // Optional is_valid: SafeDefault / omitted from contract so unconnected
  // graphs compile and activate.
  spec.input(is_valid).queue_depth(2U).expects_tensor(kDevice);
  spec.output(aoa_estimates)
      .produces_tensor(kDevice)
      .tensor_allocation(bounds);
  spec.output(angle_spectrum)
      .produces_tensor(kDevice)
      .tensor_allocation(bounds);
  // [live] peak_interpolation override. Leaving the port unconnected is
  // legal — compute() seeds from Params.
  spec.input(peak_interpolation).queue_depth(2U);
}

template <typename T>
Contract BeamscanEstimatorOp<T>::contract() const {
  // Trigger on the data tensor only. Optional is_valid and [live]
  // peak_interpolation are omitted from the temporal contract so unconnected
  // ports do not yield TEMPORAL_INPUT_UNCONNECTED (SafeDefault alone is not
  // enough on EA — same pattern as ClutterFilter method). Wiring is_valid in
  // Graph tests is deferred (RUNTIME_OPERATOR_REPLAY_MISMATCH); see
  // OPEN_ITEMS.md / HSDK5_PORT.md §1.
  Contract result;
  result.trigger(OnEach{snapshot});
  return result;
}

template <typename T>
expected<void, Error> BeamscanEstimatorOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto snap, receive_tensor(snapshot, kOpName));

  const BeamscanParams& pinned = context.params<BeamscanParams>();
  HOLOSCAN_TRY(auto peak_interp,
               resolve_sticky_override(peak_interpolation,
                                       pinned.peak_interpolation,
                                       sticky_peak_interpolation_));

  // Descriptor / vector live params are Params-only on EA (HSDK5_PORT §9).
  ArrayGeometry geometry = pinned.array_geometry;
  try {
    geometry.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const AngleConvention& convention = pinned.angle_convention;
  const std::vector<double>& az = pinned.az_angles;
  const std::vector<double>& el = pinned.el_angles;
  const WindowSpec& taper = pinned.array_taper;

  if (az.empty() || el.empty()) {
    return fail_invalid(
        "BeamscanEstimatorOp: az_angles and el_angles must be non-empty");
  }

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
        "BeamscanEstimatorOp: snapshot must have shape "
        "(num_items, num_channels) or "
        "(num_items, num_channels, num_snapshots)");
  }
  const int64_t num_items = shape[0];
  const int64_t num_channels = shape[1];
  int64_t num_snapshots = 1;
  if (shape.size() == 3) {
    num_snapshots = shape[2];
    if (num_snapshots != 1) {
      return fail_invalid(
          "BeamscanEstimatorOp: snapshot num_snapshots must be 1 for "
          "single-snapshot Beamscan");
    }
  }
  if (num_channels != geometry.num_elements()) {
    return fail_invalid(
        "BeamscanEstimatorOp: snapshot num_channels must match "
        "array_geometry.num_elements");
  }

  HOLOSCAN_TRY(auto maybe_valid, receive_optional_value(is_valid));
  // A 2-D snapshot is the num_snapshots == 1 case, so the mask contract
  // (num_items, num_snapshots) applies to both ranks.
  if (maybe_valid.has_value()) {
    try {
      detail::check_is_valid_mask(*maybe_valid, num_items, num_snapshots,
                                  kOpName);
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    } catch (const std::runtime_error& e) {
      return fail_runtime(e.what());
    }
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 "BeamscanEstimatorOp: no CUDA compute stream"});
  }

  const int64_t num_az = static_cast<int64_t>(az.size());
  const int64_t num_el = static_cast<int64_t>(el.size());
  const int64_t num_beams = num_az * num_el;
  const DLDataType out_dtype = real_dtype<T>();
  const std::vector<int64_t> aoa_shape = {num_items, 2};
  const std::vector<int64_t> spec_shape = {num_items, num_az, num_el};

  if (num_items == 0) {
    HOLOSCAN_TRY(emit_device_tensor(
        aoa_estimates, context, aoa_shape, out_dtype,
        [](void*, std::size_t) -> expected<void, Error> { return {}; }));
    HOLOSCAN_TRY(emit_device_tensor(
        angle_spectrum, context, spec_shape, out_dtype,
        [](void*, std::size_t) -> expected<void, Error> { return {}; }));
    retained_snapshot_ = std::move(snap);
    if (maybe_valid.has_value()) {
      retained_is_valid_ = std::move(*maybe_valid);
    }
    return {};
  }

  const std::string cache_key =
      steering_cache_key(geometry, convention, az, el, taper);
  // Params-seeded builds are silent (warn_on_build=false); live descriptor
  // overrides that would rebuild weights are deferred (HSDK5_PORT §9).
  const std::shared_ptr<holoscan::Tensor>& weights =
      steering_cache_.get_or_build(
          cache_key, /*warn_on_build=*/false, "steering weights", [&] {
            return build_steering_weights_device<T>(
                geometry, convention, az, el, taper, BeamOrder::kElMajor,
                SteeringLayout::kChannelMajor, /*conjugate=*/true, stream);
          });
  if (!weights) {
    return fail_runtime(
        "BeamscanEstimatorOp: internal error: steering weights missing");
  }

  const auto az_host = cast_vector<T>(az);
  const auto el_host = cast_vector<T>(el);
  auto az_grid = make_device_tensor({num_az}, out_dtype);
  auto el_grid = make_device_tensor({num_el}, out_dtype);
  HOLOSCAN_TRY(auto az_ptr, writable_ptr(*az_grid, "az_grid"));
  HOLOSCAN_TRY(auto el_ptr, writable_ptr(*el_grid, "el_grid"));
  if (cudaMemcpyAsync(az_ptr, az_host.data(), az_host.size() * sizeof(T),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(el_ptr, el_host.data(), el_host.size() * sizeof(T),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return fail_runtime("BeamscanEstimatorOp: scan-grid H2D failed");
  }

  // Both outputs are filled by one stage — allocate both loans, write on the
  // runtime stream, then commit/emit.
  auto aoa_loan = aoa_estimates.allocate_tensor(
      TensorLoanRequest{.shape = aoa_shape, .dtype = out_dtype});
  if (!aoa_loan) {
    return make_unexpected(std::move(aoa_loan).error());
  }
  auto spec_loan = angle_spectrum.allocate_tensor(
      TensorLoanRequest{.shape = spec_shape, .dtype = out_dtype});
  if (!spec_loan) {
    return make_unexpected(std::move(spec_loan).error());
  }

  auto aoa_writer = aoa_loan->write(context.cuda_stream());
  if (!aoa_writer) {
    return make_unexpected(std::move(aoa_writer).error());
  }
  auto spec_writer = spec_loan->write(context.cuda_stream());
  if (!spec_writer) {
    return make_unexpected(std::move(spec_writer).error());
  }

  auto beamformed =
      make_device_tensor({num_items, num_beams}, complex_dtype<T>());
  HOLOSCAN_TRY(auto beam_ptr, writable_ptr(*beamformed, "beamformed"));

  try {
    detail::run_beamscan_stage<T>(
        static_cast<const cuda::std::complex<T>*>(snap.data()), num_items,
        num_channels,
        static_cast<const cuda::std::complex<T>*>(weights->data()),
        static_cast<cuda::std::complex<T>*>(beam_ptr), num_az, num_el,
        static_cast<const T*>(az_grid->data()),
        static_cast<const T*>(el_grid->data()), peak_interp,
        static_cast<T*>(spec_writer->data()),
        static_cast<T*>(aoa_writer->data()), stream);
    check_cuda("beamscan stage", kOpName);
  } catch (const std::runtime_error& e) {
    return fail_runtime(e.what());
  }

  auto aoa_committed = std::move(*aoa_writer).commit();
  if (!aoa_committed) {
    return make_unexpected(std::move(aoa_committed).error());
  }
  auto spec_committed = std::move(*spec_writer).commit();
  if (!spec_committed) {
    return make_unexpected(std::move(spec_committed).error());
  }

  auto aoa_emitted = aoa_estimates.emit(
      std::move(*aoa_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!aoa_emitted) {
    return make_unexpected(std::move(aoa_emitted).error());
  }
  auto spec_emitted = angle_spectrum.emit(
      std::move(*spec_loan),
      EmitOptions{.capture_time = context.activation_time()});
  if (!spec_emitted) {
    return make_unexpected(std::move(spec_emitted).error());
  }

  retained_snapshot_ = std::move(snap);
  if (maybe_valid.has_value()) {
    retained_is_valid_ = std::move(*maybe_valid);
  }
  return {};
}

template class BeamscanEstimatorOp<float>;
template class BeamscanEstimatorOp<double>;

}  // namespace holoscan::radar
