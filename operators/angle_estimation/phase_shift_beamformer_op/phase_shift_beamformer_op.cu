// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_op.hpp"

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

#include "angle_estimation/phase_shift_beamformer_op/phase_shift_beamformer_stage.hpp"
#include "radar_common/dsp/steering_weights.hpp"
#include "radar_common/dsp/steering_weights_device.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/cuda_check.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

constexpr const char* kOpName = "PhaseShiftBeamformerOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<void, Error> fail_runtime(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kFailure, message});
}

}  // namespace

template <typename T>
void PhaseShiftBeamformerOp<T>::setup(OperatorSpec& spec) {
  // Declare device storage only; dtype is enforced in compute() so one
  // operator binary still serves F32/F64 and the harness can feed either.
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,  // rank-agnostic batching
  };
  spec.input(rd_maps).queue_depth(2U).expects_tensor(kDevice);
  spec.output(rda_cube)
      .produces_tensor(kDevice)
      .tensor_allocation(TensorAllocationBounds{
          .max_rank = kPhaseShiftBeamformerMaxRank,
          .max_byte_span = kPhaseShiftBeamformerMaxByteSpan});
  // [EA HSDK5 ONLY] No [live] Input ports — ArrayGeometry / WindowSpec /
  // unbounded az/el vectors are not PublishedPayload (HSDK5_PORT.md §9).
  // Selection is via Params (set_params); sticky live Inputs deferred.
}

template <typename T>
Contract PhaseShiftBeamformerOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  return result;
}

template <typename T>
expected<void, Error> PhaseShiftBeamformerOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto rd, receive_tensor(rd_maps, kOpName));

  const PhaseShiftBeamformerParams& pinned =
      context.params<PhaseShiftBeamformerParams>();
  // Params-only effective values (live Inputs deferred — OPEN_ITEMS / §9).
  ArrayGeometry geometry = pinned.array_geometry;
  const AngleConvention& convention = pinned.angle_convention;
  const std::vector<double>& az = pinned.az_angles;
  const std::vector<double>& el = pinned.el_angles;
  const WindowSpec& taper = pinned.array_taper;

  try {
    geometry.validate();
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  if (az.empty() || el.empty()) {
    return fail_invalid(
        "PhaseShiftBeamformerOp: az_angles and el_angles must be non-empty");
  }

  try {
    classify_input_dtype<T>(rd, "rd_maps", /*allow_real=*/false);
    require_cuda_accessible(rd, "rd_maps", kOpName);
    require_contiguous(rd, "rd_maps", kOpName);
    require_dtype(rd, complex_dtype<T>(), "rd_maps", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto& shape = rd.shape();
  const int64_t rank = static_cast<int64_t>(shape.size());
  if (rank < 3) {
    return fail_invalid(
        "PhaseShiftBeamformerOp: rd_maps must have shape "
        "(..., num_channels, num_doppler, num_range_bins) with rank >= 3");
  }
  const int64_t num_channels = shape[static_cast<size_t>(rank - 3)];
  const int64_t num_doppler = shape[static_cast<size_t>(rank - 2)];
  const int64_t num_range = shape[static_cast<size_t>(rank - 1)];
  int64_t batch = 1;
  for (int64_t i = 0; i < rank - 3; ++i) {
    batch *= shape[static_cast<size_t>(i)];
  }
  const int64_t num_cells = num_doppler * num_range;

  if (num_channels != geometry.num_elements()) {
    return fail_invalid(
        "PhaseShiftBeamformerOp: rd_maps num_channels must match "
        "array_geometry.num_elements");
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "PhaseShiftBeamformerOp: no CUDA compute stream"});
  }

  const std::string cache_key =
      steering_cache_key(geometry, convention, az, el, taper);
  // Conjugated weights, built on device, az-major beams in a
  // (num_beams, num_channels) layout — drops straight into the GEMM.
  // warn_on_build stays false while live Inputs are deferred (Params-only);
  // cache still dedupes identical keys across ticks.
  const std::shared_ptr<holoscan::Tensor>& weights =
      steering_cache_.get_or_build(
          cache_key, /*warn_on_build=*/false, "steering weights", [&] {
            return build_steering_weights_device<T>(
                geometry, convention, az, el, taper, BeamOrder::kAzMajor,
                SteeringLayout::kBeamMajor, /*conjugate=*/true, stream);
          });
  if (!weights) {
    return fail_runtime(
        "PhaseShiftBeamformerOp: internal error: steering weights missing");
  }

  const int64_t num_az = static_cast<int64_t>(az.size());
  const int64_t num_el = static_cast<int64_t>(el.size());
  const int64_t num_beams = num_az * num_el;

  // Output cube: original leading (batch) dims, then the beam grid, then the
  // range-Doppler plane. The contiguous buffer aliases [batch, num_beams,
  // num_cells] for the GEMM.
  std::vector<int64_t> out_shape;
  out_shape.reserve(static_cast<size_t>(rank + 1));
  for (int64_t i = 0; i < rank - 3; ++i) {
    out_shape.push_back(shape[static_cast<size_t>(i)]);
  }
  out_shape.push_back(num_az);
  out_shape.push_back(num_el);
  out_shape.push_back(num_doppler);
  out_shape.push_back(num_range);
  const DLDataType dtype = complex_dtype<T>();

  const void* input_ptr = rd.data();
  const void* weights_ptr = weights->data();
  HOLOSCAN_TRY(emit_device_tensor(
      rda_cube, context, out_shape, dtype,
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          detail::run_phase_shift_beamform_stage<T>(
              static_cast<const cuda::std::complex<T>*>(input_ptr),
              static_cast<cuda::std::complex<T>*>(out_ptr), batch,
              num_channels, num_cells,
              static_cast<const cuda::std::complex<T>*>(weights_ptr),
              num_beams, stream);
          check_cuda("phase-shift beamform stage", kOpName);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        } catch (const std::runtime_error& e) {
          return fail_runtime(e.what());
        }
        return {};
      }));

  retained_input_ = std::move(rd);
  return {};
}

template class PhaseShiftBeamformerOp<float>;
template class PhaseShiftBeamformerOp<double>;

}  // namespace holoscan::radar
