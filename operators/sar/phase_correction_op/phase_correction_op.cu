// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/phase_correction_op/phase_correction_op.hpp"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/core/expected_macros.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/phase_correction_op/phase_correction_stage.hpp"

namespace holoscan::radar {

namespace {
constexpr const char* kOpName = "PhaseCorrectionOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}
}  // namespace

template <typename T>
PhaseCorrectionOp<T>::PhaseCorrectionOp(const MaxTensorExtents& extents) : extents_(extents) {
  const auto pulses = require_extent(extents.max_pulses, "max_pulses", kOpName);
  const auto bins = require_extent(extents.max_range_bins, "max_range_bins", kOpName);
  max_byte_span_ = byte_span_complex_cube<T>({pulses, bins}, kOpName);
}

template <typename T>
void PhaseCorrectionOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  spec.input(range_profiles_in).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(phase_correction).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.output(range_profiles_out).produces_tensor(pooled_tensor_output(device, 2U, max_byte_span_));
}

template <typename T>
Contract PhaseCorrectionOp<T>::contract() const {
  Contract result;
  result
      .trigger(OnAligned{.ports = ports(range_profiles_in, phase_correction),
                         .window = std::chrono::nanoseconds{0}})
      .input(range_profiles_in, Exact{})
      .input(phase_correction, Exact{});
  return result;
}

template <typename T>
expected<void, Error> PhaseCorrectionOp<T>::compute(ExecutionContext& context) {
  const auto sense = context.params<PhaseCorrectionParams>().sense;
  if (sense != PhaseCorrectionSense::kSubtract && sense != PhaseCorrectionSense::kAdd) {
    return fail_invalid(std::string(kOpName) + ": sense must be subtract or add");
  }
  // Exact-join profiles and their per-pulse phase for this tick.
  HOLOSCAN_TRY(auto profiles, receive_tensor(range_profiles_in, kOpName));
  HOLOSCAN_TRY(auto phase, receive_tensor(phase_correction, kOpName));
  try {
    require_cuda_accessible(profiles, "range_profiles", kOpName);
    require_cuda_accessible(phase, "phase_correction", kOpName);
    require_contiguous(profiles, "range_profiles", kOpName);
    require_contiguous(phase, "phase_correction", kOpName);
    require_complex<T>(profiles, "range_profiles");
    require_dtype(phase, real_dtype<T>(), "phase_correction", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const auto shape = profiles.shape();
  const auto phase_shape = phase.shape();
  if (shape.size() != 2U || phase_shape.size() != 1U || shape[0] != phase_shape[0] ||
      shape[0] <= 0 || shape[1] <= 0 || shape[0] > *extents_.max_pulses ||
      shape[1] > *extents_.max_range_bins) {
    return fail_invalid(std::string(kOpName) +
                        ": expected range_profiles (num_pulses, num_range_bins) and "
                        "phase_correction (num_pulses) within MaxTensorExtents");
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, std::string(kOpName) + ": no CUDA compute stream"});
  }
  // Apply the declared add/subtract sign across every range bin of each pulse.
  HOLOSCAN_TRY(emit_device_tensor(range_profiles_out, context, shape, profiles.dtype(),
                                  [&](void* destination, std::size_t) -> expected<void, Error> {
                                    try {
                                      detail::run_phase_correction<T>(
                                          sense, profiles.data(), phase.data(), 1, shape[0],
                                          shape[1], destination, stream, kOpName);
                                    } catch (const std::exception& e) {
                                      return make_unexpected(Error{ErrorCode::kFailure, e.what()});
                                    }
                                    return {};
                                  }));
  retained_profiles_ = std::move(profiles);
  retained_phase_ = std::move(phase);
  return {};
}

template class PhaseCorrectionOp<float>;
template class PhaseCorrectionOp<double>;

}  // namespace holoscan::radar
