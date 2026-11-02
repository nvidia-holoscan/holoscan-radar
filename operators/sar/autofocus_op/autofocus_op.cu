// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/autofocus_op/autofocus_op.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/autofocus_op/autofocus_stage.hpp"

namespace holoscan::radar {

namespace {

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

template <typename T>
const std::string& op_name() {
  static const std::string name = std::string("AutofocusOp<") + precision_name<T>() + ">";
  return name;
}

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

// Validated per tick, because all four are [live]: an override carries an
// application's value, not a reviewed one.
expected<void, Error> validate_live(const detail::AutofocusConfig& cfg, const std::string& who) {
  if (cfg.num_iterations < 1) {
    return fail_invalid(who + ": num_iterations must be >= 1");
  }
  if (!(cfg.window_fraction > 0.0 && cfg.window_fraction <= 1.0)) {
    return fail_invalid(who + ": window_fraction must be in (0, 1]");
  }
  if (cfg.min_scatterers < 1) {
    return fail_invalid(who + ": min_scatterers must be >= 1");
  }
  // NaN is tested separately: it is not finite, so the positivity test below
  // short-circuits past it, and a NaN ceiling then makes every comparison
  // against it false -- the guard is silently never applied. Infinity is the
  // documented "uncapped" default and stays legal.
  if (std::isnan(cfg.max_correction)) {
    return fail_invalid(who + ": max_correction must not be NaN");
  }
  if (std::isfinite(cfg.max_correction) && cfg.max_correction <= 0.0) {
    return fail_invalid(who + ": max_correction must be positive when set");
  }
  return {};
}

}  // namespace

template <typename T>
AutofocusOp<T>::AutofocusOp(const MaxTensorExtents& extents) : extents_(extents) {
  const char* who = "AutofocusOp";
  const auto v = require_extent(extents.max_image_v, "max_image_v", who);
  const auto u = require_extent(extents.max_image_u, "max_image_u", who);
  max_image_byte_span_ = byte_span_complex_cube<T>({v, u}, who);
  // The azimuth axis is whichever of the two the caller names, so the phase
  // vector is bounded by the larger; sizing it from the named axis alone would
  // depend on a Param this constructor must not read.
  max_azimuth_ = v > u ? v : u;
  max_phase_byte_span_ = byte_span_real_cube<T>({max_azimuth_}, who);
  max_sharpness_byte_span_ = byte_span_real_cube<T>({kMaxAutofocusSharpnessSlots}, who);
  // termination and iterations are one int32 each, independent of precision:
  // they describe control flow, not samples.
  max_status_byte_span_ = byte_span_elems(1U, sizeof(std::int32_t), who);
}

template <typename T>
AutofocusOp<T>::~AutofocusOp() = default;

template <typename T>
void AutofocusOp<T>::start() {
  phase_scratch_ = detail::autofocus_malloc<T>(static_cast<size_t>(max_azimuth_), "phase scratch");
  sharpness_scratch_ = detail::autofocus_malloc<T>(static_cast<size_t>(kMaxAutofocusSharpnessSlots),
                                                   "sharpness scratch");
}

template <typename T>
void AutofocusOp<T>::stop() {
  cudaFree(phase_scratch_);
  cudaFree(sharpness_scratch_);
  phase_scratch_ = nullptr;
  sharpness_scratch_ = nullptr;
  state_.reset();
}

template <typename T>
void AutofocusOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  spec.input(image).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(num_iterations).optional().queue_depth(2U);
  spec.input(window_fraction).optional().queue_depth(2U);
  spec.input(min_scatterers).optional().queue_depth(2U);
  spec.input(max_correction).optional().queue_depth(2U);

  spec.output(image_out).produces_tensor(pooled_tensor_output(device, 2U, max_image_byte_span_));
  spec.output(phase_correction)
      .produces_tensor(pooled_tensor_output(device, 1U, max_phase_byte_span_));
  spec.output(sharpness).produces_tensor(
      pooled_tensor_output(device, 1U, max_sharpness_byte_span_));
  spec.output(termination).produces_tensor(pooled_tensor_output(device, 1U, max_status_byte_span_));
  spec.output(iterations).produces_tensor(pooled_tensor_output(device, 1U, max_status_byte_span_));
}

template <typename T>
Contract AutofocusOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{image});
  // Declared unconditionally: wiring the port is what enables the override,
  // and an unwired port simply never matches.
  add_live_override_to_contract(result, num_iterations);
  add_live_override_to_contract(result, window_fraction);
  add_live_override_to_contract(result, min_scatterers);
  add_live_override_to_contract(result, max_correction);
  return result;
}

template <typename T>
expected<void, Error> AutofocusOp<T>::compute(ExecutionContext& context) {
  const std::string& who = op_name<T>();
  const auto& params = context.params<AutofocusParams>();

  if (params.algorithm != AutofocusAlgorithm::kPga) {
    return fail_invalid(who + ": algorithm must be pga");
  }
  // Static and required: an unnamed axis is the one failure this contract
  // cannot detect from the tensor, so it is refused rather than guessed.
  if (params.azimuth_axis != AutofocusAzimuthAxis::kU &&
      params.azimuth_axis != AutofocusAzimuthAxis::kV) {
    return fail_invalid(who + ": azimuth_axis is required (u or v)");
  }

  detail::AutofocusConfig cfg;
  cfg.azimuth_axis = params.azimuth_axis;
  HOLOSCAN_TRY(cfg.num_iterations, resolve_live_override(num_iterations, params.num_iterations));
  HOLOSCAN_TRY(cfg.window_fraction, resolve_live_override(window_fraction, params.window_fraction));
  HOLOSCAN_TRY(cfg.min_scatterers, resolve_live_override(min_scatterers, params.min_scatterers));
  HOLOSCAN_TRY(cfg.max_correction, resolve_live_override(max_correction, params.max_correction));
  HOLOSCAN_TRY(validate_live(cfg, who));

  HOLOSCAN_TRY(auto formed, receive_tensor(image, who));
  try {
    require_cuda_accessible(formed, "image", who);
    require_contiguous(formed, "image", who);
    require_complex<T>(formed, "image");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const auto shape = formed.shape();
  if (shape.size() != 2U || shape[0] <= 0 || shape[1] <= 0 || shape[0] > *extents_.max_image_v ||
      shape[1] > *extents_.max_image_u) {
    return fail_invalid(who +
                        ": expected image (num_v, num_u) within MaxTensorExtents max_image_v / "
                        "max_image_u");
  }
  const bool az_u = cfg.azimuth_axis == AutofocusAzimuthAxis::kU;
  const int64_t n_az = az_u ? shape[1] : shape[0];
  // Bound before arithmetic: `num_iterations` is `[live]`, so an override of
  // INT64_MAX makes the `+ 1` signed-overflow UB before the check can run.
  if (cfg.num_iterations > kMaxAutofocusIterations) {
    return fail_invalid(who + ": num_iterations exceeds the sharpness pool bound (" +
                        std::to_string(kMaxAutofocusIterations) + ")");
  }
  const int64_t n_slots = cfg.num_iterations + 1;

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported, who + ": no CUDA compute stream"});
  }

  if (!state_) {
    state_ = std::make_unique<detail::AutofocusState<T>>();
  }

  // The run happens inside the image fill, writing the two vectors to the
  // op-owned scratch on the way past; the optional ports then copy from it.
  detail::AutofocusOutcome outcome;
  HOLOSCAN_TRY(emit_required_tensor(image_out, context, "AutofocusOp.image_out", false, [&] {
    return prepare_device_tensor(
        image_out, context, shape, formed.dtype(),
        [&](void* destination, std::size_t bytes) -> expected<void, Error> {
          if (bytes != formed.nbytes())
            return make_unexpected(Error{ErrorCode::kFailure, who + ": image_out size mismatch"});
          try {
            detail::run_autofocus<T>(cfg, *state_, formed.data(), 1, shape[0], shape[1],
                                     destination, phase_scratch_, sharpness_scratch_, &outcome,
                                     stream, who);
          } catch (const std::exception& e) {
            return make_unexpected(Error{ErrorCode::kFailure, e.what()});
          }
          return {};
        });
  }));

  const auto copy_out = [&](Output<Tensor>& port, LatchedOutput& latch, const T* source,
                            std::int64_t length, const char* label) -> expected<void, Error> {
    if (!latch.active()) {
      return {};
    }
    auto loan = prepare_device_tensor(
        port, context, std::vector<int64_t>{length}, real_dtype<T>(),
        [&](void* destination, std::size_t bytes) -> expected<void, Error> {
          // `bytes` is also the copy length, so a pool slot larger than the
          // tensor would read past the end of the scratch rather than
          // short-copy into it.
          if (bytes != static_cast<std::size_t>(length) * sizeof(T)) {
            return make_unexpected(
                Error{ErrorCode::kFailure, who + ": " + label + " size mismatch"});
          }
          if (cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToDevice, stream) !=
              cudaSuccess) {
            return make_unexpected(
                Error{ErrorCode::kFailure, who + ": " + label + " device copy failed"});
          }
          return {};
        });
    // Propagate rather than drop. Letting a failed loan fall through here
    // emits `image_out` and silently omits this port, which is exactly the
    // partially-emitted output set the prepare-then-emit split exists to
    // prevent -- and `LatchedOutput` only latches on *emit* failure, so a
    // pool exhaustion or a size mismatch would recur every tick unseen.
    if (!loan) {
      return make_unexpected(loan.error());
    }
    latch.emit(port, context, std::move(*loan), label);
    return {};
  };
  HOLOSCAN_TRY(copy_out(phase_correction, phase_correction_latch_, phase_scratch_, n_az,
                        "AutofocusOp.phase_correction"));
  HOLOSCAN_TRY(
      copy_out(sharpness, sharpness_latch_, sharpness_scratch_, n_slots, "AutofocusOp.sharpness"));

  // Provenance: what the run did, never whether it was right. Host-to-device
  // because the value was decided on the host; one int32 on the compute stream,
  // so it stays ordered behind the image without a sync.
  const auto emit_status = [&](Output<Tensor>& port, LatchedOutput& latch, std::int32_t& staging,
                               std::int32_t value, const char* label) -> expected<void, Error> {
    if (!latch.active()) {
      return {};
    }
    // Each port stages into its own member so the value outlives this call:
    // the emit is ordered on the compute stream, and the staging slot has to
    // stay valid and unmodified until that copy runs.
    staging = value;
    auto loan = prepare_device_tensor(
        port, context, std::vector<int64_t>{1}, DLDataType{kDLInt, 32, 1},
        [&](void* destination, std::size_t bytes) -> expected<void, Error> {
          // `staging` is a single int32, and `bytes` is the copy length.
          if (bytes != sizeof(std::int32_t)) {
            return make_unexpected(
                Error{ErrorCode::kFailure, who + ": " + label + " size mismatch"});
          }
          if (cudaMemcpyAsync(destination, &staging, bytes, cudaMemcpyHostToDevice, stream) !=
              cudaSuccess) {
            return make_unexpected(
                Error{ErrorCode::kFailure, who + ": " + label + " device copy failed"});
          }
          return {};
        });
    if (!loan) {
      return make_unexpected(loan.error());
    }
    latch.emit(port, context, std::move(*loan), label);
    return {};
  };
  HOLOSCAN_TRY(emit_status(termination, termination_latch_, termination_staging_,
                           static_cast<std::int32_t>(outcome.termination),
                           "AutofocusOp.termination"));
  HOLOSCAN_TRY(emit_status(iterations, iterations_latch_, iterations_staging_,
                           static_cast<std::int32_t>(outcome.iterations_accepted),
                           "AutofocusOp.iterations"));

  retained_input_ = std::move(formed);
  return {};
}

template class AutofocusOp<float>;
template class AutofocusOp<double>;

}  // namespace holoscan::radar
