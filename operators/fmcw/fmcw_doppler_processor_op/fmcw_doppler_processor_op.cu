// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "fmcw/fmcw_doppler_processor_op/doppler_stage.hpp"
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

constexpr const char* kOpName = "FMCWDopplerProcessorOp";

// Effective per-tick values own converted schema payloads.
struct EffectiveDopplerConfig {
  WindowSpec window;
  bool center_zero_doppler;
  FMCWWaveformDescriptor waveform;
  FftNorm fft_norm;
  int64_t nfft;
};

// Validated stage configuration plus the output-axis decisions derived from
// the current input shape.
struct ValidatedDopplerConfig {
  detail::DopplerStageConfig stage;
  int64_t num_doppler;
  bool waveform_present;
};

// Resolve Exact inputs with the pinned parameters. Unwired ports
// (expect=false) must not be received.
expected<EffectiveDopplerConfig, Error> resolve_overrides(
    Input<wire::WindowSpecT>& doppler_window_in,
    Input<bool>& center_zero_doppler_in,
    Input<wire::FMCWWaveformT>& waveform_in, Input<FftNorm>& fft_norm_in,
    Input<int64_t>& nfft_in, const FMCWDopplerProcessorParams& pinned,
    std::optional<WindowSpec>& sticky_doppler_window,
    std::optional<bool>& sticky_center_zero_doppler,
    std::optional<FMCWWaveformDescriptor>& sticky_waveform,
    std::optional<FftNorm>& sticky_fft_norm,
    std::optional<int64_t>& sticky_nfft) {
  HOLOSCAN_TRY(
      auto doppler_window,
      resolve_expected_override_via(
          pinned.expect_doppler_window, doppler_window_in,
          pinned.doppler_window, sticky_doppler_window,
          pinned.sticky_doppler_window,
          [](const wire::WindowSpecT& value) { return from_wire(value); }));
  HOLOSCAN_TRY(auto center_zero_doppler,
               resolve_expected_override(
                   pinned.expect_center_zero_doppler, center_zero_doppler_in,
                   pinned.center_zero_doppler, sticky_center_zero_doppler,
                   pinned.sticky_center_zero_doppler));
  HOLOSCAN_TRY(
      auto waveform,
      resolve_expected_override_via(
          pinned.expect_waveform, waveform_in, pinned.waveform,
          sticky_waveform, pinned.sticky_waveform,
          [](const wire::FMCWWaveformT& value) { return from_wire(value); }));
  HOLOSCAN_TRY(auto fft_norm,
               resolve_expected_override(pinned.expect_fft_norm, fft_norm_in,
                                         pinned.fft_norm, sticky_fft_norm,
                                         pinned.sticky_fft_norm));
  HOLOSCAN_TRY(auto nfft,
               resolve_expected_override(pinned.expect_nfft, nfft_in,
                                         pinned.nfft, sticky_nfft,
                                         pinned.sticky_nfft));

  return EffectiveDopplerConfig{
      .window = std::move(doppler_window),
      .center_zero_doppler = center_zero_doppler,
      .waveform = std::move(waveform),
      .fft_norm = fft_norm,
      .nfft = nfft};
}

// Collapse the named leading axes for MatX and enforce their extent bounds.
expected<BatchedShape, Error> resolve_batched_shape(
    const Tensor& input, const MaxTensorExtents& extents) {
  BatchedShape batched;
  try {
    batched = collapse_batch(input, "range_profiles");
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }

  const std::size_t max_leading = named_tx_rx_leading_rank(extents);
  if (batched.leading.size() > max_leading) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        unexpected_leading_message(kOpName, batched.leading.size(),
                                   max_leading)});
  }
  const std::int64_t max_rx = extent_rx_or_channels(extents);
  const std::int64_t max_tx = extent_or_one(extents.max_tx);
  if (batched.batch > max_tx * max_rx) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": leading product " +
            std::to_string(batched.batch) +
            " exceeds max_tx*max_rx from MaxTensorExtents"});
  }
  return batched;
}

// Validate configuration-dependent constraints and build the stage config.
expected<ValidatedDopplerConfig, Error> validate_configuration(
    const EffectiveDopplerConfig& effective, const BatchedShape& batched) {
  std::optional<int64_t> opt_nfft;
  if (effective.nfft != 0) {
    opt_nfft = effective.nfft;
  }

  if (opt_nfft.has_value() && *opt_nfft < batched.rows) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": nfft (" + std::to_string(*opt_nfft) +
            ") must be >= num_chirps (" + std::to_string(batched.rows) +
            ") on input port 'range_profiles'"});
  }

  const bool waveform_present = effective.waveform.is_valid();
  std::optional<FMCWWaveformDescriptor> opt_waveform;
  if (waveform_present) {
    opt_waveform = effective.waveform;
  }

  detail::DopplerStageConfig stage;
  stage.window = effective.window;
  stage.center_zero_doppler = effective.center_zero_doppler;
  stage.waveform = std::move(opt_waveform);
  stage.fft_norm = effective.fft_norm;
  stage.nfft = opt_nfft;
  const int64_t num_doppler =
      detail::doppler_num_bins(batched.rows, stage.nfft);

  return ValidatedDopplerConfig{
      .stage = std::move(stage),
      .num_doppler = num_doppler,
      .waveform_present = waveform_present};
}

}  // namespace

template <typename T>
FMCWDopplerProcessorOp<T>::FMCWDopplerProcessorOp(
    const MaxTensorExtents& extents)
    : state_(std::make_shared<detail::DopplerStageState<T>>()) {  // NOLINT
  const char* who = "FMCWDopplerProcessorOp";
  const std::int64_t max_chirps =
      require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_range_bins =
      require_extent(
          extents.max_range_bins, "max_range_bins", who);
  const std::int64_t max_rx =
      extent_rx_or_channels(extents);
  const std::int64_t max_tx =
      extent_or_one(extents.max_tx);
  if (max_rx <= 0 || max_tx <= 0) {
    throw std::invalid_argument(
        "FMCWDopplerProcessorOp: max_rx/max_tx extents "
        "must be positive when set");
  }
  extents_ = extents;
  max_rank_ = static_cast<std::uint32_t>(
      2U + named_tx_rx_leading_rank(extents_));
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_tx, max_rx, max_chirps, max_range_bins}, who);
  velocity_bins_byte_span_ =
      byte_span_real_cube<T>({max_chirps}, who);
}

template <typename T>
FMCWDopplerProcessorOp<T>::~FMCWDopplerProcessorOp() = default;

template <typename T>
void FMCWDopplerProcessorOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds fat_bounds{
      .max_rank = max_rank_,
      .max_byte_span = max_byte_span_};
  const TensorBounds vel_bounds{
      .max_rank = 1U,
      .max_byte_span = velocity_bins_byte_span_};
  spec.input(range_profiles).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(doppler_window_in).queue_depth(2U);
  spec.input(waveform_in).queue_depth(2U);
  spec.output(rd_maps)
      .produces_tensor(pooled_tensor_output(kDevice, fat_bounds));
  spec.output(velocity_bins)
      .produces_tensor(pooled_tensor_output(kDevice, vel_bounds));

  // Sticky [live] POD override ports.
  spec.input(center_zero_doppler_in).queue_depth(2U);
  spec.input(fft_norm_in).queue_depth(2U);
  spec.input(nfft_in).queue_depth(2U);
}

template <typename T>
Contract FMCWDopplerProcessorOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{range_profiles});
  maybe_add_live_override_to_contract(
      result, doppler_window_in, params().expect_doppler_window);
  maybe_add_live_override_to_contract(
      result, center_zero_doppler_in, params().expect_center_zero_doppler);
  maybe_add_live_override_to_contract(
      result, waveform_in, params().expect_waveform);
  maybe_add_live_override_to_contract(
      result, fft_norm_in, params().expect_fft_norm);
  maybe_add_live_override_to_contract(
      result, nfft_in, params().expect_nfft);
  return result;
}

template <typename T>
expected<void, Error> FMCWDopplerProcessorOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto input, receive_tensor(range_profiles, kOpName));

  try {
    require_cuda_accessible(input, "range_profiles", kOpName);
    require_contiguous(input, "range_profiles", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const FMCWDopplerProcessorParams& pinned =
      context.params<FMCWDopplerProcessorParams>();

  HOLOSCAN_TRY(auto effective,
               resolve_overrides(
                   doppler_window_in, center_zero_doppler_in, waveform_in,
                   fft_norm_in, nfft_in, pinned, sticky_doppler_window_,
                   sticky_center_zero_doppler_, sticky_waveform_,
                   sticky_fft_norm_, sticky_nfft_));

  try {
    classify_input_dtype<T>(input, "range_profiles", /*allow_real=*/false);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  HOLOSCAN_TRY(auto batched, resolve_batched_shape(input, extents_));

  HOLOSCAN_TRY(auto validated,
               validate_configuration(effective, batched));

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    const std::string& msg = std::string(kOpName) + ": no CUDA compute stream";
    return make_unexpected(Error{ErrorCode::kNotSupported, msg});
  }

  // Conditional velocity_bins: emitted only when waveform is effective.
  // Enqueued before the Doppler FFT: its host-staged upload waits on the
  // stream, so this ordering keeps that wait off the FFT hot path.
  std::optional<TensorOutputLoan> velocity_loan;
  if (validated.waveform_present) {
    HOLOSCAN_TRY(auto loan,
                 prepare_device_tensor(
                     velocity_bins, context,
                     std::vector<int64_t>{validated.num_doppler},
                     real_dtype<T>(),
                     [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_velocity_bins<T>(
                             validated.stage, validated.num_doppler,
                             static_cast<T*>(out_ptr),
                             stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       }
                       return {};
                     }));
    velocity_loan.emplace(std::move(loan));
  }

  // Run the Doppler stage.
  const std::vector<int64_t> out_shape =
      full_rank_shape(batched, validated.num_doppler, batched.cols);
  const DLDataType dtype = complex_dtype<T>();

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(auto rd_loan,
               prepare_device_tensor(
                   rd_maps, context, out_shape, dtype,
                   [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                     try {
                       detail::run_doppler_stage<T>(
                           validated.stage, *state_, input_ptr, batched.batch,
                           batched.rows, batched.cols, out_ptr,
                           validated.num_doppler, stream);
                     } catch (const std::invalid_argument& e) {
                       return fail_invalid(e.what());
                     }
                     return {};
                   }));

  // Emit all prepared loans together.
  if (velocity_loan.has_value()) {
    HOLOSCAN_TRY(emit_prepared_tensor(velocity_bins, context,
                                      std::move(*velocity_loan)));
  }
  HOLOSCAN_TRY(emit_prepared_tensor(rd_maps, context, std::move(rd_loan)));

  retained_input_ = std::move(input);
  return {};
}

template class FMCWDopplerProcessorOp<float>;
template class FMCWDopplerProcessorOp<double>;

}  // namespace holoscan::radar
