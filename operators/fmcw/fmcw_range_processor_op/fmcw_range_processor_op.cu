// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_range_processor_op/fmcw_range_processor_op.hpp"

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

#include "fmcw/fmcw_range_processor_op/range_stage.hpp"
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

constexpr const char* kOpName = "FMCWRangeProcessorOp";

// Effective per-tick values after applying sticky scalar overrides. Params-only
// descriptor values are borrowed from the operator parameter storage.
struct EffectiveRangeConfig {
  const WindowSpec* window;
  bool apply_dechirp;
  const FMCWWaveformDescriptor* waveform;
  FftNorm fft_norm;
  double propagation_speed;
  int64_t nfft;
};

// Validated stage configuration plus the output-axis decisions derived from
// the current input dtype and shape.
struct ValidatedRangeConfig {
  detail::RangeStageConfig stage;
  int64_t num_bins;
  bool waveform_present;
};

// Resolve sticky scalar inputs and combine them with the pinned parameters.
expected<EffectiveRangeConfig, Error> resolve_overrides(
    Input<bool>& apply_dechirp_in, Input<FftNorm>& fft_norm_in,
    Input<int64_t>& nfft_in, const FMCWRangeProcessorParams& pinned,
    std::optional<bool>& sticky_apply_dechirp,
    std::optional<FftNorm>& sticky_fft_norm,
    std::optional<int64_t>& sticky_nfft) {
  HOLOSCAN_TRY(auto apply_dechirp,
               resolve_sticky_override(apply_dechirp_in,
                                       pinned.apply_dechirp,
                                       sticky_apply_dechirp));
  HOLOSCAN_TRY(auto fft_norm,
               resolve_sticky_override(fft_norm_in, pinned.fft_norm,
                                       sticky_fft_norm));
  HOLOSCAN_TRY(auto nfft,
               resolve_sticky_override(nfft_in, pinned.nfft, sticky_nfft));

  return EffectiveRangeConfig{.window = &pinned.range_window,
                              .apply_dechirp = apply_dechirp,
                              .waveform = &pinned.waveform,
                              .fft_norm = fft_norm,
                              .propagation_speed = pinned.propagation_speed,
                              .nfft = nfft};
}

// Classify the input precision and return whether the input is complex-IF.
template <typename T>
expected<bool, Error> classify_complex_input(const Tensor& input) {
  try {
    const InputPrecision precision =
        classify_input_dtype<T>(input, "adc_cube", /*allow_real=*/true);
    return precision == InputPrecision::kComplex;
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }
}

// Collapse the named leading axes for MatX and enforce their extent bounds.
expected<BatchedShape, Error> resolve_batched_shape(
    const Tensor& input, const MaxTensorExtents& extents) {
  BatchedShape batched;
  try {
    batched = collapse_batch(input, "adc_cube");
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(e.what())});
  }

  const std::size_t max_leading = named_tx_rx_leading_rank(extents);
  if (batched.leading.size() > max_leading) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument,
                                 unexpected_leading_message(
                                     kOpName, batched.leading.size(),
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
expected<ValidatedRangeConfig, Error> validate_configuration(
    const EffectiveRangeConfig& effective, bool complex_input,
    const BatchedShape& batched) {
  std::optional<int64_t> opt_nfft;
  if (effective.nfft != 0) {
    opt_nfft = effective.nfft;
  }

  if (opt_nfft.has_value() && *opt_nfft < batched.cols) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": nfft (" + std::to_string(*opt_nfft) +
            ") must be >= samples_per_chirp (" +
            std::to_string(batched.cols) + ") on port adc_cube"});
  }
  if (effective.apply_dechirp) {
    if (!effective.waveform->is_valid()) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          std::string(kOpName) +
              ": apply_dechirp = true requires a waveform descriptor "
              "(dechirp reference source)"});
    }
    if (!complex_input) {
      return make_unexpected(Error{
          ErrorCode::kInvalidArgument,
          std::string(kOpName) +
              ": apply_dechirp = true requires complex-IF input on port "
              "adc_cube (received real-IF samples; software dechirp "
              "produces complex samples, inconsistent with the real-IF "
              "output size)"});
    }
  }

  const bool waveform_present = effective.waveform->is_valid();
  std::optional<FMCWWaveformDescriptor> opt_waveform;
  if (waveform_present) {
    opt_waveform = *effective.waveform;
  }

  detail::RangeStageConfig stage;
  stage.window = *effective.window;
  stage.apply_dechirp = effective.apply_dechirp;
  stage.waveform = std::move(opt_waveform);
  stage.fft_norm = effective.fft_norm;
  stage.propagation_speed = effective.propagation_speed;
  stage.nfft = opt_nfft;

  return ValidatedRangeConfig{
      .stage = std::move(stage),
      .num_bins = detail::range_num_bins(complex_input, batched.cols, opt_nfft),
      .waveform_present = waveform_present};
}

}  // namespace

template <typename T>
FMCWRangeProcessorOp<T>::FMCWRangeProcessorOp(
    const MaxTensorExtents& extents)
    : state_(std::make_shared<detail::RangeStageState<T>>()) {
  const char* who = "FMCWRangeProcessorOp";
  const std::int64_t max_chirps =
      require_extent(extents.max_chirps, "max_chirps", who);
  const std::int64_t max_range_bins =
      require_extent(extents.max_range_bins, "max_range_bins",
                     who);
  const std::int64_t max_rx =
      extent_rx_or_channels(extents);
  const std::int64_t max_tx = extent_or_one(extents.max_tx);
  if (max_rx <= 0 || max_tx <= 0) {
    throw std::invalid_argument(
        "FMCWRangeProcessorOp: max_rx/max_tx extents must be "
        "positive when set");
  }
  extents_ = extents;
  max_rank_ = static_cast<std::uint32_t>(
      2U + named_tx_rx_leading_rank(extents_));
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_tx, max_rx, max_chirps, max_range_bins}, who);
  range_bins_byte_span_ = byte_span_real_cube<T>(
      {max_range_bins}, who);
}

template <typename T>
FMCWRangeProcessorOp<T>::~FMCWRangeProcessorOp() = default;

template <typename T>
void FMCWRangeProcessorOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorAllocationBounds fat_bounds{
      .max_rank = max_rank_,
      .max_byte_span = max_byte_span_};
  const TensorAllocationBounds bins_bounds{
      .max_rank = 1U,
      .max_byte_span = range_bins_byte_span_};

  spec.input(adc_cube).queue_depth(2U).expects_tensor(kDevice);
  spec.output(range_profiles)
      .produces_tensor(kDevice)
      .tensor_allocation(fat_bounds);
  spec.output(range_bins)
      .produces_tensor(kDevice)
      .tensor_allocation(bins_bounds);

  // [live] sticky override ports for POD scalars.
  spec.input(apply_dechirp_in).queue_depth(2U);
  spec.input(fft_norm_in).queue_depth(2U);
  spec.input(nfft_in).queue_depth(2U);
}

template <typename T>
Contract FMCWRangeProcessorOp<T>::contract() const {
  // Trigger on the data tensor only. Sticky override ports and the optional
  // range_bins output are intentionally omitted from the contract so
  // unconnected ports do not yield TEMPORAL_INPUT_UNCONNECTED and an
  // unwired range_bins sink does not gate scheduling.
  Contract result;
  result.trigger(OnEach{adc_cube});
  return result;
}

template <typename T>
expected<void, Error> FMCWRangeProcessorOp<T>::compute(
    ExecutionContext& context) {
  // Receive the input tensor, validate the format, and prepare for processing.
  HOLOSCAN_TRY(auto input, receive_tensor(adc_cube, kOpName));

  try {
    require_cuda_accessible(input, "adc_cube", kOpName);
    require_contiguous(input, "adc_cube", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  HOLOSCAN_TRY(auto complex_input, classify_complex_input<T>(input));
  HOLOSCAN_TRY(auto batched, resolve_batched_shape(input, extents_));

  // Resolve the configuration parameters from the operator context.
  const FMCWRangeProcessorParams& pinned =
      context.params<FMCWRangeProcessorParams>();

  HOLOSCAN_TRY(auto effective,
               resolve_overrides(
                   apply_dechirp_in, fft_norm_in, nfft_in, pinned,
                   sticky_apply_dechirp_, sticky_fft_norm_, sticky_nfft_));

  HOLOSCAN_TRY(auto validated,
               validate_configuration(effective, complex_input, batched));

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    const std::string& msg = std::string(kOpName) + ": no CUDA compute stream";
    return make_unexpected(Error{ErrorCode::kNotSupported, msg});
  }

  // Conditionally prepare range_bins first (before range_profiles) so both
  // loans are ready before either is emitted.
  const bool emit_bins = validated.waveform_present;
  std::optional<TensorOutputLoan> bins_loan;
  if (emit_bins) {
    const std::vector<int64_t> bins_shape{validated.num_bins};
    const DLDataType bins_dtype = real_dtype<T>();
    HOLOSCAN_TRY(auto prepared,
                 prepare_device_tensor(
                     range_bins, context, bins_shape, bins_dtype,
                     [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_range_bins<T>(
                             validated.stage, batched.cols, validated.num_bins,
                             static_cast<T*>(dst), stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       } catch (const std::runtime_error& e) {
                         return make_unexpected(
                             Error{ErrorCode::kFailure, e.what()});
                       }
                       return {};
                     }));
    bins_loan.emplace(std::move(prepared));
  }

  // Run the range stage into a freshly allocated output tensor.
  const std::vector<int64_t> out_shape =
      full_rank_shape(batched, batched.rows, validated.num_bins);
  const DLDataType out_dtype = complex_dtype<T>();
  const void* input_ptr = input.data();

  HOLOSCAN_TRY(auto profiles_loan,
               prepare_device_tensor(
                   range_profiles, context, out_shape, out_dtype,
                   [&](void* dst, std::size_t) -> expected<void, Error> {
                     try {
                       detail::run_range_stage<T>(
                           validated.stage, *state_, input_ptr, complex_input,
                           batched.batch, batched.rows, batched.cols, dst,
                           validated.num_bins, stream);
                     } catch (const std::invalid_argument& e) {
                       return fail_invalid(e.what());
                     }
                     return {};
                   }));

  // Emit all prepared loans.
  HOLOSCAN_TRY(emit_prepared_tensor(range_profiles, context,
                                    std::move(profiles_loan)));
  if (bins_loan.has_value()) {
    HOLOSCAN_TRY(emit_prepared_tensor(range_bins, context,
                                      std::move(*bins_loan)));
  }

  retained_input_ = std::move(input);
  return {};
}

template class FMCWRangeProcessorOp<float>;
template class FMCWRangeProcessorOp<double>;

}  // namespace holoscan::radar
