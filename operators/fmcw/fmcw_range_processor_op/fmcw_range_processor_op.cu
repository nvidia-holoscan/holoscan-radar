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

}  // namespace

template <typename T>
FMCWRangeProcessorOp<T>::FMCWRangeProcessorOp(
    const MaxTensorExtents& extents) {
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
  HOLOSCAN_TRY(auto input, receive_tensor(adc_cube, kOpName));

  const FMCWRangeProcessorParams& pinned =
      context.params<FMCWRangeProcessorParams>();

  // Resolve sticky [live] overrides for POD scalars.
  HOLOSCAN_TRY(auto eff_apply_dechirp,
               resolve_sticky_override(apply_dechirp_in,
                                       pinned.apply_dechirp,
                                       sticky_apply_dechirp_));
  HOLOSCAN_TRY(auto eff_fft_norm,
               resolve_sticky_override(fft_norm_in, pinned.fft_norm,
                                       sticky_fft_norm_));
  HOLOSCAN_TRY(auto eff_nfft,
               resolve_sticky_override(nfft_in, pinned.nfft,
                                       sticky_nfft_));

  // [EA HSDK5 ONLY] range_window and waveform are Params-only (§9).
  const WindowSpec& eff_window = pinned.range_window;
  const FMCWWaveformDescriptor& eff_waveform = pinned.waveform;

  // Classify and validate input against the locked contract.
  InputPrecision precision;
  try {
    precision = classify_input_dtype<T>(input, "adc_cube",
                                        /*allow_real=*/true);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const bool complex_input = precision == InputPrecision::kComplex;

  try {
    require_cuda_accessible(input, "adc_cube", kOpName);
    require_contiguous(input, "adc_cube", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  BatchedShape batched;
  try {
    batched = collapse_batch(input, "adc_cube");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const std::size_t max_leading =
      named_tx_rx_leading_rank(extents_);
  if (batched.leading.size() > max_leading) {
    return fail_invalid(unexpected_leading_message(
        kOpName, batched.leading.size(), max_leading));
  }
  const std::int64_t max_rx = extent_rx_or_channels(extents_);
  const std::int64_t max_tx = extent_or_one(extents_.max_tx);
  if (batched.batch > max_tx * max_rx) {
    return fail_invalid(
        std::string(kOpName) + ": leading product " +
        std::to_string(batched.batch) +
        " exceeds max_tx*max_rx from MaxTensorExtents");
  }

  // nfft: 0 means exact (no zero-padding); translate to optional for
  // the stage config.
  std::optional<int64_t> opt_nfft;
  if (eff_nfft != 0) {
    opt_nfft = eff_nfft;
  }

  if (opt_nfft.has_value() && *opt_nfft < batched.cols) {
    return fail_invalid(
        std::string(kOpName) + ": nfft (" + std::to_string(*opt_nfft) +
        ") must be >= samples_per_chirp (" +
        std::to_string(batched.cols) + ") on port adc_cube");
  }
  if (eff_apply_dechirp) {
    if (!eff_waveform.is_valid()) {
      return fail_invalid(
          std::string(kOpName) +
          ": apply_dechirp = true requires a waveform descriptor "
          "(dechirp reference source)");
    }
    if (!complex_input) {
      return fail_invalid(
          std::string(kOpName) +
          ": apply_dechirp = true requires complex-IF input on port "
          "adc_cube (received real-IF samples; software dechirp "
          "produces complex samples, inconsistent with the real-IF "
          "output size)");
    }
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 std::string(kOpName) +
                                     ": no CUDA compute stream"});
  }

  // Build the RangeStageConfig from effective parameters.
  const bool waveform_present = eff_waveform.is_valid();
  std::optional<FMCWWaveformDescriptor> opt_waveform;
  if (waveform_present) {
    opt_waveform = eff_waveform;
  }

  detail::RangeStageConfig cfg;
  cfg.window = eff_window;
  cfg.apply_dechirp = eff_apply_dechirp;
  cfg.waveform = opt_waveform;
  cfg.fft_norm = eff_fft_norm;
  cfg.propagation_speed = pinned.propagation_speed;
  cfg.nfft = opt_nfft;

  const int64_t num_bins =
      detail::range_num_bins(complex_input, batched.cols, opt_nfft);

  if (!state_) {
    state_ = std::make_shared<detail::RangeStageState<T>>();
  }

  // Conditionally prepare range_bins first (before range_profiles) so both
  // loans are ready before either is emitted.
  const bool emit_bins = waveform_present;
  std::optional<TensorOutputLoan> bins_loan;
  if (emit_bins) {
    const std::vector<int64_t> bins_shape{num_bins};
    const DLDataType bins_dtype = real_dtype<T>();
    HOLOSCAN_TRY(auto prepared,
                 prepare_device_tensor(
                     range_bins, context, bins_shape, bins_dtype,
                     [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_range_bins<T>(
                             cfg, batched.cols, num_bins,
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
      full_rank_shape(batched, batched.rows, num_bins);
  const DLDataType out_dtype = complex_dtype<T>();
  const void* input_ptr = input.data();

  // warn_on_rebuild: false — Params-only window/waveform never trigger
  // mid-stream rebuilds; POD scalars do not cause ConfigCache invalidation
  // in the range stage.
  constexpr bool kWarnOnRebuild = false;

  HOLOSCAN_TRY(auto profiles_loan,
               prepare_device_tensor(
                   range_profiles, context, out_shape, out_dtype,
                   [&](void* dst, std::size_t) -> expected<void, Error> {
                     try {
                       detail::run_range_stage<T>(
                           cfg, *state_, input_ptr, complex_input,
                           batched.batch, batched.rows, batched.cols,
                           dst, num_bins, kWarnOnRebuild, stream);
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
