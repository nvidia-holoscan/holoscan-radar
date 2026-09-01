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

// Kernel knobs plus output-axis decisions derived from this tick's input
// dtype and shape.
struct RangeStagePlan {
  detail::RangeStageConfig stage;
  int64_t num_bins;
};

// Merge Exact [live] ports onto pinned Params and map sentinels to the
// stage form (nfft == 0 and an empty waveform become nullopt).
expected<detail::RangeStageConfig, Error> resolve_params(
    Input<wire::WindowSpecT>& range_window_in,
    Input<bool>& apply_dechirp_in,
    Input<wire::FMCWWaveformT>& waveform_in, Input<FftNorm>& fft_norm_in,
    Input<int64_t>& nfft_in, const FMCWRangeProcessorParams& pinned) {
  HOLOSCAN_TRY(
      auto range_window,
      resolve_live_override_via(
          range_window_in, pinned.range_window,
          [](const wire::WindowSpecT& value) { return from_wire(value); }));
  HOLOSCAN_TRY(
      auto apply_dechirp,
      resolve_live_override(apply_dechirp_in, pinned.apply_dechirp));
  HOLOSCAN_TRY(
      auto waveform,
      resolve_live_override_via(
          waveform_in, pinned.waveform,
          [](const wire::FMCWWaveformT& value) { return from_wire(value); }));
  HOLOSCAN_TRY(auto fft_norm,
               resolve_live_override(fft_norm_in, pinned.fft_norm));
  HOLOSCAN_TRY(auto nfft_raw, resolve_live_override(nfft_in, pinned.nfft));

  std::optional<int64_t> nfft;
  if (nfft_raw != 0) {
    nfft = nfft_raw;
  }
  std::optional<FMCWWaveformDescriptor> opt_waveform;
  if (waveform.is_valid()) {
    opt_waveform = std::move(waveform);
  }

  return detail::RangeStageConfig{
      .window = std::move(range_window),
      .apply_dechirp = apply_dechirp,
      .waveform = std::move(opt_waveform),
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

// Check the stage config against this tick's input and size the range axis.
expected<RangeStagePlan, Error> build_range_stage_plan(
    detail::RangeStageConfig stage, bool complex_input,
    const BatchedShape& batched) {
  if (stage.nfft.has_value() && *stage.nfft < batched.cols) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": nfft (" + std::to_string(*stage.nfft) +
            ") must be >= samples_per_chirp (" +
            std::to_string(batched.cols) + ") on port adc_cube"});
  }
  if (stage.apply_dechirp) {
    if (!stage.waveform.has_value()) {
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

  const int64_t num_bins =
      detail::range_num_bins(complex_input, batched.cols, stage.nfft);
  return RangeStagePlan{.stage = std::move(stage), .num_bins = num_bins};
}

}  // namespace

template <typename T>
FMCWRangeProcessorOp<T>::FMCWRangeProcessorOp(
    const MaxTensorExtents& extents)
    : state_(std::make_shared<detail::RangeStageState<T>>()) {  // NOLINT
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
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorBounds fat_bounds{
      .max_rank = max_rank_,
      .max_byte_span = max_byte_span_};
  const TensorBounds bins_bounds{
      .max_rank = 1U,
      .max_byte_span = range_bins_byte_span_};

  spec.input(adc_cube).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(range_profiles)
      .produces_tensor(pooled_tensor_output(kDevice, fat_bounds));
  spec.output(range_bins)
      .produces_tensor(pooled_tensor_output(kDevice, bins_bounds));

  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(range_window_in).queue_depth(2U);
  spec.input(apply_dechirp_in).queue_depth(2U);
  spec.input(waveform_in).queue_depth(2U);
  spec.input(fft_norm_in).queue_depth(2U);
  spec.input(nfft_in).queue_depth(2U);
}

template <typename T>
Contract FMCWRangeProcessorOp<T>::contract() const {
  // Trigger on the data tensor. Every Exact override is declared whether or
  // not the application wires it: an unwired one compiles to a warning and
  // reads as a miss. Optional range_bins stays omitted so an unwired sink does
  // not gate scheduling.
  Contract result;
  result.trigger(OnEach{adc_cube});
  add_live_override_to_contract(result, range_window_in);
  add_live_override_to_contract(result, apply_dechirp_in);
  add_live_override_to_contract(result, waveform_in);
  add_live_override_to_contract(result, fft_norm_in);
  add_live_override_to_contract(result, nfft_in);
  return result;
}

template <typename T>
expected<void, Error> FMCWRangeProcessorOp<T>::compute(
    ExecutionContext& context) {
  // Receive adc_cube; require a device-contiguous buffer.
  HOLOSCAN_TRY(auto input, receive_tensor(adc_cube, kOpName));
  try {
    require_cuda_accessible(input, "adc_cube", kOpName);
    require_contiguous(input, "adc_cube", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Complex vs real IF, and collapse named leading axes (N_rx / M_tx).
  HOLOSCAN_TRY(auto complex_input, classify_complex_input<T>(input));
  HOLOSCAN_TRY(auto batched,
               collapse_named_tx_rx_batch(
                   input, extents_, "adc_cube", kOpName));

  // This tick's Params: pinned defaults with any matched [live] override.
  const FMCWRangeProcessorParams& pinned =
      context.params<FMCWRangeProcessorParams>();
  HOLOSCAN_TRY(auto stage,
               resolve_params(range_window_in, apply_dechirp_in, waveform_in,
                              fft_norm_in, nfft_in, pinned));

  // Bind the stage config to this input's dtype/shape.
  HOLOSCAN_TRY(auto plan,
               build_range_stage_plan(std::move(stage), complex_input,
                                      batched));

  // MatX work runs on the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    const std::string& msg = std::string(kOpName) + ": no CUDA compute stream";
    return make_unexpected(Error{ErrorCode::kNotSupported, msg});
  }

  // Range compression into a pool loan, then publish range_profiles.
  const std::vector<int64_t> out_shape =
      full_rank_shape(batched, batched.rows, plan.num_bins);
  const DLDataType out_dtype = complex_dtype<T>();
  const void* input_ptr = input.data();

  // Compute range profiles and emit.
  HOLOSCAN_TRY(auto profiles_loan,
               prepare_device_tensor(
                   range_profiles, context, out_shape, out_dtype,
                   [&](void* dst, std::size_t) -> expected<void, Error> {
                     try {
                       detail::run_range_stage<T>(
                           plan.stage, *state_, input_ptr, complex_input,
                           batched.batch, batched.rows, batched.cols, dst,
                           plan.num_bins, stream);
                     } catch (const std::invalid_argument& e) {
                       return fail_invalid(e.what());
                     }
                     return {};
                   }));
  HOLOSCAN_TRY(emit_prepared_tensor(range_profiles, context,
                                    std::move(profiles_loan)));

  // Optional range_bins: waveform present and the latched sink still
  // active. Independent of the FFT; an unwired emit is absorbed.
  if (plan.stage.waveform.has_value() && range_bins_out_.active()) {
    HOLOSCAN_TRY(auto bins_loan,
                 prepare_device_tensor(
                     range_bins, context,
                     std::vector<int64_t>{plan.num_bins},
                     real_dtype<T>(),
                     [&](void* dst, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_range_bins<T>(
                             plan.stage, batched.cols, plan.num_bins,
                             static_cast<T*>(dst), stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       } catch (const std::runtime_error& e) {
                         return make_unexpected(
                             Error{ErrorCode::kFailure, e.what()});
                       }
                       return {};
                     }));
    range_bins_out_.emit(range_bins, context, std::move(bins_loan),
                         "FMCWRangeProcessorOp.range_bins");
  }

  // Keep adc_cube alive until stream-ordered reads complete.
  retained_input_ = std::move(input);
  return {};
}

template class FMCWRangeProcessorOp<float>;
template class FMCWRangeProcessorOp<double>;

}  // namespace holoscan::radar
