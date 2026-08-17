// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/fmcw_doppler_processor_op/fmcw_doppler_processor_op.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
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

template <typename T>
constexpr const char* precision_name() {
  return std::is_same_v<T, float> ? "float" : "double";
}

}  // namespace

template <typename T>
FMCWDopplerProcessorOp<T>::FMCWDopplerProcessorOp(
    const MaxTensorExtents& extents) {
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
  constexpr TensorPortLayout kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  const TensorAllocationBounds fat_bounds{
      .max_rank = max_rank_,
      .max_byte_span = max_byte_span_};
  const TensorAllocationBounds vel_bounds{
      .max_rank = 1U,
      .max_byte_span = velocity_bins_byte_span_};
  spec.input(range_profiles).queue_depth(2U).expects_tensor(kDevice);
  spec.output(rd_maps)
      .produces_tensor(kDevice)
      .tensor_allocation(fat_bounds);
  spec.output(velocity_bins)
      .produces_tensor(kDevice)
      .tensor_allocation(vel_bounds);

  // Sticky [live] POD override ports.
  spec.input(center_zero_doppler_in).queue_depth(2U);
  spec.input(fft_norm_in).queue_depth(2U);
  spec.input(nfft_in).queue_depth(2U);
}

template <typename T>
Contract FMCWDopplerProcessorOp<T>::contract() const {
  // Trigger on the data tensor only. The [live] sticky override inputs
  // are intentionally omitted from the temporal contract so an unconnected
  // override port does not yield TEMPORAL_INPUT_UNCONNECTED (same pattern
  // as ClutterFilterOp / MIMOCoherentCompensationOp).
  Contract result;
  result.trigger(OnEach{range_profiles});
  return result;
}

template <typename T>
expected<void, Error> FMCWDopplerProcessorOp<T>::compute(
    ExecutionContext& context) {
  const std::string op_name =
      std::string("FMCWDopplerProcessorOp<") + precision_name<T>() + ">";

  HOLOSCAN_TRY(auto input,
               receive_tensor(range_profiles, op_name));

  const FMCWDopplerProcessorParams& pinned =
      context.params<FMCWDopplerProcessorParams>();

  // Resolve sticky [live] overrides from POD input ports.
  HOLOSCAN_TRY(auto eff_center,
               resolve_sticky_override(center_zero_doppler_in,
                                       pinned.center_zero_doppler,
                                       sticky_center_zero_doppler_));
  HOLOSCAN_TRY(auto eff_fft_norm,
               resolve_sticky_override(fft_norm_in, pinned.fft_norm,
                                       sticky_fft_norm_));
  HOLOSCAN_TRY(auto eff_nfft_raw,
               resolve_sticky_override(nfft_in, pinned.nfft,
                                       sticky_nfft_));

  // nfft 0 = "use num_chirps" (Params "none" convention).
  const std::optional<int64_t> eff_nfft =
      eff_nfft_raw != 0 ? std::optional<int64_t>{eff_nfft_raw} : std::nullopt;

  // doppler_window and waveform are Params-only on EA (HSDK5_PORT §9).
  const WindowSpec& eff_window = pinned.doppler_window;
  const bool waveform_effective = pinned.waveform.is_valid();
  const std::optional<FMCWWaveformDescriptor> eff_waveform =
      waveform_effective ? std::optional{pinned.waveform} : std::nullopt;

  try {
    classify_input_dtype<T>(input, "range_profiles", /*allow_real=*/false);
    require_cuda_accessible(input, "range_profiles", op_name);
    require_contiguous(input, "range_profiles", op_name);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  BatchedShape batched;
  try {
    batched = collapse_batch(input, "range_profiles");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const std::size_t max_leading =
      named_tx_rx_leading_rank(extents_);
  if (batched.leading.size() > max_leading) {
    return fail_invalid(unexpected_leading_message(
        op_name.c_str(), batched.leading.size(),
        max_leading));
  }
  const std::int64_t max_rx_val =
      extent_rx_or_channels(extents_);
  const std::int64_t max_tx_val =
      extent_or_one(extents_.max_tx);
  if (batched.batch > max_tx_val * max_rx_val) {
    return fail_invalid(
        op_name + ": leading product " +
        std::to_string(batched.batch) +
        " exceeds max_tx*max_rx from "
        "MaxTensorExtents");
  }
  const int64_t num_chirps = batched.rows;
  const int64_t num_range_bins = batched.cols;
  if (eff_nfft.has_value() && *eff_nfft < num_chirps) {
    std::ostringstream msg;
    msg << op_name << ": nfft (" << *eff_nfft
        << ") must be >= num_chirps (" << num_chirps
        << ") on input port 'range_profiles'";
    return fail_invalid(msg.str());
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotSupported,
                                 op_name + ": no CUDA compute stream"});
  }

  if (!state_) {
    state_ = std::make_shared<detail::DopplerStageState<T>>();
  }

  detail::DopplerStageConfig cfg;
  cfg.window = eff_window;
  cfg.center_zero_doppler = eff_center;
  cfg.waveform = eff_waveform;
  cfg.fft_norm = eff_fft_norm;
  cfg.nfft = eff_nfft;

  const int64_t num_doppler =
      detail::doppler_num_bins(num_chirps, eff_nfft);

  // Conditional velocity_bins: emitted only when waveform is effective.
  // Enqueued before the Doppler FFT: its host-staged upload waits on the
  // stream, so this ordering keeps that wait off the FFT hot path.
  std::optional<TensorOutputLoan> velocity_loan;
  if (eff_waveform.has_value()) {
    HOLOSCAN_TRY(auto loan,
                 prepare_device_tensor(
                     velocity_bins, context,
                     std::vector<int64_t>{num_doppler},
                     real_dtype<T>(),
                     [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                       try {
                         detail::fill_velocity_bins<T>(
                             cfg, num_doppler, static_cast<T*>(out_ptr),
                             stream);
                       } catch (const std::invalid_argument& e) {
                         return fail_invalid(e.what());
                       }
                       return {};
                     }));
    velocity_loan.emplace(std::move(loan));
  }

  const std::vector<int64_t> out_shape =
      full_rank_shape(batched, num_doppler, num_range_bins);
  const DLDataType dtype = complex_dtype<T>();

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(auto rd_loan,
               prepare_device_tensor(
                   rd_maps, context, out_shape, dtype,
                   [&](void* out_ptr, std::size_t) -> expected<void, Error> {
                     try {
                       detail::run_doppler_stage<T>(
                           cfg, *state_, input_ptr, batched.batch,
                           num_chirps, num_range_bins, out_ptr,
                           num_doppler, /*warn_on_rebuild=*/false, stream);
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
