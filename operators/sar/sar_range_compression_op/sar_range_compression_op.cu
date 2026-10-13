// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sar/sar_range_compression_op/sar_range_compression_op.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/sar_range_compression_op/range_compression_stage.hpp"

namespace holoscan::radar {
namespace {
constexpr const char* kOpName = "SARRangeCompressionOp";

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

expected<detail::RangeCompressionConfig, Error> resolve_stage_config(
    Input<WireWindowSpec>& window_in, Input<WireLFMWaveform>& waveform_in,
    Input<FftNorm>& fft_norm_in, Input<int64_t>& nfft_in, Input<double>& propagation_speed_in,
    const SARRangeCompressionParams& pinned) {
  HOLOSCAN_TRY(auto window, resolve_live_override_via(window_in, pinned.range_window,
                                                      [](const WireWindowSpec& wire) {
                                                        return from_wire(wire);
                                                      }));
  HOLOSCAN_TRY(auto waveform, resolve_live_override_via(waveform_in, pinned.waveform,
                                                        [](const WireLFMWaveform& wire) {
                                                          return from_wire_lfm(wire);
                                                        }));
  HOLOSCAN_TRY(auto fft_norm, resolve_live_override(fft_norm_in, pinned.fft_norm));
  HOLOSCAN_TRY(auto nfft, resolve_live_override(nfft_in, pinned.nfft));
  HOLOSCAN_TRY(auto propagation_speed,
               resolve_live_override(propagation_speed_in, pinned.propagation_speed));
  if (nfft < 0 || !std::isfinite(propagation_speed) || propagation_speed <= 0.0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              std::string(kOpName) + ": nfft must be nonnegative and propagation_speed positive"});
  }
  detail::RangeCompressionConfig result;
  result.window = std::move(window);
  if (waveform.is_valid()) {
    result.waveform = std::move(waveform);
  }
  result.fft_norm = fft_norm;
  if (nfft > 0) {
    result.nfft = nfft;
  }
  result.propagation_speed = propagation_speed;
  return result;
}

expected<TensorOutputLoan, Error> prepare_window_output(Output<WireWindowSpec>& output,
                                                        const WindowSpec& window) {
  const std::vector<double>& coeff = window.coefficients;
  const std::vector<int64_t> shape{static_cast<int64_t>(coeff.size())};
  auto loan = output.allocate_tensor(
      TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLFloat, 64U, 1U}});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto committed = loan->write_host_and_commit([&coeff](const TensorOutputWriteGuard& writer) {
    std::copy(coeff.begin(), coeff.end(), writer.data_as<double>());
  });
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return std::move(*loan);
}
}  // namespace

template <typename T>
SARRangeCompressionOp<T>::SARRangeCompressionOp(const MaxTensorExtents& extents)
: state_(std::make_shared<detail::RangeCompressionState<T>>()), extents_(extents) {
  const auto pulses = require_extent(extents.max_pulses, "max_pulses", kOpName);
  const auto samples =
      require_extent(extents.max_samples_per_pulse, "max_samples_per_pulse", kOpName);
  const auto bins = require_extent(extents.max_range_bins, "max_range_bins", kOpName);
  max_byte_span_ = byte_span_complex_cube<T>({pulses, bins}, kOpName);
  bins_byte_span_ = byte_span_real_cube<T>({bins}, kOpName);
  window_byte_span_ = byte_span_real_cube<double>({samples}, kOpName);
}

template <typename T>
SARRangeCompressionOp<T>::~SARRangeCompressionOp() = default;

template <typename T>
void SARRangeCompressionOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  constexpr TensorRepresentation host_window{
      .memory_kind = MemoryKind::kHost, .dtype = DLDataType{kDLFloat, 64U, 1U}, .rank = 1U};
  spec.input(phase_history).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.output(range_profiles).produces_tensor(pooled_tensor_output(device, 2U, max_byte_span_));
  spec.output(range_bins).produces_tensor(pooled_tensor_output(device, 1U, bins_byte_span_));
  spec.output(range_window_out)
      .max_emits_per_compute(1U)
      .produces_tensor(pooled_tensor_output(host_window, 1U, window_byte_span_));
  spec.input(range_window_in).optional().queue_depth(2U);
  spec.input(waveform_in).optional().queue_depth(2U);
  spec.input(fft_norm_in).optional().queue_depth(2U);
  spec.input(nfft_in).optional().queue_depth(2U);
  spec.input(propagation_speed_in).optional().queue_depth(2U);
}

template <typename T>
Contract SARRangeCompressionOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{phase_history});
  add_live_override_to_contract(result, range_window_in);
  add_live_override_to_contract(result, waveform_in);
  add_live_override_to_contract(result, fft_norm_in);
  add_live_override_to_contract(result, nfft_in);
  add_live_override_to_contract(result, propagation_speed_in);
  return result;
}

template <typename T>
expected<void, Error> SARRangeCompressionOp<T>::compute(ExecutionContext& context) {
  HOLOSCAN_TRY(auto input, receive_tensor(phase_history, kOpName));
  try {
    require_cuda_accessible(input, "phase_history", kOpName);
    require_contiguous(input, "phase_history", kOpName);
    require_complex<T>(input, "phase_history");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  const auto shape = input.shape();
  if (shape.size() != 2U || shape[0] <= 0 || shape[1] <= 0 || shape[0] > *extents_.max_pulses ||
      shape[1] > *extents_.max_samples_per_pulse) {
    return fail_invalid(std::string(kOpName) +
                        ": phase_history must be (num_pulses, num_samples) within bounds");
  }
  // Resolve this tick's optional overrides, then size the centred IFFT output.
  HOLOSCAN_TRY(auto cfg, resolve_stage_config(range_window_in, waveform_in, fft_norm_in, nfft_in,
                                              propagation_speed_in,
                                              context.params<SARRangeCompressionParams>()));
  if (cfg.nfft.has_value() && *cfg.nfft < shape[1]) {
    return fail_invalid(std::string(kOpName) + ": nfft must be >= num_samples");
  }
  const int64_t bins = detail::compression_num_bins(shape[1], cfg.nfft);
  if (bins > *extents_.max_range_bins) {
    return fail_invalid(std::string(kOpName) + ": output exceeds max_range_bins");
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, std::string(kOpName) + ": no CUDA compute stream"});
  }
  const std::vector<int64_t> out_shape{shape[0], bins};
  const bool warn_on_rebuild = !range_window_in.is_missing() || !nfft_in.is_missing();
  HOLOSCAN_TRY(
      auto profiles_loan,
      prepare_device_tensor(range_profiles, context, out_shape, complex_dtype<T>(),
                            [&](void* dst, std::size_t) -> expected<void, Error> {
                              try {
                                detail::run_range_compression<T>(cfg, *state_, input.data(), 1,
                                                                 shape[0], shape[1], dst, bins,
                                                                 warn_on_rebuild, stream);
                              } catch (const std::invalid_argument& e) {
                                return fail_invalid(e.what());
                              } catch (const std::exception& e) {
                                return make_unexpected(Error{ErrorCode::kFailure, e.what()});
                              }
                              return {};
                            }));

  // Physical range bins and the applied window are computed only for wired taps.
  if (cfg.waveform.has_value() && range_bins_out_.active()) {
    HOLOSCAN_TRY(
        auto bins_loan,
        prepare_device_tensor(range_bins, context, std::vector<int64_t>{bins}, real_dtype<T>(),
                              [&](void* dst, std::size_t) -> expected<void, Error> {
                                try {
                                  detail::fill_compressed_range_bins<T>(
                                      cfg, shape[1], bins, static_cast<T*>(dst), stream);
                                } catch (const std::exception& e) {
                                  return make_unexpected(Error{ErrorCode::kFailure, e.what()});
                                }
                                return {};
                              }));
    range_bins_out_.emit(range_bins, context, std::move(bins_loan),
                         "SARRangeCompressionOp.range_bins");
  }
  if (range_window_out_.active()) {
    HOLOSCAN_TRY(auto window_loan, prepare_window_output(range_window_out, cfg.window));
    range_window_out_.emit_schema_tensor(range_window_out, context, std::move(window_loan),
                                         to_wire(cfg.window),
                                         "SARRangeCompressionOp.range_window_out");
  }
  HOLOSCAN_TRY(emit_prepared_tensor(range_profiles, context, std::move(profiles_loan)));
  retained_input_ = std::move(input);
  return {};
}

template class SARRangeCompressionOp<float>;
template class SARRangeCompressionOp<double>;

}  // namespace holoscan::radar
