// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/tdm_dense_compensation_op/tdm_dense_compensation_op.hpp"

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {
constexpr const char* kOpName = "TDMDenseCompensationOp";
expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}
template <typename T>
std::vector<std::complex<T>> cast_correction(const std::vector<std::complex<double>>& host) {
  std::vector<std::complex<T>> out(host.size());
  for (size_t i = 0; i < host.size(); ++i)
    out[i] = {static_cast<T>(host[i].real()), static_cast<T>(host[i].imag())};
  return out;
}
expected<FMCWWaveformDescriptor, Error> resolve_waveform(Input<wire::FMCWWaveformT>& waveform_in,
                                                         const TDMDenseCompensationParams& pinned) {
  HOLOSCAN_TRY(auto waveform, resolve_live_override_via(waveform_in, pinned.waveform,
                                                        [](const wire::FMCWWaveformT& value) {
                                                          return from_wire(value);
                                                        }));
  if (!waveform.is_valid())
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string(kOpName) +
                                               ": waveform must be a valid FMCWWaveformDescriptor "
                                               "(build-time or [live] override)"});
  return waveform;
}
}  // namespace
template <typename T>
TDMDenseCompensationOp<T>::TDMDenseCompensationOp(const MaxTensorExtents& extents) {
  const int64_t mt = require_extent(extents.max_tx, "max_tx", kOpName);
  const int64_t mr = require_extent(extents.max_rx, "max_rx", kOpName);
  const int64_t md = require_extent(extents.max_doppler_bins, "max_doppler_bins", kOpName);
  const int64_t mg = require_extent(extents.max_range_bins, "max_range_bins", kOpName);
  if (extents.max_samples_per_chirp && *extents.max_samples_per_chirp <= 0)
    throw std::invalid_argument(std::string(kOpName) +
                                ": max_samples_per_chirp must be positive when set");
  extents_ = extents;
  max_byte_span_ = byte_span_complex_cube<T>({mt, mr, md, mg}, kOpName);
}
template <typename T>
void TDMDenseCompensationOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{
      .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.output(rd_maps_out)
      .produces_tensor(
          pooled_tensor_output(device, {.max_rank = max_rank_, .max_byte_span = max_byte_span_}));
  spec.input(waveform_in).optional().queue_depth(2U);
}
template <typename T>
Contract TDMDenseCompensationOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  add_live_override_to_contract(result, waveform_in);
  return result;
}
template <typename T>
expected<void, Error> TDMDenseCompensationOp<T>::run_compute(
    ExecutionContext& context, const detail::TdmRdMapsLayout& layout,
    const FMCWWaveformDescriptor& waveform, const std::vector<int64_t>& slots,
    const std::vector<int64_t>& out_shape, const Tensor& input, cudaStream_t stream) {
  const auto key =
      detail::tdm_correction_cache_key(waveform, layout.num_tx, layout.num_doppler, slots);
  const CachedCorrection& cached = correction_cache_.get_or_build(key, "TDM correction table", [&] {
    CachedCorrection entry;
    auto typed = cast_correction<T>(
        detail::build_tdm_correction_table(waveform, layout.num_tx, layout.num_doppler, slots));
    entry.device_table =
        make_device_tensor({layout.num_tx, layout.num_doppler}, complex_dtype<T>());
    auto mut = entry.device_table->mutable_data();
    if (!mut)
      throw std::runtime_error(std::string(kOpName) + ": correction-table not writable");
    if (cudaMemcpyAsync(*mut, typed.data(), typed.size() * sizeof(std::complex<T>),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess)
      throw std::runtime_error(std::string(kOpName) + ": correction-table H2D failed");
    return entry;
  });
  const void* ptr = input.data();
  HOLOSCAN_TRY(emit_device_tensor(rd_maps_out, context, out_shape, input.dtype(),
                                  [&](void* out, std::size_t) -> expected<void, Error> {
                                    try {
                                      detail::run_tdm_compensation_stage<T>(
                                          ptr, out, layout, cached.device_table->data(), stream);
                                    } catch (const std::invalid_argument& e) {
                                      return fail_invalid(e.what());
                                    }
                                    return {};
                                  }));
  return {};
}
template <typename T>
expected<void, Error> TDMDenseCompensationOp<T>::compute(ExecutionContext& context) {
  // Receive and validate inputs before deriving any tick-local state.
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));
  try {
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
    require_complex<T>(input, "rd_maps");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  // Resolve tick-local configuration; Exact misses revert to pinned Params.
  const auto& pinned = context.params<TDMDenseCompensationParams>();
  HOLOSCAN_TRY(auto waveform, resolve_waveform(waveform_in, pinned));
  if (extents_.max_samples_per_chirp &&
      waveform.reference_samples().size() > static_cast<size_t>(*extents_.max_samples_per_chirp))
    return fail_invalid(std::string(kOpName) +
                        ": waveform reference_samples length exceeds max_samples_per_chirp");
  detail::TdmRdMapsLayout layout;
  std::vector<int64_t> slots;
  try {
    if (input.shape().size() != 4U) {
      return fail_invalid(std::string(kOpName) +
                          ": rd_maps rank must be 4 (M_tx, N_rx, Doppler, range)");
    }
    layout = detail::parse_tdm_rd_maps_layout(input, "rd_maps");
    if (layout.num_tx > *extents_.max_tx || layout.num_rx > *extents_.max_rx ||
        layout.num_doppler > *extents_.max_doppler_bins ||
        layout.num_range > *extents_.max_range_bins) {
      return fail_invalid(std::string(kOpName) + ": rd_maps shape exceeds MaxTensorExtents");
    }
    slots = detail::tdm_firing_slots(pinned.tx_schedule, layout.num_tx);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (!stream)
    return make_unexpected(
        Error{ErrorCode::kNotSupported, std::string(kOpName) + ": no CUDA compute stream"});
  // Execute the device stage on the input runtime stream.
  HOLOSCAN_TRY(run_compute(context, layout, waveform, slots, detail::tdm_output_rd_shape(layout),
                           input, stream));
  // Publish coherent outputs in dependency order and retain borrowed inputs.
  retained_input_ = std::move(input);
  return {};
}
template class TDMDenseCompensationOp<float>;
template class TDMDenseCompensationOp<double>;
}  // namespace holoscan::radar
