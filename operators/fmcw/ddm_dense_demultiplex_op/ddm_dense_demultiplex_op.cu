// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/ddm_dense_demultiplex_op/ddm_dense_demultiplex_op.hpp"

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {
namespace {
constexpr const char* kOpName = "DDMDenseDemultiplexOp";
expected<void, Error> fail_invalid(const std::string& m) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, m});
}
template <typename T>
std::vector<std::complex<T>> cast_correction(
    const std::vector<std::complex<double>>& h) {
  std::vector<std::complex<T>> o(h.size());
  for (size_t i = 0; i < h.size(); ++i)
    o[i] = {static_cast<T>(h[i].real()), static_cast<T>(h[i].imag())};
  return o;
}
}  // namespace
template <typename T>
DDMDenseDemultiplexOp<T>::DDMDenseDemultiplexOp(const MaxTensorExtents& e) {
  const auto mt = require_extent(e.max_tx, "max_tx", kOpName);
  const auto mr = require_extent(e.max_rx, "max_rx", kOpName);
  const auto md =
      require_extent(e.max_doppler_bins, "max_doppler_bins", kOpName);
  const auto mg = require_extent(e.max_range_bins, "max_range_bins", kOpName);
  extents_ = e;
  max_byte_span_ = byte_span_complex_cube<T>({mt, mr, md, mg}, kOpName);
  velocity_bins_max_byte_span_ =
      byte_span_real_cube<T>({md}, kOpName);
}
template <typename T>
DDMDenseDemultiplexOp<T>::~DDMDenseDemultiplexOp() {
  detail::release_ddm_correlate_scratch(this);
}
template <typename T>
void DDMDenseDemultiplexOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation device{.memory_kind = MemoryKind::kCudaDevice,
                                        .dtype = std::nullopt,
                                        .rank = std::nullopt};
  constexpr TensorRepresentation axis_input{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt};
  constexpr TensorRepresentation axis_output{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = 1U};
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(device));
  spec.input(velocity_bins)
      .optional()
      .queue_depth(2U)
      .expects_tensor(tensor_input(axis_input));
  spec.output(rd_maps_out)
      .produces_tensor(pooled_tensor_output(
          device, {.max_rank = max_rank_, .max_byte_span = max_byte_span_}));
  spec.output(velocity_bins_out)
      .produces_tensor(pooled_tensor_output(
          axis_output, {.max_rank = 1U,
                        .max_byte_span = velocity_bins_max_byte_span_}));
}
template <typename T>
Contract DDMDenseDemultiplexOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  result.input(velocity_bins, Exact{})
      .on_missing(velocity_bins, SafeDefault{});
  return result;
}
template <typename T>
expected<TensorOutputLoan, Error>
DDMDenseDemultiplexOp<T>::prepare_demux(
    ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
    const DDMCodeDescriptor& code, int64_t width,
    const std::vector<int64_t>& shape, const Tensor& input,
    cudaStream_t stream) {
  const auto offsets = detail::ddm_resolve_offsets(code);
  const auto key = detail::ddm_demux_cache_key(code, layout.num_doppler);
  const CachedDemux& cached =
      demux_cache_.get_or_build(key, "DDM demux table", [&] {
        CachedDemux e;
        auto h = detail::build_ddm_demux_table(offsets, code.num_tx,
                                               layout.num_doppler, width);
        const auto phase = cast_correction<T>(
            detail::ddm_initial_phase_weights(code));
        e.device_table =
            make_device_tensor({code.num_tx, width}, DLDataType{kDLInt, 64, 1});
        e.device_initial_phase_weights =
            make_device_tensor({code.num_tx}, complex_dtype<T>());
        auto p = e.device_table->mutable_data();
        if (!p)
          throw std::runtime_error(std::string(kOpName) +
                                   ": demux-table not writable");
        if (cudaMemcpyAsync(*p, h.data(), h.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
          throw std::runtime_error(std::string(kOpName) +
                                   ": demux-table H2D failed");
        auto phase_p = e.device_initial_phase_weights->mutable_data();
        if (!phase_p ||
            cudaMemcpyAsync(*phase_p, phase.data(),
                            phase.size() * sizeof(std::complex<T>),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
          throw std::runtime_error(std::string(kOpName) +
                                   ": initial-phase H2D failed");
        return e;
      });
  const void* p = input.data();
  return prepare_device_tensor(
      rd_maps_out, context, shape, input.dtype(),
      [&](void* out, std::size_t) -> expected<void, Error> {
        try {
          detail::run_ddm_demux_stage<T>(
              p, out, layout, code.num_tx, width,
              static_cast<const int64_t*>(cached.device_table->data()),
              cached.device_initial_phase_weights->data(), stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      });
}
template <typename T>
expected<TensorOutputLoan, Error>
DDMDenseDemultiplexOp<T>::prepare_correlate(
    ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
    const DDMCodeDescriptor& code, const std::vector<int64_t>& shape,
    const Tensor& input, cudaStream_t stream) {
  const auto key = detail::ddm_correlate_cache_key(code, layout.num_doppler);
  const CachedCorrelate& cached =
      correlate_cache_.get_or_build(key, "DDM correlate FFT weights", [&] {
        CachedCorrelate e;
        auto h = cast_correction<T>(
            detail::build_ddm_corr_weights(code, layout.num_doppler));
        e.device_corr_weights = make_device_tensor(
            {code.num_tx, layout.num_doppler}, complex_dtype<T>());
        auto p = e.device_corr_weights->mutable_data();
        if (!p)
          throw std::runtime_error(std::string(kOpName) +
                                   ": corr-weights not writable");
        if (cudaMemcpyAsync(*p, h.data(), h.size() * sizeof(std::complex<T>),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess)
          throw std::runtime_error(std::string(kOpName) +
                                   ": corr-weights H2D failed");
        return e;
      });
  const void* p = input.data();
  return prepare_device_tensor(
      rd_maps_out, context, shape, input.dtype(),
      [&](void* out, std::size_t) -> expected<void, Error> {
        try {
          detail::run_ddm_correlate_stage<T>(p, out, layout, code.num_tx,
                                             cached.device_corr_weights->data(),
                                             this, stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      });
}
template <typename T>
expected<void, Error> DDMDenseDemultiplexOp<T>::compute(
    ExecutionContext& context) {
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));
  try {
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
    require_complex<T>(input, "rd_maps");
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  HOLOSCAN_TRY(auto maybe_velocity_bins,
               receive_optional_value(velocity_bins));
  detail::DdmRdMapsLayout layout;
  const auto& code = context.params<DDMDenseDemultiplexParams>().ddm_code;
  detail::DdmCodeMode mode;
  int64_t width = 0;
  std::vector<int64_t> shape;
  try {
    if (input.shape().size() != 3U) {
      return fail_invalid(
          std::string(kOpName) +
          ": rd_maps rank must be 3 (N_rx, Doppler, range)");
    }
    layout = detail::parse_ddm_rd_maps_layout(input, "rd_maps");
    mode = detail::ddm_classify_codes(code);
    if (code.num_tx > *extents_.max_tx ||
        layout.num_rx > *extents_.max_rx ||
        layout.num_doppler > *extents_.max_doppler_bins ||
        layout.num_range > *extents_.max_range_bins) {
      return fail_invalid(
          std::string(kOpName) +
          ": rd_maps or ddm_code shape exceeds MaxTensorExtents");
    }
    if (mode == detail::DdmCodeMode::kNonLinear) {
      if (code.guard_bands != 0)
        return fail_invalid(
            std::string(kOpName) +
            ": non-linear DDM phase_codes require guard_bands = 0 (got " +
            std::to_string(code.guard_bands) + ")");
      const int64_t chirps =
          static_cast<int64_t>(code.phase_codes.size()) / code.num_tx;
      if (chirps != layout.num_doppler)
        return fail_invalid(
            std::string(kOpName) +
            ": non-linear DDM phase_codes imply num_chirps = " +
            std::to_string(chirps) +
            " but input num_doppler = " + std::to_string(layout.num_doppler) +
            "; these must match for the correlate path");
      shape = detail::ddm_correlate_output_shape(layout, code.num_tx);
    } else {
      (void)detail::ddm_exact_band_indices(code, layout.num_doppler);
      width = detail::ddm_num_doppler_per_tx(layout.num_doppler, code.num_tx,
                                             code.guard_bands);
      shape = detail::ddm_output_rd_shape(layout, code.num_tx, width);
    }
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }
  if (maybe_velocity_bins.has_value()) {
    try {
      require_cuda_accessible(
          *maybe_velocity_bins, "velocity_bins", kOpName);
      require_contiguous(*maybe_velocity_bins, "velocity_bins", kOpName);
      require_dtype(
          *maybe_velocity_bins, real_dtype<T>(), "velocity_bins", kOpName);
      if (maybe_velocity_bins->shape().size() != 1U ||
          maybe_velocity_bins->shape()[0] != layout.num_doppler) {
        return fail_invalid(
            std::string(kOpName) +
            ": velocity_bins must have shape (num_doppler,) matching "
            "rd_maps");
      }
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
  }
  cudaStream_t stream = context.cuda_stream().get();
  if (!stream)
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              std::string(kOpName) + ": no CUDA compute stream"});
  HOLOSCAN_TRY(
      auto rd_loan,
      mode == detail::DdmCodeMode::kNonLinear
          ? prepare_correlate(context, layout, code, shape, input, stream)
          : prepare_demux(
                context, layout, code, width, shape, input, stream));

  // The optional physical axis is a causal companion to rd_maps_out.
  // Prepare and publish it first so a downstream Exact join sees the value
  // when rd_maps_out triggers its consumer.
  if (maybe_velocity_bins.has_value() && velocity_bins_out_.active()) {
    const int64_t num_doppler_out =
        mode == detail::DdmCodeMode::kNonLinear
            ? layout.num_doppler
            : width;
    HOLOSCAN_TRY(
        auto axis_loan,
        prepare_device_tensor(
            velocity_bins_out, context,
            std::vector<int64_t>{num_doppler_out}, real_dtype<T>(),
            [&](void* output, std::size_t) -> expected<void, Error> {
              try {
                detail::run_ddm_velocity_axis_stage<T>(
                    maybe_velocity_bins->data(), output,
                    layout.num_doppler, num_doppler_out, stream);
              } catch (const std::invalid_argument& e) {
                return fail_invalid(e.what());
              } catch (const std::runtime_error& e) {
                return make_unexpected(
                    Error{ErrorCode::kFailure, e.what()});
              }
              return {};
            }));
    velocity_bins_out_.emit(
        velocity_bins_out, context, std::move(axis_loan),
        "DDMDenseDemultiplexOp.velocity_bins_out");
  }

  HOLOSCAN_TRY(
      emit_prepared_tensor(rd_maps_out, context, std::move(rd_loan)));
  retained_input_ = std::move(input);
  if (maybe_velocity_bins.has_value()) {
    retained_velocity_bins_ = std::move(*maybe_velocity_bins);
  } else {
    retained_velocity_bins_ = {};
  }
  return {};
}
template class DDMDenseDemultiplexOp<float>;
template class DDMDenseDemultiplexOp<double>;
}  // namespace holoscan::radar
