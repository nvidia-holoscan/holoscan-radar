// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/payload_options.hpp>

#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/extents_layout.hpp"
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

template <typename T>
std::vector<std::complex<T>> cast_correction(
    const std::vector<std::complex<double>>& host) {
  std::vector<std::complex<T>> out(host.size());
  for (size_t i = 0; i < host.size(); ++i) {
    out[i] = std::complex<T>(static_cast<T>(host[i].real()),
                             static_cast<T>(host[i].imag()));
  }
  return out;
}

}  // namespace

template <typename T>
MIMOCoherentCompensationOp<T>::MIMOCoherentCompensationOp(
    const MaxTensorExtents& extents) {
  const char* who = "MIMOCoherentCompensationOp";
  const std::int64_t max_tx =
      require_extent(extents.max_tx, "max_tx", who);
  const std::int64_t max_rx =
      require_extent(extents.max_rx, "max_rx", who);
  const std::int64_t max_doppler =
      require_extent(extents.max_doppler_bins,
                     "max_doppler_bins", who);
  const std::int64_t max_range =
      require_extent(extents.max_range_bins,
                     "max_range_bins", who);
  if (extents.max_samples_per_chirp.has_value() &&
      *extents.max_samples_per_chirp <= 0) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: max_samples_per_chirp must be "
        "positive when set");
  }
  extents_ = extents;
  max_rank_ = 4U;
  max_byte_span_ = byte_span_complex_cube<T>(
      {max_tx, max_rx, max_doppler, max_range}, who);
}

template <typename T>
MIMOCoherentCompensationOp<T>::~MIMOCoherentCompensationOp() {
  detail::release_ddm_correlate_scratch(this);
}

template <typename T>
void MIMOCoherentCompensationOp<T>::setup(OperatorSpec& spec) {
  constexpr TensorRepresentation kDevice{
      .memory_kind = MemoryKind::kCudaDevice,
      .dtype = std::nullopt,
      .rank = std::nullopt,
  };
  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.input(waveform_in).queue_depth(2U);
  spec.output(rd_maps_out)
      .produces_tensor(pooled_tensor_output(kDevice, max_rank_, max_byte_span_));
}

template <typename T>
Contract MIMOCoherentCompensationOp<T>::contract() const {
  Contract result;
  result.trigger(OnEach{rd_maps});
  add_live_override_to_contract(result, waveform_in);
  return result;
}

template <typename T>
expected<void, Error> MIMOCoherentCompensationOp<T>::run_tdm_compute(
    ExecutionContext& context, const detail::TdmRdMapsLayout& layout,
    const Tensor& input, cudaStream_t stream) {
  if (!effective_waveform_.is_valid()) {
    return fail_invalid(
        "MIMOCoherentCompensationOp: waveform must be a valid "
        "FMCWWaveformDescriptor (build-time or [live] override)");
  }

  const std::string cache_key = detail::tdm_correction_cache_key(
      effective_waveform_, layout.num_tx, layout.num_doppler);
  const CachedTdmCorrection& cached = correction_cache_.get_or_build(
      cache_key, "TDM correction table", [&] {
        CachedTdmCorrection entry;
        const auto host = detail::build_tdm_correction_table(
            effective_waveform_, layout.num_tx, layout.num_doppler);
        const auto typed = cast_correction<T>(host);
        entry.device_table = make_device_tensor(
            {layout.num_tx, layout.num_doppler}, complex_dtype<T>());
        auto mut = entry.device_table->mutable_data();
        if (!mut) {
          throw std::runtime_error(
              "MIMOCoherentCompensationOp: correction-table not writable");
        }
        if (cudaMemcpyAsync(*mut, typed.data(),
                            typed.size() * sizeof(std::complex<T>),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
          throw std::runtime_error(
              "MIMOCoherentCompensationOp: correction-table H2D failed");
        }
        return entry;
      });

  const std::vector<int64_t> out_shape =
      detail::tdm_output_rd_shape(layout);

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(emit_device_tensor(
      rd_maps_out, context, out_shape, input.dtype(),
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          detail::run_tdm_compensation_stage<T>(
              input_ptr, out_ptr, layout,
              cached.device_table->data(), stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      }));

  return {};
}

template <typename T>
expected<void, Error> MIMOCoherentCompensationOp<T>::run_ddm_compute(
    ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
    const Tensor& input, cudaStream_t stream) {
  if (!effective_waveform_.is_valid()) {
    return fail_invalid(
        "MIMOCoherentCompensationOp: waveform must be a valid "
        "FMCWWaveformDescriptor (build-time or [live] override)");
  }

  const MIMOCoherentCompensationParams& pinned =
      context.params<MIMOCoherentCompensationParams>();
  const DDMCodeDescriptor& code = pinned.ddm_code;
  const auto mode = detail::ddm_classify_codes(code);

  if (mode == detail::DdmCodeMode::kNonLinear) {
    return run_ddm_correlate(context, layout, input, stream);
  }

  const std::vector<double> offsets = detail::ddm_resolve_offsets(code);
  const int64_t num_tx = code.num_tx;
  int64_t num_doppler_per_tx = 0;
  try {
    num_doppler_per_tx = detail::ddm_num_doppler_per_tx(
        layout.num_doppler, num_tx, code.guard_bands);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const std::string cache_key =
      detail::ddm_demux_cache_key(code, layout.num_doppler);
  const CachedDdmDemux& cached = demux_cache_.get_or_build(
      cache_key, "DDM demux table", [&] {
        CachedDdmDemux entry;
        const auto host = detail::build_ddm_demux_table(
            offsets, num_tx, layout.num_doppler, num_doppler_per_tx);
        entry.device_table =
            make_device_tensor({num_tx, num_doppler_per_tx},
                               DLDataType{kDLInt, 64, 1});
        auto mut = entry.device_table->mutable_data();
        if (!mut) {
          throw std::runtime_error(
              "MIMOCoherentCompensationOp: demux-table not writable");
        }
        if (cudaMemcpyAsync(*mut, host.data(),
                            host.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              "MIMOCoherentCompensationOp: demux-table H2D failed");
        }
        return entry;
      });

  const std::vector<int64_t> out_shape =
      detail::ddm_output_rd_shape(layout, num_tx, num_doppler_per_tx);

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(emit_device_tensor(
      rd_maps_out, context, out_shape, input.dtype(),
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          detail::run_ddm_demux_stage<T>(
              input_ptr, out_ptr, layout, num_tx, num_doppler_per_tx,
              static_cast<const int64_t*>(cached.device_table->data()),
              stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      }));

  return {};
}

template <typename T>
expected<void, Error> MIMOCoherentCompensationOp<T>::run_ddm_correlate(
    ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
    const Tensor& input, cudaStream_t stream) {
  const MIMOCoherentCompensationParams& pinned =
      context.params<MIMOCoherentCompensationParams>();
  const DDMCodeDescriptor& code = pinned.ddm_code;
  const int64_t num_tx = code.num_tx;
  const int64_t num_chirps =
      static_cast<int64_t>(code.phase_codes.size()) / num_tx;

  if (code.guard_bands != 0) {
    return fail_invalid(
        "MIMOCoherentCompensationOp: non-linear DDM phase_codes require "
        "guard_bands = 0 (got " +
        std::to_string(code.guard_bands) + ")");
  }
  if (num_chirps != layout.num_doppler) {
    return fail_invalid(
        "MIMOCoherentCompensationOp: non-linear DDM phase_codes imply "
        "num_chirps = " +
        std::to_string(num_chirps) +
        " but input num_doppler = " + std::to_string(layout.num_doppler) +
        "; these must match for the correlate path");
  }

  const std::string cache_key =
      detail::ddm_correlate_cache_key(code, layout.num_doppler);
  const CachedDdmCorrelate& cached = correlate_cache_.get_or_build(
      cache_key, "DDM correlate FFT weights", [&] {
        CachedDdmCorrelate entry;
        const auto host_weights =
            detail::build_ddm_corr_weights(code, layout.num_doppler);
        const int64_t D = layout.num_doppler;
        entry.device_corr_weights =
            make_device_tensor({num_tx, D}, complex_dtype<T>());
        auto mut = entry.device_corr_weights->mutable_data();
        if (!mut) {
          throw std::runtime_error(
              "MIMOCoherentCompensationOp: corr-weights not writable");
        }
        if constexpr (sizeof(T) == 8) {
          if (cudaMemcpyAsync(
                  *mut, host_weights.data(),
                  host_weights.size() * sizeof(std::complex<double>),
                  cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error(
                "MIMOCoherentCompensationOp: corr-weights H2D failed");
          }
        } else {
          std::vector<std::complex<float>> typed(host_weights.size());
          for (size_t i = 0; i < host_weights.size(); ++i) {
            typed[i] = std::complex<float>(
                static_cast<float>(host_weights[i].real()),
                static_cast<float>(host_weights[i].imag()));
          }
          if (cudaMemcpyAsync(*mut, typed.data(),
                              typed.size() * sizeof(std::complex<float>),
                              cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            throw std::runtime_error(
                "MIMOCoherentCompensationOp: corr-weights H2D failed");
          }
        }
        return entry;
      });

  const std::vector<int64_t> out_shape =
      detail::ddm_correlate_output_shape(layout, num_tx);

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(emit_device_tensor(
      rd_maps_out, context, out_shape, input.dtype(),
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          detail::run_ddm_correlate_stage<T>(
              input_ptr, out_ptr, layout, num_tx,
              cached.device_corr_weights->data(), this, stream);
        } catch (const std::invalid_argument& e) {
          return fail_invalid(e.what());
        }
        return {};
      }));

  return {};
}

template <typename T>
expected<void, Error> MIMOCoherentCompensationOp<T>::compute(
    ExecutionContext& context) {
  constexpr const char* kOpName = "MIMOCoherentCompensationOp";

  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));

  try {
    classify_input_dtype<T>(input, "rd_maps", /*allow_real=*/false);
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  const MIMOCoherentCompensationParams& pinned =
      context.params<MIMOCoherentCompensationParams>();

  // Convert only a matched wire sample. A changed waveform yields a new
  // correction-table cache key, so
  // ConfigCache handles the warn-once itself.
  HOLOSCAN_TRY(
      effective_waveform_,
      resolve_live_override_via(
          waveform_in, pinned.waveform,
          [](const wire::FMCWWaveformT& value) { return from_wire(value); }));
  if (extents_.max_samples_per_chirp.has_value() &&
      effective_waveform_.reference_samples().size() >
          static_cast<std::size_t>(*extents_.max_samples_per_chirp)) {
    return fail_invalid(
        std::string(kOpName) + ": waveform reference_samples length " +
        std::to_string(effective_waveform_.reference_samples().size()) +
        " exceeds max_samples_per_chirp (" +
        std::to_string(*extents_.max_samples_per_chirp) + ")");
  }

  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "MIMOCoherentCompensationOp: no CUDA compute stream"});
  }

  const MultiplexingScheme scheme = pinned.multiplexing_scheme;
  if (scheme == MultiplexingScheme::kUnspecified) {
    return fail_invalid(
        std::string(kOpName) +
        ": multiplexing_scheme is required (tdm or ddm)");
  }

  if (scheme == MultiplexingScheme::kDdm) {
    detail::DdmRdMapsLayout layout;
    try {
      layout = detail::parse_ddm_rd_maps_layout(input, "rd_maps");
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    HOLOSCAN_TRY(run_ddm_compute(context, layout, input, stream));
  } else if (scheme == MultiplexingScheme::kTdm) {
    detail::TdmRdMapsLayout layout;
    try {
      layout = detail::parse_tdm_rd_maps_layout(input, "rd_maps");
    } catch (const std::invalid_argument& e) {
      return fail_invalid(e.what());
    }
    HOLOSCAN_TRY(run_tdm_compute(context, layout, input, stream));
  } else {
    return fail_invalid(
        std::string(kOpName) + ": unknown multiplexing_scheme");
  }

  retained_input_ = std::move(input);
  return {};
}

template class MIMOCoherentCompensationOp<float>;
template class MIMOCoherentCompensationOp<double>;

}  // namespace holoscan::radar
