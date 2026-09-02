// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
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
#include "radar_common/tensor/max_byte_span.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

expected<void, Error> fail_invalid(const std::string& message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, message});
}

constexpr const char* kOpName = "MIMOCoherentCompensationOp";

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

// Stage config plus this tick's parsed layout and emit geometry.
// TDM trailing rank is (M_tx, N_rx, D, R); DDM is (N_rx, D, R). Neither
// is an FMCW (chirps, bins) cube, so this does not call
// collapse_named_tx_rx_batch.
struct CompensationStagePlan {
  detail::CompensationStageConfig stage;
  std::vector<int64_t> out_shape;
  std::optional<detail::TdmRdMapsLayout> tdm;
  std::optional<detail::DdmRdMapsLayout> ddm;
  std::vector<int64_t> tdm_slot_by_tx;
  detail::DdmCodeMode ddm_mode{detail::DdmCodeMode::kLinear};
  int64_t num_doppler_per_tx{0};
};

// Merge the Exact [live] waveform onto pinned Params. Map the empty
// descriptor sentinel by rejecting it (waveform is required on every
// path). multiplexing_scheme, ddm_code, and tx_schedule are Params-only.
expected<detail::CompensationStageConfig, Error> resolve_params(
    Input<wire::FMCWWaveformT>& waveform_in,
    const MIMOCoherentCompensationParams& pinned) {
  HOLOSCAN_TRY(
      auto waveform,
      resolve_live_override_via(
          waveform_in, pinned.waveform,
          [](const wire::FMCWWaveformT& value) { return from_wire(value); }));
  if (!waveform.is_valid()) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": waveform must be a valid FMCWWaveformDescriptor "
            "(build-time or [live] override)"});
  }

  const MultiplexingScheme scheme = pinned.multiplexing_scheme;
  if (scheme == MultiplexingScheme::kUnspecified) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) +
            ": multiplexing_scheme is required (tdm or ddm)"});
  }
  if (scheme != MultiplexingScheme::kTdm &&
      scheme != MultiplexingScheme::kDdm) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": unknown multiplexing_scheme"});
  }

  return detail::CompensationStageConfig{
      .multiplexing_scheme = scheme,
      .waveform = std::move(waveform),
      .ddm_code = pinned.ddm_code,
      .tx_schedule = pinned.tx_schedule};
}

// Bind the stage config to this input: parse TDM/DDM layout, size the
// output, and reject illegal DDM code / chirp combinations.
expected<CompensationStagePlan, Error> build_compensation_stage_plan(
    detail::CompensationStageConfig stage, const Tensor& input,
    const MaxTensorExtents& extents) {
  if (extents.max_samples_per_chirp.has_value() &&
      stage.waveform.reference_samples().size() >
          static_cast<std::size_t>(*extents.max_samples_per_chirp)) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        std::string(kOpName) + ": waveform reference_samples length " +
            std::to_string(stage.waveform.reference_samples().size()) +
            " exceeds max_samples_per_chirp (" +
            std::to_string(*extents.max_samples_per_chirp) + ")"});
  }

  CompensationStagePlan plan;
  plan.stage = std::move(stage);

  try {
    if (plan.stage.multiplexing_scheme == MultiplexingScheme::kTdm) {
      plan.tdm = detail::parse_tdm_rd_maps_layout(input, "rd_maps");
      plan.out_shape = detail::tdm_output_rd_shape(*plan.tdm);
      plan.tdm_slot_by_tx = detail::tdm_firing_slots(
          plan.stage.tx_schedule, plan.tdm->num_tx);
      return plan;
    }

    plan.ddm = detail::parse_ddm_rd_maps_layout(input, "rd_maps");
    const DDMCodeDescriptor& code = plan.stage.ddm_code;
    plan.ddm_mode = detail::ddm_classify_codes(code);
    if (plan.ddm_mode == detail::DdmCodeMode::kNonLinear) {
      if (code.guard_bands != 0) {
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument,
            std::string(kOpName) +
                ": non-linear DDM phase_codes require guard_bands = 0 "
                "(got " +
                std::to_string(code.guard_bands) + ")"});
      }
      const int64_t num_chirps =
          static_cast<int64_t>(code.phase_codes.size()) / code.num_tx;
      if (num_chirps != plan.ddm->num_doppler) {
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument,
            std::string(kOpName) +
                ": non-linear DDM phase_codes imply num_chirps = " +
                std::to_string(num_chirps) +
                " but input num_doppler = " +
                std::to_string(plan.ddm->num_doppler) +
                "; these must match for the correlate path"});
      }
      plan.out_shape =
          detail::ddm_correlate_output_shape(*plan.ddm, code.num_tx);
    } else {
      // Linear path: reject empty code (validate allows both-empty when
      // num_tx > 0) and size the reduced Doppler axis.
      (void)detail::ddm_resolve_offsets(code);
      plan.num_doppler_per_tx = detail::ddm_num_doppler_per_tx(
          plan.ddm->num_doppler, code.num_tx, code.guard_bands);
      plan.out_shape = detail::ddm_output_rd_shape(
          *plan.ddm, code.num_tx, plan.num_doppler_per_tx);
    }
  } catch (const std::invalid_argument& e) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, e.what()});
  }
  return plan;
}

}  // namespace

template <typename T>
MIMOCoherentCompensationOp<T>::MIMOCoherentCompensationOp(
    const MaxTensorExtents& extents) {
  const char* who = kOpName;
  const std::int64_t max_tx = require_extent(extents.max_tx, "max_tx", who);
  const std::int64_t max_rx = require_extent(extents.max_rx, "max_rx", who);
  const std::int64_t max_doppler =
      require_extent(extents.max_doppler_bins, "max_doppler_bins", who);
  const std::int64_t max_range =
      require_extent(extents.max_range_bins, "max_range_bins", who);
  if (extents.max_samples_per_chirp.has_value() &&
      *extents.max_samples_per_chirp <= 0) {
    throw std::invalid_argument(
        std::string(kOpName) +
        ": max_samples_per_chirp must be positive when set");
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
  const TensorBounds fat_bounds{.max_rank = max_rank_,
                                .max_byte_span = max_byte_span_};

  spec.input(rd_maps).queue_depth(2U).expects_tensor(tensor_input(kDevice));
  spec.output(rd_maps_out)
      .produces_tensor(pooled_tensor_output(kDevice, fat_bounds));

  // [live] Exact overrides, Params order (schema and POD alike).
  spec.input(waveform_in).queue_depth(2U);
}

template <typename T>
Contract MIMOCoherentCompensationOp<T>::contract() const {
  // Trigger on the data tensor. The Exact override is declared whether or
  // not the application wires it: an unwired one compiles to a warning and
  // reads as a miss.
  Contract result;
  result.trigger(OnEach{rd_maps});
  add_live_override_to_contract(result, waveform_in);
  return result;
}

template <typename T>
expected<void, Error> MIMOCoherentCompensationOp<T>::run_tdm_compute(
    ExecutionContext& context, const detail::TdmRdMapsLayout& layout,
    const FMCWWaveformDescriptor& waveform,
    const std::vector<int64_t>& slot_by_tx,
    const std::vector<int64_t>& out_shape, const Tensor& input,
    cudaStream_t stream) {
  const std::string cache_key = detail::tdm_correction_cache_key(
      waveform, layout.num_tx, layout.num_doppler, slot_by_tx);
  const CachedTdmCorrection& cached = correction_cache_.get_or_build(
      cache_key, "TDM correction table", [&] {
        CachedTdmCorrection entry;
        const auto host = detail::build_tdm_correction_table(
            waveform, layout.num_tx, layout.num_doppler, slot_by_tx);
        const auto typed = cast_correction<T>(host);
        entry.device_table = make_device_tensor(
            {layout.num_tx, layout.num_doppler}, complex_dtype<T>());
        auto mut = entry.device_table->mutable_data();
        if (!mut) {
          throw std::runtime_error(
              std::string(kOpName) + ": correction-table not writable");
        }
        if (cudaMemcpyAsync(*mut, typed.data(),
                            typed.size() * sizeof(std::complex<T>),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
          throw std::runtime_error(
              std::string(kOpName) + ": correction-table H2D failed");
        }
        return entry;
      });

  const void* input_ptr = input.data();
  HOLOSCAN_TRY(emit_device_tensor(
      rd_maps_out, context, out_shape, input.dtype(),
      [&](void* out_ptr, std::size_t) -> expected<void, Error> {
        try {
          detail::run_tdm_compensation_stage<T>(
              input_ptr, out_ptr, layout, cached.device_table->data(),
              stream);
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
    const DDMCodeDescriptor& code, int64_t num_doppler_per_tx,
    const std::vector<int64_t>& out_shape, const Tensor& input,
    cudaStream_t stream) {
  const std::vector<double> offsets = detail::ddm_resolve_offsets(code);
  const int64_t num_tx = code.num_tx;

  const std::string cache_key =
      detail::ddm_demux_cache_key(code, layout.num_doppler);
  const CachedDdmDemux& cached = demux_cache_.get_or_build(
      cache_key, "DDM demux table", [&] {
        CachedDdmDemux entry;
        const auto host = detail::build_ddm_demux_table(
            offsets, num_tx, layout.num_doppler, num_doppler_per_tx);
        entry.device_table = make_device_tensor(
            {num_tx, num_doppler_per_tx}, DLDataType{kDLInt, 64, 1});
        auto mut = entry.device_table->mutable_data();
        if (!mut) {
          throw std::runtime_error(
              std::string(kOpName) + ": demux-table not writable");
        }
        if (cudaMemcpyAsync(*mut, host.data(),
                            host.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              std::string(kOpName) + ": demux-table H2D failed");
        }
        return entry;
      });

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
    const DDMCodeDescriptor& code, const std::vector<int64_t>& out_shape,
    const Tensor& input, cudaStream_t stream) {
  const int64_t num_tx = code.num_tx;

  const std::string cache_key =
      detail::ddm_correlate_cache_key(code, layout.num_doppler);
  const CachedDdmCorrelate& cached = correlate_cache_.get_or_build(
      cache_key, "DDM correlate FFT weights", [&] {
        CachedDdmCorrelate entry;
        const auto host_weights =
            detail::build_ddm_corr_weights(code, layout.num_doppler);
        const auto typed = cast_correction<T>(host_weights);
        const int64_t D = layout.num_doppler;
        entry.device_corr_weights =
            make_device_tensor({num_tx, D}, complex_dtype<T>());
        auto mut = entry.device_corr_weights->mutable_data();
        if (!mut) {
          throw std::runtime_error(
              std::string(kOpName) + ": corr-weights not writable");
        }
        if (cudaMemcpyAsync(*mut, typed.data(),
                            typed.size() * sizeof(std::complex<T>),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              std::string(kOpName) + ": corr-weights H2D failed");
        }
        return entry;
      });

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
  // Receive rd_maps; require a device-contiguous buffer.
  HOLOSCAN_TRY(auto input, receive_tensor(rd_maps, kOpName));
  try {
    require_cuda_accessible(input, "rd_maps", kOpName);
    require_contiguous(input, "rd_maps", kOpName);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // Complex-only input. Trailing rank is TDM/DDM-specific, not an FMCW
  // (chirps, bins) cube — layout parse lives in the plan builder.
  try {
    classify_input_dtype<T>(input, "rd_maps", /*allow_real=*/false);
  } catch (const std::invalid_argument& e) {
    return fail_invalid(e.what());
  }

  // This tick's Params: pinned defaults with any matched [live] override.
  const MIMOCoherentCompensationParams& pinned =
      context.params<MIMOCoherentCompensationParams>();
  HOLOSCAN_TRY(auto stage, resolve_params(waveform_in, pinned));

  // Bind the stage config to this input's scheme-specific layout.
  HOLOSCAN_TRY(auto plan,
               build_compensation_stage_plan(std::move(stage), input,
                                             extents_));

  // Kernels / cuFFT run on the runtime compute stream.
  cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              std::string(kOpName) + ": no CUDA compute stream"});
  }

  // Required rd_maps_out: TDM multiply, DDM gather, or DDM-NL correlate.
  if (plan.stage.multiplexing_scheme == MultiplexingScheme::kTdm) {
    HOLOSCAN_TRY(run_tdm_compute(context, *plan.tdm, plan.stage.waveform,
                                 plan.tdm_slot_by_tx, plan.out_shape, input,
                                 stream));
  } else if (plan.ddm_mode == detail::DdmCodeMode::kNonLinear) {
    HOLOSCAN_TRY(run_ddm_correlate(context, *plan.ddm, plan.stage.ddm_code,
                                   plan.out_shape, input, stream));
  } else {
    HOLOSCAN_TRY(run_ddm_compute(context, *plan.ddm, plan.stage.ddm_code,
                                 plan.num_doppler_per_tx, plan.out_shape,
                                 input, stream));
  }

  // Keep rd_maps alive until stream-ordered reads complete.
  retained_input_ = std::move(input);
  return {};
}

template class MIMOCoherentCompensationOp<float>;
template class MIMOCoherentCompensationOp<double>;

}  // namespace holoscan::radar
