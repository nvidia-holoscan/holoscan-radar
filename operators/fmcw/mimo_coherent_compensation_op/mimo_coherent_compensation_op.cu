// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_op.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "radar_common/sdk/arg_setters.hpp"
#include "radar_common/sdk/override_param.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar {

namespace {

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
MIMOCoherentCompensationOp<T>::~MIMOCoherentCompensationOp() = default;

template <typename T>
void MIMOCoherentCompensationOp<T>::initialize() {
  ensure_radar_arg_setters();
  Operator::initialize();
  effective_waveform_ = waveform_.get();
}

template <typename T>
void MIMOCoherentCompensationOp<T>::setup(OperatorSpec& spec) {
  spec.input<holoscan::TensorMap>("rd_maps");
  spec.output<holoscan::TensorMap>("rd_maps");

  spec.param(multiplexing_scheme_, "multiplexing_scheme",
             "MIMO multiplexing scheme",
             "TDM: per-Doppler-bin inter-TX phase correction. DDM: band demux "
             "+ code removal. Static.",
             MultiplexingScheme::kTdm);
  spec.param(waveform_, "waveform", "FMCW waveform descriptor",
             "Per-TX PRI and carrier for Doppler-dependent phase math.",
             FMCWWaveformDescriptor{});
  add_override_port<FMCWWaveformDescriptor>(spec, "waveform");
  spec.param(ddm_code_, "ddm_code", "DDM code descriptor",
             "Doppler-band / phase-code layout for DDM mode. Unused for TDM.",
             DDMCodeDescriptor{});
}

template <typename T>
void MIMOCoherentCompensationOp<T>::run_tdm_compute(
    OutputContext& op_output, const detail::TdmRdMapsLayout& layout,
    const std::shared_ptr<holoscan::Tensor>& input, cudaStream_t stream,
    bool waveform_changed) {
  if (!effective_waveform_.is_valid()) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: waveform must be a valid "
        "FMCWWaveformDescriptor (build-time or [live] override)");
  }

  const std::string cache_key = detail::tdm_correction_cache_key(
      effective_waveform_, layout.num_tx, layout.num_doppler);
  const CachedTdmCorrection& cached = correction_cache_.get_or_build(
      cache_key, waveform_changed, "TDM correction table", [&] {
        CachedTdmCorrection entry;
        const auto host = detail::build_tdm_correction_table(
            effective_waveform_, layout.num_tx, layout.num_doppler);
        const auto typed = cast_correction<T>(host);
        entry.device_table = make_device_tensor(
            {layout.num_tx, layout.num_doppler}, complex_dtype<T>());
        if (cudaMemcpyAsync(entry.device_table->data(), typed.data(),
                            typed.size() * sizeof(std::complex<T>),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
          throw std::runtime_error(
              "MIMOCoherentCompensationOp: correction-table H2D failed");
        }
        return entry;
      });

  auto output = make_device_tensor(detail::tdm_output_rd_shape(layout),
                                   input->dtype());
  detail::run_tdm_compensation_stage<T>(
      input->data(), output->data(), layout, cached.device_table->data(),
      stream);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "rd_maps");
  }
  holoscan::TensorMap out_map;
  out_map.insert({"rd_maps", output});
  op_output.emit(out_map, "rd_maps");
}

template <typename T>
void MIMOCoherentCompensationOp<T>::run_ddm_compute(
    OutputContext& op_output, const detail::DdmRdMapsLayout& layout,
    const std::shared_ptr<holoscan::Tensor>& input, cudaStream_t stream) {
  // waveform is a required parameter on both schemes (contract parity). The
  // linear-phase DDM demux does not use it, but we still require a valid
  // descriptor so the contract "required, no default" holds.
  if (!effective_waveform_.is_valid()) {
    throw std::invalid_argument(
        "MIMOCoherentCompensationOp: waveform must be a valid "
        "FMCWWaveformDescriptor (build-time or [live] override)");
  }
  const DDMCodeDescriptor& code = ddm_code_.get();
  // Resolve + validate the code up front (offsets, or linear-phase codes;
  // arbitrary codes throw here) so errors surface before any allocation.
  const std::vector<double> offsets = detail::ddm_resolve_offsets(code);
  const int64_t num_tx = code.num_tx;
  const int64_t num_doppler_per_tx = detail::ddm_num_doppler_per_tx(
      layout.num_doppler, num_tx, code.guard_bands);

  const std::string cache_key =
      detail::ddm_demux_cache_key(code, layout.num_doppler);
  const CachedDdmDemux& cached = demux_cache_.get_or_build(
      cache_key, /*changed=*/false, "DDM demux table", [&] {
        CachedDdmDemux entry;
        entry.num_tx = num_tx;
        entry.num_doppler_per_tx = num_doppler_per_tx;
        const auto host = detail::build_ddm_demux_table(
            offsets, num_tx, layout.num_doppler, num_doppler_per_tx,
            code.guard_bands);
        entry.device_table =
            make_device_tensor({num_tx, num_doppler_per_tx},
                               DLDataType{kDLInt, 64, 1});
        if (cudaMemcpyAsync(entry.device_table->data(), host.data(),
                            host.size() * sizeof(int64_t),
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          throw std::runtime_error(
              "MIMOCoherentCompensationOp: demux-table H2D failed");
        }
        return entry;
      });

  auto output = make_device_tensor(
      detail::ddm_output_rd_shape(layout, num_tx, num_doppler_per_tx),
      input->dtype());
  detail::run_ddm_demux_stage<T>(
      input->data(), output->data(), layout, num_tx, num_doppler_per_tx,
      static_cast<const int64_t*>(cached.device_table->data()), stream);

  if (stream != cudaStreamDefault) {
    op_output.set_cuda_stream(stream, "rd_maps");
  }
  holoscan::TensorMap out_map;
  out_map.insert({"rd_maps", output});
  op_output.emit(out_map, "rd_maps");
}

template <typename T>
void MIMOCoherentCompensationOp<T>::compute(InputContext& op_input,
                                          OutputContext& op_output,
                                          ExecutionContext& context) {
  (void)context;
  constexpr const char* kOpName = "MIMOCoherentCompensationOp";

  std::shared_ptr<holoscan::Tensor> input =
      receive_single(op_input, "rd_maps", kOpName);

  classify_input_dtype<T>(*input, "rd_maps", /*allow_real=*/false);
  require_cuda_accessible(*input, "rd_maps", kOpName);
  require_contiguous(*input, "rd_maps", kOpName);

  // waveform is a [live] override on both schemes; sticky per invariant #5.
  // (For linear-phase DDM the demux gather is waveform-independent, but the
  // override is still accepted and validated for contract parity.)
  bool waveform_changed = false;
  effective_waveform_ =
      resolve_override_param(op_input, "waveform", effective_waveform_,
                             &waveform_changed);

  cudaStream_t stream = op_input.receive_cuda_stream("rd_maps");

  if (multiplexing_scheme_.get() == MultiplexingScheme::kDdm) {
    const detail::DdmRdMapsLayout layout =
        detail::parse_ddm_rd_maps_layout(*input, "rd_maps");
    run_ddm_compute(op_output, layout, input, stream);
  } else {
    const detail::TdmRdMapsLayout layout =
        detail::parse_tdm_rd_maps_layout(*input, "rd_maps");
    run_tdm_compute(op_output, layout, input, stream, waveform_changed);
  }

  retained_input_ = std::move(input);
}

template class MIMOCoherentCompensationOp<float>;
template class MIMOCoherentCompensationOp<double>;

}  // namespace holoscan::radar
