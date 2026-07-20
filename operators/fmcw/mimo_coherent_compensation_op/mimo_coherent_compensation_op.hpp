// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/sdk/config_cache.hpp"

namespace holoscan::radar {

// MIMO marshaling: per-Doppler-bin inter-TX phase correction (TDM) or Doppler-
// band demux + code removal (DDM). Contract reference: ports, shapes, dtypes,
// parameters.
//
// TDM in:  rd_maps — complex64/complex128, (..., M_tx, N_rx, num_doppler,
//          num_range_bins).
// TDM out: rd_maps — same shape; phase-aligned (tx, rx) block.
//
// DDM in:  rd_maps — complex64/complex128, (..., N_rx, num_doppler,
//          num_range_bins); the M TX sit in distinct Doppler sub-bands.
// DDM out: rd_maps — (..., M_tx, N_rx, num_doppler_per_tx, num_range_bins);
//          each TX demuxed out of its sub-band into an explicit axis, all lanes
//          aligned to a common reduced-Doppler axis. Standard linear-phase DDM
//          only (doppler_offsets, or a linear-phase phase_codes reduced to
//          offsets); the demux is a pure Doppler-bin gather and is
//          waveform-independent (waveform is carried for contract parity and
//          the deferred arbitrary-code path). See OPEN_ITEMS.
template <typename T>
class MIMOCoherentCompensationOp : public holoscan::Operator {
 public:
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "MIMOCoherentCompensationOp supports float and double only");

  HOLOSCAN_OPERATOR_FORWARD_ARGS(MIMOCoherentCompensationOp)

  MIMOCoherentCompensationOp() = default;
  ~MIMOCoherentCompensationOp() override;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  struct CachedTdmCorrection {
    std::shared_ptr<holoscan::Tensor> device_table;
  };

  struct CachedDdmDemux {
    std::shared_ptr<holoscan::Tensor> device_table;
  };

  void run_tdm_compute(OutputContext& op_output,
                       const detail::TdmRdMapsLayout& layout,
                       const std::shared_ptr<holoscan::Tensor>& input,
                       cudaStream_t stream, bool waveform_changed);

  void run_ddm_compute(OutputContext& op_output,
                       const detail::DdmRdMapsLayout& layout,
                       const std::shared_ptr<holoscan::Tensor>& input,
                       cudaStream_t stream);

  Parameter<MultiplexingScheme> multiplexing_scheme_;
  Parameter<FMCWWaveformDescriptor> waveform_;
  Parameter<DDMCodeDescriptor> ddm_code_;

  FMCWWaveformDescriptor effective_waveform_{};
  ConfigCache<CachedTdmCorrection> correction_cache_{};
  ConfigCache<CachedDdmDemux> demux_cache_{};

  std::shared_ptr<holoscan::Tensor> retained_input_;
};

}  // namespace holoscan::radar
