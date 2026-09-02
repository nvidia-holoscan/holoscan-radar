// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "fmcw/mimo_coherent_compensation_op/mimo_coherent_compensation_stage.hpp"
#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"

namespace holoscan::radar {

// MIMO marshaling: per-Doppler-bin inter-TX phase correction (TDM) or Doppler-
// band demux + code removal (DDM). Contract reference: ports, shapes, dtypes,
// parameters.
//
// TDM in:  rd_maps — complex64/complex128, (..., M_tx, N_rx, num_doppler,
//          num_range_bins).
// TDM out: rd_maps_out — same shape; phase-aligned (tx, rx) block.
//          [EA HSDK5 ONLY] unique in/out port names required.
// TDM scope: the per-Doppler-bin correction uses a fixed PRI/M inter-TX offset
//          and indexes it by TX id — i.e. it assumes the **default identity
//          round-robin** `tx_order` (TX id == firing slot). A non-identity
//          `tx_order` permutation (still uniform-repeating, so accepted by
//          `MIMODeinterleaveOp`) is NOT compensated correctly: each lane would
//          need the correction for its actual firing slot, and this operator
//          does not receive `tx_order`. Known limitation — see
//          `OPEN_ITEMS.md` item 1.
//          Staggered / per-TX-timing schedules are separately out of scope
//          (STATUS.md → Roadmap; settled as L12).
//
// DDM in:  rd_maps — complex64/complex128, (..., N_rx, num_doppler,
//          num_range_bins); the M TX sit in distinct Doppler sub-bands.
// DDM out (linear-phase / offsets):
//          rd_maps_out — (..., M_tx, N_rx, num_doppler_per_tx,
//          num_range_bins); each TX demuxed out of its sub-band into an
//          explicit axis, all lanes aligned to a common reduced-Doppler axis.
// DDM out (non-linear phase_codes):
//          rd_maps_out — (..., M_tx, N_rx, num_doppler, num_range_bins);
//          full Doppler axis preserved; conjugate-correlation demux.
//
// Precision is fixed at compile time by the template instantiation.
// Fat output pool sizing requires MaxTensorExtents at graph.op() (ctor).

struct MIMOCoherentCompensationParams {
  MultiplexingScheme multiplexing_scheme{MultiplexingScheme::kUnspecified};
  FMCWWaveformDescriptor waveform{};
  DDMCodeDescriptor ddm_code{};
};

HOLOSCAN_PARAM_SCHEMA(
    MIMOCoherentCompensationParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.MIMOCoherentCompensationParams", 1),
    holoscan::field<&MIMOCoherentCompensationParams::multiplexing_scheme>(
        "multiplexing_scheme")
        .description(
            "MIMO multiplexing scheme {tdm, ddm}. TDM: per-Doppler-bin "
            "inter-TX phase correction. DDM: band demux + code removal. "
            "Static. Required."),
    holoscan::field<&MIMOCoherentCompensationParams::waveform>("waveform")
        .description(
            "Per-TX PRI and carrier for Doppler-dependent phase math. [live]."),
    holoscan::field<&MIMOCoherentCompensationParams::ddm_code>("ddm_code")
        .description(
            "Doppler-band / phase-code layout for DDM mode. Unused for "
            "TDM. Static."));

static_assert(
    holoscan::param_schema_availability<MIMOCoherentCompensationParams>() ==
        holoscan::ParamSchemaAvailability::kDeclared,
    "HOLOSCAN_PARAM_SCHEMA must be in "
    "MIMOCoherentCompensationParams's namespace");

template <typename T>
class MIMOCoherentCompensationOp
    : public holoscan::Operator<MIMOCoherentCompensationParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MIMOCoherentCompensationOp supports float and double only");

 public:
  // Required: sizes the rd_maps_out Tensor pool at setup().
  explicit MIMOCoherentCompensationOp(const MaxTensorExtents& extents);
  ~MIMOCoherentCompensationOp() override;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;

  Input<Tensor> rd_maps{"rd_maps"};
  // HSDK 5 requires unique port names within an operator (inputs and
  // outputs share one namespace). OPERATOR_API.md's logical output channel
  // remains "rd_maps"; the physical port name is rd_maps_out.
  // [EA HSDK5 ONLY] revisit naming once the full port is settled.
  Output<Tensor> rd_maps_out{"rd_maps_out"};

  // [live] Exact override (OPERATOR_API.md). Schema wire type; converted
  // back to FMCWWaveformDescriptor in resolve_params.
  Input<wire::FMCWWaveformT> waveform_in{"waveform"};

 private:
  struct CachedTdmCorrection {
    std::shared_ptr<holoscan::Tensor> device_table;
  };

  struct CachedDdmDemux {
    std::shared_ptr<holoscan::Tensor> device_table;
  };

  struct CachedDdmCorrelate {
    std::shared_ptr<holoscan::Tensor> device_corr_weights;
  };

  expected<void, Error> run_tdm_compute(
      ExecutionContext& context, const detail::TdmRdMapsLayout& layout,
      const FMCWWaveformDescriptor& waveform,
      const std::vector<int64_t>& out_shape, const Tensor& input,
      cudaStream_t stream);

  expected<void, Error> run_ddm_compute(
      ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
      const DDMCodeDescriptor& code, int64_t num_doppler_per_tx,
      const std::vector<int64_t>& out_shape, const Tensor& input,
      cudaStream_t stream);

  expected<void, Error> run_ddm_correlate(
      ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
      const DDMCodeDescriptor& code, const std::vector<int64_t>& out_shape,
      const Tensor& input, cudaStream_t stream);

  // Keep the previous activation's input alive across ticks so async
  // device reads cannot outlive the buffer.
  Tensor retained_input_;

  // No latched optional. Table caches are stage state (warn-once).
  ConfigCache<CachedTdmCorrection> correction_cache_{};
  ConfigCache<CachedDdmDemux> demux_cache_{};
  ConfigCache<CachedDdmCorrelate> correlate_cache_{};

  MaxTensorExtents extents_;
  std::uint32_t max_rank_{4U};
  std::size_t max_byte_span_{0U};
};

extern template class MIMOCoherentCompensationOp<float>;
extern template class MIMOCoherentCompensationOp<double>;

}  // namespace holoscan::radar
