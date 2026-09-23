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
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_schemas/fmcw_waveform_bridge.hpp"

namespace holoscan::radar {
struct TDMDenseCompensationParams {
  FMCWWaveformDescriptor waveform{};
  TXSchedule tx_schedule{};
};
HOLOSCAN_PARAM_SCHEMA(
    TDMDenseCompensationParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.TDMDenseCompensationParams", 1),
    holoscan::field<&TDMDenseCompensationParams::waveform>("waveform")
        .description("Per-TX PRI for Doppler-dependent inter-TX phase "
                     "correction. [live]."),
    holoscan::field<&TDMDenseCompensationParams::tx_schedule>("tx_schedule")
        .description("Static TDM firing cycle. Empty = identity round-robin."));
static_assert(
    holoscan::param_schema_availability<TDMDenseCompensationParams>() ==
    holoscan::ParamSchemaAvailability::kDeclared);
template <typename T>
class TDMDenseCompensationOp
    : public holoscan::Operator<TDMDenseCompensationParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit TDMDenseCompensationOp(const MaxTensorExtents& extents);
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;
  Input<Tensor> rd_maps{"rd_maps"};
  Output<Tensor> rd_maps_out{"rd_maps_out"};
  Input<wire::FMCWWaveformT> waveform_in{"waveform"};

 private:
  struct CachedCorrection {
    std::shared_ptr<Tensor> device_table;
  };
  expected<void, Error> run_compute(ExecutionContext& context,
                                    const detail::TdmRdMapsLayout& layout,
                                    const FMCWWaveformDescriptor& waveform,
                                    const std::vector<int64_t>& slot_by_tx,
                                    const std::vector<int64_t>& out_shape,
                                    const Tensor& input, cudaStream_t stream);
  Tensor retained_input_;
  ConfigCache<CachedCorrection> correction_cache_{};
  MaxTensorExtents extents_;
  std::uint32_t max_rank_{4U};
  std::size_t max_byte_span_{0U};
};
extern template class TDMDenseCompensationOp<float>;
extern template class TDMDenseCompensationOp<double>;
}  // namespace holoscan::radar
