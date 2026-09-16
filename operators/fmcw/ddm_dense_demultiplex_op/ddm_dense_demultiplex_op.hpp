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

#include "radar_common/descriptors/ddm_code_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/sdk/config_cache.hpp"
#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar {
struct DDMDenseDemultiplexParams {
  DDMCodeDescriptor ddm_code{};
};
HOLOSCAN_PARAM_SCHEMA(
    DDMDenseDemultiplexParams,
    holoscan::param_schema(
        "com.nvidia.holoscan.radar.DDMDenseDemultiplexParams", 1),
    holoscan::field<&DDMDenseDemultiplexParams::ddm_code>("ddm_code")
        .description("Static DDM Doppler-band or phase-code layout."));
static_assert(
    holoscan::param_schema_availability<DDMDenseDemultiplexParams>() ==
    holoscan::ParamSchemaAvailability::kDeclared);
template <typename T>
class DDMDenseDemultiplexOp
    : public holoscan::Operator<DDMDenseDemultiplexParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

 public:
  explicit DDMDenseDemultiplexOp(const MaxTensorExtents& extents);
  ~DDMDenseDemultiplexOp() override;
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(
      ExecutionContext& context) override;
  Input<Tensor> rd_maps{"rd_maps"};
  Input<Tensor> velocity_bins{"velocity_bins"};
  Output<Tensor> rd_maps_out{"rd_maps_out"};
  Output<Tensor> velocity_bins_out{"velocity_bins_out"};

 private:
  struct CachedDemux {
    std::shared_ptr<Tensor> device_table;
    std::shared_ptr<Tensor> device_initial_phase_weights;
  };
  struct CachedCorrelate {
    std::shared_ptr<Tensor> device_corr_weights;
  };
  expected<TensorOutputLoan, Error> prepare_demux(
      ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
      const DDMCodeDescriptor& code, int64_t num_doppler_per_tx,
      const std::vector<int64_t>& out_shape, const Tensor& input,
      cudaStream_t stream);
  expected<TensorOutputLoan, Error> prepare_correlate(
      ExecutionContext& context, const detail::DdmRdMapsLayout& layout,
      const DDMCodeDescriptor& code, const std::vector<int64_t>& out_shape,
      const Tensor& input, cudaStream_t stream);
  Tensor retained_input_;
  Tensor retained_velocity_bins_;
  ConfigCache<CachedDemux> demux_cache_{};
  ConfigCache<CachedCorrelate> correlate_cache_{};
  LatchedOutput velocity_bins_out_{};
  MaxTensorExtents extents_;
  std::uint32_t max_rank_{4U};
  std::size_t max_byte_span_{0U};
  std::size_t velocity_bins_max_byte_span_{0U};
};
extern template class DDMDenseDemultiplexOp<float>;
extern template class DDMDenseDemultiplexOp<double>;
}  // namespace holoscan::radar
