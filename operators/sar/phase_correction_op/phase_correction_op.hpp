// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_codecs.hpp"

namespace holoscan::radar {

// Static correction sense; no implicit sign is safe for autofocus.
struct PhaseCorrectionParams {
  PhaseCorrectionSense sense{PhaseCorrectionSense::kUnspecified};
};

HOLOSCAN_PARAM_SCHEMA(PhaseCorrectionParams,
                      holoscan::param_schema("com.nvidia.holoscan.radar.PhaseCorrectionParams", 3),
                      holoscan::field<&PhaseCorrectionParams::sense>("sense").description(
                          "subtract multiplies by exp(-i phi); add by exp(+i phi). Required."));

template <typename T>
class PhaseCorrectionOp : public holoscan::Operator<PhaseCorrectionParams> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "PhaseCorrectionOp supports float and double precision");

 public:
  explicit PhaseCorrectionOp(const MaxTensorExtents& extents);

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> range_profiles_in{"range_profiles"};
  Input<Tensor> phase_correction{"phase_correction"};
  Output<Tensor> range_profiles_out{"range_profiles_out"};

 private:
  Tensor retained_profiles_;
  Tensor retained_phase_;
  MaxTensorExtents extents_;
  std::size_t max_byte_span_{0U};
};

extern template class PhaseCorrectionOp<float>;
extern template class PhaseCorrectionOp<double>;

}  // namespace holoscan::radar
