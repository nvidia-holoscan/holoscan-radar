// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

namespace holoscan::radar::app::sar {
class SarImageVizOp : public Operator<> {
 public:
  SarImageVizOp(int64_t max_v, int64_t max_u, int64_t display_side, double hold_s);
  ~SarImageVizOp() override;
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  Input<Tensor> image{"image"};
  Output<schema::ImageT> frame{"frame"};

 private:
  int64_t max_v_{0}, max_u_{0}, display_side_{0};
  double hold_s_{0.0};
  float* sampled_magnitude_{nullptr};
};
}  // namespace holoscan::radar::app::sar
