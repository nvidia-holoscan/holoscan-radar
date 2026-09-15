// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-local bridge for the beamform-then-detect path. CFAR peak suppression
// already emits one row per local power maximum; this operator gives each row
// an identity label and unit size for the shared report and display consumers.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"

namespace holoscan::radar::app::simo {

namespace detail {

void launch_peak_identity_labels(int64_t* labels, int64_t count,
                                 cudaStream_t stream);
void launch_peak_identity_sizes(int64_t* sizes, int64_t count,
                                cudaStream_t stream);

}  // namespace detail

class PeakIdentityOp final : public holoscan::Operator<> {
 public:
  explicit PeakIdentityOp(const MaxTensorExtents& extents);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Output<holoscan::Tensor> labels{"labels"};
  holoscan::Output<holoscan::Tensor> sizes{"sizes"};
  holoscan::Output<holoscan::Tensor> rep_idx{"rep_idx"};

 private:
  std::size_t vector_max_byte_span_{0};
  std::size_t index_max_byte_span_{0};
};

}  // namespace holoscan::radar::app::simo
