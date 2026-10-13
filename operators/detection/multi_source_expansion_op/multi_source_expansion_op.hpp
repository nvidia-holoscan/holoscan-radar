// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"

namespace holoscan::radar {

template <typename T>
class MultiSourceExpansionOp : public holoscan::Operator<> {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "MultiSourceExpansionOp supports float and double precision");

 public:
  explicit MultiSourceExpansionOp(const MaxTensorExtents& extents);
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> detection_idx{"detection_idx"};
  Input<Tensor> baseline_aoa{"baseline_aoa"};
  Input<Tensor> selected_idx{"selected_idx"};
  Input<Tensor> resolved_aoa{"resolved_aoa"};
  Input<Tensor> doa_multi{"doa_multi"};
  Input<Tensor> source_count{"source_count"};
  Output<Tensor> detection_idx_out{"detection_idx_out"};
  Output<Tensor> aoa_estimates_out{"aoa_estimates_out"};

 private:
  struct RuntimeState;
  std::shared_ptr<RuntimeState> state_;
  Tensor retained_detection_idx_;
  Tensor retained_baseline_;
  Tensor retained_selected_;
  Tensor retained_resolved_;
  Tensor retained_multi_;
  Tensor retained_count_;
  std::int64_t max_items_{0};
  std::int64_t max_points_{0};
  std::int64_t max_sources_{0};
  std::size_t idx_max_byte_span_{0U};
  std::size_t aoa_max_byte_span_{0U};
};

extern template class MultiSourceExpansionOp<float>;
extern template class MultiSourceExpansionOp<double>;

}  // namespace holoscan::radar
