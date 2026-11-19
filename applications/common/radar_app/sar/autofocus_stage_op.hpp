// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "radar_common/sar/formation_hud.hpp"
#include "radar_common/sdk/tensor_io.hpp"

namespace holoscan::radar::app::sar {

// A stage checkpoint at an existing application boundary. Forward the original
// sample and its storage, readiness and capture metadata; never allocate a pool,
// copy the aperture/image, or wait for CUDA. This reports input admission to the
// next stage, not an assertion that its GPU kernels have started executing.
class AutofocusStageOp : public Operator<> {
 public:
  AutofocusStageOp(std::shared_ptr<FormationHud> hud, AutofocusHudStage stage)
  : hud_(std::move(hud)), stage_(stage) {}

  void setup(OperatorSpec& spec) override {
    constexpr TensorRepresentation device{
        .memory_kind = MemoryKind::kCudaDevice, .dtype = std::nullopt, .rank = std::nullopt};
    spec.input(input).queue_depth(1U).expects_tensor(tensor_input(device));
    spec.output(output).produces_tensor(tensor_output(device));
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample)
      return make_unexpected(std::move(sample).error());
    if (hud_)
      hud_->publish_autofocus_stage(stage_, context.activation_time().timestamp_ns);
    return emit_required_result(context, "AutofocusStageOp.output", false, [&] {
      return output.emit(*sample);
    });
  }

  Input<Tensor> input{"input"};
  Output<Tensor> output{"output"};

 private:
  std::shared_ptr<FormationHud> hud_;
  AutofocusHudStage stage_;
};
}  // namespace holoscan::radar::app::sar
