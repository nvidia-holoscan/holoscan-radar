// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Rank-agnostic synthetic ADC source for the FMCW example apps (HSDK 5 Graph).
// Emits one complex-IF ADC cube per OnClock tick from the GPU point-target
// simulator, advancing targets between ticks. Shape follows the scene: rank-2
// SISO or rank-3 SIMO. Apps must link holoscan::radar_common_simulator.
//
// Graph op handles are const — pass runtime config via the ctor (not member
// assignment after graph.op()).

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "radar_app/report/cadence_meter.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/simulator.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app {

struct SimAdcSourceConfig {
  sim::SimScene scene;
  sim::SceneVisibility scene_visibility{};
  double frame_period{0.1};
  int64_t num_frames{1};
  sim::SceneMover mover;
  std::shared_ptr<CadenceMeter> cadence;
};

class SimAdcSourceOp : public holoscan::Operator<> {
 public:
  SimAdcSourceOp(const MaxTensorExtents& extents, SimAdcSourceConfig config)
      : config_(std::move(config)) {
    const int64_t chirps =
        require_extent(extents.max_chirps, "max_chirps", "SimAdcSourceOp");
    const int64_t samples = require_extent(extents.max_samples_per_chirp,
                                           "max_samples_per_chirp",
                                           "SimAdcSourceOp");
    int64_t leading = 1;
    if (extents.max_rx.has_value() && *extents.max_rx > 1) {
      leading = *extents.max_rx;
    }
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{leading, chirps, samples}, complex_dtype<float>());
    max_rank_ = (leading > 1) ? 3U : 2U;
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(adc_cube)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, max_rank_, max_byte_span_));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(config_.frame_period));
    result.trigger(holoscan::OnClock{.period = period});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (frames_emitted_ >= config_.num_frames) {
      return {};
    }

    cudaStream_t stream = context.cuda_stream().get();
    std::shared_ptr<holoscan::Tensor> staged;
    try {
      staged = sim::simulate_gpu(
          sim::visible_scene(config_.scene, config_.scene_visibility), stream);
      sim::add_thermal_noise_gpu(*staged, config_.scene.snr_db,
                                 config_.scene.seed + frames_emitted_, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            "SimAdcSourceOp: stream sync failed"});
      }
    } catch (const std::exception& e) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("SimAdcSourceOp: simulate failed: ") + e.what()});
    }

    auto loan = adc_cube.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = staged->shape(), .dtype = staged->dtype()});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemcpyAsync(writer->data(), staged->data(), writer->byte_size(),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "SimAdcSourceOp: D2D failed"});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    HOLOSCAN_RETURN_IF_ERROR(adc_cube.emit(
        std::move(*loan),
        holoscan::EmitOptions{.capture_time = context.activation_time()}));

    if (config_.cadence) {
      config_.cadence->on_source_emit();
    }
    if (config_.mover) {
      config_.mover(config_.scene, config_.frame_period);
    } else {
      sim::advance_scene(&config_.scene, config_.frame_period);
    }
    ++frames_emitted_;
    return {};
  }

  holoscan::Output<holoscan::Tensor> adc_cube{"adc_cube"};

 private:
  SimAdcSourceConfig config_;
  std::size_t max_byte_span_{0U};
  std::uint32_t max_rank_{2U};
  int64_t frames_emitted_{0};
};

}  // namespace holoscan::radar::app
