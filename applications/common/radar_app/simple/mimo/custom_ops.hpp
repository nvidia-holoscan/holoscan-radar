// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
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
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/simulator.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_app/simple/mimo/config.hpp"

namespace holoscan::radar::app::mimo {

struct MimoTdmAdcSourceConfig {
  sim::SimScene scene;
  sim::SceneVisibility scene_visibility{};
  TXSchedule tx_schedule;
  double frame_period{0.1};
  int64_t num_frames{1};
  sim::SceneMover mover;
  std::shared_ptr<CadenceMeter> cadence;
};

class MimoTdmAdcSourceOp : public holoscan::Operator<> {
 public:
  MimoTdmAdcSourceOp(const MaxTensorExtents& extents,
                     MimoTdmAdcSourceConfig config)
      : config_(std::move(config)) {
    const char* who = "MimoTdmAdcSourceOp";
    const int64_t max_rx =
        require_extent(extents.max_rx, "max_rx", who);
    const int64_t samples = require_extent(
        extents.max_samples_per_chirp, "max_samples_per_chirp", who);
    const int64_t interleaved = require_extent(
        extents.max_interleaved_chirps, "max_interleaved_chirps", who);
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_rx, interleaved, samples},
        complex_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(adc_cube)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 3U, max_byte_span_));
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
    const int64_t num_tx =
        static_cast<int64_t>(config_.scene.tx_positions.size());
    const int64_t slow_time = num_tx * config_.scene.num_chirps;
    const std::vector<int64_t> pattern =
        config_.tx_schedule.expand_chirp_tx_index(slow_time);
    std::vector<std::complex<double>> weights(
        static_cast<std::size_t>(num_tx * slow_time), {0.0, 0.0});
    for (int64_t chirp = 0; chirp < slow_time; ++chirp) {
      weights[static_cast<std::size_t>(
          pattern[static_cast<std::size_t>(chirp)] * slow_time + chirp)] =
          {1.0, 0.0};
    }
    return emit_frame(context, std::move(weights), slow_time);
  }

  holoscan::Output<holoscan::Tensor> adc_cube{"adc_cube"};

 private:
  holoscan::expected<void, holoscan::Error> emit_frame(
      holoscan::ExecutionContext& context,
      std::vector<std::complex<double>> weights, int64_t slow_time) {
    cudaStream_t stream = context.cuda_stream().get();
    std::shared_ptr<holoscan::Tensor> staged;
    try {
      staged = sim::simulate_mimo_gpu(
          sim::visible_scene(config_.scene, config_.scene_visibility),
          weights, slow_time, stream);
      sim::add_thermal_noise_gpu(
          *staged, config_.scene.snr_db,
          config_.scene.seed + frames_emitted_, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("stream synchronization failed");
      }
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("MimoTdmAdcSourceOp: simulate failed: ") +
              error.what()});
    }
    HOLOSCAN_RETURN_IF_ERROR(copy_and_emit(staged, context));
    finish_frame();
    return {};
  }

  holoscan::expected<void, holoscan::Error> copy_and_emit(
      const std::shared_ptr<holoscan::Tensor>& staged,
      holoscan::ExecutionContext& context) {
    auto loan = adc_cube.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = staged->shape(), .dtype = staged->dtype()});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemcpyAsync(
            writer->data(), staged->data(), writer->byte_size(),
            cudaMemcpyDeviceToDevice,
            context.cuda_stream().get()) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "MimoTdmAdcSourceOp: output D2D failed"});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    return adc_cube.emit(
        std::move(*loan),
        holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  void finish_frame() {
    if (config_.cadence) {
      config_.cadence->on_source_emit();
    }
    if (config_.mover) {
      config_.mover(config_.scene, config_.frame_period);
    } else {
      sim::advance_scene(&config_.scene, config_.frame_period);
    }
    ++frames_emitted_;
  }

  MimoTdmAdcSourceConfig config_;
  std::size_t max_byte_span_{0U};
  int64_t frames_emitted_{0};
};

struct MimoDdmAdcSourceConfig {
  sim::SimScene scene;
  sim::SceneVisibility scene_visibility{};
  std::vector<double> doppler_offsets;
  std::vector<std::complex<double>> phase_codes;
  double frame_period{0.1};
  int64_t num_frames{1};
  sim::SceneMover mover;
  std::shared_ptr<CadenceMeter> cadence;
};

class MimoDdmAdcSourceOp : public holoscan::Operator<> {
 public:
  MimoDdmAdcSourceOp(const MaxTensorExtents& extents,
                     MimoDdmAdcSourceConfig config)
      : config_(std::move(config)) {
    const char* who = "MimoDdmAdcSourceOp";
    const int64_t max_rx =
        require_extent(extents.max_rx, "max_rx", who);
    const int64_t max_chirps =
        require_extent(extents.max_chirps, "max_chirps", who);
    const int64_t samples = require_extent(
        extents.max_samples_per_chirp, "max_samples_per_chirp", who);
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_rx, max_chirps, samples},
        complex_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(adc_cube)
        .produces_tensor(holoscan::radar::pooled_tensor_output(
            kDevice, 3U, max_byte_span_));
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
    const int64_t num_tx =
        static_cast<int64_t>(config_.scene.tx_positions.size());
    const int64_t slow_time = config_.scene.num_chirps;
    std::vector<std::complex<double>> weights;
    if (!config_.phase_codes.empty()) {
      if (static_cast<int64_t>(config_.phase_codes.size()) !=
          num_tx * slow_time) {
        return invalid_code(
            "phase_codes length must equal num_tx * num_chirps");
      }
      weights = config_.phase_codes;
    } else {
      if (static_cast<int64_t>(config_.doppler_offsets.size()) != num_tx) {
        return invalid_code(
            "doppler_offsets length must equal num_tx");
      }
      weights.assign(
          static_cast<std::size_t>(num_tx * slow_time), {0.0, 0.0});
      for (int64_t tx = 0; tx < num_tx; ++tx) {
        for (int64_t chirp = 0; chirp < slow_time; ++chirp) {
          const double phase =
              2.0 * kPi *
              config_.doppler_offsets[static_cast<std::size_t>(tx)] *
              static_cast<double>(chirp);
          weights[static_cast<std::size_t>(
              tx * slow_time + chirp)] =
              {std::cos(phase), std::sin(phase)};
        }
      }
    }
    return emit_frame(context, std::move(weights), slow_time);
  }

  holoscan::Output<holoscan::Tensor> adc_cube{"adc_cube"};

 private:
  holoscan::expected<void, holoscan::Error> invalid_code(
      const char* message) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        std::string("MimoDdmAdcSourceOp: ") + message});
  }

  holoscan::expected<void, holoscan::Error> emit_frame(
      holoscan::ExecutionContext& context,
      std::vector<std::complex<double>> weights, int64_t slow_time) {
    cudaStream_t stream = context.cuda_stream().get();
    std::shared_ptr<holoscan::Tensor> staged;
    try {
      staged = sim::simulate_mimo_gpu(
          sim::visible_scene(config_.scene, config_.scene_visibility),
          weights, slow_time, stream);
      sim::add_thermal_noise_gpu(
          *staged, config_.scene.snr_db,
          config_.scene.seed + frames_emitted_, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("stream synchronization failed");
      }
    } catch (const std::exception& error) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("MimoDdmAdcSourceOp: simulate failed: ") +
              error.what()});
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
    if (cudaMemcpyAsync(
            writer->data(), staged->data(), writer->byte_size(),
            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          "MimoDdmAdcSourceOp: output D2D failed"});
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

  MimoDdmAdcSourceConfig config_;
  std::size_t max_byte_span_{0U};
  int64_t frames_emitted_{0};
};

}  // namespace holoscan::radar::app::mimo
