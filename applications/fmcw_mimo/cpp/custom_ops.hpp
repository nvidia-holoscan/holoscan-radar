// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-specific Graph operators for fmcw_mimo:
//   DdmReportVelocityBinsOp — post-DDM-demux velocity axis for reporting
//   MimoTdmAdcSourceOp / MimoDdmAdcSourceOp — synthetic multiplexed ADC sources
// Sources build per-(tx, slot) excitation weights and drive
// simulate_mimo_gpu; apps must link holoscan::radar_common_simulator.
// Graph op handles are const — pass runtime config via the ctor.

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

namespace holoscan::radar::app {

// ---------------------------------------------------------------------------
// DdmReportVelocityBinsOp
// ---------------------------------------------------------------------------

struct DdmReportVelocityBinsConfig {
  FMCWWaveformDescriptor waveform;
  int64_t num_doppler{0};
  // Demux keeps PRF/N spacing (N = pre-demux Doppler length), not PRF/W.
  double bin_spacing_hz{0.0};
};

class DdmReportVelocityBinsOp : public holoscan::Operator<> {
 public:
  DdmReportVelocityBinsOp(const MaxTensorExtents& extents,
                          DdmReportVelocityBinsConfig config)
      : config_(std::move(config)) {
    const int64_t max_d =
        require_extent(extents.max_doppler_bins, "max_doppler_bins",
                       "DdmReportVelocityBinsOp");
    max_byte_span_ =
        tensor_byte_span(std::vector<int64_t>{max_d}, real_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.input(sync).queue_depth(2U).expects_tensor(kDevice);
    spec.output(velocity_bins)
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 1U, .max_byte_span = max_byte_span_});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{sync});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const char* who = "DdmReportVelocityBinsOp";
    HOLOSCAN_TRY(auto unused, receive_tensor(sync, who));
    (void)unused;

    if (config_.num_doppler <= 0) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DdmReportVelocityBinsOp: num_doppler must be positive"});
    }
    if (!config_.waveform.is_valid()) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DdmReportVelocityBinsOp: waveform descriptor is invalid"});
    }
    if (config_.bin_spacing_hz <= 0.0) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kInvalidArgument,
          "DdmReportVelocityBinsOp: bin_spacing_hz must be > 0"});
    }

    if (!host_axis_) {
      const double wavelength = config_.waveform.wavelength();
      host_axis_ = std::make_shared<std::vector<float>>(
          static_cast<size_t>(config_.num_doppler));
      for (int64_t i = 0; i < config_.num_doppler; ++i) {
        const double f =
            static_cast<double>(i - config_.num_doppler / 2) *
            config_.bin_spacing_hz;
        (*host_axis_)[static_cast<size_t>(i)] =
            static_cast<float>(-f * wavelength / 2.0);
      }
    }

    const std::vector<int64_t> shape{config_.num_doppler};
    return emit_device_tensor(
        velocity_bins, context, shape, real_dtype<float>(),
        [&](void* dst, std::size_t bytes)
            -> holoscan::expected<void, holoscan::Error> {
          if (bytes != host_axis_->size() * sizeof(float)) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                "DdmReportVelocityBinsOp: velocity_bins loan size mismatch"});
          }
          if (cudaMemcpyAsync(dst, host_axis_->data(), bytes,
                              cudaMemcpyHostToDevice,
                              context.cuda_stream().get()) != cudaSuccess) {
            return holoscan::make_unexpected(holoscan::Error{
                holoscan::ErrorCode::kFailure,
                "DdmReportVelocityBinsOp: velocity_bins H2D failed"});
          }
          return {};
        });
  }

  holoscan::Input<holoscan::Tensor> sync{"sync"};
  holoscan::Output<holoscan::Tensor> velocity_bins{"velocity_bins"};

 private:
  DdmReportVelocityBinsConfig config_;
  std::size_t max_byte_span_{0U};
  std::shared_ptr<std::vector<float>> host_axis_;
};

// ---------------------------------------------------------------------------
// MimoTdmAdcSourceOp
// ---------------------------------------------------------------------------

struct MimoTdmAdcSourceConfig {
  sim::SimScene scene;
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
    const int64_t max_rx = require_extent(extents.max_rx, "max_rx", who);
    const int64_t samples = require_extent(extents.max_samples_per_chirp,
                                           "max_samples_per_chirp", who);
    int64_t interleaved = 0;
    if (extents.max_interleaved_chirps.has_value()) {
      interleaved = require_extent(extents.max_interleaved_chirps,
                                   "max_interleaved_chirps", who);
    } else {
      const int64_t max_tx = require_extent(extents.max_tx, "max_tx", who);
      const int64_t max_chirps =
          require_extent(extents.max_chirps, "max_chirps", who);
      interleaved = max_tx * max_chirps;
    }
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_rx, interleaved, samples},
        complex_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(adc_cube)
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 3U, .max_byte_span = max_byte_span_});
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
    const int64_t num_tx =
        static_cast<int64_t>(config_.scene.tx_positions.size());
    const int64_t slow = num_tx * config_.scene.num_chirps;
    TXSchedule sched = config_.tx_schedule;
    sched.num_tx = num_tx;
    const auto pattern = sched.expand_chirp_tx_index(slow);
    std::vector<std::complex<double>> weights(
        static_cast<size_t>(num_tx * slow), {0.0, 0.0});
    for (int64_t k = 0; k < slow; ++k) {
      weights[static_cast<size_t>(
          pattern[static_cast<size_t>(k)] * slow + k)] = {1.0, 0.0};
    }

    std::shared_ptr<holoscan::Tensor> staged;
    try {
      staged = sim::simulate_mimo_gpu(config_.scene, weights, slow, stream);
      sim::add_thermal_noise_gpu(*staged, config_.scene.snr_db,
                                 config_.scene.seed + frames_emitted_, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            "MimoTdmAdcSourceOp: stream sync failed"});
      }
    } catch (const std::exception& e) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("MimoTdmAdcSourceOp: simulate failed: ") + e.what()});
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
          holoscan::ErrorCode::kFailure, "MimoTdmAdcSourceOp: D2D failed"});
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
  MimoTdmAdcSourceConfig config_;
  std::size_t max_byte_span_{0U};
  int64_t frames_emitted_{0};
};

// ---------------------------------------------------------------------------
// MimoDdmAdcSourceOp
// ---------------------------------------------------------------------------

struct MimoDdmAdcSourceConfig {
  sim::SimScene scene;
  std::vector<double> doppler_offsets;
  bool use_phase_codes{false};
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
    const int64_t max_rx = require_extent(extents.max_rx, "max_rx", who);
    const int64_t max_chirps =
        require_extent(extents.max_chirps, "max_chirps", who);
    const int64_t samples = require_extent(extents.max_samples_per_chirp,
                                           "max_samples_per_chirp", who);
    max_byte_span_ = tensor_byte_span(
        std::vector<int64_t>{max_rx, max_chirps, samples},
        complex_dtype<float>());
  }

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorPortLayout kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    spec.output(adc_cube)
        .produces_tensor(kDevice)
        .tensor_allocation(holoscan::TensorAllocationBounds{
            .max_rank = 3U, .max_byte_span = max_byte_span_});
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
    const int64_t num_tx =
        static_cast<int64_t>(config_.scene.tx_positions.size());
    const int64_t slow = config_.scene.num_chirps;
    std::vector<std::complex<double>> weights;
    try {
      if (config_.use_phase_codes) {
        if (static_cast<int64_t>(config_.phase_codes.size()) !=
            num_tx * slow) {
          throw std::runtime_error(
              "phase_codes must be (num_tx * num_chirps)");
        }
        weights = config_.phase_codes;
      } else {
        if (static_cast<int64_t>(config_.doppler_offsets.size()) != num_tx) {
          throw std::runtime_error(
              "doppler_offsets length must equal num_tx");
        }
        weights.assign(static_cast<size_t>(num_tx * slow), {0.0, 0.0});
        for (int64_t tx = 0; tx < num_tx; ++tx) {
          for (int64_t k = 0; k < slow; ++k) {
            const double ph =
                2.0 * kPi * config_.doppler_offsets[static_cast<size_t>(tx)] *
                static_cast<double>(k);
            weights[static_cast<size_t>(tx * slow + k)] = {std::cos(ph),
                                                           std::sin(ph)};
          }
        }
      }
    } catch (const std::exception& e) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("MimoDdmAdcSourceOp: ") + e.what()});
    }

    std::shared_ptr<holoscan::Tensor> staged;
    try {
      staged = sim::simulate_mimo_gpu(config_.scene, weights, slow, stream);
      sim::add_thermal_noise_gpu(*staged, config_.scene.snr_db,
                                 config_.scene.seed + frames_emitted_, stream);
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kFailure,
            "MimoDdmAdcSourceOp: stream sync failed"});
      }
    } catch (const std::exception& e) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("MimoDdmAdcSourceOp: simulate failed: ") + e.what()});
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
          holoscan::ErrorCode::kFailure, "MimoDdmAdcSourceOp: D2D failed"});
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
  MimoDdmAdcSourceConfig config_;
  std::size_t max_byte_span_{0U};
  int64_t frames_emitted_{0};
};

}  // namespace holoscan::radar::app
