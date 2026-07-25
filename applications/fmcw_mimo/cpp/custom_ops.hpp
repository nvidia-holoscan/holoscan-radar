// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-specific operators for fmcw_mimo, colocated with the app:
//   DdmReportVelocityBinsOp — post-DDM-demux velocity axis for reporting
//   MimoTdmAdcSourceOp / MimoDdmAdcSourceOp — synthetic multiplexed ADC sources
// The source operators build the per-(tx, slot) excitation weights for their
// multiplexing scheme and drive the unified GPU simulator
// (radar_common/sim/sim_gpu.hpp::simulate_mimo_gpu), so the cube is synthesized
// on the device and emitted directly — no host synthesis, no cube H2D. Apps
// using them must link holoscan::radar_common_sim_gpu.

#pragma once

#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_app/report/cadence_meter.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/sim_gpu.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace radar = holoscan::radar;
using holoscan::radar::app::CadenceMeter;

class DdmReportVelocityBinsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DdmReportVelocityBinsOp)
  DdmReportVelocityBinsOp() = default;

  radar::FMCWWaveformDescriptor waveform;
  int64_t num_doppler{0};
  // The DDM demux keeps the full slow-time FFT bin resolution PRF/N (N = the
  // pre-demux Doppler length = num_chirps) but retains only W = N/num_tx minus
  // guard bins, re-centered. So the demuxed velocity axis has W bins spaced at
  // PRF/N (NOT PRF/W); using PRF/W would inflate velocities by ~num_tx and
  // fragment each target across Doppler in clustering.
  double bin_spacing_hz{0.0};

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::TensorMap>("sync");
    spec.output<holoscan::TensorMap>("velocity_bins");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    (void)radar::receive_single(op_input, "sync", "DdmReportVelocityBinsOp");
    // The axis is constant per run; build and upload it once, then re-emit the
    // cached device tensor each tick (the "sync" input just paces the emit).
    if (!velocity_bins_) {
      velocity_bins_ = build_axis();
    }
    holoscan::TensorMap out;
    out.insert({"velocity_bins", velocity_bins_});
    op_output.emit(out, "velocity_bins");
  }

 private:
  std::shared_ptr<holoscan::Tensor> build_axis() const {
    if (num_doppler <= 0) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: num_doppler must be positive");
    }
    if (!waveform.is_valid()) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: waveform descriptor is invalid");
    }
    if (bin_spacing_hz <= 0.0) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: bin_spacing_hz must be > 0");
    }
    const double wavelength = waveform.wavelength();
    // Center-zero axis: bin i maps to Doppler frequency (i - W/2) * df.
    std::vector<float> axis(static_cast<size_t>(num_doppler));
    for (int64_t i = 0; i < num_doppler; ++i) {
      const double f =
          static_cast<double>(i - num_doppler / 2) * bin_spacing_hz;
      axis[static_cast<size_t>(i)] =
          static_cast<float>(-f * wavelength / 2.0);
    }
    auto tensor = radar::make_device_tensor({num_doppler},
                                            radar::real_dtype<float>());
    if (cudaMemcpy(tensor->data(), axis.data(), sizeof(float) * axis.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error(
          "DdmReportVelocityBinsOp: velocity_bins H2D upload failed");
    }
    return tensor;
  }

  std::shared_ptr<holoscan::Tensor> velocity_bins_;
};

class MimoTdmAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MimoTdmAdcSourceOp)
  MimoTdmAdcSourceOp() = default;

  radar::sim::SimScene scene;
  radar::TXSchedule tx_schedule;
  double frame_period{0.1};
  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void start() override {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("MimoTdmAdcSourceOp: cudaStreamCreate failed");
    }
  }
  void stop() override {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    // TDM excitation: one TX active per interleaved slow-time slot. Build the
    // one-hot (tx, slot) weight matrix, then synthesize on the device.
    const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
    const int64_t slow = num_tx * scene.num_chirps;
    radar::TXSchedule sched = tx_schedule;
    sched.num_tx = num_tx;
    const auto pattern = sched.expand_chirp_tx_index(slow);
    std::vector<std::complex<double>> weights(
        static_cast<size_t>(num_tx * slow), {0.0, 0.0});
    for (int64_t k = 0; k < slow; ++k) {
      weights[static_cast<size_t>(pattern[static_cast<size_t>(k)] * slow + k)] =
          {1.0, 0.0};
    }

    auto tensor = radar::sim::simulate_mimo_gpu(scene, weights, slow, stream_);
    radar::sim::add_thermal_noise_gpu(*tensor, scene.snr_db, scene.seed,
                                      stream_);
    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    if (cadence) {
      cadence->on_source_emit();
    }
    radar::sim::advance_scene(&scene, frame_period);
  }

 private:
  cudaStream_t stream_{nullptr};
};

class MimoDdmAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MimoDdmAdcSourceOp)
  MimoDdmAdcSourceOp() = default;

  radar::sim::SimScene scene;
  std::vector<double> doppler_offsets;
  // When true, emit via simulate_mimo_ddm_phase_coded(phase_codes) instead of
  // the linear-offset gather path.
  bool use_phase_codes{false};
  std::vector<std::complex<double>> phase_codes;
  double frame_period{0.1};
  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void start() override {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("MimoDdmAdcSourceOp: cudaStreamCreate failed");
    }
  }
  void stop() override {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    // DDM excitation: all TX fire every chirp. The (tx, chirp) weight is either
    // the explicit unit-magnitude phase code, or the linear Doppler offset
    // exp(i*2*pi*offset_tx*k). slow_time = num_chirps.
    const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
    const int64_t slow = scene.num_chirps;
    std::vector<std::complex<double>> weights;
    if (use_phase_codes) {
      if (static_cast<int64_t>(phase_codes.size()) != num_tx * slow) {
        throw std::runtime_error(
            "MimoDdmAdcSourceOp: phase_codes must be (num_tx * num_chirps)");
      }
      weights = phase_codes;  // row-major (tx, chirp)
    } else {
      if (static_cast<int64_t>(doppler_offsets.size()) != num_tx) {
        throw std::runtime_error(
            "MimoDdmAdcSourceOp: doppler_offsets length must equal num_tx");
      }
      weights.assign(static_cast<size_t>(num_tx * slow), {0.0, 0.0});
      for (int64_t tx = 0; tx < num_tx; ++tx) {
        for (int64_t k = 0; k < slow; ++k) {
          const double ph = 2.0 * radar::kPi *
                            doppler_offsets[static_cast<size_t>(tx)] *
                            static_cast<double>(k);
          weights[static_cast<size_t>(tx * slow + k)] = {std::cos(ph),
                                                         std::sin(ph)};
        }
      }
    }

    auto tensor = radar::sim::simulate_mimo_gpu(scene, weights, slow, stream_);
    radar::sim::add_thermal_noise_gpu(*tensor, scene.snr_db, scene.seed,
                                      stream_);
    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    if (cadence) {
      cadence->on_source_emit();
    }
    radar::sim::advance_scene(&scene, frame_period);
  }

 private:
  cudaStream_t stream_{nullptr};
};
