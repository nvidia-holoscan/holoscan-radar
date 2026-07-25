// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// App-specific operators for fmcw_mimo, colocated with the app:
//   DdmReportVelocityBinsOp — post-DDM-demux velocity axis for reporting
//   MimoTdmAdcSourceOp / MimoDdmAdcSourceOp — synthetic multiplexed ADC sources
// These are thin Holoscan wrappers around the shared MIMO simulators
// (radar_common/sim/mimo_sim.hpp), so they stay local while the signal model
// lives in radar_common.

#pragma once

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_app/report/cadence_meter.hpp"
#include "radar_common/descriptors/fmcw_waveform_descriptor.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/mimo_sim.hpp"
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

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    radar::sim::SimResult result =
        radar::sim::simulate_mimo_tdm_interleaved(scene, tx_schedule);
    if (result.shape.size() != 3) {
      throw std::runtime_error(
          "MimoTdmAdcSourceOp: expected rank-3 (N_rx, M*L, samples)");
    }
    // Apply I/Q additives (thermal noise) on top of the clean solver response.
    radar::sim::add_thermal_noise(result.cube, scene.snr_db, scene.seed);

    std::vector<std::complex<float>> cube(result.cube.size());
    for (size_t i = 0; i < result.cube.size(); ++i) {
      cube[i] = std::complex<float>(static_cast<float>(result.cube[i].real()),
                                    static_cast<float>(result.cube[i].imag()));
    }

    auto tensor =
        radar::make_device_tensor(result.shape, radar::complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("MimoTdmAdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    if (cadence) {
      cadence->on_source_emit();
    }
    radar::sim::advance_scene(&scene, frame_period);
  }
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

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context)
      override {
    radar::sim::SimResult result =
        use_phase_codes ? radar::sim::simulate_mimo_ddm_phase_coded(
                              scene, phase_codes)
                        : radar::sim::simulate_mimo_ddm_coded(
                              scene, doppler_offsets);
    if (result.shape.size() != 3) {
      throw std::runtime_error(
          "MimoDdmAdcSourceOp: expected rank-3 (N_rx, num_chirps, samples)");
    }
    // Apply I/Q additives (thermal noise) on top of the clean solver response.
    radar::sim::add_thermal_noise(result.cube, scene.snr_db, scene.seed);

    std::vector<std::complex<float>> cube(result.cube.size());
    for (size_t i = 0; i < result.cube.size(); ++i) {
      cube[i] = std::complex<float>(static_cast<float>(result.cube[i].real()),
                                    static_cast<float>(result.cube[i].imag()));
    }

    auto tensor =
        radar::make_device_tensor(result.shape, radar::complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("MimoDdmAdcSourceOp: H2D upload failed");
    }

    holoscan::TensorMap out;
    out.insert({"adc_cube", tensor});
    op_output.emit(out, "adc_cube");
    if (cadence) {
      cadence->on_source_emit();
    }
    radar::sim::advance_scene(&scene, frame_period);
  }
};
