// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Rank-agnostic synthetic ADC source for the FMCW example apps. Emits one
// complex-IF ADC cube per tick from the GPU point-target simulator
// (radar_common/sim/sim_gpu.hpp), advancing the targets through space by one
// frame period between ticks. The cube is synthesized on the device and emitted
// directly — no host synthesis, no cube H2D. Its shape follows the scene: rank-2
// (num_chirps, samples_per_chirp) for single-RX SISO, rank-3
// (num_rx, num_chirps, samples_per_chirp) for SIMO — so one operator serves
// both. (MIMO drives its own sources: TDM/DDM multiplexing needs a bespoke
// simulator that does not fit the single-TX simulate_gpu(SimScene) shape.)
//
// Header-only: methods are in-class inline, so multiple app TUs can include it.
// Apps that use this operator must link holoscan::radar_common_sim_gpu.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>

#include <holoscan/holoscan.hpp>

#include "radar_app/report/cadence_meter.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/sim_gpu.hpp"

namespace holoscan::radar::app {

class SimAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimAdcSourceOp)
  SimAdcSourceOp() = default;

  sim::SimScene scene;        // set from the parsed config before run()
  double frame_period{0.1};   // s, between emitted cubes (and target motion)
  // Per-frame scene-evolution strategy (see sim::SceneMover). Unset → the
  // default constant-velocity mover; set a custom one for orbits/maneuvers or a
  // procedural target source, without touching the simulator.
  sim::SceneMover mover;
  // Optional shared cadence probe (same instance as DetectionReportOp).
  std::shared_ptr<CadenceMeter> cadence;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  // A source operator has no input port to draw a CUDA stream from, so it owns
  // one for the lifetime of the run.
  void start() override {
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
      throw std::runtime_error("SimAdcSourceOp: cudaStreamCreate failed");
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
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Synthesize the clean scatterer response on the device, then apply the
    // thermal-noise I/Q additive in place — both stream-ordered on stream_ and
    // synchronized before the tensor is emitted. The noise seed advances per
    // frame so each tick draws a fresh, independent realization (deterministic
    // per run); a constant seed would freeze the same noise across all frames.
    auto tensor = sim::simulate_gpu(scene, stream_);
    sim::add_thermal_noise_gpu(*tensor, scene.snr_db, scene.seed + frame_index_,
                               stream_);

    emit_single(op_output, "adc_cube", tensor);
    if (cadence) {
      cadence->on_source_emit();
    }

    // Step the scene forward for the next frame via the configured mover
    // (default: constant-velocity advance).
    if (mover) {
      mover(scene, frame_period);
    } else {
      sim::advance_scene(&scene, frame_period);
    }
    ++frame_index_;
  }

 private:
  cudaStream_t stream_{nullptr};
  uint64_t frame_index_{0};  // advances the per-frame noise seed
};

}  // namespace holoscan::radar::app
