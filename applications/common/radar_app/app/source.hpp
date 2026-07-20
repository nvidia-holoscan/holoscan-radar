// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Rank-agnostic synthetic ADC source for the FMCW example apps. Emits one
// complex-IF ADC cube per tick from the shared analytic point-target simulator,
// advancing the targets through space by one frame period between ticks. The
// emitted cube follows the simulator's shape verbatim — rank-2
// (num_chirps, samples_per_chirp) for single-RX SISO, rank-3
// (num_rx, num_chirps, samples_per_chirp) for SIMO — so one operator serves
// both. (MIMO drives its own sources: TDM/DDM multiplexing needs a bespoke
// simulator that does not fit the plain simulate(SimScene) shape.)
//
// Header-only: methods are in-class inline, so multiple app TUs can include it.

#pragma once

#include <cuda_runtime_api.h>

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "radar_common/sdk/tensor_io.hpp"
#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::app {

class SimAdcSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SimAdcSourceOp)
  SimAdcSourceOp() = default;

  sim::SimScene scene;        // set from the parsed config before run()
  double frame_period{0.1};   // s, between emitted cubes (and target motion)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<holoscan::TensorMap>("adc_cube");
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // The simulator computes in double precision; cast to the float ADC sample
    // type before upload. The cube rank follows scene.rx_positions (rank-2 for
    // one RX, rank-3 for many); allocate straight from sim.shape.
    const sim::SimResult result = sim::simulate(scene);
    std::vector<std::complex<float>> cube(result.cube.size());
    for (size_t i = 0; i < result.cube.size(); ++i) {
      cube[i] = std::complex<float>(static_cast<float>(result.cube[i].real()),
                                    static_cast<float>(result.cube[i].imag()));
    }

    auto tensor = make_device_tensor(result.shape, complex_dtype<float>());
    if (cudaMemcpy(tensor->data(), cube.data(),
                   sizeof(std::complex<float>) * cube.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("SimAdcSourceOp: H2D upload failed");
    }

    emit_single(op_output, "adc_cube", tensor);

    // Step the scene forward for the next frame.
    sim::advance_scene(&scene, frame_period);
  }
};

}  // namespace holoscan::radar::app
