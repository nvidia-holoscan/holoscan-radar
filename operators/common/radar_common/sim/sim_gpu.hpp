// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GPU-accelerated point-target FMCW simulator (MatX). The device-resident
// counterpart of point_target_sim.hpp::simulate(): same closed-form signal
// model (far-field two-way delay, beat frequency, slow-time carrier phase),
// synthesized on the device and returned as a complex<float> ADC cube ready to
// emit — no host synthesis, no H2D of the cube.
//
// The physics matches the CPU reference to a tight numerical tolerance (not
// bit-exact: GPU transcendentals and reduction order differ). The CPU simulate()
// remains the analytic ground truth against which this is validated.
//
// The host launch wrapper is declared here (CUDA-free, host-includable) and
// defined in sim_gpu.cu, compiled into holoscan::radar_common_sim_gpu. Link that
// library; do not recompile the .cu into another target (ODR). radar_common
// itself stays CUDA-free.

#pragma once

#include <cuda_runtime_api.h>

#include <memory>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/sim/point_target_sim.hpp"

namespace holoscan::radar::sim {

// Synthesize the single-TX ADC cube for `scene` on the device and return it as a
// complex<float> holoscan::Tensor (compact row-major): (num_chirps, samples) for
// one RX, (num_rx, num_chirps, samples) for many — the same shape simulate()
// produces. Contributions are accumulated in double then cast to float (matching
// the CPU path's double model → float ADC). Clean scatterer response only;
// thermal noise / additives are applied separately (additives.hpp). Requires
// exactly one TX element (use the MIMO GPU path for multi-TX). All work is
// enqueued on `stream`.
std::shared_ptr<holoscan::Tensor> simulate_gpu(const SimScene& scene,
                                               cudaStream_t stream);

}  // namespace holoscan::radar::sim
