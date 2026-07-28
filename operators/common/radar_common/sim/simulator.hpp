// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GPU-accelerated point-target FMCW simulator (MatX). The device-resident
// counterpart of scene.hpp::simulate(): same closed-form signal
// model (far-field two-way delay, beat frequency, slow-time carrier phase),
// synthesized on the device and returned as a complex<float> ADC cube ready to
// emit — no host synthesis, no H2D of the cube.
//
// The physics matches the CPU reference to a tight numerical tolerance (not
// bit-exact: GPU transcendentals and
// reduction order differ). The CPU simulate()
// remains the analytic ground truth against which this is validated.
//
// The host launch wrapper is declared here (CUDA-free, host-includable) and
// defined in simulator.cu, compiled into
// holoscan::radar_common_simulator. Link that
// library; do not recompile the .cu into another target (ODR). radar_common
// itself stays CUDA-free.

#pragma once

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "radar_common/sim/scene.hpp"

namespace holoscan::radar::sim {

// Synthesize the single-TX ADC cube for
// `scene` on the device and return it as a
// complex<float> holoscan::Tensor (compact
// row-major): (num_chirps, samples) for
// one RX, (num_rx, num_chirps, samples) for many — the same shape simulate()
// produces. Contributions are accumulated
// in double then cast to float (matching
// the CPU path's double model → float ADC). Clean scatterer response only;
// thermal noise / additives are applied separately (additives.hpp). Requires
// exactly one TX element (use the MIMO GPU path for multi-TX). All work is
// enqueued on `stream`.
std::shared_ptr<holoscan::Tensor> simulate_gpu(const SimScene& scene,
                                               cudaStream_t stream);

// Synthesize a MULTI-TX ADC cube on the device (MatX) and return it as a
// complex<float> holoscan::Tensor of shape (num_rx, slow_time, samples) — the
// device counterpart of the CPU mimo solvers (the tests-only reference),
// unified into one weight-driven core. The multiplexing scheme is captured
// entirely by `weights` (row-major (num_tx, slow_time) complex), the
// per-(tx, slot) excitation:
//   - TDM: slow_time = num_tx * num_chirps; weight 1 on the slot's active TX,
//          0 elsewhere.
//   - DDM (coded): slow_time = num_chirps; weight exp(i*2*pi*offset_tx*k).
//   - DDM (phase-coded): slow_time = num_chirps; weight = the (tx, k) code.
// num_tx = scene.tx_positions.size(); the slow-time step uses
// scene.waveform.chirp_repetition_interval. Clean scatterer response only.
std::shared_ptr<holoscan::Tensor> simulate_mimo_gpu(
    const SimScene& scene, const std::vector<std::complex<double>>& weights,
    int64_t slow_time, cudaStream_t stream);

// I/Q additive (device): add seeded complex white Gaussian noise in place to a
// complex<float> cube at `snr_db` relative to a unit-amplitude target — the
// device counterpart of additives.hpp::add_thermal_noise, so the noisy cube
// never leaves the GPU. No-op when `snr_db` is unset. Not bit-identical to the
// CPU additive's realization (device RNG);
// statistically equivalent. Enqueued on
// `stream`.
void add_thermal_noise_gpu(holoscan::Tensor& cube,
                           const std::optional<double>& snr_db, uint64_t seed,
                           cudaStream_t stream);

}  // namespace holoscan::radar::sim
