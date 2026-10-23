// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace holoscan::radar::detail {

struct DopplerNotchStageConfig {
  double center_velocity;
  double stop_half_width;
  double transition_width;
  bool cyclic_velocity_axis;
};

template <typename T>
__global__ void doppler_notch_kernel(const cuda::std::complex<T>* input, const T* velocity_bins,
                                     cuda::std::complex<T>* output, int64_t total_elements,
                                     int64_t num_doppler, int64_t num_range,
                                     DopplerNotchStageConfig config) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements)
    return;

  const int64_t doppler = (index / num_range) % num_doppler;
  const double velocity = static_cast<double>(velocity_bins[doppler]);
  double distance = fabs(velocity - config.center_velocity);

  int64_t nearest = 0;
  if (num_doppler > 1) {
    const double first = static_cast<double>(velocity_bins[0]);
    const double last = static_cast<double>(velocity_bins[num_doppler - 1]);
    const double spacing = (last - first) / static_cast<double>(num_doppler - 1);
    // Velocity producers may emit either axis direction; retain the signed index step.
    if (spacing != 0.0 && isfinite(spacing)) {
      int64_t raw_nearest =
          static_cast<int64_t>(llround((config.center_velocity - first) / spacing));
      if (config.cyclic_velocity_axis) {
        raw_nearest %= num_doppler;
        if (raw_nearest < 0)
          raw_nearest += num_doppler;
        nearest = raw_nearest;
        const double period = fabs(spacing) * static_cast<double>(num_doppler);
        distance = fmod(distance, period);
        distance = fmin(distance, period - distance);
      } else {
        nearest = raw_nearest;
        if (nearest < 0)
          nearest = 0;
        if (nearest >= num_doppler)
          nearest = num_doppler - 1;
      }
    }
  }

  double gain = 1.0;
  if (doppler == nearest || distance <= config.stop_half_width) {
    gain = 0.0;
  } else if (config.transition_width > 0.0 &&
             distance < config.stop_half_width + config.transition_width) {
    const double fraction = (distance - config.stop_half_width) / config.transition_width;
    gain = 0.5 * (1.0 - cos(3.14159265358979323846 * fraction));
  }
  output[index] = input[index] * static_cast<T>(gain);
}

template <typename T>
void run_doppler_notch_stage(const cuda::std::complex<T>* input, const T* velocity_bins,
                             cuda::std::complex<T>* output, int64_t total_elements,
                             int64_t num_doppler, int64_t num_range,
                             const DopplerNotchStageConfig& config, cudaStream_t stream) {
  if (input == nullptr || velocity_bins == nullptr || output == nullptr) {
    throw std::invalid_argument("doppler notch stage: null device pointer");
  }
  if (total_elements <= 0 || num_doppler <= 0 || num_range <= 0) {
    throw std::invalid_argument("doppler notch stage: invalid extent");
  }
  constexpr int kBlock = 256;
  const int grid = static_cast<int>((total_elements + kBlock - 1) / kBlock);
  doppler_notch_kernel<T><<<grid, kBlock, 0, stream>>>(input, velocity_bins, output, total_elements,
                                                       num_doppler, num_range, config);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("doppler notch stage: kernel launch failed");
  }
}

}  // namespace holoscan::radar::detail
