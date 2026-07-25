// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Validates the GPU point-target simulator (simulate_gpu, MatX) against the CPU
// analytic reference (simulate) to a tight numerical tolerance — not bit-exact
// (GPU transcendentals / reduction order differ), but far below the float32 ADC
// quantization, so it is equivalent for the downstream pipeline.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "radar_common/sim/point_target_sim.hpp"
#include "radar_common/sim/sim_gpu.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace sim = holoscan::radar::sim;
using holoscan::radar::testing::to_host;

sim::SimScene base_scene() {
  sim::SimScene scene;
  scene.waveform.carrier_frequency = 77e9;
  scene.waveform.bandwidth = 150e6;
  scene.waveform.chirp_duration = 20e-6;
  scene.waveform.sample_rate = 12.8e6;
  scene.waveform.chirp_repetition_interval = 25e-6;
  scene.num_chirps = 64;
  scene.samples_per_chirp = 128;
  scene.targets.push_back(
      {sim::Vec3{0.0, 60.0, 0.0}, sim::Vec3{0.0, 0.0, 0.0}, {2.0, 0.0}});
  scene.targets.push_back(
      {sim::Vec3{5.0, 42.0, 0.0}, sim::Vec3{0.0, 5.5, 0.0}, {1.0, -0.3}});
  return scene;
}

// Max |gpu - cpu| over the cube, comparing the GPU complex<float> cube against
// the CPU double cube cast to float.
double max_abs_diff(const sim::SimScene& scene) {
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }
  auto dev = sim::simulate_gpu(scene, stream);
  const auto cpu = sim::simulate(scene);
  const auto host = to_host(*dev);
  cudaStreamDestroy(stream);

  EXPECT_EQ(static_cast<size_t>(host.size()), cpu.cube.size());
  const auto* g = host.as<std::complex<float>>();
  double worst = 0.0;
  for (size_t i = 0; i < cpu.cube.size(); ++i) {
    const std::complex<float> ref(static_cast<float>(cpu.cube[i].real()),
                                  static_cast<float>(cpu.cube[i].imag()));
    worst = std::max(worst, static_cast<double>(std::abs(g[i] - ref)));
  }
  std::fprintf(stderr, "[sim_gpu] max|gpu-cpu| = %.3e\n", worst);
  return worst;
}

TEST(SimGpu, MatchesCpuSingleRx) {
  EXPECT_LT(max_abs_diff(base_scene()), 1e-3);
}

TEST(SimGpu, MatchesCpuMultiRx) {
  auto scene = base_scene();
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0}, sim::Vec3{0.002, 0.0, 0.0},
                        sim::Vec3{0.004, 0.0, 0.0}, sim::Vec3{0.006, 0.0, 0.0}};
  EXPECT_LT(max_abs_diff(scene), 1e-3);
}

TEST(SimGpu, MatchesCpuPreMixer) {
  auto scene = base_scene();
  scene.dechirped = false;
  EXPECT_LT(max_abs_diff(scene), 1e-3);
}

TEST(SimGpu, ShapeMatchesRank) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto siso = sim::simulate_gpu(base_scene(), stream);
  EXPECT_EQ(siso->shape().size(), 2u);  // (chirps, samples)
  auto scene = base_scene();
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0}, sim::Vec3{0.002, 0.0, 0.0}};
  auto simo = sim::simulate_gpu(scene, stream);
  EXPECT_EQ(simo->shape().size(), 3u);  // (rx, chirps, samples)
  cudaStreamDestroy(stream);
}

}  // namespace
