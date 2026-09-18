// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstdio>
#include <memory>
#include <vector>

#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/math/constants.hpp"
#include "utils/mimo_reference.hpp"
#include "radar_common/sim/simulator.hpp"
#include "utils/sim_reference.hpp"
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

sim::SimScene scene_for_beat(double beat_frequency,
                             int64_t ambiguity_order = 0) {
  auto scene = base_scene();
  scene.num_chirps = 2;
  scene.targets.clear();
  const double delay =
      beat_frequency / scene.waveform.slope() +
      static_cast<double>(ambiguity_order) *
          scene.waveform.chirp_repetition_interval;
  const double range = 0.5 * scene.propagation_speed * delay;
  scene.targets.push_back(
      {sim::Vec3{range, 0.0, 0.0}, sim::Vec3{}, {1.0, 0.0}});
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

TEST(SimGpu, MatchesCpuInRaisedCosineTransition) {
  const double fs = base_scene().waveform.sample_rate;
  const auto scene = scene_for_beat(0.95 * fs);
  EXPECT_NEAR(std::abs(sim::simulate(scene).cube.front()), 0.5, 1e-12);
  EXPECT_LT(max_abs_diff(scene), 1e-3);
}

TEST(SimGpu, MatchesCpuInFilteredStopband) {
  const double fs = base_scene().waveform.sample_rate;
  const auto scene = scene_for_beat(1.05 * fs);
  EXPECT_NEAR(std::abs(sim::simulate(scene).cube.front()), 0.0, 1e-12);
  EXPECT_LT(max_abs_diff(scene), 1e-3);
}

TEST(SimGpu, MatchesCpuAtRealIfBoundary) {
  const double fs = base_scene().waveform.sample_rate;
  auto scene = scene_for_beat(0.5 * fs);
  scene.analog_if.sampling_mode = sim::IfSamplingMode::kReal;
  EXPECT_NEAR(std::abs(sim::simulate(scene).cube.front()), 0.0, 1e-12);
  EXPECT_LT(max_abs_diff(scene), 1e-3);
}

TEST(SimGpu, MatchesCpuWithExplicitUnfilteredResponse) {
  const double fs = base_scene().waveform.sample_rate;
  auto scene = scene_for_beat(1.05 * fs);
  scene.analog_if.response = sim::AnalogIfResponseType::kNone;
  EXPECT_NEAR(std::abs(sim::simulate(scene).cube.front()), 1.0, 1e-12);
  EXPECT_LT(max_abs_diff(scene), 1e-3);
}

TEST(SimGpu, MatchesCpuSimoAcrossMultipleCriIntervals) {
  const double fs = base_scene().waveform.sample_rate;
  auto scene = scene_for_beat(0.2 * fs, 2);
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.002, 0.0, 0.0},
                        sim::Vec3{0.004, 0.0, 0.0}};
  EXPECT_LT(max_abs_diff(scene), 1e-3);
}

TEST(SimGpu, MatchesCpuWithRangeLossAtReferenceTwiceAndFolded) {
  for (const double range : {40.0, 80.0, 3787.405725}) {
    auto scene = base_scene();
    scene.targets = {
        {sim::Vec3{range, 0.0, 0.0}, sim::Vec3{}, {1.0, -0.2}}};
    scene.range_loss = {
        sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
    EXPECT_LT(max_abs_diff(scene), 1e-3);
  }

  auto pre_mixer = base_scene();
  pre_mixer.targets = {
      {sim::Vec3{80.0, 0.0, 0.0}, sim::Vec3{}, {1.0, -0.2}}};
  pre_mixer.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  pre_mixer.analog_if.response = sim::AnalogIfResponseType::kNone;
  pre_mixer.dechirped = false;
  EXPECT_LT(max_abs_diff(pre_mixer), 1e-3);
}

TEST(SimGpu, NoiseHasExpectedVariance) {
  // Pure-noise cube (no targets): E[|z|^2] = 2*sigma^2 = noise_power.
  auto scene = base_scene();
  scene.targets.clear();
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto dev = sim::simulate_gpu(scene, stream);
  const double snr_db = 10.0;
  sim::add_thermal_noise_gpu(*dev, snr_db, 0xC0FFEEULL, stream);
  const auto host = to_host(*dev);
  const auto* g = host.as<std::complex<float>>();
  double sumsq = 0.0;
  const size_t n = static_cast<size_t>(host.size());
  for (size_t i = 0; i < n; ++i) {
    sumsq += static_cast<double>(std::norm(g[i]));
  }
  const double measured = sumsq / static_cast<double>(n);
  const double expected = std::pow(10.0, -snr_db / 10.0);
  cudaStreamDestroy(stream);
  EXPECT_NEAR(measured, expected, 0.1 * expected);  // ~8k samples → <10%
}

// --- MIMO GPU core (simulate_mimo_gpu) vs the CPU mimo_sim solvers ---

namespace rd = holoscan::radar;
using holoscan::radar::kPi;

sim::SimScene mimo_scene() {
  auto scene = base_scene();
  scene.targets = {{sim::Vec3{0.0, 40.0, 0.0}, sim::Vec3{0.0, 5.0, 0.0},
                    {1.0, 0.0}}};
  const double lambda = scene.waveform.wavelength();
  scene.tx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.0, 2.0 * lambda, 0.0}};
  scene.rx_positions = {sim::Vec3{0.0, 0.0, 0.0},
                        sim::Vec3{0.0, 0.5 * lambda, 0.0}};
  return scene;
}

double cube_diff(const std::shared_ptr<holoscan::Tensor>& dev,
                 const std::vector<std::complex<double>>& cpu) {
  const auto host = to_host(*dev);
  EXPECT_EQ(static_cast<size_t>(host.size()), cpu.size());
  const auto* g = host.as<std::complex<float>>();
  double worst = 0.0;
  for (size_t i = 0; i < cpu.size(); ++i) {
    const std::complex<float> ref(static_cast<float>(cpu[i].real()),
                                  static_cast<float>(cpu[i].imag()));
    worst = std::max(worst, static_cast<double>(std::abs(g[i] - ref)));
  }
  std::fprintf(stderr, "[sim_gpu_mimo] max|gpu-cpu| = %.3e\n", worst);
  return worst;
}

TEST(SimGpuMimo, TdmCriAmbiguousMatchesCpu) {
  auto scene = mimo_scene();
  scene.targets[0].position[1] +=
      0.5 * scene.propagation_speed *
      scene.waveform.chirp_repetition_interval;
  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
  const int64_t slow = num_tx * scene.num_chirps;
  rd::TXSchedule sched;
  sched.num_tx = num_tx;
  const auto pattern = sched.expand_chirp_tx_index(slow);
  std::vector<std::complex<double>> w(static_cast<size_t>(num_tx * slow),
                                      {0.0, 0.0});
  for (int64_t k = 0; k < slow; ++k) {
    w[static_cast<size_t>(pattern[static_cast<size_t>(k)] * slow + k)] = {1.0,
                                                                          0.0};
  }
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto dev = sim::simulate_mimo_gpu(scene, w, slow, stream);
  const auto cpu = sim::simulate_mimo_tdm_interleaved(scene, sched);
  cudaStreamDestroy(stream);
  EXPECT_LT(cube_diff(dev, cpu.cube), 1e-3);
}

TEST(SimGpuMimo, DdmCodedCriAmbiguousMatchesCpu) {
  auto scene = mimo_scene();
  scene.targets[0].position[1] +=
      0.5 * scene.propagation_speed *
      scene.waveform.chirp_repetition_interval;
  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
  const int64_t slow = scene.num_chirps;
  const auto offsets = sim::uniform_ddm_offsets(num_tx);
  std::vector<std::complex<double>> w(static_cast<size_t>(num_tx * slow));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t k = 0; k < slow; ++k) {
      const double ph = 2.0 * kPi * offsets[static_cast<size_t>(tx)] *
                        static_cast<double>(k);
      w[static_cast<size_t>(tx * slow + k)] = {std::cos(ph), std::sin(ph)};
    }
  }
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto dev = sim::simulate_mimo_gpu(scene, w, slow, stream);
  const auto cpu = sim::simulate_mimo_ddm_coded(scene, offsets);
  cudaStreamDestroy(stream);
  EXPECT_LT(cube_diff(dev, cpu.cube), 1e-3);
}

TEST(SimGpuMimo, DdmPhaseCodedPreMixerCriAmbiguousMatchesCpu) {
  auto scene = mimo_scene();
  scene.dechirped = false;
  scene.targets[0].position[1] +=
      0.5 * scene.propagation_speed *
      scene.waveform.chirp_repetition_interval;
  scene.range_loss = {
      sim::RangeLossModel::kMonostaticFreeSpace, 40.0};
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
  const int64_t slow = scene.num_chirps;
  std::vector<std::complex<double>> codes(static_cast<size_t>(num_tx * slow));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t k = 0; k < slow; ++k) {
      const double ph = 0.3 * static_cast<double>(tx) +
                        0.11 * static_cast<double>(k);
      codes[static_cast<size_t>(tx * slow + k)] = {std::cos(ph), std::sin(ph)};
    }
  }
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  auto dev = sim::simulate_mimo_gpu(scene, codes, slow, stream);
  const auto cpu = sim::simulate_mimo_ddm_phase_coded(scene, codes);
  cudaStreamDestroy(stream);
  EXPECT_LT(cube_diff(dev, cpu.cube), 1e-3);
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
