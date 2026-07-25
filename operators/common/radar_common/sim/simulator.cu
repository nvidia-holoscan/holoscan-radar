// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MatX implementation of the GPU simulators (declared in simulator.hpp). Mirrors
// the closed-form model in scene.hpp: per target, per TX, per
// (rx, slot), a two-way delay tau and beat frequency drive the fast-time phase
// ramp; contributions (weighted by the per-(tx, slot) excitation) accumulate in
// double, then cast to the complex<float> ADC cube. The small per-(rx, slot) tau
// factors are computed host-side (cheap, convention-dependent) and uploaded; the
// O(rx*slow*samples) synthesis runs on the device.
//
// Single-TX and MIMO share one accumulate kernel (accumulate_scene): single-TX
// is just num_tx == 1 with a unit weight, so there is a single physics
// implementation; the public entry points differ only in output shape (rank-2
// SISO vs rank-3), the optional pre-mixer multiply, and how the weights are set.

#include "radar_common/sim/simulator.hpp"

#include <cuda/std/complex>

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "radar_common/math/constants.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/matx_utils.hpp"
#include "radar_common/tensor/tensor_utils.hpp"

namespace holoscan::radar::sim {

namespace {

// Synchronous H2D upload of a 2-D (rows, cols) double array (blocking so the
// host staging vector can be freed on return).
matx::tensor_t<double, 2> upload2d(const std::vector<double>& h, int64_t rows,
                                   int64_t cols, cudaStream_t stream) {
  auto d = matx::make_tensor<double, 2>({rows, cols}, matx::MATX_DEVICE_MEMORY);
  if (cudaMemcpyAsync(d.Data(), h.data(), sizeof(double) * h.size(),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("simulator: H2D copy failed");
  }
  return d;
}

// Fast-time sample times t_s = s / fs (samples,), on the device.
matx::tensor_t<double, 1> sample_times_device(int64_t samples, double fs,
                                              cudaStream_t stream) {
  std::vector<double> t_host(static_cast<size_t>(samples));
  for (int64_t s = 0; s < samples; ++s) {
    t_host[static_cast<size_t>(s)] = static_cast<double>(s) / fs;
  }
  return upload_coefficients<double>(t_host, stream);
}

// Accumulate the scene's clean scatterer response into `acc`
// (num_rx, slow_time, samples), summing over every target and TX with the
// per-(tx, slot) excitation `weights` (row-major (num_tx, slow_time)). The one
// physics kernel behind both simulate_gpu (num_tx == 1, unit weights) and
// simulate_mimo_gpu.
void accumulate_scene(matx::tensor_t<cuda::std::complex<double>, 3>& acc,
                      const SimScene& scene,
                      const std::vector<std::complex<double>>& weights,
                      int64_t slow_time,
                      const matx::tensor_t<double, 1>& t_dev,
                      matx::cudaExecutor& exec, cudaStream_t stream) {
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
  const int64_t samples = scene.samples_per_chirp;
  const double c = scene.propagation_speed;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double pri = scene.waveform.chirp_repetition_interval;
  const double two_pi = 2.0 * kPi;
  const cuda::std::complex<double> kI(0.0, 1.0);

  for (const PointTarget& target : scene.targets) {
    const TargetTruth truth = derive_truth(target);
    const cuda::std::complex<double> amp(target.amplitude.real(),
                                         target.amplitude.imag());
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      // Per-(rx, slot) beat frequency (Hz) + slow-time carrier phase (rad) for
      // this TX's aperture.
      std::vector<double> beat_host(static_cast<size_t>(num_rx * slow_time));
      std::vector<double> slow_host(static_cast<size_t>(num_rx * slow_time));
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const double aperture =
            dot(scene.tx_positions[static_cast<size_t>(tx)],
                truth.look_direction) +
            dot(scene.rx_positions[static_cast<size_t>(rx)],
                truth.look_direction);
        for (int64_t k = 0; k < slow_time; ++k) {
          const double range_k =
              truth.range + truth.radial_velocity * static_cast<double>(k) * pri;
          const double tau = (2.0 * range_k - aperture) / c;
          const size_t idx = static_cast<size_t>(rx * slow_time + k);
          beat_host[idx] = slope * tau;
          slow_host[idx] = -two_pi * f_c * tau;
        }
      }
      auto beat_rk = upload2d(beat_host, num_rx, slow_time, stream);
      auto slow_rk = upload2d(slow_host, num_rx, slow_time, stream);

      // This TX's excitation weight per slot (broadcast over rx and samples).
      std::vector<std::complex<double>> w_host(
          weights.begin() + tx * slow_time,
          weights.begin() + (tx + 1) * slow_time);
      auto w_dev = upload_complex_coefficients<double>(w_host, stream);

      // Broadcast to (rx, slot, samples): phase = 2*pi*beat*t_s + slow.
      auto beat3 = matx::clone<3>(
          beat_rk, {matx::matxKeepDim, matx::matxKeepDim, samples});
      auto slow3 = matx::clone<3>(
          slow_rk, {matx::matxKeepDim, matx::matxKeepDim, samples});
      auto t3 = matx::clone<3>(t_dev, {num_rx, slow_time, matx::matxKeepDim});
      auto w3 = matx::clone<3>(w_dev, {num_rx, matx::matxKeepDim, samples});
      auto phase = two_pi * beat3 * t3 + slow3;
      (acc = acc + amp * w3 * matx::exp(kI * phase)).run(exec);
    }
  }
}

}  // namespace

std::shared_ptr<holoscan::Tensor> simulate_gpu(const SimScene& scene,
                                               cudaStream_t stream) {
  if (scene.rx_positions.empty()) {
    throw std::invalid_argument("simulate_gpu: need >= 1 RX element");
  }
  if (scene.tx_positions.size() != 1) {
    throw std::invalid_argument(
        "simulate_gpu: models a single TX; use the MIMO GPU path for multi-TX");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_chirps = scene.num_chirps;
  const int64_t samples = scene.samples_per_chirp;

  const std::vector<int64_t> shape =
      (num_rx == 1) ? std::vector<int64_t>{num_chirps, samples}
                    : std::vector<int64_t>{num_rx, num_chirps, samples};
  auto out = make_device_tensor(shape, complex_dtype<float>());

  matx::cudaExecutor exec{stream};
  auto t_dev = sample_times_device(samples, scene.waveform.sample_rate, stream);
  auto acc = matx::make_tensor<cuda::std::complex<double>, 3>(
      {num_rx, num_chirps, samples}, matx::MATX_DEVICE_MEMORY);
  (acc = cuda::std::complex<double>(0.0, 0.0)).run(exec);

  // Single TX, unit excitation on every chirp: slow_time = num_chirps.
  const std::vector<std::complex<double>> weights(
      static_cast<size_t>(num_chirps), {1.0, 0.0});
  accumulate_scene(acc, scene, weights, num_chirps, t_dev, exec, stream);

  // Pre-mixer output: multiply the dechirped model back up by the reference
  // chirp r[s] = exp(+i*pi*slope*t_s^2).
  if (!scene.dechirped) {
    const double fs = scene.waveform.sample_rate;
    const double slope = scene.waveform.slope();
    std::vector<std::complex<double>> ref_host(static_cast<size_t>(samples));
    for (int64_t s = 0; s < samples; ++s) {
      const double t = static_cast<double>(s) / fs;
      const double ph = kPi * slope * t * t;
      ref_host[static_cast<size_t>(s)] = {std::cos(ph), std::sin(ph)};
    }
    auto ref_dev = upload_complex_coefficients<double>(ref_host, stream);
    auto ref3 = matx::clone<3>(ref_dev, {num_rx, num_chirps, matx::matxKeepDim});
    (acc = acc * ref3).run(exec);
  }

  auto out_view =
      make_complex_view3<float>(out->data(), num_rx, num_chirps, samples);
  (out_view = matx::as_type<cuda::std::complex<float>>(acc)).run(exec);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("simulate_gpu: final stream sync failed");
  }
  return out;
}

std::shared_ptr<holoscan::Tensor> simulate_mimo_gpu(
    const SimScene& scene, const std::vector<std::complex<double>>& weights,
    int64_t slow_time, cudaStream_t stream) {
  if (scene.rx_positions.empty() || scene.tx_positions.empty()) {
    throw std::invalid_argument("simulate_mimo_gpu: need >= 1 TX and RX");
  }
  const int64_t num_rx = static_cast<int64_t>(scene.rx_positions.size());
  const int64_t num_tx = static_cast<int64_t>(scene.tx_positions.size());
  const int64_t samples = scene.samples_per_chirp;
  if (static_cast<int64_t>(weights.size()) != num_tx * slow_time) {
    throw std::invalid_argument(
        "simulate_mimo_gpu: weights must be (num_tx * slow_time)");
  }

  auto out = make_device_tensor({num_rx, slow_time, samples},
                                complex_dtype<float>());
  matx::cudaExecutor exec{stream};
  auto t_dev = sample_times_device(samples, scene.waveform.sample_rate, stream);
  auto acc = matx::make_tensor<cuda::std::complex<double>, 3>(
      {num_rx, slow_time, samples}, matx::MATX_DEVICE_MEMORY);
  (acc = cuda::std::complex<double>(0.0, 0.0)).run(exec);

  accumulate_scene(acc, scene, weights, slow_time, t_dev, exec, stream);

  auto out_view =
      make_complex_view3<float>(out->data(), num_rx, slow_time, samples);
  (out_view = matx::as_type<cuda::std::complex<float>>(acc)).run(exec);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("simulate_mimo_gpu: final stream sync failed");
  }
  return out;
}

void add_thermal_noise_gpu(holoscan::Tensor& cube,
                           const std::optional<double>& snr_db, uint64_t seed,
                           cudaStream_t stream) {
  if (!snr_db.has_value()) {
    return;
  }
  const auto shape = cube.shape();
  int64_t num_rx = 1;
  int64_t num_chirps = 0;
  int64_t samples = 0;
  if (shape.size() == 2) {
    num_chirps = shape[0];
    samples = shape[1];
  } else if (shape.size() == 3) {
    num_rx = shape[0];
    num_chirps = shape[1];
    samples = shape[2];
  } else {
    throw std::runtime_error(
        "add_thermal_noise_gpu: expected a rank-2 or rank-3 cube");
  }

  const double noise_power = std::pow(10.0, -(*snr_db) / 10.0);
  const float sigma = static_cast<float>(std::sqrt(noise_power / 2.0));

  matx::cudaExecutor exec{stream};
  // View the complex<float> buffer as real<float> with a doubled last dim:
  // adding independent N(0, sigma) to each real/imag component yields complex
  // white Gaussian noise with per-component std sigma (matches the CPU model).
  auto real_view =
      make_real_view3<float>(cube.data(), num_rx, num_chirps, 2 * samples);
  auto noise = matx::make_tensor<float, 3>({num_rx, num_chirps, 2 * samples},
                                           matx::MATX_DEVICE_MEMORY);
  (noise = matx::random<float>(noise.Shape(), matx::NORMAL, seed)).run(exec);
  (real_view = real_view + sigma * noise).run(exec);

  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("add_thermal_noise_gpu: stream sync failed");
  }
}

}  // namespace holoscan::radar::sim
