// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// MatX implementation of simulate_gpu (declared in sim_gpu.hpp). Mirrors the
// closed-form model in point_target_sim.hpp::simulate(): per target, per
// (rx, chirp), a two-way delay tau and beat frequency drive the fast-time phase
// ramp; contributions accumulate in double, then cast to the complex<float> ADC
// cube. The small per-(rx, chirp) tau factors are computed host-side (cheap,
// convention-dependent) and uploaded; the O(rx*chirps*samples) synthesis runs
// on the device.

#include "radar_common/sim/sim_gpu.hpp"

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
  const double c = scene.propagation_speed;
  const double fs = scene.waveform.sample_rate;
  const double slope = scene.waveform.slope();
  const double f_c = scene.waveform.carrier_frequency;
  const double pri = scene.waveform.chirp_repetition_interval;
  const double two_pi = 2.0 * kPi;

  const std::vector<int64_t> shape =
      (num_rx == 1) ? std::vector<int64_t>{num_chirps, samples}
                    : std::vector<int64_t>{num_rx, num_chirps, samples};
  auto out = make_device_tensor(shape, complex_dtype<float>());

  matx::cudaExecutor exec{stream};

  // Host→device upload of a 2-D (rows, cols) double array (synchronous so the
  // host staging vector can be freed on return).
  const auto upload2d = [&](const std::vector<double>& h, int64_t rows,
                            int64_t cols) {
    auto d = matx::make_tensor<double, 2>({rows, cols}, matx::MATX_DEVICE_MEMORY);
    if (cudaMemcpyAsync(d.Data(), h.data(), sizeof(double) * h.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
      throw std::runtime_error("simulate_gpu: H2D copy failed");
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("simulate_gpu: stream sync failed");
    }
    return d;
  };

  // Fast-time sample times t_s = s / fs (samples,), device double.
  std::vector<double> t_host(static_cast<size_t>(samples));
  for (int64_t s = 0; s < samples; ++s) {
    t_host[static_cast<size_t>(s)] = static_cast<double>(s) / fs;
  }
  auto t_dev = upload_coefficients<double>(t_host, stream);  // (samples,)

  // Double-precision accumulator (rx, chirps, samples), zero-initialized.
  auto acc = matx::make_tensor<cuda::std::complex<double>, 3>(
      {num_rx, num_chirps, samples}, matx::MATX_DEVICE_MEMORY);
  (acc = cuda::std::complex<double>(0.0, 0.0)).run(exec);

  const cuda::std::complex<double> kI(0.0, 1.0);
  for (const PointTarget& target : scene.targets) {
    const TargetTruth truth = derive_truth(target);
    // Per-(rx, chirp) beat frequency (Hz) and slow-time carrier phase (rad).
    std::vector<double> beat_host(static_cast<size_t>(num_rx * num_chirps));
    std::vector<double> slow_host(static_cast<size_t>(num_rx * num_chirps));
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      const double aperture =
          dot(scene.tx_positions[0], truth.look_direction) +
          dot(scene.rx_positions[static_cast<size_t>(rx)], truth.look_direction);
      for (int64_t k = 0; k < num_chirps; ++k) {
        const double range_k =
            truth.range + truth.radial_velocity * static_cast<double>(k) * pri;
        const double tau = (2.0 * range_k - aperture) / c;
        const size_t idx = static_cast<size_t>(rx * num_chirps + k);
        beat_host[idx] = slope * tau;         // Hz
        slow_host[idx] = -two_pi * f_c * tau;  // rad
      }
    }
    auto beat_rk = upload2d(beat_host, num_rx, num_chirps);
    auto slow_rk = upload2d(slow_host, num_rx, num_chirps);

    // Broadcast to (rx, chirps, samples): phase = 2*pi*beat*t_s + slow.
    auto beat3 =
        matx::clone<3>(beat_rk, {matx::matxKeepDim, matx::matxKeepDim, samples});
    auto slow3 =
        matx::clone<3>(slow_rk, {matx::matxKeepDim, matx::matxKeepDim, samples});
    auto t3 = matx::clone<3>(t_dev, {num_rx, num_chirps, matx::matxKeepDim});
    auto phase = two_pi * beat3 * t3 + slow3;  // (rx, chirps, samples) real

    const cuda::std::complex<double> amp(target.amplitude.real(),
                                         target.amplitude.imag());
    (acc = acc + amp * matx::exp(kI * phase)).run(exec);
  }

  // Pre-mixer output: multiply the dechirped model back up by the reference
  // chirp r[s] = exp(+i*pi*slope*t_s^2).
  if (!scene.dechirped) {
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

  // Cast the double accumulator to the complex<float> ADC cube.
  auto out_view = make_complex_view3<float>(out->data(), num_rx, num_chirps,
                                            samples);
  (out_view = matx::as_type<cuda::std::complex<float>>(acc)).run(exec);

  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("simulate_gpu: final stream sync failed");
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
