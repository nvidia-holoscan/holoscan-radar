// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CPU I/Q additives — the tests-only reference for the non-scatterer returns
// added linearly on top of a simulator's clean scatterer response. The runtime
// applies these on the device (simulator.hpp::add_thermal_noise_gpu); this host
// version is the deterministic oracle used by the tests. Additives operate on a
// raw complex IQ cube + parameters, not on a SimScene, so this header does NOT
// depend on the scene/solver headers —
// additives are independent of how the cube
// was generated.
//
// Today this holds thermal noise; future non-target returns (clutter, CW /
// interference tones, DC / IF leakage) would follow the same add_*(cube, ...)
// shape that composes linearly.

#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace holoscan::radar::sim {

// Seeded complex white Gaussian noise added linearly to a solver cube at
// `snr_db` relative to a unit-amplitude target (no-op when unset). Same
// (seed, snr_db) → a bit-identical realization; the same seed each call means a
// static realization across frames unless the caller varies the seed.
inline void add_thermal_noise(std::vector<std::complex<double>>& cube,
                              const std::optional<double>& snr_db, uint64_t seed) {
  if (!snr_db.has_value()) {
    return;
  }
  const double noise_power = std::pow(10.0, -(*snr_db) / 10.0);
  const double sigma = std::sqrt(noise_power / 2.0);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss(0.0, sigma);
  for (auto& v : cube) {
    v += std::complex<double>(gauss(rng), gauss(rng));
  }
}

}  // namespace holoscan::radar::sim
