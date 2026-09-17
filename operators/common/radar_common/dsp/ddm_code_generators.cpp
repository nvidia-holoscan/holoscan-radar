// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "radar_common/dsp/ddm_code_generators.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/math/constants.hpp"

namespace holoscan::radar {
namespace {

void require_generator_extents(int64_t num_tx, int64_t num_chirps,
                               const char* generator) {
  if (num_tx <= 0) {
    throw std::invalid_argument(std::string(generator) +
                                ": num_tx must be positive");
  }
  if (num_chirps <= 0) {
    throw std::invalid_argument(std::string(generator) +
                                ": num_chirps must be positive");
  }
}

}  // namespace

DDMCodeDescriptor make_zadoff_chu_ddm_code(int64_t num_tx,
                                           int64_t num_chirps) {
  constexpr const char* kGenerator = "make_zadoff_chu_ddm_code";
  require_generator_extents(num_tx, num_chirps, kGenerator);

  std::vector<int64_t> roots;
  for (int64_t root = 1;
       root < num_chirps &&
       static_cast<int64_t>(roots.size()) < num_tx;
       ++root) {
    if (std::gcd(root, num_chirps) == 1) {
      roots.push_back(root);
    }
  }
  if (static_cast<int64_t>(roots.size()) < num_tx) {
    throw std::invalid_argument(
        "make_zadoff_chu_ddm_code: need one distinct coprime root per TX; "
        "increase num_chirps");
  }

  DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.phase_codes.resize(
      static_cast<std::size_t>(num_tx * num_chirps));
  const bool odd = num_chirps % 2 != 0;
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    const double root =
        static_cast<double>(roots[static_cast<std::size_t>(tx)]);
    for (int64_t chirp = 0; chirp < num_chirps; ++chirp) {
      const double k = static_cast<double>(chirp);
      const double phase =
          odd ? -kPi * root * k * (k + 1.0) /
                    static_cast<double>(num_chirps)
              : -kPi * root * k * k /
                    static_cast<double>(num_chirps);
      code.phase_codes[static_cast<std::size_t>(
          tx * num_chirps + chirp)] = std::polar(1.0, phase);
    }
  }
  code.validate();
  return code;
}

DDMCodeDescriptor make_sinusoidal_ddm_stress_code(int64_t num_tx,
                                                  int64_t num_chirps) {
  constexpr const char* kGenerator =
      "make_sinusoidal_ddm_stress_code";
  require_generator_extents(num_tx, num_chirps, kGenerator);

  DDMCodeDescriptor code;
  code.num_tx = num_tx;
  code.phase_codes.resize(
      static_cast<std::size_t>(num_tx * num_chirps));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t chirp = 0; chirp < num_chirps; ++chirp) {
      const double phase =
          std::sin(0.7 * static_cast<double>(chirp) +
                   static_cast<double>(tx)) *
          2.0;
      code.phase_codes[static_cast<std::size_t>(
          tx * num_chirps + chirp)] = std::polar(1.0, phase);
    }
  }
  code.validate();
  return code;
}

}  // namespace holoscan::radar
