// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "radar_common/dsp/ddm_code_generators.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/math/constants.hpp"

namespace {

namespace detail = holoscan::radar::detail;
using holoscan::radar::DDMCodeDescriptor;
using holoscan::radar::kPi;
using holoscan::radar::make_sinusoidal_ddm_stress_code;
using holoscan::radar::make_zadoff_chu_ddm_code;

void expect_nonlinear_descriptor(const DDMCodeDescriptor& code, int64_t num_tx,
                                 int64_t num_chirps) {
  EXPECT_EQ(code.num_tx, num_tx);
  EXPECT_TRUE(code.doppler_offsets.empty());
  EXPECT_EQ(code.guard_bands, 0);
  ASSERT_EQ(code.phase_codes.size(), static_cast<std::size_t>(num_tx * num_chirps));
  for (const std::complex<double>& sample : code.phase_codes) {
    EXPECT_NEAR(std::abs(sample), 1.0, 1e-12);
  }
  EXPECT_NO_THROW(code.validate());
  EXPECT_EQ(detail::ddm_classify_codes(code), detail::DdmCodeMode::kNonLinear);
}

TEST(DdmCodeGenerators, SinusoidalStressCodeMatchesRecipe) {
  constexpr int64_t kNumTx = 3;
  constexpr int64_t kNumChirps = 11;
  const DDMCodeDescriptor code = make_sinusoidal_ddm_stress_code(kNumTx, kNumChirps);
  expect_nonlinear_descriptor(code, kNumTx, kNumChirps);

  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t chirp = 0; chirp < kNumChirps; ++chirp) {
      const double phase =
          std::sin(0.7 * static_cast<double>(chirp) + static_cast<double>(tx)) * 2.0;
      const std::complex<double> expected = std::polar(1.0, phase);
      const auto actual = code.phase_codes[static_cast<std::size_t>(tx * kNumChirps + chirp)];
      EXPECT_NEAR(actual.real(), expected.real(), 1e-12);
      EXPECT_NEAR(actual.imag(), expected.imag(), 1e-12);
    }
  }
}

TEST(DdmCodeGenerators, ZadoffChuEvenLengthUsesQuadraticPhase) {
  constexpr int64_t kNumTx = 2;
  constexpr int64_t kNumChirps = 8;
  const DDMCodeDescriptor code = make_zadoff_chu_ddm_code(kNumTx, kNumChirps);
  expect_nonlinear_descriptor(code, kNumTx, kNumChirps);

  const std::array<int64_t, kNumTx> roots{1, 3};
  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t chirp = 0; chirp < kNumChirps; ++chirp) {
      const double k = static_cast<double>(chirp);
      const double phase =
          -kPi * static_cast<double>(roots[tx]) * k * k / static_cast<double>(kNumChirps);
      const std::complex<double> expected = std::polar(1.0, phase);
      const auto actual = code.phase_codes[static_cast<std::size_t>(tx * kNumChirps + chirp)];
      EXPECT_NEAR(actual.real(), expected.real(), 1e-12);
      EXPECT_NEAR(actual.imag(), expected.imag(), 1e-12);
    }
  }
}

TEST(DdmCodeGenerators, ZadoffChuOddLengthUsesAdjacentProductPhase) {
  constexpr int64_t kNumTx = 3;
  constexpr int64_t kNumChirps = 7;
  const DDMCodeDescriptor code = make_zadoff_chu_ddm_code(kNumTx, kNumChirps);
  expect_nonlinear_descriptor(code, kNumTx, kNumChirps);

  const std::array<int64_t, kNumTx> roots{1, 2, 3};
  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t chirp = 0; chirp < kNumChirps; ++chirp) {
      const double k = static_cast<double>(chirp);
      const double phase =
          -kPi * static_cast<double>(roots[tx]) * k * (k + 1.0) / static_cast<double>(kNumChirps);
      const std::complex<double> expected = std::polar(1.0, phase);
      const auto actual = code.phase_codes[static_cast<std::size_t>(tx * kNumChirps + chirp)];
      EXPECT_NEAR(actual.real(), expected.real(), 1e-12);
      EXPECT_NEAR(actual.imag(), expected.imag(), 1e-12);
    }
  }
}

TEST(DdmCodeGenerators, GenerationIsDeterministic) {
  EXPECT_EQ(make_zadoff_chu_ddm_code(2, 64).phase_codes,
            make_zadoff_chu_ddm_code(2, 64).phase_codes);
  EXPECT_EQ(make_sinusoidal_ddm_stress_code(2, 64).phase_codes,
            make_sinusoidal_ddm_stress_code(2, 64).phase_codes);
}

TEST(DdmCodeGenerators, RejectsInvalidExtentsAndInsufficientRoots) {
  EXPECT_THROW(make_zadoff_chu_ddm_code(0, 64), std::invalid_argument);
  EXPECT_THROW(make_zadoff_chu_ddm_code(2, 0), std::invalid_argument);
  EXPECT_THROW(make_zadoff_chu_ddm_code(2, 2), std::invalid_argument);
  EXPECT_THROW(make_sinusoidal_ddm_stress_code(0, 64), std::invalid_argument);
  EXPECT_THROW(make_sinusoidal_ddm_stress_code(2, 0), std::invalid_argument);
}

TEST(DdmCodeGenerators, RejectsUnrepresentableCountsBeforeSearchingRootsOrAllocating) {
  constexpr int64_t huge = std::numeric_limits<int64_t>::max();
  for (const auto generator : {make_zadoff_chu_ddm_code, make_sinusoidal_ddm_stress_code}) {
    EXPECT_THROW(generator(huge, huge), std::invalid_argument);
    EXPECT_THROW(generator(huge, 2), std::invalid_argument);
    EXPECT_THROW(generator(1, huge / 16 + 1), std::invalid_argument);
    EXPECT_THROW(generator(-1, 64), std::invalid_argument);
  }
}

}  // namespace
