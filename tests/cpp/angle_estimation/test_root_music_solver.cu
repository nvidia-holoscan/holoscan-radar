// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "angle_estimation/root_music_estimator_op/root_music_stage.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

template <typename T>
void check_mixed_companions() {
  using Complex = cuda::std::complex<T>;
  using holoscan::radar::complex_dtype;
  using holoscan::radar::detail::require_mutable;
  using holoscan::radar::make_device_tensor;
  constexpr int64_t kDegree = 2;
  constexpr int64_t kItems = 3;
  auto coefficients = make_device_tensor({kItems, kDegree + 1}, complex_dtype<T>());
  auto companion = make_device_tensor({kItems, kDegree, kDegree}, complex_dtype<T>());
  auto roots = make_device_tensor({kItems, kDegree}, complex_dtype<T>());
  auto counts = make_device_tensor({kItems}, DLDataType{kDLInt, 64, 1});
  const std::vector<int64_t> valid{16, 16, 0};
  ASSERT_EQ(cudaMemcpy(require_mutable<int64_t>(*counts, "counts"), valid.data(),
                       valid.size() * sizeof(int64_t), cudaMemcpyHostToDevice),
            cudaSuccess);
  holoscan::radar::detail::RootMUSICSolverState solver;
  for (int tick = 0; tick < 3; ++tick) {
    // z^2 - 1/4 has the analytic roots +/-1/2. The other polynomial
    // has zero leading coefficient; swap their item positions on each tick
    // to expose stale validity/info/roots when reusing solver workspaces.
    std::vector<Complex> coeff{{T{-0.25}, T{0}}, {T{0}, T{0}}, {T{1}, T{0}},
                               {T{0}, T{0}},     {T{1}, T{0}}, {T{0}, T{0}},
                               {T{0}, T{0}},     {T{1}, T{0}}, {T{0}, T{0}}};
    const int finite_item = tick % 2;
    if (finite_item == 1) {
      for (int index = 0; index < 3; ++index) {
        std::swap(coeff[index], coeff[index + 3]);
      }
    }
    ASSERT_EQ(cudaMemcpy(require_mutable<Complex>(*coefficients, "coefficients"), coeff.data(),
                         coeff.size() * sizeof(Complex), cudaMemcpyHostToDevice),
              cudaSuccess);
    holoscan::radar::detail::root_music_companion_kernel<T>
        <<<kItems, 256>>>(static_cast<const Complex*>(coefficients->data()),
                          static_cast<const int64_t*>(counts->data()), kItems, kDegree,
                          require_mutable<Complex>(*companion, "companion"));
    ASSERT_NO_THROW(holoscan::radar::detail::solve_companion_roots<T>(
        &solver, kDegree, require_mutable<Complex>(*companion, "companion"),
        require_mutable<Complex>(*roots, "roots"), kItems, nullptr));
    const auto host = holoscan::radar::testing::to_host(*roots);
    const auto values = host.template as<Complex>();
    const T a = values[finite_item * kDegree].real();
    const T b = values[finite_item * kDegree + 1].real();
    EXPECT_NEAR(a + b, T{0}, 1e-6);
    EXPECT_NEAR(a * b, T{-0.25}, 1e-6);
    EXPECT_NEAR(values[finite_item * kDegree].imag(), T{0}, 1e-6);
    EXPECT_NEAR(values[finite_item * kDegree + 1].imag(), T{0}, 1e-6);
    const int blank_item = 1 - finite_item;
    for (int root = 0; root < kDegree; ++root) {
      EXPECT_TRUE(std::isnan(values[blank_item * kDegree + root].real()));
      EXPECT_TRUE(std::isnan(values[blank_item * kDegree + root].imag()));
      EXPECT_EQ(values[2 * kDegree + root], Complex(T{0}, T{0}));
    }
  }
}

TEST(RootMusicSolver, FloatMixedDegenerateAndFiniteItemsAcrossFrames) {
  check_mixed_companions<float>();
}

TEST(RootMusicSolver, DoubleMixedDegenerateAndFiniteItemsAcrossFrames) {
  check_mixed_companions<double>();
}

}  // namespace
