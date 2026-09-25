// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "fmcw/ddm_power_fold_op/ddm_power_fold_op.hpp"
#include "radar_common/dsp/mimo_coherent_compensation_stage.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"
#include "utils/test_extents.hpp"

namespace {
namespace hr = holoscan::radar;

hr::DDMCodeDescriptor code(int64_t guard = 0) {
  hr::DDMCodeDescriptor result;
  result.num_tx = 2;
  result.doppler_offsets = {0.5, 0.0};
  result.guard_bands = guard;
  return result;
}

template <typename T>
hr::testing::HostTensor run(const hr::DDMCodeDescriptor& ddm_code) {
  constexpr int64_t kRx = 2;
  constexpr int64_t kD = 8;
  constexpr int64_t kR = 3;
  std::vector<std::complex<T>> host(static_cast<size_t>(kRx * kD * kR));
  for (int64_t rx = 0; rx < kRx; ++rx) {
    for (int64_t d = 0; d < kD; ++d) {
      for (int64_t r = 0; r < kR; ++r) {
        host[static_cast<size_t>((rx * kD + d) * kR + r)] = {static_cast<T>(1 + rx + d),
                                                             static_cast<T>(r)};
      }
    }
  }
  hr::MaxTensorExtents extents;
  extents.max_rx = kRx;
  extents.max_doppler_bins = kD;
  extents.max_range_bins = kR;
  auto input = hr::testing::to_device_tensor(host, {kRx, kD, kR}, hr::complex_dtype<T>());
  auto fixture = holoscan::testing::create_operator_test<hr::DDMPowerFoldOp<T>>(extents);
  fixture.set_params(hr::DDMPowerFoldParams{.ddm_code = ddm_code});
  fixture.push(&hr::DDMPowerFoldOp<T>::rd_maps, *input);
  fixture.template capture_snapshot<holoscan::Tensor>("power_map", 1U, input->nbytes());
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("power_map", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("power_map", 1U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  if (!observed || !result.ok()) {
    throw std::runtime_error("DDMPowerFoldOp failed: " + result.describe_failure());
  }
  return hr::testing::to_host(result.template snapshot<holoscan::Tensor>("power_map"));
}

template <typename T>
void check_fold() {
  const auto ddm_code = code(1);
  const auto table =
      hr::detail::build_ddm_demux_table(hr::detail::ddm_resolve_offsets(ddm_code), 2, 8, 3);
  const auto output = run<T>(ddm_code);
  ASSERT_EQ(output.shape, (std::vector<int64_t>{3, 3}));
  const auto* actual = output.template as<T>();
  constexpr double kTolerance = std::is_same_v<T, float> ? 1e-5 : 1e-12;
  for (int64_t d = 0; d < 3; ++d) {
    for (int64_t r = 0; r < 3; ++r) {
      T expected = 0;
      for (int64_t tx = 0; tx < 2; ++tx) {
        const int64_t source_d = table[static_cast<size_t>(tx * 3 + d)];
        for (int64_t rx = 0; rx < 2; ++rx) {
          const T re = static_cast<T>(1 + rx + source_d);
          const T im = static_cast<T>(r);
          expected += re * re + im * im;
        }
      }
      EXPECT_NEAR(actual[d * 3 + r], expected, kTolerance);
    }
  }
}

TEST(DDMPowerFoldOp, FloatHostOracleAndGuardExclusion) {
  check_fold<float>();
}

TEST(DDMPowerFoldOp, DoubleHostOracleAndGuardExclusion) {
  check_fold<double>();
}

TEST(DDMPowerFoldMath, RejectsInvalidLinearLayouts) {
  EXPECT_THROW(hr::detail::ddm_exact_band_indices(
                   hr::DDMCodeDescriptor{.num_tx = 2, .doppler_offsets = {0.0, 0.3}}, 8),
               std::invalid_argument);
  EXPECT_THROW(hr::detail::ddm_num_doppler_per_tx(7, 2, 0), std::invalid_argument);
  auto nonlinear = hr::DDMCodeDescriptor{
      .num_tx = 2,
      .phase_codes = {{1, 0}, {1, 0}, {1, 0}, {1, 0}, {1, 0}, {0, 1}, {1, 0}, {0, -1}}};
  EXPECT_THROW(hr::detail::ddm_exact_band_indices(nonlinear, 4), std::invalid_argument);
}

}  // namespace
