// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for MIMODeinterleaveOp: TDM slow-time gather into an explicit TX
// axis (round-robin and custom tx_order), non-uniform schedule rejection,
// dtype preservation, leading batch axes, and float/double agreement.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "fmcw/mimo_deinterleave_op/mimo_deinterleave_op.hpp"
#include "radar_common/descriptors/tx_schedule.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::MIMODeinterleaveOp;
using holoscan::radar::TXSchedule;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;
using holoscan::test::create_operator_test;

constexpr int64_t kNumRx = 2;
constexpr int64_t kNumTx = 2;
constexpr int64_t kChirpsPerTx = 3;
constexpr int64_t kSamples = 4;
constexpr int64_t kSlowTime = kNumTx * kChirpsPerTx;

double marker_value(int64_t tx, int64_t l, int64_t rx, int64_t sample) {
  return 1000.0 * static_cast<double>(tx) +
         10.0 * static_cast<double>(l) +
         0.1 * static_cast<double>(rx) +
         0.01 * static_cast<double>(sample);
}

std::vector<int64_t> pattern_for(const TXSchedule& schedule) {
  TXSchedule copy = schedule;
  copy.validate();
  return copy.expand_chirp_tx_index(kSlowTime);
}

template <typename ElementT>
std::vector<ElementT> build_interleaved_cube(
    const std::vector<int64_t>& pattern, const std::vector<int64_t>& shape) {
  const int64_t num_rx = shape[shape.size() - 3];
  const int64_t slow_time = shape[shape.size() - 2];
  const int64_t samples = shape.back();
  std::vector<ElementT> data(static_cast<size_t>(num_rx * slow_time * samples));
  std::vector<int64_t> seen(static_cast<size_t>(kNumTx), 0);
  for (int64_t k = 0; k < slow_time; ++k) {
    const int64_t tx = pattern[static_cast<size_t>(k)];
    const int64_t l = seen[static_cast<size_t>(tx)]++;
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      for (int64_t s = 0; s < samples; ++s) {
        const size_t idx = static_cast<size_t>(
            (rx * slow_time + k) * samples + s);
        const double m = marker_value(tx, l, rx, s);
        if constexpr (std::is_same_v<ElementT, float> ||
                      std::is_same_v<ElementT, double>) {
          data[idx] = static_cast<ElementT>(m);
        } else {
          data[idx] = ElementT(static_cast<typename ElementT::value_type>(m),
                               static_cast<typename ElementT::value_type>(0));
        }
      }
    }
  }
  return data;
}

template <typename T, typename ElementT>
HostTensor run_deinterleave(const std::vector<ElementT>& host,
                            const std::vector<int64_t>& shape,
                            DLDataType dtype, const TXSchedule& schedule) {
  TensorMapCapture capture;
  auto harness = create_operator_test<MIMODeinterleaveOp<T>>(
      holoscan::Arg("tx_schedule") = schedule);
  auto frame = single_map(to_device_tensor(host, shape, dtype), "adc_cube");
  harness->template add_input_port<holoscan::TensorMap>("adc_cube", {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "adc_cube", capture.validators());
  harness->run_test();
  if (capture.frames.size() != 1) {
    throw std::runtime_error("expected exactly one captured output frame");
  }
  return std::move(capture.frames.front());
}

template <typename ElementT>
void expect_deinterleaved(const HostTensor& out,
                          const std::vector<int64_t>& shape) {
  const int64_t num_tx = shape[shape.size() - 4];
  const int64_t num_rx = shape[shape.size() - 3];
  const int64_t chirps_per_tx = shape[shape.size() - 2];
  const int64_t samples = shape.back();
  ASSERT_EQ(shape.size(), 4U);
  ASSERT_EQ(num_tx, kNumTx);
  ASSERT_EQ(num_rx, kNumRx);
  ASSERT_EQ(chirps_per_tx, kChirpsPerTx);
  ASSERT_EQ(samples, kSamples);

  const auto* data = out.as<ElementT>();
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      for (int64_t l = 0; l < chirps_per_tx; ++l) {
        for (int64_t s = 0; s < samples; ++s) {
          const size_t idx = static_cast<size_t>(
              (((tx * num_rx + rx) * chirps_per_tx + l) * samples) + s);
          const double expected = marker_value(tx, l, rx, s);
          double got = 0.0;
          if constexpr (std::is_same_v<ElementT, float> ||
                        std::is_same_v<ElementT, double>) {
            got = static_cast<double>(data[idx]);
          } else {
            got = static_cast<double>(data[idx].real());
          }
          EXPECT_NEAR(got, expected, 1e-4) << "tx=" << tx << " rx=" << rx
                                           << " l=" << l << " s=" << s;
        }
      }
    }
  }
}

TXSchedule round_robin_schedule() {
  TXSchedule schedule;
  schedule.num_tx = kNumTx;
  return schedule;
}

TXSchedule custom_order_schedule() {
  TXSchedule schedule;
  schedule.num_tx = kNumTx;
  schedule.tx_order = {1, 0};
  return schedule;
}

TEST(MIMODeinterleaveOp, RoundRobinComplexFloat) {
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = round_robin_schedule();
  const auto pattern = pattern_for(schedule);
  const auto host =
      build_interleaved_cube<std::complex<float>>(pattern, shape);
  const HostTensor out = run_deinterleave<float>(
      host, shape, complex_dtype<float>(), schedule);
  EXPECT_TRUE(
      holoscan::radar::dldatatype_equals(out.dtype, complex_dtype<float>()));
  expect_deinterleaved<std::complex<float>>(out, out.shape);
}

TEST(MIMODeinterleaveOp, CustomTxOrderComplexFloat) {
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = custom_order_schedule();
  const auto pattern = pattern_for(schedule);
  const auto host =
      build_interleaved_cube<std::complex<float>>(pattern, shape);
  const HostTensor out = run_deinterleave<float>(
      host, shape, complex_dtype<float>(), schedule);
  expect_deinterleaved<std::complex<float>>(out, out.shape);
}

TEST(MIMODeinterleaveOp, RealIfPreserved) {
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = round_robin_schedule();
  const auto pattern = pattern_for(schedule);
  const auto host = build_interleaved_cube<float>(pattern, shape);
  const HostTensor out =
      run_deinterleave<float>(host, shape, real_dtype<float>(), schedule);
  EXPECT_TRUE(
      holoscan::radar::dldatatype_equals(out.dtype, real_dtype<float>()));
  expect_deinterleaved<float>(out, out.shape);
}

TEST(MIMODeinterleaveOp, LeadingBatchAxis) {
  const std::vector<int64_t> shape = {2, kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = round_robin_schedule();
  const auto pattern = pattern_for(schedule);
  const std::vector<int64_t> slice_shape = {kNumRx, kSlowTime, kSamples};
  auto slice =
      build_interleaved_cube<std::complex<float>>(pattern, slice_shape);
  std::vector<std::complex<float>> host(slice.size() * 2);
  host.assign(slice.begin(), slice.end());
  host.insert(host.end(), slice.begin(), slice.end());
  const HostTensor out = run_deinterleave<float>(
      host, shape, complex_dtype<float>(), schedule);
  ASSERT_EQ(out.shape.size(), 5U);
  EXPECT_EQ(out.shape[0], 2);
}

TEST(MIMODeinterleaveOp, RejectsNonUniformSchedule) {
  TXSchedule schedule;
  schedule.num_tx = kNumTx;
  schedule.chirp_tx_index = {0, 1, 0, 1, 0, 0};
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const auto host = std::vector<std::complex<float>>(
      static_cast<size_t>(kNumRx * kSlowTime * kSamples), {1.0f, 0.0f});
  EXPECT_THROW(
      (void)run_deinterleave<float>(host, shape, complex_dtype<float>(),
                                    schedule),
      std::invalid_argument);
}

TEST(MIMODeinterleaveOp, RejectsIndivisibleSlowTime) {
  TXSchedule schedule = round_robin_schedule();
  const std::vector<int64_t> shape = {kNumRx, kSlowTime + 1, kSamples};
  const auto host = std::vector<std::complex<float>>(
      static_cast<size_t>(kNumRx * (kSlowTime + 1) * kSamples), {1.0f, 0.0f});
  EXPECT_THROW(
      (void)run_deinterleave<float>(host, shape, complex_dtype<float>(),
                                    schedule),
      std::invalid_argument);
}

TEST(MIMODeinterleaveOp, FloatDoubleAgreement) {
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = round_robin_schedule();
  const auto pattern = pattern_for(schedule);
  const auto host_f =
      build_interleaved_cube<std::complex<float>>(pattern, shape);
  const auto host_d =
      build_interleaved_cube<std::complex<double>>(pattern, shape);
  const HostTensor out_f = run_deinterleave<float>(
      host_f, shape, complex_dtype<float>(), schedule);
  const HostTensor out_d = run_deinterleave<double>(
      host_d, shape, complex_dtype<double>(), schedule);
  ASSERT_EQ(out_f.shape, out_d.shape);
  const auto* f = out_f.as<std::complex<float>>();
  const auto* d = out_d.as<std::complex<double>>();
  for (int64_t i = 0; i < out_f.size(); ++i) {
    EXPECT_NEAR(static_cast<double>(f[i].real()),
                static_cast<double>(d[i].real()), 1e-4);
    EXPECT_NEAR(static_cast<double>(f[i].imag()),
                static_cast<double>(d[i].imag()), 1e-4);
  }
}

}  // namespace
