// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for MIMODeinterleaveOp: TDM slow-time gather into an explicit TX
// axis (round-robin, custom tx_order, explicit chirp_tx_index), non-uniform
// schedule rejection, dtype preservation, leading batch axes, cross-cutting
// input validation, and float/double agreement.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
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
using holoscan::radar::testing::to_pageable_host_tensor;
using holoscan::test::create_operator_test;

constexpr int64_t kNumRx = 2;
constexpr int64_t kNumTx = 2;
constexpr int64_t kChirpsPerTx = 3;
constexpr int64_t kSamples = 4;
constexpr int64_t kSlowTime = kNumTx * kChirpsPerTx;

double marker_value(int64_t batch, int64_t tx, int64_t l, int64_t rx,
                    int64_t sample) {
  return 10000.0 * static_cast<double>(batch) +
         1000.0 * static_cast<double>(tx) +
         10.0 * static_cast<double>(l) +
         0.1 * static_cast<double>(rx) +
         0.01 * static_cast<double>(sample);
}

int64_t tensor_count(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  return count;
}

int64_t input_leading_batch(const std::vector<int64_t>& shape) {
  int64_t batch = 1;
  for (size_t i = 0; i + 3 < shape.size(); ++i) {
    batch *= shape[i];
  }
  return batch;
}

int64_t output_leading_batch(const std::vector<int64_t>& shape) {
  int64_t batch = 1;
  for (size_t i = 0; i + 4 < shape.size(); ++i) {
    batch *= shape[i];
  }
  return batch;
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
  const int64_t batch = input_leading_batch(shape);
  const int64_t slice_elems = num_rx * slow_time * samples;
  std::vector<ElementT> data(static_cast<size_t>(batch * slice_elems));
  for (int64_t b = 0; b < batch; ++b) {
    std::vector<int64_t> seen(static_cast<size_t>(kNumTx), 0);
    for (int64_t k = 0; k < slow_time; ++k) {
      const int64_t tx = pattern[static_cast<size_t>(k)];
      const int64_t l = seen[static_cast<size_t>(tx)]++;
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t s = 0; s < samples; ++s) {
          const size_t idx = static_cast<size_t>(
              b * slice_elems + (rx * slow_time + k) * samples + s);
          const double m = marker_value(b, tx, l, rx, s);
          if constexpr (std::is_same_v<ElementT, float> ||
                        std::is_same_v<ElementT, double>) {
            data[idx] = static_cast<ElementT>(m);
          } else {
            data[idx] =
                ElementT(static_cast<typename ElementT::value_type>(m),
                           static_cast<typename ElementT::value_type>(0));
          }
        }
      }
    }
  }
  return data;
}

template <typename T>
HostTensor run_harness_single(const TXSchedule& schedule,
                              holoscan::TensorMap frame) {
  TensorMapCapture capture;
  auto harness = create_operator_test<MIMODeinterleaveOp<T>>(
      holoscan::Arg("tx_schedule") = schedule);
  harness->template add_input_port<holoscan::TensorMap>("adc_cube", {frame});
  harness->template add_output_port<holoscan::TensorMap>(
      "adc_cube", capture.validators());
  harness->run_test();
  if (capture.frames.size() != 1) {
    throw std::runtime_error("expected exactly one captured output frame");
  }
  return std::move(capture.frames.front());
}

template <typename T, typename ElementT>
HostTensor run_deinterleave(const std::vector<ElementT>& host,
                            const std::vector<int64_t>& shape,
                            DLDataType dtype, const TXSchedule& schedule) {
  return run_harness_single<T>(
      schedule, single_map(to_device_tensor(host, shape, dtype), "adc_cube"));
}

template <typename ElementT>
void expect_deinterleaved(const HostTensor& out,
                          const std::vector<int64_t>& shape) {
  ASSERT_GE(shape.size(), 4U);
  const int64_t samples = shape.back();
  const int64_t chirps_per_tx = shape[shape.size() - 2];
  const int64_t num_rx = shape[shape.size() - 3];
  const int64_t num_tx = shape[shape.size() - 4];
  const int64_t batch = output_leading_batch(shape);
  ASSERT_EQ(num_tx, kNumTx);
  ASSERT_EQ(num_rx, kNumRx);
  ASSERT_EQ(chirps_per_tx, kChirpsPerTx);
  ASSERT_EQ(samples, kSamples);

  const auto* data = out.as<ElementT>();
  const int64_t slice_elems = num_tx * num_rx * chirps_per_tx * samples;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t l = 0; l < chirps_per_tx; ++l) {
          for (int64_t s = 0; s < samples; ++s) {
            const size_t idx = static_cast<size_t>(
                b * slice_elems +
                (((tx * num_rx + rx) * chirps_per_tx + l) * samples) + s);
            const double expected = marker_value(b, tx, l, rx, s);
            double got = 0.0;
            if constexpr (std::is_same_v<ElementT, float> ||
                          std::is_same_v<ElementT, double>) {
              got = static_cast<double>(data[idx]);
            } else {
              got = static_cast<double>(data[idx].real());
            }
            const double tol = std::max(1e-4, 1e-6 * std::abs(expected));
            EXPECT_NEAR(got, expected, tol)
                << "batch=" << b << " tx=" << tx << " rx=" << rx << " l=" << l
                << " s=" << s;
          }
        }
      }
    }
  }
}

struct StridedAdcCubeHolder {
  void* data{nullptr};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor dl{};

  ~StridedAdcCubeHolder() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

// Device tensor with non-compact row-major strides on adc_cube.
std::shared_ptr<holoscan::Tensor> make_non_contiguous_adc_cube(
    const std::vector<std::complex<float>>& host,
    const std::vector<int64_t>& shape) {
  const int64_t num_rx = shape[0];
  const int64_t slow_time = shape[1];
  const int64_t samples = shape[2];
  const int64_t slow_stride = samples + 1;
  const size_t alloc_elems =
      static_cast<size_t>(num_rx * slow_time * slow_stride);
  auto holder = std::make_unique<StridedAdcCubeHolder>();
  holder->shape = shape;
  holder->strides = {slow_time * slow_stride, slow_stride, 1};
  if (cudaMalloc(&holder->data, alloc_elems * sizeof(std::complex<float>)) !=
      cudaSuccess) {
    throw std::runtime_error("make_non_contiguous_adc_cube: cudaMalloc failed");
  }
  std::vector<std::complex<float>> padded(alloc_elems, {0.0f, 0.0f});
  for (int64_t rx = 0; rx < num_rx; ++rx) {
    for (int64_t k = 0; k < slow_time; ++k) {
      for (int64_t s = 0; s < samples; ++s) {
        const size_t dst = static_cast<size_t>(
            rx * slow_time * slow_stride + k * slow_stride + s);
        const size_t src =
            static_cast<size_t>((rx * slow_time + k) * samples + s);
        padded[dst] = host[src];
      }
    }
  }
  if (cudaMemcpy(holder->data, padded.data(),
                 alloc_elems * sizeof(std::complex<float>),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_non_contiguous_adc_cube: H2D failed");
  }
  int device_id = 0;
  cudaGetDevice(&device_id);
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = DLDevice{kDLCUDA, device_id};
  holder->dl.dl_tensor.ndim = 3;
  holder->dl.dl_tensor.dtype = complex_dtype<float>();
  holder->dl.dl_tensor.shape = holder->shape.data();
  holder->dl.dl_tensor.strides = holder->strides.data();
  holder->dl.dl_tensor.byte_offset = 0;
  holder->dl.manager_ctx = holder.get();
  holder->dl.deleter = [](DLManagedTensor* self) {
    delete static_cast<StridedAdcCubeHolder*>(self->manager_ctx);
  };
  auto tensor = std::make_shared<holoscan::Tensor>(&holder->dl);
  holder.release();
  return tensor;
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

TXSchedule explicit_chirp_index_schedule() {
  TXSchedule schedule;
  schedule.num_tx = kNumTx;
  schedule.chirp_tx_index = {0, 1, 0, 1, 0, 1};
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

TEST(MIMODeinterleaveOp, ExplicitChirpTxIndexUniformPattern) {
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = explicit_chirp_index_schedule();
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

TEST(MIMODeinterleaveOp, ComplexDoublePreserved) {
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = round_robin_schedule();
  const auto pattern = pattern_for(schedule);
  const auto host =
      build_interleaved_cube<std::complex<double>>(pattern, shape);
  const HostTensor out = run_deinterleave<double>(
      host, shape, complex_dtype<double>(), schedule);
  EXPECT_TRUE(
      holoscan::radar::dldatatype_equals(out.dtype, complex_dtype<double>()));
  expect_deinterleaved<std::complex<double>>(out, out.shape);
}

TEST(MIMODeinterleaveOp, RealDoublePreserved) {
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = round_robin_schedule();
  const auto pattern = pattern_for(schedule);
  const auto host = build_interleaved_cube<double>(pattern, shape);
  const HostTensor out =
      run_deinterleave<double>(host, shape, real_dtype<double>(), schedule);
  EXPECT_TRUE(
      holoscan::radar::dldatatype_equals(out.dtype, real_dtype<double>()));
  expect_deinterleaved<double>(out, out.shape);
}

TEST(MIMODeinterleaveOp, LeadingBatchAxesPreservedAndIndependent) {
  const std::vector<int64_t> shape = {2, kNumRx, kSlowTime, kSamples};
  const TXSchedule schedule = round_robin_schedule();
  const auto pattern = pattern_for(schedule);
  const auto host =
      build_interleaved_cube<std::complex<float>>(pattern, shape);
  const HostTensor out = run_deinterleave<float>(
      host, shape, complex_dtype<float>(), schedule);
  ASSERT_EQ(out.shape.size(), 5U);
  EXPECT_EQ(out.shape[0], 2);
  expect_deinterleaved<std::complex<float>>(out, out.shape);
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

TEST(MIMODeinterleaveOp, RejectsPageableHostInput) {
  const TXSchedule schedule = round_robin_schedule();
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const auto host = std::vector<std::complex<float>>(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  holoscan::TensorMap map;
  map.insert({"adc_cube",
              to_pageable_host_tensor(host, shape, complex_dtype<float>())});
  EXPECT_THROW((void)run_harness_single<float>(schedule, map),
               std::invalid_argument);
}

TEST(MIMODeinterleaveOp, RejectsMultiEntryTensorMap) {
  const TXSchedule schedule = round_robin_schedule();
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const auto host = std::vector<std::complex<float>>(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  holoscan::TensorMap map;
  map.insert({"a", to_device_tensor(host, shape, complex_dtype<float>())});
  map.insert({"b", to_device_tensor(host, shape, complex_dtype<float>())});
  EXPECT_THROW((void)run_harness_single<float>(schedule, map),
               std::invalid_argument);
}

TEST(MIMODeinterleaveOp, RejectsNonContiguousInput) {
  const TXSchedule schedule = round_robin_schedule();
  const std::vector<int64_t> shape = {kNumRx, kSlowTime, kSamples};
  const auto pattern = pattern_for(schedule);
  const auto host =
      build_interleaved_cube<std::complex<float>>(pattern, shape);
  holoscan::TensorMap map;
  map.insert({"adc_cube", make_non_contiguous_adc_cube(host, shape)});
  EXPECT_THROW((void)run_harness_single<float>(schedule, map),
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
