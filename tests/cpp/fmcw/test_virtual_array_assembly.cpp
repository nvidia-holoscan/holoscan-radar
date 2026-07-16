// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Contract tests for VirtualArrayAssemblyOp: virtual layout, overlap policies,
// array_geometry emission, rank-agnostic batching, [live] wavelength overrides,
// cross-cutting input validation, and float/double agreement.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

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

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ArrayGeometry;
using holoscan::radar::OverlapPolicy;
using holoscan::radar::VirtualArrayAssemblyOp;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;
using holoscan::radar::testing::to_pageable_host_tensor;
using holoscan::radar::testing::wire_capture_sink;
using holoscan::radar::testing::wire_data_source;
using holoscan::radar::testing::wire_leading_override;
using holoscan::test::create_operator_test;

constexpr int64_t kNumTx = 2;
constexpr int64_t kNumRx = 2;
constexpr int64_t kDoppler = 2;
constexpr int64_t kRange = 2;
constexpr double kWavelength = 0.03;

int64_t tensor_count(const std::vector<int64_t>& shape) {
  int64_t count = 1;
  for (int64_t d : shape) {
    count *= d;
  }
  return count;
}

double channel_marker(int64_t batch, int64_t tx, int64_t rx, int64_t doppler,
                      int64_t range) {
  return 10000.0 * static_cast<double>(batch) +
         100.0 * static_cast<double>(tx) +
         10.0 * static_cast<double>(rx) +
         static_cast<double>(doppler) + 0.1 * static_cast<double>(range);
}

std::vector<double> separated_tx_positions() {
  return {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
}

std::vector<double> overlapping_tx_positions() {
  return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

std::vector<double> default_rx_positions() {
  return {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
}

std::vector<double> analytic_virtual_positions(
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, int64_t num_tx, int64_t num_rx) {
  std::vector<double> out(static_cast<size_t>(num_tx * num_rx * 3));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      const size_t v = static_cast<size_t>((tx * num_rx + rx) * 3);
      for (int i = 0; i < 3; ++i) {
        out[v + static_cast<size_t>(i)] =
            tx_positions[static_cast<size_t>(tx * 3 + i)] +
            rx_positions[static_cast<size_t>(rx * 3 + i)];
      }
    }
  }
  return out;
}

template <typename T>
std::vector<std::complex<T>> build_rd_host(const std::vector<int64_t>& shape) {
  if (shape.size() < 4) {
    throw std::invalid_argument("build_rd_host: expected rank >= 4");
  }
  const int64_t num_tx = shape[shape.size() - 4];
  const int64_t num_rx = shape[shape.size() - 3];
  const int64_t doppler = shape[shape.size() - 2];
  const int64_t range = shape.back();
  int64_t batch = 1;
  for (size_t i = 0; i + 4 < shape.size(); ++i) {
    batch *= shape[i];
  }
  const int64_t count = tensor_count(shape);
  std::vector<std::complex<T>> host(static_cast<size_t>(count));
  const int64_t slice = num_tx * num_rx * doppler * range;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        for (int64_t d = 0; d < doppler; ++d) {
          for (int64_t r = 0; r < range; ++r) {
            const size_t idx = static_cast<size_t>(
                b * slice + (((tx * num_rx + rx) * doppler) + d) * range + r);
            const T m =
                static_cast<T>(channel_marker(b, tx, rx, d, r));
            host[idx] = {m, static_cast<T>(0)};
          }
        }
      }
    }
  }
  return host;
}

struct AssemblyResult {
  HostTensor rd_maps;
  ArrayGeometry geometry;
};

template <typename T>
AssemblyResult run_harness(
    const std::vector<holoscan::TensorMap>& frames,
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, double wavelength,
    OverlapPolicy policy,
    const std::function<void(holoscan::Fragment*,
                             const std::shared_ptr<holoscan::Operator>&)>&
        wire_extra = nullptr) {
  TensorMapCapture rd_capture;
  std::vector<ArrayGeometry> geometries;
  class AssemblyApp : public holoscan::Application {
   public:
    std::vector<holoscan::TensorMap> frames;
    std::vector<double> tx_positions;
    std::vector<double> rx_positions;
    double wavelength{0.0};
    OverlapPolicy policy{OverlapPolicy::kDedup};
    TensorMapCapture* rd_capture{nullptr};
    std::vector<ArrayGeometry>* geometries{nullptr};
    std::function<void(holoscan::Fragment*,
                       const std::shared_ptr<holoscan::Operator>&)>
        wire_extra;

    void compose() override {
      using holoscan::Arg;
      auto op = make_operator<VirtualArrayAssemblyOp<T>>(
          "virtual_array_assembly", Arg("tx_positions") = tx_positions,
          Arg("rx_positions") = rx_positions, Arg("wavelength") = wavelength,
          Arg("overlap_policy") = policy);
      wire_data_source(this, op, "rd_maps", frames);
      if (wire_extra) {
        wire_extra(this, op);
      }
      wire_capture_sink(this, op, "rd_maps", rd_capture);
      auto geom_sink =
          make_operator<holoscan::test::TestHarnessSinkOp<ArrayGeometry>>(
              "array_geometry_sink");
      geom_sink->set_validators({[this](const ArrayGeometry& geom) {
        geometries->push_back(geom);
      }});
      add_flow(op, geom_sink, {{"array_geometry", "input"}});
    }
  };

  AssemblyApp app;
  app.frames = frames;
  app.tx_positions = tx_positions;
  app.rx_positions = rx_positions;
  app.wavelength = wavelength;
  app.policy = policy;
  app.rd_capture = &rd_capture;
  app.geometries = &geometries;
  app.wire_extra = wire_extra;
  app.run();

  if (rd_capture.frames.size() != frames.size()) {
    throw std::runtime_error("expected one rd_maps frame per input tick");
  }
  if (geometries.size() != frames.size()) {
    throw std::runtime_error("expected one array_geometry per input tick");
  }
  return {std::move(rd_capture.frames.back()),
          std::move(geometries.back())};
}

template <typename T>
AssemblyResult run_assembly(
    const std::vector<std::complex<T>>& host,
    const std::vector<int64_t>& shape, const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, double wavelength,
    OverlapPolicy policy) {
  auto rd_maps =
      single_map(to_device_tensor(host, shape, complex_dtype<T>()), "rd_maps");
  return run_harness<T>({rd_maps}, tx_positions, rx_positions, wavelength,
                        policy);
}

template <typename T>
AssemblyResult run_assembly(
    const std::vector<std::complex<T>>& host,
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, OverlapPolicy policy) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  return run_assembly<T>(host, shape, tx_positions, rx_positions, kWavelength,
                         policy);
}

void expect_geometry_positions(const ArrayGeometry& geom,
                               const std::vector<double>& expected,
                               double wavelength = kWavelength) {
  ASSERT_EQ(geom.element_positions.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(geom.element_positions[i], expected[i], 1e-9);
  }
  EXPECT_NEAR(geom.wavelength, wavelength, 1e-12);
}

void expect_keep_all_values(const HostTensor& out,
                            const std::vector<int64_t>& shape) {
  ASSERT_GE(shape.size(), 4U);
  const int64_t num_tx = shape[shape.size() - 4];
  const int64_t num_rx = shape[shape.size() - 3];
  const int64_t doppler = shape[shape.size() - 2];
  const int64_t range = shape.back();
  int64_t batch = 1;
  for (size_t i = 0; i + 4 < shape.size(); ++i) {
    batch *= shape[i];
  }
  const auto* data = out.as<std::complex<float>>();
  const int64_t slice = num_tx * num_rx * doppler * range;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t tx = 0; tx < num_tx; ++tx) {
      for (int64_t rx = 0; rx < num_rx; ++rx) {
        const int64_t v = tx * num_rx + rx;
        for (int64_t d = 0; d < doppler; ++d) {
          for (int64_t r = 0; r < range; ++r) {
            const size_t in_idx = static_cast<size_t>(
                b * slice + (((tx * num_rx + rx) * doppler) + d) * range + r);
            const size_t out_idx = static_cast<size_t>(
                b * (num_tx * num_rx * doppler * range) +
                ((v * doppler + d) * range + r));
            EXPECT_NEAR(data[out_idx].real(), data[in_idx].real(), 1e-5)
                << "batch=" << b << " tx=" << tx << " rx=" << rx;
          }
        }
      }
    }
  }
}

struct StridedRdMapsHolder {
  void* data{nullptr};
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  DLManagedTensor dl{};

  ~StridedRdMapsHolder() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

std::shared_ptr<holoscan::Tensor> make_non_contiguous_rd_maps(
    const std::vector<std::complex<float>>& host,
    const std::vector<int64_t>& shape) {
  const int64_t num_tx = shape[shape.size() - 4];
  const int64_t num_rx = shape[shape.size() - 3];
  const int64_t doppler = shape[shape.size() - 2];
  const int64_t range = shape.back();
  const int64_t range_stride = range + 1;
  const size_t alloc_elems = static_cast<size_t>(num_tx * num_rx * doppler *
                                                 range_stride);
  auto holder = std::make_unique<StridedRdMapsHolder>();
  holder->shape = shape;
  holder->strides = {num_rx * doppler * range_stride, doppler * range_stride,
                     range_stride, 1};
  if (cudaMalloc(&holder->data, alloc_elems * sizeof(std::complex<float>)) !=
      cudaSuccess) {
    throw std::runtime_error("make_non_contiguous_rd_maps: cudaMalloc failed");
  }
  std::vector<std::complex<float>> padded(alloc_elems, {0.0f, 0.0f});
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      for (int64_t d = 0; d < doppler; ++d) {
        for (int64_t r = 0; r < range; ++r) {
          const size_t dst = static_cast<size_t>(
              tx * num_rx * doppler * range_stride +
              rx * doppler * range_stride + d * range_stride + r);
          const size_t src = static_cast<size_t>(
              (((tx * num_rx + rx) * doppler) + d) * range + r);
          padded[dst] = host[src];
        }
      }
    }
  }
  if (cudaMemcpy(holder->data, padded.data(),
                 alloc_elems * sizeof(std::complex<float>),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("make_non_contiguous_rd_maps: H2D failed");
  }
  int device_id = 0;
  cudaGetDevice(&device_id);
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = DLDevice{kDLCUDA, device_id};
  holder->dl.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  holder->dl.dl_tensor.dtype = complex_dtype<float>();
  holder->dl.dl_tensor.shape = holder->shape.data();
  holder->dl.dl_tensor.strides = holder->strides.data();
  holder->dl.dl_tensor.byte_offset = 0;
  holder->dl.manager_ctx = holder.get();
  holder->dl.deleter = [](DLManagedTensor* self) {
    delete static_cast<StridedRdMapsHolder*>(self->manager_ctx);
  };
  auto tensor = std::make_shared<holoscan::Tensor>(&holder->dl);
  holder.release();
  return tensor;
}

TEST(VirtualArrayAssemblyOp, EmittedGeometryMatchesAnalyticLayout) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto result = run_assembly<float>(
      build_rd_host<float>(shape), shape, separated_tx_positions(),
      default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  const auto expected = analytic_virtual_positions(
      separated_tx_positions(), default_rx_positions(), kNumTx, kNumRx);
  expect_geometry_positions(result.geometry, expected);
}

TEST(VirtualArrayAssemblyOp, KeepAllPreservesChannels) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto result = run_assembly<float>(
      build_rd_host<float>(shape), shape, separated_tx_positions(),
      default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  ASSERT_EQ(result.rd_maps.shape.size(), 3U);
  EXPECT_EQ(result.rd_maps.shape[0], kNumTx * kNumRx);
  EXPECT_EQ(result.geometry.num_elements(), kNumTx * kNumRx);
  expect_keep_all_values(result.rd_maps, shape);
}

TEST(VirtualArrayAssemblyOp, DedupCollapsesCoincidentPositions) {
  const auto result = run_assembly<float>(
      build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
      overlapping_tx_positions(), default_rx_positions(),
      OverlapPolicy::kDedup);
  EXPECT_EQ(result.rd_maps.shape[0], 2);
  EXPECT_EQ(result.geometry.num_elements(), 2);
  expect_geometry_positions(result.geometry, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
}

TEST(VirtualArrayAssemblyOp, DedupKeepsFirstRepresentativeChannel) {
  const auto result = run_assembly<float>(
      build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
      overlapping_tx_positions(), default_rx_positions(),
      OverlapPolicy::kDedup);
  const auto* out = result.rd_maps.as<std::complex<float>>();
  for (int64_t d = 0; d < kDoppler; ++d) {
    for (int64_t r = 0; r < kRange; ++r) {
      const size_t origin_idx = static_cast<size_t>(d * kRange + r);
      const size_t rx1_idx =
          static_cast<size_t>(1 * kDoppler * kRange + d * kRange + r);
      EXPECT_NEAR(out[origin_idx].real(), channel_marker(0, 0, 0, d, r), 1e-5)
          << "dedup must keep the first colliding (tx, rx) channel";
      EXPECT_NEAR(out[rx1_idx].real(), channel_marker(0, 0, 1, d, r), 1e-5)
          << "second virtual slot is rx=1 at x=1";
    }
  }
}

TEST(VirtualArrayAssemblyOp, DedupIsDefaultPolicy) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  const auto explicit_dedup = run_assembly<float>(
      host, overlapping_tx_positions(), default_rx_positions(),
      OverlapPolicy::kDedup);
  TensorMapCapture capture;
  ArrayGeometry geometry;
  auto harness = create_operator_test<VirtualArrayAssemblyOp<float>>(
      holoscan::Arg("tx_positions") = overlapping_tx_positions(),
      holoscan::Arg("rx_positions") = default_rx_positions(),
      holoscan::Arg("wavelength") = kWavelength);
  harness->template add_input_port<holoscan::TensorMap>(
      "rd_maps",
      {single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                  "rd_maps")});
  harness->template add_output_port<holoscan::TensorMap>(
      "rd_maps", capture.validators());
  harness->template add_output_port<ArrayGeometry>(
      "array_geometry", {[&geometry](const ArrayGeometry& geom) {
        geometry = geom;
      }});
  harness->run_test();
  ASSERT_EQ(capture.frames.size(), 1U);
  EXPECT_EQ(capture.frames.front().shape, explicit_dedup.rd_maps.shape);
  EXPECT_EQ(geometry.num_elements(), explicit_dedup.geometry.num_elements());
}

TEST(VirtualArrayAssemblyOp, AverageCombinesCoincidentChannels) {
  const auto result = run_assembly<float>(
      build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
      overlapping_tx_positions(), default_rx_positions(),
      OverlapPolicy::kAverage);
  ASSERT_EQ(result.rd_maps.shape[0], 2);
  const auto* out = result.rd_maps.as<std::complex<float>>();
  for (int64_t d = 0; d < kDoppler; ++d) {
    for (int64_t r = 0; r < kRange; ++r) {
      const size_t idx = static_cast<size_t>(d * kRange + r);
      const double expected =
          0.5 * (channel_marker(0, 0, 0, d, r) + channel_marker(0, 1, 0, d, r));
      EXPECT_NEAR(out[idx].real(), expected, 1e-5);
    }
  }
}

TEST(VirtualArrayAssemblyOp, AverageCombinesAllCollidingChannels) {
  const std::vector<double> origin_rx = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto result = run_assembly<float>(
      build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
      overlapping_tx_positions(), origin_rx, OverlapPolicy::kAverage);
  EXPECT_EQ(result.rd_maps.shape[0], 1);
  EXPECT_EQ(result.geometry.num_elements(), 1);
  const auto* out = result.rd_maps.as<std::complex<float>>();
  for (int64_t d = 0; d < kDoppler; ++d) {
    for (int64_t r = 0; r < kRange; ++r) {
      const size_t idx = static_cast<size_t>(d * kRange + r);
      double expected = 0.0;
      for (int64_t tx = 0; tx < kNumTx; ++tx) {
        for (int64_t rx = 0; rx < kNumRx; ++rx) {
          expected += channel_marker(0, tx, rx, d, r);
        }
      }
      expected /= static_cast<double>(kNumTx * kNumRx);
      EXPECT_NEAR(out[idx].real(), expected, 1e-5);
    }
  }
}

TEST(VirtualArrayAssemblyOp, ComplexDoublePreservesValues) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto result = run_assembly<double>(
      build_rd_host<double>(shape), shape, separated_tx_positions(),
      default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  EXPECT_TRUE(
      holoscan::radar::dldatatype_equals(result.rd_maps.dtype,
                                         complex_dtype<double>()));
  const auto* out = result.rd_maps.as<std::complex<double>>();
  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t rx = 0; rx < kNumRx; ++rx) {
      const int64_t v = tx * kNumRx + rx;
      for (int64_t d = 0; d < kDoppler; ++d) {
        for (int64_t r = 0; r < kRange; ++r) {
          const size_t idx =
              static_cast<size_t>((v * kDoppler + d) * kRange + r);
          EXPECT_NEAR(out[idx].real(), channel_marker(0, tx, rx, d, r), 1e-9);
        }
      }
    }
  }
}

TEST(VirtualArrayAssemblyOp, LeadingBatchAxesPreservedAndIndependent) {
  const std::vector<int64_t> shape = {2, kNumTx, kNumRx, kDoppler, kRange};
  const auto result = run_assembly<float>(
      build_rd_host<float>(shape), shape, separated_tx_positions(),
      default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  ASSERT_EQ(result.rd_maps.shape.size(), 4U);
  EXPECT_EQ(result.rd_maps.shape[0], 2);
  EXPECT_EQ(result.rd_maps.shape[1], kNumTx * kNumRx);
  expect_keep_all_values(result.rd_maps, shape);
}

TEST(VirtualArrayAssemblyOp, LiveWavelengthOverrideSticky) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  auto frame = [&]() {
    return single_map(to_device_tensor(host, shape, complex_dtype<float>()),
                      "rd_maps");
  };
  std::vector<ArrayGeometry> geometries;
  TensorMapCapture rd_capture;
  class WavelengthApp : public holoscan::Application {
   public:
    std::vector<holoscan::TensorMap> frames;
    std::vector<double> wavelength_overrides;
    TensorMapCapture* rd_capture{nullptr};
    std::vector<ArrayGeometry>* geometries{nullptr};

    void compose() override {
      using holoscan::Arg;
      auto op = make_operator<VirtualArrayAssemblyOp<float>>(
          "virtual_array_assembly",
          Arg("tx_positions") = separated_tx_positions(),
          Arg("rx_positions") = default_rx_positions(),
          Arg("wavelength") = kWavelength,
          Arg("overlap_policy") = OverlapPolicy::kKeepAll);
      wire_data_source(this, op, "rd_maps", frames);
      wire_leading_override<double>(this, op, "wavelength",
                                    wavelength_overrides);
      wire_capture_sink(this, op, "rd_maps", rd_capture);
      auto geom_sink =
          make_operator<holoscan::test::TestHarnessSinkOp<ArrayGeometry>>(
              "array_geometry_sink");
      geom_sink->set_validators({[this](const ArrayGeometry& geom) {
        geometries->push_back(geom);
      }});
      add_flow(op, geom_sink, {{"array_geometry", "input"}});
    }
  };

  WavelengthApp app;
  app.frames = {frame(), frame(), frame()};
  app.wavelength_overrides = {0.04, 0.05};
  app.rd_capture = &rd_capture;
  app.geometries = &geometries;
  app.run();

  ASSERT_EQ(geometries.size(), 3U);
  EXPECT_NEAR(geometries[0].wavelength, 0.04, 1e-12);
  EXPECT_NEAR(geometries[1].wavelength, 0.05, 1e-12);
  EXPECT_NEAR(geometries[2].wavelength, 0.05, 1e-12);
}

TEST(VirtualArrayAssemblyOp, FloatDoubleAgreement) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto f = run_assembly<float>(build_rd_host<float>(shape), shape,
                                     separated_tx_positions(),
                                     default_rx_positions(), kWavelength,
                                     OverlapPolicy::kKeepAll);
  const auto d = run_assembly<double>(build_rd_host<double>(shape), shape,
                                      separated_tx_positions(),
                                      default_rx_positions(), kWavelength,
                                      OverlapPolicy::kKeepAll);
  ASSERT_EQ(f.rd_maps.shape, d.rd_maps.shape);
  const int64_t count = f.rd_maps.size();
  ASSERT_EQ(count, d.rd_maps.size());
  const auto* f_data = f.rd_maps.as<std::complex<float>>();
  const auto* d_data = d.rd_maps.as<std::complex<double>>();
  for (int64_t i = 0; i < count; ++i) {
    EXPECT_NEAR(static_cast<double>(f_data[static_cast<size_t>(i)].real()),
                d_data[static_cast<size_t>(i)].real(), 1e-5);
    EXPECT_NEAR(static_cast<double>(f_data[static_cast<size_t>(i)].imag()),
                d_data[static_cast<size_t>(i)].imag(), 1e-5);
  }
}

TEST(VirtualArrayAssemblyOp, RejectsMismatchedTxPositions) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  EXPECT_THROW(
      (void)run_assembly<float>(host, shape, {0.0, 0.0, 0.0},
                                default_rx_positions(), kWavelength,
                                OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

TEST(VirtualArrayAssemblyOp, RejectsMismatchedRxPositions) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  EXPECT_THROW(
      (void)run_assembly<float>(host, shape, separated_tx_positions(),
                                {0.0, 0.0, 0.0}, kWavelength,
                                OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

TEST(VirtualArrayAssemblyOp, RejectsInvalidWavelength) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  EXPECT_THROW(
      (void)run_assembly<float>(host, shape, separated_tx_positions(),
                                default_rx_positions(), 0.0,
                                OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

TEST(VirtualArrayAssemblyOp, RejectsRealInput) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const std::vector<float> host(static_cast<size_t>(tensor_count(shape)), 1.0f);
  holoscan::TensorMap map;
  map.insert({"rd_maps",
              to_device_tensor(host, shape, real_dtype<float>())});
  EXPECT_THROW(
      (void)run_harness<float>({map}, separated_tx_positions(),
                               default_rx_positions(), kWavelength,
                               OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

TEST(VirtualArrayAssemblyOp, RejectsRankTooSmall) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  EXPECT_THROW(
      (void)run_assembly<float>(host, shape, separated_tx_positions(),
                                default_rx_positions(), kWavelength,
                                OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

TEST(VirtualArrayAssemblyOp, RejectsPageableHostInput) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  holoscan::TensorMap map;
  map.insert({"rd_maps",
              to_pageable_host_tensor(host, shape, complex_dtype<float>())});
  EXPECT_THROW(
      (void)run_harness<float>({map}, separated_tx_positions(),
                               default_rx_positions(), kWavelength,
                               OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

TEST(VirtualArrayAssemblyOp, RejectsMultiEntryTensorMap) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  holoscan::TensorMap map;
  map.insert({"a", to_device_tensor(host, shape, complex_dtype<float>())});
  map.insert({"b", to_device_tensor(host, shape, complex_dtype<float>())});
  EXPECT_THROW(
      (void)run_harness<float>({map}, separated_tx_positions(),
                               default_rx_positions(), kWavelength,
                               OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

TEST(VirtualArrayAssemblyOp, RejectsNonContiguousInput) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto host = build_rd_host<float>(shape);
  holoscan::TensorMap map;
  map.insert({"rd_maps", make_non_contiguous_rd_maps(host, shape)});
  EXPECT_THROW(
      (void)run_harness<float>({map}, separated_tx_positions(),
                               default_rx_positions(), kWavelength,
                               OverlapPolicy::kKeepAll),
      std::invalid_argument);
}

}  // namespace
