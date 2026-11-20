// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Contract tests for VirtualArrayAssemblyOp (HSDK 5 Graph harness):
// virtual layout, overlap policies, WireArrayGeometry emission,
// opaque leading-batch rejection, [live] wavelength overrides, cross-cutting
// input validation, and float/double agreement.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "angle_estimation/beamscan_estimator_op/beamscan_estimator_op.hpp"
#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/angle_convention.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "radar_schemas/array_geometry_bridge.hpp"
#include "utils/integration_graph_utils.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ArrayGeometry;
using holoscan::radar::BeamscanEstimatorOp;
using holoscan::radar::BeamscanParams;
using holoscan::radar::OverlapPolicy;
using holoscan::radar::VirtualArrayAssemblyOp;
using holoscan::radar::VirtualArrayAssemblyParams;
using holoscan::radar::WireArrayGeometry;
using holoscan::radar::complex_dtype;
using holoscan::radar::from_wire;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;

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

double channel_marker(int64_t batch, int64_t tx, int64_t rx, int64_t doppler, int64_t range) {
  return 10000.0 * static_cast<double>(batch) + 100.0 * static_cast<double>(tx) +
         10.0 * static_cast<double>(rx) + static_cast<double>(doppler) +
         0.1 * static_cast<double>(range);
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

std::vector<double> analytic_virtual_positions(const std::vector<double>& tx_positions,
                                               const std::vector<double>& rx_positions,
                                               int64_t num_tx, int64_t num_rx) {
  std::vector<double> out(static_cast<size_t>(num_tx * num_rx * 3));
  for (int64_t tx = 0; tx < num_tx; ++tx) {
    for (int64_t rx = 0; rx < num_rx; ++rx) {
      const size_t v = static_cast<size_t>((tx * num_rx + rx) * 3);
      for (int i = 0; i < 3; ++i) {
        out[v + static_cast<size_t>(i)] = tx_positions[static_cast<size_t>(tx * 3 + i)] +
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
            const size_t idx =
                static_cast<size_t>(b * slice + (((tx * num_rx + rx) * doppler) + d) * range + r);
            const T m = static_cast<T>(channel_marker(b, tx, rx, d, r));
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
holoscan::testing::TestResult run_assembly_result_t(std::shared_ptr<holoscan::Tensor> input,
                                                    VirtualArrayAssemblyParams params) {
  using Op = VirtualArrayAssemblyOp<T>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  fixture.push(&Op::rd_maps, *input);
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.template capture_snapshot<holoscan::Tensor>("rd_maps_out", 1U, kMaximumBytes);
  fixture.template capture_snapshot<WireArrayGeometry>("array_geometry", 1U, kMaximumBytes);
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "rd_maps_out", 1U);
  (void)completion.wait_for_output(running, "array_geometry", 1U);
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

holoscan::testing::TestResult run_assembly_result(std::shared_ptr<holoscan::Tensor> input,
                                                  VirtualArrayAssemblyParams params) {
  return run_assembly_result_t<float>(std::move(input), std::move(params));
}

std::vector<HostTensor> assembly_maps(const holoscan::testing::TestResult& result) {
  std::vector<HostTensor> frames;
  for (const auto& snapshot : result.template snapshots<holoscan::Tensor>("rd_maps_out")) {
    frames.push_back(holoscan::radar::testing::to_host(snapshot));
  }
  return frames;
}

std::vector<WireArrayGeometry> assembly_geometries(const holoscan::testing::TestResult& result) {
  std::vector<WireArrayGeometry> geometries;
  for (const auto& snapshot : result.template snapshots<WireArrayGeometry>("array_geometry")) {
    auto wire = snapshot.metadata();
    const auto& packed =
        snapshot.tensor(holoscan::radar::wire::ArrayGeometryTensorFields::packed_geometry);
    std::vector<double> values(packed.bytes().size() / sizeof(double));
    std::memcpy(values.data(), packed.bytes().data(), packed.bytes().size());
    wire.packed_geometry =
        holoscan::radar::testing::to_pageable_host_tensor(values, packed.shape(), packed.dtype());
    geometries.push_back(std::move(wire));
  }
  return geometries;
}

template <typename T>
AssemblyResult run_assembly(const std::vector<std::complex<T>>& host,
                            const std::vector<int64_t>& shape,
                            const std::vector<double>& tx_positions,
                            const std::vector<double>& rx_positions, double wavelength,
                            OverlapPolicy policy) {
  VirtualArrayAssemblyParams params;
  params.tx_positions = tx_positions;
  params.rx_positions = rx_positions;
  params.wavelength = wavelength;
  params.overlap_policy = policy;
  const auto result = run_assembly_result_t<T>(to_device_tensor(host, shape, complex_dtype<T>()),
                                               std::move(params));
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  auto decoded = from_wire(assembly_geometries(result).front());
  if (!decoded) {
    throw std::runtime_error("VirtualArrayAssemblyOp test: invalid WireArrayGeometry");
  }
  return {
      holoscan::radar::testing::to_host(result.template snapshot<holoscan::Tensor>("rd_maps_out")),
      std::move(*decoded)};
}
template <typename T>
AssemblyResult run_assembly(const std::vector<std::complex<T>>& host,
                            const std::vector<double>& tx_positions,
                            const std::vector<double>& rx_positions, OverlapPolicy policy) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  return run_assembly<T>(host, shape, tx_positions, rx_positions, kWavelength, policy);
}

void expect_geometry_positions(const ArrayGeometry& geom, const std::vector<double>& expected,
                               double wavelength = kWavelength) {
  ASSERT_EQ(geom.element_positions.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(geom.element_positions[i], expected[i], 1e-9);
  }
  EXPECT_NEAR(geom.wavelength, wavelength, 1e-12);
}

void expect_keep_all_values(const HostTensor& out, const std::vector<int64_t>& shape) {
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
            const size_t out_idx = static_cast<size_t>(b * (num_tx * num_rx * doppler * range) +
                                                       ((v * doppler + d) * range + r));
            const double expected_val = channel_marker(b, tx, rx, d, r);
            const double tol = std::max(1e-4, 1e-6 * std::abs(expected_val));
            EXPECT_NEAR(data[out_idx].real(), expected_val, tol)
                << "batch=" << b << " tx=" << tx << " rx=" << rx;
          }
        }
      }
    }
  }
}

TEST(VirtualArrayAssemblyOp, EmittedGeometryMatchesAnalyticLayout) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto result =
      run_assembly<float>(build_rd_host<float>(shape), shape, separated_tx_positions(),
                          default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  const auto expected =
      analytic_virtual_positions(separated_tx_positions(), default_rx_positions(), kNumTx, kNumRx);
  expect_geometry_positions(result.geometry, expected);
}

TEST(VirtualArrayAssemblyOp, GeometryDirectlyDrivesBeamscan) {
  namespace detail = holoscan::radar::testing::integration;

  // Construct before Graph/RunSession so native input handles are released
  // after their teardown, including ASSERT exits, rather than at static exit.
  struct BankCleanup {
    ~BankCleanup() {
      detail::FrameBank::reset();
      detail::FrameBankB::reset();
    }
  };
  const BankCleanup cleanup;
  detail::FrameBank::reset();
  detail::FrameBankB::reset();
  const std::vector<int64_t> rd_shape{2, 2, 3, 1};
  detail::FrameBank::input_frames = {to_device_tensor(
      std::vector<std::complex<float>>(12U, {1.0F, 0.0F}), rd_shape, complex_dtype<float>())};

  holoscan::Graph graph{"vaa-geometry-to-beamscan"};
  const auto source = graph.op<detail::TensorSourceOp>("source");
  auto extents = holoscan::radar::testing::default_test_extents();
  auto vaa = graph.op<VirtualArrayAssemblyOp<float>>("vaa", extents);
  VirtualArrayAssemblyParams vaa_params;
  vaa_params.tx_positions = {0.0, 0.0, 0.0, 0.0, 0.5, 0.0};
  vaa_params.rx_positions = {0.0, 0.0, 0.0, 0.0, 0.5, 0.0};
  vaa_params.wavelength = 1.0;
  vaa.set_params(std::move(vaa_params));

  auto beamscan = graph.op<BeamscanEstimatorOp<float>>("beamscan", extents);
  BeamscanParams beamscan_params;
  // Deliberately incompatible with the 3-channel snapshot. Successful
  // steering proves the Exact-matched VAA geometry replaced this SafeDefault.
  beamscan_params.array_geometry =
      ArrayGeometry{.element_positions = {0.0, 0.0, 0.0, 0.0, 0.5, 0.0}, .wavelength = 1.0};
  beamscan_params.az_coordinates = {-0.2, 0.0, 0.2};
  beamscan_params.el_coordinates = {0.0};
  beamscan.set_params(std::move(beamscan_params));

  const auto aoa_sink = graph.op<detail::TensorCaptureOp>("aoa_sink");
  const auto spectrum_sink = graph.op<detail::TensorCaptureBOp>("spectrum_sink");
  const holoscan::ConnectionOptions queue{.queue_depth = 4U};
  graph.add_flow(source->out, vaa->rd_maps, queue);
  graph.add_flow(vaa->rd_maps_out, beamscan->snapshot, queue);
  graph.add_flow(vaa->array_geometry, beamscan->array_geometry, queue);
  graph.add_flow(beamscan->aoa_estimates, aoa_sink->in, queue);
  graph.add_flow(beamscan->angle_spectrum, spectrum_sink->in, queue);
  graph.partition("integration")
      .add(source)
      .add(vaa)
      .add(beamscan)
      .add(aoa_sink)
      .add(spectrum_sink);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  auto bind = [&options](const char* op, const char* port) {
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = holoscan::DeviceId{0}});
  };
  bind("source", "out");
  bind("vaa", "rd_maps_out");
  bind("beamscan", "aoa_estimates");
  bind("beamscan", "angle_spectrum");

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + detail::kHarnessDeadline;
  while (session.diagnostics().empty() && std::chrono::steady_clock::now() < deadline) {
    bool captured = false;
    {
      const std::lock_guard lock(detail::FrameBank::mutex);
      captured = !detail::FrameBank::captured.empty();
    }
    if (captured) {
      break;
    }
    std::this_thread::sleep_for(detail::kHarnessPoll);
  }
  session.request_stop();
  session.wait();
  ASSERT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(detail::FrameBank::mutex);
  ASSERT_EQ(detail::FrameBank::captured.size(), 1U);
  const auto& aoa = detail::FrameBank::captured.front();
  ASSERT_EQ(aoa.shape, (std::vector<int64_t>{3, 2}));
  for (int64_t item = 0; item < 3; ++item) {
    EXPECT_NEAR(aoa.as<float>()[item * 2], 0.0F, 1e-6F);
  }
}

TEST(VirtualArrayAssemblyOp, KeepAllPreservesChannels) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto result =
      run_assembly<float>(build_rd_host<float>(shape), shape, separated_tx_positions(),
                          default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  ASSERT_EQ(result.rd_maps.shape.size(), 3U);
  EXPECT_EQ(result.rd_maps.shape[0], kNumTx * kNumRx);
  EXPECT_EQ(result.geometry.num_elements(), kNumTx * kNumRx);
  expect_keep_all_values(result.rd_maps, shape);
}

TEST(VirtualArrayAssemblyOp, DedupCollapsesCoincidentPositions) {
  const auto result = run_assembly<float>(build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
                                          overlapping_tx_positions(), default_rx_positions(),
                                          OverlapPolicy::kDedup);
  EXPECT_EQ(result.rd_maps.shape[0], 2);
  EXPECT_EQ(result.geometry.num_elements(), 2);
  expect_geometry_positions(result.geometry, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
}

TEST(VirtualArrayAssemblyOp, DedupKeepsFirstRepresentativeChannel) {
  const auto result = run_assembly<float>(build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
                                          overlapping_tx_positions(), default_rx_positions(),
                                          OverlapPolicy::kDedup);
  const auto* out = result.rd_maps.as<std::complex<float>>();
  for (int64_t d = 0; d < kDoppler; ++d) {
    for (int64_t r = 0; r < kRange; ++r) {
      const size_t origin_idx = static_cast<size_t>(d * kRange + r);
      const size_t rx1_idx = static_cast<size_t>(1 * kDoppler * kRange + d * kRange + r);
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
  const auto explicit_dedup = run_assembly<float>(host, overlapping_tx_positions(),
                                                  default_rx_positions(), OverlapPolicy::kDedup);

  VirtualArrayAssemblyParams params;
  params.tx_positions = overlapping_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = kWavelength;
  // overlap_policy defaults to kDedup in VirtualArrayAssemblyParams.
  const auto result = run_assembly_result_t<float>(
      to_device_tensor(host, shape, complex_dtype<float>()), std::move(params));
  ASSERT_TRUE(result.ok());
  ASSERT_FALSE(assembly_maps(result).empty());
  ASSERT_FALSE(assembly_geometries(result).empty());
  EXPECT_EQ(assembly_maps(result).front().shape, explicit_dedup.rd_maps.shape);
  auto geometry = from_wire(assembly_geometries(result).front());
  ASSERT_TRUE(geometry);
  EXPECT_EQ(geometry->num_elements(), explicit_dedup.geometry.num_elements());
}

TEST(VirtualArrayAssemblyOp, AverageCombinesCoincidentChannels) {
  const auto result = run_assembly<float>(build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
                                          overlapping_tx_positions(), default_rx_positions(),
                                          OverlapPolicy::kAverage);
  ASSERT_EQ(result.rd_maps.shape[0], 2);
  const auto* out = result.rd_maps.as<std::complex<float>>();
  for (int64_t d = 0; d < kDoppler; ++d) {
    for (int64_t r = 0; r < kRange; ++r) {
      const size_t idx = static_cast<size_t>(d * kRange + r);
      const double expected = 0.5 * (channel_marker(0, 0, 0, d, r) + channel_marker(0, 1, 0, d, r));
      EXPECT_NEAR(out[idx].real(), expected, 1e-5);
    }
  }
}

TEST(VirtualArrayAssemblyOp, AverageCombinesAllCollidingChannels) {
  const std::vector<double> origin_rx = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto result =
      run_assembly<float>(build_rd_host<float>({kNumTx, kNumRx, kDoppler, kRange}),
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
  const auto result =
      run_assembly<double>(build_rd_host<double>(shape), shape, separated_tx_positions(),
                           default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  EXPECT_TRUE(holoscan::radar::dldatatype_equals(result.rd_maps.dtype, complex_dtype<double>()));
  const auto* out = result.rd_maps.as<std::complex<double>>();
  for (int64_t tx = 0; tx < kNumTx; ++tx) {
    for (int64_t rx = 0; rx < kNumRx; ++rx) {
      const int64_t v = tx * kNumRx + rx;
      for (int64_t d = 0; d < kDoppler; ++d) {
        for (int64_t r = 0; r < kRange; ++r) {
          const size_t idx = static_cast<size_t>((v * kDoppler + d) * kRange + r);
          EXPECT_NEAR(out[idx].real(), channel_marker(0, tx, rx, d, r), 1e-9);
        }
      }
    }
  }
}

TEST(VirtualArrayAssemblyOp, RejectsOpaqueLeadingBatch) {
  // Contract is rank 4 (M_tx, N_rx, D, R); an extra leading dim is
  // opaque batch (Part B), not a named axis. parse_rd_maps_layout fail-closes.
  const std::vector<int64_t> shape = {2, kNumTx, kNumRx, kDoppler, kRange};
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = kWavelength;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const auto result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape, complex_dtype<float>()),
      std::move(params));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(assembly_maps(result).empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(VirtualArrayAssemblyOp, WavelengthConfigurableFromParams) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  constexpr double kCustomWl = 0.05;
  const auto result =
      run_assembly<float>(build_rd_host<float>(shape), shape, separated_tx_positions(),
                          default_rx_positions(), kCustomWl, OverlapPolicy::kKeepAll);
  EXPECT_NEAR(result.geometry.wavelength, kCustomWl, 1e-12);
}

TEST(VirtualArrayAssemblyOp, LiveWavelengthAppliesOneTickThenReverts) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto input = [&] {
    return to_device_tensor(build_rd_host<float>(shape), shape, complex_dtype<float>());
  };
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = kWavelength;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  using Op = VirtualArrayAssemblyOp<float>;
  auto fixture =
      holoscan::testing::create_operator_test<Op>(holoscan::radar::testing::default_test_extents());
  fixture.set_params(std::move(params));
  for (std::size_t index = 0; index < 3U; ++index) {
    const auto capture_time = std::chrono::milliseconds{static_cast<int64_t>((index + 1U) * 100U)};
    auto frame = input();
    if (index == 1U) {
      fixture.push_frame(holoscan::testing::frame_at(
          capture_time, holoscan::testing::input("wavelength", 2.0 * kWavelength),
          holoscan::testing::input("rd_maps", *frame)));
    } else {
      fixture.push_frame(
          holoscan::testing::frame_at(capture_time, holoscan::testing::input("rd_maps", *frame)));
    }
  }
  constexpr std::size_t kMaximumBytes = 256U * 1024U * 1024U;
  fixture.capture_snapshot<holoscan::Tensor>("rd_maps_out", 3U, kMaximumBytes);
  fixture.capture_snapshot<WireArrayGeometry>("array_geometry", 3U, kMaximumBytes);
  fixture.input_source_device("rd_maps", holoscan::DeviceId{0});
  fixture.output_device("rd_maps_out", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  (void)completion.wait_for_output(running, "rd_maps_out", 3U);
  (void)completion.wait_for_output(running, "array_geometry", 3U);
  running.request_stop();
  const auto result = running.finish();
  completion.check(result);
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  ASSERT_EQ(assembly_maps(result).size(), 3U);

  const auto captured = assembly_geometries(result);
  ASSERT_EQ(captured.size(), 3U);
  std::vector<double> wavelengths;
  for (const auto& wire : captured) {
    const auto geometry = from_wire(wire);
    ASSERT_TRUE(geometry.has_value());
    wavelengths.push_back(geometry->wavelength);
  }
  EXPECT_DOUBLE_EQ(wavelengths[0], kWavelength);
  EXPECT_DOUBLE_EQ(wavelengths[1], 2.0 * kWavelength);
  EXPECT_DOUBLE_EQ(wavelengths[2], kWavelength);
}

TEST(VirtualArrayAssemblyOp, FloatDoubleAgreement) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto f = run_assembly<float>(build_rd_host<float>(shape), shape, separated_tx_positions(),
                                     default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  const auto d = run_assembly<double>(build_rd_host<double>(shape), shape, separated_tx_positions(),
                                      default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
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
  VirtualArrayAssemblyParams params;
  params.tx_positions = {0.0, 0.0, 0.0};
  params.rx_positions = default_rx_positions();
  params.wavelength = kWavelength;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const auto result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape, complex_dtype<float>()),
      std::move(params));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(assembly_maps(result).empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(VirtualArrayAssemblyOp, RejectsMismatchedRxPositions) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = {0.0, 0.0, 0.0};
  params.wavelength = kWavelength;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const auto result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape, complex_dtype<float>()),
      std::move(params));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(assembly_maps(result).empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(VirtualArrayAssemblyOp, RejectsInvalidWavelength) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = 0.0;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const auto result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape, complex_dtype<float>()),
      std::move(params));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(assembly_maps(result).empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

TEST(VirtualArrayAssemblyOp, RejectsRankTooSmall) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler};
  const std::vector<std::complex<float>> host(static_cast<size_t>(tensor_count(shape)),
                                              {1.0f, 0.0f});
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = kWavelength;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const auto result = run_assembly_result_t<float>(
      to_device_tensor(host, shape, complex_dtype<float>()), std::move(params));
  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(assembly_maps(result).empty());
  EXPECT_FALSE(result.diagnostics().empty());
}

// Cell and channel markers are integers below 2^24 even during summation,
// so both precisions have an exact analytic answer, including averaging the
// three coincident channels in the centre of the 3-by-3 aperture. Expected
// groups are authored here, independently of the production virtual plan.
template <typename T>
class VirtualArrayAssemblyTileBoundary : public ::testing::Test {};
using AssemblyTilePrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(VirtualArrayAssemblyTileBoundary, AssemblyTilePrecisions);

struct AssemblyPlane {
  int64_t doppler;
  int64_t range;
};

std::vector<double> linear_positions(int64_t elements, int64_t spacing = 1) {
  std::vector<double> positions;
  for (int64_t element = 0; element < elements; ++element) {
    positions.insert(positions.end(), {static_cast<double>(element * spacing), 0.0, 0.0});
  }
  return positions;
}

template <typename T>
void check_analytic_tile(const AssemblyPlane& shape, const std::vector<double>& tx_positions,
                         const std::vector<double>& rx_positions,
                         const std::vector<std::vector<int64_t>>& collision_groups,
                         OverlapPolicy policy) {
  const int64_t tx = static_cast<int64_t>(tx_positions.size() / 3);
  const int64_t rx = static_cast<int64_t>(rx_positions.size() / 3);
  const int64_t channels = tx * rx;
  const int64_t plane = shape.doppler * shape.range;
  std::vector<std::complex<T>> input(static_cast<std::size_t>(channels * plane));
  for (int64_t channel = 0; channel < channels; ++channel) {
    for (int64_t cell = 0; cell < plane; ++cell) {
      input[static_cast<std::size_t>(channel * plane + cell)] = {
          static_cast<T>(8192 * channel + cell + 1), static_cast<T>(3 * cell - 4096 * channel - 1)};
    }
  }
  const auto result = run_assembly<T>(input, {tx, rx, shape.doppler, shape.range}, tx_positions,
                                      rx_positions, kWavelength, policy);
  const int64_t output_channels =
      policy == OverlapPolicy::kKeepAll ? channels : static_cast<int64_t>(collision_groups.size());
  ASSERT_EQ(result.rd_maps.shape,
            (std::vector<int64_t>{output_channels, shape.doppler, shape.range}));
  ASSERT_TRUE(holoscan::radar::dldatatype_equals(result.rd_maps.dtype, complex_dtype<T>()));
  ASSERT_EQ(result.geometry.num_elements(), output_channels);
  const auto* output = result.rd_maps.template as<std::complex<T>>();
  std::vector<double> expected_positions;
  for (int64_t channel = 0; channel < output_channels; ++channel) {
    int64_t representative = channel;
    int64_t mean_source = channel;
    if (policy != OverlapPolicy::kKeepAll) {
      const auto& group = collision_groups[static_cast<std::size_t>(channel)];
      ASSERT_FALSE(group.empty());
      representative = group.front();
      mean_source = representative;
      if (policy == OverlapPolicy::kAverage) {
        int64_t sum = 0;
        for (const int64_t member : group) {
          sum += member;
        }
        const int64_t members = static_cast<int64_t>(group.size());
        ASSERT_EQ(sum % members, 0) << "Fixture requires an exactly integral mean channel";
        mean_source = sum / members;
      }
    }
    const std::size_t tx_offset = static_cast<std::size_t>((representative / rx) * 3);
    const std::size_t rx_offset = static_cast<std::size_t>((representative % rx) * 3);
    expected_positions.insert(expected_positions.end(),
                              {tx_positions[tx_offset] + rx_positions[rx_offset], 0.0, 0.0});
    for (int64_t cell = 0; cell < plane; ++cell) {
      const std::complex<T> expected{static_cast<T>(8192 * mean_source + cell + 1),
                                     static_cast<T>(3 * cell - 4096 * mean_source - 1)};
      ASSERT_EQ(output[channel * plane + cell], expected)
          << "policy=" << static_cast<int>(policy) << " plane=" << plane
          << " output_channel=" << channel << " cell=" << cell;
    }
  }
  EXPECT_EQ(result.geometry.element_positions, expected_positions);
  EXPECT_EQ(result.geometry.wavelength, kWavelength);
}

template <typename T>
void check_plane_boundaries(OverlapPolicy policy) {
  // Factor large planes to remain within the existing fixture's D/R bounds;
  // 257/4097 exercise partial tiles and 255/4095 leave one cell short.
  const std::vector<AssemblyPlane> planes{{1, 1},   {3, 85},   {4, 64},  {1, 257},
                                          {9, 455}, {16, 256}, {17, 241}};
  const auto positions = linear_positions(3);
  const std::vector<std::vector<int64_t>> groups{{0}, {1, 3}, {2, 4, 6}, {5, 7}, {8}};
  for (const auto& plane : planes) {
    SCOPED_TRACE(::testing::Message() << "D=" << plane.doppler << " R=" << plane.range);
    check_analytic_tile<T>(plane, positions, positions, groups, policy);
  }
}

TYPED_TEST(VirtualArrayAssemblyTileBoundary, KeepAllPreservesEveryComplexCellAtPlaneBoundaries) {
  check_plane_boundaries<TypeParam>(OverlapPolicy::kKeepAll);
}

TYPED_TEST(VirtualArrayAssemblyTileBoundary, DedupKeepsFirstComplexCellAtPlaneBoundaries) {
  check_plane_boundaries<TypeParam>(OverlapPolicy::kDedup);
}

TYPED_TEST(VirtualArrayAssemblyTileBoundary, AverageCombinesAllMembersAtPlaneBoundaries) {
  check_plane_boundaries<TypeParam>(OverlapPolicy::kAverage);
}

TYPED_TEST(VirtualArrayAssemblyTileBoundary, UniqueChannelCountsAroundEightPreservePartialPlanes) {
  for (const int64_t channels : {1, 7, 8, 9}) {
    // Nine channels use 3 TX by 3 RX, rather than exceeding max_rx=8.
    const auto tx = linear_positions(channels == 9 ? 3 : 1, 3);
    const auto rx = linear_positions(channels == 9 ? 3 : channels);
    std::vector<std::vector<int64_t>> groups;
    for (int64_t channel = 0; channel < channels; ++channel) {
      groups.push_back({channel});
    }
    for (const auto policy :
         {OverlapPolicy::kKeepAll, OverlapPolicy::kDedup, OverlapPolicy::kAverage}) {
      SCOPED_TRACE(::testing::Message() << "channels=" << channels);
      check_analytic_tile<TypeParam>({1, 257}, tx, rx, groups, policy);
    }
  }
}

}  // namespace
