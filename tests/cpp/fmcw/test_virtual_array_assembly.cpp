// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Contract tests for VirtualArrayAssemblyOp (HSDK 5 Graph harness):
// virtual layout, overlap policies, array_geometry packed Tensor emission,
// opaque leading-batch rejection, [live] wavelength overrides (mid-stream
// harness §1 deferred), cross-cutting input validation, and float/double
// agreement.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "fmcw/virtual_array_assembly_op/virtual_array_assembly_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/descriptors/enums.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ArrayGeometry;
using holoscan::radar::OverlapPolicy;
using holoscan::radar::VirtualArrayAssemblyOp;
using holoscan::radar::VirtualArrayAssemblyParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::real_dtype;
using holoscan::radar::unpack_array_geometry;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
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
std::vector<std::complex<T>> build_rd_host(
    const std::vector<int64_t>& shape) {
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

// Graph wiring lambda for 1-in / 2-out VirtualArrayAssemblyOp.
template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_assembly(holoscan::Graph& graph, const Source& source, OpRef& op,
                   const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->rd_maps, q);
  graph.add_flow(op->rd_maps_out, sink0->in, q);
  graph.add_flow(op->array_geometry, sink1->in, q);
}

struct AssemblyResult {
  HostTensor rd_maps;
  ArrayGeometry geometry;
};

DualIOGraphRunResult run_assembly_result(
    std::shared_ptr<holoscan::Tensor> input,
    VirtualArrayAssemblyParams params) {
  return run_single_in_dual_out_operator_graph_result<
      VirtualArrayAssemblyOp<float>>(
      {std::move(input)}, std::move(params), "rd_maps_out", "array_geometry",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_assembly(graph, source, op, sink0, sink1);
      });
}

template <typename T>
DualIOGraphRunResult run_assembly_result_t(
    std::shared_ptr<holoscan::Tensor> input,
    VirtualArrayAssemblyParams params) {
  return run_single_in_dual_out_operator_graph_result<
      VirtualArrayAssemblyOp<T>>(
      {std::move(input)}, std::move(params), "rd_maps_out", "array_geometry",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_assembly(graph, source, op, sink0, sink1);
      });
}

template <typename T>
AssemblyResult run_assembly(
    const std::vector<std::complex<T>>& host,
    const std::vector<int64_t>& shape,
    const std::vector<double>& tx_positions,
    const std::vector<double>& rx_positions, double wavelength,
    OverlapPolicy policy) {
  VirtualArrayAssemblyParams params;
  params.tx_positions = tx_positions;
  params.rx_positions = rx_positions;
  params.wavelength = wavelength;
  params.overlap_policy = policy;
  DualIOGraphRunResult result = run_assembly_result_t<T>(
      to_device_tensor(host, shape, complex_dtype<T>()), std::move(params));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames0.size() != 1U || result.frames1.size() != 1U) {
    throw std::runtime_error(
        "VirtualArrayAssemblyOp test: expected one output frame pair, got " +
        std::to_string(result.frames0.size()) + "/" +
        std::to_string(result.frames1.size()));
  }
  const HostTensor& geom_t = result.frames1.front();
  if (geom_t.shape.size() != 2U) {
    throw std::runtime_error(
        "VirtualArrayAssemblyOp test: array_geometry rank != 2");
  }
  ArrayGeometry geom = unpack_array_geometry(
      geom_t.as<double>(), geom_t.shape[0], geom_t.shape[1]);
  return {std::move(result.frames0.front()), std::move(geom)};
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
            const size_t out_idx = static_cast<size_t>(
                b * (num_tx * num_rx * doppler * range) +
                ((v * doppler + d) * range + r));
            const double expected_val =
                channel_marker(b, tx, rx, d, r);
            const double tol =
                std::max(1e-4, 1e-6 * std::abs(expected_val));
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
      EXPECT_NEAR(out[origin_idx].real(),
                  channel_marker(0, 0, 0, d, r), 1e-5)
          << "dedup must keep the first colliding (tx, rx) channel";
      EXPECT_NEAR(out[rx1_idx].real(),
                  channel_marker(0, 0, 1, d, r), 1e-5)
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

  VirtualArrayAssemblyParams params;
  params.tx_positions = overlapping_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = kWavelength;
  // overlap_policy defaults to kDedup in VirtualArrayAssemblyParams.
  DualIOGraphRunResult result = run_assembly_result_t<float>(
      to_device_tensor(host, shape, complex_dtype<float>()),
      std::move(params));
  ASSERT_TRUE(result.plan_ok);
  ASSERT_FALSE(result.frames0.empty());
  ASSERT_FALSE(result.frames1.empty());
  EXPECT_EQ(result.frames0.front().shape, explicit_dedup.rd_maps.shape);
  // Unpack and check geometry element count.
  const HostTensor& geom_t = result.frames1.front();
  ArrayGeometry geom = unpack_array_geometry(
      geom_t.as<double>(), geom_t.shape[0], geom_t.shape[1]);
  EXPECT_EQ(geom.num_elements(), explicit_dedup.geometry.num_elements());
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
          0.5 * (channel_marker(0, 0, 0, d, r) +
                 channel_marker(0, 1, 0, d, r));
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
          EXPECT_NEAR(out[idx].real(),
                      channel_marker(0, tx, rx, d, r), 1e-9);
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
  const DualIOGraphRunResult result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape,
                       complex_dtype<float>()),
      std::move(params));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(VirtualArrayAssemblyOp, LiveWavelengthOverrideMidStream) {
  // [EA HSDK5 ONLY] Graph mid-stream override harness is deferred
  // (HSDK5_PORT.md §1). Skip until the Graph harness supports multi-tick
  // override injection.
  GTEST_SKIP() << "HSDK 5 Graph: mid-stream override harness "
                  "deferred (HSDK5_PORT §1)";
}

TEST(VirtualArrayAssemblyOp, WavelengthConfigurableFromParams) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  constexpr double kCustomWl = 0.05;
  const auto result = run_assembly<float>(
      build_rd_host<float>(shape), shape, separated_tx_positions(),
      default_rx_positions(), kCustomWl, OverlapPolicy::kKeepAll);
  EXPECT_NEAR(result.geometry.wavelength, kCustomWl, 1e-12);
}

TEST(VirtualArrayAssemblyOp, FloatDoubleAgreement) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  const auto f = run_assembly<float>(
      build_rd_host<float>(shape), shape, separated_tx_positions(),
      default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  const auto d = run_assembly<double>(
      build_rd_host<double>(shape), shape, separated_tx_positions(),
      default_rx_positions(), kWavelength, OverlapPolicy::kKeepAll);
  ASSERT_EQ(f.rd_maps.shape, d.rd_maps.shape);
  const int64_t count = f.rd_maps.size();
  ASSERT_EQ(count, d.rd_maps.size());
  const auto* f_data = f.rd_maps.as<std::complex<float>>();
  const auto* d_data = d.rd_maps.as<std::complex<double>>();
  for (int64_t i = 0; i < count; ++i) {
    EXPECT_NEAR(
        static_cast<double>(f_data[static_cast<size_t>(i)].real()),
        d_data[static_cast<size_t>(i)].real(), 1e-5);
    EXPECT_NEAR(
        static_cast<double>(f_data[static_cast<size_t>(i)].imag()),
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
  const DualIOGraphRunResult result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape,
                       complex_dtype<float>()),
      std::move(params));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(VirtualArrayAssemblyOp, RejectsMismatchedRxPositions) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = {0.0, 0.0, 0.0};
  params.wavelength = kWavelength;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const DualIOGraphRunResult result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape,
                       complex_dtype<float>()),
      std::move(params));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(VirtualArrayAssemblyOp, RejectsInvalidWavelength) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler, kRange};
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = 0.0;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const DualIOGraphRunResult result = run_assembly_result_t<float>(
      to_device_tensor(build_rd_host<float>(shape), shape,
                       complex_dtype<float>()),
      std::move(params));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

TEST(VirtualArrayAssemblyOp, RejectsPageableHostInput) {
  // [EA HSDK5 ONLY] Graph harness materializes inputs through device-bound
  // tensor loans, so pageable host data never reaches compute(). Guard
  // remains but is untestable through the Graph model.
  GTEST_SKIP() << "HSDK 5 Graph: pageable host input cannot reach "
                  "the operator through device-bound tensor loans";
}

TEST(VirtualArrayAssemblyOp, RejectsRankTooSmall) {
  const std::vector<int64_t> shape = {kNumTx, kNumRx, kDoppler};
  const std::vector<std::complex<float>> host(
      static_cast<size_t>(tensor_count(shape)), {1.0f, 0.0f});
  VirtualArrayAssemblyParams params;
  params.tx_positions = separated_tx_positions();
  params.rx_positions = default_rx_positions();
  params.wavelength = kWavelength;
  params.overlap_policy = OverlapPolicy::kKeepAll;
  const DualIOGraphRunResult result = run_assembly_result_t<float>(
      to_device_tensor(host, shape, complex_dtype<float>()),
      std::move(params));
  EXPECT_TRUE(result.plan_ok);
  EXPECT_TRUE(result.frames0.empty());
  EXPECT_FALSE(result.diagnostics.empty());
}

}  // namespace
