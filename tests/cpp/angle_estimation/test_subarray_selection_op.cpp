// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SubarraySelectionOp contract tests (HSDK 5 Graph harness): S3 select a
// uniform sub-array from a holey MIMO virtual array; emit reduced snapshot +
// packed array_geometry Tensor.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/descriptors/array_geometry_tensor.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::ArrayGeometry;
using holoscan::radar::SubarraySelectionOp;
using holoscan::radar::SubarraySelectionParams;
using holoscan::radar::complex_dtype;
using holoscan::radar::pack_array_geometry;
using holoscan::radar::testing::DualIOGraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_single_in_dual_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::unpack_array_geometry;

constexpr double kD = 0.5;

ArrayGeometry make_linear(const std::vector<int>& iy) {
  ArrayGeometry g;
  g.wavelength = 1.0;
  for (int i : iy) {
    g.element_positions.insert(g.element_positions.end(), {0.0, i * kD, 0.0});
  }
  return g;
}

// snapshot (num_items, N, K) with value[item,c,s] = (c, s) encoded — the real
// part carries the channel index and the imag part the snapshot index.
std::shared_ptr<holoscan::Tensor> make_snapshot(int64_t items, int64_t n,
                                                int64_t k) {
  std::vector<std::complex<double>> x(
      static_cast<size_t>(items * n * k), {0.0, 0.0});
  for (int64_t it = 0; it < items; ++it) {
    for (int64_t c = 0; c < n; ++c) {
      for (int64_t s = 0; s < k; ++s) {
        x[static_cast<size_t>((it * n + c) * k + s)] = {
            static_cast<double>(c), static_cast<double>(s)};
      }
    }
  }
  return to_device_tensor(x, {items, n, k}, complex_dtype<double>());
}

struct Result {
  HostTensor snapshot;
  ArrayGeometry geometry;
};

template <typename Source, typename OpRef, typename Sink0, typename Sink1>
void wire_subarray(holoscan::Graph& graph, const Source& source, OpRef& op,
                   const Sink0& sink0, const Sink1& sink1) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->snapshot, q);
  graph.add_flow(op->snapshot_out, sink0->in, q);
  graph.add_flow(op->array_geometry, sink1->in, q);
}

DualIOGraphRunResult run_select_result(
    std::shared_ptr<holoscan::Tensor> snap, SubarraySelectionParams params) {
  return run_single_in_dual_out_operator_graph_result<
      SubarraySelectionOp<double>>(
      {std::move(snap)}, std::move(params), "snapshot_out", "array_geometry",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink0, const auto& sink1) {
        wire_subarray(graph, source, op, sink0, sink1);
      },
      // Optional is_valid_out is declared but unwired; still needs a device
      // bind (IMPLICIT_DEVICE_ZERO_REJECTED otherwise).
      {"is_valid_out"});
}

Result run_select(std::shared_ptr<holoscan::Tensor> snap,
                  SubarraySelectionParams params) {
  DualIOGraphRunResult result =
      run_select_result(std::move(snap), std::move(params));
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
        "SubarraySelectionOp test: expected one output frame pair, got " +
        std::to_string(result.frames0.size()) + "/" +
        std::to_string(result.frames1.size()));
  }
  const HostTensor& geom_t = result.frames1.front();
  if (geom_t.shape.size() != 2U) {
    throw std::runtime_error(
        "SubarraySelectionOp test: array_geometry rank != 2");
  }
  ArrayGeometry geom = unpack_array_geometry(
      geom_t.as<double>(), geom_t.shape[0], geom_t.shape[1]);
  return {std::move(result.frames0.front()), std::move(geom)};
}

std::set<double> y_set(const ArrayGeometry& g) {
  std::set<double> s;
  for (int64_t e = 0; e < g.num_elements(); ++e) {
    s.insert(g.element_positions[static_cast<size_t>(e * 3 + 1)]);
  }
  return s;
}

void expect_channels(const HostTensor& snapshot,
                     const std::vector<int64_t>& channels, int64_t num_snap) {
  const int64_t l = static_cast<int64_t>(channels.size());
  ASSERT_EQ(snapshot.shape, (std::vector<int64_t>{1, l, num_snap}));
  const auto* snap = snapshot.as<std::complex<double>>();
  for (int64_t j = 0; j < l; ++j) {
    for (int64_t s = 0; s < num_snap; ++s) {
      const auto& got = snap[static_cast<size_t>(j * num_snap + s)];
      EXPECT_NEAR(got.real(), static_cast<double>(channels[static_cast<size_t>(j)]),
                  1e-6)
          << "out channel " << j << " snapshot " << s;
      EXPECT_NEAR(got.imag(), static_cast<double>(s), 1e-6)
          << "out channel " << j << " snapshot " << s;
    }
  }
}

SubarraySelectionParams make_params(ArrayGeometry geom,
                                    std::vector<int64_t> subarray = {},
                                    int64_t min_size = 2) {
  SubarraySelectionParams params{};
  params.array_geometry = std::move(geom);
  params.subarray = std::move(subarray);
  params.min_subarray_size = min_size;
  return params;
}

TEST(SubarraySelectionOpTest, HoleyUlaSelectsLongestRun) {
  const auto geom = make_linear({0, 1, 2, 3, 5, 6});  // longest run 0..3 → L=4
  auto snap = make_snapshot(1, 6, 8);
  const auto out =
      run_select(snap, make_params(geom, {}, /*min_size=*/2));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 8}));
  EXPECT_EQ(out.geometry.num_elements(), 4);
  EXPECT_EQ(y_set(out.geometry),
            (std::set<double>{0.0, kD, 2 * kD, 3 * kD}));
  expect_channels(out.snapshot, {0, 1, 2, 3}, 8);
}

TEST(SubarraySelectionOpTest, FilledUlaPassesThrough) {
  const auto geom = make_linear({0, 1, 2, 3});
  auto snap = make_snapshot(1, 4, 5);
  const auto out = run_select(snap, make_params(geom));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 5}));
  EXPECT_EQ(out.geometry.num_elements(), 4);
  EXPECT_EQ(out.geometry.wavelength, 1.0);
}

TEST(SubarraySelectionOpTest, ExplicitSubarray) {
  const auto geom = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto snap = make_snapshot(1, 8, 3);
  const auto out =
      run_select(snap, make_params(geom, {2, 3, 4, 5}));
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 3}));
  EXPECT_EQ(y_set(out.geometry),
            (std::set<double>{2 * kD, 3 * kD, 4 * kD, 5 * kD}));
  expect_channels(out.snapshot, {2, 3, 4, 5}, 3);
}

TEST(SubarraySelectionOpTest, SubarrayConfigurableFromParams) {
  const auto geom = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto snap = make_snapshot(1, 8, 2);
  SubarraySelectionParams params = make_params(geom, {1, 2, 3});
  const auto out = run_select(snap, params);
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 3, 2}));
  expect_channels(out.snapshot, {1, 2, 3}, 2);
}

TEST(SubarraySelectionOpTest, PackUnpackRoundTrip) {
  auto geom = make_linear({0, 2, 4});
  geom.validate();
  const auto packed = pack_array_geometry(geom);
  const auto back = unpack_array_geometry(
      packed.data(), geom.num_elements() + 1, 3);
  EXPECT_EQ(back, geom);
}

}  // namespace
