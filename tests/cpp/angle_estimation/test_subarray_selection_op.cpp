// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SubarraySelectionOp contract tests (S3: select a uniform sub-array from a
// holey MIMO virtual array; emit reduced snapshot + array_geometry).

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "angle_estimation/subarray_selection_op/subarray_selection_op.hpp"
#include "radar_common/descriptors/array_geometry.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::ArrayGeometry;
using holoscan::radar::complex_dtype;
using holoscan::radar::SubarraySelectionOp;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::test::create_operator_test;

constexpr double kD = 0.5;

ArrayGeometry make_linear(const std::vector<int>& iy) {
  ArrayGeometry g;
  g.wavelength = 1.0;
  for (int i : iy) {
    g.element_positions.insert(g.element_positions.end(), {0.0, i * kD, 0.0});
  }
  return g;
}

// snapshot (num_items, N, K) with value[item,c,s] = (c, s) encoded; only the
// shape and emitted geometry are asserted here.
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

template <typename... Args>
Result run_select(std::shared_ptr<holoscan::Tensor> snap, Args&&... args) {
  auto harness = create_operator_test<SubarraySelectionOp<double>>(
      std::forward<Args>(args)...);
  harness->template add_input_port<holoscan::TensorMap>(
      "snapshot", {single_map(std::move(snap), "snapshot")});
  TensorMapCapture snap_cap;
  ArrayGeometry geom;
  harness->template add_output_port<holoscan::TensorMap>(
      "snapshot", snap_cap.validators());
  harness->template add_output_port<ArrayGeometry>(
      "array_geometry",
      {[&geom](const ArrayGeometry& g) { geom = g; }});
  harness->run_test();
  return {snap_cap.frames.front(), geom};
}

std::set<double> y_set(const ArrayGeometry& g) {
  std::set<double> s;
  for (int64_t e = 0; e < g.num_elements(); ++e) {
    s.insert(g.element_positions[e * 3 + 1]);
  }
  return s;
}

TEST(SubarraySelectionOpTest, HoleyUlaSelectsLongestRun) {
  const auto geom = make_linear({0, 1, 2, 3, 5, 6});  // longest run 0..3 → L=4
  auto snap = make_snapshot(1, 6, 8);
  const auto out =
      run_select(snap, holoscan::Arg("array_geometry") = geom,
                 holoscan::Arg("min_subarray_size") = int64_t{2});
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 8}));
  EXPECT_EQ(out.geometry.num_elements(), 4);
  EXPECT_EQ(y_set(out.geometry),
            (std::set<double>{0.0, kD, 2 * kD, 3 * kD}));
}

TEST(SubarraySelectionOpTest, FilledUlaPassesThrough) {
  const auto geom = make_linear({0, 1, 2, 3});
  auto snap = make_snapshot(1, 4, 5);
  const auto out = run_select(snap, holoscan::Arg("array_geometry") = geom);
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 5}));
  EXPECT_EQ(out.geometry.num_elements(), 4);
}

TEST(SubarraySelectionOpTest, ExplicitSubarray) {
  const auto geom = make_linear({0, 1, 2, 3, 4, 5, 6, 7});
  auto snap = make_snapshot(1, 8, 3);
  const auto out = run_select(
      snap, holoscan::Arg("array_geometry") = geom,
      holoscan::Arg("subarray") = std::vector<int64_t>{2, 3, 4, 5});
  EXPECT_EQ(out.snapshot.shape, (std::vector<int64_t>{1, 4, 3}));
  EXPECT_EQ(y_set(out.geometry),
            (std::set<double>{2 * kD, 3 * kD, 4 * kD, 5 * kD}));
}

}  // namespace
