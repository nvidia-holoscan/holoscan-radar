// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include "radar_common/math/uniform_grid.hpp"

namespace {

using holoscan::radar::make_uniform_grid;

TEST(UniformGridTest, IncludesAlignedMaximum) {
  const auto actual = make_uniform_grid(-0.3, 0.3, 0.05);
  const std::vector<double> expected{-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0,
                                     0.05, 0.1,   0.15, 0.2,   0.25, 0.3};
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1.0e-14);
  }
  EXPECT_DOUBLE_EQ(actual.back(), 0.3);
}

TEST(UniformGridTest, SupportsSingleton) {
  EXPECT_EQ(make_uniform_grid(0.0, 0.0, 1.0), (std::vector<double>{0.0}));
}

TEST(UniformGridTest, RejectsInvalidSpecifications) {
  EXPECT_THROW(make_uniform_grid(1.0, 0.0, 0.1), std::invalid_argument);
  EXPECT_THROW(make_uniform_grid(0.0, 1.0, 0.0), std::invalid_argument);
  EXPECT_THROW(make_uniform_grid(0.0, 1.0, -0.1), std::invalid_argument);
  EXPECT_THROW(make_uniform_grid(0.0, 1.0, 0.3), std::invalid_argument);
  EXPECT_THROW(make_uniform_grid(0.0, std::numeric_limits<double>::infinity(), 0.1),
               std::invalid_argument);
  EXPECT_THROW(make_uniform_grid(std::numeric_limits<double>::quiet_NaN(), 1.0, 0.1),
               std::invalid_argument);
  EXPECT_THROW(make_uniform_grid(0.0, 1.0, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(make_uniform_grid(0.0, 10.0, 1.0, "grid", 10U), std::invalid_argument);
}

}  // namespace
