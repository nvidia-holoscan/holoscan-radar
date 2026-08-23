// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "radar_common/sdk/live_override_payloads.hpp"

namespace holoscan::radar {
namespace {

TEST(LiveOverridePayloads, CfarCellWindowConvertsToAxisPair) {
  const auto result = from_wire(
      CfarCellWindowOverride{.doppler = 2, .range = 3});
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, (std::vector<std::int64_t>{2, 3}));
}

TEST(LiveOverridePayloads, SpatialSmoothingPreservesLogicalForms) {
  auto result = from_wire(SpatialSmoothingOverride{});
  ASSERT_TRUE(result);
  EXPECT_TRUE(result->empty());

  result = from_wire(
      SpatialSmoothingOverride{.count = 1, .first = 3});
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, (std::vector<std::int64_t>{3}));

  result = from_wire(
      SpatialSmoothingOverride{.count = 2, .first = 2, .second = 4});
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, (std::vector<std::int64_t>{2, 4}));
}

TEST(LiveOverridePayloads, SpatialSmoothingRejectsInvalidCount) {
  const auto result =
      from_wire(SpatialSmoothingOverride{.count = 3});
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace holoscan::radar
