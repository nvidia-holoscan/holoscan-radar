// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "radar_app/report/report_ops.hpp"

namespace holoscan::radar::app {
namespace {

TEST(ReportShapes, RejectsNegativeDimensionsBeforeAllocation) {
  for (const std::vector<int64_t>& shape :
       {std::vector<int64_t>{-1, 2}, std::vector<int64_t>{2, -1}}) {
    const auto count =
        checked_report_elements(shape, 0, sizeof(int64_t), "ReportOp", "detection_idx");
    EXPECT_FALSE(count.has_value());
  }
}

TEST(ReportShapes, RejectsOverflowAndForgedByteCount) {
  const auto overflow = checked_report_elements({std::numeric_limits<int64_t>::max(), 2}, 0,
                                                sizeof(int64_t), "ReportOp", "detection_idx");
  EXPECT_FALSE(overflow.has_value());

  const auto forged =
      checked_report_elements({1024, 2}, 0, sizeof(int64_t), "ReportOp", "detection_idx");
  EXPECT_FALSE(forged.has_value());

  const auto negative_bytes =
      checked_report_elements({1, 2}, -1, sizeof(int64_t), "ReportOp", "detection_idx");
  EXPECT_FALSE(negative_bytes.has_value());
}

TEST(ReportShapes, AcceptsValidAndEmptyFrames) {
  const auto valid = checked_report_elements({3, 2}, 6 * sizeof(int64_t), sizeof(int64_t),
                                             "ReportOp", "detection_idx");
  ASSERT_TRUE(valid.has_value());
  EXPECT_EQ(*valid, 6U);

  const auto empty =
      checked_report_elements({0, 2}, 0, sizeof(int64_t), "ReportOp", "detection_idx");
  ASSERT_TRUE(empty.has_value());
  EXPECT_EQ(*empty, 0U);
}

}  // namespace
}  // namespace holoscan::radar::app
