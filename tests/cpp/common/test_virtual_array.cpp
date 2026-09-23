// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "radar_common/descriptors/enums.hpp"
#include "radar_common/dsp/virtual_array.hpp"

namespace {

namespace hr = holoscan::radar;

constexpr const char* kCaller = "PlannerTestOp";

TEST(VirtualArrayPlan, KeepAllUsesTxMajorRxMinorPositionSums) {
  const auto plan = hr::detail::build_virtual_plan({0.0, 0.0, 0.0, 10.0, 20.0, 30.0},
                                                   {0.0, 0.0, 0.0, 1.0, 2.0, 3.0}, 2, 2,
                                                   hr::OverlapPolicy::kKeepAll, kCaller);

  EXPECT_EQ(plan.num_virtual, 4);
  EXPECT_EQ(plan.virtual_positions, (std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 10.0, 20.0,
                                                         30.0, 11.0, 22.0, 33.0}));
  EXPECT_EQ(plan.src_channel, (std::vector<int64_t>{0, 1, 2, 3}));
  EXPECT_TRUE(plan.member_offsets.empty());
  EXPECT_TRUE(plan.member_indices.empty());
}

TEST(VirtualArrayPlan, DedupKeepsFirstScanOrderRepresentative) {
  const auto plan =
      hr::detail::build_virtual_plan({0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                     2, 2, hr::OverlapPolicy::kDedup, kCaller);

  EXPECT_EQ(plan.virtual_positions,
            (std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0}));
  EXPECT_EQ(plan.src_channel, (std::vector<int64_t>{0, 1, 3}));
  EXPECT_TRUE(plan.member_indices.empty());
}

TEST(VirtualArrayPlan, AverageBuildsDeterministicCsrGroups) {
  const auto plan =
      hr::detail::build_virtual_plan({0.0, 0.0, 0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                     2, 2, hr::OverlapPolicy::kAverage, kCaller);

  EXPECT_EQ(plan.num_virtual, 3);
  EXPECT_TRUE(plan.src_channel.empty());
  EXPECT_EQ(plan.member_offsets, (std::vector<int64_t>{0, 1, 3, 4}));
  EXPECT_EQ(plan.member_indices, (std::vector<int64_t>{0, 1, 2, 3}));
}

TEST(VirtualArrayPlan, CoincidenceToleranceIsOneNanometer) {
  const auto merged = hr::detail::build_virtual_plan(
      {0.0, 0.0, 0.0, 0.5e-9, 0.0, 0.0}, {0.0, 0.0, 0.0}, 2, 1, hr::OverlapPolicy::kDedup, kCaller);
  const auto separate = hr::detail::build_virtual_plan(
      {0.0, 0.0, 0.0, 2.0e-9, 0.0, 0.0}, {0.0, 0.0, 0.0}, 2, 1, hr::OverlapPolicy::kDedup, kCaller);

  EXPECT_EQ(merged.num_virtual, 1);
  EXPECT_EQ(separate.num_virtual, 2);
}

TEST(VirtualArrayPlan, ValidationNamesTheCallingOperator) {
  try {
    (void)hr::detail::build_virtual_plan({0.0, 0.0}, {0.0, 0.0, 0.0}, 1, 1,
                                         hr::OverlapPolicy::kKeepAll, "TDMSparseCompensationOp");
    FAIL() << "expected invalid_argument";
  } catch (const std::invalid_argument& error) {
    EXPECT_EQ(std::string(error.what()),
              "TDMSparseCompensationOp: tx_positions length must be num_tx * 3");
  }
}

TEST(VirtualArrayPlan, RejectsNonFinitePositionsAndEmptyArrays) {
  EXPECT_THROW((void)hr::detail::build_virtual_plan(
                   {std::numeric_limits<double>::infinity(), 0.0, 0.0}, {0.0, 0.0, 0.0}, 1, 1,
                   hr::OverlapPolicy::kDedup, kCaller),
               std::invalid_argument);
  EXPECT_THROW((void)hr::detail::build_virtual_plan({}, {0.0, 0.0, 0.0}, 0, 1,
                                                    hr::OverlapPolicy::kDedup, kCaller),
               std::invalid_argument);
}

TEST(VirtualArrayPlan, GeometryPreservesPlannedOrderAndWavelength) {
  const auto plan =
      hr::detail::build_virtual_plan({0.0, 0.0, 0.0, 2.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
                                     2, 2, hr::OverlapPolicy::kAverage, kCaller);
  const auto geometry = hr::detail::make_array_geometry(plan, 0.0039);

  EXPECT_EQ(geometry.element_positions, plan.virtual_positions);
  EXPECT_DOUBLE_EQ(geometry.wavelength, 0.0039);
  EXPECT_FALSE(geometry.geometry_type.has_value());
  EXPECT_EQ(geometry.num_elements(), plan.num_virtual);
}

}  // namespace
