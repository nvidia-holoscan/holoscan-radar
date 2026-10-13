// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <string>

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/max_tensor_extents_yaml.hpp"
#include "radar_common/tensor/max_byte_span.hpp"

namespace {

using holoscan::radar::MaxTensorExtents;
using holoscan::radar::MaxTensorExtentsFromYaml;
using holoscan::radar::byte_span_complex_cube;
using holoscan::radar::require_extent;

TEST(MaxTensorExtents, FromYamlParsesOptionalFields) {
  const auto extents = MaxTensorExtentsFromYaml(YAML::Load(R"(
max_chirps: 128
max_range_bins: 512
max_rx: 8
)"));
  ASSERT_TRUE(extents.max_chirps.has_value());
  EXPECT_EQ(*extents.max_chirps, 128);
  ASSERT_TRUE(extents.max_range_bins.has_value());
  EXPECT_EQ(*extents.max_range_bins, 512);
  ASSERT_TRUE(extents.max_rx.has_value());
  EXPECT_EQ(*extents.max_rx, 8);
  EXPECT_FALSE(extents.max_az.has_value());
}

TEST(MaxTensorExtents, FromYamlRejectsNonPositive) {
  EXPECT_THROW(MaxTensorExtentsFromYaml(YAML::Load("max_chirps: 0")), std::invalid_argument);
  EXPECT_THROW(MaxTensorExtentsFromYaml(YAML::Load("max_chirps: -1")), std::invalid_argument);
}

TEST(MaxTensorExtents, FromYamlRejectsNonMap) {
  EXPECT_THROW(MaxTensorExtentsFromYaml(YAML::Load("[1, 2]")), std::invalid_argument);
}

TEST(MaxTensorExtents, YamlConvertRoundTrip) {
  MaxTensorExtents src;
  src.max_chirps = 64;
  src.max_az = 101;
  const YAML::Node node = YAML::convert<MaxTensorExtents>::encode(src);
  const auto dst = node.as<MaxTensorExtents>();
  EXPECT_EQ(dst, src);
}

TEST(MaxTensorExtents, RequireExtent) {
  MaxTensorExtents extents;
  extents.max_chirps = 16;
  EXPECT_EQ(require_extent(extents.max_chirps, "max_chirps", "TestOp"), 16);
  EXPECT_THROW(require_extent(extents.max_range_bins, "max_range_bins", "TestOp"),
               std::invalid_argument);
}

TEST(MaxByteSpan, ComplexCubeProduct) {
  EXPECT_EQ(byte_span_complex_cube<float>({2, 8, 16}, "test"),
            static_cast<std::size_t>(2 * 8 * 16 * 8));
  EXPECT_THROW(byte_span_complex_cube<float>({2, 0, 16}, "test"), std::invalid_argument);
}

}  // namespace
