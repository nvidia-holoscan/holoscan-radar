// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include <holoscan/config/io.hpp>

#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/sdk/config_codecs.hpp"
#include "radar_common/tensor/max_byte_span.hpp"

namespace {

using holoscan::radar::MaxTensorExtents;
using holoscan::radar::byte_span_complex_cube;
using holoscan::radar::require_extent;

holoscan::config::Document parse(const char* text) {
  return holoscan::config_io::parse(text, holoscan::config_io::Format::kYaml);
}

TEST(MaxTensorExtents, DecodesOptionalFields) {
  const auto extents = parse(R"(
max_chirps: 128
max_range_bins: 512
max_rx: 8
max_az: null
)")
                           .get<MaxTensorExtents>("");
  ASSERT_TRUE(extents.max_chirps.has_value());
  EXPECT_EQ(*extents.max_chirps, 128);
  ASSERT_TRUE(extents.max_range_bins.has_value());
  EXPECT_EQ(*extents.max_range_bins, 512);
  ASSERT_TRUE(extents.max_rx.has_value());
  EXPECT_EQ(*extents.max_rx, 8);
  EXPECT_FALSE(extents.max_az.has_value());
  EXPECT_FALSE(extents.max_tx.has_value());
}

TEST(MaxTensorExtents, RejectsNonPositive) {
  EXPECT_FALSE(parse("max_chirps: 0").try_get<MaxTensorExtents>(""));
  EXPECT_FALSE(parse("max_chirps: -1").try_get<MaxTensorExtents>(""));
}

TEST(MaxTensorExtents, RejectsNonMapAndUnknownKeys) {
  EXPECT_FALSE(parse("[1, 2]").try_get<MaxTensorExtents>(""));
  EXPECT_FALSE(parse("max_chirp: 4").try_get<MaxTensorExtents>(""));
}

TEST(MaxTensorExtents, EncodeDecodeRoundTrip) {
  MaxTensorExtents src;
  src.max_chirps = 64;
  src.max_az = 101;
  const auto encoded = holoscan::config::Document::from_value(holoscan::config::encode_config(src));
  EXPECT_EQ(encoded.get<MaxTensorExtents>(""), src);
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
