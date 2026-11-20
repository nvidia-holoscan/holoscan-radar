// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "radar_common/tensor/tensor_utils.hpp"
#include "sar/backprojection_op/backprojection_convert.cuh"
#include "utils/tensor_fixtures.hpp"

namespace {
namespace detail = holoscan::radar::detail;

TEST(BackprojectionConversions, LargeCountsRemainWideAndUnrepresentableCountsReject) {
  constexpr int64_t expected = int64_t{1} << 32;
  EXPECT_EQ(detail::profile_scalar_count<float>(16384, 131072), expected);
  EXPECT_EQ(detail::profile_scalar_count<double>(16384, 131072), expected);
  EXPECT_EQ(detail::conversion_blocks(expected), 4096);
  EXPECT_THROW(detail::profile_scalar_count<double>(std::numeric_limits<int64_t>::max(), 2),
               std::invalid_argument);
  EXPECT_THROW(detail::profile_scalar_count<float>(1LL << 32, 1LL << 32), std::invalid_argument);
  EXPECT_THROW(detail::profile_scalar_count<float>(0, 128), std::invalid_argument);
  EXPECT_THROW(detail::profile_scalar_count<double>(128, -1), std::invalid_argument);
}

template <typename T>
class BackprojectionConversions : public ::testing::Test {};
using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(BackprojectionConversions, Precisions);

TYPED_TEST(BackprojectionConversions, ConversionCoversGridStrideTailAndPreservesGuards) {
  using In = TypeParam;
  using Out = std::conditional_t<std::is_same_v<In, float>, double, float>;
  for (const int64_t count : {1, 255, 256, 257, 1048593}) {
    std::vector<In> input(static_cast<std::size_t>(count));
    for (int64_t i = 0; i < count; ++i)
      input[static_cast<std::size_t>(i)] = static_cast<In>((i % 129) - 64) / In{8};
    std::vector<Out> output(static_cast<std::size_t>(count) + 2, Out{-123});
    auto src = holoscan::radar::testing::to_device_tensor(input, {count},
                                                          holoscan::radar::real_dtype<In>());
    auto dst = holoscan::radar::testing::to_device_tensor(output, {count + 2},
                                                          holoscan::radar::real_dtype<Out>());
    auto writable = dst->mutable_data();
    ASSERT_TRUE(writable);
    detail::convert_backprojection(static_cast<const In*>(src->data()),
                                   static_cast<Out*>(*writable) + 1, count, nullptr);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(output.data(), dst->data(), output.size() * sizeof(Out), cudaMemcpyDeviceToHost),
        cudaSuccess);
    EXPECT_EQ(output.front(), Out{-123});
    EXPECT_EQ(output.back(), Out{-123});
    for (int64_t i = 0; i < count; ++i) {
      ASSERT_EQ(output[static_cast<std::size_t>(i) + 1], static_cast<Out>(input[i])) << i;
    }
  }
}
}  // namespace
