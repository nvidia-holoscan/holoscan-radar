// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "detection/multi_source_expansion_op/multi_source_expansion_stage.hpp"

namespace {

using holoscan::radar::detail::ExpansionError;

template <typename T>
class MultiSourceExpansionStage : public ::testing::Test {};
using ExpansionPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MultiSourceExpansionStage, ExpansionPrecisions);

template <typename T>
int32_t run_plan(const std::vector<int64_t>& selected, const std::vector<T>& resolved,
                 const std::vector<T>& multi, const std::vector<int32_t>& counts, int64_t items,
                 int64_t max_sources, std::vector<uint32_t>* row_counts,
                 std::vector<int64_t>* ordinals, int64_t* total) {
  int64_t* selected_device = nullptr;
  T* resolved_device = nullptr;
  T* multi_device = nullptr;
  int32_t* counts_device = nullptr;
  uint32_t* rows_device = nullptr;
  int64_t* ordinals_device = nullptr;
  int64_t* total_device = nullptr;
  int32_t* error_device = nullptr;
  cudaMalloc(&selected_device, std::max<std::size_t>(1, selected.size()) * sizeof(int64_t));
  cudaMalloc(&resolved_device, std::max<std::size_t>(1, resolved.size()) * sizeof(T));
  cudaMalloc(&multi_device, std::max<std::size_t>(1, multi.size()) * sizeof(T));
  cudaMalloc(&counts_device, std::max<std::size_t>(1, counts.size()) * sizeof(int32_t));
  cudaMalloc(&rows_device,
             std::max<std::size_t>(1, static_cast<std::size_t>(items)) * sizeof(uint32_t));
  cudaMalloc(&ordinals_device,
             std::max<std::size_t>(1, static_cast<std::size_t>(items)) * sizeof(int64_t));
  cudaMalloc(&total_device, sizeof(int64_t));
  cudaMalloc(&error_device, sizeof(int32_t));
  if (!selected.empty()) {
    cudaMemcpy(selected_device, selected.data(), selected.size() * sizeof(int64_t),
               cudaMemcpyHostToDevice);
  }
  if (!resolved.empty()) {
    cudaMemcpy(resolved_device, resolved.data(), resolved.size() * sizeof(T),
               cudaMemcpyHostToDevice);
  }
  if (!multi.empty()) {
    cudaMemcpy(multi_device, multi.data(), multi.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  if (!counts.empty()) {
    cudaMemcpy(counts_device, counts.data(), counts.size() * sizeof(int32_t),
               cudaMemcpyHostToDevice);
  }
  holoscan::radar::detail::build_expansion_plan_kernel<T>
      <<<1, 1>>>(selected_device, selected.size(), resolved_device, multi_device, counts_device,
                 items, max_sources, rows_device, ordinals_device, total_device, error_device);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int32_t error = 0;
  row_counts->resize(items);
  ordinals->resize(items);
  cudaMemcpy(&error, error_device, sizeof(error), cudaMemcpyDeviceToHost);
  if (items > 0) {
    cudaMemcpy(row_counts->data(), rows_device, items * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  }
  if (items > 0) {
    cudaMemcpy(ordinals->data(), ordinals_device, items * sizeof(int64_t), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(total, total_device, sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaFree(error_device);
  cudaFree(total_device);
  cudaFree(ordinals_device);
  cudaFree(rows_device);
  cudaFree(counts_device);
  cudaFree(multi_device);
  cudaFree(resolved_device);
  cudaFree(selected_device);
  return error;
}

TYPED_TEST(MultiSourceExpansionStage, BuildsStableMixedExpansionPlan) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const int32_t error =
      run_plan<TypeParam>({1, 3}, {-0.1, 0.0, 0.2, 0.0}, {-0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0},
                          {2, 1}, 4, 2, &rows, &ordinals, &total);
  EXPECT_EQ(error, static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_EQ(rows, (std::vector<uint32_t>{1, 2, 1, 1}));
  EXPECT_EQ(ordinals, (std::vector<int64_t>{-1, 0, -1, 1}));
  EXPECT_EQ(total, 5);
}

TYPED_TEST(MultiSourceExpansionStage, RejectsMalformedSelectionsAndCounts) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  EXPECT_EQ(run_plan<TypeParam>({2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {2, 2}, 3, 2, &rows,
                                &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSelectedOrder));
  EXPECT_EQ(run_plan<TypeParam>({1}, {0, 0}, {0, 0, 0, 0}, {3}, 3, 2, &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSourceCount));
}

TYPED_TEST(MultiSourceExpansionStage, RejectsNonFiniteResolvedAngles) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
  EXPECT_EQ(
      run_plan<TypeParam>({0}, {nan, 0}, {0, 0, 0.1F, 0}, {2}, 1, 2, &rows, &ordinals, &total),
      static_cast<int32_t>(ExpansionError::kNonFiniteDoa));
}

TYPED_TEST(MultiSourceExpansionStage, RejectsNegativeUnsortedAndOutOfRangeRows) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  for (const auto& selected :
       {std::vector<int64_t>{-1}, std::vector<int64_t>{2, 1}, std::vector<int64_t>{3}}) {
    const int64_t count = static_cast<int64_t>(selected.size());
    EXPECT_EQ(run_plan<TypeParam>(selected, std::vector<TypeParam>(count * 2),
                                  std::vector<TypeParam>(count * 4), std::vector<int32_t>(count, 2),
                                  3, 2, &rows, &ordinals, &total),
              static_cast<int32_t>(ExpansionError::kSelectedOrder));
  }
}

TYPED_TEST(MultiSourceExpansionStage, ZeroAndOneSourceUseSingleRowFallback) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  EXPECT_EQ(run_plan<TypeParam>({0, 2}, {0, 0, 0, 0}, std::vector<TypeParam>(8), {0, 1}, 3, 2,
                                &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_EQ(rows, (std::vector<uint32_t>{1, 1, 1}));
  EXPECT_EQ(total, 3);
}

TYPED_TEST(MultiSourceExpansionStage, RejectsNonFiniteSecondaryAngle) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
  EXPECT_EQ(run_plan<TypeParam>({0}, {0, 0}, {0, 0, nan, 0}, {2}, 1, 2, &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNonFiniteDoa));
}

TYPED_TEST(MultiSourceExpansionStage, EmptySelectionKeepsEveryBaselineRow) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = -1;
  EXPECT_EQ(run_plan<TypeParam>({}, {}, {}, {}, 4, 2, &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_EQ(rows, (std::vector<uint32_t>{1, 1, 1, 1}));
  EXPECT_EQ(ordinals, (std::vector<int64_t>{-1, -1, -1, -1}));
  EXPECT_EQ(total, 4);
}

TYPED_TEST(MultiSourceExpansionStage, EmptyItemsAndSelectionProduceZeroRows) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = -1;
  EXPECT_EQ(run_plan<TypeParam>({}, {}, {}, {}, 0, 2, &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_TRUE(rows.empty());
  EXPECT_TRUE(ordinals.empty());
  EXPECT_EQ(total, 0);
}

TYPED_TEST(MultiSourceExpansionStage, FourSourceCapacityPreservesMixedCounts) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = -1;
  EXPECT_EQ(run_plan<TypeParam>({0, 2}, std::vector<TypeParam>(4), std::vector<TypeParam>(16),
                                {4, 3}, 3, 4, &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_EQ(rows, (std::vector<uint32_t>{4, 1, 3}));
  EXPECT_EQ(ordinals, (std::vector<int64_t>{0, -1, 1}));
  EXPECT_EQ(total, 8);
}

TYPED_TEST(MultiSourceExpansionStage, FirstMalformedSelectedRowDeterminesReportedError) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  // The first row's bad count precedes the later duplicate-index error.
  EXPECT_EQ(run_plan<TypeParam>({1, 1}, std::vector<TypeParam>(4), std::vector<TypeParam>(8),
                                {3, 2}, 3, 2, &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSourceCount));
  const TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
  // The first row's non-finite angle also precedes the later duplicate.
  EXPECT_EQ(run_plan<TypeParam>({1, 1}, {nan, 0, 0, 0}, std::vector<TypeParam>(8), {2, 2}, 3, 2,
                                &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNonFiniteDoa));
}

TYPED_TEST(MultiSourceExpansionStage, SameRowValidationPrecedenceRemainsStable) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
  EXPECT_EQ(run_plan<TypeParam>({3}, {nan, 0}, std::vector<TypeParam>(4), {3}, 3, 2, &rows,
                                &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSelectedOrder));
  EXPECT_EQ(run_plan<TypeParam>({0}, {nan, 0}, std::vector<TypeParam>(4), {3}, 3, 2, &rows,
                                &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSourceCount));
}

TYPED_TEST(MultiSourceExpansionStage, FallbackRowsDoNotValidateUnusedAngles) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
  EXPECT_EQ(
      run_plan<TypeParam>({0, 2}, std::vector<TypeParam>(4, nan), std::vector<TypeParam>(8, nan),
                          {0, 1}, 3, 2, &rows, &ordinals, &total),
      static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_EQ(rows, (std::vector<uint32_t>{1, 1, 1}));
  EXPECT_EQ(total, 3);
}

TYPED_TEST(MultiSourceExpansionStage, RejectsNegativeSourceCount) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  EXPECT_EQ(run_plan<TypeParam>({0}, {0, 0}, std::vector<TypeParam>(4), {-1}, 1, 2, &rows,
                                &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSourceCount));
}

TYPED_TEST(MultiSourceExpansionStage, LargeNonPrefixSelectionKeepsOriginalRowOrdinals) {
  constexpr int64_t kItems = 16384;
  std::vector<int64_t> selected;
  for (int64_t row = 1; row < kItems; row += 2) {
    selected.push_back(row);
  }
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  ASSERT_EQ(run_plan<TypeParam>(selected, std::vector<TypeParam>(selected.size() * 2),
                                std::vector<TypeParam>(selected.size() * 4),
                                std::vector<int32_t>(selected.size(), 2), kItems, 2, &rows,
                                &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNone));
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(kItems));
  ASSERT_EQ(ordinals.size(), static_cast<std::size_t>(kItems));
  EXPECT_EQ(total, kItems + kItems / 2);
  for (int64_t row = 0; row < kItems; ++row) {
    EXPECT_EQ(rows[row], row % 2 == 0 ? 1U : 2U);
    EXPECT_EQ(ordinals[row], row % 2 == 0 ? -1 : row / 2);
  }
}

}  // namespace
