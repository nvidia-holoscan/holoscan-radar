// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "detection/multi_source_expansion_op/multi_source_expansion_stage.hpp"

namespace {

using holoscan::radar::detail::ExpansionError;

template <typename T>
int32_t run_plan(const std::vector<int64_t>& selected,
                 const std::vector<T>& resolved,
                 const std::vector<T>& multi,
                 const std::vector<int32_t>& counts, int64_t items,
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
  cudaMalloc(&selected_device, selected.size() * sizeof(int64_t));
  cudaMalloc(&resolved_device, resolved.size() * sizeof(T));
  cudaMalloc(&multi_device, multi.size() * sizeof(T));
  cudaMalloc(&counts_device, counts.size() * sizeof(int32_t));
  cudaMalloc(&rows_device, items * sizeof(uint32_t));
  cudaMalloc(&ordinals_device, items * sizeof(int64_t));
  cudaMalloc(&total_device, sizeof(int64_t));
  cudaMalloc(&error_device, sizeof(int32_t));
  cudaMemcpy(selected_device, selected.data(), selected.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(resolved_device, resolved.data(), resolved.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(multi_device, multi.data(), multi.size() * sizeof(T),
             cudaMemcpyHostToDevice);
  cudaMemcpy(counts_device, counts.data(), counts.size() * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  holoscan::radar::detail::build_expansion_plan_kernel<T><<<1, 1>>>(
      selected_device, selected.size(), resolved_device, multi_device,
      counts_device, items, max_sources, rows_device, ordinals_device,
      total_device, error_device);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int32_t error = 0;
  row_counts->resize(items);
  ordinals->resize(items);
  cudaMemcpy(&error, error_device, sizeof(error), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_counts->data(), rows_device, items * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(ordinals->data(), ordinals_device, items * sizeof(int64_t),
             cudaMemcpyDeviceToHost);
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

TEST(MultiSourceExpansionStage, BuildsStableMixedExpansionPlan) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const int32_t error = run_plan<double>(
      {1, 3}, {-0.1, 0.0, 0.2, 0.0},
      {-0.1, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.0}, {2, 1}, 4, 2,
      &rows, &ordinals, &total);
  EXPECT_EQ(error, static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_EQ(rows, (std::vector<uint32_t>{1, 2, 1, 1}));
  EXPECT_EQ(ordinals, (std::vector<int64_t>{-1, 0, -1, 1}));
  EXPECT_EQ(total, 5);
}

TEST(MultiSourceExpansionStage, RejectsMalformedSelectionsAndCounts) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  EXPECT_EQ(run_plan<float>({2, 2}, {0, 0, 0, 0},
                            {0, 0, 0, 0, 0, 0, 0, 0}, {2, 2}, 3, 2,
                            &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSelectedOrder));
  EXPECT_EQ(run_plan<float>({1}, {0, 0}, {0, 0, 0, 0}, {3}, 3, 2,
                            &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kSourceCount));
}

TEST(MultiSourceExpansionStage, RejectsNonFiniteResolvedAngles) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_EQ(run_plan<float>({0}, {nan, 0}, {0, 0, 0.1F, 0}, {2}, 1, 2,
                            &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNonFiniteDoa));
}

TEST(MultiSourceExpansionStage, RejectsNegativeUnsortedAndOutOfRangeRows) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  for (const auto& selected : {std::vector<int64_t>{-1},
                               std::vector<int64_t>{2, 1},
                               std::vector<int64_t>{3}}) {
    const int64_t count = static_cast<int64_t>(selected.size());
    EXPECT_EQ(run_plan<float>(selected, std::vector<float>(count * 2),
                              std::vector<float>(count * 4),
                              std::vector<int32_t>(count, 2), 3, 2,
                              &rows, &ordinals, &total),
              static_cast<int32_t>(ExpansionError::kSelectedOrder));
  }
}

TEST(MultiSourceExpansionStage, ZeroAndOneSourceUseSingleRowFallback) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  EXPECT_EQ(run_plan<double>({0, 2}, {0, 0, 0, 0},
                             std::vector<double>(8), {0, 1}, 3, 2,
                             &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNone));
  EXPECT_EQ(rows, (std::vector<uint32_t>{1, 1, 1}));
  EXPECT_EQ(total, 3);
}

TEST(MultiSourceExpansionStage, RejectsNonFiniteSecondaryAngle) {
  std::vector<uint32_t> rows;
  std::vector<int64_t> ordinals;
  int64_t total = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(run_plan<double>({0}, {0, 0}, {0, 0, nan, 0}, {2}, 1, 2,
                             &rows, &ordinals, &total),
            static_cast<int32_t>(ExpansionError::kNonFiniteDoa));
}

}  // namespace
