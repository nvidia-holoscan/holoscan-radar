// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/testing/operator_test_harness.hpp>

#include "detection/multi_source_expansion_op/multi_source_expansion_op.hpp"
#include "detection/multi_source_expansion_op/multi_source_expansion_stage.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/tensor_fixtures.hpp"
#include "utils/test_extents.hpp"

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

// These SDK-native snapshots prove public scatter/validation and metadata, not
// retained output-loan lifetime; that is covered by the whole-chain integration.
template <typename T>
struct ExpansionFrame {
  std::vector<int64_t> indices{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int64_t> index_shape{4, 2};
  std::vector<T> baseline{10, 11, 12, 13, 14, 15, 16, 17};
  std::vector<int64_t> baseline_shape{4, 2};
  std::vector<int64_t> selected{1, 3};
  std::vector<T> resolved{20, 21, 30, 31};
  std::vector<T> multi{90, 91, 22, 23, 92, 93, 32, 33};
  std::vector<int64_t> multi_shape{2, 2, 2};
  std::vector<int32_t> counts{2, 1};
};

template <typename T>
holoscan::testing::TestResult run_public_expansion(
    const ExpansionFrame<T>& data,
    holoscan::radar::MaxTensorExtents extents = holoscan::radar::testing::default_test_extents()) {
  using Op = holoscan::radar::MultiSourceExpansionOp<T>;
  using holoscan::radar::testing::to_device_tensor;
  using holoscan::testing::input;
  constexpr DLDataType kInt64{kDLInt, 64, 1};
  constexpr DLDataType kInt32{kDLInt, 32, 1};
  const int64_t selected_rows = static_cast<int64_t>(data.selected.size());
  const auto real = holoscan::radar::real_dtype<T>();
  auto fixture = holoscan::testing::create_operator_test<Op>(extents);
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{7},
      input("detection_idx", *to_device_tensor(data.indices, data.index_shape, kInt64)),
      input("baseline_aoa", *to_device_tensor(data.baseline, data.baseline_shape, real)),
      input("selected_idx", *to_device_tensor(data.selected, {selected_rows}, kInt64)),
      input("resolved_aoa", *to_device_tensor(data.resolved, {selected_rows, 2}, real)),
      input("doa_multi", *to_device_tensor(data.multi, data.multi_shape, real)),
      input("source_count", *to_device_tensor(data.counts, {selected_rows}, kInt32))));
  const std::size_t kSnapshotBytes =
      static_cast<std::size_t>(extents.max_points.value_or(1)) * 10U * sizeof(int64_t);
  fixture.template capture_snapshot<holoscan::Tensor>("detection_idx_out", 1U, kSnapshotBytes);
  fixture.template capture_snapshot<holoscan::Tensor>("aoa_estimates_out", 1U, kSnapshotBytes);
  for (const char* name : {"detection_idx", "baseline_aoa", "selected_idx", "resolved_aoa",
                           "doa_multi", "source_count"}) {
    fixture.input_source_device(name, holoscan::DeviceId{0});
  }
  fixture.output_device("detection_idx_out", holoscan::DeviceId{0});
  fixture.output_device("aoa_estimates_out", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool indices_observed =
      running.wait_for_output("detection_idx_out", 1U, std::chrono::seconds{5});
  const bool angles_observed =
      running.wait_for_output("aoa_estimates_out", 1U, std::chrono::seconds{5});
  (void)indices_observed;
  (void)angles_observed;
  running.request_stop();
  return running.finish();
}

template <typename T>
void expect_public_expansion(const holoscan::testing::TestResult& result,
                             const std::vector<int64_t>& expected_indices, int64_t columns,
                             const std::vector<T>& expected_angles) {
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto indices = result.template snapshots<holoscan::Tensor>("detection_idx_out");
  const auto angles = result.template snapshots<holoscan::Tensor>("aoa_estimates_out");
  ASSERT_EQ(indices.size(), 1U);
  ASSERT_EQ(angles.size(), 1U);
  const int64_t rows = static_cast<int64_t>(expected_angles.size() / 2);
  EXPECT_EQ(indices[0].shape(), (std::vector<int64_t>{rows, columns}));
  EXPECT_EQ(angles[0].shape(), (std::vector<int64_t>{rows, 2}));
  EXPECT_EQ(indices[0].source_device().device_type, kDLCUDA);
  EXPECT_EQ(angles[0].source_device().device_type, kDLCUDA);
  EXPECT_EQ(indices[0].dtype().code, kDLInt);
  EXPECT_EQ(indices[0].dtype().bits, 64U);
  EXPECT_EQ(angles[0].dtype().code, kDLFloat);
  EXPECT_EQ(angles[0].dtype().bits, sizeof(T) * 8U);
  const auto& idx_metadata = indices[0].sample_metadata();
  const auto& aoa_metadata = angles[0].sample_metadata();
  EXPECT_EQ(idx_metadata.capture_timestamp_ns, 7000000);
  EXPECT_EQ(aoa_metadata.capture_timestamp_ns, idx_metadata.capture_timestamp_ns);
  EXPECT_TRUE(idx_metadata.source_clock_id.valid());
  EXPECT_EQ(aoa_metadata.source_clock_id, idx_metadata.source_clock_id);
  EXPECT_EQ(idx_metadata.sequence, 1U);
  EXPECT_EQ(aoa_metadata.sequence, 1U);
  const auto host_indices = holoscan::radar::testing::to_host(indices[0]);
  const auto host_angles = holoscan::radar::testing::to_host(angles[0]);
  ASSERT_EQ(host_indices.data.size(), expected_indices.size() * sizeof(int64_t));
  ASSERT_EQ(host_angles.data.size(), expected_angles.size() * sizeof(T));
  for (std::size_t index = 0; index < expected_indices.size(); ++index) {
    EXPECT_EQ(host_indices.template as<int64_t>()[index], expected_indices[index]);
  }
  for (std::size_t index = 0; index < expected_angles.size(); ++index) {
    EXPECT_EQ(host_angles.template as<T>()[index], expected_angles[index]);
  }
}

void expect_expansion_rejected(const holoscan::testing::TestResult& result,
                               const std::string& diagnostic_token) {
  EXPECT_FALSE(result.ok());
  EXPECT_FALSE(result.diagnostics().empty());
  EXPECT_NE(result.describe_failure().find(diagnostic_token), std::string::npos)
      << result.describe_failure();
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("detection_idx_out").empty());
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("aoa_estimates_out").empty());
}

template <typename T>
class MultiSourceExpansionNative : public ::testing::Test {};
TYPED_TEST_SUITE(MultiSourceExpansionNative, ExpansionPrecisions);

TYPED_TEST(MultiSourceExpansionNative, MixedScatterPreservesRowsAnglesAndCaptureCoordinates) {
  const ExpansionFrame<TypeParam> data;
  expect_public_expansion<TypeParam>(run_public_expansion(data), {0, 1, 2, 3, 2, 3, 4, 5, 6, 7}, 2,
                                     {10, 11, 20, 21, 22, 23, 14, 15, 16, 17});
}

TYPED_TEST(MultiSourceExpansionNative, ZeroAndOneCountsIgnoreUnusedNonFiniteAngles) {
  ExpansionFrame<TypeParam> data;
  data.counts = {0, 1};
  const TypeParam nan = std::numeric_limits<TypeParam>::quiet_NaN();
  data.resolved.assign(4, nan);
  data.multi.assign(8, nan);
  expect_public_expansion<TypeParam>(run_public_expansion(data), data.indices, 2, data.baseline);
}

TYPED_TEST(MultiSourceExpansionNative, EmptySelectionEmitsBaselineRows) {
  ExpansionFrame<TypeParam> data;
  data.selected.clear();
  data.resolved.clear();
  data.multi.clear();
  data.multi_shape = {0, 2, 2};
  data.counts.clear();
  expect_public_expansion<TypeParam>(run_public_expansion(data), data.indices, 2, data.baseline);
}

TYPED_TEST(MultiSourceExpansionNative, EmptyItemsEmitBothEmptyTensors) {
  ExpansionFrame<TypeParam> data;
  data.indices.clear();
  data.index_shape = {0, 2};
  data.baseline.clear();
  data.baseline_shape = {0, 2};
  data.selected.clear();
  data.resolved.clear();
  data.multi.clear();
  data.multi_shape = {0, 2, 2};
  data.counts.clear();
  expect_public_expansion<TypeParam>(run_public_expansion(data), {}, 2, {});
}

TYPED_TEST(MultiSourceExpansionNative, FullySelectedFourSourcesPreserveAllTenIndexColumns) {
  ExpansionFrame<TypeParam> data;
  data.indices.resize(20);
  for (int64_t column = 0; column < 20; ++column) {
    data.indices[column] = column;
  }
  data.index_shape = {2, 10};
  data.baseline = {10, 11, 12, 13};
  data.baseline_shape = {2, 2};
  data.selected = {0, 1};
  data.counts = {4, 4};
  data.multi = {90, 91, 22, 23, 24, 25, 26, 27, 92, 93, 32, 33, 34, 35, 36, 37};
  data.multi_shape = {2, 4, 2};
  std::vector<int64_t> expected;
  for (int64_t row = 0; row < 2; ++row) {
    for (int64_t source = 0; source < 4; ++source) {
      for (int64_t column = 0; column < 10; ++column) {
        expected.push_back(data.indices[row * 10 + column]);
      }
    }
  }
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_items = 2;
  extents.max_points = 8;
  extents.max_sources = 4;
  expect_public_expansion<TypeParam>(
      run_public_expansion(data, extents), expected, 10,
      {20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37});
}

TYPED_TEST(MultiSourceExpansionNative, RejectsMissingAndNonPositiveConstructorBounds) {
  using Op = holoscan::radar::MultiSourceExpansionOp<TypeParam>;
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_items.reset();
  EXPECT_THROW((Op{extents}), std::invalid_argument);
  extents = holoscan::radar::testing::default_test_extents();
  extents.max_points = 0;
  EXPECT_THROW((Op{extents}), std::invalid_argument);
  extents = holoscan::radar::testing::default_test_extents();
  extents.max_sources = -1;
  EXPECT_THROW((Op{extents}), std::invalid_argument);
}

TYPED_TEST(MultiSourceExpansionNative, RejectsOutputCapacityOverflowBeforeEitherEmission) {
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_points = 4;
  expect_expansion_rejected(run_public_expansion(ExpansionFrame<TypeParam>{}, extents),
                            "expanded rows exceed max_points");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsInputItemsBeyondConstructorCapacity) {
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_items = 3;
  expect_expansion_rejected(run_public_expansion(ExpansionFrame<TypeParam>{}, extents),
                            "num_items exceeds max_items");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsSourceSlotsBeyondConstructorCapacity) {
  auto extents = holoscan::radar::testing::default_test_extents();
  extents.max_sources = 1;
  expect_expansion_rejected(run_public_expansion(ExpansionFrame<TypeParam>{}, extents),
                            "doa_multi must have shape");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsBaselineRowMismatchBeforeEitherEmission) {
  ExpansionFrame<TypeParam> data;
  data.baseline.resize(6);
  data.baseline_shape = {3, 2};
  expect_expansion_rejected(run_public_expansion(data), "baseline_aoa must have shape");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsNonMatrixDetectionIndicesBeforeEitherEmission) {
  ExpansionFrame<TypeParam> data;
  data.index_shape = {2, 2, 2};
  expect_expansion_rejected(run_public_expansion(data), "detection_idx must have shape");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsDuplicateSelectedRowsBeforeEitherEmission) {
  ExpansionFrame<TypeParam> data;
  data.selected = {1, 1};
  expect_expansion_rejected(run_public_expansion(data), "selected_idx must be strictly ascending");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsInvalidCountBeforeLaterDuplicateSelection) {
  ExpansionFrame<TypeParam> data;
  data.selected = {1, 1};
  data.counts = {3, 2};
  expect_expansion_rejected(run_public_expansion(data), "source_count must be within");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsNonFiniteConfirmedAngleBeforeLaterDuplicate) {
  ExpansionFrame<TypeParam> data;
  data.selected = {1, 1};
  data.resolved[0] = std::numeric_limits<TypeParam>::quiet_NaN();
  expect_expansion_rejected(run_public_expansion(data), "confirmed source angles must be finite");
}

TYPED_TEST(MultiSourceExpansionNative, RejectsNonFiniteSecondaryAngleBeforeEitherEmission) {
  ExpansionFrame<TypeParam> data;
  data.multi[2] = std::numeric_limits<TypeParam>::infinity();
  expect_expansion_rejected(run_public_expansion(data), "confirmed source angles must be finite");
}

// Span multiple 256-thread blocks in both item and selected-row domains. Values
// are exact integers, so float and double have the same independent host oracle.
template <typename T>
ExpansionFrame<T> large_expansion_frame() {
  constexpr int64_t kItems = 1024;
  constexpr int64_t kSources = 4;
  const T nan = std::numeric_limits<T>::quiet_NaN();
  ExpansionFrame<T> data;
  data.index_shape = {kItems, 4};
  data.baseline_shape = {kItems, 2};
  data.indices.resize(kItems * 4);
  data.baseline.resize(kItems * 2);
  data.selected.clear();
  data.resolved.clear();
  data.multi.clear();
  data.counts.clear();
  for (int64_t row = 0; row < kItems; ++row) {
    for (int64_t column = 0; column < 4; ++column) {
      data.indices[row * 4 + column] = row * 10 + column;
    }
    data.baseline[row * 2] = static_cast<T>(row * 2);
    data.baseline[row * 2 + 1] = static_cast<T>(row * 2 + 1);
    if (row % 2 == 0) {
      continue;
    }
    const auto ordinal = static_cast<int32_t>(data.selected.size());
    const int32_t count = ordinal % 5;
    data.selected.push_back(row);
    data.counts.push_back(count);
    data.resolved.push_back(count < 2 ? nan : static_cast<T>(10000 + row * 10));
    data.resolved.push_back(count < 2 ? nan : static_cast<T>(10001 + row * 10));
    for (int64_t source = 0; source < kSources; ++source) {
      // Slot zero is unused even for confirmed rows: dominant angles come from
      // resolved_aoa. Inactive secondary slots must also remain unvalidated.
      const bool used = source > 0 && source < count && count >= 2;
      data.multi.push_back(used ? static_cast<T>(20000 + row * 10 + source * 2) : nan);
      data.multi.push_back(used ? static_cast<T>(20001 + row * 10 + source * 2) : nan);
    }
  }
  data.multi_shape = {static_cast<int64_t>(data.selected.size()), kSources, 2};
  return data;
}

TYPED_TEST(MultiSourceExpansionNative, LargeSparseMixedRowsMatchIndependentHostMerge) {
  const auto data = large_expansion_frame<TypeParam>();
  std::vector<int64_t> expected_indices;
  std::vector<TypeParam> expected_angles;
  std::size_t selected_ordinal = 0;
  for (int64_t row = 0; row < data.index_shape[0]; ++row) {
    const bool selected =
        selected_ordinal < data.selected.size() && data.selected[selected_ordinal] == row;
    const int32_t count = selected ? data.counts[selected_ordinal] : 0;
    const int32_t output_rows = std::max<int32_t>(1, count);
    for (int32_t source = 0; source < output_rows; ++source) {
      for (int64_t column = 0; column < 4; ++column) {
        expected_indices.push_back(data.indices[row * 4 + column]);
      }
      for (int64_t angle = 0; angle < 2; ++angle) {
        if (count < 2) {
          expected_angles.push_back(data.baseline[row * 2 + angle]);
        } else if (source == 0) {
          expected_angles.push_back(data.resolved[selected_ordinal * 2 + angle]);
        } else {
          expected_angles.push_back(data.multi[(selected_ordinal * 4 + source) * 2 + angle]);
        }
      }
    }
    if (selected) {
      ++selected_ordinal;
    }
  }
  ASSERT_EQ(selected_ordinal, data.selected.size());
  expect_public_expansion<TypeParam>(run_public_expansion(data), expected_indices, 4,
                                     expected_angles);
}

TYPED_TEST(MultiSourceExpansionNative, LargeEarlyCountErrorPrecedesDistantSelectionError) {
  auto data = large_expansion_frame<TypeParam>();
  data.counts[17] = 5;
  data.selected[401] = data.selected[400];
  expect_expansion_rejected(run_public_expansion(data), "source_count must be within");
}

TYPED_TEST(MultiSourceExpansionNative, LargeEarlyFiniteErrorPrecedesDistantCountError) {
  auto data = large_expansion_frame<TypeParam>();
  data.resolved[17 * 2] = std::numeric_limits<TypeParam>::quiet_NaN();
  data.counts[401] = 5;
  expect_expansion_rejected(run_public_expansion(data), "confirmed source angles must be finite");
}

TYPED_TEST(MultiSourceExpansionNative, LargeEarlyFiniteErrorPrecedesDistantSelectionError) {
  auto data = large_expansion_frame<TypeParam>();
  data.resolved[17 * 2] = std::numeric_limits<TypeParam>::quiet_NaN();
  data.selected[401] = data.selected[400];
  expect_expansion_rejected(run_public_expansion(data), "confirmed source angles must be finite");
}

TYPED_TEST(MultiSourceExpansionNative, LargeSameRowOrderErrorPrecedesCountAndFiniteErrors) {
  auto data = large_expansion_frame<TypeParam>();
  data.selected[401] = data.selected[400];
  data.counts[401] = 5;
  data.resolved[401 * 2] = std::numeric_limits<TypeParam>::quiet_NaN();
  expect_expansion_rejected(run_public_expansion(data), "selected_idx must be strictly ascending");
}

TYPED_TEST(MultiSourceExpansionNative, LargeSameRowCountErrorPrecedesFiniteError) {
  auto data = large_expansion_frame<TypeParam>();
  data.counts[401] = 5;
  data.resolved[401 * 2] = std::numeric_limits<TypeParam>::quiet_NaN();
  expect_expansion_rejected(run_public_expansion(data), "source_count must be within");
}

}  // namespace
