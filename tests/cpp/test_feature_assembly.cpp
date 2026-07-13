// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FeatureAssemblyOp contract tests (gather / bin lookup → points matrix).

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/test/test_harness.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {
using holoscan::radar::AssemblyFeature;
using holoscan::radar::FeatureAssemblyOp;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::single_map;
using holoscan::radar::testing::TensorMapCapture;
using holoscan::radar::testing::to_device_tensor;
using holoscan::test::create_operator_test;

template <typename T>
HostTensor run_assembly(
    std::shared_ptr<holoscan::Tensor> detection_idx,
    const std::vector<AssemblyFeature>& features,
    std::shared_ptr<holoscan::Tensor> range_bins = nullptr,
    std::shared_ptr<holoscan::Tensor> velocity_bins = nullptr,
    std::shared_ptr<holoscan::Tensor> detection_score = nullptr,
    std::shared_ptr<holoscan::Tensor> aoa_estimates = nullptr) {
  auto harness = create_operator_test<FeatureAssemblyOp<T>>(
      holoscan::Arg("features") = features);
  harness->template add_input_port<holoscan::TensorMap>(
      "detection_idx", {single_map(std::move(detection_idx), "detection_idx")});
  if (range_bins) {
    harness->template add_input_port<holoscan::TensorMap>(
        "range_bins", {single_map(std::move(range_bins), "range_bins")});
  }
  if (velocity_bins) {
    harness->template add_input_port<holoscan::TensorMap>(
        "velocity_bins",
        {single_map(std::move(velocity_bins), "velocity_bins")});
  }
  if (detection_score) {
    harness->template add_input_port<holoscan::TensorMap>(
        "detection_score",
        {single_map(std::move(detection_score), "detection_score")});
  }
  if (aoa_estimates) {
    harness->template add_input_port<holoscan::TensorMap>(
        "aoa_estimates",
        {single_map(std::move(aoa_estimates), "aoa_estimates")});
  }

  TensorMapCapture points_cap;
  harness->template add_output_port<holoscan::TensorMap>(
      "points", points_cap.validators());
  harness->run_test();
  if (points_cap.frames.size() != 1u) {
    throw std::runtime_error("FeatureAssemblyOp test: expected 1 points frame");
  }
  return points_cap.frames.front();
}

TEST(FeatureAssemblyOpTest, AssemblesRawBinEscapeHatches) {
  // Two detections: (d,r) = (3,5) and (7,2).
  std::vector<int64_t> idx = {3, 5, 7, 2};
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  const auto points = run_assembly<double>(
      idx_t, {AssemblyFeature::kDopplerBin, AssemblyFeature::kRangeBin});
  ASSERT_EQ(points.shape.size(), 2u);
  EXPECT_EQ(points.shape[0], 2);
  EXPECT_EQ(points.shape[1], 2);
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 3.0);
  EXPECT_DOUBLE_EQ(p[1], 5.0);
  EXPECT_DOUBLE_EQ(p[2], 7.0);
  EXPECT_DOUBLE_EQ(p[3], 2.0);
}

TEST(FeatureAssemblyOpTest, LooksUpPhysicalRangeVelocityAndScore) {
  constexpr int64_t kD = 8;
  constexpr int64_t kR = 16;
  std::vector<int64_t> idx = {2, 4};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});

  std::vector<double> range_bins(kR);
  for (int64_t r = 0; r < kR; ++r) {
    range_bins[static_cast<size_t>(r)] = 10.0 * static_cast<double>(r);
  }
  std::vector<double> vel_bins(kD);
  for (int64_t d = 0; d < kD; ++d) {
    vel_bins[static_cast<size_t>(d)] = -1.5 * static_cast<double>(d);
  }
  std::vector<double> score(static_cast<size_t>(kD * kR), 0.0);
  score[static_cast<size_t>(2 * kR + 4)] = 12.5;

  auto rb = to_device_tensor(range_bins, {kR}, real_dtype<double>());
  auto vb = to_device_tensor(vel_bins, {kD}, real_dtype<double>());
  auto sc = to_device_tensor(score, {kD, kR}, real_dtype<double>());

  const auto points = run_assembly<double>(
      idx_t,
      {AssemblyFeature::kRange, AssemblyFeature::kVelocity,
       AssemblyFeature::kScore},
      rb, vb, sc);
  ASSERT_EQ(points.shape[0], 1);
  ASSERT_EQ(points.shape[1], 3);
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 40.0);   // range_bins[4]
  EXPECT_DOUBLE_EQ(p[1], -3.0);   // velocity_bins[2]
  EXPECT_DOUBLE_EQ(p[2], 12.5);   // score[2,4]
}

TEST(FeatureAssemblyOpTest, RangeFallsBackToRawBinWhenBinsAbsent) {
  std::vector<int64_t> idx = {1, 9};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  const auto points = run_assembly<float>(
      idx_t, {AssemblyFeature::kRange, AssemblyFeature::kVelocity});
  const float* p = points.as<float>();
  EXPECT_FLOAT_EQ(p[0], 9.0f);
  EXPECT_FLOAT_EQ(p[1], 1.0f);
}

TEST(FeatureAssemblyOpTest, AppendsAoaEstimates) {
  std::vector<int64_t> idx = {0, 1, 2, 3};
  auto idx_t = to_device_tensor(idx, {2, 2}, DLDataType{kDLInt, 64, 1});
  std::vector<double> aoa = {0.1, -0.2, 0.3, 0.4};
  auto aoa_t = to_device_tensor(aoa, {2, 2}, real_dtype<double>());
  const auto points = run_assembly<double>(
      idx_t,
      {AssemblyFeature::kDopplerBin, AssemblyFeature::kAz,
       AssemblyFeature::kEl},
      nullptr, nullptr, nullptr, aoa_t);
  ASSERT_EQ(points.shape[1], 3);
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 0.0);
  EXPECT_DOUBLE_EQ(p[1], 0.1);
  EXPECT_DOUBLE_EQ(p[2], -0.2);
  EXPECT_DOUBLE_EQ(p[3], 2.0);
  EXPECT_DOUBLE_EQ(p[4], 0.3);
  EXPECT_DOUBLE_EQ(p[5], 0.4);
}

TEST(FeatureAssemblyOpTest, RejectsScoreFeatureWithoutScoreInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(
      run_assembly<double>(idx_t, {AssemblyFeature::kScore}),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsEmptyFeatures) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(run_assembly<double>(idx_t, {}), std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, EmptyDetectionIdxEmitsEmptyPoints) {
  std::vector<int64_t> idx;
  auto idx_t = to_device_tensor(idx, {0, 2}, DLDataType{kDLInt, 64, 1});
  const auto points = run_assembly<double>(
      idx_t, {AssemblyFeature::kDopplerBin, AssemblyFeature::kRangeBin});
  ASSERT_EQ(points.shape.size(), 2u);
  EXPECT_EQ(points.shape[0], 0);
  EXPECT_EQ(points.shape[1], 2);
}

TEST(FeatureAssemblyOpTest, FloatDoubleAgreeOnPhysicalLookup) {
  constexpr int64_t kR = 4;
  std::vector<int64_t> idx = {1, 2};
  auto idx_d = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  auto idx_f = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});

  std::vector<double> range_d = {0, 10, 20, 30};
  std::vector<float> range_f = {0, 10, 20, 30};
  auto rb_d = to_device_tensor(range_d, {kR}, real_dtype<double>());
  auto rb_f = to_device_tensor(range_f, {kR}, real_dtype<float>());

  const auto pd = run_assembly<double>(idx_d, {AssemblyFeature::kRange}, rb_d);
  const auto pf = run_assembly<float>(idx_f, {AssemblyFeature::kRange}, rb_f);
  EXPECT_NEAR(pd.as<double>()[0], static_cast<double>(pf.as<float>()[0]),
              1e-5);
}

TEST(FeatureAssemblyOpTest, GathersScoreAcrossPreservedBatch) {
  // detection_idx columns: (batch, doppler, range) — matches CFAR with one
  // preserved leading axis. Score map shape (B=2, D=4, R=4).
  constexpr int64_t kB = 2;
  constexpr int64_t kD = 4;
  constexpr int64_t kR = 4;
  // Two detections in different batch slices.
  std::vector<int64_t> idx = {
      0, 1, 2,  // batch 0, d=1, r=2 → score should be 11
      1, 2, 3,  // batch 1, d=2, r=3 → score should be 23
  };
  auto idx_t = to_device_tensor(idx, {2, 3}, DLDataType{kDLInt, 64, 1});

  std::vector<double> score(static_cast<size_t>(kB * kD * kR), 0.0);
  score[static_cast<size_t>(0 * kD * kR + 1 * kR + 2)] = 11.0;
  score[static_cast<size_t>(1 * kD * kR + 2 * kR + 3)] = 23.0;
  auto sc = to_device_tensor(score, {kB, kD, kR}, real_dtype<double>());

  const auto points = run_assembly<double>(
      idx_t, {AssemblyFeature::kScore}, nullptr, nullptr, sc);
  ASSERT_EQ(points.shape[0], 2);
  ASSERT_EQ(points.shape[1], 1);
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 11.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], 23.0);
}

TEST(FeatureAssemblyOpTest, RejectsAzFeatureWithoutAoaInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(run_assembly<double>(idx_t, {AssemblyFeature::kAz}),
               std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsBadDetectionIdxRank) {
  std::vector<int64_t> idx = {0, 1, 2};
  auto idx_t = to_device_tensor(idx, {3}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(
      run_assembly<double>(idx_t, {AssemblyFeature::kDopplerBin}),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsElFeatureWithoutAoaInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  EXPECT_THROW(run_assembly<double>(idx_t, {AssemblyFeature::kEl}),
               std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsScoreLeadingRankMismatch) {
  // Score has a leading batch dim but detection_idx has no preserved cols.
  std::vector<int64_t> idx = {0, 1};
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  std::vector<double> score(2 * 4 * 4, 1.0);
  auto sc = to_device_tensor(score, {2, 4, 4}, real_dtype<double>());
  EXPECT_THROW(
      run_assembly<double>(idx_t, {AssemblyFeature::kScore}, nullptr, nullptr,
                           sc),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsOobRangeBinWhenBinsPresent) {
  std::vector<int64_t> idx = {0, 99};  // range_bin 99 with only 4 bins
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  std::vector<double> range_bins = {0, 10, 20, 30};
  auto rb = to_device_tensor(range_bins, {4}, real_dtype<double>());
  EXPECT_THROW(run_assembly<double>(idx_t, {AssemblyFeature::kRange}, rb),
               std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsOobScoreCoords) {
  std::vector<int64_t> idx = {8, 1};  // doppler 8 out of range for D=4
  auto idx_t = to_device_tensor(idx, {1, 2}, DLDataType{kDLInt, 64, 1});
  std::vector<double> score(4 * 4, 1.0);
  auto sc = to_device_tensor(score, {4, 4}, real_dtype<double>());
  EXPECT_THROW(
      run_assembly<double>(idx_t, {AssemblyFeature::kScore}, nullptr, nullptr,
                           sc),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsOobPreservedBatchCoord) {
  // batch coord 2 is OOB for score leading dim B=2.
  std::vector<int64_t> idx = {2, 0, 0};
  auto idx_t = to_device_tensor(idx, {1, 3}, DLDataType{kDLInt, 64, 1});
  std::vector<double> score(2 * 4 * 4, 1.0);
  auto sc = to_device_tensor(score, {2, 4, 4}, real_dtype<double>());
  EXPECT_THROW(
      run_assembly<double>(idx_t, {AssemblyFeature::kScore}, nullptr, nullptr,
                           sc),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsNonContiguousDetectionIdx) {
  // Shape (1, 2) with strides (2, 2): elements at offsets 0 and 2 — not
  // compact row-major, so is_contiguous() is false.
  struct Holder {
    void* data{nullptr};
    std::vector<int64_t> shape{1, 2};
    std::vector<int64_t> strides{2, 2};
    DLManagedTensor dl{};
  };
  auto holder = std::make_unique<Holder>();
  if (cudaMalloc(&holder->data, 4 * sizeof(int64_t)) != cudaSuccess) {
    FAIL() << "cudaMalloc failed";
  }
  std::vector<int64_t> host = {0, 0, 1, 0};  // values at stride slots 0 and 2
  cudaMemcpy(holder->data, host.data(), 4 * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  int device_id = 0;
  cudaGetDevice(&device_id);
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = DLDevice{kDLCUDA, device_id};
  holder->dl.dl_tensor.ndim = 2;
  holder->dl.dl_tensor.dtype = DLDataType{kDLInt, 64, 1};
  holder->dl.dl_tensor.shape = holder->shape.data();
  holder->dl.dl_tensor.strides = holder->strides.data();
  holder->dl.dl_tensor.byte_offset = 0;
  holder->dl.manager_ctx = holder.get();
  holder->dl.deleter = [](DLManagedTensor* self) {
    auto* h = static_cast<Holder*>(self->manager_ctx);
    cudaFree(h->data);
    delete h;
  };
  auto idx_t = std::make_shared<holoscan::Tensor>(&holder->dl);
  holder.release();
  ASSERT_FALSE(idx_t->is_contiguous());
  EXPECT_THROW(
      run_assembly<double>(idx_t, {AssemblyFeature::kDopplerBin,
                                   AssemblyFeature::kRangeBin}),
      std::invalid_argument);
}

}  // namespace
