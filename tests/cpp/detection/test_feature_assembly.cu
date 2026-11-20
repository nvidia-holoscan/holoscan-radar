// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FeatureAssemblyOp contract tests (HSDK 5 Graph harness + stage kernels):
// static features defines both output columns and the compiled temporal join.
// Selected sources are exact-aligned; unselected gathers may remain unwired.
// detection_idx is rank 2; extra leading axes fail closed.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/testing/operator_test_harness.hpp>

#include "utils/fixture_completion.hpp"
#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_stage.hpp"
#include "radar_common/descriptors/max_tensor_extents.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/test_extents.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AssemblyFeature;
using holoscan::radar::FeatureAssemblyOp;
using holoscan::radar::make_device_tensor;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};

std::vector<AssemblyFeature> make_features(std::vector<AssemblyFeature> features) {
  return features;
}

HostTensor require_one_frame(holoscan::testing::TestResult result);

template <typename T>
holoscan::testing::TestResult run_assembly_frame(
    std::vector<AssemblyFeature> features, const holoscan::testing::TestFrame& frame,
    std::initializer_list<std::string_view> device_inputs,
    const holoscan::radar::MaxTensorExtents& extents =
        holoscan::radar::testing::default_test_extents()) {
  using Op = FeatureAssemblyOp<T>;
  const std::size_t max_bytes =
      static_cast<std::size_t>(extents.max_points.value_or(1) * extents.max_features.value_or(1) *
                               static_cast<std::int64_t>(sizeof(T)));
  auto fixture = holoscan::testing::create_operator_test<Op>(extents, std::move(features));
  fixture.push_frame(frame);
  fixture.template capture_snapshot<holoscan::Tensor>("points", 1U, max_bytes);
  for (const std::string_view name : device_inputs) {
    fixture.input_source_device(name, holoscan::DeviceId{0});
  }
  fixture.output_device("points", holoscan::DeviceId{0});
  holoscan::radar::testing::FixtureCompletion completion;
  auto running = fixture.run_async(completion.run_options());
  const bool observed = completion.wait_for_output(running, "points", 1U);
  (void)observed;
  running.request_stop();
  auto fixture_result = running.finish();
  completion.check(fixture_result);
  return fixture_result;
}

template <typename T>
holoscan::testing::TestResult run_assembly_result(std::shared_ptr<holoscan::Tensor> idx,
                                                  std::vector<AssemblyFeature> features) {
  return run_assembly_frame<T>(
      std::move(features),
      holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                  holoscan::testing::input("detection_idx", *idx)),
      {"detection_idx"});
}

template <typename T>
HostTensor run_assembly(std::shared_ptr<holoscan::Tensor> idx,
                        std::vector<AssemblyFeature> features) {
  return require_one_frame(run_assembly_result<T>(std::move(idx), std::move(features)));
}

bool run_failed_with_diagnostic(const holoscan::testing::TestResult& result) {
  return !result.ok() && !result.diagnostics().empty();
}

HostTensor require_one_frame(holoscan::testing::TestResult result) {
  if (!result.ok()) {
    throw std::runtime_error(result.describe_failure());
  }
  return to_host(result.template snapshot<holoscan::Tensor>("points"));
}

// ---------------------------------------------------------------------------
// Stage helpers — focused kernel and error-path coverage.
// ---------------------------------------------------------------------------

template <typename T>
HostTensor stage_assemble(const std::vector<int64_t>& idx_host, int64_t num_points,
                          int64_t idx_cols, const std::vector<AssemblyFeature>& features,
                          const T* range_bins, int64_t num_range, bool have_range,
                          const T* velocity_bins, int64_t num_doppler, bool have_velocity,
                          const T* score_map, int64_t num_score_doppler, int64_t num_score_range,
                          const int64_t* score_plane_offsets, bool have_score, const T* aoa,
                          bool have_aoa, const T* az_grid, int64_t num_az, const T* el_grid,
                          int64_t num_el) {
  const int64_t num_features = static_cast<int64_t>(features.size());
  const int64_t n_preserved = idx_cols - 2;

  auto idx_dev = to_device_tensor(idx_host, {num_points, idx_cols}, kInt64Dtype);
  std::vector<int> feature_ids(static_cast<size_t>(num_features));
  for (int64_t i = 0; i < num_features; ++i) {
    switch (features[static_cast<size_t>(i)]) {
      case AssemblyFeature::kRange:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatRange;
        break;
      case AssemblyFeature::kVelocity:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatVelocity;
        break;
      case AssemblyFeature::kScore:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatScore;
        break;
      case AssemblyFeature::kAz:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatAz;
        break;
      case AssemblyFeature::kEl:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatEl;
        break;
      case AssemblyFeature::kAzEstimate:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatAzEstimate;
        break;
      case AssemblyFeature::kElEstimate:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatElEstimate;
        break;
      case AssemblyFeature::kAzBin:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatAzBin;
        break;
      case AssemblyFeature::kElBin:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatElBin;
        break;
      case AssemblyFeature::kDopplerBin:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatDopplerBin;
        break;
      case AssemblyFeature::kRangeBin:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatRangeBin;
        break;
    }
  }
  auto feat_dev = make_device_tensor({num_features}, kInt32Dtype);
  auto feat_mut = feat_dev->mutable_data();
  if (!feat_mut) {
    throw std::runtime_error(feat_mut.error().message);
  }
  if (cudaMemcpy(feat_mut.value(), feature_ids.data(), feature_ids.size() * sizeof(int),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    throw std::runtime_error("feature id H2D failed");
  }

  auto err_dev = make_device_tensor({1}, kInt32Dtype);
  auto err_mut = err_dev->mutable_data();
  if (!err_mut) {
    throw std::runtime_error(err_mut.error().message);
  }
  if (cudaMemset(err_mut.value(), 0, sizeof(int)) != cudaSuccess) {
    throw std::runtime_error("error flag clear failed");
  }

  auto points_dev = make_device_tensor({num_points, num_features}, real_dtype<T>());
  auto points_mut = points_dev->mutable_data();
  if (!points_mut) {
    throw std::runtime_error(points_mut.error().message);
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    throw std::runtime_error("cudaStreamCreate failed");
  }

  if (have_score && n_preserved > 0 && score_plane_offsets == nullptr) {
    cudaStreamDestroy(stream);
    throw std::invalid_argument("stage_assemble: missing score plane offsets");
  }

  holoscan::radar::detail::launch_assemble_features<T>(
      static_cast<const int64_t*>(idx_dev->data()), num_points, idx_cols, n_preserved, range_bins,
      num_range, have_range, velocity_bins, num_doppler, have_velocity, score_map,
      num_score_doppler, num_score_range, score_plane_offsets, have_score, aoa, have_aoa, az_grid,
      num_az, el_grid, num_el, static_cast<const int*>(feat_dev->data()), num_features,
      static_cast<T*>(points_mut.value()), static_cast<int*>(err_mut.value()), stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaStreamDestroy(stream);
    throw std::runtime_error("assemble sync failed");
  }
  cudaStreamDestroy(stream);

  int host_error = 0;
  if (cudaMemcpy(&host_error, err_dev->data(), sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    throw std::runtime_error("error flag D2H failed");
  }
  if (host_error != 0) {
    throw std::invalid_argument("stage assemble kernel error " + std::to_string(host_error));
  }
  return to_host(*points_dev);
}

template <typename T>
std::shared_ptr<holoscan::Tensor> upload_real(const std::vector<T>& host,
                                              const std::vector<int64_t>& shape) {
  return to_device_tensor(host, shape, real_dtype<T>());
}

// ---------------------------------------------------------------------------
// Graph: constructor-derived temporal joins.
// ---------------------------------------------------------------------------

TEST(FeatureAssemblyOpTest, AssemblesRawBinEscapeHatches) {
  std::vector<int64_t> idx = {3, 5, 7, 2};
  auto idx_t = to_device_tensor(idx, {2, 2}, kInt64Dtype);
  const auto points = run_assembly<double>(
      idx_t, make_features({AssemblyFeature::kDopplerBin, AssemblyFeature::kRangeBin}));
  ASSERT_EQ(points.shape.size(), 2u);
  EXPECT_EQ(points.shape[0], 2);
  EXPECT_EQ(points.shape[1], 2);
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 3.0);
  EXPECT_DOUBLE_EQ(p[1], 5.0);
  EXPECT_DOUBLE_EQ(p[2], 7.0);
  EXPECT_DOUBLE_EQ(p[3], 2.0);
}

TEST(FeatureAssemblyOpTest, AssemblesRawDenseAngleBinIndices) {
  std::vector<int64_t> idx = {3, 2, 7, 5};
  auto idx_t = to_device_tensor(idx, {1, 4}, kInt64Dtype);
  const auto points = run_assembly<double>(
      idx_t, make_features({AssemblyFeature::kAzBin, AssemblyFeature::kElBin,
                            AssemblyFeature::kDopplerBin, AssemblyFeature::kRangeBin}));
  ASSERT_EQ(points.shape, (std::vector<int64_t>{1, 4}));
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 3.0);
  EXPECT_DOUBLE_EQ(p[1], 2.0);
  EXPECT_DOUBLE_EQ(p[2], 7.0);
  EXPECT_DOUBLE_EQ(p[3], 5.0);
}

TEST(FeatureAssemblyOpTest, RejectsPhysicalFeaturesWithoutAxes) {
  std::vector<int64_t> idx = {1, 9};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<float>(
      idx_t, make_features({AssemblyFeature::kRange, AssemblyFeature::kVelocity}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsScoreFeatureWithoutScoreInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(idx_t, make_features({AssemblyFeature::kScore}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsEmptyFeatures) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(idx_t, make_features({}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, EmptyDetectionIdxEmitsEmptyPoints) {
  std::vector<int64_t> idx;
  auto idx_t = to_device_tensor(idx, {0, 2}, kInt64Dtype);
  const auto points = run_assembly<double>(
      idx_t, make_features({AssemblyFeature::kDopplerBin, AssemblyFeature::kRangeBin}));
  ASSERT_EQ(points.shape.size(), 2u);
  EXPECT_EQ(points.shape[0], 0);
  EXPECT_EQ(points.shape[1], 2);
}

TEST(FeatureAssemblyOpTest, FeaturesConfiguredAtConstruction) {
  std::vector<int64_t> idx = {4, 8};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto points = run_assembly<double>(
      idx_t, make_features({AssemblyFeature::kRangeBin, AssemblyFeature::kDopplerBin}));
  ASSERT_EQ(points.shape[1], 2);
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 8.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], 4.0);
}

TEST(FeatureAssemblyOpTest, RejectsAzEstimateWithoutAoaInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result =
      run_assembly_result<double>(idx_t, make_features({AssemblyFeature::kAzEstimate}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsElEstimateWithoutAoaInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result =
      run_assembly_result<double>(idx_t, make_features({AssemblyFeature::kElEstimate}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsBadDetectionIdxRank) {
  std::vector<int64_t> idx = {0, 1, 2};
  auto idx_t = to_device_tensor(idx, {3}, kInt64Dtype);
  const auto result =
      run_assembly_result<double>(idx_t, make_features({AssemblyFeature::kDopplerBin}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsOpaqueLeadingBatch) {
  // Rank 3 is not a named axis; parse_detection_idx_layout enforces
  // the module named-axis invariant.
  std::vector<int64_t> idx = {0, 1, 2, 3};
  auto idx_t = to_device_tensor(idx, {2, 1, 2}, kInt64Dtype);
  const auto result =
      run_assembly_result<double>(idx_t, make_features({AssemblyFeature::kDopplerBin}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsAzWithoutGrid) {
  std::vector<int64_t> idx = {0, 0, 0, 0};
  auto idx_t = to_device_tensor(idx, {1, 4}, kInt64Dtype);
  const auto result = run_assembly_result<double>(idx_t, make_features({AssemblyFeature::kAz}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, GathersAlignedRangeVelocity) {
  auto idx = to_device_tensor(std::vector<int64_t>{2, 3}, {1, 2}, kInt64Dtype);
  auto ranges = upload_real(std::vector<double>{0, 10, 20, 30}, {4});
  auto velocities = upload_real(std::vector<double>{-3, -2, -1, 0}, {4});
  auto result = run_assembly_frame<double>(
      make_features({AssemblyFeature::kRange, AssemblyFeature::kVelocity}),
      holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("range_bins", *ranges),
                                  holoscan::testing::input("velocity_bins", *velocities)),
      {"detection_idx", "range_bins", "velocity_bins"});
  const HostTensor points = require_one_frame(std::move(result));
  ASSERT_EQ(points.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 30.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], -1.0);
}

TEST(FeatureAssemblyOpTest, WiredUnselectedGatherDoesNotChangeTrigger) {
  auto idx = to_device_tensor(std::vector<int64_t>{2, 3}, {1, 2}, kInt64Dtype);
  auto ranges = upload_real(std::vector<double>{0, 10, 20, 30}, {4});
  auto result = run_assembly_frame<double>(
      make_features({AssemblyFeature::kRangeBin}),
      holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("range_bins", *ranges)),
      {"detection_idx", "range_bins"});
  const HostTensor points = require_one_frame(std::move(result));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 3.0);
}

TEST(FeatureAssemblyOpTest, GathersAlignedScore) {
  auto idx = to_device_tensor(std::vector<int64_t>{1, 2}, {1, 2}, kInt64Dtype);
  std::vector<double> score(12, 0.0);
  score[6] = 7.5;
  auto scores = upload_real(score, {3, 4});
  auto result = run_assembly_frame<double>(
      make_features({AssemblyFeature::kScore}),
      holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("detection_score", *scores)),
      {"detection_idx", "detection_score"});
  const HostTensor points = require_one_frame(std::move(result));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 7.5);
}

TEST(FeatureAssemblyOpTest, WaitsForLaterAoaFromSameFrame) {
  // DualTensorSourceOp emits out0 before out1. The zero-window OnAligned
  // trigger waits for the later companion carrying the same capture time.
  auto idx = to_device_tensor(std::vector<int64_t>{0, 1}, {1, 2}, kInt64Dtype);
  auto aoa = upload_real(std::vector<double>{0.25, -0.5}, {1, 2});
  auto result = run_assembly_frame<double>(
      make_features({AssemblyFeature::kAzEstimate, AssemblyFeature::kElEstimate}),
      holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("aoa_estimates", *aoa)),
      {"detection_idx", "aoa_estimates"});
  const HostTensor points = require_one_frame(std::move(result));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 0.25);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], -0.5);
}

TEST(FeatureAssemblyOpTest, GathersAlignedDenseAngleGrids) {
  auto idx = to_device_tensor(std::vector<int64_t>{1, 2, 0, 0}, {1, 4}, kInt64Dtype);
  auto az = upload_real(std::vector<double>{-0.2, 0.0, 0.2}, {3});
  auto el = upload_real(std::vector<double>{0.0, 0.1, 0.2}, {3});
  auto result = run_assembly_frame<double>(
      make_features({AssemblyFeature::kAz, AssemblyFeature::kEl}),
      holoscan::testing::frame_at(
          std::chrono::nanoseconds{0}, holoscan::testing::input("detection_idx", *idx),
          holoscan::testing::input("az_grid", *az), holoscan::testing::input("el_grid", *el)),
      {"detection_idx", "az_grid", "el_grid"});
  const HostTensor points = require_one_frame(std::move(result));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 0.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], 0.2);
}

// ---------------------------------------------------------------------------
// Stage: optional gather / lookup paths.
// ---------------------------------------------------------------------------

TEST(FeatureAssemblyOpTest, LooksUpPhysicalRangeVelocityAndScore) {
  constexpr int64_t kD = 8;
  constexpr int64_t kR = 16;
  std::vector<int64_t> idx = {2, 4};

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

  auto rb = upload_real(range_bins, {kR});
  auto vb = upload_real(vel_bins, {kD});
  auto sc = upload_real(score, {kD, kR});

  const auto points = stage_assemble<double>(
      idx, 1, 2, {AssemblyFeature::kRange, AssemblyFeature::kVelocity, AssemblyFeature::kScore},
      static_cast<const double*>(rb->data()), kR, true, static_cast<const double*>(vb->data()), kD,
      true, static_cast<const double*>(sc->data()), kD, kR, nullptr, true, nullptr, false, nullptr,
      0, nullptr, 0);
  ASSERT_EQ(points.shape[0], 1);
  ASSERT_EQ(points.shape[1], 3);
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 40.0);
  EXPECT_DOUBLE_EQ(p[1], -3.0);
  EXPECT_DOUBLE_EQ(p[2], 12.5);
}

TEST(FeatureAssemblyOpTest, AppendsAoaEstimates) {
  std::vector<int64_t> idx = {0, 1, 2, 3};
  std::vector<double> aoa = {0.1, -0.2, 0.3, 0.4};
  auto aoa_t = upload_real(aoa, {2, 2});
  const auto points = stage_assemble<double>(
      idx, 2, 2,
      {AssemblyFeature::kDopplerBin, AssemblyFeature::kAzEstimate, AssemblyFeature::kElEstimate},
      nullptr, 0, false, nullptr, 0, false, nullptr, 0, 0, nullptr, false,
      static_cast<const double*>(aoa_t->data()), true, nullptr, 0, nullptr, 0);
  ASSERT_EQ(points.shape[1], 3);
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 0.0);
  EXPECT_DOUBLE_EQ(p[1], 0.1);
  EXPECT_DOUBLE_EQ(p[2], -0.2);
  EXPECT_DOUBLE_EQ(p[3], 2.0);
  EXPECT_DOUBLE_EQ(p[4], 0.3);
  EXPECT_DOUBLE_EQ(p[5], 0.4);
}

TEST(FeatureAssemblyOpTest, FloatDoubleAgreeOnPhysicalLookup) {
  constexpr int64_t kR = 4;
  std::vector<int64_t> idx = {1, 2};
  std::vector<double> range_d = {0, 10, 20, 30};
  std::vector<float> range_f = {0, 10, 20, 30};
  auto rb_d = upload_real(range_d, {kR});
  auto rb_f = upload_real(range_f, {kR});

  const auto pd = stage_assemble<double>(
      idx, 1, 2, {AssemblyFeature::kRange}, static_cast<const double*>(rb_d->data()), kR, true,
      nullptr, 0, false, nullptr, 0, 0, nullptr, false, nullptr, false, nullptr, 0, nullptr, 0);
  const auto pf = stage_assemble<float>(
      idx, 1, 2, {AssemblyFeature::kRange}, static_cast<const float*>(rb_f->data()), kR, true,
      nullptr, 0, false, nullptr, 0, 0, nullptr, false, nullptr, false, nullptr, 0, nullptr, 0);
  EXPECT_NEAR(pd.as<double>()[0], static_cast<double>(pf.as<float>()[0]), 1e-5);
}

TEST(FeatureAssemblyOpTest, GathersScoreAcrossPreservedBatch) {
  constexpr int64_t kB = 2;
  constexpr int64_t kD = 4;
  constexpr int64_t kR = 4;
  std::vector<int64_t> idx = {0, 1, 2, 1, 2, 3};
  std::vector<double> score(static_cast<size_t>(kB * kD * kR), 0.0);
  score[static_cast<size_t>(0 * kD * kR + 1 * kR + 2)] = 11.0;
  score[static_cast<size_t>(1 * kD * kR + 2 * kR + 3)] = 23.0;
  auto sc = upload_real(score, {kB, kD, kR});

  auto idx_dev = to_device_tensor(idx, {2, 3}, kInt64Dtype);
  std::vector<int64_t> leading = {kB};
  auto lead_dev = to_device_tensor(leading, {1}, kInt64Dtype);
  auto offsets = make_device_tensor({2}, kInt64Dtype);
  auto err = make_device_tensor({1}, kInt32Dtype);
  auto err_mut = err->mutable_data();
  auto off_mut = offsets->mutable_data();
  ASSERT_TRUE(err_mut);
  ASSERT_TRUE(off_mut);
  ASSERT_EQ(cudaMemset(err_mut.value(), 0, sizeof(int)), cudaSuccess);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  holoscan::radar::detail::launch_score_plane_offsets(
      static_cast<const int64_t*>(idx_dev->data()), 2, 3, 1,
      static_cast<const int64_t*>(lead_dev->data()), kD * kR,
      static_cast<int64_t*>(off_mut.value()), static_cast<int*>(err_mut.value()), stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaStreamDestroy(stream);

  const auto points = stage_assemble<double>(
      idx, 2, 3, {AssemblyFeature::kScore}, nullptr, 0, false, nullptr, 0, false,
      static_cast<const double*>(sc->data()), kD, kR, static_cast<const int64_t*>(offsets->data()),
      true, nullptr, false, nullptr, 0, nullptr, 0);
  ASSERT_EQ(points.shape[0], 2);
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 11.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], 23.0);
}

TEST(FeatureAssemblyOpTest, RejectsOobRangeBinWhenBinsPresent) {
  std::vector<int64_t> idx = {0, 99};
  std::vector<double> range_bins = {0, 10, 20, 30};
  auto rb = upload_real(range_bins, {4});
  EXPECT_THROW(
      stage_assemble<double>(idx, 1, 2, {AssemblyFeature::kRange},
                             static_cast<const double*>(rb->data()), 4, true, nullptr, 0, false,
                             nullptr, 0, 0, nullptr, false, nullptr, false, nullptr, 0, nullptr, 0),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsOobScoreCoords) {
  std::vector<int64_t> idx = {8, 1};
  std::vector<double> score(4 * 4, 1.0);
  auto sc = upload_real(score, {4, 4});
  EXPECT_THROW(stage_assemble<double>(idx, 1, 2, {AssemblyFeature::kScore}, nullptr, 0, false,
                                      nullptr, 0, false, static_cast<const double*>(sc->data()), 4,
                                      4, nullptr, true, nullptr, false, nullptr, 0, nullptr, 0),
               std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsOobPreservedBatchCoord) {
  std::vector<int64_t> idx = {2, 0, 0};
  std::vector<double> score(2 * 4 * 4, 1.0);
  auto sc = upload_real(score, {2, 4, 4});

  auto idx_dev = to_device_tensor(idx, {1, 3}, kInt64Dtype);
  std::vector<int64_t> leading = {2};
  auto lead_dev = to_device_tensor(leading, {1}, kInt64Dtype);
  auto offsets = make_device_tensor({1}, kInt64Dtype);
  auto err = make_device_tensor({1}, kInt32Dtype);
  auto err_mut = err->mutable_data();
  auto off_mut = offsets->mutable_data();
  ASSERT_TRUE(err_mut);
  ASSERT_TRUE(off_mut);
  ASSERT_EQ(cudaMemset(err_mut.value(), 0, sizeof(int)), cudaSuccess);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  holoscan::radar::detail::launch_score_plane_offsets(
      static_cast<const int64_t*>(idx_dev->data()), 1, 3, 1,
      static_cast<const int64_t*>(lead_dev->data()), 4 * 4, static_cast<int64_t*>(off_mut.value()),
      static_cast<int*>(err_mut.value()), stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  int host_error = 0;
  ASSERT_EQ(cudaMemcpy(&host_error, err->data(), sizeof(int), cudaMemcpyDeviceToHost), cudaSuccess);
  cudaStreamDestroy(stream);
  EXPECT_EQ(host_error, holoscan::radar::detail::kErrBatchCoord);
  (void)sc;
}

TEST(FeatureAssemblyOpTest, DenseAzElGridLookup) {
  std::vector<int64_t> idx = {1, 2, 0, 0, 3, 0, 0, 0};
  std::vector<double> az = {-0.3, -0.1, 0.1, 0.3};
  std::vector<double> el = {0.0, 0.05, 0.1};
  auto az_t = upload_real(az, {4});
  auto el_t = upload_real(el, {3});

  const auto points = stage_assemble<double>(
      idx, 2, 4, {AssemblyFeature::kAz, AssemblyFeature::kEl}, nullptr, 0, false, nullptr, 0, false,
      nullptr, 0, 0, nullptr, false, nullptr, false, static_cast<const double*>(az_t->data()), 4,
      static_cast<const double*>(el_t->data()), 3);
  ASSERT_EQ(points.shape, (std::vector<int64_t>{2, 2}));
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], -0.1);
  EXPECT_DOUBLE_EQ(p[1], 0.1);
  EXPECT_DOUBLE_EQ(p[2], 0.3);
  EXPECT_DOUBLE_EQ(p[3], 0.0);
}

TEST(FeatureAssemblyOpTest, DenseAzElWithRangeVelocity) {
  std::vector<int64_t> idx = {2, 1, 3, 4};
  std::vector<double> range_bins = {0, 10, 20, 30, 40, 50};
  std::vector<double> vel_bins = {-5, -2.5, 0, 2.5};
  std::vector<double> az = {-0.2, 0.0, 0.2};
  std::vector<double> el = {0.0, 0.1};
  auto rb = upload_real(range_bins, {6});
  auto vb = upload_real(vel_bins, {4});
  auto az_t = upload_real(az, {3});
  auto el_t = upload_real(el, {2});

  const auto points = stage_assemble<double>(
      idx, 1, 4,
      {AssemblyFeature::kRange, AssemblyFeature::kVelocity, AssemblyFeature::kAz,
       AssemblyFeature::kEl},
      static_cast<const double*>(rb->data()), 6, true, static_cast<const double*>(vb->data()), 4,
      true, nullptr, 0, 0, nullptr, false, nullptr, false, static_cast<const double*>(az_t->data()),
      3, static_cast<const double*>(el_t->data()), 2);
  ASSERT_EQ(points.shape, (std::vector<int64_t>{1, 4}));
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 40.0);
  EXPECT_DOUBLE_EQ(p[1], 2.5);
  EXPECT_DOUBLE_EQ(p[2], 0.2);
  EXPECT_DOUBLE_EQ(p[3], 0.1);
}

TEST(FeatureAssemblyOpTest, RejectsAzWithTooFewPreservedColumns) {
  auto idx = to_device_tensor(std::vector<int64_t>{0, 0}, {1, 2}, kInt64Dtype);
  auto az = upload_real(std::vector<double>{0.0}, {1});
  const auto result = run_assembly_frame<double>(
      make_features({AssemblyFeature::kAz}),
      holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("az_grid", *az)),
      {"detection_idx", "az_grid"});
  EXPECT_TRUE(run_failed_with_diagnostic(result));
  EXPECT_NE(result.describe_failure().find("at least two"), std::string::npos);
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("points").empty());
}

TEST(FeatureAssemblyOpTest, RejectsAzBinWithTooFewPreservedColumns) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(idx_t, make_features({AssemblyFeature::kAzBin}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsOutOfRangeAzBin) {
  std::vector<int64_t> idx = {5, 0, 0, 0};
  std::vector<double> az = {-0.2, 0.0, 0.2};
  auto az_t = upload_real(az, {3});
  EXPECT_THROW(stage_assemble<double>(idx, 1, 4, {AssemblyFeature::kAz}, nullptr, 0, false, nullptr,
                                      0, false, nullptr, 0, 0, nullptr, false, nullptr, false,
                                      static_cast<const double*>(az_t->data()), 3, nullptr, 0),
               std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsScoreLeadingRankMismatch) {
  auto idx = to_device_tensor(std::vector<int64_t>{0, 1}, {1, 2}, kInt64Dtype);
  auto score = upload_real(std::vector<double>(8, 1.0), {2, 2, 2});
  const auto result = run_assembly_frame<double>(
      make_features({AssemblyFeature::kScore}),
      holoscan::testing::frame_at(std::chrono::nanoseconds{0},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("detection_score", *score)),
      {"detection_idx", "detection_score"});
  EXPECT_TRUE(run_failed_with_diagnostic(result));
  EXPECT_NE(result.describe_failure().find("leading rank must match"), std::string::npos);
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("points").empty());
}

// Native public-port coverage for the expansion consumer in both precisions.
// These owning snapshots verify values and errors; retained-loan ownership is
// exercised separately by the whole selective-angle integration graph.
template <typename T>
class FeatureAssemblyNative : public ::testing::Test {};
using AssemblyPrecisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(FeatureAssemblyNative, AssemblyPrecisions);

void expect_rejected_without_points(const holoscan::testing::TestResult& result,
                                    const char* message) {
  ASSERT_FALSE(result.ok()) << "Invalid input unexpectedly succeeded";
  EXPECT_FALSE(result.diagnostics().empty());
  EXPECT_NE(result.describe_failure().find(message), std::string::npos)
      << result.describe_failure();
  EXPECT_TRUE(result.snapshots<holoscan::Tensor>("points").empty());
}

TYPED_TEST(FeatureAssemblyNative, PreservesSparseAnglesRowAndFeatureOrder) {
  using T = TypeParam;
  auto idx = to_device_tensor(std::vector<int64_t>{2, 3, 0, 1}, {2, 2}, kInt64Dtype);
  auto aoa = upload_real(std::vector<T>{T{0.25}, T{-0.5}, T{-0.75}, T{0.125}}, {2, 2});
  const auto result = run_assembly_frame<T>(
      make_features({AssemblyFeature::kElEstimate, AssemblyFeature::kRangeBin,
                     AssemblyFeature::kAzEstimate, AssemblyFeature::kDopplerBin}),
      holoscan::testing::frame_at(std::chrono::milliseconds{1},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("aoa_estimates", *aoa)),
      {"detection_idx", "aoa_estimates"});
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshot = result.template snapshot<holoscan::Tensor>("points");
  const auto points = to_host(snapshot);
  ASSERT_EQ(points.shape, (std::vector<int64_t>{2, 4}));
  const std::vector<T> expected{T{-0.5}, T{3}, T{0.25}, T{2}, T{0.125}, T{1}, T{-0.75}, T{0}};
  EXPECT_EQ(std::vector<T>(points.template as<T>(), points.template as<T>() + expected.size()),
            expected);
  EXPECT_EQ(snapshot.sample_metadata().capture_time().timestamp_ns, 1000000);
  EXPECT_EQ(snapshot.source_device().device_type, kDLCUDA);
}

TYPED_TEST(FeatureAssemblyNative, RejectsMismatchedAoaShapeWithoutPublication) {
  using T = TypeParam;
  for (const auto& shape : {std::vector<int64_t>{1, 3}, std::vector<int64_t>{2, 2}}) {
    auto idx = to_device_tensor(std::vector<int64_t>{0, 1}, {1, 2}, kInt64Dtype);
    auto aoa = upload_real(std::vector<T>(static_cast<std::size_t>(shape[0] * shape[1])), shape);
    const auto result = run_assembly_frame<T>(
        make_features({AssemblyFeature::kAzEstimate}),
        holoscan::testing::frame_at(std::chrono::milliseconds{1},
                                    holoscan::testing::input("detection_idx", *idx),
                                    holoscan::testing::input("aoa_estimates", *aoa)),
        {"detection_idx", "aoa_estimates"});
    expect_rejected_without_points(result, "aoa_estimates");
  }
}

TYPED_TEST(FeatureAssemblyNative, RejectsNonVectorPhysicalAxisWithoutPublication) {
  using T = TypeParam;
  auto idx = to_device_tensor(std::vector<int64_t>{0, 1}, {1, 2}, kInt64Dtype);
  auto ranges = upload_real(std::vector<T>{T{0}, T{10}}, {1, 2});
  const auto result = run_assembly_frame<T>(
      make_features({AssemblyFeature::kRange}),
      holoscan::testing::frame_at(std::chrono::milliseconds{1},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("range_bins", *ranges)),
      {"detection_idx", "range_bins"});
  expect_rejected_without_points(result, "rank-1");
}

TYPED_TEST(FeatureAssemblyNative, RejectsWrongPrecisionAoaWithoutPublication) {
  using T = TypeParam;
  auto idx = to_device_tensor(std::vector<int64_t>{0, 1}, {1, 2}, kInt64Dtype);
  using Other = std::conditional_t<std::is_same_v<T, float>, double, float>;
  auto aoa = upload_real(std::vector<Other>{Other{0}, Other{0}}, {1, 2});
  const auto result = run_assembly_frame<T>(
      make_features({AssemblyFeature::kAzEstimate}),
      holoscan::testing::frame_at(std::chrono::milliseconds{1},
                                  holoscan::testing::input("detection_idx", *idx),
                                  holoscan::testing::input("aoa_estimates", *aoa)),
      {"detection_idx", "aoa_estimates"});
  expect_rejected_without_points(result, "dtype");
}

TYPED_TEST(FeatureAssemblyNative, RejectsPointAndFeatureCapacityOverflowWithoutPublication) {
  using T = TypeParam;
  auto idx = to_device_tensor(std::vector<int64_t>{0, 1, 2, 3}, {2, 2}, kInt64Dtype);
  for (const bool point_overflow : {true, false}) {
    auto extents = holoscan::radar::testing::default_test_extents();
    extents.max_points = point_overflow ? 1 : 2;
    extents.max_features = point_overflow ? 2 : 1;
    const auto result = run_assembly_frame<T>(
        make_features({AssemblyFeature::kDopplerBin, AssemblyFeature::kRangeBin}),
        holoscan::testing::frame_at(std::chrono::milliseconds{1},
                                    holoscan::testing::input("detection_idx", *idx)),
        {"detection_idx"}, extents);
    expect_rejected_without_points(result, point_overflow ? "max_points" : "max_features");
  }
}

TYPED_TEST(FeatureAssemblyNative, RejectsNegativeAndEndExclusiveRangeIndices) {
  using T = TypeParam;
  auto ranges = upload_real(std::vector<T>{T{0}, T{10}}, {2});
  for (const int64_t range : {-1, 2}) {
    auto idx = to_device_tensor(std::vector<int64_t>{0, range}, {1, 2}, kInt64Dtype);
    const auto result = run_assembly_frame<T>(
        make_features({AssemblyFeature::kRange}),
        holoscan::testing::frame_at(std::chrono::milliseconds{1},
                                    holoscan::testing::input("detection_idx", *idx),
                                    holoscan::testing::input("range_bins", *ranges)),
        {"detection_idx", "range_bins"});
    expect_rejected_without_points(result, "range_bin out of range");
  }
}

TYPED_TEST(FeatureAssemblyNative, SelectedAoaTracksEachExactFrameAndChangedRowCount) {
  using T = TypeParam;
  using Op = FeatureAssemblyOp<T>;
  auto fixture = holoscan::testing::create_operator_test<Op>(
      holoscan::radar::testing::default_test_extents(),
      make_features({AssemblyFeature::kAzEstimate, AssemblyFeature::kRangeBin}));
  auto first_idx = to_device_tensor(std::vector<int64_t>{0, 1, 2, 3}, {2, 2}, kInt64Dtype);
  auto first_aoa = upload_real(std::vector<T>{T{0.25}, T{0}, T{0.5}, T{0}}, {2, 2});
  auto second_idx = to_device_tensor(std::vector<int64_t>{4, 5}, {1, 2}, kInt64Dtype);
  auto second_aoa = upload_real(std::vector<T>{T{-0.75}, T{0}}, {1, 2});
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{1}, holoscan::testing::input("detection_idx", *first_idx),
      holoscan::testing::input("aoa_estimates", *first_aoa)));
  fixture.push_frame(holoscan::testing::frame_at(
      std::chrono::milliseconds{2}, holoscan::testing::input("detection_idx", *second_idx),
      holoscan::testing::input("aoa_estimates", *second_aoa)));
  fixture.template capture_snapshot<holoscan::Tensor>("points", 2U, 4U * sizeof(T));
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.input_source_device("aoa_estimates", holoscan::DeviceId{0});
  fixture.output_device("points", holoscan::DeviceId{0});
  auto running = fixture.run_async({.expect_stop = true});
  const bool observed = running.wait_for_output("points", 2U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(observed) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshots = result.template snapshots<holoscan::Tensor>("points");
  ASSERT_EQ(snapshots.size(), 2U);
  const auto first = to_host(snapshots[0]);
  const auto second = to_host(snapshots[1]);
  ASSERT_EQ(first.shape, (std::vector<int64_t>{2, 2}));
  ASSERT_EQ(second.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(first.template as<T>()[0], T{0.25});
  EXPECT_EQ(first.template as<T>()[1], T{1});
  EXPECT_EQ(first.template as<T>()[2], T{0.5});
  EXPECT_EQ(first.template as<T>()[3], T{3});
  EXPECT_EQ(second.template as<T>()[0], T{-0.75});
  EXPECT_EQ(second.template as<T>()[1], T{5});
  EXPECT_EQ(snapshots[0].sample_metadata().capture_time().timestamp_ns, 1000000);
  EXPECT_EQ(snapshots[1].sample_metadata().capture_time().timestamp_ns, 2000000);
}

// Admission of one manual notification does not mean the source callback has
// retired. Retry only the documented transient admission errors; finish() owns
// shutdown and diagnostic collection even if a test assertion fails.
bool release_assembly_frame(holoscan::testing::RunningFixture& running) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    const auto released = running.release_frame();
    if (released) {
      return true;
    }
    if (released.error().code != holoscan::ErrorCode::kNotReady &&
        released.error().code != holoscan::ErrorCode::kBackpressured) {
      return false;
    }
    std::this_thread::yield();
  }
  return false;
}

TYPED_TEST(FeatureAssemblyNative, SelectedAoaRejectsMissingAndSkewedFramesThenRecoversExactly) {
  using T = TypeParam;
  using Op = FeatureAssemblyOp<T>;
  using holoscan::testing::frame_at;
  using holoscan::testing::input;
  auto fixture = holoscan::testing::create_operator_test<Op>(
      holoscan::radar::testing::default_test_extents(),
      make_features({AssemblyFeature::kAzEstimate, AssemblyFeature::kRangeBin}));
  auto first_idx = to_device_tensor(std::vector<int64_t>{1, 10}, {1, 2}, kInt64Dtype);
  auto missing_idx = to_device_tensor(std::vector<int64_t>{2, 20}, {1, 2}, kInt64Dtype);
  auto skewed_idx = to_device_tensor(std::vector<int64_t>{3, 30}, {1, 2}, kInt64Dtype);
  auto final_idx = to_device_tensor(std::vector<int64_t>{5, 50}, {1, 2}, kInt64Dtype);
  auto first_aoa = upload_real(std::vector<T>{T{0.125}, T{0}}, {1, 2});
  auto skewed_aoa = upload_real(std::vector<T>{T{0.75}, T{0}}, {1, 2});
  auto final_aoa = upload_real(std::vector<T>{T{-0.5}, T{0}}, {1, 2});
  fixture.push_frame(frame_at(std::chrono::milliseconds{1}, input("detection_idx", *first_idx),
                              input("aoa_estimates", *first_aoa)));
  fixture.push_frame(frame_at(std::chrono::milliseconds{2}, input("detection_idx", *missing_idx)));
  auto skewed = frame_at(std::chrono::milliseconds{3}, input("detection_idx", *skewed_idx));
  auto later_aoa = frame_at(std::chrono::milliseconds{4}, input("aoa_estimates", *skewed_aoa));
  skewed.fields.push_back(std::move(later_aoa.fields.front()));
  fixture.push_frame(skewed);
  fixture.push_frame(frame_at(std::chrono::milliseconds{5}, input("detection_idx", *final_idx),
                              input("aoa_estimates", *final_aoa)));
  fixture.template capture_snapshot<holoscan::Tensor>("points", 4U, 2U * sizeof(T));
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.input_source_device("aoa_estimates", holoscan::DeviceId{0});
  fixture.output_device("points", holoscan::DeviceId{0});
  auto running =
      fixture.run_async({.expect_stop = true, .frames = holoscan::testing::FrameRelease::kManual});
  ASSERT_TRUE(release_assembly_frame(running));
  ASSERT_TRUE(running.wait_for_output("points", 1U, std::chrono::seconds{5}));
  ASSERT_TRUE(release_assembly_frame(running));
  EXPECT_FALSE(running.wait_for_output("points", 2U, std::chrono::milliseconds{50}));
  ASSERT_TRUE(release_assembly_frame(running));
  EXPECT_FALSE(running.wait_for_output("points", 2U, std::chrono::milliseconds{50}));
  ASSERT_TRUE(release_assembly_frame(running));
  const bool recovered = running.wait_for_output("points", 2U, std::chrono::seconds{5});
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(recovered) << result.describe_failure();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshots = result.template snapshots<holoscan::Tensor>("points");
  // Recovery with the t5 values is a positive progress witness, not merely a
  // timeout interpreted as success. Extra cross-tick outputs fail this count.
  ASSERT_EQ(snapshots.size(), 2U);
  const auto first = to_host(snapshots[0]);
  const auto final = to_host(snapshots[1]);
  ASSERT_EQ(first.shape, (std::vector<int64_t>{1, 2}));
  ASSERT_EQ(final.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(first.template as<T>()[0], T{0.125});
  EXPECT_EQ(first.template as<T>()[1], T{10});
  EXPECT_EQ(final.template as<T>()[0], T{-0.5});
  EXPECT_EQ(final.template as<T>()[1], T{50});
  EXPECT_EQ(snapshots[0].sample_metadata().capture_time().timestamp_ns, 1000000);
  EXPECT_EQ(snapshots[1].sample_metadata().capture_time().timestamp_ns, 5000000);
}

TYPED_TEST(FeatureAssemblyNative, WiredUnselectedAoaCannotStallMissingOrSkewedFrames) {
  using T = TypeParam;
  using Op = FeatureAssemblyOp<T>;
  using holoscan::testing::frame_at;
  using holoscan::testing::input;
  auto fixture = holoscan::testing::create_operator_test<Op>(
      holoscan::radar::testing::default_test_extents(),
      make_features({AssemblyFeature::kRangeBin, AssemblyFeature::kDopplerBin}));
  const std::vector<std::vector<int64_t>> indices{{1, 10}, {2, 20, 3, 30}, {4, 40}, {5, 50, 6, 60}};
  std::vector<std::shared_ptr<holoscan::Tensor>> frames;
  for (const auto& values : indices) {
    frames.push_back(
        to_device_tensor(values, {static_cast<int64_t>(values.size() / 2), 2}, kInt64Dtype));
  }
  auto first_aoa = upload_real(std::vector<T>{T{0.125}, T{0}}, {1, 2});
  auto skewed_aoa = upload_real(std::vector<T>{T{0.75}, T{0}}, {1, 2});
  fixture.push_frame(frame_at(std::chrono::milliseconds{1}, input("detection_idx", *frames[0]),
                              input("aoa_estimates", *first_aoa)));
  fixture.push_frame(frame_at(std::chrono::milliseconds{2}, input("detection_idx", *frames[1])));
  auto skewed = frame_at(std::chrono::milliseconds{3}, input("detection_idx", *frames[2]));
  auto later_aoa = frame_at(std::chrono::milliseconds{4}, input("aoa_estimates", *skewed_aoa));
  skewed.fields.push_back(std::move(later_aoa.fields.front()));
  fixture.push_frame(skewed);
  fixture.push_frame(frame_at(std::chrono::milliseconds{5}, input("detection_idx", *frames[3])));
  fixture.template capture_snapshot<holoscan::Tensor>("points", 4U, 4U * sizeof(T));
  fixture.input_source_device("detection_idx", holoscan::DeviceId{0});
  fixture.input_source_device("aoa_estimates", holoscan::DeviceId{0});
  fixture.output_device("points", holoscan::DeviceId{0});
  auto running =
      fixture.run_async({.expect_stop = true, .frames = holoscan::testing::FrameRelease::kManual});
  for (std::size_t index = 0; index < indices.size(); ++index) {
    ASSERT_TRUE(release_assembly_frame(running));
    ASSERT_TRUE(running.wait_for_output("points", index + 1, std::chrono::seconds{5}));
  }
  running.request_stop();
  const auto result = running.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto snapshots = result.template snapshots<holoscan::Tensor>("points");
  ASSERT_EQ(snapshots.size(), indices.size());
  for (std::size_t index = 0; index < indices.size(); ++index) {
    const auto points = to_host(snapshots[index]);
    ASSERT_EQ(points.shape,
              (std::vector<int64_t>{static_cast<int64_t>(indices[index].size() / 2), 2}));
    for (std::size_t row = 0; row < indices[index].size() / 2; ++row) {
      EXPECT_EQ(points.template as<T>()[row * 2], static_cast<T>(indices[index][row * 2 + 1]));
      EXPECT_EQ(points.template as<T>()[row * 2 + 1], static_cast<T>(indices[index][row * 2]));
    }
    const int64_t timestamp_ms = index < 3 ? static_cast<int64_t>(index + 1) : 5;
    EXPECT_EQ(snapshots[index].sample_metadata().capture_time().timestamp_ns,
              timestamp_ms * 1000000);
  }
}

}  // namespace
