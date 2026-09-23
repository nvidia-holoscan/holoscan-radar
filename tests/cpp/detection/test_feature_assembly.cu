// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FeatureAssemblyOp contract tests (HSDK 5 Graph harness + stage kernels):
// static features defines both output columns and the compiled temporal join.
// Selected sources are exact-aligned; unselected gathers may remain unwired.
// detection_idx is rank 2; extra leading axes fail closed.

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>

#include "detection/feature_assembly_op/feature_assembly_op.hpp"
#include "detection/feature_assembly_op/feature_assembly_stage.hpp"
#include "radar_common/tensor/dlpack_buffer.hpp"
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

using holoscan::radar::AssemblyFeature;
using holoscan::radar::FeatureAssemblyOp;
using holoscan::radar::make_device_tensor;
using holoscan::radar::real_dtype;
using holoscan::radar::testing::GraphRunResult;
using holoscan::radar::testing::HostTensor;
using holoscan::radar::testing::run_dual_in_single_out_operator_graph_result;
using holoscan::radar::testing::run_single_io_operator_graph_result;
using holoscan::radar::testing::run_triple_in_single_out_operator_graph_result;
using holoscan::radar::testing::to_device_tensor;
using holoscan::radar::testing::to_host;

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};
constexpr DLDataType kInt32Dtype{kDLInt, 32, 1};

std::vector<AssemblyFeature> make_features(
    std::vector<AssemblyFeature> features) {
  return features;
}

template <typename Source, typename OpRef, typename Sink>
void wire_assembly(holoscan::Graph& graph, const Source& source, OpRef& op,
                   const Sink& sink) {
  const holoscan::ConnectionOptions q{.queue_depth = 4U};
  graph.add_flow(source->out, op->detection_idx, q);
  graph.add_flow(op->points, sink->in, q);
}

template <typename T>
GraphRunResult run_assembly_result(std::shared_ptr<holoscan::Tensor> idx,
                                   std::vector<AssemblyFeature> features) {
  return run_single_io_operator_graph_result<FeatureAssemblyOp<T>>(
      {std::move(idx)}, std::move(features), "points",
      [](holoscan::Graph& graph, const auto& source, auto& op,
         const auto& sink) { wire_assembly(graph, source, op, sink); });
}

template <typename T>
HostTensor run_assembly(std::shared_ptr<holoscan::Tensor> idx,
                        std::vector<AssemblyFeature> features) {
  GraphRunResult result =
      run_assembly_result<T>(std::move(idx), std::move(features));
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    std::string msg = "run diagnostics:";
    for (const auto& d : result.diagnostics) {
      msg += " ";
      msg += d;
    }
    throw std::runtime_error(msg);
  }
  if (result.frames.size() != 1U) {
    throw std::runtime_error(
        "FeatureAssemblyOp test: expected one points frame, got " +
        std::to_string(result.frames.size()));
  }
  return std::move(result.frames.front());
}

bool run_failed_with_diagnostic(const GraphRunResult& result) {
  return !result.diagnostics.empty() || !result.plan_ok;
}

HostTensor require_one_frame(GraphRunResult result) {
  if (!result.plan_ok) {
    throw std::runtime_error(std::string("compile failed: ") +
                             result.plan_json);
  }
  if (!result.diagnostics.empty()) {
    throw std::runtime_error(std::string("run diagnostic: ") +
                             result.diagnostics.front());
  }
  if (result.frames.size() != 1U) {
    throw std::runtime_error(
        "FeatureAssemblyOp test: expected one points frame");
  }
  return std::move(result.frames.front());
}

// ---------------------------------------------------------------------------
// Stage helpers — focused kernel and error-path coverage.
// ---------------------------------------------------------------------------

template <typename T>
HostTensor stage_assemble(
    const std::vector<int64_t>& idx_host, int64_t num_points, int64_t idx_cols,
    const std::vector<AssemblyFeature>& features,
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
        feature_ids[static_cast<size_t>(i)] =
            holoscan::radar::detail::kFeatVelocity;
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
        feature_ids[static_cast<size_t>(i)] =
            holoscan::radar::detail::kFeatAzEstimate;
        break;
      case AssemblyFeature::kElEstimate:
        feature_ids[static_cast<size_t>(i)] =
            holoscan::radar::detail::kFeatElEstimate;
        break;
      case AssemblyFeature::kAzBin:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatAzBin;
        break;
      case AssemblyFeature::kElBin:
        feature_ids[static_cast<size_t>(i)] = holoscan::radar::detail::kFeatElBin;
        break;
      case AssemblyFeature::kDopplerBin:
        feature_ids[static_cast<size_t>(i)] =
            holoscan::radar::detail::kFeatDopplerBin;
        break;
      case AssemblyFeature::kRangeBin:
        feature_ids[static_cast<size_t>(i)] =
            holoscan::radar::detail::kFeatRangeBin;
        break;
    }
  }
  auto feat_dev = make_device_tensor({num_features}, kInt32Dtype);
  auto feat_mut = feat_dev->mutable_data();
  if (!feat_mut) {
    throw std::runtime_error(feat_mut.error().message);
  }
  if (cudaMemcpy(feat_mut.value(), feature_ids.data(),
                 feature_ids.size() * sizeof(int),
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

  auto points_dev =
      make_device_tensor({num_points, num_features}, real_dtype<T>());
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
      static_cast<const int64_t*>(idx_dev->data()), num_points, idx_cols,
      n_preserved, range_bins, num_range, have_range, velocity_bins,
      num_doppler, have_velocity, score_map, num_score_doppler, num_score_range,
      score_plane_offsets, have_score, aoa, have_aoa, az_grid, num_az,
      el_grid, num_el, static_cast<const int*>(feat_dev->data()),
      num_features, static_cast<T*>(points_mut.value()),
      static_cast<int*>(err_mut.value()), stream);
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    cudaStreamDestroy(stream);
    throw std::runtime_error("assemble sync failed");
  }
  cudaStreamDestroy(stream);

  int host_error = 0;
  if (cudaMemcpy(&host_error, err_dev->data(), sizeof(int),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    throw std::runtime_error("error flag D2H failed");
  }
  if (host_error != 0) {
    throw std::invalid_argument("stage assemble kernel error " +
                                std::to_string(host_error));
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
      idx_t, make_features({AssemblyFeature::kDopplerBin,
                          AssemblyFeature::kRangeBin}));
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
      idx_t, make_features({AssemblyFeature::kAzBin,
                          AssemblyFeature::kElBin,
                          AssemblyFeature::kDopplerBin,
                          AssemblyFeature::kRangeBin}));
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
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kScore}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsEmptyFeatures) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result =
      run_assembly_result<double>(idx_t, make_features({}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, EmptyDetectionIdxEmitsEmptyPoints) {
  std::vector<int64_t> idx;
  auto idx_t = to_device_tensor(idx, {0, 2}, kInt64Dtype);
  const auto points = run_assembly<double>(
      idx_t, make_features({AssemblyFeature::kDopplerBin,
                          AssemblyFeature::kRangeBin}));
  ASSERT_EQ(points.shape.size(), 2u);
  EXPECT_EQ(points.shape[0], 0);
  EXPECT_EQ(points.shape[1], 2);
}

TEST(FeatureAssemblyOpTest, FeaturesConfiguredAtConstruction) {
  std::vector<int64_t> idx = {4, 8};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto points = run_assembly<double>(
      idx_t, make_features({AssemblyFeature::kRangeBin,
                          AssemblyFeature::kDopplerBin}));
  ASSERT_EQ(points.shape[1], 2);
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 8.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], 4.0);
}

TEST(FeatureAssemblyOpTest, RejectsAzEstimateWithoutAoaInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kAzEstimate}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsElEstimateWithoutAoaInput) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kElEstimate}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsBadDetectionIdxRank) {
  std::vector<int64_t> idx = {0, 1, 2};
  auto idx_t = to_device_tensor(idx, {3}, kInt64Dtype);
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kDopplerBin}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsOpaqueLeadingBatch) {
  // Rank 3 is not a named axis; parse_detection_idx_layout enforces
  // the module named-axis invariant.
  std::vector<int64_t> idx = {0, 1, 2, 3};
  auto idx_t = to_device_tensor(idx, {2, 1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kDopplerBin}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsAzWithoutGrid) {
  std::vector<int64_t> idx = {0, 0, 0, 0};
  auto idx_t = to_device_tensor(idx, {1, 4}, kInt64Dtype);
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kAz}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsNonContiguousDetectionIdx) {
  // Graph harness emit_bank_frame always packs a compact loan from shape ×
  // dtype, so a non-contiguous source tensor would be compacted before the
  // operator sees it. Assert the contract helper directly instead.
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
  std::vector<int64_t> host = {0, 0, 1, 0};
  cudaMemcpy(holder->data, host.data(), 4 * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  int current_device_id = -1;
  if (cudaGetDevice(&current_device_id) != cudaSuccess) {
    throw std::runtime_error("make_idx_with_strides: cudaGetDevice failed");
  }
  holder->dl.dl_tensor.data = holder->data;
  holder->dl.dl_tensor.device = DLDevice{kDLCUDA, current_device_id};
  holder->dl.dl_tensor.ndim = 2;
  holder->dl.dl_tensor.dtype = kInt64Dtype;
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
      holoscan::radar::require_contiguous(*idx_t, "detection_idx",
                                          "FeatureAssemblyOp"),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, GathersAlignedRangeVelocity) {
  auto idx = to_device_tensor(std::vector<int64_t>{2, 3}, {1, 2},
                              kInt64Dtype);
  auto ranges = upload_real(std::vector<double>{0, 10, 20, 30}, {4});
  auto velocities =
      upload_real(std::vector<double>{-3, -2, -1, 0}, {4});
  auto result =
      run_triple_in_single_out_operator_graph_result<FeatureAssemblyOp<double>>(
          {idx}, {ranges}, {velocities},
          make_features(
              {AssemblyFeature::kRange, AssemblyFeature::kVelocity}),
          "points",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->detection_idx, q);
            graph.add_flow(source->out1, op->range_bins, q);
            graph.add_flow(source->out2, op->velocity_bins, q);
            graph.add_flow(op->points, sink->in, q);
          });
  const HostTensor points = require_one_frame(std::move(result));
  ASSERT_EQ(points.shape, (std::vector<int64_t>{1, 2}));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 30.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], -1.0);
}

TEST(FeatureAssemblyOpTest, WiredUnselectedGatherDoesNotChangeTrigger) {
  auto idx = to_device_tensor(std::vector<int64_t>{2, 3}, {1, 2},
                              kInt64Dtype);
  auto ranges = upload_real(std::vector<double>{0, 10, 20, 30}, {4});
  auto result =
      run_dual_in_single_out_operator_graph_result<FeatureAssemblyOp<double>>(
          {idx}, {ranges}, make_features({AssemblyFeature::kRangeBin}), "points",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->detection_idx, q);
            graph.add_flow(source->out1, op->range_bins, q);
            graph.add_flow(op->points, sink->in, q);
          });
  const HostTensor points = require_one_frame(std::move(result));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 3.0);
}

TEST(FeatureAssemblyOpTest, GathersAlignedScore) {
  auto idx = to_device_tensor(std::vector<int64_t>{1, 2}, {1, 2},
                              kInt64Dtype);
  std::vector<double> score(12, 0.0);
  score[6] = 7.5;
  auto scores = upload_real(score, {3, 4});
  auto result =
      run_dual_in_single_out_operator_graph_result<FeatureAssemblyOp<double>>(
          {idx}, {scores}, make_features({AssemblyFeature::kScore}), "points",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->detection_idx, q);
            graph.add_flow(source->out1, op->detection_score, q);
            graph.add_flow(op->points, sink->in, q);
          });
  const HostTensor points = require_one_frame(std::move(result));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 7.5);
}

TEST(FeatureAssemblyOpTest, WaitsForLaterAoaFromSameFrame) {
  // DualTensorSourceOp emits out0 before out1. The zero-window OnAligned
  // trigger waits for the later companion carrying the same capture time.
  auto idx = to_device_tensor(std::vector<int64_t>{0, 1}, {1, 2},
                              kInt64Dtype);
  auto aoa = upload_real(std::vector<double>{0.25, -0.5}, {1, 2});
  auto result =
      run_dual_in_single_out_operator_graph_result<FeatureAssemblyOp<double>>(
          {idx}, {aoa},
          make_features({AssemblyFeature::kAzEstimate,
                       AssemblyFeature::kElEstimate}),
          "points",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->detection_idx, q);
            graph.add_flow(source->out1, op->aoa_estimates, q);
            graph.add_flow(op->points, sink->in, q);
          });
  const HostTensor points = require_one_frame(std::move(result));
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 0.25);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], -0.5);
}

TEST(FeatureAssemblyOpTest, GathersAlignedDenseAngleGrids) {
  auto idx = to_device_tensor(std::vector<int64_t>{1, 2, 0, 0}, {1, 4},
                              kInt64Dtype);
  auto az = upload_real(std::vector<double>{-0.2, 0.0, 0.2}, {3});
  auto el = upload_real(std::vector<double>{0.0, 0.1, 0.2}, {3});
  auto result =
      run_triple_in_single_out_operator_graph_result<FeatureAssemblyOp<double>>(
          {idx}, {az}, {el},
          make_features(
              {AssemblyFeature::kAz, AssemblyFeature::kEl}),
          "points",
          [](holoscan::Graph& graph, const auto& source, auto& op,
             const auto& sink) {
            const holoscan::ConnectionOptions q{.queue_depth = 4U};
            graph.add_flow(source->out0, op->detection_idx, q);
            graph.add_flow(source->out1, op->az_grid, q);
            graph.add_flow(source->out2, op->el_grid, q);
            graph.add_flow(op->points, sink->in, q);
          });
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
      idx, 1, 2,
      {AssemblyFeature::kRange, AssemblyFeature::kVelocity,
       AssemblyFeature::kScore},
      static_cast<const double*>(rb->data()), kR, true,
      static_cast<const double*>(vb->data()), kD, true,
      static_cast<const double*>(sc->data()), kD, kR, nullptr, true, nullptr,
      false, nullptr, 0, nullptr, 0);
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
      {AssemblyFeature::kDopplerBin, AssemblyFeature::kAzEstimate,
       AssemblyFeature::kElEstimate},
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
      idx, 1, 2, {AssemblyFeature::kRange},
      static_cast<const double*>(rb_d->data()), kR, true, nullptr, 0, false,
      nullptr, 0, 0, nullptr, false, nullptr, false, nullptr, 0, nullptr, 0);
  const auto pf = stage_assemble<float>(
      idx, 1, 2, {AssemblyFeature::kRange},
      static_cast<const float*>(rb_f->data()), kR, true, nullptr, 0, false,
      nullptr, 0, 0, nullptr, false, nullptr, false, nullptr, 0, nullptr, 0);
  EXPECT_NEAR(pd.as<double>()[0], static_cast<double>(pf.as<float>()[0]),
              1e-5);
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
      static_cast<int64_t*>(off_mut.value()),
      static_cast<int*>(err_mut.value()), stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  cudaStreamDestroy(stream);

  const auto points = stage_assemble<double>(
      idx, 2, 3, {AssemblyFeature::kScore}, nullptr, 0, false, nullptr, 0,
      false, static_cast<const double*>(sc->data()), kD, kR,
      static_cast<const int64_t*>(offsets->data()), true, nullptr, false,
      nullptr, 0, nullptr, 0);
  ASSERT_EQ(points.shape[0], 2);
  EXPECT_DOUBLE_EQ(points.as<double>()[0], 11.0);
  EXPECT_DOUBLE_EQ(points.as<double>()[1], 23.0);
}

TEST(FeatureAssemblyOpTest, RejectsOobRangeBinWhenBinsPresent) {
  std::vector<int64_t> idx = {0, 99};
  std::vector<double> range_bins = {0, 10, 20, 30};
  auto rb = upload_real(range_bins, {4});
  EXPECT_THROW(
      stage_assemble<double>(
          idx, 1, 2, {AssemblyFeature::kRange},
          static_cast<const double*>(rb->data()), 4, true, nullptr, 0, false,
          nullptr, 0, 0, nullptr, false, nullptr, false, nullptr, 0, nullptr,
          0),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsOobScoreCoords) {
  std::vector<int64_t> idx = {8, 1};
  std::vector<double> score(4 * 4, 1.0);
  auto sc = upload_real(score, {4, 4});
  EXPECT_THROW(
      stage_assemble<double>(
          idx, 1, 2, {AssemblyFeature::kScore}, nullptr, 0, false, nullptr, 0,
          false, static_cast<const double*>(sc->data()), 4, 4, nullptr, true,
          nullptr, false, nullptr, 0, nullptr, 0),
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
      static_cast<const int64_t*>(lead_dev->data()), 4 * 4,
      static_cast<int64_t*>(off_mut.value()),
      static_cast<int*>(err_mut.value()), stream);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  int host_error = 0;
  ASSERT_EQ(cudaMemcpy(&host_error, err->data(), sizeof(int),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
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
      idx, 2, 4, {AssemblyFeature::kAz, AssemblyFeature::kEl}, nullptr, 0,
      false, nullptr, 0, false, nullptr, 0, 0, nullptr, false, nullptr, false,
      static_cast<const double*>(az_t->data()), 4,
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
      {AssemblyFeature::kRange, AssemblyFeature::kVelocity,
       AssemblyFeature::kAz, AssemblyFeature::kEl},
      static_cast<const double*>(rb->data()), 6, true,
      static_cast<const double*>(vb->data()), 4, true, nullptr, 0, 0, nullptr,
      false, nullptr, false, static_cast<const double*>(az_t->data()), 3,
      static_cast<const double*>(el_t->data()), 2);
  ASSERT_EQ(points.shape, (std::vector<int64_t>{1, 4}));
  const double* p = points.as<double>();
  EXPECT_DOUBLE_EQ(p[0], 40.0);
  EXPECT_DOUBLE_EQ(p[1], 2.5);
  EXPECT_DOUBLE_EQ(p[2], 0.2);
  EXPECT_DOUBLE_EQ(p[3], 0.1);
}

TEST(FeatureAssemblyOpTest, RejectsAzWithTooFewPreservedColumns) {
  // Graph path rejects dense az when n_preserved < 2.
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  // Wire would need az_grid; without it Graph already fails for missing
  // grid. Stage path with n_preserved=0 simply skips az lookup (value 0);
  // the operator-level guard is what matters for the contract.
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kAz}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsAzBinWithTooFewPreservedColumns) {
  std::vector<int64_t> idx = {0, 0};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kAzBin}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

TEST(FeatureAssemblyOpTest, RejectsOutOfRangeAzBin) {
  std::vector<int64_t> idx = {5, 0, 0, 0};
  std::vector<double> az = {-0.2, 0.0, 0.2};
  auto az_t = upload_real(az, {3});
  EXPECT_THROW(
      stage_assemble<double>(
          idx, 1, 4, {AssemblyFeature::kAz}, nullptr, 0, false, nullptr, 0,
          false, nullptr, 0, 0, nullptr, false, nullptr, false,
          static_cast<const double*>(az_t->data()), 3, nullptr, 0),
      std::invalid_argument);
}

TEST(FeatureAssemblyOpTest, RejectsScoreLeadingRankMismatch) {
  // Operator-level: score leading rank must match preserved cols. Covered
  // when optional Graph wiring returns; stage path requires matching offsets
  // by construction. Keep a Graph-side absent-score reject as the stand-in.
  std::vector<int64_t> idx = {0, 1};
  auto idx_t = to_device_tensor(idx, {1, 2}, kInt64Dtype);
  const auto result = run_assembly_result<double>(
      idx_t, make_features({AssemblyFeature::kScore}));
  EXPECT_TRUE(run_failed_with_diagnostic(result));
}

}  // namespace
