// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "radar_app/report/report_ops.hpp"
#include "radar_common/sdk/tensor_io.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace app = holoscan::radar::app;
namespace testing = holoscan::radar::testing;

constexpr DLDataType kFloat32{kDLFloat, 32, 1};
constexpr DLDataType kInt64{kDLInt, 64, 1};

struct ReportFrames {
  std::shared_ptr<holoscan::Tensor> points;
  std::shared_ptr<holoscan::Tensor> detection_idx;
  std::shared_ptr<holoscan::Tensor> labels;
  std::shared_ptr<holoscan::Tensor> sizes;
  std::shared_ptr<holoscan::Tensor> range_bins;
  std::shared_ptr<holoscan::Tensor> velocity_bins;
};

class ReportSourceOp final : public holoscan::Operator<> {
 public:
  ReportSourceOp(ReportFrames frames, bool clustered)
      : frames_(std::move(frames)), clustered_(clustered) {}

  void setup(holoscan::OperatorSpec& spec) override {
    constexpr holoscan::TensorRepresentation kDevice{
        .memory_kind = holoscan::MemoryKind::kCudaDevice,
        .dtype = std::nullopt,
        .rank = std::nullopt,
    };
    const auto output = holoscan::radar::pooled_tensor_output(
        kDevice, 2U, 4096U);
    spec.output(points).produces_tensor(output);
    spec.output(detection_idx).produces_tensor(output);
    if (clustered_) {
      spec.output(labels).produces_tensor(output);
      spec.output(sizes).produces_tensor(output);
    }
    spec.output(range_bins).produces_tensor(output);
    spec.output(velocity_bins).produces_tensor(output);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{
        .period = std::chrono::milliseconds{5}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    if (clustered_) {
      auto result = testing::detail::emit_bank_frame(
          labels, context, frames_.labels, "ReportSourceOp");
      if (!result) {
        return result;
      }
      result = testing::detail::emit_bank_frame(
          sizes, context, frames_.sizes, "ReportSourceOp");
      if (!result) {
        return result;
      }
    }
    auto result = testing::detail::emit_bank_frame(
        range_bins, context, frames_.range_bins, "ReportSourceOp");
    if (!result) {
      return result;
    }
    result = testing::detail::emit_bank_frame(
        velocity_bins, context, frames_.velocity_bins, "ReportSourceOp");
    if (!result) {
      return result;
    }
    result = testing::detail::emit_bank_frame(
        detection_idx, context, frames_.detection_idx, "ReportSourceOp");
    if (!result) {
      return result;
    }
    return testing::detail::emit_bank_frame(
        points, context, frames_.points, "ReportSourceOp");
  }

  holoscan::Output<holoscan::Tensor> points{"points"};
  holoscan::Output<holoscan::Tensor> detection_idx{"detection_idx"};
  holoscan::Output<holoscan::Tensor> labels{"labels"};
  holoscan::Output<holoscan::Tensor> sizes{"sizes"};
  holoscan::Output<holoscan::Tensor> range_bins{"range_bins"};
  holoscan::Output<holoscan::Tensor> velocity_bins{"velocity_bins"};

 private:
  ReportFrames frames_;
  bool clustered_{false};
  bool emitted_{false};
};

std::vector<float> sequence(int64_t count) {
  std::vector<float> result(static_cast<size_t>(count));
  for (int64_t index = 0; index < count; ++index) {
    result[static_cast<size_t>(index)] = static_cast<float>(index);
  }
  return result;
}

std::vector<std::string> run_report(
    ReportFrames frames, app::DetectionReportOp::Config config) {
  const bool clustered =
      config.mode == app::DetectionReportMode::kClusters;
  app::AppFrameBank::reset(1);
  holoscan::Graph graph{"detection-report-test"};
  auto source =
      graph.op<ReportSourceOp>("source", std::move(frames), clustered);
  auto report = graph.op<app::DetectionReportOp>(
      "report", std::move(config));
  const holoscan::ConnectionOptions queue{.queue_depth = 2U};
  graph.add_flow(source->points, report->points, queue);
  graph.add_flow(source->detection_idx, report->detection_idx, queue);
  graph.add_flow(source->range_bins, report->range_bins, queue);
  graph.add_flow(source->velocity_bins, report->velocity_bins, queue);
  if (clustered) {
    graph.add_flow(source->labels, report->labels, queue);
    graph.add_flow(source->sizes, report->sizes, queue);
  }
  graph.partition("application").add(source).add(report);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  for (const char* port :
       {"points", "detection_idx", "range_bins", "velocity_bins"}) {
    options.deployment.bind_tensor_output_device(
        holoscan::TensorOutputDevicePlacement{
            .operator_path = "source",
            .output_port = port,
            .device = holoscan::DeviceId{0},
        });
  }
  if (clustered) {
    for (const char* port : {"labels", "sizes"}) {
      options.deployment.bind_tensor_output_device(
          holoscan::TensorOutputDevicePlacement{
              .operator_path = "source",
              .output_port = port,
              .device = holoscan::DeviceId{0},
          });
    }
  }
  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    return {std::string(plan.json())};
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline =
      std::chrono::steady_clock::now() + testing::detail::kHarnessDeadline;
  while (session.diagnostics().empty() &&
         std::chrono::steady_clock::now() < deadline) {
    {
      const std::lock_guard lock(app::AppFrameBank::mutex);
      if (app::AppFrameBank::captured >= 1) {
        break;
      }
    }
    std::this_thread::sleep_for(testing::detail::kHarnessPoll);
  }
  session.request_stop();
  session.wait();
  std::vector<std::string> diagnostics;
  for (const auto& diagnostic : session.diagnostics()) {
    diagnostics.emplace_back(diagnostic.token);
  }
  return diagnostics;
}

ReportFrames raw_frames(const std::vector<int64_t>& indices,
                        int64_t rows, int64_t idx_cols) {
  std::vector<float> points;
  points.reserve(static_cast<size_t>(rows * 5));
  for (int64_t row = 0; row < rows; ++row) {
    points.insert(points.end(),
                  {5.0F + static_cast<float>(row), 1.0F, 0.1F, 0.2F, 3.0F});
  }
  return ReportFrames{
      testing::to_device_tensor(points, {rows, 5}, kFloat32),
      testing::to_device_tensor(indices, {rows, idx_cols}, kInt64),
      nullptr,
      nullptr,
      testing::to_device_tensor(sequence(64), {64}, kFloat32),
      testing::to_device_tensor(sequence(64), {64}, kFloat32)};
}

ReportFrames single_cluster_frames(std::vector<int64_t> representative) {
  return ReportFrames{
      testing::to_device_tensor(
          std::vector<float>{
              5.0F, 1.0F, 0.10F, 0.00F, 3.0F, 20.0F, 10.0F},
          {1, 7}, kFloat32),
      testing::to_device_tensor(representative, {1, 2}, kInt64),
      testing::to_device_tensor(std::vector<int64_t>{0}, {1}, kInt64),
      testing::to_device_tensor(std::vector<int64_t>{1}, {1}, kInt64),
      testing::to_device_tensor(sequence(64), {64}, kFloat32),
      testing::to_device_tensor(sequence(64), {64}, kFloat32)};
}

void expect_report_ok(ReportFrames frames,
                      app::DetectionReportOp::Config config) {
  const auto diagnostics = run_report(std::move(frames), std::move(config));
  EXPECT_TRUE(diagnostics.empty())
      << (diagnostics.empty() ? "" : diagnostics.front());
}

TEST(DetectionReportOpTest, ReportsRawTwoColumnDetections) {
  expect_report_ok(raw_frames({1, 2, 3, 4}, 2, 2),
                   app::DetectionReportOp::Config{});
}

TEST(DetectionReportOpTest, ReportsRawFourColumnDetections) {
  expect_report_ok(raw_frames({0, 1, 2, 3, 1, 0, 4, 5}, 2, 4),
                   app::DetectionReportOp::Config{});
}

TEST(DetectionReportOpTest, ReportsEmptyRawDetectionSet) {
  expect_report_ok(raw_frames({}, 0, 4),
                   app::DetectionReportOp::Config{});
}

TEST(DetectionReportOpTest, ReportsClustersWithRepresentativeLookup) {
  const std::vector<float> points{
      5.0F, 1.0F, 0.10F, 0.00F, 3.0F, 20.0F, 10.0F,
      5.2F, 1.2F, 0.15F, 0.01F, 4.0F, 20.0F, 11.0F,
      9.0F, 3.0F, 0.40F, 0.02F, 5.0F, 40.0F, 30.0F};
  ReportFrames frames{
      testing::to_device_tensor(points, {3, 7}, kFloat32),
      testing::to_device_tensor(
          std::vector<int64_t>{11, 20, 30, 40}, {2, 2}, kInt64),
      testing::to_device_tensor(
          std::vector<int64_t>{0, 0, 1}, {3}, kInt64),
      testing::to_device_tensor(std::vector<int64_t>{2, 1}, {2}, kInt64),
      testing::to_device_tensor(sequence(64), {64}, kFloat32),
      testing::to_device_tensor(sequence(64), {64}, kFloat32)};
  expect_report_ok(
      std::move(frames),
      app::DetectionReportOp::Config{
          app::DetectionReportMode::kClusters, {6, 5}, nullptr});
}

TEST(DetectionReportOpTest, ReportsAllNoiseAsNoClusters) {
  ReportFrames frames{
      testing::to_device_tensor(
          std::vector<float>{
              5.0F, 1.0F, 0.10F, 0.00F, 3.0F, 20.0F, 10.0F},
          {1, 7}, kFloat32),
      testing::to_device_tensor(std::vector<int64_t>{}, {0, 2}, kInt64),
      testing::to_device_tensor(std::vector<int64_t>{-1}, {1}, kInt64),
      testing::to_device_tensor(std::vector<int64_t>{}, {0}, kInt64),
      testing::to_device_tensor(sequence(64), {64}, kFloat32),
      testing::to_device_tensor(sequence(64), {64}, kFloat32)};
  expect_report_ok(
      std::move(frames),
      app::DetectionReportOp::Config{
          app::DetectionReportMode::kClusters, {6, 5}, nullptr});
}

TEST(DetectionReportOpTest, RejectsIncompletePointIndexMapping) {
  const auto diagnostics = run_report(
      single_cluster_frames({10, 20}),
      app::DetectionReportOp::Config{
          app::DetectionReportMode::kClusters, {6}, nullptr});
  EXPECT_FALSE(diagnostics.empty());
}

TEST(DetectionReportOpTest, RejectsDuplicatePointIndexMapping) {
  const auto diagnostics = run_report(
      single_cluster_frames({10, 20}),
      app::DetectionReportOp::Config{
          app::DetectionReportMode::kClusters, {6, 6}, nullptr});
  EXPECT_FALSE(diagnostics.empty());
}

TEST(DetectionReportOpTest, RejectsOutOfRangePointIndexMapping) {
  const auto diagnostics = run_report(
      single_cluster_frames({10, 20}),
      app::DetectionReportOp::Config{
          app::DetectionReportMode::kClusters, {6, 7}, nullptr});
  EXPECT_FALSE(diagnostics.empty());
}

TEST(DetectionReportOpTest, RejectsUnmatchedRepresentativeIndex) {
  const auto diagnostics = run_report(
      single_cluster_frames({11, 20}),
      app::DetectionReportOp::Config{
          app::DetectionReportMode::kClusters, {6, 5}, nullptr});
  EXPECT_FALSE(diagnostics.empty());
}

}  // namespace
