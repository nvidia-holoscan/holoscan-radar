// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <holoscan/core/graph.hpp>

#include "custom_ops.hpp"  // NOLINT(build/include_subdir)
#include "radar_common/tensor/tensor_utils.hpp"
#include "utils/sdk_harness.hpp"
#include "utils/tensor_fixtures.hpp"

namespace {

namespace simo = holoscan::radar::app::simo;
namespace testing = holoscan::radar::testing;

constexpr DLDataType kInt64Dtype{kDLInt, 64, 1};

testing::ThreeOutGraphRunResult run_peak_identity(
    std::shared_ptr<holoscan::Tensor> input,
    const holoscan::radar::MaxTensorExtents& extents) {
  testing::ThreeOutGraphRunResult result;
  testing::detail::FrameBank::reset();
  testing::detail::FrameBankB::reset();
  testing::detail::FrameBankC::reset();
  testing::detail::FrameBank::input_frames = {std::move(input)};

  holoscan::Graph graph{"peak-identity-test"};
  const auto source =
      graph.op<testing::detail::TensorSourceOp>("source");
  auto op = graph.op<simo::PeakIdentityOp>("op", extents);
  const auto labels =
      graph.op<testing::detail::TensorCaptureOp>("labels_sink");
  const auto sizes =
      graph.op<testing::detail::TensorCaptureBOp>("sizes_sink");
  const auto rep =
      graph.op<testing::detail::TensorCaptureCOp>("rep_sink");
  const holoscan::ConnectionOptions queue{.queue_depth = 2U};
  graph.add_flow(source->out, op->detection_idx, queue);
  graph.add_flow(op->labels, labels->in, queue);
  graph.add_flow(op->sizes, sizes->in, queue);
  graph.add_flow(op->rep_idx, rep->in, queue);
  graph.set_default_clock(
      graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions options;
  testing::detail::bind_single_in_dual_out_device_pools(
      options, "labels", "sizes");
  testing::detail::bind_op_device_output(options, "rep_idx");
  const holoscan::ExecutionPlan plan =
      holoscan::compile(graph, std::move(options));
  result.plan_ok = plan.ok();
  result.plan_json = plan.json();
  if (!plan.ok()) {
    return result;
  }

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline =
      std::chrono::steady_clock::now() + testing::detail::kHarnessDeadline;
  while (session.diagnostics().empty() &&
         std::chrono::steady_clock::now() < deadline) {
    bool captured = false;
    {
      const std::scoped_lock lock(testing::detail::FrameBank::mutex,
                                  testing::detail::FrameBankB::mutex,
                                  testing::detail::FrameBankC::mutex);
      captured = !testing::detail::FrameBank::captured.empty() &&
                 !testing::detail::FrameBankB::captured.empty() &&
                 !testing::detail::FrameBankC::captured.empty();
    }
    if (captured) {
      break;
    }
    std::this_thread::sleep_for(testing::detail::kHarnessPoll);
  }
  session.request_stop();
  session.wait();
  for (const auto& diagnostic : session.diagnostics()) {
    result.diagnostics.push_back(std::string(diagnostic.token));
  }
  {
    const std::scoped_lock lock(testing::detail::FrameBank::mutex,
                                testing::detail::FrameBankB::mutex,
                                testing::detail::FrameBankC::mutex);
    result.frames0 = std::move(testing::detail::FrameBank::captured);
    result.frames1 = std::move(testing::detail::FrameBankB::captured);
    result.frames2 = std::move(testing::detail::FrameBankC::captured);
  }
  return result;
}

void expect_identity_outputs(const std::vector<int64_t>& indices,
                             int64_t count) {
  auto input =
      testing::to_device_tensor(indices, {count, 4}, kInt64Dtype);
  auto extents = testing::default_test_extents();
  extents.max_detections = 8;
  const auto result = run_peak_identity(std::move(input), extents);
  ASSERT_TRUE(result.plan_ok) << result.plan_json;
  ASSERT_TRUE(result.diagnostics.empty());
  ASSERT_EQ(result.frames0.size(), 1U);
  ASSERT_EQ(result.frames1.size(), 1U);
  ASSERT_EQ(result.frames2.size(), 1U);
  EXPECT_EQ(result.frames0[0].shape, (std::vector<int64_t>{count}));
  EXPECT_EQ(result.frames1[0].shape, (std::vector<int64_t>{count}));
  EXPECT_EQ(result.frames2[0].shape,
            (std::vector<int64_t>{count, 4}));
  const int64_t* labels = result.frames0[0].as<int64_t>();
  const int64_t* sizes = result.frames1[0].as<int64_t>();
  const int64_t* reps = result.frames2[0].as<int64_t>();
  for (int64_t index = 0; index < count; ++index) {
    EXPECT_EQ(labels[index], index);
    EXPECT_EQ(sizes[index], 1);
    for (int64_t column = 0; column < 4; ++column) {
      EXPECT_EQ(reps[index * 4 + column],
                indices[static_cast<size_t>(index * 4 + column)]);
    }
  }
}

TEST(PeakIdentityOpTest, ProducesIdentityLabelsSizesAndRepresentatives) {
  expect_identity_outputs({0, 1, 2, 3, 4, 5, 6, 7}, 2);
}

TEST(PeakIdentityOpTest, PreservesEmptyPeakShapes) {
  expect_identity_outputs({}, 0);
}

}  // namespace
